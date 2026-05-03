import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer

# ==========================================
# 0. 路径定位与组件导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# 假设 DPODataset 在这里
from dataset.lm_dataset import DPODataset


# ==========================================
# 1. 配置模块 (Configuration)
# ==========================================
@dataclass
class DPOConfig:
    batch_size: int = 2  # 实际处理 2*2=4 条序列
    epochs: int = 1
    learning_rate: float = 5e-6
    beta: float = 0.1  # 偏离惩罚系数
    max_seq_len: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. 核心损失函数模块 (Loss Computation)
# ==========================================
class DPOLoss(torch.nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta

    def get_batch_logprobs(self, logits, labels, mask):
        """
        计算特定 Token 的对数概率
        """
        # 1. 算出所有词的 log_softmax 概率
        log_probs = F.log_softmax(logits, dim=-1) # logits：形状 [B, T, V]

        # 2. 用 gather 把正确 label 对应的概率“抠”出来 [B, T]，labels：形状 [B, T]，每个位置上的真实 token id。
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        # per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        # labels 形状 [B, T] → 增加一维 → [B, T, 1]，作为索引，在 dim=2（词表维度）上取出实际 token 对应的对数概率。之后 squeeze(-1) 去掉最后一维，得到 [B, T]，每个元素是当前位置上正确 token 的 log 概率。
        # 3. 施加 mask，只保留回答部分的概率
        per_token_logps = per_token_logps * mask

        # 4. 对有效长度求和，得到整句话的对数概率
        return per_token_logps.sum(-1) # 在序列长度 T 上求和，得到每个样本整个回答序列的对数概率，形状 [B]

    def forward(self, policy_logits, ref_logits, y_combined, m_combined):
        """
        计算 DPO 核心 Loss
        """
        # 计算新旧模型的对数概率
        policy_logprobs = self.get_batch_logprobs(policy_logits, y_combined, m_combined)
        ref_logprobs = self.get_batch_logprobs(ref_logits, y_combined, m_combined)

        # 拆分为 Chosen 和 Rejected
        # policy_logits / ref_logits：形状 [2N, T, V]（前 N 个是 chosen，后 N 个是 rejected）。
        pi_logrbs_c, pi_logrbs_r = policy_logprobs.chunk(2, dim=0)
        ref_logrbs_c, ref_logrbs_r = ref_logprobs.chunk(2, dim=0)

        # 算出隐式奖励 (Implicit Reward)
        reward_chosen = self.beta * (pi_logrbs_c - ref_logrbs_c)
        reward_rejected = self.beta * (pi_logrbs_r - ref_logrbs_r)

        # 核心损失：使 chosen 的奖励尽可能大于 rejected 的奖励
        logits_diff = reward_chosen - reward_rejected
        loss = -F.logsigmoid(logits_diff).mean()

        return loss, reward_chosen.mean().item(), reward_rejected.mean().item()


# ==========================================
# 3. 训练器模块 (Trainer)
# ==========================================
class DPOTrainer:
    def __init__(self, policy_model, ref_model, optimizer, config: DPOConfig):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = DPOLoss(beta=config.beta)

    def _prepare_batch(self, batch):
        """将数据移至GPU，并拼接 chosen 和 rejected 以节省显存"""
        x_c, y_c, m_c = batch['x_chosen'].to(self.config.device), batch['y_chosen'].to(self.config.device), batch[
            'mask_chosen'].to(self.config.device)
        x_r, y_r, m_r = batch['x_rejected'].to(self.config.device), batch['y_rejected'].to(self.config.device), batch[
            'mask_rejected'].to(self.config.device)

        x_combined = torch.cat([x_c, x_r], dim=0)
        y_combined = torch.cat([y_c, y_r], dim=0)
        m_combined = torch.cat([m_c, m_r], dim=0)

        return x_combined, y_combined, m_combined

    def step(self, batch):
        """执行单步训练"""
        x_combined, y_combined, m_combined = self._prepare_batch(batch)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. 策略模型前向传播 (带梯度)
            self.policy_model.train()
            policy_logits = self.policy_model(x_combined).logits

            # 2. 参考模型前向传播 (无梯度)
            with torch.no_grad():
                self.ref_model.eval()
                ref_logits = self.ref_model(x_combined).logits

            # 3. 计算 Loss
            loss, ch_r, re_r = self.loss_fn(policy_logits, ref_logits, y_combined, m_combined)

        # 4. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), ch_r, re_r

    def train(self, dataloader):
        """完整训练循环"""
        print("🔥 开始 DPO 训练...")
        for epoch in range(self.config.epochs):
            pbar = tqdm(dataloader, desc=f"DPO Epoch {epoch + 1}/{self.config.epochs}")

            for batch in pbar:
                loss, ch_r, re_r = self.step(batch)

                # 正常现象：Ch_R 稳步上升，且逐渐大于 Re_R
                pbar.set_postfix({
                    "Loss": f"{loss:.4f}",
                    "Ch_R": f"{ch_r:.3f}",
                    "Re_R": f"{re_r:.3f}"
                })
        print("\n🎉 DPO 训练完成！")


# ==========================================
# 4. 主程序入口 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 初始化配置
    cfg = DPOConfig()

    print("🚀 正在初始化环境与 Tokenizer...")
    model_dir = os.path.join(root_dir, 'model')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("🧠 正在初始化 DPO 双模型 (Policy, Reference)...")
    dpo_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8)
    policy_model = MiniMindForCausalLM(dpo_config).to(cfg.device)
    ref_model = MiniMindForCausalLM(dpo_config).to(cfg.device)

    # 加载 SFT 权重
    sft_model_path = os.path.join(root_dir, 'out', 'full_sft_768.pth')
    if os.path.exists(sft_model_path):
        print(f"📦 正在加载 SFT 权重: {sft_model_path}")
        state_dict = torch.load(sft_model_path, map_location=cfg.device)
        policy_model.load_state_dict(state_dict, strict=False)
        ref_model.load_state_dict(state_dict, strict=False)
        print("✅ SFT 权重加载成功！")
    else:
        print("⚠️ 未找到 SFT 权重，将从头开始训练！")

    ref_model.eval().requires_grad_(False)
    optimizer = AdamW(policy_model.parameters(), lr=cfg.learning_rate)

    print("📚 正在加载数据...")
    dataset_path = os.path.join(root_dir, 'dataset', 'test_dpo.jsonl')
    train_ds = DPODataset(file_path=dataset_path, tokenizer=tokenizer, max_length=cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # 实例化 Trainer 并启动
    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        config=cfg
    )

    trainer.train(dataloader=train_loader)

    # 保存权重
    save_dir = os.path.join(root_dir, 'out')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'dpo_minimind.pth')
    print(f"\n💾 正在保存 DPO 训练后的权重至: {save_path}")
    torch.save({k: v.cpu() for k, v in policy_model.state_dict().items()}, save_path)
    print("✅ 权重保存成功！")