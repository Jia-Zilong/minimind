import os
import sys
import re
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 假设你的路径和自建模块导入
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset


# ==========================================
# 1. 配置模块 (Configuration)
# 将所有超参数集中管理，防止散落在代码各处
# ==========================================
@dataclass
class GRPOConfig:
    batch_size: int = 4  # B: 批次中有几个问题
    group_size: int = 4  # G: 每个问题生成几个回答来“内卷”
    max_seq_len: int = 128
    max_gen_len: int = 150
    grpo_epochs: int = 3  # 每次 Rollout 后策略更新的轮数
    learning_rate: float = 1e-5
    temperature: float = 0.85
    clip_eps: float = 0.2  # PPO/GRPO 剪切阈值
    beta_kl: float = 0.04  # KL 惩罚系数
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. 奖励模型模块 (Reward System)
# 负责对环境/生成结果进行打分评估
# ==========================================
class MockRewardModel:
    def __init__(self, device):
        self.device = device

    def _rep_penalty(self, text, n=3, cap=0.5):
        toks = re.findall(r"\w+|[^\w\s]", text.lower())
        grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        if not grams: return 0.0
        return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))

    def get_reward(self, response_texts):
        """输入一组回答，返回打分张量 [B*G]"""
        rewards = []
        for response in response_texts:
            score = 0.0
            score += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            answer = response
            if '</think>' in response:
                try:
                    thinking_content, answer_content = response.split('</think>', 1)
                    score += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    score += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                except ValueError:
                    score -= 0.5
            score -= self._rep_penalty(answer)
            if "请" in answer or "谢谢" in answer or "您" in answer:
                score += 0.5
            rewards.append(score)
        return torch.tensor(rewards, dtype=torch.bfloat16, device=self.device)


# ==========================================
# 3. 核心损失函数模块 (Loss Computation)
# 专门负责数学计算：优势值与 PPO/KL Loss
# ==========================================
class GRPOLoss(torch.nn.Module):
    def __init__(self, clip_eps=0.2, beta_kl=0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl

    def compute_group_advantage(self, rewards_tensor, b_size, g_size):
        """GRPO 的灵魂：组内归一化计算优势值"""
        rewards_matrix = rewards_tensor.view(b_size, g_size)
        group_mean = rewards_matrix.mean(dim=1, keepdim=True)
        group_std = rewards_matrix.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_matrix - group_mean) / group_std
        return advantages.view(-1)

    def forward(self, new_logits, old_logprobs, ref_logprobs, actions, advantages, mask):
        """计算最终的 GRPO 损失"""
        policy_dist = torch.distributions.Categorical(logits=new_logits)
        new_logprobs = policy_dist.log_prob(actions)

        # 1. 策略损失 (Clipped Surrogate)
        ratio = torch.exp(new_logprobs - old_logprobs)
        adv_expanded = advantages.unsqueeze(1).expand_as(ratio)

        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded
        policy_loss = - (torch.min(surr1, surr2) * mask).sum() / mask.sum()

        # 2. KL 散度惩罚 (DeepSeek-R1 改进版)
        kl_div = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0
        kl_loss = (kl_div * mask).sum() / mask.sum()

        total_loss = policy_loss + self.beta_kl * kl_loss
        return total_loss, policy_loss, kl_loss


# ==========================================
# 4. 训练器模块 (Trainer)
# 统筹规划：数据流转、模型 Rollout、反向传播
# ==========================================
class GRPOTrainer:
    def __init__(self, actor, ref_model, reward_model, tokenizer, optimizer, config: GRPOConfig):
        self.actor = actor
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.loss_fn = GRPOLoss(clip_eps=config.clip_eps, beta_kl=config.beta_kl)

    def _prepare_inputs(self, batch_prompts):
        """将问题复制 G 份，构建输入"""
        repeated_prompts = [p for p in batch_prompts for _ in range(self.config.group_size)]
        inputs = self.tokenizer(
            repeated_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.config.max_seq_len
        ).to(self.config.device)
        return inputs.input_ids, len(repeated_prompts)

    @torch.no_grad()
    def rollout_phase(self, prompt_ids):
        """阶段 A: 生成回答，获取原始概率与奖励"""
        self.actor.eval()
        prompt_length = prompt_ids.size(1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. 采样生成
            output_ids = self.actor.generate(
                prompt_ids, max_new_tokens=self.config.max_gen_len,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True, temperature=self.config.temperature
            )
            generated_ids = output_ids[:, prompt_length:]
            mask = (generated_ids != self.tokenizer.pad_token_id).float()

            # 2. 奖励模型打分
            responses_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            terminal_rewards = self.reward_model.get_reward(responses_text)

            # 3. 提取旧概率与参考概率
            actor_logits = self.actor(output_ids).logits[:, prompt_length - 1:-1, :]
            ref_logits = self.ref_model(output_ids).logits[:, prompt_length - 1:-1, :]

            old_logprobs = torch.distributions.Categorical(logits=actor_logits).log_prob(generated_ids)
            ref_logprobs = torch.distributions.Categorical(logits=ref_logits).log_prob(generated_ids)

        return generated_ids, mask, terminal_rewards, old_logprobs.detach(), ref_logprobs.detach()

    def update_phase(self, prompt_ids, generated_ids, mask, terminal_rewards, old_logprobs, ref_logprobs,
                     current_bg_size):
        """阶段 B & C: 计算优势，执行多轮网络更新"""
        self.actor.train()
        prompt_length = prompt_ids.size(1)
        actual_b_size = current_bg_size // self.config.group_size

        # 计算优势值
        advantages = self.loss_fn.compute_group_advantage(terminal_rewards, actual_b_size,
                                                          self.config.group_size).detach()
        train_input_ids = torch.cat([prompt_ids, generated_ids], dim=1)

        # 记录最后一次更新的 loss 供面板显示
        last_p_loss, last_kl_loss = 0.0, 0.0

        for _ in range(self.config.grpo_epochs):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                new_logits = self.actor(train_input_ids).logits[:, prompt_length - 1:-1, :]

                loss, p_loss, kl_loss = self.loss_fn(
                    new_logits=new_logits, old_logprobs=old_logprobs,
                    ref_logprobs=ref_logprobs, actions=generated_ids.clone(),
                    advantages=advantages, mask=mask
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.optimizer.step()

            last_p_loss, last_kl_loss = p_loss.item(), kl_loss.item()

        return terminal_rewards.mean().item(), last_p_loss, last_kl_loss

    def train(self, dataloader, epochs):
        """主训练循环"""
        print(f"🔥 开始纯血 GRPO 训练 (Epochs: {epochs})")
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"GRPO Epoch {epoch + 1}/{epochs}")

            for batch in pbar:
                batch_prompts = batch['prompt']

                # 1. 准备输入
                prompt_ids, current_bg_size = self._prepare_inputs(batch_prompts)

                # 2. Rollout 阶段 (生成 & 评估)
                generated_ids, mask, terminal_rewards, old_logprobs, ref_logprobs = self.rollout_phase(prompt_ids)

                # 3. Update 阶段 (优化 Actor)
                avg_reward, p_loss, kl_loss = self.update_phase(
                    prompt_ids, generated_ids, mask, terminal_rewards,
                    old_logprobs, ref_logprobs, current_bg_size
                )

                # 4. 更新面板
                pbar.set_postfix({"Rwd": f"{avg_reward:.2f}", "Act_L": f"{p_loss:.4f}", "KL": f"{kl_loss:.4f}"})

        print("\n🎉 GRPO 训练完美跑通！")


# ==========================================
# 5. 主程序入口 (Main Execution)
# ==========================================
if __name__ == "__main__":
    # 实例化配置
    cfg = GRPOConfig()

    print("🚀 正在初始化环境与 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_dir, 'model'))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("🧠 正在初始化 GRPO 模型 (Actor, Ref)...")
    ppo_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8)
    actor = MiniMindForCausalLM(ppo_config).to(cfg.device)
    ref_model = MiniMindForCausalLM(ppo_config).to(cfg.device)

    sft_model_path = os.path.join(root_dir, 'out', 'full_sft_768.pth')
    if os.path.exists(sft_model_path):
        state_dict = torch.load(sft_model_path, map_location=cfg.device)
        actor.load_state_dict(state_dict, strict=False)
        ref_model.load_state_dict(state_dict, strict=False)
        print("✅ SFT 权重加载成功！")

    ref_model.eval().requires_grad_(False)

    # 实例化打分器与优化器
    reward_model = MockRewardModel(cfg.device)
    optimizer = AdamW(actor.parameters(), lr=cfg.learning_rate)

    print("📚 正在加载数据...")
    dataset_path = os.path.join(root_dir, 'dataset', 'test_rlaif.jsonl')
    train_ds = RLAIFDataset(jsonl_path=dataset_path, tokenizer=tokenizer, max_length=cfg.max_seq_len,
                            thinking_ratio=0.5)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    # 实例化 Trainer 并开始训练
    trainer = GRPOTrainer(
        actor=actor, ref_model=ref_model, reward_model=reward_model,
        tokenizer=tokenizer, optimizer=optimizer, config=cfg
    )

    trainer.train(dataloader=train_loader, epochs=1)

    # 保存权重
    save_dir = os.path.join(root_dir, 'out')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'grpo_test.pth')
    torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, save_path)
    print(f"💾 Actor 权重已保存至: {save_path}")