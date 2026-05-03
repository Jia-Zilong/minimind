import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# ==========================================
# 0. 路径定位与核心组件导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
max_seq_len = 128
max_gen_len = 150

print("🚀 正在加载本地 Tokenizer...")
model_dir = os.path.join(root_dir, 'model')
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ==========================================
# 1. 显式定义 PPO 四大核心模型
# ==========================================
print("🧠 正在初始化 PPO 四大模型 (Actor, Ref, Critic, Reward)...")
ppo_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8)

actor = MiniMindForCausalLM(ppo_config).to(device)
ref_model = MiniMindForCausalLM(ppo_config).to(device)


class CriticModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = MiniMindForCausalLM(config)
        self.value_head = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids):
        hidden_states = self.base.model(input_ids)[0]
        return self.value_head(hidden_states).squeeze(-1)  # [B, Seq_Len]


critic = CriticModel(ppo_config).to(device)

sft_model_path = os.path.join(root_dir, 'out', 'full_sft_768.pth')
if os.path.exists(sft_model_path):
    print(f"📦 正在加载 SFT 权重作为 PPO 起点: {sft_model_path}")
    state_dict = torch.load(sft_model_path, map_location=device)
    actor.load_state_dict(state_dict, strict=False)
    ref_model.load_state_dict(state_dict, strict=False)
    critic.base.load_state_dict(state_dict, strict=False)
    print("✅ SFT 权重加载成功！")

actor.train()
ref_model.eval().requires_grad_(False)
critic.train()

# --- Reward 模型部分保持不变 ---
import re


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams: return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


class MockRewardModel: # 处理纯文本，对文本进行打分
    def __init__(self, device):
        self.device = device

    def get_reward(self, response_texts): # 输入的是Batch_Size 句话（比如 ["等于2。", "我是AI。"]）。
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
            score -= rep_penalty(answer)
            if "请" in answer or "谢谢" in answer or "您" in answer:
                score += 0.5
            rewards.append(score)
        return torch.tensor(rewards, dtype=torch.bfloat16, device=self.device) # 输出：经过一系列 if-else 后，返回了一个形状为 [Batch_Size] 的一维张量（比如 tensor([1.25, -0.5])）


'''
            第一关：存在性检查 (if '</think>' in response)
            作用：设立准入门槛。

            物理意义：如果模型生成的文本里压根没有这个标签，它就拿不到任何后续的高分，甚至可能因为字数不够被扣分。这会逼迫模型在生成时，必须先敲出 <think> 并以 </think> 结尾。

            第二关：逻辑密度奖励 (20 <= len(...) <= 300)
            作用：打击“假装思考”。

            物理意义：

            如果太短（<20字）：模型可能会耍小聪明，写一句“我在思考...”就直接给答案。这种无效思考会被扣 0.5分。

            如果适中（20-300字）：这是黄金区间。模型必须真的写点逻辑推演才能填满这部分。一旦达到，直接奖励 1.0分（全场最高奖励）。

            如果太长（>300字）：虽然代码里没写上限，但通常为了防止模型陷入死循环（只思考不给答案），我们会通过 max_gen_len 来限制。

            第三关：格式严谨性奖励 (count('</think>') == 1)
            作用：规范输出协议。

            物理意义：有些模型在 PPO 初期会因为概率抖动，输出多个 </think> 或者标签不闭合。通过这个判断，奖励那些格式工整、严格遵守 Markdown 或 XML 规范的模型。
'''

reward_model = MockRewardModel(device)
optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=1e-5) # 把「策略模型 (Actor)」和「价值模型 (Critic)」的所有可训练参数，合并成一个参数列表，交给 AdamW 优化器统一管理、一起更新！


# ==========================================
# 算法核心模块：GAE 与 PPO Loss
# ==========================================
def compute_gae(rewards_tensor, values, gamma=0.99, lam=0.95):
    """
    输入:
        rewards_tensor: [B, T] 每个 token 获得的奖励 (通常只有最后一个 token 有奖励)
        values: [B, T] Critic 预测的每个 token 的状态价值
    """
    advantages = torch.zeros_like(rewards_tensor)
    returns = torch.zeros_like(rewards_tensor)
    next_value = 0.0

    # GAE 从后往前计算
    for t in reversed(range(rewards_tensor.size(1))):
        delta = rewards_tensor[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = delta + gamma * lam * (advantages[:, t + 1] if t + 1 < rewards_tensor.size(1) else 0.0)
        returns[:, t] = advantages[:, t] + values[:, t]  # Return = Advantage + Value
        next_value = values[:, t]

    return advantages, returns # 返回size:[B, T]


def ppo_loss_fn(new_logits, old_logprobs, ref_logprobs, actions, advantages, returns, values, clip_eps=0.2,
                value_coef=0.5, entropy_coef=0.01):
    """
    修复点：使用 Rollout 时真实生成的 actions 去取 log_prob，而不是重新 sample()
    """
    policy_dist = torch.distributions.Categorical(logits=new_logits) # [B,seq_len,vocab_size]
    ref_dist = torch.distributions.Categorical(logits=ref_logits)

    # 获取真实执行动作的概率
    new_logprobs = policy_dist.log_prob(actions) # [B,seq_len]

    # 1. 策略损失 (Clipped Surrogate)
    ratio = torch.exp(new_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 2. 价值损失 (MSE)
    value_loss = F.mse_loss(values, returns)

    # 3. 熵损失 (鼓励探索)
    entropy_loss = -policy_dist.entropy().mean()

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    return total_loss, policy_loss, value_loss, entropy_loss


# ==========================================
# 2. 加载数据
# ==========================================
print("📚 正在使用 RLAIFDataset 加载数据...")
dataset_path = os.path.join(root_dir, 'dataset', 'test_rlaif.jsonl')
train_ds = RLAIFDataset(jsonl_path=dataset_path, tokenizer=tokenizer, max_length=max_seq_len, thinking_ratio=0.5)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# ==========================================
# 3. PPO 完整时序训练循环
# ==========================================
epochs = 1
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"PPO Epoch {epoch + 1}/{epochs}")

    for batch in pbar:
        batch_prompts = batch['prompt']
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_seq_len).to(device)
        prompt_ids = inputs.input_ids
        prompt_length = prompt_ids.size(1)

        # ------------------------------------------
        # 阶段 A: Rollout
        # ------------------------------------------
        actor.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Actor 开启采样生成
                output_ids = actor.generate(prompt_ids, max_new_tokens=max_gen_len, pad_token_id=tokenizer.eos_token_id,
                                            do_sample=True, temperature=0.8)
                generated_ids = output_ids[:, prompt_length:]  # 这就是我们要评估的 actions [B, T]
                # 在 Hugging Face 的底座模型（如 LLaMA, Qwen, MiniMind）中，当你调用 .generate() 函数时，模型吐出来的结果（output_ids）并不是单纯的回答，而是“问题 + 回答”的拼接体。
                # 假设输入的 prompt_ids 长度是 5。模型生成了 3 个字的回答。那么 output_ids 的总长度会是 8。所以这里切片出回答部分
                T = generated_ids.size(1)

                responses_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                terminal_rewards = reward_model.get_reward(responses_text)  # [B]
                # 输入：是一个普通的 Python 列表，里面装了 Batch_Size 句话（比如 ["等于2。", "我是AI。"]）。
                # 输出：经过一系列 if-else 后，返回了一个形状为 [Batch_Size] 的一维张量（比如 tensor([1.25, -0.5])）

                # 💡 核心构建：将单点奖励转化为时序奖励矩阵 [B, T]
                token_rewards = torch.zeros((batch_size, T), device=device, dtype=torch.bfloat16)
                token_rewards[:, -1] = terminal_rewards  # 假设奖励只在最后一句话说完时结算，由于裁判只能对完整的一句话打分，所以这 +10.0 分只放在最后一个字“。”上。
                # 最终的 token_rewards 矩阵：
                # [
                #   [0.0,  0.0,  10.0],  <-- 10.0 放在了第一句话的句尾
                #   [0.0,  0.0,  -5.0]   <-- -5.0 放在了第二句话的句尾
                # ]

                # 获取各个模型的时序输出
                actor_logits = actor(output_ids).logits[:, prompt_length - 1:-1, :]
                ref_logits = ref_model(output_ids).logits[:, prompt_length - 1:-1, :]
                '''
                大模型输出的 logits 是一个三维张量，形状为 [Batch_Size, Sequence_Length, Vocab_Size]。
                : （第一维）：取所有批次（Batch）。
                prompt_length - 1:-1 （第二维，最关键）：取序列长度中特定的那一段。
                : （第三维）：取全部词表大小（比如 6400），因为我们要得到所有的概率分布，才能知道哪个词最有可能。
                
                为什么是 -1 结束？
                在 Python 切片里，-1 代表“倒数第一个元素（不包含本身）”。
                
                回顾刚才的例子：
                
                最后一个生成的字是“的”（索引 5）。
                
                预测“的”的 Logit 存在于索引 4。
                
                索引 5 的 Logit 是用来预测这句话之后的废话（比如 <eos>）的，我们根本不需要它，因为它不在我们要评估的 actions 范围内。
                
                所以，我们要把最后一个没用的 Logit 砍掉，切片切到 -1 为止。
                '''

                # 使用真实生成的动作 ID 提取旧对数概率
                old_dist = torch.distributions.Categorical(logits=actor_logits)
                old_logprobs = old_dist.log_prob(generated_ids) # [Batch_Size, Sequence_Length]
                '''
                old_dist 是什么？ 它不是一个普通的张量（Tensor），它是一个 PyTorch 的概率分布对象（Distribution Object）。

                它的内部结构：当你把 [B, T, Vocab_Size] 喂给 Categorical 时，PyTorch 会自动把最后一个维度（Vocab_Size，也就是 64000）当成候选类别。它会在底层构建出一个形状为 [B, T] 的“分布矩阵”。
                
                通俗理解：你可以把 old_dist 想象成一个 [2, 3] 的大柜子。柜子里有 2 行 3 列，一共 6 个抽屉。每一个抽屉里，都装着一份包含了 64000 个词的打分表。
                
                PyTorch 看到 generated_ids 的形状也是 [B, T]，正好和 old_dist 柜子的形状一一对应。

                它会走到第 1 行第 1 列的抽屉，拿出那份 64000 个词的打分表。
                
                然后它看了一眼 generated_ids 第 1 行第 1 列填的词是 872（“宫”字）。
                
                它就在打分表里查出：“哦，生成『宫』字当时的对数概率是 -1.2”。
                
                查完之后，它把这个 -1.2 记下来。
                '''

                ref_dist = torch.distributions.Categorical(logits=ref_logits)
                ref_logprobs = ref_dist.log_prob(generated_ids)

                # 获取 Critic 对每一步的估值 [B, T],reward则是对生成的总体回答的打分
                old_values = critic(output_ids)[:, prompt_length:] # 只选择answer的价值v

        # ------------------------------------------
        # 阶段 B: 计算 GAE 与 Returns
        # ------------------------------------------
        advantages, returns = compute_gae(token_rewards, old_values) # advantages 和 returns 的 Size（维度）都是 [Batch_Size, Sequence_Length]
        '''
        advantages（优势值）：这个字表现得“比预期好多少”。
        returns（真实回报）：这个字及之后的所有真实得分的总和（用于后续给 Critic 当“标准答案”来训练）。
        '''

        # 优势归一化 (让训练更稳定)，防止突然冒出一个很大的advantage，导致梯度爆炸
        if advantages.shape[0] * advantages.shape[1] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = advantages.detach() # 对这些后续不用更新的，剥离梯度计算
        returns = returns.detach()
        old_logprobs = old_logprobs.detach()
        ref_logprobs = ref_logprobs.detach()

        # ------------------------------------------
        # 阶段 C: PPO 更新
        # ------------------------------------------
        actor.train()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 💡 必须用训练模式的 actor 重新过一遍完整的 prompt_ids + generated_ids
            # 不要直接复用 generate 吐出来的 output_ids

            # 把 prompt 和刚刚生成的回答拼起来，作为训练数据
            train_input_ids = torch.cat([prompt_ids, generated_ids], dim=1)

            # 让 actor 重新看一遍完整的句子（带梯度）
            new_logits = actor(train_input_ids).logits[:, prompt_length - 1:-1, :] # [:, prompt_length-1:-1, :] → 只取出「回答部分」的 logits
            new_values = critic(train_input_ids)[:, prompt_length:]
            '''
            注意这里的train_input_ids:PPO 训练需要的是「已经生成的回答」的概率 / 价值我们不是要让模型重新生成回答，而是要：
            让模型重新看一遍刚才生成的完整回答，算出每一步生成这个回答时的概率 (new_logits) 和 价值 (new_values)，用来更新策略！
            
            actor.generate() (采样生成)：这是一个循环过程。模型每蹦出一个词，就把这个词拼到输入里，再跑一遍模型。
            它确实是在“续写”。
            actor(train_input_ids) (前向传播)：这是一个静态的过程。你给它一个长度为 N的序列，它就只看这N个词，然后一次性吐出N组Logits。
            它绝对不会自动在后面多吐出一个 N+1 的词。
            '''

            # 调用分离的 PPO 损失函数
            loss, p_loss, v_loss, e_loss = ppo_loss_fn(
                new_logits=new_logits,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                actions=generated_ids.clone(),  # 这里的 clone 是为了避免报错
                advantages=advantages,
                returns=returns,
                values=new_values
            )

        optimizer.zero_grad()
        loss.backward()
        '''
        actor和critic共用同一个优化器，optimizer = AdamW(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)，二者一起训练：
        它发现 total_loss 里包含了 p_loss，于是它顺着计算图爬到了 Actor 的神经网络里，计算出 Actor 权重的梯度。
        它发现 total_loss 里也包含了 v_loss，于是它顺着计算图爬到了 Critic 的神经网络里，计算出 Critic 权重的梯度。
        也就是说，这一句 backward()，同时把 Actor 和 Critic 两个大脑的梯度都算好了！
        '''
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()

        # ------------------------------------------
        # 面板更新
        # ------------------------------------------
        pbar.set_postfix({
            "Rwd": f"{terminal_rewards.mean().item():.2f}",
            "Act_L": f"{p_loss.item():.4f}",
            "Cri_L": f"{v_loss.item():.4f}",
            "Ent": f"{e_loss.item():.4f}"
        })

print("\n🎉 带有 GAE 和完整 PPO Loss 的终极训练跑通了！")

save_dir = os.path.join(root_dir, 'out')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'ppo_test_gae.pth')
print(f"\n💾 正在保存 PPO 训练后的 Actor 权重至: {save_path}")
torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, save_path)
print("✅ 权重保存成功！")

'''
为什么只保存actor的权重：
1. 任务使命不同
Actor (演员)：它的任务是 “做决策”。它负责把 Prompt 转换成人类能看懂的文字。你部署模型到网页上或者 App 里去陪用户聊天，实际上运行的就是 Actor 的权重。

Critic (评论家)：它的任务是 “估值”。它只在训练过程中，帮 Actor 算 Advantage（优势值）时才有用。当模型上线面对真实用户时，用户会直接给出反馈，我们不再需要 Critic 来预测能拿多少分。

2. 模型结构的差异
如果你观察你的代码，你会发现：

Actor 的输出维度是 [Vocab_Size]（比如 6400），它输出的是词。

Critic 的输出维度是 [1]，它输出的是一个浮点数。

当你训练结束后，你的目的是得到一个会说话的机器人。如果你加载了 Critic 的权重，你得到的只是一个“只会给问题打分，却一个字都吐不出来”的哑巴模型。从推理部署的角度看，Critic 没有任何价值。
'''