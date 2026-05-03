import os

# 🌟 加上这一句，解决 Matplotlib 和 PyTorch 的冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt  # 🌟 新增：导入绘图库

# 导入 MiniMind 仓库里的模型和数据集类
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from dataset.lm_dataset import SFTDataset


def main():
    # ================= 0. 动态计算项目的根目录绝对路径 =================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"当前项目根目录已自动识别为: {BASE_DIR}")

    # ================= 1. 基础配置 (针对 4GB 显存优化) =================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 1
    accumulation_steps = 8
    # ⚠️ 友情提示：既然数据量到了 1 万条，建议把 epochs 调小。1万条跑20轮容易严重过拟合。建议先设为 3 到 5 看看效果。
    epochs = 3
    learning_rate = 5e-5

    # 用 BASE_DIR 拼接所有文件的绝对路径
    pretrain_weight_path = os.path.join(BASE_DIR, "out", "pretrain_768.pth")
    tokenizer_path = os.path.join(BASE_DIR, "model")
    # ⚠️ 请确保这里填的是你那 1 万条数据的文件名
    dataset_path = os.path.join(BASE_DIR, "dataset", "test_sft.jsonl")

    # ================= 2. 加载 Tokenizer 和 数据 =================
    print(f"正在从 {tokenizer_path} 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"正在加载数据集: {dataset_path} ...")
    # 如果你的1万条对话比较长，建议把 max_length 稍微调大（比如256），前提是显存不爆
    train_ds = SFTDataset(dataset_path, tokenizer, max_length=256)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ================= 3. 初始化模型 =================
    print("正在初始化 MiniMind 模型...")
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, num_attention_heads=8)
    model = MiniMindForCausalLM(lm_config).to(device)

    if os.path.exists(pretrain_weight_path):
        model.load_state_dict(torch.load(pretrain_weight_path, map_location=device))
        print("成功加载预训练权重！")
    else:
        print("未找到预训练权重，从头开始随机训练（仅供学习流程）。")

    model.train()

    # ================= 4. 优化器 =================
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ================= 5. 核心 SFT 训练循环 =================
    print("🚀 开始极简 SFT 训练...")
    step = 0
    nan_count = 0

    # 🌟 新增：用于记录每次更新的 Loss 列表
    loss_history = []

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for input_ids, labels in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if (labels == -100).all():
                nan_count += 1
                pbar.set_postfix({'Loss': 'N/A', 'Skipped': nan_count})
                continue

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                res = model(input_ids, labels=labels)
                loss = res.loss / accumulation_steps

            if torch.isnan(loss):
                nan_count += 1
                optimizer.zero_grad()
                pbar.set_postfix({'Loss': 'NaN', 'Skipped': nan_count})
                continue
            '''
            在干什么：模型拿到一个 Batch 的数据（input_ids 和 labels）后，先做个“安检”。
            检查这批数据里是不是所有的标签都被设为了 -100（这通常是因为数据太长，回答部分被截断了，只剩下了不参与计算的提问部分）。
            为什么重要：如果没有这句话，模型去算一堆全是一百的 Loss，会直接爆出 NaN（Not a Number），瞬间污染模型的所有权重。这相当于在吃东西前，先把有毒的剔除掉。
            '''


            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                current_loss = loss.item() * accumulation_steps
                '''
                为什么需要 .item()？（张量 vs. 数值）
                在大模型训练中，你的 loss 变量并不是一个简单的浮点数，它是一个住在 GPU 里的 Tensor 对象。
                
                loss (Tensor)：它像是一个带着“家谱”的数字。它不仅存着当前的损失值，还背负着复杂的计算图（Computation Graph）。通过它，PyTorch 知道如何一路往回找，找到每一个神经元并计算梯度。
                
                loss.item() (Float)：它是一个“孑然一身”的普通 Python 浮点数。它只代表那个数值，切断了与计算图、GPU 显存以及所有梯度信息的联系。
                '''

                # 🌟 新增：将当前真实的 loss 记录到列表中
                loss_history.append(current_loss)

                pbar.set_postfix({'Loss': f"{current_loss:.4f}", 'Skipped': nan_count})

            step += 1

    # ================= 6. 保存 SFT 后的模型 =================
    save_path = os.path.join(BASE_DIR, "out", "full_sft_test_768.pth")
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练完成，模型已保存为 {save_path}")

    # ================= 7. 绘制并保存 Loss 曲线 =================
    print("📊 正在绘制 Loss 曲线...")
    plt.figure(figsize=(10, 6))

    # 绘制原始波动的 Loss 曲线 (设置透明度使其作为背景)
    plt.plot(loss_history, label='Raw Training Loss', color='blue', alpha=0.3)

    # 计算并绘制平滑曲线 (滑动平均法)，使趋势更清晰
    if len(loss_history) > 50:
        window_size = min(50, len(loss_history) // 10)
        smoothed_loss = [sum(loss_history[i - window_size:i]) / window_size for i in
                         range(window_size, len(loss_history))]
        plt.plot(range(window_size, len(loss_history)), smoothed_loss, label='Smoothed Loss (Moving Avg)', color='red',
                 linewidth=2)

    plt.title('SFT Training Loss Curve')
    plt.xlabel('Update Steps (Accumulated)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图片到 out 目录
    plot_path = os.path.join(BASE_DIR, "out", "sft_loss_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Loss 曲线图已成功保存至: {plot_path}")


if __name__ == "__main__":
    main()