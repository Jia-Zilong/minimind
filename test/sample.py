import random
import os


def create_mini_dataset():
    # 1. 配置路径
    # 假设原数据集在 dataset 文件夹下，根据你的实际情况修改文件名
    source_file = "../dataset/sft_t2t_mini.jsonl"
    target_file = "../dataset/test_sft.jsonl"

    # 你的 3050Ti 跑 1 小时左右，抽取 10000 条是最合适的甜点区
    sample_size = 10000

    if not os.path.exists(source_file):
        print(f"❌ 找不到源文件：{source_file}，请检查路径是否正确！")
        return

    print(f"⏳ 正在将大文件加载到内存: {source_file} ...")
    with open(source_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_lines = len(all_lines)
    print(f"✅ 加载成功！原数据集共有 {total_lines} 条数据。")

    # 如果原数据比我们想要的还少，就直接全拿
    if total_lines <= sample_size:
        print(f"⚠️ 原数据量小于 {sample_size}，将使用全部数据。")
        sampled_lines = all_lines
    else:
        # 2. 核心操作：全局随机抽样
        print(f"🎲 正在从 {total_lines} 条中随机抽取 {sample_size} 条，保证数据多样性...")
        sampled_lines = random.sample(all_lines, sample_size)

    # 3. 写入新文件
    print(f"💾 正在写入目标文件: {target_file} ...")
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(sampled_lines)

    print(f"🎉 搞定！抽样完成。你的训练集现已准备就绪，大小约 {os.path.getsize(target_file) / (1024 * 1024):.2f} MB。")


if __name__ == "__main__":
    create_mini_dataset()