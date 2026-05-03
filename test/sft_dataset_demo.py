import torch
from transformers import AutoTokenizer


def demo_sft_processing():
    print("1. 正在加载 Tokenizer (这里以 Qwen2.5-0.5B 为例演示)...")
    # 你也可以把它换成 './model/tokenizer' 用你本地的 MiniMind 分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # 获取特殊的标识符 (不同模型的标识符不一样，这里适配 Qwen/MiniMind 体系)
    # bos_id 用于定位 assistant 开始说话的地方
    bos_text = "<|im_start|>assistant\n" # im:Interactive Message
    eos_text = "<|im_end|>\n"

    bos_id = tokenizer(bos_text, add_special_tokens=False).input_ids
    eos_id = tokenizer(eos_text, add_special_tokens=False).input_ids
    print(f"   [定位锚点] Assistant 开始符 ID: {bos_id}")
    print(f"   [定位锚点] Assistant 结束符 ID: {eos_id}\n")

    # 2. 模拟一条 JSONL 里的对话数据
    conversation = [
        {"role": "user", "content": "1+1等于几？"},
        {"role": "assistant", "content": "1+1等于2。"}
    ]

    # 3. 拼接对话 (相当于原代码的 create_chat_prompt)
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    print("2. 拼接后的完整 Raw String:")
    print("-" * 30)
    print(prompt)
    print("-" * 30 + "\n")

    # 4. Tokenize 成 Input IDs
    input_ids = tokenizer(prompt).input_ids

    # 5. 生成 Labels (核心算法！！！)
    labels = [-100] * len(input_ids)  # 第一步：全盘否定，全部设为 -100
    '''
    选 -1 也可以，选 -999 也可以。只要它不在正常 Token ID 的范围（$\ge 0$）内，就不会和正常的字发生冲突。PyTorch 官方当年拍脑袋选了 -100，后来就成了整个深度学习界的行业行规。
    '''

    i = 0
    while i < len(input_ids):
        # 如果匹配到了 "<|im_start|>assistant\n" 的 Token ID 序列
        if input_ids[i:i + len(bos_id)] == bos_id:
            start = i + len(bos_id)  # 记录有效回答的起点
            end = start

            # 一直往后找，直到找到结束符 "<|im_end|>\n"
            while end < len(input_ids):
                if input_ids[end:end + len(eos_id)] == eos_id:
                    break
                end += 1

            # 第二步：拨乱反正，把起点到终点区间的 label 恢复成正确的 Input ID
            for j in range(start, end + len(eos_id)):
                labels[j] = input_ids[j]

            i = end + len(eos_id)
        else:
            i += 1

        '''
        为什么assistant回答的末尾的|im_end|不屏蔽？
        1. 如果屏蔽掉 <|im_end|> 会发生什么？（灾难现场）
        假设我们把 <|im_end|> 也设为了 -100。
        
        模型在训练时，学到了遇到“1+1等于”要输出“2”，遇到“2”要输出“。”。
        
        当时间步走到“。”时，它预测下一个词。但因为下一个词 <|im_end|> 被你屏蔽了（-100），所以哪怕它瞎猜了一个“猪”，或者猜了一个“3”，系统的 Loss 也是 0，不会给它任何惩罚。
        
        部署推理时的后果： 当用户问“1+1等于几？”时，模型流畅地回答“1+1等于2。”，然后呢？因为它从来没学过在句号后面应该输出结束符，它就会开始漫无目的地胡言乱语（比如接着写“2+2等于4。这道题真简单……”），直到达到你设定的最大生成长度（Max New Tokens）才被迫强行切断。
        
        2. 为什么必须保留 <|im_end|>？（交出麦克风）
        我们要让模型学到的完整逻辑不仅仅是“如何回答问题”，更重要的是“回答完毕后，主动交出控制权”。
        
        当我们保留 <|im_end|> 作为有效的 Label 时：
        
        当模型走到“。”时，它的目标词（Label）必须是 <|im_end|>。
        
        如果它敢猜别的字，Loss 就会飙升，梯度就会狠狠地教训它。
        
        通过成千上万次的训练，模型形成了一种“肌肉记忆”：只要我的意思表达完整了，下一个 Token 我必须雷打不动地输出 <|im_end|>。
        '''

    # 6. 对齐打印展示魔法效果
    print("3. 最终的 Token vs Label 映射关系 (注意看 -100):")
    print(f"{'索引':<5} | {'Token 文本':<20} | {'Input ID':<10} | {'Label (目标)'}")
    print("-" * 60)

    for idx, (x, y) in enumerate(zip(input_ids, labels)):
        # 把 ID 解码回文字方便人类阅读
        token_str = tokenizer.decode([x])
        # 把换行符转义一下，不然打印会乱
        token_str = token_str.replace('\n', '\\n')

        # 为了美观，做一下颜色标记 (-100是灰色，正常是绿色)
        label_str = str(y)
        if y == -100:
            label_display = f"\033[90m{label_str:<8} (不计算)\033[0m"
        else:
            label_display = f"\033[92m{label_str:<8} (要计算)\033[0m"

        print(f"{idx:<5} | {token_str:<20} | {x:<10} | {label_display}")


if __name__ == "__main__":
    demo_sft_processing()