from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以80%概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids #input_ids输出bos_id的分词结果
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    '''
    features:在大模型训练中，我们面对的往往是几个 G 甚至几十个 G 的 JSONL 文件。如果数据格式稍微有一点错（比如本来该是字符串的地方填成了数字），整个训练就会在中途崩溃。
    features 就是用来强制规定数据长什么样的。它就像数据库建表时的 Schema（表结构）。
    features = Features({
    'conversations': [{
        'role': Value('string'), 
        'content': Value('string'), 
        'reasoning_content': Value('string'), 
        'tools': Value('string'), 
        'tool_calls': Value('string')
    }]
    })
    因为大模型的训练数据里，有时候为了方便，可能没有 tools 这一项，或者 reasoning_content 是空的。有了这个 features 蓝图，即使遇到残缺的数据，datasets 库也会自动用 None 或空字符串帮你补齐，保证所有数据排列得像军队一样整齐，绝不报错。
    
    samples:
    这行代码执行后，self.samples 就变成了一个极其强大的 Hugging Face Dataset 对象。

    你可以把它理解为一个**“超级增强版的 Python 列表（List）”**。它里面装的就是你硬盘上那个 .jsonl 文件里所有的对话数据，但是已经被 features 规范化了。
    它的厉害之处在于（底层机制）：
    如果你加载了 50GB 的数据，普通的 Python json.load() 会直接把你的内存撑爆。但 load_dataset 底层使用了 Apache Arrow 技术，它不会把 50GB 全塞进内存，而是放在硬盘上，你需要第几条数据，它就瞬间去硬盘上取第几条（内存映射），这也就是为什么几十万条数据的 SFT 训练在普通电脑上也能跑的原因。
    
    例子：
    {"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
    执行：print(self.samples[0])
    输出{
    'conversations': [
        {
            'role': 'user', 
            'content': '你好', 
            'reasoning_content': None, 
            'tools': None, 
            'tool_calls': None
        }, 
        {
            'role': 'assistant', 
            'content': '你好！', 
            'reasoning_content': None, 
            'tools': None, 
            'tool_calls': None
        }
    ]
}
    '''
    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations): #它把面向程序员的结构化数据（JSON），翻译并排版成了面向大模型底层神经网络的纯文本格式，并且严格贴上了防伪的特殊标签！
        messages = []
        tools = None
        for message in conversations:
            # dict(message) 是为了拷贝一下，防止修改原数据
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False, # 当add_generation_prompt=True时，会自动在对话末尾加上<|im_start|>assistant\n，用于推理环节，这里是训练环节，不需要引导模型说话
            tools=tools
        )
    '''
    create_chat_prompt的作用：
    假设你的模型被设定为一个能查天气的智能体（Agent）。在原始的 JSONL 数据集里，tools 往往被粗暴地存成了一个字符串。输入给 create_chat_prompt 的数据：
    [
    {
        "role": "system", 
        "content": "你可以使用工具。", 
        # 注意：这里在原始数据中可能是一大坨字符串格式的 JSON
        "tools": "[{\"name\": \"get_weather\", \"description\": \"获取天气\"}]" 
    },
    {"role": "user", "content": "北京天气如何？"}
    ]
    create_chat_prompt 内部的清洗动作：它识别到 tools 是一个字符串，于是执行了 json.loads()，把它在内存里变成了一个真正的 Python 列表：[{"name": "get_weather", "description": "获取天气"}]。然后传给模板。
    经过json.loads后的结果：
    <|im_start|>user
    北京天气如何？<|im_end|>
    <|im_start|>assistant
    <tool_call>
    {"name": "get_weather", "arguments": {"city": "北京"}}
    </tool_call><|im_end|>
    '''

    def generate_labels(self, input_ids):
        # 让所有input_id变为-100，-100是交叉熵损失默认忽略的值
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations) # 把conversations转化为chat格式
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 截断prompt
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) # 如果input_ids太短，则pad对齐
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt) # 准备训练数据的时候，数据里就已经包含了<think>\n\n</think>\n\n，这是噪音。

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        # 动作：这是标准的自回归训练写法。x 是输入序列（去掉最后一个字）。y 是预测目标（去掉第一个字）。意义：让模型学习“看到第n个词，预测第 n+1 个词”。
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }
    '''
    DPO 的核心是「偏好对齐」，不是「强制思考」
    DPO 的重点是让模型偏好好回答、远离坏回答，思考标签只是辅助，不是必须。所以空的就去掉，有内容的（如果有的话）才保留。
    '''

    def generate_loss_mask(self, input_ids): # 选取回答部分，对prompt部分进行掩码处理
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio # 掷骰子决定要不要强制思考
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking, # 追加一个 <think>\n，直接替模型把思考的开头写好
            add_generation_prompt=True # # 当add_generation_prompt=True时，会自动在对话末尾加上<|im_start|>assistant\n
        )
    '''
    假设原始的 conversations 是这样的 4 轮对话（包含最后的空回答）：
        [用户问1, 助手答1, 用户问2, 助手答空]
        conversations[:-1]：在 Python 中，[:-1] 代表切片，意思是“抛弃列表里的最后一个元素”。
        所以，那个用来占位的 助手答空 直接被扔进垃圾桶了！此时列表里只剩前 3 轮真实的上下文。
        
    ❌ 当 tokenize=True 时（默认情况）
    分词器会做两件事：
    
    先把对话套进模型的专属模板里（比如加上 <|im_start|> 等特殊字符）。
    
    立刻把它切成词，并转换成数字 ID。
    
    输出结果：[151644, 872, 108336, 151645, 151644, 77091, 198] （这是一个 Python 列表或 PyTorch Tensor）。
    
    适用场景：当你只有一条数据，且想立刻把它喂给模型推理时。
    
    ✅ 当 tokenize=False 时（你的代码用法）
    分词器只做一件事：
    
    只套用模板，进行字符串拼接，绝不进行数字转换。
    
    输出结果："<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" （这是一段纯文本字符串）。
    
    适用场景：构建 Dataset、需要查看打印结果、或者需要把多条文本聚拢后统一处理时。
    '''
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass