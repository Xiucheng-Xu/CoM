"""LLM 客户端封装"""
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import List, Dict, Optional
import time


class BaseLLM(ABC):
    """LLM 抽象基类，定义统一接口"""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], return_usage: bool = False):
        """
        生成回复
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            return_usage: 是否返回token使用统计
            
        Returns:
            如果 return_usage=False: 返回模型生成的回复文本
            如果 return_usage=True: 返回 (回复文本, usage对象)
        """
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM 实现"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, 
                 model: str = "gpt-4o", temperature: float = 0.0, 
                 max_tokens: int = 2000):
        """
        初始化 OpenAI LLM 客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL（可选）
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not api_key or api_key.strip() == "":
            api_key = "NONE"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # 简单进程内缓存，仅用于 return_usage=False 的调用
        # key: (model, temperature, max_tokens, tuple(serialized messages)) -> str(content)
        self._cache = {}
    
    def generate(self, messages: List[Dict[str, str]], return_usage: bool = False):
        """
        生成回复
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            return_usage: 是否返回token使用统计
            
        Returns:
            如果 return_usage=False: 返回模型生成的回复文本
            如果 return_usage=True: 返回 (回复文本, usage对象)
        """
        # 仅在不需要 usage 时启用缓存
        if not return_usage:
            # 将消息序列化为不可变结构作为 key
            msg_key = tuple((m.get("role", ""), m.get("content", "")) for m in messages)
            cache_key = (self.model, self.temperature, self.max_tokens, msg_key)
            if cache_key in self._cache:
                return self._cache[cache_key]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                
                # 检查响应是否有效
                if not response or not response.choices or len(response.choices) == 0:
                    raise Exception("API返回的响应无效或为空")
                
                content = response.choices[0].message.content
                
                # 检查内容是否为空
                if content is None:
                    raise Exception("API返回的内容为空")
                
                if return_usage:
                    return (content, response.usage)
                else:
                    # 写入缓存
                    if not return_usage:
                        self._cache[cache_key] = content
                    return content
                
            except Exception as e:
                error_msg = f"API调用失败 (尝试 {attempt + 1}/3): {str(e)}"
                if attempt == 2:
                    raise Exception(error_msg)
                print(f"警告: {error_msg}，正在重试...")
                time.sleep(2 ** attempt)

    def generate_batch(self, messages_list: List[List[Dict[str, str]]], return_usage: bool = False):
        """
        批量生成回复（顺序调用，复用单条 generate 的缓存与重试机制）。
        返回：
          - return_usage=False: List[str]
          - return_usage=True: Tuple[List[str], List[Any]]  # usage 列表
        """
        outputs = []
        usages = []
        for msgs in messages_list:
            if return_usage:
                content, usage = self.generate(msgs, return_usage=True)
                outputs.append(content)
                usages.append(usage)
            else:
                content = self.generate(msgs, return_usage=False)
                outputs.append(content)
        return (outputs, usages) if return_usage else outputs

