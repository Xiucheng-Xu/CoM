"""异步LLM客户端 - 极致性能优化版本"""
import httpx
import asyncio
import json
from typing import List, Dict, Optional, Any, Tuple
import time


class AsyncOpenAILLM:
    """异步OpenAI LLM客户端，支持高并发"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, 
                 model: str = "gpt-4o-mini", temperature: float = 0.0, 
                 max_tokens: int = 8192, timeout: float = 120.0,
                 max_concurrent: int = 20):
        """
        初始化异步LLM客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 请求超时时间（秒）
            max_concurrent: 最大并发数
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        if not api_key or api_key.strip() == "":
            api_key = "NONE"
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else "https://api.openai.com/v1"
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 共享的 httpx 客户端（复用连接池，避免连接耗尽）
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        
        # 缓存
        self._cache = {}
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(
                        timeout=self.timeout,
                        limits=httpx.Limits(
                            max_connections=500,
                            max_keepalive_connections=100
                        )
                    )
        return self._client
    
    async def close(self):
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def generate(self, messages: List[Dict[str, str]], 
                      return_usage: bool = False,
                      max_retries: int = 3) -> Any:
        """
        异步生成回复
        
        Args:
            messages: 消息列表
            return_usage: 是否返回token使用统计
            max_retries: 最大重试次数
            
        Returns:
            如果return_usage=False: 返回回复文本
            如果return_usage=True: 返回(回复文本, usage对象)
        """
        # 检查缓存（仅在不需要usage时）
        if not return_usage:
            msg_key = tuple((m.get("role", ""), m.get("content", "")) for m in messages)
            cache_key = (self.model, self.temperature, self.max_tokens, msg_key)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # 使用信号量控制并发
        async with self.semaphore:
            client = await self._get_client()
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"API返回错误状态码: {response.status_code}, {response.text}")
                    
                    result = response.json()
                    
                    if not result.get("choices") or len(result["choices"]) == 0:
                        raise Exception("API返回的响应无效或为空")
                    
                    content = result["choices"][0]["message"]["content"]
                    
                    if content is None:
                        raise Exception("API返回的内容为空")
                    
                    if not return_usage:
                        self._cache[cache_key] = content
                    
                    if return_usage:
                        usage = result.get("usage", {})
                        class Usage:
                            def __init__(self, data):
                                self.prompt_tokens = data.get("prompt_tokens", 0)
                                self.completion_tokens = data.get("completion_tokens", 0)
                                self.total_tokens = data.get("total_tokens", 0)
                        return (content, Usage(usage))
                    else:
                        return content
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"API调用失败（重试{max_retries}次后）: {str(e)}")
                    
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
    
    async def generate_batch(self, messages_list: List[List[Dict[str, str]]], 
                           return_usage: bool = False) -> Any:
        """
        批量异步生成回复（真正并发）
        
        Args:
            messages_list: 消息列表的列表
            return_usage: 是否返回token使用统计
            
        Returns:
            如果return_usage=False: List[str]
            如果return_usage=True: (List[str], List[usage对象])
        """
        # 创建所有任务
        tasks = [self.generate(msgs, return_usage=return_usage) for msgs in messages_list]
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果和异常
        outputs = []
        usages = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 如果某个请求失败，返回空字符串
                print(f"警告: 第{i}个请求失败: {str(result)}")
                outputs.append("")
                if return_usage:
                    class EmptyUsage:
                        prompt_tokens = 0
                        completion_tokens = 0
                        total_tokens = 0
                    usages.append(EmptyUsage())
            else:
                if return_usage:
                    outputs.append(result[0])
                    usages.append(result[1])
                else:
                    outputs.append(result)
        
        return (outputs, usages) if return_usage else outputs
