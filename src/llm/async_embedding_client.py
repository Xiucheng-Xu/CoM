"""异步Embedding客户端 - 极致性能优化版本"""
import httpx
import asyncio
from typing import List, Optional


class AsyncOpenAIEmbedding:
    """异步OpenAI Embedding客户端，支持高并发"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None,
                 model: str = "text-embedding-3-small",
                 timeout: float = 60.0,
                 max_concurrent: int = 10,
                 max_token_limit: int = 8192,
                 chars_per_token: float = 3.5):
        """
        初始化异步Embedding客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: Embedding模型名称
            timeout: 请求超时时间（秒）
            max_concurrent: 最大并发数
            max_token_limit: 模型最大token数
            chars_per_token: 每token对应的字符数估算（3.5适合纯英文，2.0适合中英混合）
        """
        if not api_key or api_key.strip() == "":
            api_key = "NONE"
        
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/') if base_url else "https://api.openai.com/v1"
        self.timeout = timeout
        self._max_chars = int(max_token_limit * chars_per_token * 0.90)
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 缓存
        self._cache = {}
        
        # 共享的 httpx 客户端（复用连接池）
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建共享的 httpx 客户端（线程安全）"""
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                # 双重检查
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
        """关闭客户端连接"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """
        异步获取单个文本的embedding
        
        Args:
            text: 输入文本
            max_retries: 最大重试次数
            
        Returns:
            Embedding向量
        """
        # 截断超长文本
        if len(text) > self._max_chars:
            text = text[:self._max_chars]
        
        # 检查缓存
        cache_key = (self.model, text)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 获取共享客户端
        client = await self._get_client()
        
        # 使用信号量控制并发
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "input": text
                        }
                    )
                    
                    if response.status_code != 200:
                        raise Exception(f"API返回错误状态码: {response.status_code}, {response.text}")
                    
                    result = response.json()
                    if "data" not in result:
                        raise Exception(f"API返回格式异常，缺少'data'字段，完整返回: {str(result)[:500]}")
                    embedding = result["data"][0]["embedding"]
                    
                    # 缓存结果
                    self._cache[cache_key] = embedding
                    
                    return embedding
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Embedding API调用失败: {str(e)}")
                    
                    # 指数退避
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
    
    async def get_embeddings(self, texts: List[str], 
                           batch_size: int = 200,
                           max_retries: int = 3) -> List[List[float]]:
        """
        批量异步获取多段文本的embedding
        
        Args:
            texts: 文本列表
            batch_size: 每批处理的文本数量
            max_retries: 最大重试次数
            
        Returns:
            Embedding向量列表
        """
        # 截断超长文本
        texts = [t[:self._max_chars] if len(t) > self._max_chars else t for t in texts]
        
        # 先检查缓存
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for idx, text in enumerate(texts):
            cache_key = (self.model, text)
            if cache_key in self._cache:
                results[idx] = self._cache[cache_key]
            else:
                uncached_indices.append(idx)
                uncached_texts.append(text)
        
        if not uncached_texts:
            return results
        
        # 获取共享客户端
        client = await self._get_client()
        
        # 分批处理未缓存的文本
        async def process_batch(batch_texts: List[str], batch_indices: List[int]):
            """处理一批文本"""
            async with self.semaphore:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            f"{self.base_url}/embeddings",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "input": batch_texts
                            }
                        )
                        
                        if response.status_code != 200:
                            raise Exception(f"API返回错误状态码: {response.status_code}, {response.text[:500]}")
                        
                        result = response.json()
                        if "data" not in result:
                            raise Exception(f"API返回格式异常，缺少'data'字段，完整返回: {str(result)[:500]}")
                        
                        for i, emb_data in enumerate(result["data"]):
                            embedding = emb_data["embedding"]
                            orig_idx = batch_indices[i]
                            results[orig_idx] = embedding
                            # 缓存
                            self._cache[(self.model, batch_texts[i])] = embedding
                        
                        return
                    
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"批量Embedding API调用失败: {str(e)}")
                        
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
        
        # 创建批次任务
        tasks = []
        for i in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[i:i+batch_size]
            batch_indices = uncached_indices[i:i+batch_size]
            tasks.append(process_batch(batch_texts, batch_indices))
        
        # 并发执行所有批次
        await asyncio.gather(*tasks)
        
        # 返回结果（所有位置都应该已填充）
        return [r if r is not None else [] for r in results]
