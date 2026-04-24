import asyncio
import time
from typing import List, Dict
from src.llm.async_llm_client import AsyncOpenAILLM
from src.llm.async_embedding_client import AsyncOpenAIEmbedding

async def run_embedding_tests():
    print("🚀 开始初始化 Embedding 客户端...")
    
    # 1. 初始化客户端
    # 注意：base_url 依然指向你通过 SSH 隧道映射的本地端口
    # model: 请替换为你服务器上实际部署的 Embedding 模型名称（如 "bge-m3", "text-embedding-3-small" 等）
    client = AsyncOpenAIEmbedding(
        api_key="EMPTY", 
        base_url="http://127.0.0.1:8082/v1", 
        model="Qwen3-Embedding-8B", # <-- 这里记得改成你的模型名
        max_concurrent=5
    )

    try:
        # ==========================================
        # 测试 1：基础单条 Embedding 测试
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 1: 基础单条 Embedding 生成")
        print("="*50)
        
        text1 = "人工智能正在改变世界。"
        print(f"输入文本: '{text1}'")
        
        start_time = time.time()
        embedding_vector = await client.get_embedding(text1)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 耗时: {elapsed:.2f} 秒")
        print(f"📊 向量维度: {len(embedding_vector)}")
        print(f"🔢 向量前5个数值: {embedding_vector[:5]} ...")

        # ==========================================
        # 测试 2：批量 Embedding 测试 (验证并发与分批)
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 2: 批量 Embedding 生成")
        print("="*50)
        
        batch_texts = [
            "苹果是一种很好吃的水果。",
            "香蕉通常是黄色的。",
            "量子力学是物理学的一个分支。",
            "机器学习属于人工智能的范畴。",
            "今天天气真不错，适合出去玩。"
        ]
        print(f"准备处理 {len(batch_texts)} 条文本...")
        
        start_time = time.time()
        # 调用批量获取方法
        batch_embeddings = await client.get_embeddings(batch_texts)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 批量处理总耗时: {elapsed:.2f} 秒")
        print(f"✅ 成功获取了 {len(batch_embeddings)} 个向量。")
        for i, emb in enumerate(batch_embeddings):
            print(f"  -> 文本 {i+1} 向量维度: {len(emb)}")

        # ==========================================
        # 测试 3：缓存机制验证
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 3: 缓存机制验证 (Cache命中测试)")
        print("="*50)
        
        # 使用第一步和第二步中已经处理过的文本
        cache_test_texts = [
            "人工智能正在改变世界。", # 在测试1中处理过
            "苹果是一种很好吃的水果。"  # 在测试2中处理过
        ]
        
        start_time = time.time()
        cached_embeddings = await client.get_embeddings(cache_test_texts)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 缓存命中耗时 (应当接近 0 秒): {elapsed:.4f} 秒")
        print(f"✅ 成功从缓存获取了 {len(cached_embeddings)} 个向量。")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        print("👉 请检查：")
        print("   1. 你的 LLM 服务端是否同时支持 Embedding 接口 (通常是 `/v1/embeddings`)。")
        print("   2. `model` 参数是否填入了你服务器正确加载的 Embedding 模型名称。")

    finally:
        # 释放连接池资源
        print("\n🧹 清理资源，关闭客户端连接...")
        await client.close()
        print("✅ 测试完成！")

async def run_tests():
    print("🚀 开始初始化 LLM 客户端...")
    
    # 1. 初始化客户端
    # 注意：base_url 指向你通过 SSH 隧道映射的本地端口
    # api_key: 如果你的本地开源 LLM 服务不需要 Key，可以随便填一个非空字符串
    # model: 请替换为你服务器上实际部署的模型名称（如 "qwen-max", "llama-3-8b" 等）
    llm = AsyncOpenAILLM(
        api_key="EMPTY", 
        base_url="http://127.0.0.1:8081/v1", 
        model="Qwen3-32B", # <-- 这里记得改成你的模型名
        temperature=0.7,
        max_concurrent=10 # 测试并发控制
    )

    try:
        # ==========================================
        # 测试 1：基础单次生成测试
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 1: 基础单次调用 (无 Usage)")
        print("="*50)
        
        single_msg = [{"role": "user", "content": "用一句话解释什么是量子计算。"}]
        
        start_time = time.time()
        response = await llm.generate(single_msg)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 耗时: {elapsed:.2f} 秒")
        print(f"🤖 回复: {response}")

        # ==========================================
        # 测试 2：带 Token 统计的单次生成测试
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 2: 带 Usage 统计的单次调用")
        print("="*50)
        
        usage_msg = [{"role": "user", "content": "写一首五言绝句描写春天。"}]
        
        response_text, usage = await llm.generate(usage_msg, return_usage=True)
        print(f"🤖 回复: {response_text}")
        print(f"📊 Token统计 -> Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")

        # ==========================================
        # 测试 3：高并发批量测试
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 3: 并发批量调用 (验证 Semaphore 和 异步性能)")
        print("="*50)
        
        # 准备 5 个不同的并发请求
        batch_messages = [
            [{"role": "user", "content": "中国的首都是哪里？只回答城市名。"}],
            [{"role": "user", "content": "1 + 1 等于几？只回答数字。"}],
            [{"role": "user", "content": "地球是圆的还是平的？简单回答。"}],
            [{"role": "user", "content": "光速大约是多少 km/s？"}],
            [{"role": "user", "content": "水是由哪两种元素组成的？"}]
        ]
        
        start_time = time.time()
        # 调用你的并发方法
        batch_results, batch_usages = await llm.generate_batch(batch_messages, return_usage=True)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 批量请求 5 个任务总耗时: {elapsed:.2f} 秒")
        for i, (text, usage) in enumerate(zip(batch_results, batch_usages)):
            print(f"任务 {i+1} 结果: [{text.strip()}] (耗费 Tokens: {usage.total_tokens})")

        # ==========================================
        # 测试 4：缓存机制验证
        # ==========================================
        print("\n" + "="*50)
        print("🧪 测试 4: 缓存机制验证 (Cache命中测试)")
        print("="*50)
        
        cache_msg = [{"role": "user", "content": "用一句话解释什么是量子计算。"}] # 与测试1完全相同
        
        start_time = time.time()
        # 注意：你的代码中写了 return_usage=True 时不走缓存，所以这里设为 False
        cached_response = await llm.generate(cache_msg, return_usage=False)
        elapsed = time.time() - start_time
        
        print(f"⏱️ 缓存命中耗时 (应当接近 0 秒): {elapsed:.4f} 秒")
        print(f"🤖 回复: {cached_response}")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        print("👉 请检查：")
        print("   1. SSH 隧道是否仍然存活。")
        print("   2. 服务器 8081 端口的 LLM 服务是否正常运行。")
        print("   3. `model` 参数是否填入了你服务器支持的模型名称。")

    finally:
        # 必须显式关闭 httpx 客户端，释放连接池资源
        print("\n🧹 清理资源，关闭客户端连接...")
        await llm.close()
        print("✅ 测试完成！")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(run_embedding_tests())
    asyncio.run(run_tests())
    