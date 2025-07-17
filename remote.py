import asyncio
import os
import tiktoken
from toolz.curried import filter, pipe
import numpy as np
from typing import List
from openai import AsyncOpenAI

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_BASE_URL")
if not api_key or not base_url:
    print("QWEN_API_KEY or QWEN_URL is not set")
    quit()

llm = AsyncOpenAI(api_key=api_key, base_url=base_url)
model = "text-embedding-v4"


async def remote_embedding(input: str) -> List[float]:
    # 预处理输入，去除多余空格和换行
    input = " ".join(input.strip().split())

    # 使用适用于 text-embedding-v4 的编码器
    encoding = tiktoken.get_encoding("o200k_base")
    max_tokens_per_chunk = 3000  # 预留安全空间

    def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
        tokens = encoding.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            end = i + max_tokens
            chunk_tokens = tokens[i:end]
            chunk = encoding.decode(chunk_tokens)

            # 再次检查 token 数是否超标
            while len(encoding.encode(chunk)) > max_tokens:
                end -= 1
                chunk_tokens = tokens[i:end]
                chunk = encoding.decode(chunk_tokens)
                if i >= end:
                    break  # 防止死循环

            # ✅ 新增：跳过空 chunk
            if not chunk.strip():
                i += max_tokens - overlap
                continue

            chunks.append(chunk)

            if i + max_tokens >= len(tokens):
                break
            i += max_tokens - overlap
            if i < 0:
                i = 0
        return chunks

    # 分块处理
    token_count = len(encoding.encode(input))
    if token_count <= max_tokens_per_chunk:
        data = [input]
    else:
        data = chunk_text(input, max_tokens_per_chunk, overlap=0)

    # 打印每个 chunk 的 token 数量用于调试
    for idx, chunk in enumerate(data):
        print(f"Chunk {idx} token count: {len(encoding.encode(chunk))}")

    batch_size = 20  # 并发 batch 大小（Qwen 限制建议控制在 20 以内）
    all_embeddings = []

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        if not batch:
            continue

        # ✅ 新增：再次过滤空内容和超标 chunk
        valid_batch = [
            chunk
            for chunk in batch
            if chunk.strip() and len(encoding.encode(chunk)) <= max_tokens_per_chunk
        ]
        if len(valid_batch) != len(batch):
            print(f"Filtered out {len(batch) - len(valid_batch)} invalid chunks")

        # ✅ 使用 asyncio.gather 实现并发请求
        tasks = [fetch_embedding(chunk, model, llm, encoding) for chunk in valid_batch]
        try:
            batch_embeddings = await asyncio.gather(*tasks)
            all_embeddings.extend([e for e in batch_embeddings if e is not None])
        except Exception as e:
            print(f"Error processing batch: {e}")

    if not all_embeddings:
        return []

    embeddings_np = np.array(all_embeddings, dtype=np.float64)
    averaged_embedding_np = np.mean(embeddings_np, axis=0)
    return averaged_embedding_np.tolist()


# ✅ 单个请求函数，支持重试和错误处理
async def fetch_embedding(
    chunk: str, model: str, client: AsyncOpenAI, encoding, retry=3
):
    for i in range(retry):
        try:
            resp = await client.embeddings.create(input=chunk, model=model)
            return resp.data[0].embedding
        except Exception as e:
            print(f"Attempt {i + 1} failed for chunk '{chunk[:50]}...': {e}")
            await asyncio.sleep(1)
    return None
