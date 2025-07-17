import asyncio
import tiktoken  # 用于Token计数和分段
from toolz.curried import filter, pipe
import numpy as np
from typing import List, get_type_hints
from openai import AsyncOpenAI

llm = AsyncOpenAI()


# async def remote_embedding(input: str) -> List[float]:
#     data = (
#         [input] if len(input) < 8192 else [line for line in input.split("\n") if line]
#     )
#     tasks = [
#         llm.embeddings.create(input=x, model="text-embedding-3-small") for x in data
#     ]
#     resps = await asyncio.gather(*tasks)
#     embeddings = [resp.data[0].embedding for resp in resps]
#     return np.mean(embeddings, axis=0).tolist()


llm = AsyncOpenAI()


async def remote_embedding(input: str) -> List[float]:
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # text-embedding-3-small 使用此编码器
    max_tokens_per_chunk = 8192  # OpenAI embedding API 的最大 token 限制

    def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
        tokens = encoding.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
            if i + max_tokens >= len(tokens):
                break
            i += max_tokens - overlap  # 移动步长，考虑重叠
            if i < 0:  # 避免负数索引在某些边缘情况下
                i = 0
        return chunks

    if len(encoding.encode(input)) <= max_tokens_per_chunk:
        data = [input]
    else:
        data = chunk_text(input, max_tokens_per_chunk, overlap=0)

    batch_size = 200  # 示例批次大小，实际应测试优化
    all_embeddings = []

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        # print(f"Processing batch from index {i} to {i + len(batch)}") # 调试

        # 确保 batch 不为空
        if not batch:
            continue

        try:
            resp = await llm.embeddings.create(
                input=batch, model="text-embedding-3-small"
            )
            # resp.data 是一个 Embedding 对象列表，每个对象有 .embedding 属性
            batch_embeddings = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # 根据需求处理错误，例如跳过此批次或重试

    if not all_embeddings:
        return []  # 或者抛出异常，根据你希望的行为

    embeddings_np = np.array(all_embeddings, dtype=np.float64)

    averaged_embedding_np = np.mean(embeddings_np, axis=0)

    return averaged_embedding_np.tolist()
