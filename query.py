import asyncio
import os
from typing import cast
from toolz.curried import pipe, map
from toolz.dicttoolz import get_in
from chromadb import PersistentClient
from openai import AsyncOpenAI


llm = AsyncOpenAI()

openai_api_key = os.getenv("OPENAI_API_KEY")

env = os.getenv("ENV") or "dev"


async def query(input: str, top_k: int = 5) -> tuple[list[int], list[str]] | None:
    path = "/app/blueapex" if env == "production" else "/root/develop/rag/blueapex"
    db = PersistentClient(path=path)
    col = db.get_or_create_collection("hire")
    resp = await llm.embeddings.create(
        input=input,
        model="text-embedding-3-small",
    )
    if resp.data:
        embedding = resp.data[0].embedding
        loop = asyncio.get_running_loop()

        def query_wrapper(embeddings: list[float], n_results: int) -> dict:
            return cast(
                dict, col.query(query_embeddings=embeddings, n_results=n_results)
            )

        result = await loop.run_in_executor(None, query_wrapper, embedding, top_k)
        ids = cast(list[str], get_in(["ids", 0], result))
        names = cast(list[str], get_in(["documents", 0], result))
        return cast(list[int], pipe(ids, map(int), list)), names
    else:
        return None


if __name__ == "__main__":
    input = "男性，25岁，推荐几个养老保险，安盛AXA的产品优先"
    print(f"User search:{input}")
    asyncio.run(query(input=input))
