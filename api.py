import asyncio
from pydantic import BaseModel, Field

from fastapi import FastAPI

from chromadb import PersistentClient

from query import query
from remote import remote_embedding

app = FastAPI()


client = PersistentClient(path="/root/develop/rag/blueapex")
col = client.get_or_create_collection(name="hire")


class EmbeddingModel(BaseModel):
    id: int = Field(..., gt=0, lt=1000000000)
    name: str = Field(..., min_length=1, max_length=100)
    category: str | None = Field(default="")
    company: str | None = Field(default="")
    description: str
    prices: str | None = Field(default="")


@app.get("/search")
async def search(s: str, n: int = 5):
    r = await query(input=s, top_k=n)
    if r:
        ids, names = r
        return {"ids": ids, "names": names}
    return {"code": 3, "error": "No results found"}


@app.post("/embedding")
async def embedding(model: EmbeddingModel):
    doc = f"{model.name}\nCategory: {model.category}\n Company:{model.company} \n {model.description} \n\n {model.prices}"
    tensor = await remote_embedding(doc)

    async def add_wrapper(
        ids: list[str], documents: list[str], embeddings: list[float]
    ):
        return col.add(ids=ids, documents=documents, embeddings=embeddings)

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, add_wrapper, [str(model.id)], [model.name], tensor)

    return {"code": 0}


@app.delete("/embeddings")
async def delete_embeddings():
    pass


@app.put("/embeddings")
async def update_embeddings():
    """
    need renew vector
    """
    pass
