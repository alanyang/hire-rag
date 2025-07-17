import os
import asyncio
import aiomysql
import numpy as np
from chromadb import PersistentClient
# from sentence_transformers import SentenceTransformer

from remote import remote_embedding


# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")


env = os.getenv("ENV") or "dev"
path = "/app/blueapex" if env == "production" else "/root/develop/rag/blueapex"

client = PersistentClient(path=path)
col = client.get_or_create_collection(name="hire")


def split_text(text, max_length=512):
    # 按字符分段，简单实现
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


async def mean_embedding_data(last_id: int, limit: int = 10) -> int:
    try:
        conn = await aiomysql.connect(
            host="8.217.197.114",
            port=3306,
            user="root",
            password="BB2LjIwei4Cs9XXNPhWx",
            db="hire",
        )
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    p.description,
                    p.prices,
                    c.name AS category_name,
                    comp.name AS company_name
                FROM
                    product AS p
                JOIN
                    category AS c ON p.category_id = c.id
                JOIN
                    company AS comp ON p.company_id = comp.id
                WHERE
                    p.id > %s
                ORDER BY
                    p.id
                LIMIT %s;
                """,
                (last_id, limit),
            )
            rows = await cur.fetchall()

        nearest_id: int = 0
        ids, docs, embeddings = [], [], []

        if rows:
            for row in rows:
                nearest_id = row[0]
                id = str(row[0])
                doc = f"{row[1]}\nCategory: {row[4]}\n Company:{row[5]} \n {row[2]} {row[3]}"
                # 分段
                # segments = split_text(doc, max_length=512)
                # 批量embedding
                # segment_embeddings = model.encode(segments, device="cuda")
                # 聚合（取平均）
                # embedding = np.mean(segment_embeddings, axis=0).tolist()

                embedding = await remote_embedding(doc)
                ids.append(id)
                docs.append(row[1])  # 原始文档name
                embeddings.append(embedding)
                print(f"Prepared document {id}")

            # 批量写入
            col.add(ids=ids, documents=docs, embeddings=embeddings)

        await conn.ensure_closed()
        return nearest_id

    except aiomysql.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 0


async def main():
    last_id = 1
    while last_id < 200:
        last_id = await mean_embedding_data(last_id=last_id, limit=3)


if __name__ == "__main__":
    asyncio.run(main())
