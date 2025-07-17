import asyncio
import logging
import uvloop
import httpx
from dataclasses import dataclass
from typing import cast


from pydantic import BaseModel, Field
from returns.context import RequiresContext
from returns.future import Future, FutureResult, future_safe, FutureResultE
from returns.result import Success, ResultE
from returns.io import IO, IOResult
from toolz.curried import filter, map, pipe
from toolz.dicttoolz import get_in

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@future_safe
async def fetch_user_info(id: int) -> dict:
    await asyncio.sleep(0.3)
    print("fetch user")
    return {
        "id": id,
        "name": "John Doe",
        "info": "some  info",
        "contact": {"phone": "1234567890"},
    }


@future_safe
async def embedding_user_info(phone: str) -> list[float]:
    await asyncio.sleep(0.5)
    print("embedding user info")
    if phone == "16687670359":
        raise ValueError("embedding error")
    return [1.2, 3.4, 5.6, 7.8, 9.10, 11.12]


def spread_vector(v: list[float]) -> ResultE[list[float]]:
    return Success([i * 2.0322 + 100 for i in v])


class Post(BaseModel):
    id: int = Field(..., gt=0, description="id must be greater than 0")
    userId: int = Field(..., gt=0, description="userId must be greater than 0")
    title: str = Field(..., min_length=1, max_length=1024)
    body: str


@future_safe
async def fetch_posts() -> list[Post]:
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://jsonplaceholder.typicode.com/posts")
        return [Post.model_validate(v) for v in resp.json()]


async def main():
    a = (
        await fetch_user_info(1)
        .map(lambda v: cast(str, get_in(["contact", "phone"], v)))
        .bind(embedding_user_info)
        .bind(lambda v: FutureResult.from_result(spread_vector(v)))
        .awaitable()
    )

    match a:
        case IOResult(v):
            print(v.value_or([]))

    posts = await fetch_posts().awaitable()
    posts.lash(lambda e: IOResult.from_io((print(e), IO([]))[1])).map(
        lambda v: [
            Post.model_validate(
                {
                    k: val if k != "title" else val.upper()
                    for k, val in post.model_dump().items()
                }
            )
            for post in v
        ]
    )


@dataclass
class Context:
    logger: logging.Logger
    memory: dict | None = None


def fetch_product_by_id(
    id: int,
) -> RequiresContext[FutureResultE[ResultE[dict]], Context]:
    async def _fetch(ctx: Context) -> ResultE[dict]:
        ctx.logger.info(f"Fetching product by id {id}")
        return Success({"id": id, "name": "Product Name", "price": 100.0})

    return RequiresContext(future_safe(_fetch))


if __name__ == "__main__":
    assert asyncio.run(Future.from_value(11).awaitable()) == IO(11)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ctx = Context(logger=logger)
    task = (
        fetch_product_by_id(1128)
        .map(lambda v: v.map(lambda x: {**x.value_or({}), "extra_field": 123}))(ctx)
        .awaitable()
    )

    print(asyncio.run(task))

    products = [
        {"id": 1, "name": "Product 1"},
        {"id": 2, "name": "Product 2"},
        {"id": 3, "name": "Product 3"},
        {"id": 4, "name": "Product Complex 4"},
    ]

    print(
        pipe(
            products,
            filter(lambda v: v["id"] % 2 == 0),
            map(lambda v: {**v, "name": v["name"].upper()}),
            filter(lambda v: len(v["name"]) < 10),
            map(lambda v: {**v, "name": "_".join(v["name"].split(" ")).lower()}),
            list,
        )
    )

    asyncio.run(main())
