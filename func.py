import asyncio
import logging
import uvloop
import httpx
import json
from dataclasses import dataclass
from collections.abc import AsyncGenerator
from typing import cast, Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from returns.context import RequiresContext
from returns.future import Future, FutureResult, future_safe, FutureResultE, future
from returns.result import Success, ResultE, safe, Failure
from returns.io import IO, IOResult
from toolz.curried import filter, map, pipe
from toolz.dicttoolz import get_in

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

llm = AsyncOpenAI()


class Tool(BaseModel):
    name: str
    arguments_string: str
    arguments: dict


class ToolCall(BaseModel):
    id: str = Field(..., description="The id of the tool call")
    type: Literal["function"] = "function"
    function: Tool


class Event(BaseModel):
    name: Literal["chunk", "finish", "error"]
    tool: Tool | None = None
    error: str | None = None
    chunks: list[str] = []


async def stream_completion(
    input="What's the weather like in Los Angeles today?",
) -> AsyncGenerator[Event, str]:
    try:
        stream = await llm.chat.completions.create(
            model="gpt-4.1",
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": input,
                },
            ],
            tool_choice="auto",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather of a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city name",
                                },
                                "date": {
                                    "type": "string",
                                    "description": "The date",
                                },
                            },
                            "required": ["city", "date"],
                        },
                    },
                }
            ],
        )
        assembled_tool_calls: dict[int, ToolCall] = {}
        async for chunk in stream:
            tool_calls = chunk.choices[0].delta.tool_calls
            if tool_calls is None:
                continue
            # print(tool_calls)
            pieces = [tool_call.to_json() for tool_call in tool_calls]
            yield Event(name="chunk", chunks=pieces)
            for tool_call in tool_calls:
                if tool_call.index not in assembled_tool_calls:
                    assembled_tool_calls[tool_call.index] = ToolCall(
                        id="",
                        function=Tool(name="", arguments_string="", arguments={}),
                    )

                if tool_call.id:
                    assembled_tool_calls[tool_call.index].id += tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        assembled_tool_calls[
                            tool_call.index
                        ].function.name += tool_call.function.name
                    if tool_call.function.arguments:
                        assembled_tool_calls[
                            tool_call.index
                        ].function.arguments_string += tool_call.function.arguments

        for tool_call in assembled_tool_calls.values():
            arguments_result = safe(json.loads)(tool_call.function.arguments_string)
            match arguments_result:
                case Success(arguments):
                    tool_call.function.arguments = arguments
                    yield Event(name="finish", tool=tool_call.function)
                case Failure(error):
                    yield Event(name="error", error=str(error))
    except Exception as e:
        yield Event(name="error", error=str(e))


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
    async for num in stream_completion():
        print(num)
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
