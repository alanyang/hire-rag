import asyncio
import inspect
from dataclasses import dataclass
from collections.abc import Callable, Awaitable
from typing import overload, TypeVar, Generic, cast
from pydantic import BaseModel

type ToolCallReturn = BaseModel | str

PType = TypeVar("PType", bound=BaseModel)
RType = TypeVar("RType", bound=BaseModel | str)

type ToolCall[PType, RType] = (
    Callable[[], RType]
    | Callable[[PType], RType]
    | Callable[[], Awaitable[RType]]
    | Callable[[BaseModel], Awaitable[RType]]
)


@dataclass
class Tool(Generic[RType]):
    schema: dict
    action: ToolCall

    def __call__(self, *args, **kwargs) -> Awaitable[RType]:
        result = (
            self.action(*args, **kwargs)
            if inspect.signature(self.action).parameters
            else cast(Callable[[], RType | Awaitable[RType]], self.action)()
        )
        if inspect.isawaitable(result):
            return cast(Awaitable[RType], result)
        else:

            async def inner() -> RType:
                return cast(RType, result)

            return inner()


@overload
def tool(
    func: ToolCall,
) -> Tool[BaseModel | str]: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[..., Tool[BaseModel | str]]: ...


def tool[PType: BaseModel, RType: BaseModel | str](
    func: ToolCall[PType, RType] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool[RType] | Callable[[ToolCall[PType, RType]], Tool[RType]]:
    """Decorator to mark a function as a tool."""

    def create(f: ToolCall) -> Tool:
        signs = inspect.signature(f)
        if len(signs.parameters) > 1:
            raise ValueError("Tool functions must have at most one parameter")
        first_param = next(iter(signs.parameters.values()), None)
        if first_param:
            param_type = first_param.annotation
            if param_type == inspect.Signature.empty:
                raise ValueError("Tool functions must have a type annotation")
            if not isinstance(param_type, type) or not issubclass(
                param_type, BaseModel
            ):
                raise ValueError(
                    "Tool functions must have a pydantic model as parameter"
                )
        schema = {
            "type": "function",
            "function": {
                "name": name or f.__name__,
                "description": description or inspect.getdoc(func),
                "parameters": first_param.annotation.model_json_schema()
                if first_param
                else {},
            },
        }
        return Tool(schema=schema, action=f)

    if func is None:
        return create
    else:
        return create(func)


class WeatherModel(BaseModel):
    city: str


@tool
def get_weather(param: WeatherModel) -> str:
    """Get the weather in a city"""
    return f"The weather in {param.city} is sunny"


@tool(description="Get the weather in a city asynchronously")
async def get_weather_async(param: WeatherModel) -> str:
    return f"The weather in {param.city} is sunny"


@tool(name="empty_parameter_tool")
def empty_parameter_tool() -> str:
    """Get the weather in a city"""
    return "The weather is sunny"


async def main():
    print(get_weather.schema)
    print(get_weather_async.schema)
    print(empty_parameter_tool.schema)
    print(await get_weather_async(WeatherModel(city="London")))
    print(await get_weather(WeatherModel(city="Paris")))
    print(await empty_parameter_tool({}))


if __name__ == "__main__":
    asyncio.run(main())
