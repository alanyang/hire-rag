import asyncio
import inspect
from collections.abc import Awaitable, Callable, Coroutine, Generator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, get_type_hints, overload

from pydantic import BaseModel
from returns.context import RequiresContext
from returns.future import FutureResultE, future_safe

type ToolCallReturn = BaseModel | str

PType = TypeVar("PType", bound=BaseModel)
RType = TypeVar("RType")

type ToolCall[PType, RType] = (
    Callable[[], RType]
    | Callable[[PType], RType]
    | Callable[[], Awaitable[RType]]
    | Callable[[PType], Awaitable[RType]]
    | Callable[[PType, Any], RType]
    | Callable[[PType, Any], Awaitable[RType]]
    | Callable[[Any], RType]
    | Callable[[Any], Awaitable[RType]]
)


@dataclass
class Tool(Generic[RType]):
    schema: dict
    action: ToolCall
    timeout: float = 0.0

    _task: Awaitable[RType] | None = None

    async def __await_with_timeout__(self) -> RType:
        if self._task is None:
            raise RuntimeError("Tool has not been called yet")

        if self.timeout == 0.0:
            return await self._task

        timeout_task = asyncio.create_task(asyncio.sleep(self.timeout))
        main_coroutine = cast(Coroutine[Any, Any, RType], self._task)

        main_task = asyncio.create_task(main_coroutine)
        done, pending = await asyncio.wait(
            [timeout_task, main_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if timeout_task in done:
            main_task.cancel()
            raise TimeoutError(f"Tool call timed out after {self.timeout} seconds")
        else:
            timeout_task.cancel()
            return await main_task

    def __await__(self) -> Generator[Any, None, RType]:
        return (yield from self.__await_with_timeout__().__await__())

    def __call__(self, *args, **kwargs) -> "Tool[RType]":
        params = inspect.signature(self.action).parameters
        if "ctx" not in params:
            kwargs.pop("ctx", None)
        result = (
            self.action(*args, **kwargs)
            if params
            else cast(Callable[[], RType | Awaitable[RType]], self.action)()
        )
        if inspect.isawaitable(result):
            self._task = result
        else:

            async def inner() -> RType:
                return cast(RType, result)

            self._task = inner()
        return self


@dataclass
class SafeTool(Generic[RType]):
    tool: Tool[RType]
    runner: Callable[..., FutureResultE[RType]] | None = None

    @property
    def schema(self) -> dict:
        return self.tool.schema

    def __call__(self, *args, **kwargs) -> FutureResultE[RType]:
        configured_tool = self.tool(*args, **kwargs)

        async def pure_awaitable() -> RType:
            return await configured_tool

        return future_safe(pure_awaitable)()

    def run_with_env(self) -> Callable[..., RequiresContext[FutureResultE[RType], Any]]:
        def _runner(*args, **kwargs):
            def _call_inner(ctx: Any):
                return self(*args, **{**kwargs, "ctx": ctx})

            return RequiresContext(_call_inner)

        return _runner

    def run_with_context(
        self, *args, **kwargs
    ) -> RequiresContext[FutureResultE[RType], Any]:
        def _call_inner(ctx: Any):
            return self(*args, **{**kwargs, "ctx": ctx})

        return RequiresContext(_call_inner)


@overload
def tool(
    func: ToolCall,
) -> SafeTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 0.0,
) -> Callable[..., SafeTool]: ...


def tool[PType: BaseModel, RType](
    func: ToolCall[PType, RType] | None = None,
    name: str | None = None,
    *,
    description: str | None = None,
    timeout: float = 0.0,
) -> SafeTool[RType] | Callable[[ToolCall[PType, RType]], SafeTool[RType]]:
    """Decorator to mark a function as a tool."""

    def create(f: ToolCall) -> SafeTool[RType]:
        signs = inspect.signature(f)
        valid_params = [
            p
            for p in signs.parameters.values()
            if p.POSITIONAL_ONLY or p.POSITIONAL_OR_KEYWORD
        ]
        if len(valid_params) > 2:
            raise ValueError(
                "Tool functions must have at most two optional parameters, BaseModel object for logic and Context"
            )
        first_param = next(iter(valid_params), None)
        param_type: Any = None
        if first_param:
            hints = get_type_hints(f)
            param_type = hints.get(first_param.name, None) or first_param.annotation
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
                "parameters": param_type.model_json_schema() if param_type else {},
            },
        }
        tool: Tool[RType] = Tool(
            schema=schema,
            action=f,
            timeout=timeout,
        )
        return safe_tool(tool)

    return create(func) if func is not None else create


def safe_tool(tool: Tool[RType] | SafeTool[RType]) -> SafeTool[RType]:
    if isinstance(tool, SafeTool):
        return tool
    return SafeTool(tool=tool)


class WeatherModel(BaseModel):
    city: str


class IPModel(BaseModel):
    ip: str


class WeatherAlertModel(BaseModel):
    weather: str


class WeatherResult(BaseModel):
    description: str
    yesterday_weather: str
    alert: str | None


@dataclass
class TestContext:
    id: str
    yesterday_weather: str


@tool(description="Get the weather in a city asynchronously", timeout=1.5)
async def get_weather(param: WeatherModel, ctx: TestContext) -> str:
    await asyncio.sleep(0.3)
    return f"The weather in {param.city} is sunny, yestody is {ctx.yesterday_weather}"


@tool
async def get_location_by_ip(ip: IPModel) -> WeatherModel:
    """Get the location by ip"""
    return WeatherModel(city="Shanghai")


@tool
async def get_alert(param: WeatherAlertModel, ctx: TestContext) -> WeatherResult:
    """Get the weather alert"""
    return WeatherResult(
        alert="No weather alert",
        yesterday_weather=f"Yesterday is {ctx.yesterday_weather}",
        description=param.weather,
    )


async def main():
    # print(get_weather.schema)
    # print(empty_parameter_tool.schema)
    # print(await get_weather(WeatherModel(city="Paris")))
    # print(await empty_parameter_tool())
    ctx = TestContext(id="123123123123", yesterday_weather="rain")
    # print(
    # await get_location_by_ip(IPModel(ip="8.8.8.8"))
    # .bind(lambda v: get_weather(WeatherModel(city=cast(WeatherModel, v).city)))
    # .bind(lambda v: get_alert(WeatherAlertModel(weather=cast(str, v))))
    # .map(lambda v: cast(BaseModel, v).model_dump())
    # .awaitable()
    # )

    result = (
        await get_location_by_ip.run_with_context("23.8.1.32")
        .map(lambda v: v.map(lambda x: get_weather.run_with_context(x)))
        .map(lambda v: v.map(lambda x: get_alert.run_with_context(x)))(ctx)
        .awaitable()
    )

    print(result)

    print(
        await get_location_by_ip(IPModel(ip="8.8.8.8"))
        .bind(
            lambda v: get_weather(
                WeatherModel(city=cast(WeatherModel, v).city), ctx=ctx
            )
        )
        .bind(lambda v: get_alert(WeatherAlertModel(weather=cast(str, v)), ctx=ctx))
        .map(lambda v: cast(BaseModel, v).model_dump())
        .awaitable()
    )


if __name__ == "__main__":
    asyncio.run(main())
