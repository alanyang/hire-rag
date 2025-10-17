"""
完整的工作流框架 - Dataclass版本
修复了所有类型错误和警告
"""

import asyncio
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    Set,
    Callable,
    runtime_checkable,
    cast,
)
from dataclasses import dataclass, field
import logging

# === 类型定义 ===
Action = str
Context = Dict[str, Any]
Params = Dict[str, Any]

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === 协议定义 ===
@runtime_checkable
class Node(Protocol):
    """节点协议定义"""

    name: str
    params: Params
    successors: Dict[Action, List["Node"]]
    id: str = field(default_factory=str)

    async def prep(self, ctx: Context) -> Any:
        """准备阶段 - 数据预处理"""
        ...

    async def exec(self, prep_res: Any) -> Any:
        """执行阶段 - 核心逻辑"""
        ...

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List["Node"]]:
        """后处理阶段 - 返回新上下文和后续节点"""
        ...

    def get_next_nodes(self, action: Action) -> List["Node"]:
        """获取指定动作的后续节点"""
        ...

    def next(self, *nodes: "Node") -> "Node":
        """设置默认后续节点，返回第一个节点或self"""
        ...

    def on(self, action: Action, *nodes: "Node") -> "Node":
        """为指定动作设置后续节点"""
        ...

    def set_params(self, params: Params) -> "Node":
        """设置节点参数"""
        ...


# === 基础节点数据类 ===
@dataclass
class BaseNode:
    """基础节点数据类"""

    name: str = ""
    params: Params = field(default_factory=dict)
    successors: Dict[Action, List["Node"]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    def get_next_nodes(self, action: Action) -> List["Node"]:
        """获取指定动作的后续节点"""
        return self.successors.get(action, [])

    def next(self, *nodes: "Node") -> "Node":
        """设置默认后续节点"""
        self.successors["default"] = list(nodes)
        return self  # type: ignore

    def on(self, action: Action, *nodes: "Node") -> "Node":
        """为指定动作设置后续节点"""
        self.successors[action] = list(nodes)
        return self  # type: ignore

    def set_params(self, params: Params) -> "Node":
        """设置节点参数"""
        self.params = params
        return self  # type: ignore

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# === 标准节点实现 ===
@dataclass
class StandardNode(BaseNode):
    """标准节点实现 - 支持重试和回退"""

    retries: int = 1
    wait_sec: float = 0.0

    async def prep(self, ctx: Context) -> Any:
        """默认准备 - 可以被子类重写"""
        return ctx.get("data")

    async def exec(self, prep_res: Any) -> Any:
        """默认执行 - 应该被子类重写"""
        logger.info(f"执行节点: {self.name}")
        return prep_res

    async def exec_with_retry(self, prep_res: Any) -> Any:
        """带重试的执行"""
        last_exception = None
        for i in range(self.retries):
            try:
                return await self.exec(prep_res)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"节点 {self.name} 执行失败 (尝试 {i + 1}/{self.retries}): {e}"
                )
                if i < self.retries - 1 and self.wait_sec > 0:
                    await asyncio.sleep(self.wait_sec)

        if last_exception is not None:
            return await self.fallback(prep_res, last_exception)
        return prep_res

    async def fallback(self, prep_res: Any, error: Exception) -> Any:
        """回退处理 - 可以被子类重写"""
        logger.error(f"节点 {self.name} 所有重试都失败，执行回退")
        raise error

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List[Node]]:
        """默认后处理"""
        new_ctx = ctx.copy()
        new_ctx["data"] = exec_res
        next_nodes = self.get_next_nodes("default")
        return new_ctx, next_nodes


# === 条件分支节点 ===
@dataclass
class ConditionalNode(StandardNode):
    """条件分支节点"""

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List[Node]]:
        if await self.should_branch_true(ctx, prep_res, exec_res):
            next_nodes = self.get_next_nodes("true")
            logger.info(f"条件节点 {self.name} 走 true 分支")
        else:
            next_nodes = self.get_next_nodes("false")
            logger.info(f"条件节点 {self.name} 走 false 分支")

        new_ctx = ctx.copy()
        new_ctx["data"] = exec_res
        return new_ctx, next_nodes

    async def should_branch_true(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> bool:
        """判断分支条件 - 子类需要重写"""
        return True

    def if_true(self, *nodes: "Node") -> "Node":
        """设置true分支"""
        return self.on("true", *nodes)

    def if_false(self, *nodes: "Node") -> "Node":
        """设置false分支"""
        return self.on("false", *nodes)


# === 并行分割节点 ===
@dataclass
class ParallelSplitNode(StandardNode):
    """并行分割节点"""

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List[Node]]:
        parallel_branches = self.get_next_nodes("parallel")
        if not parallel_branches:
            parallel_branches = self.get_next_nodes("default")

        logger.info(f"并行节点 {self.name} 启动 {len(parallel_branches)} 个分支")
        new_ctx = ctx.copy()
        new_ctx["data"] = exec_res
        return new_ctx, parallel_branches

    def parallel(self, *nodes: "Node") -> "Node":
        """设置并行分支"""
        return self.on("parallel", *nodes)


# === 合并节点 ===
@dataclass
class MergeNode(StandardNode):
    """合并节点 - 等待多个分支完成"""

    expected_inputs: int = 1
    received_contexts: List[Context] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def prep(self, ctx: Context) -> Any:
        async with self._lock:
            self.received_contexts.append(ctx)
            logger.info(
                f"合并节点 {self.name} 收到输入 {len(self.received_contexts)}/{self.expected_inputs}"
            )
            return self.received_contexts.copy()

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List[Node]]:
        async with self._lock:
            if len(self.received_contexts) < self.expected_inputs:
                logger.info(f"合并节点 {self.name} 等待更多输入")
                return ctx, []  # 暂停执行

            # 合并所有上下文
            merged_context = await self.merge_contexts(self.received_contexts)
            next_nodes = self.get_next_nodes("default")

            # 重置状态
            self.received_contexts = []
            logger.info(f"合并节点 {self.name} 完成合并，继续执行")

            return merged_context, next_nodes

    async def merge_contexts(self, contexts: List[Context]) -> Context:
        """合并上下文 - 可以被子类重写"""
        merged = {}
        for ctx in contexts:
            merged.update(ctx)
        return merged


# === 循环节点 ===
@dataclass
class LoopNode(StandardNode):
    """循环节点"""

    max_iterations: int = 10
    current_iteration: int = field(default=0, init=False)

    async def post(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> tuple[Context, List[Node]]:
        self.current_iteration += 1

        if (
            self.current_iteration < self.max_iterations
            and await self.should_continue_loop(ctx, prep_res, exec_res)
        ):
            loop_nodes = self.get_next_nodes("loop")
            logger.info(
                f"循环节点 {self.name} 继续循环 (第 {self.current_iteration} 次)"
            )
            new_ctx = ctx.copy()
            new_ctx["data"] = exec_res
            return new_ctx, loop_nodes
        else:
            self.current_iteration = 0
            exit_nodes = self.get_next_nodes("exit")
            logger.info(f"循环节点 {self.name} 退出循环")
            new_ctx = ctx.copy()
            new_ctx["data"] = exec_res
            return new_ctx, exit_nodes

    async def should_continue_loop(
        self, ctx: Context, prep_res: Any, exec_res: Any
    ) -> bool:
        """判断是否继续循环 - 子类需要重写"""
        return False

    def loop_to(self, *nodes: "Node") -> "Node":
        """设置循环目标"""
        return self.on("loop", *nodes)

    def exit_to(self, *nodes: "Node") -> "Node":
        """设置退出目标"""
        return self.on("exit", *nodes)


# === 并行批量处理节点 ===
@dataclass
class ParallelBatchNode(StandardNode):
    """并行批量处理节点"""

    async def exec(self, prep_res: Any) -> List[Any]:
        if not isinstance(prep_res, list):
            return []

        logger.info(f"批量节点 {self.name} 处理 {len(prep_res)} 个项目")
        tasks = [self.exec_item_with_retry(item) for item in prep_res]
        return await asyncio.gather(*tasks)

    async def exec_item(self, item: Any) -> Any:
        """处理单个项目 - 子类需要重写"""
        return item

    async def exec_item_with_retry(self, item: Any) -> Any:
        """带重试的单项处理"""
        last_exception = None
        for i in range(self.retries):
            try:
                return await self.exec_item(item)
            except Exception as e:
                last_exception = e
                if i < self.retries - 1 and self.wait_sec > 0:
                    await asyncio.sleep(self.wait_sec)
        if last_exception is not None:
            raise last_exception
        return item


# === 执行引擎 ===
@dataclass
class ExecutionEngine:
    """统一执行引擎"""

    executed_nodes: Dict[int, Any] = field(default_factory=dict, init=False)

    async def execute_nodes(
        self, start_nodes: List[Node], initial_context: Context
    ) -> Context:
        """执行节点图"""
        queue: List[tuple[Node, Context]] = [
            (node, initial_context.copy()) for node in start_nodes
        ]
        executed: Set[int] = set()
        final_context = initial_context.copy()

        iteration = 0
        while queue:
            iteration += 1
            logger.info(f"执行批次 {iteration}，待处理节点: {len(queue)}")

            # 当前批次
            current_batch: List[tuple[Node, Context]] = []
            remaining_queue: List[tuple[Node, Context]] = []

            for node, context in queue:
                node_id = id(node)
                if node_id not in executed:
                    current_batch.append((node, context))
                    executed.add(node_id)
                else:
                    remaining_queue.append((node, context))

            queue = remaining_queue

            if not current_batch:
                break

            # 并行执行当前批次
            tasks = [
                self.execute_single_node(node, context)
                for node, context in current_batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            for (node, _), result in zip(current_batch, results):
                if isinstance(result, Exception):
                    logger.error(f"节点 {node.name} 执行失败: {result}")
                    continue

                # 类型守卫：此时 result 一定是 tuple[Context, List[Node]]
                result_context, next_nodes = cast(tuple[Context, List[Node]], result)
                final_context.update(result_context)

                # 添加后续节点到队列
                for next_node in next_nodes:
                    queue.append((next_node, result_context.copy()))

        logger.info(f"执行完成，总共 {iteration} 个批次")
        return final_context

    async def execute_single_node(
        self, node: Node, context: Context
    ) -> tuple[Context, List[Node]]:
        """执行单个节点"""
        logger.info(f"开始执行节点: {node.name}")

        prep_res = await node.prep(context)

        if isinstance(node, StandardNode):
            exec_res = await node.exec_with_retry(prep_res)
        else:
            exec_res = await node.exec(prep_res)

        result = await node.post(context, prep_res, exec_res)
        logger.info(f"节点 {node.name} 执行完成")
        return result


# === Flow API ===
@dataclass
class FlowAPI:
    """简单的流式API"""

    _engine: ExecutionEngine = field(default_factory=ExecutionEngine, init=False)

    async def run(self, start_node: Node, initial_context: Context) -> Context:
        """运行简单流程"""
        logger.info("使用 Flow API 执行")
        return await self._engine.execute_nodes([start_node], initial_context)


# === Graph API ===
@dataclass
class GraphBuilder:
    """图构建器"""

    _nodes: List[Node] = field(default_factory=list, init=False)
    _start_nodes: List[Node] = field(default_factory=list, init=False)
    _engine: ExecutionEngine = field(default_factory=ExecutionEngine, init=False)

    def add_node(self, node: Node) -> "GraphBuilder":
        """添加节点"""
        if node not in self._nodes:
            self._nodes.append(node)
        return self

    def add_edge(
        self, from_node: Node, to_node: Node, action: str = "default"
    ) -> "GraphBuilder":
        """添加边"""
        self.add_node(from_node).add_node(to_node)
        from_node.on(action, to_node)
        return self

    def add_conditional_edge(
        self, from_node: Node, edge_map: Dict[str, Node]
    ) -> "GraphBuilder":
        """添加条件边"""
        self.add_node(from_node)
        for action, to_node in edge_map.items():
            self.add_node(to_node)
            from_node.on(action, to_node)
        return self

    def set_start_nodes(self, *nodes: Node) -> "GraphBuilder":
        """设置起始节点"""
        self._start_nodes = list(nodes)
        for node in nodes:
            self.add_node(node)
        return self

    async def run(
        self,
        start_nodes: Union[Node, List[Node], None] = None,
        initial_context: Optional[Context] = None,
    ) -> Context:
        """运行图"""
        if start_nodes is None:
            start_nodes = self._start_nodes
        elif isinstance(start_nodes, Node):
            start_nodes = [start_nodes]

        if not start_nodes:
            raise ValueError("必须指定起始节点")

        if initial_context is None:
            initial_context = {}

        logger.info("使用 Graph API 执行")
        return await self._engine.execute_nodes(start_nodes, initial_context)


# === 统一框架入口 ===
@dataclass
class WorkflowFramework:
    """统一工作流框架"""

    flow: FlowAPI = field(default_factory=FlowAPI, init=False)

    # === 简单用法 ===
    async def run_flow(self, start_node: Node, context: Context) -> Context:
        """运行简单流程"""
        return await self.flow.run(start_node, context)

    # === 高级用法 ===
    def create_graph(self) -> GraphBuilder:
        """创建图构建器"""
        return GraphBuilder()

    # === 智能用法 ===
    async def run_auto(self, start_node: Node, context: Context) -> Context:
        """智能执行 - 自动选择最适合的方式"""
        return await self.run_flow(start_node, context)


# === 实用节点实现 ===
@dataclass
class FunctionNode(StandardNode):
    """函数节点 - 包装普通函数"""

    func: Callable[[Any], Any] = field(default=lambda x: x)

    async def exec(self, prep_res: Any) -> Any:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(prep_res)
        else:
            return self.func(prep_res)


@dataclass
class DataTransformNode(StandardNode):
    """数据转换节点"""

    transform_func: Callable[[Any], Any] = field(default=lambda x: x)

    async def exec(self, prep_res: Any) -> Any:
        return self.transform_func(prep_res)


@dataclass
class PrintNode(StandardNode):
    """打印节点 - 用于调试"""

    prefix: str = ""

    async def exec(self, prep_res: Any) -> Any:
        print(f"{self.prefix}{prep_res}")
        return prep_res


# === 测试用例和示例 ===
async def test_simple_flow():
    """测试简单流程"""
    print("\n=== 测试简单流程 ===")

    framework = WorkflowFramework()

    # 创建节点
    node1 = FunctionNode(name="步骤1", func=lambda x: f"{x}_step1")
    node2 = FunctionNode(name="步骤2", func=lambda x: f"{x}_step2")
    node3 = PrintNode(name="打印", prefix="最终结果: ")

    # 构建链
    node1.next(node2)
    node2.next(node3)

    # 执行
    result = await framework.run_flow(node1, {"data": "开始"})
    print("Flow 执行结果:", result)


async def test_conditional_flow():
    """测试条件分支"""
    print("\n=== 测试条件分支 ===")

    @dataclass
    class NumberCheckNode(ConditionalNode):
        async def should_branch_true(
            self, ctx: Context, prep_res: Any, exec_res: Any
        ) -> bool:
            return isinstance(prep_res, int) and prep_res > 5

    framework = WorkflowFramework()

    checker = NumberCheckNode(name="数字检查")
    big_handler = FunctionNode(name="大数处理", func=lambda x: f"大数: {x}")
    small_handler = FunctionNode(name="小数处理", func=lambda x: f"小数: {x}")
    final = PrintNode(name="最终", prefix="条件分支结果: ")

    # 构建分支
    checker.if_true(big_handler)
    checker.if_false(small_handler)
    big_handler.next(final)
    small_handler.next(final)

    # 测试大数
    result1 = await framework.run_flow(checker, {"data": 10})
    print("大数结果:", result1)

    # 测试小数
    result2 = await framework.run_flow(checker, {"data": 3})
    print("小数结果:", result2)


async def test_parallel_flow():
    """测试并行流程"""
    print("\n=== 测试并行流程 ===")

    framework = WorkflowFramework()

    # 创建图
    graph = framework.create_graph()

    start = FunctionNode(name="开始", func=lambda x: x)
    branch1 = FunctionNode(name="分支1", func=lambda x: f"{x}_branch1")
    branch2 = FunctionNode(name="分支2", func=lambda x: f"{x}_branch2")
    merge = MergeNode(name="合并", expected_inputs=2)
    end = PrintNode(name="结束", prefix="并行结果: ")

    # 构建图结构
    (
        graph.add_edge(start, branch1)
        .add_edge(start, branch2)
        .add_edge(branch1, merge)
        .add_edge(branch2, merge)
        .add_edge(merge, end)
    )

    result = await graph.run(start, {"data": "并行开始"})
    print("并行执行结果:", result)


async def test_complex_graph():
    """测试复杂图结构"""
    print("\n=== 测试复杂图结构 ===")

    @dataclass
    class LoopCounterNode(LoopNode):
        max_count: int = 3
        counter: int = field(default=0, init=False)

        def __post_init__(self):
            super().__post_init__()
            self.max_iterations = self.max_count

        async def exec(self, prep_res: Any) -> Any:
            self.counter += 1
            return f"{prep_res}_loop{self.counter}"

        async def should_continue_loop(
            self, ctx: Context, prep_res: Any, exec_res: Any
        ) -> bool:
            return self.counter < 3

    framework = WorkflowFramework()
    graph = framework.create_graph()

    start = FunctionNode(name="开始", func=lambda x: "start")
    loop_node = LoopCounterNode(name="循环节点", max_count=3)
    process = FunctionNode(name="处理", func=lambda x: f"{x}_processed")
    end = PrintNode(name="结束", prefix="复杂图结果: ")

    # 构建循环结构
    loop_node.loop_to(process)
    loop_node.exit_to(end)
    process.next(loop_node)  # 形成循环

    (graph.add_edge(start, loop_node).set_start_nodes(start))

    result = await graph.run(initial_context={"data": "复杂开始"})
    print("复杂图执行结果:", result)


async def main():
    """主测试函数"""
    print("🚀 开始测试工作流框架")

    try:
        await test_simple_flow()
        await test_conditional_flow()
        await test_parallel_flow()
        await test_complex_graph()

        print("\n✅ 所有测试完成！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
