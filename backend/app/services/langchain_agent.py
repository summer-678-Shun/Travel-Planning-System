"""LangChain Agent 与 MCP 工具适配层。

这个模块提供与原 hello_agents.SimpleAgent / MCPTool 近似的同步接口，
让上层业务代码基本不用改动，同时底层切换为 LangChain。
"""

import asyncio
import json
import re
import threading
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


_TOOL_CALL_RE = re.compile(r"\[TOOL_CALL:([^:\]]+):([^\]]*)\]")


def _run_async(coro):
    """在同步代码中安全运行 async 协程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder: Dict[str, Any] = {}
    error_holder: Dict[str, BaseException] = {}

    def runner():
        try:
            result_holder["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - 需要跨线程传递异常
            error_holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("result")


def _stringify_result(result: Any) -> str:
    """把 LangChain/MCP 工具或模型结果统一转成字符串。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return _stringify_result(result.content)
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


def _parse_tool_arguments(raw_args: str) -> Dict[str, Any]:
    """解析旧代码中的 [TOOL_CALL:tool:key=value,key=value] 参数格式。"""
    arguments: Dict[str, Any] = {}
    if not raw_args.strip():
        return arguments

    for part in raw_args.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        arguments[key.strip()] = value.strip()
    return arguments


class LangChainMCPTool:
    """MCP 工具的 LangChain 适配器，保留原 MCPTool 的常用同步接口。"""

    def __init__(
        self,
        name: str,
        description: str,
        server_command: List[str],
        env: Optional[Dict[str, str]] = None,
        auto_expand: bool = True,
    ):
        if not server_command:
            raise ValueError("server_command不能为空")

        self.name = name
        self.description = description
        self.server_command = server_command
        self.env = env or {}
        self.auto_expand = auto_expand
        self.expandable = auto_expand

        self._client = MultiServerMCPClient(
            {
                self.name: {
                    "transport": "stdio",
                    "command": self.server_command[0],
                    "args": self.server_command[1:],
                    "env": self.env,
                }
            }
        )
        self._tools = None
        self._tool_map: Dict[str, Any] = {}
        self._available_tools: List[Dict[str, str]] = []

    async def _ensure_tools_async(self) -> List[Any]:
        if self._tools is None:
            self._tools = await self._client.get_tools()
            self._tool_map = {tool.name: tool for tool in self._tools}

            # 兼容旧代码中 amap_maps_xxx 这种“服务名前缀+工具名”的写法
            for tool in self._tools:
                prefixed_name = f"{self.name}_{tool.name}"
                self._tool_map[prefixed_name] = tool

            self._available_tools = [
                {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                }
                for tool in self._tools
            ]
        return self._tools

    def _ensure_tools(self) -> List[Any]:
        return _run_async(self._ensure_tools_async())

    def list_tools(self) -> List[Any]:
        return self._ensure_tools()

    def _normalize_tool_name(self, tool_name: str) -> str:
        if tool_name in self._tool_map:
            return tool_name
        prefix = f"{self.name}_"
        if tool_name.startswith(prefix):
            without_prefix = tool_name[len(prefix):]
            if without_prefix in self._tool_map:
                return without_prefix
        return tool_name

    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        await self._ensure_tools_async()
        normalized_name = self._normalize_tool_name(tool_name)
        tool = self._tool_map.get(normalized_name)
        if tool is None:
            available = ", ".join(sorted(self._tool_map.keys()))
            raise ValueError(f"未找到MCP工具: {tool_name}; 可用工具: {available}")
        result = await tool.ainvoke(arguments)
        return _stringify_result(result)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        return _run_async(self.call_tool_async(tool_name, arguments))

    def run(self, payload: Dict[str, Any]) -> str:
        """兼容旧 MCPTool.run({...}) 调用。"""
        if not isinstance(payload, dict):
            raise TypeError("MCP工具调用参数必须是dict")
        action = payload.get("action", "call_tool")
        if action != "call_tool":
            raise ValueError(f"暂不支持的MCP action: {action}")
        return self.call_tool(payload["tool_name"], payload.get("arguments", {}))


class LangChainSimpleAgent:
    """轻量级 LangChain Agent，保留 SimpleAgent 的 run/add_tool/list_tools 接口。"""

    def __init__(self, name: str, llm: BaseChatModel, system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self._tools: List[LangChainMCPTool] = []

    def add_tool(self, tool: LangChainMCPTool) -> None:
        if tool.auto_expand:
            for tool in tool.list_tools():
                self._tools.append(tool)
        else:
            self._tools.append(tool)

    def list_tools(self) -> List[LangChainMCPTool]:
        return self._tools

    async def _execute_tool_call_if_present_async(self, text: str) -> Optional[str]:
        match = _TOOL_CALL_RE.search(text)
        if not match:
            return None
        if not self._tools:
            raise ValueError(f"Agent {self.name} 没有可用工具，无法执行: {match.group(0)}")

        tool_name = match.group(1).strip()
        arguments = _parse_tool_arguments(match.group(2))

        # 当前项目只有一个高德 MCP 工具(内部有很多工具)；若后续有多个工具，可以按 tool_name 前缀选择。
        return await self._tools[0].call_tool_async(tool_name, arguments)

    async def arun(self, query: str) -> str:
        """异步执行 Agent。

        这个方法用于 FastAPI 异步路由和 asyncio.gather 并行调用。
        它保留原来的 [TOOL_CALL:...] 文本工具调用协议，但内部改用
        llm.ainvoke() 和 MCP tool.ainvoke()，避免再绕回同步调用。
        """
        # 旧代码会直接把 [TOOL_CALL:...] 放进 query，这里优先执行，避免模型改写导致工具名/参数漂移。
        direct_tool_result = await self._execute_tool_call_if_present_async(query)
        if direct_tool_result is not None:
            return direct_tool_result

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query),
            ]
        )
        response_text = _stringify_result(response)

        # 如果模型根据系统提示返回了旧格式工具调用，也执行对应工具。
        tool_result = await self._execute_tool_call_if_present_async(response_text)
        return tool_result if tool_result is not None else response_text

    def run(self, query: str) -> str:
        """同步兼容接口，保留旧代码的 agent.run(...) 调用方式。"""
        return _run_async(self.arun(query))
