import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI Python SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置日志输出到文件，而不是控制台
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "mcp_client.log")),
        # 不再使用 StreamHandler，避免日志输出到控制台
    ]
)


class Configuration:
    """
    配置加载类

    管理 MCP 客户端的环境变量和配置文件
    """

    def __init__(self) -> None:
        load_dotenv()
        # 从环境变量中加载 API key, base_url 和 model
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 文件中配置")

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """
        从 JSON 文件加载服务器配置
        
        Args:
            file_path: JSON 配置文件路径
        
        Returns:
            包含服务器配置的字典
        """
        with open(file_path, "r") as f:
            return json.load(f)


class Server:
    """
    MCP 服务器封装类

    管理单个 MCP 服务器连接和工具调用
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """初始化与 MCP 服务器的连接"""
        # command 字段直接从配置获取
        command = self.config["command"]
        if command is None:
            raise ValueError("command 不能为空")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[Any]:
        """获取服务器可用的工具列表

        Returns:
            工具列表
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        tools_response = await self.session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))
        return tools

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], retries: int = 2, delay: float = 1.0
    ) -> Any:
        """执行指定工具，并支持重试机制

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            retries: 重试次数
            delay: 重试间隔秒数

        Returns:
            工具调用结果
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name} on server {self.name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")

                    # 注意：这里返回的是错误信息，而不是接着抛异常
                    return f"Error executing tool: {e}"
                    # raise

    async def cleanup(self) -> None:
        """清理服务器资源"""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """
    工具封装类

    封装 MCP 返回的工具信息
    """

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """生成用于 LLM 提示的工具描述"""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)
        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """
    LLM 客户端封装类（使用 OpenAI SDK)

    使用 OpenAI SDK 与大模型交互
    """

    def __init__(self, api_key: str, base_url: Optional[str], model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_response(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        发送消息给大模型 API,支持传入工具参数(function calling 格式）
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": True,  # 启用流式模式
        }
        try:
            logging.debug(f"Sending payload to LLM: {payload}")

            # 处理流式响应
            stream_resp = self.client.chat.completions.create(**payload)
            
            # 收集完整响应
            collected_chunks = []
            collected_messages = []
            collected_reasoning = []  # 新增：收集reasoning_content
            has_printed_content = False  # 跟踪是否已经打印过内容
            has_printed_reasoning = False  # 跟踪是否已经打印过reasoning_content
            
            # 收集工具调用信息
            tool_calls_info = {}  # 用于存储合并的工具调用信息
            
            # 处理流式响应
            for chunk in stream_resp:
                collected_chunks.append(chunk)  # 保存所有块
                # 打印每个块的内容，便于调试
                logging.debug(f"Stream chunk: {chunk}")
                
                if hasattr(chunk.choices[0], 'delta'):
                    # 处理常规内容
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        collected_messages.append(chunk.choices[0].delta.content)
                        # 实时打印流式内容片段
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        has_printed_content = True
                    
                    # 处理reasoning_content（如果有）
                    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        content = chunk.choices[0].delta.reasoning_content
                        collected_reasoning.append(content)
                        
                        # 只有当没有常规content时才打印reasoning_content
                        if not has_printed_content:
                            # 避免重复打印相同的内容（如"好的"）
                            if not has_printed_reasoning or (has_printed_reasoning and content != "好的"):
                                print(content, end="", flush=True)
                                has_printed_reasoning = True
                    
                    # 收集工具调用信息
                    if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            if tool_call.index is not None:
                                idx = tool_call.index
                                if idx not in tool_calls_info:
                                    tool_calls_info[idx] = {
                                        "id": tool_call.id if tool_call.id else "",
                                        "type": tool_call.type if tool_call.type else "function",
                                        "function": {
                                            "name": tool_call.function.name if hasattr(tool_call.function, 'name') and tool_call.function.name else "",
                                            "arguments": tool_call.function.arguments if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments else ""
                                        }
                                    }
                                else:
                                    # 合并ID
                                    if tool_call.id:
                                        tool_calls_info[idx]["id"] += tool_call.id
                                    
                                    # 合并类型
                                    if tool_call.type:
                                        tool_calls_info[idx]["type"] = tool_call.type
                                    
                                    # 合并函数名称和参数
                                    if hasattr(tool_call, 'function'):
                                        if hasattr(tool_call.function, 'name') and tool_call.function.name:
                                            tool_calls_info[idx]["function"]["name"] = tool_call.function.name
                                        
                                        if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments:
                                            tool_calls_info[idx]["function"]["arguments"] += tool_call.function.arguments
            
            # 这里修正了缩进问题，将下面的代码移出循环
            print()  # 换行
            
            # 使用最后一个块作为完整响应
            if collected_chunks:
                full_response = collected_chunks[-1]
                
                # 确保响应对象有正确的结构
                if hasattr(full_response.choices[0], 'delta'):
                    # 如果delta中没有content属性或content为None，设置为空字符串
                    if not hasattr(full_response.choices[0].delta, 'content') or full_response.choices[0].delta.content is None:
                        # 如果有收集到的消息，使用它们
                        if collected_messages:
                            # 为delta添加content属性
                            full_response.choices[0].delta.content = "".join(collected_messages)
                        elif collected_reasoning:  # 如果有reasoning_content但没有常规content
                            # 使用reasoning_content作为content
                            full_response.choices[0].delta.content = "".join(collected_reasoning)
                        else:
                            # 否则设置为空字符串
                            full_response.choices[0].delta.content = ""
                    
                    # 如果是工具调用，将合并后的工具调用信息添加到响应中
                    if full_response.choices[0].finish_reason == "tool_calls" and tool_calls_info:
                        # 创建合并后的工具调用列表
                        merged_tool_calls = []
                        for idx in sorted(tool_calls_info.keys()):
                            merged_tool_calls.append(tool_calls_info[idx])
                        
                        # 将合并后的工具调用信息添加到响应中
                        full_response.choices[0].delta.tool_calls = merged_tool_calls
                    
                    return full_response
                else:
                    raise Exception("响应格式不正确,缺少delta属性")
            else:
                raise Exception("没有收到任何响应块")
        except (Exception, httpx.HTTPError) as e:
            # 捕获 LLM 调用过程中的异常
            logging.error(f"Error during LLM call: {e}")
            raise


class MultiServerMCPClient:
    """
    多服务器 MCP 客户端类
    
    集成配置文件、工具格式转换与 OpenAI SDK 调用
    """
    def __init__(self) -> None:
        """
        管理多个 MCP 服务器，并使用 OpenAI Function Calling 风格的接口调用大模型
        """
        self.exit_stack = AsyncExitStack()
        config = Configuration()
        self.openai_api_key = config.api_key
        self.base_url = config.base_url
        self.model = config.model
        self.client = LLMClient(self.openai_api_key, self.base_url, self.model)
        # (server_name -> Server 对象)
        self.servers: Dict[str, Server] = {}
        # 各个 server 的工具列表
        self.tools_by_server: Dict[str, List[Any]] = {}
        self.all_tools: List[Dict[str, Any]] = []

    async def connect_to_servers(self, servers_config: Dict[str, Any]) -> None:
        """
        根据配置文件同时启动多个服务器并获取工具
        servers_config 的格式为：
        {
          "mcpServers": {
              "sqlite": { "command": "uvx", "args": [ ... ] },
              "puppeteer": { "command": "npx", "args": [ ... ] },
              ...
          }
        }
        """
        mcp_servers = servers_config.get("mcpServers", {})
        for server_name, srv_config in mcp_servers.items():
            server = Server(server_name, srv_config)
            await server.initialize()
            self.servers[server_name] = server
            tools = await server.list_tools()
            self.tools_by_server[server_name] = tools

            for tool in tools:
                # 统一重命名：serverName_toolName
                function_name = f"{server_name}_{tool.name}"
                self.all_tools.append({
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    }
                })

        # 转换为 OpenAI Function Calling 所需格式
        self.all_tools = await self.transform_json(self.all_tools)

        logging.info("\n✅ 已连接到下列服务器:")
        for name in self.servers:
            srv_cfg = mcp_servers[name]
            logging.info(f"  - {name}: command={srv_cfg['command']}, args={srv_cfg['args']}")
        logging.info("\n汇总的工具:")
        for t in self.all_tools:
            logging.info(f"  - {t['function']['name']}")

    async def transform_json(self, json_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将工具的 input_schema 转换为 OpenAI 所需的 parameters 格式，并删除多余字段
        """
        result = []
        for item in json_data:
            if not isinstance(item, dict) or "type" not in item or "function" not in item:
                continue
            old_func = item["function"]
            if not isinstance(old_func, dict) or "name" not in old_func or "description" not in old_func:
                continue
            new_func = {
                "name": old_func["name"],
                "description": old_func["description"],
                "parameters": {}
            }
            if "input_schema" in old_func and isinstance(old_func["input_schema"], dict):
                old_schema = old_func["input_schema"]
                new_func["parameters"]["type"] = old_schema.get("type", "object")
                new_func["parameters"]["properties"] = old_schema.get("properties", {})
                new_func["parameters"]["required"] = old_schema.get("required", [])
            new_item = {
                "type": item["type"],
                "function": new_func
            }
            result.append(new_item)
        return result

    async def chat_base(self, messages: List[Dict[str, Any]], max_tool_calls: int = 5) -> Any:
        """
        使用 OpenAI 接口进行对话,并支持多次工具调用(Function Calling).
        如果返回 finish_reason 为 "tool_calls",则进行工具调用后再发起请求.
        
        Args:
            messages: 对话历史消息
            max_tool_calls: 最大工具调用次数，防止无限循环
        """
        response = self.client.get_response(messages, tools=self.all_tools)
        # 如果模型返回工具调用
        tool_call_count = 0
        if response.choices[0].finish_reason == "tool_calls":
            while tool_call_count < max_tool_calls:
                tool_call_count += 1
                logging.info(f"执行第 {tool_call_count} 次工具调用")
                
                # 创建包含工具调用结果的新消息
                messages = await self.create_function_response_messages(messages, response)
                
                # 发送新消息给模型
                response = self.client.get_response(messages, tools=self.all_tools)
                
                # 如果模型不再返回工具调用，则退出循环
                if response.choices[0].finish_reason != "tool_calls":
                    break
            
            if tool_call_count >= max_tool_calls:
                logging.warning(f"达到最大工具调用次数 {max_tool_calls}，强制退出工具调用循环")
        
        return response

    async def create_function_response_messages(self, messages: List[Dict[str, Any]], response: Any) -> List[Dict[str, Any]]:
        """
        将模型返回的工具调用解析执行，并将结果追加到消息队列中
        """
        # 初始化 function_call_messages 为空列表
        function_call_messages = []
        
        # 检查是否为流式响应，并获取工具调用信息
        if hasattr(response.choices[0], 'delta') and hasattr(response.choices[0].delta, 'tool_calls'):
            # 流式响应处理
            if response.choices[0].delta.tool_calls is not None:  # 确保 tool_calls 不为 None
                function_call_messages = response.choices[0].delta.tool_calls
                logging.debug(f"从流式响应中获取到的工具调用: {function_call_messages}")
                
                # 添加模型消息到历史
                model_message = {"role": "assistant", "tool_calls": []}
                for tool_call in function_call_messages:
                    try:
                        model_message["tool_calls"].append({
                            "id": tool_call["id"] if isinstance(tool_call, dict) else tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"] if isinstance(tool_call, dict) else tool_call.function.name,
                                "arguments": tool_call["function"]["arguments"] if isinstance(tool_call, dict) else tool_call.function.arguments
                            }
                        })
                    except (AttributeError, KeyError) as e:
                        logging.warning(f"处理工具调用时出错: {e}")
                        continue
                messages.append(model_message)
        elif hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'tool_calls'):
            # 非流式响应处理
            if response.choices[0].message.tool_calls is not None:  # 确保 tool_calls 不为 None
                function_call_messages = response.choices[0].message.tool_calls
                logging.debug(f"从非流式响应中获取到的工具调用: {function_call_messages}")
                
                # 添加模型消息到历史
                try:
                    model_message = response.choices[0].message.model_dump()
                    messages.append(model_message)
                except AttributeError as e:
                    logging.warning(f"处理消息时出错: {e}")
                    # 尝试手动构建消息
                    model_message = {"role": "assistant", "tool_calls": []}
                    for tool_call in function_call_messages:
                        try:
                            model_message["tool_calls"].append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            })
                        except AttributeError:
                            continue
                messages.append(model_message)
        
        # 处理每个工具调用
        for function_call_message in function_call_messages:
            try:
                # 支持字典和对象两种形式
                if isinstance(function_call_message, dict):
                    tool_name = function_call_message["function"]["name"]
                    tool_args_str = function_call_message["function"]["arguments"]
                    tool_call_id = function_call_message["id"]
                else:
                    tool_name = function_call_message.function.name
                    tool_args_str = function_call_message.function.arguments
                    tool_call_id = function_call_message.id
                
                logging.info(f"准备调用工具: {tool_name}, 参数: {tool_args_str}")
                
                # 解析参数
                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError as e:
                    logging.error(f"解析工具参数时出错: {e}, 原始参数: {tool_args_str}")
                    # 尝试修复常见的JSON格式问题
                    fixed_args_str = tool_args_str.replace("'", "\"")
                    try:
                        tool_args = json.loads(fixed_args_str)
                        logging.info(f"成功修复并解析参数: {tool_args}")
                    except:
                        # 如果仍然失败，使用空字典
                        tool_args = {}
                    
                # 调用 MCP 工具
                function_response = await self._call_mcp_tool(tool_name, tool_args)
                logging.info(f"工具调用结果: {function_response}")
                
                messages.append({
                    "role": "tool",
                    "content": function_response,
                    "tool_call_id": tool_call_id,
                })
            except Exception as e:
                logging.error(f"处理工具调用时出错: {e}")
                continue
        
        return messages

    async def process_query(self, user_query: str) -> str:
        """
        OpenAI Function Calling 流程：
         1. 发送用户消息 + 工具信息
         2. 若模型返回 finish_reason 为 "tool_calls"，则解析并调用 MCP 工具
         3. 将工具调用结果返回给模型，获得最终回答
        """
        messages = [{"role": "user", "content": user_query}]
        response = self.client.get_response(messages, tools=self.all_tools)
        content = response.choices[0]
        logging.info(content)
        if content.finish_reason == "tool_calls":
            # 处理工具调用，注意区分流式和非流式响应
            tool_call = content.delta.tool_calls[0] if hasattr(content, 'delta') else content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            logging.info(f"\n[ 调用工具: {tool_name}, 参数: {tool_args} ]\n")
            result = await self._call_mcp_tool(tool_name, tool_args)
            
            # 添加模型消息到历史
            if hasattr(content, 'delta'):
                messages.append(content.delta.model_dump())
            else:
                messages.append(content.message.model_dump())
                
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
            })
            response = self.client.get_response(messages, tools=self.all_tools)
            return response.choices[0].delta.content if hasattr(response.choices[0], 'delta') else response.choices[0].message.content
        return content.delta.content if hasattr(content, 'delta') else content.message.content

    async def _call_mcp_tool(self, tool_full_name: str, tool_args: Dict[str, Any]) -> str:
        """
        根据 "serverName_toolName" 格式调用相应 MCP 工具
        """
        parts = tool_full_name.split("_", 1)
        if len(parts) != 2:
            return f"无效的工具名称: {tool_full_name}"
        server_name, tool_name = parts
        server = self.servers.get(server_name)
        if not server:
            return f"找不到服务器: {server_name}"
        resp = await server.execute_tool(tool_name, tool_args)

        # 如果返回的是字符串,即execute_tool调用失败时，则直接返回
        if isinstance(resp, str):
            return resp
        else:
            return resp.content if resp.content else "工具执行无输出"

    async def chat_loop(self) -> None:
        """多服务器 MCP + OpenAI Function Calling 客户端主循环"""
        logging.info("\n🤖 多服务器 MCP + Function Calling 客户端已启动！输入 'quit' 退出。")
        print("\n🤖 多服务器 MCP + Function Calling 客户端已启动！输入 'quit' 退出。")
        messages: List[Dict[str, Any]] = []
        while True:
            query = input("\n你: ").strip()
            if query.lower() == "quit":
                break
            # 检查用户输入是否为空
            if not query:
                continue  # 如果输入为空，跳过本次循环
            try:
                messages.append({"role": "user", "content": query})
                messages = messages[-20:]  # 保持最新 20 条上下文
                print("\nAI: ", end="", flush=True)  # 提前打印AI前缀，后续内容会在get_response中流式输出
                response = await self.chat_base(messages)
                
                # 处理流式响应的结果
                content = ""
                if hasattr(response.choices[0], 'delta'):
                    # 流式响应
                    if hasattr(response.choices[0].delta, 'content') and response.choices[0].delta.content:
                        content = response.choices[0].delta.content
                    # 添加助手消息到历史
                    messages.append({"role": "assistant", "content": content})
                else:
                    # 非流式响应
                    content = response.choices[0].message.content
                    messages.append(response.choices[0].message.model_dump())
                
            except Exception as e:
                print(f"\n⚠️  调用过程出错: {e}")
                logging.exception("详细错误信息")

    async def cleanup(self) -> None:
        """关闭所有资源"""
        await self.exit_stack.aclose()


async def main() -> None:
    """主函数"""

    logging.basicConfig(level=logging.INFO)
    logging.info("初始化...")
    # 从配置文件加载服务器配置
    config = Configuration()
    servers_config = config.load_config("servers_config.json")
    client = MultiServerMCPClient()
    try:
        await client.connect_to_servers(servers_config)
        await client.chat_loop()
    finally:
        try:
            await asyncio.sleep(0.1)
            await client.cleanup()
        except RuntimeError as e:
            # 如果是因为退出 cancel scope 导致的异常，可以选择忽略
            if "Attempted to exit cancel scope" in str(e):
                logging.info("退出时检测到 cancel scope 异常，已忽略。")
            else:
                raise

if __name__ == "__main__":
    asyncio.run(main())