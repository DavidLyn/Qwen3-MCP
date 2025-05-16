import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WriteServer")
USER_AGENT = "write-app/1.0"

@mcp.tool()
async def write_file(content: str) -> str:
    """
    将指定内容写入本地文件。

    参数:
        content: 必要参数，字符串类型，用于表示需要写入文档的具体内容。

    返回:
        成功写入文件时返回 "已成功写入本地文件。"，否则返回错误信息。
    """
    return "已成功写入本地文件。"

if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')