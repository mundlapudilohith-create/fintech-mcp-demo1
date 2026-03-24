"""
MCP Client Wrapper — Bank AI Assistant
Single bank MCP server: data_server.py
"""
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Dict, List, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for a single MCP server."""

    def __init__(self, server_module: str, server_name: str):
        self.server_module  = server_module
        self.server_name    = server_name
        self.available_tools: List[Dict[str, Any]] = []

    async def connect(self):
        """Connect to discover available tools."""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=["-m", self.server_module],
                env=None
            )

            logger.info(f"Connecting to {self.server_name}...")

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools_list = await session.list_tools()
                    self.available_tools = [
                        {
                            "name":         tool.name,
                            "description":  tool.description,
                            "input_schema": tool.inputSchema
                        }
                        for tool in tools_list.tools
                    ]

            logger.info(f"✓ {self.server_name}: {len(self.available_tools)} tools")
            logger.info(f"  Tools: {[t['name'] for t in self.available_tools]}")
            return self.available_tools

        except Exception as e:
            logger.error(f"✗ {self.server_name} connection failed: {e}")
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool using a fresh connection."""
        logger.info(f"→ [{self.server_name}] {tool_name}")

        try:
            server_params = StdioServerParameters(
                command="python",
                args=["-m", self.server_module],
                env=None
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)

                    logger.info(f"✓ [{self.server_name}] {tool_name} executed")

                    if result.content:
                        result_text = result.content[0].text if result.content else None
                        return {
                            "success":  not result.isError,
                            "result":   result_text,
                            "is_error": result.isError
                        }

                    return {
                        "success":  not result.isError,
                        "result":   None,
                        "is_error": result.isError
                    }

        except Exception as e:
            logger.error(f"✗ [{self.server_name}] {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    def get_tools_for_schema(self) -> List[Dict[str, Any]]:
        """Get tools in a generic schema format."""
        schema_tools = []

        for tool in self.available_tools:
            parameter_definitions = {}

            if "properties" in tool["input_schema"]:
                for param_name, param_schema in tool["input_schema"]["properties"].items():
                    param_type  = param_schema.get("type", "string")
                    mapped_type = self._map_type(param_type)

                    parameter_definitions[param_name] = {
                        "description": param_schema.get("description", f"Parameter {param_name}"),
                        "type":        mapped_type,
                        "required":    param_name in tool["input_schema"].get("required", [])
                    }

            schema_tools.append({
                "name":                 tool["name"],
                "description":          tool["description"],
                "parameter_definitions": parameter_definitions
            })

        return schema_tools

    def _map_type(self, json_type: str) -> str:
        return {
            "string":  "str",
            "number":  "float",
            "integer": "int",
            "boolean": "bool",
            "array":   "list",
            "object":  "dict"
        }.get(json_type, "str")

    async def close(self):
        logger.info(f"{self.server_name} client closed")


class MCPClientManager:
    """Manages lifecycle of a single MCP client."""

    def __init__(self, server_module: str, server_name: str):
        self.server_module = server_module
        self.server_name   = server_name
        self._client: Optional[MCPClient] = None
        self._lock = asyncio.Lock()

    async def get_client(self) -> MCPClient:
        async with self._lock:
            if self._client is None:
                self._client = MCPClient(self.server_module, self.server_name)
                try:
                    await self._client.connect()
                except Exception as e:
                    logger.error(f"Failed to initialize {self.server_name}: {e}")
                    self._client = None
                    raise
            return self._client

    async def close(self):
        async with self._lock:
            if self._client:
                await self._client.close()
                self._client = None


# ═══════════════════════════════════════════════════════════════════════
# THREE MANAGERS
# ═══════════════════════════════════════════════════════════════════════

# Bank AI Assistant — all payment/compliance/account tools
bank_client_manager = MCPClientManager(
    server_module="mcp_server.data_server",
    server_name="Bank AI Assistant"
)

# GST Calculator — calculate_gst, reverse_calculate_gst, gst_breakdown,
#                  compare_gst_rates, validate_gstin
gst_client_manager = MCPClientManager(
    server_module="mcp_server.server",
    server_name="GST Calculator"
)

# Onboarding Info — company, bank, vendor onboarding guides & FAQs
info_client_manager = MCPClientManager(
    server_module="mcp_server.info_server",
    server_name="Onboarding Info"
)