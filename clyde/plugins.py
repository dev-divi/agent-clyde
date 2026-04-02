"""MCP/Plugin interface — external tool integration.

Allows Clyde to load tools from:
  1. MCP servers (Model Context Protocol) via stdio or SSE transport
  2. Python plugin files (drop-in .py files with tool classes)
  3. Plugin registries (JSON manifests pointing to tool implementations)

This is the extensibility layer that prevents Clyde from being
locked into only its built-in tools.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ToolDefinition, ToolParameter
from .tools.base import BaseTool
from .tools.registry import ToolRegistry

logger = logging.getLogger("clyde.plugins")


# ---------------------------------------------------------------------------
# MCP Protocol types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    command: list[str]  # e.g. ["npx", "-y", "some-mcp-server"]
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" | "sse"
    url: str = ""  # For SSE transport


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    """Client for communicating with an MCP server over stdio.

    Implements the JSON-RPC based MCP protocol for tool discovery
    and execution.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._tools: list[MCPTool] = []

    def connect(self) -> bool:
        """Start the MCP server process and initialize."""
        try:
            cmd = self.config.command + self.config.args
            env = {**dict(__import__("os").environ), **self.config.env}
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            # Send initialize request
            response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "clyde", "version": "0.1.0"},
            })
            if response and "result" in response:
                # Send initialized notification
                self._send_notification("notifications/initialized", {})
                logger.info(f"MCP server '{self.config.name}' connected")
                return True
            return False
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Failed to connect MCP server '{self.config.name}': {e}")
            return False

    def discover_tools(self) -> list[MCPTool]:
        """Discover available tools from the MCP server."""
        response = self._send_request("tools/list", {})
        if not response or "result" not in response:
            return []

        tools = []
        for raw_tool in response["result"].get("tools", []):
            tools.append(MCPTool(
                server_name=self.config.name,
                name=raw_tool["name"],
                description=raw_tool.get("description", ""),
                input_schema=raw_tool.get("inputSchema", {}),
            ))
        self._tools = tools
        logger.info(f"MCP '{self.config.name}': discovered {len(tools)} tools")
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        response = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if not response:
            return "Error: No response from MCP server"

        if "error" in response:
            return f"MCP Error: {response['error'].get('message', 'Unknown error')}"

        result = response.get("result", {})
        content_parts = result.get("content", [])
        text_parts = []
        for part in content_parts:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts) if text_parts else json.dumps(result)

    def disconnect(self) -> None:
        """Shut down the MCP server."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self._process.kill()
            self._process = None

    def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and read the response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        try:
            line = json.dumps(request) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
            if response_line:
                return json.loads(response_line)
        except (json.JSONDecodeError, OSError, BrokenPipeError) as e:
            logger.error(f"MCP communication error: {e}")
        return None

    def _send_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            line = json.dumps(notification) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()
        except (OSError, BrokenPipeError):
            pass


# ---------------------------------------------------------------------------
# MCP Tool wrapper — adapts MCP tools to Clyde's BaseTool interface
# ---------------------------------------------------------------------------

class MCPToolAdapter(BaseTool):
    """Wraps an MCP server tool as a Clyde BaseTool."""

    def __init__(self, mcp_tool: MCPTool, client: MCPClient):
        self._mcp_tool = mcp_tool
        self._client = client
        self._params: list[ToolParameter] | None = None

    @property
    def name(self) -> str:
        return f"mcp_{self._mcp_tool.server_name}_{self._mcp_tool.name}"

    @property
    def description(self) -> str:
        return self._mcp_tool.description or f"MCP tool from {self._mcp_tool.server_name}"

    @property
    def category(self) -> str:
        return f"mcp:{self._mcp_tool.server_name}"

    @property
    def requires_permission(self) -> bool:
        return True  # MCP tools always require permission

    def get_parameters(self) -> list[ToolParameter]:
        if self._params is not None:
            return self._params

        schema = self._mcp_tool.input_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        params = []
        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
                enum=prop.get("enum"),
                default=prop.get("default"),
            ))
        self._params = params
        return params

    def execute(self, **kwargs: Any) -> str:
        return self._client.call_tool(self._mcp_tool.name, kwargs)


# ---------------------------------------------------------------------------
# Python plugin loader
# ---------------------------------------------------------------------------

class PluginLoader:
    """Loads tool plugins from Python files.

    Drop a .py file in the plugins directory that defines one or more
    classes inheriting from BaseTool, and they'll be auto-registered.
    """

    def __init__(self, plugin_dirs: list[Path] | None = None):
        self.plugin_dirs = plugin_dirs or [Path(".clyde_plugins")]

    def discover(self) -> list[BaseTool]:
        """Discover and instantiate tool plugins from plugin directories."""
        tools: list[BaseTool] = []
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            for py_file in sorted(plugin_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                found = self._load_file(py_file)
                tools.extend(found)
        if tools:
            logger.info(f"Loaded {len(tools)} plugin tool(s)")
        return tools

    def _load_file(self, path: Path) -> list[BaseTool]:
        """Load BaseTool subclasses from a Python file."""
        tools: list[BaseTool] = []
        module_name = f"clyde_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return tools
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    try:
                        instance = attr()
                        tools.append(instance)
                        logger.debug(f"Loaded plugin tool: {instance.name} from {path.name}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate {attr_name} from {path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load plugin {path}: {e}")
        finally:
            sys.modules.pop(module_name, None)
        return tools


# ---------------------------------------------------------------------------
# Plugin manager — orchestrates all plugin sources
# ---------------------------------------------------------------------------

class PluginManager:
    """Manages MCP servers, Python plugins, and external tool sources."""

    def __init__(self):
        self._mcp_clients: dict[str, MCPClient] = {}
        self._plugin_loader = PluginLoader()

    def load_mcp_servers(self, configs: list[MCPServerConfig]) -> list[BaseTool]:
        """Connect to MCP servers and return their tools as BaseTools."""
        tools: list[BaseTool] = []
        for config in configs:
            client = MCPClient(config)
            if client.connect():
                self._mcp_clients[config.name] = client
                mcp_tools = client.discover_tools()
                for mt in mcp_tools:
                    tools.append(MCPToolAdapter(mt, client))
        return tools

    def load_plugins(self, plugin_dirs: list[Path] | None = None) -> list[BaseTool]:
        """Load Python plugin files."""
        if plugin_dirs:
            self._plugin_loader.plugin_dirs = plugin_dirs
        return self._plugin_loader.discover()

    def load_all(
        self,
        registry: ToolRegistry,
        mcp_configs: list[MCPServerConfig] | None = None,
        plugin_dirs: list[Path] | None = None,
    ) -> int:
        """Load all external tools into the registry. Returns count loaded."""
        count = 0
        if mcp_configs:
            for tool in self.load_mcp_servers(mcp_configs):
                registry.register(tool)
                count += 1
        for tool in self.load_plugins(plugin_dirs):
            registry.register(tool)
            count += 1
        return count

    def shutdown(self) -> None:
        """Disconnect all MCP servers."""
        for name, client in self._mcp_clients.items():
            logger.info(f"Disconnecting MCP server: {name}")
            client.disconnect()
        self._mcp_clients.clear()

    @staticmethod
    def parse_mcp_config(config_data: list[dict]) -> list[MCPServerConfig]:
        """Parse MCP server configs from clyde.json format.

        Expected format in clyde.json:
        {
          "mcp_servers": [
            {
              "name": "filesystem",
              "command": ["npx", "-y", "@anthropic/mcp-filesystem"],
              "args": ["/path/to/allow"],
              "env": {}
            }
          ]
        }
        """
        configs = []
        for raw in config_data:
            configs.append(MCPServerConfig(
                name=raw["name"],
                command=raw.get("command", []),
                args=raw.get("args", []),
                env=raw.get("env", {}),
                transport=raw.get("transport", "stdio"),
                url=raw.get("url", ""),
            ))
        return configs
