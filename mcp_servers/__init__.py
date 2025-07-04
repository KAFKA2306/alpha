"""
MCP Servers Package for Alpha Architecture Agent
Provides financial data, macro economic data, and alternative data through MCP protocol
"""

from .finance_server import FinanceServer, FinanceConfig
from .macro_data_server import MacroDataServer, MacroDataConfig
from .alternative_data_server import AlternativeDataServer, AlternativeDataConfig
from .server_manager import MCPServerManager, MCPConfig

__version__ = "1.0.0"
__author__ = "Alpha Architecture Agent Team"

__all__ = [
    "FinanceServer",
    "FinanceConfig", 
    "MacroDataServer",
    "MacroDataConfig",
    "AlternativeDataServer", 
    "AlternativeDataConfig",
    "MCPServerManager",
    "MCPConfig"
]