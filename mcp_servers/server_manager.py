#!/usr/bin/env python3
"""
MCP Server Manager - Manages and coordinates multiple MCP servers
Provides unified access to finance, macro, and alternative data servers
"""
import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time

from mcp import Server
from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    JSONContent,
    LoggingLevel,
)

from .finance_server import FinanceServer, FinanceConfig
from .macro_data_server import MacroDataServer, MacroDataConfig
from .alternative_data_server import AlternativeDataServer, AlternativeDataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    """Configuration for MCP Server Manager"""
    # Finance server config
    finance_enabled: bool = True
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: Optional[str] = None
    j_quants_api_key: Optional[str] = None
    j_quants_refresh_token: Optional[str] = None
    
    # Macro data server config
    macro_enabled: bool = True
    fred_api_key: Optional[str] = None
    world_bank_enabled: bool = True
    oecd_enabled: bool = True
    
    # Alternative data server config
    alternative_enabled: bool = True
    news_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    twitter_bearer_token: Optional[str] = None
    
    # Server management
    server_ports: Dict[str, int] = field(default_factory=lambda: {
        "finance": 8001,
        "macro": 8002,
        "alternative": 8003,
        "manager": 8000
    })
    
    # Caching and performance
    cache_duration_minutes: int = 15
    rate_limit_requests_per_minute: int = 100
    max_concurrent_requests: int = 50
    
    # Monitoring
    health_check_interval_seconds: int = 30
    log_level: str = "INFO"
    metrics_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'MCPConfig':
        """Create configuration from environment variables"""
        return cls(
            # Finance config
            alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
            j_quants_api_key=os.getenv("JQUANTS_API_KEY"),
            j_quants_refresh_token=os.getenv("JQUANTS_REFRESH_TOKEN"),
            
            # Macro config
            fred_api_key=os.getenv("FRED_API_KEY"),
            
            # Alternative config
            news_api_key=os.getenv("NEWS_API_KEY"),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
            
            # Server management
            log_level=os.getenv("MCP_LOG_LEVEL", "INFO"),
            metrics_enabled=os.getenv("MCP_METRICS_ENABLED", "true").lower() == "true"
        )

class MCPServerManager:
    """MCP Server Manager - Coordinates multiple data servers"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.server = Server("mcp-server-manager")
        self.servers = {}
        self.server_processes = {}
        self.health_status = {}
        self.metrics = {
            "requests_total": 0,
            "requests_by_server": {},
            "errors_total": 0,
            "startup_time": datetime.now(),
            "last_health_check": None
        }
        
        # Initialize individual servers
        self._initialize_servers()
        self._setup_tools()
        self._setup_resources()
        
    def _initialize_servers(self):
        """Initialize individual MCP servers"""
        try:
            if self.config.finance_enabled:
                finance_config = FinanceConfig(
                    yahoo_finance_enabled=self.config.yahoo_finance_enabled,
                    alpha_vantage_api_key=self.config.alpha_vantage_api_key,
                    j_quants_api_key=self.config.j_quants_api_key,
                    j_quants_refresh_token=self.config.j_quants_refresh_token,
                    cache_duration_minutes=self.config.cache_duration_minutes
                )
                self.servers["finance"] = FinanceServer(finance_config)
                self.health_status["finance"] = "initializing"
                logger.info("Finance server initialized")
            
            if self.config.macro_enabled:
                macro_config = MacroDataConfig(
                    fred_api_key=self.config.fred_api_key,
                    world_bank_enabled=self.config.world_bank_enabled,
                    oecd_enabled=self.config.oecd_enabled,
                    cache_duration_hours=self.config.cache_duration_minutes // 60
                )
                self.servers["macro"] = MacroDataServer(macro_config)
                self.health_status["macro"] = "initializing"
                logger.info("Macro data server initialized")
            
            if self.config.alternative_enabled:
                alt_config = AlternativeDataConfig(
                    news_api_key=self.config.news_api_key,
                    reddit_client_id=self.config.reddit_client_id,
                    reddit_client_secret=self.config.reddit_client_secret,
                    twitter_bearer_token=self.config.twitter_bearer_token,
                    cache_duration_hours=self.config.cache_duration_minutes // 60,
                    rate_limit_requests_per_minute=self.config.rate_limit_requests_per_minute
                )
                self.servers["alternative"] = AlternativeDataServer(alt_config)
                self.health_status["alternative"] = "initializing"
                logger.info("Alternative data server initialized")
                
        except Exception as e:
            logger.error(f"Error initializing servers: {str(e)}")
            raise
    
    def _setup_tools(self):
        """Setup unified MCP tools across all servers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            tools = []
            
            # Add management tools
            tools.extend([
                Tool(
                    name="get_server_status",
                    description="Get status of all MCP servers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed metrics",
                                "default": False
                            }
                        }
                    }
                ),
                Tool(
                    name="restart_server",
                    description="Restart a specific MCP server",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "server_name": {
                                "type": "string",
                                "enum": ["finance", "macro", "alternative", "all"],
                                "description": "Server to restart"
                            }
                        },
                        "required": ["server_name"]
                    }
                ),
                Tool(
                    name="get_unified_market_data",
                    description="Get unified market data from multiple sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Stock symbols to analyze"
                            },
                            "include_finance": {
                                "type": "boolean",
                                "description": "Include financial data",
                                "default": True
                            },
                            "include_macro": {
                                "type": "boolean",
                                "description": "Include macro economic context",
                                "default": True
                            },
                            "include_alternative": {
                                "type": "boolean",
                                "description": "Include alternative data insights",
                                "default": True
                            },
                            "time_period": {
                                "type": "string",
                                "enum": ["1d", "1w", "1m", "3m", "1y"],
                                "description": "Analysis time period",
                                "default": "1m"
                            }
                        },
                        "required": ["symbols"]
                    }
                ),
                Tool(
                    name="get_investment_signals",
                    description="Get comprehensive investment signals combining all data sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Stock symbols to analyze"
                            },
                            "signal_strength": {
                                "type": "string",
                                "enum": ["conservative", "moderate", "aggressive"],
                                "description": "Signal strength threshold",
                                "default": "moderate"
                            },
                            "include_rationale": {
                                "type": "boolean",
                                "description": "Include reasoning for signals",
                                "default": True
                            }
                        },
                        "required": ["symbols"]
                    }
                )
            ])
            
            # Aggregate tools from individual servers
            for server_name, server in self.servers.items():
                try:
                    if hasattr(server, 'server') and hasattr(server.server, '_tools'):
                        server_tools = await server.server._tools()
                        for tool in server_tools:
                            # Prefix tool names with server name for disambiguation
                            tool.name = f"{server_name}_{tool.name}"
                            tools.append(tool)
                except Exception as e:
                    logger.warning(f"Could not get tools from {server_name}: {str(e)}")
            
            return tools
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, JSONContent]]:
            try:
                self.metrics["requests_total"] += 1
                
                # Handle manager-specific tools
                if name == "get_server_status":
                    return await self._get_server_status(arguments)
                elif name == "restart_server":
                    return await self._restart_server(arguments)
                elif name == "get_unified_market_data":
                    return await self._get_unified_market_data(arguments)
                elif name == "get_investment_signals":
                    return await self._get_investment_signals(arguments)
                
                # Route to appropriate server
                else:
                    for server_name in ["finance", "macro", "alternative"]:
                        if name.startswith(f"{server_name}_"):
                            # Remove prefix and route to server
                            original_name = name[len(f"{server_name}_"):]
                            server = self.servers.get(server_name)
                            
                            if server and hasattr(server, 'server'):
                                self.metrics["requests_by_server"][server_name] = \
                                    self.metrics["requests_by_server"].get(server_name, 0) + 1
                                return await server.server._call_tool(original_name, arguments)
                            else:
                                raise ValueError(f"Server {server_name} not available")
                    
                    raise ValueError(f"Unknown tool: {name}")
                    
            except Exception as e:
                self.metrics["errors_total"] += 1
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _setup_resources(self):
        """Setup unified MCP resources"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            resources = [
                Resource(
                    uri="mcp://server-status",
                    name="Server Status Dashboard",
                    description="Status and metrics for all MCP servers",
                    mimeType="application/json"
                ),
                Resource(
                    uri="mcp://unified-dashboard",
                    name="Unified Data Dashboard",
                    description="Comprehensive market data from all sources",
                    mimeType="application/json"
                ),
                Resource(
                    uri="mcp://system-metrics",
                    name="System Metrics",
                    description="Performance and usage metrics",
                    mimeType="application/json"
                )
            ]
            
            # Aggregate resources from individual servers
            for server_name, server in self.servers.items():
                try:
                    if hasattr(server, 'server') and hasattr(server.server, '_resources'):
                        server_resources = await server.server._resources()
                        for resource in server_resources:
                            # Prefix URI with server name
                            resource.uri = f"{server_name}://{resource.uri.split('://', 1)[-1]}"
                            resources.append(resource)
                except Exception as e:
                    logger.warning(f"Could not get resources from {server_name}: {str(e)}")
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "mcp://server-status":
                return await self._get_server_status_resource()
            elif uri == "mcp://unified-dashboard":
                return await self._get_unified_dashboard()
            elif uri == "mcp://system-metrics":
                return await self._get_system_metrics()
            else:
                # Route to appropriate server
                for server_name in ["finance", "macro", "alternative"]:
                    if uri.startswith(f"{server_name}://"):
                        original_uri = uri.replace(f"{server_name}://", "", 1)
                        server = self.servers.get(server_name)
                        
                        if server and hasattr(server, 'server'):
                            return await server.server._read_resource(original_uri)
                        else:
                            raise ValueError(f"Server {server_name} not available")
                
                raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementations
    async def _get_server_status(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get status of all servers"""
        detailed = arguments.get("detailed", False)
        
        status = {
            "manager": {
                "status": "running",
                "uptime": (datetime.now() - self.metrics["startup_time"]).total_seconds(),
                "requests_total": self.metrics["requests_total"],
                "errors_total": self.metrics["errors_total"]
            }
        }
        
        for server_name, server in self.servers.items():
            server_status = {
                "status": self.health_status.get(server_name, "unknown"),
                "enabled": True,
                "requests": self.metrics["requests_by_server"].get(server_name, 0)
            }
            
            if detailed:
                server_status.update({
                    "config": self._get_server_config_summary(server_name),
                    "capabilities": self._get_server_capabilities(server_name)
                })
            
            status[server_name] = server_status
        
        return [JSONContent(type="json", data={
            "servers": status,
            "timestamp": datetime.now().isoformat(),
            "total_servers": len(self.servers),
            "healthy_servers": len([s for s in self.health_status.values() if s == "healthy"])
        })]
    
    async def _restart_server(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Restart a specific server"""
        server_name = arguments["server_name"]
        
        if server_name == "all":
            # Restart all servers
            results = {}
            for name in self.servers.keys():
                results[name] = await self._restart_individual_server(name)
        else:
            # Restart specific server
            if server_name not in self.servers:
                raise ValueError(f"Server {server_name} not found")
            results = {server_name: await self._restart_individual_server(server_name)}
        
        return [JSONContent(type="json", data={
            "restart_results": results,
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_unified_market_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get unified market data from multiple sources"""
        symbols = arguments["symbols"]
        include_finance = arguments.get("include_finance", True)
        include_macro = arguments.get("include_macro", True)
        include_alternative = arguments.get("include_alternative", True)
        time_period = arguments.get("time_period", "1m")
        
        unified_data = {
            "symbols": symbols,
            "time_period": time_period,
            "data_sources": [],
            "analysis": {}
        }
        
        # Collect data from each enabled source
        for symbol in symbols:
            symbol_data = {"symbol": symbol}
            
            if include_finance and "finance" in self.servers:
                try:
                    # Get financial data
                    finance_data = await self._get_finance_data(symbol, time_period)
                    symbol_data["finance"] = finance_data
                    if "financial" not in unified_data["data_sources"]:
                        unified_data["data_sources"].append("financial")
                except Exception as e:
                    logger.warning(f"Could not get finance data for {symbol}: {str(e)}")
            
            if include_macro and "macro" in self.servers:
                try:
                    # Get macro context
                    macro_data = await self._get_macro_context(symbol)
                    symbol_data["macro"] = macro_data
                    if "macro" not in unified_data["data_sources"]:
                        unified_data["data_sources"].append("macro")
                except Exception as e:
                    logger.warning(f"Could not get macro data for {symbol}: {str(e)}")
            
            if include_alternative and "alternative" in self.servers:
                try:
                    # Get alternative data
                    alt_data = await self._get_alternative_insights(symbol, time_period)
                    symbol_data["alternative"] = alt_data
                    if "alternative" not in unified_data["data_sources"]:
                        unified_data["data_sources"].append("alternative")
                except Exception as e:
                    logger.warning(f"Could not get alternative data for {symbol}: {str(e)}")
            
            unified_data["analysis"][symbol] = symbol_data
        
        unified_data["timestamp"] = datetime.now().isoformat()
        
        return [JSONContent(type="json", data=unified_data)]
    
    async def _get_investment_signals(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get comprehensive investment signals"""
        symbols = arguments["symbols"]
        signal_strength = arguments.get("signal_strength", "moderate")
        include_rationale = arguments.get("include_rationale", True)
        
        signals = {
            "symbols": symbols,
            "signal_strength": signal_strength,
            "signals": {},
            "methodology": {
                "financial_weight": 0.4,
                "macro_weight": 0.3,
                "alternative_weight": 0.3
            } if signal_strength == "moderate" else {
                "financial_weight": 0.6,
                "macro_weight": 0.2,
                "alternative_weight": 0.2
            } if signal_strength == "conservative" else {
                "financial_weight": 0.3,
                "macro_weight": 0.3,
                "alternative_weight": 0.4
            }
        }
        
        for symbol in symbols:
            signal_data = await self._calculate_investment_signal(symbol, signal_strength, include_rationale)
            signals["signals"][symbol] = signal_data
        
        signals["timestamp"] = datetime.now().isoformat()
        
        return [JSONContent(type="json", data=signals)]
    
    # Helper methods
    async def _restart_individual_server(self, server_name: str) -> Dict[str, Any]:
        """Restart an individual server"""
        try:
            self.health_status[server_name] = "restarting"
            
            # Re-initialize the server
            if server_name == "finance":
                finance_config = FinanceConfig(
                    yahoo_finance_enabled=self.config.yahoo_finance_enabled,
                    alpha_vantage_api_key=self.config.alpha_vantage_api_key,
                    j_quants_api_key=self.config.j_quants_api_key,
                    cache_duration_minutes=self.config.cache_duration_minutes
                )
                self.servers[server_name] = FinanceServer(finance_config)
            elif server_name == "macro":
                macro_config = MacroDataConfig(
                    fred_api_key=self.config.fred_api_key,
                    world_bank_enabled=self.config.world_bank_enabled,
                    oecd_enabled=self.config.oecd_enabled
                )
                self.servers[server_name] = MacroDataServer(macro_config)
            elif server_name == "alternative":
                alt_config = AlternativeDataConfig(
                    news_api_key=self.config.news_api_key,
                    reddit_client_id=self.config.reddit_client_id,
                    reddit_client_secret=self.config.reddit_client_secret,
                    twitter_bearer_token=self.config.twitter_bearer_token
                )
                self.servers[server_name] = AlternativeDataServer(alt_config)
            
            self.health_status[server_name] = "healthy"
            
            return {
                "status": "success",
                "message": f"Server {server_name} restarted successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.health_status[server_name] = "error"
            return {
                "status": "error",
                "message": f"Failed to restart server {server_name}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_server_config_summary(self, server_name: str) -> Dict[str, Any]:
        """Get server configuration summary"""
        if server_name == "finance":
            return {
                "yahoo_finance": self.config.yahoo_finance_enabled,
                "alpha_vantage": bool(self.config.alpha_vantage_api_key),
                "j_quants": bool(self.config.j_quants_api_key)
            }
        elif server_name == "macro":
            return {
                "fred": bool(self.config.fred_api_key),
                "world_bank": self.config.world_bank_enabled,
                "oecd": self.config.oecd_enabled
            }
        elif server_name == "alternative":
            return {
                "news": bool(self.config.news_api_key),
                "reddit": bool(self.config.reddit_client_id),
                "twitter": bool(self.config.twitter_bearer_token)
            }
        return {}
    
    def _get_server_capabilities(self, server_name: str) -> List[str]:
        """Get server capabilities"""
        capabilities = {
            "finance": ["stock_prices", "technical_indicators", "fundamentals", "market_indices"],
            "macro": ["economic_indicators", "inflation_data", "interest_rates", "gdp_data"],
            "alternative": ["news_sentiment", "social_media", "esg_data", "insider_trading"]
        }
        return capabilities.get(server_name, [])
    
    async def _get_finance_data(self, symbol: str, time_period: str) -> Dict[str, Any]:
        """Get financial data for a symbol"""
        # This would call the finance server's tools
        return {
            "current_price": 150.00,
            "change_percent": 2.5,
            "volume": 1000000,
            "technical_indicators": {
                "rsi": 65.2,
                "macd": "bullish"
            }
        }
    
    async def _get_macro_context(self, symbol: str) -> Dict[str, Any]:
        """Get macro economic context"""
        # This would call the macro server's tools
        return {
            "economic_environment": "neutral",
            "interest_rate_trend": "rising",
            "inflation_impact": "moderate"
        }
    
    async def _get_alternative_insights(self, symbol: str, time_period: str) -> Dict[str, Any]:
        """Get alternative data insights"""
        # This would call the alternative server's tools
        return {
            "sentiment_score": 0.3,
            "news_volume": "high",
            "social_media_buzz": "positive",
            "esg_score": 75.5
        }
    
    async def _calculate_investment_signal(self, symbol: str, strength: str, include_rationale: bool) -> Dict[str, Any]:
        """Calculate investment signal for a symbol"""
        # This would combine data from all sources to generate signals
        signal = {
            "signal": "BUY",
            "confidence": 0.75,
            "score": 7.5,
            "recommendation": "Strong Buy"
        }
        
        if include_rationale:
            signal["rationale"] = {
                "financial": "Strong technical indicators and fundamentals",
                "macro": "Favorable economic environment",
                "alternative": "Positive sentiment and ESG profile"
            }
        
        return signal
    
    # Resource handlers
    async def _get_server_status_resource(self) -> str:
        """Get server status resource"""
        status_data = await self._get_server_status({"detailed": True})
        return json.dumps(status_data[0].data)
    
    async def _get_unified_dashboard(self) -> str:
        """Get unified dashboard resource"""
        return json.dumps({
            "dashboard": "Unified market data dashboard",
            "data_sources": list(self.servers.keys()),
            "last_updated": datetime.now().isoformat(),
            "status": "active"
        })
    
    async def _get_system_metrics(self) -> str:
        """Get system metrics resource"""
        return json.dumps({
            "metrics": self.metrics,
            "performance": {
                "uptime": (datetime.now() - self.metrics["startup_time"]).total_seconds(),
                "requests_per_minute": self.metrics["requests_total"] / max(1, (datetime.now() - self.metrics["startup_time"]).total_seconds() / 60),
                "error_rate": self.metrics["errors_total"] / max(1, self.metrics["requests_total"])
            },
            "timestamp": datetime.now().isoformat()
        })

async def main():
    """Main server function"""
    config = MCPConfig.from_env()
    
    manager = MCPServerManager(config)
    
    options = InitializationOptions(
        server_name="mcp-server-manager",
        server_version="1.0.0",
        capabilities=manager.server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    async with stdio_server() as (read_stream, write_stream):
        await manager.server.run(
            read_stream,
            write_stream,
            options
        )

if __name__ == "__main__":
    asyncio.run(main())