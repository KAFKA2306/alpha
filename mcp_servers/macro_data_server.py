#!/usr/bin/env python3
"""
Macro Economic Data MCP Server - Provides macroeconomic data access tools
Supports multiple data sources: FRED, World Bank, OECD, IMF
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import os

import pandas as pd
import numpy as np
from fredapi import Fred
from mcp import Server, get_model_context
from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    JSONContent,
    LoggingLevel,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    RESOURCE_NOT_FOUND,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MacroDataConfig:
    """Configuration for macro economic data sources"""
    fred_api_key: Optional[str] = None
    world_bank_enabled: bool = True
    oecd_enabled: bool = True
    imf_enabled: bool = True
    cache_duration_hours: int = 24
    max_history_years: int = 20

class MacroDataServer:
    """Macro Economic Data MCP Server implementation"""
    
    def __init__(self, config: MacroDataConfig):
        self.config = config
        self.server = Server("macro-data-server")
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize FRED API if key is provided
        self.fred = None
        if config.fred_api_key:
            self.fred = Fred(api_key=config.fred_api_key)
        
        self._setup_tools()
        self._setup_resources()
        
    def _setup_tools(self):
        """Setup MCP tools for macro economic data access"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_economic_indicator",
                    description="Get economic indicator data from FRED",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "series_id": {
                                "type": "string",
                                "description": "FRED series ID (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format",
                                "default": "2020-01-01"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format (optional)"
                            },
                            "frequency": {
                                "type": "string",
                                "enum": ["daily", "weekly", "monthly", "quarterly", "annual"],
                                "description": "Data frequency",
                                "default": "monthly"
                            },
                            "units": {
                                "type": "string",
                                "enum": ["lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"],
                                "description": "Data transformation units",
                                "default": "lin"
                            }
                        },
                        "required": ["series_id"]
                    }
                ),
                Tool(
                    name="search_economic_data",
                    description="Search for economic data series",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "search_text": {
                                "type": "string",
                                "description": "Search query for economic data"
                            },
                            "tag_names": {
                                "type": "string",
                                "description": "Filter by tag names (comma-separated)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 100
                            }
                        },
                        "required": ["search_text"]
                    }
                ),
                Tool(
                    name="get_key_indicators",
                    description="Get key economic indicators dashboard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "enum": ["us", "japan", "eurozone", "uk", "china", "global"],
                                "description": "Country/region for indicators",
                                "default": "us"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["overview", "growth", "inflation", "employment", "monetary", "fiscal", "external"],
                                "description": "Category of indicators",
                                "default": "overview"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_inflation_data",
                    description="Get comprehensive inflation data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "enum": ["us", "japan", "eurozone", "uk", "china"],
                                "description": "Country for inflation data",
                                "default": "us"
                            },
                            "measure": {
                                "type": "string",
                                "enum": ["headline", "core", "pce", "ppi", "import", "export"],
                                "description": "Inflation measure",
                                "default": "headline"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "10y", "max"],
                                "description": "Time period",
                                "default": "5y"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_interest_rates",
                    description="Get interest rate data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "enum": ["us", "japan", "eurozone", "uk", "china"],
                                "description": "Country for interest rates",
                                "default": "us"
                            },
                            "rate_type": {
                                "type": "string",
                                "enum": ["policy", "10y_bond", "2y_bond", "real_rates", "yield_curve"],
                                "description": "Type of interest rate",
                                "default": "policy"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "10y", "max"],
                                "description": "Time period",
                                "default": "5y"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_employment_data",
                    description="Get employment and labor market data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "enum": ["us", "japan", "eurozone", "uk"],
                                "description": "Country for employment data",
                                "default": "us"
                            },
                            "metric": {
                                "type": "string",
                                "enum": ["unemployment", "payrolls", "participation", "jobless_claims", "job_openings"],
                                "description": "Employment metric",
                                "default": "unemployment"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "10y", "max"],
                                "description": "Time period",
                                "default": "5y"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_gdp_data",
                    description="Get GDP and economic growth data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "country": {
                                "type": "string",
                                "enum": ["us", "japan", "eurozone", "uk", "china", "global"],
                                "description": "Country for GDP data",
                                "default": "us"
                            },
                            "component": {
                                "type": "string",
                                "enum": ["total", "consumption", "investment", "government", "net_exports", "per_capita"],
                                "description": "GDP component",
                                "default": "total"
                            },
                            "frequency": {
                                "type": "string",
                                "enum": ["quarterly", "annual"],
                                "description": "Data frequency",
                                "default": "quarterly"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["5y", "10y", "20y", "max"],
                                "description": "Time period",
                                "default": "10y"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_currency_data",
                    description="Get currency and exchange rate data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "base_currency": {
                                "type": "string",
                                "enum": ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD"],
                                "description": "Base currency",
                                "default": "USD"
                            },
                            "target_currencies": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY"]
                                },
                                "description": "Target currencies",
                                "default": ["EUR", "JPY", "GBP"]
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "10y"],
                                "description": "Time period",
                                "default": "2y"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_commodity_data",
                    description="Get commodity price data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "commodities": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["oil", "gold", "silver", "copper", "wheat", "corn", "natural_gas"]
                                },
                                "description": "Commodities to fetch",
                                "default": ["oil", "gold"]
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1y", "2y", "5y", "10y"],
                                "description": "Time period",
                                "default": "2y"
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, JSONContent]]:
            try:
                if name == "get_economic_indicator":
                    return await self._get_economic_indicator(arguments)
                elif name == "search_economic_data":
                    return await self._search_economic_data(arguments)
                elif name == "get_key_indicators":
                    return await self._get_key_indicators(arguments)
                elif name == "get_inflation_data":
                    return await self._get_inflation_data(arguments)
                elif name == "get_interest_rates":
                    return await self._get_interest_rates(arguments)
                elif name == "get_employment_data":
                    return await self._get_employment_data(arguments)
                elif name == "get_gdp_data":
                    return await self._get_gdp_data(arguments)
                elif name == "get_currency_data":
                    return await self._get_currency_data(arguments)
                elif name == "get_commodity_data":
                    return await self._get_commodity_data(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _setup_resources(self):
        """Setup MCP resources for macro economic data"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="macro://economic-dashboard",
                    name="Economic Dashboard",
                    description="Key economic indicators overview",
                    mimeType="application/json"
                ),
                Resource(
                    uri="macro://inflation-report",
                    name="Inflation Report",
                    description="Global inflation analysis",
                    mimeType="application/json"
                ),
                Resource(
                    uri="macro://central-banks",
                    name="Central Bank Policies",
                    description="Central bank policy rates and decisions",
                    mimeType="application/json"
                ),
                Resource(
                    uri="macro://yield-curves",
                    name="Yield Curves",
                    description="Government bond yield curves",
                    mimeType="application/json"
                ),
                Resource(
                    uri="macro://recession-indicators",
                    name="Recession Indicators",
                    description="Leading recession indicators",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "macro://economic-dashboard":
                return await self._get_economic_dashboard()
            elif uri == "macro://inflation-report":
                return await self._get_inflation_report()
            elif uri == "macro://central-banks":
                return await self._get_central_banks()
            elif uri == "macro://yield-curves":
                return await self._get_yield_curves()
            elif uri == "macro://recession-indicators":
                return await self._get_recession_indicators()
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementations
    async def _get_economic_indicator(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get economic indicator data from FRED"""
        if not self.fred:
            raise ValueError("FRED API key not configured")
        
        series_id = arguments["series_id"]
        start_date = arguments.get("start_date", "2020-01-01")
        end_date = arguments.get("end_date")
        frequency = arguments.get("frequency", "monthly")
        units = arguments.get("units", "lin")
        
        cache_key = f"fred_{series_id}_{start_date}_{end_date}_{frequency}_{units}"
        
        if self._is_cache_valid(cache_key):
            return [JSONContent(type="json", data=self.cache[cache_key])]
        
        try:
            # Map frequency to FRED API
            freq_map = {
                "daily": "d",
                "weekly": "w",
                "monthly": "m",
                "quarterly": "q",
                "annual": "a"
            }
            
            data = self.fred.get_series(
                series_id,
                start=start_date,
                end=end_date,
                frequency=freq_map.get(frequency, 'm'),
                units=units
            )
            
            # Get series info
            info = self.fred.get_series_info(series_id)
            
            result = {
                "series_id": series_id,
                "title": info.get("title", ""),
                "units": info.get("units", ""),
                "frequency": info.get("frequency", ""),
                "seasonal_adjustment": info.get("seasonal_adjustment", ""),
                "last_updated": info.get("last_updated", ""),
                "data": [
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "value": float(value) if pd.notna(value) else None
                    }
                    for date, value in data.items()
                ],
                "metadata": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "frequency": frequency,
                    "units": units,
                    "data_points": len(data)
                }
            }
            
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            return [JSONContent(type="json", data=result)]
            
        except Exception as e:
            logger.error(f"Error getting FRED data for {series_id}: {str(e)}")
            raise
    
    async def _search_economic_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Search for economic data series"""
        if not self.fred:
            raise ValueError("FRED API key not configured")
        
        search_text = arguments["search_text"]
        tag_names = arguments.get("tag_names")
        limit = arguments.get("limit", 100)
        
        try:
            results = self.fred.search(search_text, limit=limit)
            
            if tag_names:
                # Filter by tags if provided
                tag_list = [tag.strip() for tag in tag_names.split(",")]
                # This would require additional API calls to filter by tags
                # For now, we'll just return the search results
            
            search_results = []
            for _, row in results.iterrows():
                search_results.append({
                    "id": row.name,
                    "title": row.get("title", ""),
                    "units": row.get("units", ""),
                    "frequency": row.get("frequency", ""),
                    "seasonal_adjustment": row.get("seasonal_adjustment", ""),
                    "last_updated": row.get("last_updated", ""),
                    "popularity": row.get("popularity", 0)
                })
            
            return [JSONContent(type="json", data={
                "search_text": search_text,
                "results": search_results[:limit],
                "total_results": len(search_results),
                "timestamp": datetime.now().isoformat()
            })]
            
        except Exception as e:
            logger.error(f"Error searching FRED data: {str(e)}")
            raise
    
    async def _get_key_indicators(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get key economic indicators dashboard"""
        country = arguments.get("country", "us")
        category = arguments.get("category", "overview")
        
        # Define key indicators by country and category
        indicators = self._get_indicator_mapping(country, category)
        
        results = {}
        
        for indicator_name, series_id in indicators.items():
            try:
                if self.fred:
                    data = self.fred.get_series(series_id, limit=1)
                    if not data.empty:
                        latest_value = data.iloc[-1]
                        latest_date = data.index[-1]
                        
                        # Get previous value for change calculation
                        prev_data = self.fred.get_series(series_id, limit=2)
                        if len(prev_data) > 1:
                            prev_value = prev_data.iloc[-2]
                            change = latest_value - prev_value
                            change_percent = (change / prev_value) * 100 if prev_value != 0 else 0
                        else:
                            change = 0
                            change_percent = 0
                        
                        results[indicator_name] = {
                            "series_id": series_id,
                            "current_value": float(latest_value),
                            "date": latest_date.strftime("%Y-%m-%d"),
                            "change": float(change),
                            "change_percent": float(change_percent)
                        }
                else:
                    results[indicator_name] = {"error": "FRED API not available"}
            except Exception as e:
                results[indicator_name] = {"error": str(e)}
        
        return [JSONContent(type="json", data={
            "country": country,
            "category": category,
            "indicators": results,
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_inflation_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get comprehensive inflation data"""
        country = arguments.get("country", "us")
        measure = arguments.get("measure", "headline")
        period = arguments.get("period", "5y")
        
        # Map period to start date
        period_map = {
            "1y": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            "2y": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
            "5y": (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d"),
            "10y": (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d"),
            "max": "1950-01-01"
        }
        
        start_date = period_map.get(period, "2020-01-01")
        
        # Get inflation series based on country and measure
        inflation_series = self._get_inflation_series(country, measure)
        
        results = {}
        
        for series_name, series_id in inflation_series.items():
            try:
                if self.fred:
                    data = self.fred.get_series(series_id, start=start_date, units="pc1")  # Percent change
                    if not data.empty:
                        results[series_name] = {
                            "series_id": series_id,
                            "data": [
                                {
                                    "date": date.strftime("%Y-%m-%d"),
                                    "value": float(value) if pd.notna(value) else None
                                }
                                for date, value in data.items()
                            ],
                            "current_value": float(data.iloc[-1]) if not data.empty else None,
                            "average": float(data.mean()) if not data.empty else None,
                            "std_dev": float(data.std()) if not data.empty else None
                        }
                else:
                    results[series_name] = {"error": "FRED API not available"}
            except Exception as e:
                results[series_name] = {"error": str(e)}
        
        return [JSONContent(type="json", data={
            "country": country,
            "measure": measure,
            "period": period,
            "data": results,
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_interest_rates(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get interest rate data"""
        country = arguments.get("country", "us")
        rate_type = arguments.get("rate_type", "policy")
        period = arguments.get("period", "5y")
        
        # Implementation similar to inflation data
        # This would fetch various interest rate series
        
        return [JSONContent(type="json", data={
            "country": country,
            "rate_type": rate_type,
            "period": period,
            "message": "Interest rates implementation in progress",
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_employment_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get employment data"""
        country = arguments.get("country", "us")
        metric = arguments.get("metric", "unemployment")
        period = arguments.get("period", "5y")
        
        # Get employment series
        employment_series = self._get_employment_series(country, metric)
        
        results = {}
        
        for series_name, series_id in employment_series.items():
            try:
                if self.fred:
                    period_map = {
                        "1y": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                        "2y": (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d"),
                        "5y": (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d"),
                        "10y": (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d"),
                        "max": "1950-01-01"
                    }
                    
                    start_date = period_map.get(period, "2020-01-01")
                    data = self.fred.get_series(series_id, start=start_date)
                    
                    if not data.empty:
                        results[series_name] = {
                            "series_id": series_id,
                            "current_value": float(data.iloc[-1]),
                            "data": [
                                {
                                    "date": date.strftime("%Y-%m-%d"),
                                    "value": float(value) if pd.notna(value) else None
                                }
                                for date, value in data.items()
                            ]
                        }
                else:
                    results[series_name] = {"error": "FRED API not available"}
            except Exception as e:
                results[series_name] = {"error": str(e)}
        
        return [JSONContent(type="json", data={
            "country": country,
            "metric": metric,
            "period": period,
            "data": results,
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_gdp_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get GDP data"""
        country = arguments.get("country", "us")
        component = arguments.get("component", "total")
        frequency = arguments.get("frequency", "quarterly")
        period = arguments.get("period", "10y")
        
        # Implementation for GDP data
        return [JSONContent(type="json", data={
            "country": country,
            "component": component,
            "frequency": frequency,
            "period": period,
            "message": "GDP data implementation in progress",
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_currency_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get currency data"""
        base_currency = arguments.get("base_currency", "USD")
        target_currencies = arguments.get("target_currencies", ["EUR", "JPY", "GBP"])
        period = arguments.get("period", "2y")
        
        # Implementation for currency data
        return [JSONContent(type="json", data={
            "base_currency": base_currency,
            "target_currencies": target_currencies,
            "period": period,
            "message": "Currency data implementation in progress",
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _get_commodity_data(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get commodity data"""
        commodities = arguments.get("commodities", ["oil", "gold"])
        period = arguments.get("period", "2y")
        
        # Implementation for commodity data
        return [JSONContent(type="json", data={
            "commodities": commodities,
            "period": period,
            "message": "Commodity data implementation in progress",
            "timestamp": datetime.now().isoformat()
        })]
    
    # Helper methods
    def _get_indicator_mapping(self, country: str, category: str) -> Dict[str, str]:
        """Get indicator mapping for country and category"""
        mappings = {
            "us": {
                "overview": {
                    "gdp": "GDP",
                    "unemployment": "UNRATE",
                    "inflation": "CPIAUCSL",
                    "fed_funds": "FEDFUNDS",
                    "10y_treasury": "GS10"
                },
                "growth": {
                    "gdp": "GDP",
                    "gdp_growth": "A191RL1Q225SBEA",
                    "industrial_production": "INDPRO",
                    "retail_sales": "RSXFS"
                },
                "employment": {
                    "unemployment": "UNRATE",
                    "payrolls": "PAYEMS",
                    "jobless_claims": "ICSA",
                    "participation_rate": "CIVPART"
                }
            },
            "japan": {
                "overview": {
                    "gdp": "JPNRGDPEXP",
                    "inflation": "JPNCPIALLMINMEI",
                    "unemployment": "LRUNTTTTJPM156S"
                }
            }
        }
        
        return mappings.get(country, {}).get(category, {})
    
    def _get_inflation_series(self, country: str, measure: str) -> Dict[str, str]:
        """Get inflation series mapping"""
        series_map = {
            "us": {
                "headline": {"cpi": "CPIAUCSL"},
                "core": {"core_cpi": "CPILFESL"},
                "pce": {"pce": "PCEPI"}
            },
            "japan": {
                "headline": {"cpi": "JPNCPIALLMINMEI"}
            }
        }
        
        return series_map.get(country, {}).get(measure, {})
    
    def _get_employment_series(self, country: str, metric: str) -> Dict[str, str]:
        """Get employment series mapping"""
        series_map = {
            "us": {
                "unemployment": {"unemployment_rate": "UNRATE"},
                "payrolls": {"nonfarm_payrolls": "PAYEMS"},
                "jobless_claims": {"initial_claims": "ICSA"}
            }
        }
        
        return series_map.get(country, {}).get(metric, {})
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False
        
        elapsed_hours = (datetime.now() - cache_time).total_seconds() / 3600
        return elapsed_hours < self.config.cache_duration_hours
    
    # Resource handlers
    async def _get_economic_dashboard(self) -> str:
        """Get economic dashboard"""
        return json.dumps({
            "dashboard": "Economic indicators overview",
            "timestamp": datetime.now().isoformat(),
            "note": "Dashboard implementation in progress"
        })
    
    async def _get_inflation_report(self) -> str:
        """Get inflation report"""
        return json.dumps({
            "report": "Global inflation analysis",
            "timestamp": datetime.now().isoformat(),
            "note": "Inflation report implementation in progress"
        })
    
    async def _get_central_banks(self) -> str:
        """Get central bank policies"""
        return json.dumps({
            "central_banks": "Policy rates and decisions",
            "timestamp": datetime.now().isoformat(),
            "note": "Central bank data implementation in progress"
        })
    
    async def _get_yield_curves(self) -> str:
        """Get yield curves"""
        return json.dumps({
            "yield_curves": "Government bond yields",
            "timestamp": datetime.now().isoformat(),
            "note": "Yield curve implementation in progress"
        })
    
    async def _get_recession_indicators(self) -> str:
        """Get recession indicators"""
        return json.dumps({
            "recession_indicators": "Leading recession indicators",
            "timestamp": datetime.now().isoformat(),
            "note": "Recession indicators implementation in progress"
        })

async def main():
    """Main server function"""
    config = MacroDataConfig(
        fred_api_key=os.getenv("FRED_API_KEY")
    )
    
    server = MacroDataServer(config)
    
    options = InitializationOptions(
        server_name="macro-data-server",
        server_version="1.0.0",
        capabilities=server.server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            options
        )

if __name__ == "__main__":
    asyncio.run(main())