#!/usr/bin/env python3
"""
Finance MCP Server - Provides financial data access tools
Supports multiple data sources: Yahoo Finance, Alpha Vantage, J-Quants
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

import yfinance as yf
import pandas as pd
import numpy as np
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
class FinanceConfig:
    """Configuration for finance data sources"""
    yahoo_finance_enabled: bool = True
    alpha_vantage_api_key: Optional[str] = None
    j_quants_api_key: Optional[str] = None
    j_quants_refresh_token: Optional[str] = None
    default_source: str = "yahoo_finance"
    max_history_days: int = 365 * 5  # 5 years max
    cache_duration_minutes: int = 15

class FinanceServer:
    """Finance MCP Server implementation"""
    
    def __init__(self, config: FinanceConfig):
        self.config = config
        self.server = Server("finance-server")
        self.cache = {}
        self.cache_timestamps = {}
        self._setup_tools()
        self._setup_resources()
        
    def _setup_tools(self):
        """Setup MCP tools for financial data access"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="get_stock_price",
                    description="Get current or historical stock price data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol (e.g., 'AAPL', '7203.T')"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                                "description": "Time period for historical data",
                                "default": "1y"
                            },
                            "interval": {
                                "type": "string",
                                "enum": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                                "description": "Data interval",
                                "default": "1d"
                            },
                            "source": {
                                "type": "string",
                                "enum": ["yahoo_finance", "alpha_vantage", "j_quants"],
                                "description": "Data source",
                                "default": "yahoo_finance"
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="get_multiple_stocks",
                    description="Get price data for multiple stocks",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of stock symbols"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                                "description": "Time period for historical data",
                                "default": "1y"
                            },
                            "interval": {
                                "type": "string",
                                "enum": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                                "description": "Data interval",
                                "default": "1d"
                            }
                        },
                        "required": ["symbols"]
                    }
                ),
                Tool(
                    name="get_stock_info",
                    description="Get comprehensive stock information and fundamentals",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol"
                            },
                            "include_financials": {
                                "type": "boolean",
                                "description": "Include financial statements",
                                "default": True
                            },
                            "include_recommendations": {
                                "type": "boolean",
                                "description": "Include analyst recommendations",
                                "default": True
                            }
                        },
                        "required": ["symbol"]
                    }
                ),
                Tool(
                    name="calculate_technical_indicators",
                    description="Calculate technical indicators for a stock",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol"
                            },
                            "indicators": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["sma", "ema", "rsi", "macd", "bollinger", "stochastic", "atr", "adx"]
                                },
                                "description": "List of technical indicators to calculate"
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                                "description": "Time period for calculation",
                                "default": "1y"
                            }
                        },
                        "required": ["symbol", "indicators"]
                    }
                ),
                Tool(
                    name="get_market_indices",
                    description="Get major market indices data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "indices": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["^GSPC", "^DJI", "^IXIC", "^N225", "^FTSE", "^GDAXI", "^HSI"]
                                },
                                "description": "Market indices to fetch",
                                "default": ["^GSPC", "^DJI", "^IXIC", "^N225"]
                            },
                            "period": {
                                "type": "string",
                                "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                                "description": "Time period",
                                "default": "1mo"
                            }
                        }
                    }
                ),
                Tool(
                    name="screen_stocks",
                    description="Screen stocks based on criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "market": {
                                "type": "string",
                                "enum": ["us", "japan", "global"],
                                "description": "Market to screen",
                                "default": "us"
                            },
                            "criteria": {
                                "type": "object",
                                "properties": {
                                    "min_market_cap": {"type": "number", "description": "Minimum market cap"},
                                    "max_market_cap": {"type": "number", "description": "Maximum market cap"},
                                    "min_pe_ratio": {"type": "number", "description": "Minimum P/E ratio"},
                                    "max_pe_ratio": {"type": "number", "description": "Maximum P/E ratio"},
                                    "min_dividend_yield": {"type": "number", "description": "Minimum dividend yield"},
                                    "sector": {"type": "string", "description": "Sector filter"},
                                    "min_volume": {"type": "number", "description": "Minimum average volume"}
                                },
                                "description": "Screening criteria"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 50
                            }
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[Union[TextContent, JSONContent]]:
            try:
                if name == "get_stock_price":
                    return await self._get_stock_price(arguments)
                elif name == "get_multiple_stocks":
                    return await self._get_multiple_stocks(arguments)
                elif name == "get_stock_info":
                    return await self._get_stock_info(arguments)
                elif name == "calculate_technical_indicators":
                    return await self._calculate_technical_indicators(arguments)
                elif name == "get_market_indices":
                    return await self._get_market_indices(arguments)
                elif name == "screen_stocks":
                    return await self._screen_stocks(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    def _setup_resources(self):
        """Setup MCP resources for financial data"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="finance://market-summary",
                    name="Market Summary",
                    description="Current market overview and key indices",
                    mimeType="application/json"
                ),
                Resource(
                    uri="finance://top-movers",
                    name="Top Movers",
                    description="Top gaining and losing stocks",
                    mimeType="application/json"
                ),
                Resource(
                    uri="finance://earnings-calendar",
                    name="Earnings Calendar",
                    description="Upcoming earnings announcements",
                    mimeType="application/json"
                ),
                Resource(
                    uri="finance://economic-calendar",
                    name="Economic Calendar",
                    description="Upcoming economic events",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "finance://market-summary":
                return await self._get_market_summary()
            elif uri == "finance://top-movers":
                return await self._get_top_movers()
            elif uri == "finance://earnings-calendar":
                return await self._get_earnings_calendar()
            elif uri == "finance://economic-calendar":
                return await self._get_economic_calendar()
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    async def _get_stock_price(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get stock price data"""
        symbol = arguments["symbol"]
        period = arguments.get("period", "1y")
        interval = arguments.get("interval", "1d")
        source = arguments.get("source", "yahoo_finance")
        
        cache_key = f"{symbol}_{period}_{interval}_{source}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return [JSONContent(type="json", data=self.cache[cache_key])]
        
        try:
            if source == "yahoo_finance":
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    raise ValueError(f"No data found for symbol {symbol}")
                
                # Convert to JSON-serializable format
                data = {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                    "data": hist.reset_index().to_dict(orient="records"),
                    "metadata": {
                        "timezone": str(hist.index.tz) if hasattr(hist.index, 'tz') else None,
                        "start_date": hist.index[0].isoformat() if not hist.empty else None,
                        "end_date": hist.index[-1].isoformat() if not hist.empty else None,
                        "data_points": len(hist)
                    }
                }
                
                # Cache the result
                self.cache[cache_key] = data
                self.cache_timestamps[cache_key] = datetime.now()
                
                return [JSONContent(type="json", data=data)]
            
            else:
                raise ValueError(f"Source {source} not implemented yet")
                
        except Exception as e:
            logger.error(f"Error getting stock price for {symbol}: {str(e)}")
            raise
    
    async def _get_multiple_stocks(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get price data for multiple stocks"""
        symbols = arguments["symbols"]
        period = arguments.get("period", "1y")
        interval = arguments.get("interval", "1d")
        
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    results[symbol] = {
                        "data": hist.reset_index().to_dict(orient="records"),
                        "metadata": {
                            "start_date": hist.index[0].isoformat(),
                            "end_date": hist.index[-1].isoformat(),
                            "data_points": len(hist)
                        }
                    }
                else:
                    results[symbol] = {"error": "No data available"}
                    
            except Exception as e:
                results[symbol] = {"error": str(e)}
        
        return [JSONContent(type="json", data={
            "symbols": symbols,
            "period": period,
            "interval": interval,
            "results": results
        })]
    
    async def _get_stock_info(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get comprehensive stock information"""
        symbol = arguments["symbol"]
        include_financials = arguments.get("include_financials", True)
        include_recommendations = arguments.get("include_recommendations", True)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            result = {
                "symbol": symbol,
                "basic_info": {
                    "longName": info.get("longName"),
                    "shortName": info.get("shortName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "currency": info.get("currency"),
                    "exchange": info.get("exchange"),
                    "marketCap": info.get("marketCap"),
                    "enterpriseValue": info.get("enterpriseValue"),
                    "sharesOutstanding": info.get("sharesOutstanding"),
                    "floatShares": info.get("floatShares")
                },
                "valuation": {
                    "currentPrice": info.get("currentPrice"),
                    "previousClose": info.get("previousClose"),
                    "open": info.get("open"),
                    "dayLow": info.get("dayLow"),
                    "dayHigh": info.get("dayHigh"),
                    "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                    "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
                    "priceToEarningsRatio": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "priceToBook": info.get("priceToBook"),
                    "priceToSales": info.get("priceToSalesTrailing12Months"),
                    "enterpriseToRevenue": info.get("enterpriseToRevenue"),
                    "enterpriseToEbitda": info.get("enterpriseToEbitda")
                },
                "financial_metrics": {
                    "totalRevenue": info.get("totalRevenue"),
                    "revenuePerShare": info.get("revenuePerShare"),
                    "totalCash": info.get("totalCash"),
                    "totalDebt": info.get("totalDebt"),
                    "totalCashPerShare": info.get("totalCashPerShare"),
                    "currentRatio": info.get("currentRatio"),
                    "quickRatio": info.get("quickRatio"),
                    "debtToEquity": info.get("debtToEquity"),
                    "returnOnAssets": info.get("returnOnAssets"),
                    "returnOnEquity": info.get("returnOnEquity"),
                    "grossMargins": info.get("grossMargins"),
                    "operatingMargins": info.get("operatingMargins"),
                    "profitMargins": info.get("profitMargins")
                },
                "dividend_info": {
                    "dividendRate": info.get("dividendRate"),
                    "dividendYield": info.get("dividendYield"),
                    "payoutRatio": info.get("payoutRatio"),
                    "exDividendDate": info.get("exDividendDate"),
                    "fiveYearAvgDividendYield": info.get("fiveYearAvgDividendYield")
                },
                "trading_info": {
                    "volume": info.get("volume"),
                    "averageVolume": info.get("averageVolume"),
                    "averageVolume10days": info.get("averageVolume10days"),
                    "beta": info.get("beta"),
                    "impliedSharesOutstanding": info.get("impliedSharesOutstanding")
                }
            }
            
            if include_financials:
                try:
                    # Get financial statements
                    result["financials"] = {
                        "income_statement": ticker.financials.to_dict() if hasattr(ticker, 'financials') else None,
                        "balance_sheet": ticker.balance_sheet.to_dict() if hasattr(ticker, 'balance_sheet') else None,
                        "cashflow": ticker.cashflow.to_dict() if hasattr(ticker, 'cashflow') else None
                    }
                except Exception as e:
                    result["financials"] = {"error": str(e)}
            
            if include_recommendations:
                try:
                    recommendations = ticker.recommendations
                    if recommendations is not None:
                        result["recommendations"] = recommendations.to_dict(orient="records")
                except Exception as e:
                    result["recommendations"] = {"error": str(e)}
            
            return [JSONContent(type="json", data=result)]
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            raise
    
    async def _calculate_technical_indicators(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Calculate technical indicators"""
        symbol = arguments["symbol"]
        indicators = arguments["indicators"]
        period = arguments.get("period", "1y")
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            result = {
                "symbol": symbol,
                "period": period,
                "indicators": {}
            }
            
            # Calculate requested indicators
            for indicator in indicators:
                if indicator == "sma":
                    result["indicators"]["sma_20"] = hist["Close"].rolling(window=20).mean().tolist()
                    result["indicators"]["sma_50"] = hist["Close"].rolling(window=50).mean().tolist()
                    result["indicators"]["sma_200"] = hist["Close"].rolling(window=200).mean().tolist()
                
                elif indicator == "ema":
                    result["indicators"]["ema_12"] = hist["Close"].ewm(span=12).mean().tolist()
                    result["indicators"]["ema_26"] = hist["Close"].ewm(span=26).mean().tolist()
                
                elif indicator == "rsi":
                    delta = hist["Close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    result["indicators"]["rsi"] = (100 - (100 / (1 + rs))).tolist()
                
                elif indicator == "macd":
                    ema_12 = hist["Close"].ewm(span=12).mean()
                    ema_26 = hist["Close"].ewm(span=26).mean()
                    macd = ema_12 - ema_26
                    signal = macd.ewm(span=9).mean()
                    result["indicators"]["macd"] = macd.tolist()
                    result["indicators"]["macd_signal"] = signal.tolist()
                    result["indicators"]["macd_histogram"] = (macd - signal).tolist()
                
                elif indicator == "bollinger":
                    sma_20 = hist["Close"].rolling(window=20).mean()
                    std_20 = hist["Close"].rolling(window=20).std()
                    result["indicators"]["bollinger_upper"] = (sma_20 + (std_20 * 2)).tolist()
                    result["indicators"]["bollinger_lower"] = (sma_20 - (std_20 * 2)).tolist()
                    result["indicators"]["bollinger_middle"] = sma_20.tolist()
                
                elif indicator == "atr":
                    high_low = hist["High"] - hist["Low"]
                    high_close = np.abs(hist["High"] - hist["Close"].shift())
                    low_close = np.abs(hist["Low"] - hist["Close"].shift())
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    result["indicators"]["atr"] = true_range.rolling(window=14).mean().tolist()
            
            # Add dates for reference
            result["dates"] = hist.index.strftime('%Y-%m-%d').tolist()
            
            return [JSONContent(type="json", data=result)]
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            raise
    
    async def _get_market_indices(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Get market indices data"""
        indices = arguments.get("indices", ["^GSPC", "^DJI", "^IXIC", "^N225"])
        period = arguments.get("period", "1mo")
        
        results = {}
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
                    previous_close = hist["Close"].iloc[-2] if len(hist) > 1 else hist["Close"].iloc[-1]
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    results[index] = {
                        "name": self._get_index_name(index),
                        "current_price": float(current_price),
                        "previous_close": float(previous_close),
                        "change": float(change),
                        "change_percent": float(change_percent),
                        "volume": int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0,
                        "high_52w": float(hist["High"].max()),
                        "low_52w": float(hist["Low"].min()),
                        "historical_data": hist.reset_index().to_dict(orient="records")
                    }
                else:
                    results[index] = {"error": "No data available"}
                    
            except Exception as e:
                results[index] = {"error": str(e)}
        
        return [JSONContent(type="json", data={
            "indices": indices,
            "period": period,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })]
    
    async def _screen_stocks(self, arguments: Dict[str, Any]) -> List[JSONContent]:
        """Screen stocks based on criteria"""
        market = arguments.get("market", "us")
        criteria = arguments.get("criteria", {})
        limit = arguments.get("limit", 50)
        
        # This is a simplified implementation
        # In a real implementation, you would use a proper screening API
        
        if market == "us":
            # Sample US stocks for demonstration
            sample_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        elif market == "japan":
            # Sample Japanese stocks
            sample_symbols = ["7203.T", "6758.T", "9984.T", "7974.T", "4063.T"]
        else:
            sample_symbols = ["AAPL", "MSFT", "7203.T", "6758.T"]
        
        results = []
        
        for symbol in sample_symbols[:limit]:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Simple filtering based on criteria
                if self._meets_criteria(info, criteria):
                    results.append({
                        "symbol": symbol,
                        "name": info.get("longName"),
                        "sector": info.get("sector"),
                        "market_cap": info.get("marketCap"),
                        "pe_ratio": info.get("trailingPE"),
                        "dividend_yield": info.get("dividendYield"),
                        "current_price": info.get("currentPrice"),
                        "volume": info.get("volume")
                    })
            except Exception as e:
                logger.warning(f"Error screening {symbol}: {str(e)}")
                continue
        
        return [JSONContent(type="json", data={
            "market": market,
            "criteria": criteria,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        })]
    
    def _meets_criteria(self, info: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        for key, value in criteria.items():
            if key == "min_market_cap" and info.get("marketCap", 0) < value:
                return False
            elif key == "max_market_cap" and info.get("marketCap", float('inf')) > value:
                return False
            elif key == "min_pe_ratio" and (info.get("trailingPE") or 0) < value:
                return False
            elif key == "max_pe_ratio" and (info.get("trailingPE") or float('inf')) > value:
                return False
            elif key == "min_dividend_yield" and (info.get("dividendYield") or 0) < value:
                return False
            elif key == "sector" and info.get("sector") != value:
                return False
            elif key == "min_volume" and (info.get("volume") or 0) < value:
                return False
        return True
    
    def _get_index_name(self, symbol: str) -> str:
        """Get human-readable name for index symbol"""
        names = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones Industrial Average",
            "^IXIC": "NASDAQ Composite",
            "^N225": "Nikkei 225",
            "^FTSE": "FTSE 100",
            "^GDAXI": "DAX",
            "^HSI": "Hang Seng Index"
        }
        return names.get(symbol, symbol)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time:
            return False
        
        elapsed_minutes = (datetime.now() - cache_time).total_seconds() / 60
        return elapsed_minutes < self.config.cache_duration_minutes
    
    # Resource handlers
    async def _get_market_summary(self) -> str:
        """Get market summary"""
        indices = ["^GSPC", "^DJI", "^IXIC", "^N225"]
        summary = {"indices": {}, "timestamp": datetime.now().isoformat()}
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist["Close"].iloc[-1]
                    previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_percent = (change / previous) * 100
                    
                    summary["indices"][index] = {
                        "name": self._get_index_name(index),
                        "current": float(current),
                        "change": float(change),
                        "change_percent": float(change_percent)
                    }
            except Exception as e:
                logger.warning(f"Error getting {index}: {str(e)}")
        
        return json.dumps(summary)
    
    async def _get_top_movers(self) -> str:
        """Get top movers"""
        # This would typically use a proper API for real-time data
        return json.dumps({
            "gainers": [],
            "losers": [],
            "most_active": [],
            "timestamp": datetime.now().isoformat(),
            "note": "Top movers feature requires real-time data API"
        })
    
    async def _get_earnings_calendar(self) -> str:
        """Get earnings calendar"""
        return json.dumps({
            "upcoming_earnings": [],
            "timestamp": datetime.now().isoformat(),
            "note": "Earnings calendar feature requires specialized API"
        })
    
    async def _get_economic_calendar(self) -> str:
        """Get economic calendar"""
        return json.dumps({
            "upcoming_events": [],
            "timestamp": datetime.now().isoformat(),
            "note": "Economic calendar feature requires specialized API"
        })

async def main():
    """Main server function"""
    config = FinanceConfig()
    
    # Initialize server
    server = FinanceServer(config)
    
    # Setup options
    options = InitializationOptions(
        server_name="finance-server",
        server_version="1.0.0",
        capabilities=server.server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={}
        )
    )
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            options
        )

if __name__ == "__main__":
    asyncio.run(main())