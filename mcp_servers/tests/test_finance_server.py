#!/usr/bin/env python3
"""
Tests for Finance MCP Server
"""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from finance_server import FinanceServer, FinanceConfig
from mcp.types import JSONContent, TextContent

class TestFinanceServer:
    """Test cases for Finance MCP Server"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return FinanceConfig(
            yahoo_finance_enabled=True,
            alpha_vantage_api_key="test_key",
            j_quants_api_key=None,
            cache_duration_minutes=1  # Short cache for testing
        )
    
    @pytest.fixture
    def server(self, config):
        """Finance server instance"""
        return FinanceServer(config)
    
    @pytest.fixture
    def sample_stock_data(self):
        """Sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        return pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(110, 120, len(dates)),
            'Low': np.random.uniform(90, 100, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_get_stock_price(self, server, sample_stock_data):
        """Test stock price retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock yfinance response
            mock_instance = Mock()
            mock_instance.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_instance
            
            # Test stock price retrieval
            arguments = {
                "symbol": "AAPL",
                "period": "1y",
                "interval": "1d"
            }
            
            result = await server._get_stock_price(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert data["symbol"] == "AAPL"
            assert data["period"] == "1y"
            assert data["interval"] == "1d"
            assert "data" in data
            assert "metadata" in data
            assert len(data["data"]) == len(sample_stock_data)
    
    @pytest.mark.asyncio
    async def test_get_multiple_stocks(self, server, sample_stock_data):
        """Test multiple stocks retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock yfinance response
            mock_instance = Mock()
            mock_instance.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_instance
            
            arguments = {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "period": "1m",
                "interval": "1d"
            }
            
            result = await server._get_multiple_stocks(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert "results" in data
            assert len(data["results"]) == 3
            assert all(symbol in data["results"] for symbol in ["AAPL", "GOOGL", "MSFT"])
    
    @pytest.mark.asyncio
    async def test_get_stock_info(self, server):
        """Test stock info retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock comprehensive stock info
            mock_instance = Mock()
            mock_instance.info = {
                "longName": "Apple Inc.",
                "sector": "Technology",
                "marketCap": 3000000000000,
                "currentPrice": 150.00,
                "trailingPE": 25.5,
                "dividendYield": 0.005
            }
            mock_ticker.return_value = mock_instance
            
            arguments = {
                "symbol": "AAPL",
                "include_financials": False,
                "include_recommendations": False
            }
            
            result = await server._get_stock_info(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert data["symbol"] == "AAPL"
            assert "basic_info" in data
            assert "valuation" in data
            assert data["basic_info"]["longName"] == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, server, sample_stock_data):
        """Test technical indicators calculation"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_instance
            
            arguments = {
                "symbol": "AAPL",
                "indicators": ["sma", "rsi", "macd"],
                "period": "1y"
            }
            
            result = await server._calculate_technical_indicators(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert data["symbol"] == "AAPL"
            assert "indicators" in data
            indicators = data["indicators"]
            
            # Check SMA indicators
            assert "sma_20" in indicators
            assert "sma_50" in indicators
            assert "sma_200" in indicators
            
            # Check RSI
            assert "rsi" in indicators
            
            # Check MACD
            assert "macd" in indicators
            assert "macd_signal" in indicators
            assert "macd_histogram" in indicators
    
    @pytest.mark.asyncio
    async def test_get_market_indices(self, server, sample_stock_data):
        """Test market indices retrieval"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_instance = Mock()
            mock_instance.history.return_value = sample_stock_data
            mock_ticker.return_value = mock_instance
            
            arguments = {
                "indices": ["^GSPC", "^DJI"],
                "period": "1mo"
            }
            
            result = await server._get_market_indices(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert "results" in data
            assert "^GSPC" in data["results"]
            assert "^DJI" in data["results"]
    
    @pytest.mark.asyncio
    async def test_screen_stocks(self, server):
        """Test stock screening"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock stock info for screening
            mock_instance = Mock()
            mock_instance.info = {
                "longName": "Test Company",
                "marketCap": 1000000000,
                "trailingPE": 15.0,
                "dividendYield": 0.02,
                "sector": "Technology",
                "currentPrice": 100.0,
                "volume": 1000000
            }
            mock_ticker.return_value = mock_instance
            
            arguments = {
                "market": "us",
                "criteria": {
                    "min_market_cap": 500000000,
                    "max_pe_ratio": 20.0
                },
                "limit": 10
            }
            
            result = await server._screen_stocks(arguments)
            
            assert len(result) == 1
            assert isinstance(result[0], JSONContent)
            data = result[0].data
            
            assert "results" in data
            assert "market" in data
            assert data["market"] == "us"
    
    def test_cache_functionality(self, server):
        """Test caching mechanism"""
        # Test cache validity
        cache_key = "test_key"
        
        # Initially no cache
        assert not server._is_cache_valid(cache_key)
        
        # Add to cache
        server.cache[cache_key] = {"test": "data"}
        server.cache_timestamps[cache_key] = datetime.now()
        
        # Should be valid immediately
        assert server._is_cache_valid(cache_key)
        
        # Mock old timestamp
        server.cache_timestamps[cache_key] = datetime.now() - timedelta(minutes=20)
        
        # Should be invalid after expiration
        assert not server._is_cache_valid(cache_key)
    
    def test_index_name_mapping(self, server):
        """Test index name mapping"""
        assert server._get_index_name("^GSPC") == "S&P 500"
        assert server._get_index_name("^DJI") == "Dow Jones Industrial Average"
        assert server._get_index_name("^IXIC") == "NASDAQ Composite"
        assert server._get_index_name("^N225") == "Nikkei 225"
        assert server._get_index_name("UNKNOWN") == "UNKNOWN"
    
    def test_screening_criteria(self, server):
        """Test stock screening criteria"""
        # Test stock that meets criteria
        good_stock = {
            "marketCap": 1000000000,
            "trailingPE": 15.0,
            "dividendYield": 0.02,
            "sector": "Technology",
            "volume": 1000000
        }
        
        criteria = {
            "min_market_cap": 500000000,
            "max_pe_ratio": 20.0,
            "sector": "Technology"
        }
        
        assert server._meets_criteria(good_stock, criteria)
        
        # Test stock that doesn't meet criteria
        bad_stock = {
            "marketCap": 100000000,  # Too small
            "trailingPE": 15.0,
            "sector": "Technology"
        }
        
        assert not server._meets_criteria(bad_stock, criteria)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock exception
            mock_ticker.side_effect = Exception("API Error")
            
            arguments = {"symbol": "INVALID"}
            
            with pytest.raises(Exception):
                await server._get_stock_price(arguments)
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, server):
        """Test handling of empty data"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock empty dataframe
            mock_instance = Mock()
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_instance
            
            arguments = {"symbol": "NODATA"}
            
            with pytest.raises(ValueError, match="No data found"):
                await server._get_stock_price(arguments)

class TestFinanceConfig:
    """Test cases for Finance Configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = FinanceConfig()
        
        assert config.yahoo_finance_enabled is True
        assert config.alpha_vantage_api_key is None
        assert config.j_quants_api_key is None
        assert config.default_source == "yahoo_finance"
        assert config.max_history_days == 365 * 5
        assert config.cache_duration_minutes == 15
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = FinanceConfig(
            yahoo_finance_enabled=False,
            alpha_vantage_api_key="test_key",
            cache_duration_minutes=30
        )
        
        assert config.yahoo_finance_enabled is False
        assert config.alpha_vantage_api_key == "test_key"
        assert config.cache_duration_minutes == 30

if __name__ == "__main__":
    pytest.main([__file__])