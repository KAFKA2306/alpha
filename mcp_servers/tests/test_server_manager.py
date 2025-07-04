#!/usr/bin/env python3
"""
Tests for MCP Server Manager
"""
import pytest
import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from server_manager import MCPServerManager, MCPConfig
from mcp.types import JSONContent, TextContent

class TestMCPConfig:
    """Test cases for MCP Configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = MCPConfig()
        
        assert config.finance_enabled is True
        assert config.macro_enabled is True
        assert config.alternative_enabled is True
        assert config.yahoo_finance_enabled is True
        assert config.cache_duration_minutes == 15
        assert config.rate_limit_requests_per_minute == 100
        assert config.max_concurrent_requests == 50
    
    def test_from_env(self):
        """Test configuration from environment variables"""
        # Mock environment variables
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_av_key",
            "FRED_API_KEY": "test_fred_key",
            "NEWS_API_KEY": "test_news_key",
            "MCP_LOG_LEVEL": "DEBUG",
            "MCP_METRICS_ENABLED": "false"
        }
        
        with patch.dict(os.environ, env_vars):
            config = MCPConfig.from_env()
            
            assert config.alpha_vantage_api_key == "test_av_key"
            assert config.fred_api_key == "test_fred_key"
            assert config.news_api_key == "test_news_key"
            assert config.log_level == "DEBUG"
            assert config.metrics_enabled is False
    
    def test_server_ports(self):
        """Test server port configuration"""
        config = MCPConfig()
        
        assert config.server_ports["finance"] == 8001
        assert config.server_ports["macro"] == 8002
        assert config.server_ports["alternative"] == 8003
        assert config.server_ports["manager"] == 8000

class TestMCPServerManager:
    """Test cases for MCP Server Manager"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return MCPConfig(
            finance_enabled=True,
            macro_enabled=True,
            alternative_enabled=True,
            cache_duration_minutes=1,  # Short cache for testing
            rate_limit_requests_per_minute=1000  # High limit for testing
        )
    
    @pytest.fixture
    def manager(self, config):
        """Server manager instance"""
        with patch.multiple(
            'server_manager',
            FinanceServer=Mock,
            MacroDataServer=Mock,
            AlternativeDataServer=Mock
        ):
            return MCPServerManager(config)
    
    def test_initialization(self, manager):
        """Test server manager initialization"""
        assert "finance" in manager.servers
        assert "macro" in manager.servers
        assert "alternative" in manager.servers
        
        assert "finance" in manager.health_status
        assert "macro" in manager.health_status
        assert "alternative" in manager.health_status
        
        assert manager.metrics["requests_total"] == 0
        assert manager.metrics["errors_total"] == 0
        assert isinstance(manager.metrics["startup_time"], datetime)
    
    @pytest.mark.asyncio
    async def test_get_server_status(self, manager):
        """Test server status retrieval"""
        # Update health status
        manager.health_status["finance"] = "healthy"
        manager.health_status["macro"] = "healthy"
        manager.health_status["alternative"] = "healthy"
        
        arguments = {"detailed": False}
        result = await manager._get_server_status(arguments)
        
        assert len(result) == 1
        assert isinstance(result[0], JSONContent)
        data = result[0].data
        
        assert "servers" in data
        assert "finance" in data["servers"]
        assert "macro" in data["servers"]
        assert "alternative" in data["servers"]
        assert "manager" in data["servers"]
        
        assert data["servers"]["finance"]["status"] == "healthy"
        assert data["total_servers"] == 3
        assert data["healthy_servers"] == 3
    
    @pytest.mark.asyncio
    async def test_get_server_status_detailed(self, manager):
        """Test detailed server status"""
        arguments = {"detailed": True}
        result = await manager._get_server_status(arguments)
        
        assert len(result) == 1
        data = result[0].data
        
        # Check that detailed info is included
        for server_name in ["finance", "macro", "alternative"]:
            server_data = data["servers"][server_name]
            assert "config" in server_data
            assert "capabilities" in server_data
    
    @pytest.mark.asyncio
    async def test_restart_server(self, manager):
        """Test server restart functionality"""
        with patch.object(manager, '_restart_individual_server', 
                         return_value={"status": "success", "message": "Restarted"}) as mock_restart:
            
            arguments = {"server_name": "finance"}
            result = await manager._restart_server(arguments)
            
            assert len(result) == 1
            data = result[0].data
            
            assert "restart_results" in data
            assert "finance" in data["restart_results"]
            assert data["restart_results"]["finance"]["status"] == "success"
            
            mock_restart.assert_called_once_with("finance")
    
    @pytest.mark.asyncio
    async def test_restart_all_servers(self, manager):
        """Test restarting all servers"""
        with patch.object(manager, '_restart_individual_server', 
                         return_value={"status": "success", "message": "Restarted"}) as mock_restart:
            
            arguments = {"server_name": "all"}
            result = await manager._restart_server(arguments)
            
            assert len(result) == 1
            data = result[0].data
            
            assert "restart_results" in data
            assert len(data["restart_results"]) == 3  # All three servers
            
            # Check that restart was called for each server
            assert mock_restart.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_unified_market_data(self, manager):
        """Test unified market data retrieval"""
        # Mock individual data retrieval methods
        with patch.object(manager, '_get_finance_data', 
                         return_value={"price": 150.0, "change": 2.5}) as mock_finance, \
             patch.object(manager, '_get_macro_context', 
                         return_value={"environment": "neutral"}) as mock_macro, \
             patch.object(manager, '_get_alternative_insights', 
                         return_value={"sentiment": 0.3}) as mock_alt:
            
            arguments = {
                "symbols": ["AAPL", "GOOGL"],
                "include_finance": True,
                "include_macro": True,
                "include_alternative": True,
                "time_period": "1m"
            }
            
            result = await manager._get_unified_market_data(arguments)
            
            assert len(result) == 1
            data = result[0].data
            
            assert data["symbols"] == ["AAPL", "GOOGL"]
            assert data["time_period"] == "1m"
            assert "financial" in data["data_sources"]
            assert "macro" in data["data_sources"]
            assert "alternative" in data["data_sources"]
            
            assert "AAPL" in data["analysis"]
            assert "GOOGL" in data["analysis"]
            
            # Check that all data sources were called
            assert mock_finance.call_count == 2  # Once for each symbol
            assert mock_macro.call_count == 2
            assert mock_alt.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_investment_signals(self, manager):
        """Test investment signals generation"""
        with patch.object(manager, '_calculate_investment_signal', 
                         return_value={
                             "signal": "BUY",
                             "confidence": 0.75,
                             "score": 7.5
                         }) as mock_signal:
            
            arguments = {
                "symbols": ["AAPL"],
                "signal_strength": "moderate",
                "include_rationale": True
            }
            
            result = await manager._get_investment_signals(arguments)
            
            assert len(result) == 1
            data = result[0].data
            
            assert data["symbols"] == ["AAPL"]
            assert data["signal_strength"] == "moderate"
            assert "methodology" in data
            assert "AAPL" in data["signals"]
            
            signal_data = data["signals"]["AAPL"]
            assert signal_data["signal"] == "BUY"
            assert signal_data["confidence"] == 0.75
            
            mock_signal.assert_called_once_with("AAPL", "moderate", True)
    
    def test_server_config_summary(self, manager):
        """Test server configuration summary"""
        # Test finance config
        finance_config = manager._get_server_config_summary("finance")
        assert "yahoo_finance" in finance_config
        assert "alpha_vantage" in finance_config
        assert "j_quants" in finance_config
        
        # Test macro config
        macro_config = manager._get_server_config_summary("macro")
        assert "fred" in macro_config
        assert "world_bank" in macro_config
        assert "oecd" in macro_config
        
        # Test alternative config
        alt_config = manager._get_server_config_summary("alternative")
        assert "news" in alt_config
        assert "reddit" in alt_config
        assert "twitter" in alt_config
    
    def test_server_capabilities(self, manager):
        """Test server capabilities listing"""
        finance_caps = manager._get_server_capabilities("finance")
        assert "stock_prices" in finance_caps
        assert "technical_indicators" in finance_caps
        assert "fundamentals" in finance_caps
        assert "market_indices" in finance_caps
        
        macro_caps = manager._get_server_capabilities("macro")
        assert "economic_indicators" in macro_caps
        assert "inflation_data" in macro_caps
        assert "interest_rates" in macro_caps
        assert "gdp_data" in macro_caps
        
        alt_caps = manager._get_server_capabilities("alternative")
        assert "news_sentiment" in alt_caps
        assert "social_media" in alt_caps
        assert "esg_data" in alt_caps
        assert "insider_trading" in alt_caps
    
    @pytest.mark.asyncio
    async def test_restart_individual_server(self, manager):
        """Test individual server restart"""
        with patch.multiple(
            'server_manager',
            FinanceServer=Mock,
            MacroDataServer=Mock,
            AlternativeDataServer=Mock
        ):
            result = await manager._restart_individual_server("finance")
            
            assert result["status"] == "success"
            assert "finance" in result["message"]
            assert manager.health_status["finance"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_restart_server_error(self, manager):
        """Test server restart error handling"""
        with patch('server_manager.FinanceServer', side_effect=Exception("Restart failed")):
            result = await manager._restart_individual_server("finance")
            
            assert result["status"] == "error"
            assert "Restart failed" in result["message"]
            assert manager.health_status["finance"] == "error"
    
    def test_metrics_tracking(self, manager):
        """Test metrics tracking"""
        initial_requests = manager.metrics["requests_total"]
        initial_errors = manager.metrics["errors_total"]
        
        # Simulate request
        manager.metrics["requests_total"] += 1
        manager.metrics["requests_by_server"]["finance"] = 1
        
        assert manager.metrics["requests_total"] == initial_requests + 1
        assert manager.metrics["requests_by_server"]["finance"] == 1
        
        # Simulate error
        manager.metrics["errors_total"] += 1
        
        assert manager.metrics["errors_total"] == initial_errors + 1
    
    @pytest.mark.asyncio
    async def test_resource_handlers(self, manager):
        """Test resource handlers"""
        # Test server status resource
        status_resource = await manager._get_server_status_resource()
        status_data = json.loads(status_resource)
        assert "servers" in status_data
        
        # Test unified dashboard
        dashboard = await manager._get_unified_dashboard()
        dashboard_data = json.loads(dashboard)
        assert "dashboard" in dashboard_data
        assert "data_sources" in dashboard_data
        
        # Test system metrics
        metrics = await manager._get_system_metrics()
        metrics_data = json.loads(metrics)
        assert "metrics" in metrics_data
        assert "performance" in metrics_data
    
    @pytest.mark.asyncio
    async def test_error_handling_in_tools(self, manager):
        """Test error handling in tool calls"""
        # Test with invalid server name in restart
        arguments = {"server_name": "invalid_server"}
        
        with pytest.raises(ValueError, match="Server invalid_server not found"):
            await manager._restart_server(arguments)
    
    @pytest.mark.asyncio 
    async def test_data_aggregation_methods(self, manager):
        """Test data aggregation helper methods"""
        # Test finance data method
        finance_data = await manager._get_finance_data("AAPL", "1m")
        assert "current_price" in finance_data
        assert "technical_indicators" in finance_data
        
        # Test macro context method
        macro_data = await manager._get_macro_context("AAPL")
        assert "economic_environment" in macro_data
        
        # Test alternative insights method
        alt_data = await manager._get_alternative_insights("AAPL", "1m")
        assert "sentiment_score" in alt_data
        
        # Test investment signal calculation
        signal = await manager._calculate_investment_signal("AAPL", "moderate", True)
        assert "signal" in signal
        assert "confidence" in signal
        assert "rationale" in signal

if __name__ == "__main__":
    pytest.main([__file__])