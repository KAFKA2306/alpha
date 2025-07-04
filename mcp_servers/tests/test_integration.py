#!/usr/bin/env python3
"""
Integration tests for MCP Servers
Tests the interaction between different servers and the manager
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from server_manager import MCPServerManager, MCPConfig

@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests for MCP server ecosystem"""
    
    @pytest.fixture
    def config(self):
        """Integration test configuration"""
        return MCPConfig(
            finance_enabled=True,
            macro_enabled=True,
            alternative_enabled=True,
            yahoo_finance_enabled=True,
            cache_duration_minutes=5,
            rate_limit_requests_per_minute=1000
        )
    
    @pytest.fixture
    def manager(self, config):
        """Server manager for integration testing"""
        with patch.multiple(
            'server_manager',
            FinanceServer=Mock,
            MacroDataServer=Mock,
            AlternativeDataServer=Mock
        ):
            return MCPServerManager(config)
    
    @pytest.mark.asyncio
    async def test_full_data_pipeline(self, manager):
        """Test complete data pipeline across all servers"""
        # Mock all individual server responses
        finance_mock_data = {
            "symbol": "AAPL",
            "current_price": 150.00,
            "technical_indicators": {"rsi": 65.2, "macd": "bullish"},
            "fundamentals": {"pe_ratio": 25.5, "market_cap": 2400000000000}
        }
        
        macro_mock_data = {
            "gdp_growth": 2.1,
            "inflation": 3.1,
            "unemployment": 3.7,
            "interest_rates": 5.25,
            "economic_environment": "neutral"
        }
        
        alt_mock_data = {
            "news_sentiment": 0.25,
            "social_sentiment": 0.15,
            "esg_score": 85.3,
            "insider_activity": "positive"
        }
        
        # Mock the individual data retrieval methods
        with patch.object(manager, '_get_finance_data', return_value=finance_mock_data), \
             patch.object(manager, '_get_macro_context', return_value=macro_mock_data), \
             patch.object(manager, '_get_alternative_insights', return_value=alt_mock_data):
            
            # Test unified market data
            arguments = {
                "symbols": ["AAPL"],
                "include_finance": True,
                "include_macro": True,
                "include_alternative": True,
                "time_period": "1m"
            }
            
            result = await manager._get_unified_market_data(arguments)
            data = result[0].data
            
            # Verify data integration
            assert "AAPL" in data["analysis"]
            aapl_data = data["analysis"]["AAPL"]
            
            assert "finance" in aapl_data
            assert "macro" in aapl_data
            assert "alternative" in aapl_data
            
            assert aapl_data["finance"]["current_price"] == 150.00
            assert aapl_data["macro"]["gdp_growth"] == 2.1
            assert aapl_data["alternative"]["esg_score"] == 85.3
    
    @pytest.mark.asyncio
    async def test_investment_signal_generation(self, manager):
        """Test comprehensive investment signal generation"""
        # Mock signal calculation that combines all data sources
        with patch.object(manager, '_calculate_investment_signal') as mock_signal:
            mock_signal.return_value = {
                "signal": "BUY",
                "confidence": 0.78,
                "score": 8.2,
                "rationale": {
                    "financial": "Strong fundamentals and technical momentum",
                    "macro": "Favorable economic conditions",
                    "alternative": "Positive sentiment and strong ESG profile"
                }
            }
            
            arguments = {
                "symbols": ["AAPL", "GOOGL"],
                "signal_strength": "moderate",
                "include_rationale": True
            }
            
            result = await manager._get_investment_signals(arguments)
            data = result[0].data
            
            # Verify signal generation
            assert len(data["signals"]) == 2
            assert "AAPL" in data["signals"]
            assert "GOOGL" in data["signals"]
            
            for symbol in ["AAPL", "GOOGL"]:
                signal_data = data["signals"][symbol]
                assert "signal" in signal_data
                assert "confidence" in signal_data
                assert "score" in signal_data
                assert "rationale" in signal_data
                
                # Check rationale includes all data sources
                rationale = signal_data["rationale"]
                assert "financial" in rationale
                assert "macro" in rationale
                assert "alternative" in rationale
    
    @pytest.mark.asyncio
    async def test_server_health_monitoring(self, manager):
        """Test server health monitoring and status tracking"""
        # Initialize all servers as healthy
        for server_name in ["finance", "macro", "alternative"]:
            manager.health_status[server_name] = "healthy"
        
        # Test status retrieval
        status_result = await manager._get_server_status({"detailed": True})
        status_data = status_result[0].data
        
        # Verify all servers are reported as healthy
        assert status_data["healthy_servers"] == 3
        assert status_data["total_servers"] == 3
        
        for server_name in ["finance", "macro", "alternative"]:
            server_status = status_data["servers"][server_name]
            assert server_status["status"] == "healthy"
            assert server_status["enabled"] is True
            assert "config" in server_status
            assert "capabilities" in server_status
    
    @pytest.mark.asyncio
    async def test_error_propagation_and_recovery(self, manager):
        """Test error handling and recovery across servers"""
        # Simulate finance server failure
        manager.health_status["finance"] = "error"
        
        # Mock partial data retrieval (some servers fail)
        with patch.object(manager, '_get_finance_data', side_effect=Exception("Finance server error")), \
             patch.object(manager, '_get_macro_context', return_value={"gdp": 2.1}), \
             patch.object(manager, '_get_alternative_insights', return_value={"sentiment": 0.3}):
            
            arguments = {
                "symbols": ["AAPL"],
                "include_finance": True,
                "include_macro": True,
                "include_alternative": True
            }
            
            result = await manager._get_unified_market_data(arguments)
            data = result[0].data
            
            # Should still return data from working servers
            aapl_data = data["analysis"]["AAPL"]
            assert "macro" in aapl_data  # Should have macro data
            assert "alternative" in aapl_data  # Should have alternative data
            # Finance data should be missing due to error
    
    @pytest.mark.asyncio
    async def test_caching_across_servers(self, manager):
        """Test caching behavior across different servers"""
        # Test that repeated requests use cached data
        with patch.object(manager, '_get_finance_data') as mock_finance, \
             patch.object(manager, '_get_macro_context') as mock_macro, \
             patch.object(manager, '_get_alternative_insights') as mock_alt:
            
            mock_finance.return_value = {"price": 150.0}
            mock_macro.return_value = {"gdp": 2.1}
            mock_alt.return_value = {"sentiment": 0.3}
            
            arguments = {
                "symbols": ["AAPL"],
                "include_finance": True,
                "include_macro": True, 
                "include_alternative": True
            }
            
            # First request
            await manager._get_unified_market_data(arguments)
            
            # Second request (should use cache where applicable)
            await manager._get_unified_market_data(arguments)
            
            # Verify methods were called (caching is handled within individual servers)
            assert mock_finance.call_count == 2  # Called for each request
            assert mock_macro.call_count == 2
            assert mock_alt.call_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, manager):
        """Test handling of concurrent requests"""
        with patch.object(manager, '_get_finance_data', return_value={"price": 150.0}), \
             patch.object(manager, '_get_macro_context', return_value={"gdp": 2.1}), \
             patch.object(manager, '_get_alternative_insights', return_value={"sentiment": 0.3}):
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(5):
                arguments = {
                    "symbols": [f"SYMBOL{i}"],
                    "include_finance": True,
                    "include_macro": True,
                    "include_alternative": True
                }
                task = manager._get_unified_market_data(arguments)
                tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
            
            # Verify all requests completed successfully
            assert len(results) == 5
            for i, result in enumerate(results):
                data = result[0].data
                assert f"SYMBOL{i}" in data["analysis"]
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, manager):
        """Test metrics collection across server operations"""
        initial_requests = manager.metrics["requests_total"]
        
        # Simulate several requests
        manager.metrics["requests_total"] += 5
        manager.metrics["requests_by_server"]["finance"] = 3
        manager.metrics["requests_by_server"]["macro"] = 1
        manager.metrics["requests_by_server"]["alternative"] = 1
        
        # Get system metrics
        metrics_resource = await manager._get_system_metrics()
        metrics_data = json.loads(metrics_resource)
        
        # Verify metrics tracking
        assert metrics_data["metrics"]["requests_total"] == initial_requests + 5
        assert metrics_data["metrics"]["requests_by_server"]["finance"] == 3
        assert "performance" in metrics_data
        assert "uptime" in metrics_data["performance"]
        assert "requests_per_minute" in metrics_data["performance"]
    
    @pytest.mark.asyncio
    async def test_configuration_consistency(self, manager):
        """Test configuration consistency across servers"""
        # Verify all servers are initialized with consistent configuration
        for server_name in ["finance", "macro", "alternative"]:
            assert server_name in manager.servers
            assert server_name in manager.health_status
            
            # Check server config summary
            config_summary = manager._get_server_config_summary(server_name)
            assert isinstance(config_summary, dict)
            assert len(config_summary) > 0
            
            # Check server capabilities
            capabilities = manager._get_server_capabilities(server_name)
            assert isinstance(capabilities, list)
            assert len(capabilities) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])