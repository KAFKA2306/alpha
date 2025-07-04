#!/usr/bin/env python3
"""
Pytest configuration for MCP Servers tests
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    test_env = {
        "ALPHA_VANTAGE_API_KEY": "test_av_key",
        "FRED_API_KEY": "test_fred_key", 
        "NEWS_API_KEY": "test_news_key",
        "REDDIT_CLIENT_ID": "test_reddit_id",
        "REDDIT_CLIENT_SECRET": "test_reddit_secret",
        "TWITTER_BEARER_TOKEN": "test_twitter_token",
        "JQUANTS_API_KEY": "test_jquants_key",
        "MCP_LOG_LEVEL": "DEBUG",
        "DEBUG": "true"
    }
    
    # Set test environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # Clean up test environment variables
    for key in test_env.keys():
        if key in os.environ:
            del os.environ[key]

@pytest.fixture
def mock_data_directory(tmp_path):
    """Create a temporary directory for test data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create sample configuration file
    config_file = data_dir / "test_config.yaml"
    config_file.write_text("""
mcp_servers:
  finance:
    enabled: true
    data_sources:
      yahoo_finance:
        enabled: true
  macro:
    enabled: true
  alternative:
    enabled: true
""")
    
    return data_dir

@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing"""
    return {
        "AAPL": {
            "price": 150.00,
            "change": 2.50,
            "change_percent": 1.69,
            "volume": 50000000,
            "market_cap": 2400000000000
        },
        "GOOGL": {
            "price": 2800.00,
            "change": -15.00,
            "change_percent": -0.53,
            "volume": 1200000,
            "market_cap": 1800000000000
        }
    }

@pytest.fixture
def sample_economic_data():
    """Sample economic data for testing"""
    return {
        "GDP": {
            "value": 25000,
            "unit": "Billions USD",
            "change": 2.1,
            "period": "Q4 2024"
        },
        "UNEMPLOYMENT": {
            "value": 3.7,
            "unit": "Percent",
            "change": -0.1,
            "period": "December 2024"
        },
        "INFLATION": {
            "value": 3.1,
            "unit": "Percent",
            "change": 0.2,
            "period": "December 2024"
        }
    }

@pytest.fixture
def sample_alternative_data():
    """Sample alternative data for testing"""
    return {
        "news_sentiment": {
            "score": 0.25,
            "articles_count": 150,
            "positive_ratio": 0.65,
            "negative_ratio": 0.35
        },
        "social_sentiment": {
            "twitter_score": 0.15,
            "reddit_score": 0.35,
            "mentions_count": 2500
        },
        "esg_scores": {
            "environmental": 85.2,
            "social": 78.5,
            "governance": 92.1,
            "overall": 85.3
        }
    }

# Async test utilities
@pytest.fixture
async def async_mock_response():
    """Mock async response for testing"""
    class MockResponse:
        def __init__(self, json_data, status=200):
            self.json_data = json_data
            self.status = status
            
        async def json(self):
            return self.json_data
            
        async def text(self):
            return str(self.json_data)
            
        def __aenter__(self):
            return self
            
        def __aexit__(self, exc_type, exc, tb):
            pass
    
    return MockResponse

# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require external API access"
    )