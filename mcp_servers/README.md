# Alpha Architecture Agent - MCP Servers

A comprehensive Model Context Protocol (MCP) server ecosystem for financial data, macroeconomic indicators, and alternative data sources. Designed to support AI-driven investment strategy generation and analysis.

## Overview

This package provides three specialized MCP servers that work together to deliver comprehensive market intelligence:

- **Finance Server**: Real-time and historical financial market data
- **Macro Data Server**: Macroeconomic indicators and analysis
- **Alternative Data Server**: Non-traditional data sources for investment insights

## Features

### 🏦 Finance Server
- **Stock Data**: Real-time prices, historical data, technical indicators
- **Market Indices**: Major global indices tracking
- **Fundamentals**: Company financials, ratios, and metrics
- **Screening**: Custom stock screening with multiple criteria
- **Data Sources**: Yahoo Finance, Alpha Vantage, J-Quants (Japanese markets)

### 📊 Macro Data Server
- **Economic Indicators**: GDP, inflation, unemployment, interest rates
- **Central Bank Data**: Policy rates and monetary policy decisions
- **Global Coverage**: US (FRED), World Bank, OECD, IMF data
- **Time Series**: Historical economic data with multiple frequencies
- **Forecasting Support**: Data formatted for economic modeling

### 🔍 Alternative Data Server
- **Sentiment Analysis**: News and social media sentiment tracking
- **ESG Data**: Environmental, social, governance metrics
- **Insider Trading**: Corporate insider transaction monitoring
- **Patent Data**: Innovation and IP tracking
- **Supply Chain**: Logistics and supply chain intelligence
- **Satellite Data**: Geospatial economic indicators

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install additional requirements for specific features
pip install textblob  # For sentiment analysis
pip install fredapi  # For FRED economic data
```

### Configuration

1. **Environment Variables**: Set up API keys in your environment
```bash
export ALPHA_VANTAGE_API_KEY="your_key"
export FRED_API_KEY="your_key"
export NEWS_API_KEY="your_key"
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export TWITTER_BEARER_TOKEN="your_token"
```

2. **Configuration File**: Update `config/mcp_servers.yaml`
```yaml
mcp_servers:
  finance:
    enabled: true
    data_sources:
      yahoo_finance:
        enabled: true
      alpha_vantage:
        enabled: true  # Set to true when API key is available
```

### Running Servers

#### Individual Servers
```bash
# Finance Server
python finance_server.py

# Macro Data Server  
python macro_data_server.py

# Alternative Data Server
python alternative_data_server.py
```

#### Unified Manager
```bash
# Run all servers through the manager
python server_manager.py
```

## Usage Examples

### Getting Stock Data
```python
# Using the finance server
arguments = {
    "symbol": "AAPL",
    "period": "1y",
    "interval": "1d"
}
result = await finance_server.get_stock_price(arguments)
```

### Economic Indicators
```python
# Using the macro server
arguments = {
    "series_id": "GDP",
    "start_date": "2020-01-01",
    "frequency": "quarterly"
}
result = await macro_server.get_economic_indicator(arguments)
```

### Sentiment Analysis
```python
# Using the alternative data server
arguments = {
    "query": "Apple Inc",
    "time_period": "1w",
    "sentiment_analysis": True
}
result = await alt_server.get_news_sentiment(arguments)
```

### Unified Market Intelligence
```python
# Using the server manager for comprehensive analysis
arguments = {
    "symbols": ["AAPL", "GOOGL"],
    "include_finance": True,
    "include_macro": True,
    "include_alternative": True
}
result = await manager.get_unified_market_data(arguments)
```

## Architecture

### Server Architecture
```
┌─────────────────────────────────────────────┐
│           MCP Server Manager                │
├─────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │
│  │   Finance   │ │    Macro    │ │   Alt   │ │
│  │   Server    │ │    Data     │ │  Data   │ │
│  │             │ │   Server    │ │ Server  │ │
│  └─────────────┘ └─────────────┘ └─────────┘ │
└─────────────────────────────────────────────┘
```

### Data Flow
1. **Request Routing**: Manager routes requests to appropriate servers
2. **Data Aggregation**: Combines responses from multiple sources
3. **Signal Generation**: Creates investment signals from unified data
4. **Caching**: Intelligent caching to minimize API calls
5. **Error Handling**: Graceful degradation when sources are unavailable

## API Reference

### Finance Server Tools

#### `get_stock_price`
Get current or historical stock price data.

**Parameters:**
- `symbol` (string): Stock symbol (e.g., 'AAPL', '7203.T')
- `period` (string): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
- `interval` (string): Data interval ('1m', '2m', '5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
- `source` (string): Data source ('yahoo_finance', 'alpha_vantage', 'j_quants')

#### `calculate_technical_indicators`
Calculate technical indicators for a stock.

**Parameters:**
- `symbol` (string): Stock symbol
- `indicators` (array): List of indicators ('sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic', 'atr', 'adx')
- `period` (string): Time period for calculation

### Macro Data Server Tools

#### `get_economic_indicator`
Get economic indicator data from FRED.

**Parameters:**
- `series_id` (string): FRED series ID (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')
- `start_date` (string): Start date in YYYY-MM-DD format
- `end_date` (string): End date in YYYY-MM-DD format
- `frequency` (string): Data frequency ('daily', 'weekly', 'monthly', 'quarterly', 'annual')
- `units` (string): Data transformation units

#### `get_key_indicators`
Get key economic indicators dashboard.

**Parameters:**
- `country` (string): Country/region ('us', 'japan', 'eurozone', 'uk', 'china', 'global')
- `category` (string): Category ('overview', 'growth', 'inflation', 'employment', 'monetary', 'fiscal')

### Alternative Data Server Tools

#### `get_news_sentiment`
Get news sentiment analysis for stocks or topics.

**Parameters:**
- `query` (string): Search query (company name, ticker, or topic)
- `language` (string): Language filter ('en', 'ja', 'all')
- `time_period` (string): Time period ('1d', '3d', '1w', '1m')
- `sources` (array): News sources to include
- `sentiment_analysis` (boolean): Include sentiment analysis

#### `get_esg_data`
Get ESG (Environmental, Social, Governance) data.

**Parameters:**
- `symbol` (string): Company stock symbol
- `esg_category` (string): ESG category ('environmental', 'social', 'governance', 'all')
- `include_scores` (boolean): Include ESG scores
- `include_controversies` (boolean): Include ESG controversies

## Resources

Each server provides resources that can be accessed through the MCP protocol:

### Finance Resources
- `finance://market-summary`: Current market overview
- `finance://top-movers`: Top gaining and losing stocks
- `finance://earnings-calendar`: Upcoming earnings announcements

### Macro Resources
- `macro://economic-dashboard`: Key economic indicators overview
- `macro://inflation-report`: Global inflation analysis
- `macro://central-banks`: Central bank policy rates and decisions

### Alternative Data Resources
- `alt-data://market-sentiment`: Overall market sentiment dashboard
- `alt-data://trending-topics`: Trending topics in financial social media
- `alt-data://esg-leaders`: Top ESG performing companies

## Configuration

### Server Configuration (`config/mcp_servers.yaml`)

```yaml
mcp_servers:
  manager:
    enabled: true
    port: 8000
    log_level: "INFO"
    
  finance:
    enabled: true
    port: 8001
    data_sources:
      yahoo_finance:
        enabled: true
        rate_limit: 2000
      alpha_vantage:
        enabled: false
        api_key: ${ALPHA_VANTAGE_API_KEY}
        
  macro:
    enabled: true
    port: 8002
    data_sources:
      fred:
        enabled: false
        api_key: ${FRED_API_KEY}
        
  alternative:
    enabled: true
    port: 8003
    data_sources:
      news_api:
        enabled: false
        api_key: ${NEWS_API_KEY}
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | Optional |
| `FRED_API_KEY` | FRED (Federal Reserve) API key | Optional |
| `NEWS_API_KEY` | News API key | Optional |
| `REDDIT_CLIENT_ID` | Reddit API client ID | Optional |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret | Optional |
| `TWITTER_BEARER_TOKEN` | Twitter API bearer token | Optional |
| `JQUANTS_API_KEY` | J-Quants API key (Japanese markets) | Optional |
| `MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | Optional |

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests

# Run with coverage
pytest --cov=mcp_servers --cov-report=html
```

### Test Structure
- `tests/test_finance_server.py`: Finance server tests
- `tests/test_server_manager.py`: Server manager tests
- `tests/test_integration.py`: Integration tests
- `tests/conftest.py`: Test configuration and fixtures

## Development

### Adding New Data Sources

1. **Finance Server**: Add new provider in `finance_server.py`
2. **Macro Server**: Add new economic data source in `macro_data_server.py`
3. **Alternative Server**: Add new alternative data provider in `alternative_data_server.py`

### Extending Functionality

1. Create new tool methods in the appropriate server
2. Update the tool list in `_setup_tools()`
3. Add corresponding tests
4. Update documentation

### Performance Optimization

- **Caching**: Implement intelligent caching strategies
- **Rate Limiting**: Respect API rate limits
- **Async Operations**: Use async/await for I/O operations
- **Connection Pooling**: Reuse HTTP connections
- **Data Compression**: Compress large datasets

## Monitoring and Observability

### Health Checks
- Endpoint: `/health`
- Checks: Server status, API connectivity, cache status

### Metrics
- Request counts by server
- Response times
- Error rates
- Cache hit rates
- API quota usage

### Logging
- Structured logging with JSON format
- Log levels: DEBUG, INFO, WARNING, ERROR
- Request/response logging
- Error tracking and alerting

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000 8001 8002 8003

CMD ["python", "server_manager.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  mcp-servers:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
      - "8003:8003"
    environment:
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - FRED_API_KEY=${FRED_API_KEY}
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
```

### Security Considerations

- **API Key Management**: Secure storage and rotation
- **Rate Limiting**: Implement request throttling
- **Input Validation**: Sanitize all inputs
- **Access Control**: Implement authentication/authorization
- **Audit Logging**: Track all data access

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set
2. **Rate Limiting**: Check API quotas and implement backoff
3. **Data Quality**: Validate data sources and handle missing data
4. **Network Issues**: Implement retry logic and timeout handling

### Debug Mode
```bash
export MCP_LOG_LEVEL=DEBUG
export DEBUG=true
python server_manager.py
```

### Logs Location
- Application logs: `logs/mcp_servers.log`
- Error logs: `logs/errors.log`
- Access logs: `logs/access.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Ensure all tests pass

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Include comprehensive tests

## License

This project is part of the Alpha Architecture Agent and follows the same licensing terms.

## Support

For questions and support:
- Check the troubleshooting section
- Review the test cases for usage examples
- Create an issue in the repository

---

**Alpha Architecture Agent MCP Servers** - Powering AI-driven investment intelligence through comprehensive data integration.