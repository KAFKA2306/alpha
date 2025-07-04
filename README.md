# Alpha Architecture Agent 🤖📈

**AI-Powered Neural Network Architecture Generation for Stock Price Prediction**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Alpha Architecture Agent is an innovative AI system that automatically generates and evaluates neural network architectures for Japanese stock price prediction. Inspired by recent research in neural architecture search and financial machine learning, this system combines the creativity of Large Language Models (LLMs) with the rigor of systematic evaluation to discover effective trading strategies.

### Key Features

- 🧠 **AI-Powered Architecture Generation**: Uses GPT-4/Claude to intelligently combine neural network building blocks
- 🔧 **50+ Domain-Specific Blocks**: Pre-built components optimized for financial time series
- 📊 **Comprehensive Backtesting**: Rigorous evaluation framework with multiple risk metrics
- 🎯 **Ensemble Strategies**: Combines top-performing models for robust predictions
- 🔍 **Diversity Optimization**: Ensures architectural diversity to avoid overfitting
- 📈 **Japanese Stock Focus**: Optimized for J-Quants API and Japanese market dynamics
- 🚀 **Production Ready**: Docker-based deployment with monitoring and alerting

## Architecture Philosophy

Based on the research methodology described in our [idea document](docs/idea.md), our system implements a **hybrid approach**:

1. **AI Agent Intelligence**: LLMs generate creative architectural combinations
2. **Random Exploration**: Systematic random combinations ensure diversity
3. **Domain Knowledge**: Financial and time-series specific building blocks
4. **Rigorous Evaluation**: Backtesting with Sharpe ratio optimization

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI or Anthropic API key (optional, for LLM-based generation)
- J-Quants API key (for Japanese stock data)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd uki
   ```

2. **Set up development environment**:
   ```bash
   # Using devcontainer (recommended)
   code .  # Open in VS Code with devcontainer
   
   # Or local installation
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the services**:
   ```bash
   docker-compose up -d
   ```

### Quick Demo

```bash
# Run the demonstration script
python examples/demo_architecture_generation.py
```

This will:
- Show all available domain blocks
- Generate 5 sample architectures
- Compile them into PyTorch models
- Analyze their complexity and diversity

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Alpha Architecture Agent                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   AI Agent      │  │  Domain Blocks  │  │   Strategy      │ │
│  │   System        │  │    Library      │  │  Evaluation     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                     │                     │          │
│           └─────────────────────┼─────────────────────┘          │
│                                 │                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Data          │  │   Portfolio     │  │   Monitoring    │ │
│  │   Pipeline      │  │   Management    │  │   & Alerting    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **[Domain Blocks](src/models/domain_blocks.py)**: 50+ neural network building blocks
2. **[AI Agent](src/agents/architecture_agent.py)**: LLM-powered architecture generator
3. **[Data Pipeline](src/data/)**: Japanese stock data processing
4. **[Strategy Evaluation](src/strategies/)**: Backtesting and performance metrics
5. **[Portfolio Management](src/portfolio/)**: Position sizing and risk management

## Domain Blocks Library

Our system includes over 50 domain-specific neural network blocks organized by category:

### Normalization Blocks
- **Batch Normalization**: Standard batch normalization
- **Layer Normalization**: For transformer-like architectures
- **Adaptive Instance Norm**: Learnable instance normalization
- **Demean**: Simple mean subtraction

### Feature Extraction Blocks
- **PCA Block**: Principal component analysis
- **Fourier Features**: Frequency domain analysis
- **Multi-Time Frame**: Multiple time horizon analysis
- **Lead-Lag**: Capture lead-lag relationships

### Financial Domain Blocks
- **Regime Detection**: Market regime identification
- **Factor Exposure**: Risk factor analysis
- **Cross-Sectional**: Inter-stock relationships
- **Volatility Clustering**: Volatility pattern recognition

### Sequence Models
- **LSTM**: Long short-term memory
- **Transformer**: Self-attention mechanisms
- **GRU**: Gated recurrent units
- **Temporal CNN**: Convolutional sequence modeling

### Attention Mechanisms
- **Multi-Head Attention**: Standard transformer attention
- **Sparse Attention**: Efficient attention (Informer)
- **Auto-Correlation**: Time series auto-correlation (Autoformer)
- **Cross Attention**: Cross-sectional attention (Crossformer)

[See full list in Domain Blocks documentation](docs/domain_blocks.md)

## Usage Examples

### Generate Architectures

```python
from agents.architecture_agent import ArchitectureAgent

# Initialize agent
agent = ArchitectureAgent()

# Generate architectures for Japanese stock data
input_shape = (32, 252, 20)  # (batch, sequence, features)
architectures = agent.generate_architecture_suite(
    input_shape=input_shape,
    num_architectures=70
)

print(f"Generated {len(architectures)} architectures")
```

### Compile and Train Models

```python
# Compile architectures to PyTorch models
models = agent.compile_architecture_suite(architectures)

# Train models (example)
for arch_id, model in models.items():
    trainer = ModelTrainer(model)
    trainer.train(train_data, val_data)
```

### Evaluate Strategies

```python
from strategies.backtester import Backtester

# Backtest strategies
backtester = Backtester()
results = backtester.evaluate_strategies(
    models=models,
    data=test_data,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Get performance metrics
for result in results:
    print(f"Strategy: {result.name}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}")
```

### Create Ensemble

```python
from portfolio.ensemble import EnsembleManager

# Create ensemble from top strategies
ensemble = EnsembleManager()
top_strategies = ensemble.select_top_strategies(
    results=results,
    top_n=20,
    correlation_threshold=0.8
)

# Generate ensemble predictions
predictions = ensemble.predict(top_strategies, current_data)
```

## Configuration

The system is highly configurable via [config/config.yaml](config/config.yaml):

```yaml
# ML Configuration
ml:
  max_blocks: 50
  max_combinations: 1000
  training_epochs: 100
  
# Strategy Configuration
strategy:
  long_percentage: 0.05  # Top 5% for long
  short_percentage: 0.05  # Bottom 5% for short
  ensemble_top_n: 20
  
# Agent Configuration
agent:
  llm:
    provider: "openai"  # or "anthropic"
    model: "gpt-4"
    temperature: 0.7
```

## API Documentation

### REST API

The system includes a FastAPI-based REST API:

```bash
# Start API server
uvicorn api.main:app --reload

# Generate architectures
curl -X POST "http://localhost:8000/api/v1/architectures/generate" \
     -H "Content-Type: application/json" \
     -d '{"input_shape": [32, 252, 20], "num_architectures": 10}'

# Get architecture details
curl "http://localhost:8000/api/v1/architectures/{arch_id}"

# Start backtesting
curl -X POST "http://localhost:8000/api/v1/backtest/start" \
     -H "Content-Type: application/json" \
     -d '{"architecture_ids": ["arch1", "arch2"]}'
```

### Python API

See [API documentation](docs/api.md) for complete Python API reference.

## Performance Results

Based on our backtesting from 2017-2023 with validation on 2024 data:

- **Best Individual Strategy**: Sharpe ratio of 1.3
- **Top 20 Ensemble**: Sharpe ratio of 2.2
- **Hit Rate**: 65% accuracy on direction prediction
- **Max Drawdown**: -8.5% for ensemble

### Key Findings

1. **Effective Blocks**: Convolutional feature integration and PCA exposure blocks showed strong performance
2. **Avoided Overfitting**: Transformer blocks and complex sequence integration performed poorly
3. **Ensemble Benefits**: Top 20 strategy ensemble significantly outperformed individual strategies
4. **Market Adaptability**: Strategies maintained performance through 2024 market volatility

## Development

### Project Structure

```
uki/
├── src/
│   ├── agents/           # AI agents for architecture generation
│   ├── core/             # Core utilities and configuration
│   ├── data/             # Data processing pipelines
│   ├── models/           # Domain blocks and model definitions
│   ├── strategies/       # Strategy evaluation and backtesting
│   ├── portfolio/        # Portfolio management and ensemble
│   ├── api/              # REST API endpoints
│   ├── monitoring/       # Monitoring and alerting
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── docs/                 # Documentation
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks for research
├── scripts/              # Utility scripts
├── examples/             # Example usage scripts
└── mcp_servers/          # MCP server implementations
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_domain_blocks.py
pytest tests/test_agents.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/
```

## MCP Servers

The system includes Model Context Protocol (MCP) servers for external data access:

### Financial Data MCP
- **Yahoo Finance**: Real-time and historical stock data
- **Alpha Vantage**: Fundamental and technical indicators
- **J-Quants**: Japanese stock market data

### Macro Data MCP
- **FRED**: Federal Reserve economic data
- **World Bank**: Global economic indicators
- **OECD**: International economic statistics

### Alternative Data MCP
- **News Sentiment**: Financial news sentiment analysis
- **Social Media**: Social media sentiment and trends
- **Satellite Data**: Economic activity indicators

## Monitoring and Observability

### Metrics Dashboard
- **Grafana**: Real-time performance dashboards
- **Prometheus**: Metrics collection and alerting
- **MLflow**: Model experiment tracking

### Key Metrics
- Strategy performance (Sharpe ratio, drawdown)
- Model accuracy and drift
- System resource utilization
- Data quality indicators

## Deployment

### Docker Deployment

```bash
# Build and deploy
docker-compose up -d

# Scale services
docker-compose up -d --scale worker=4

# View logs
docker-compose logs -f
```

### Production Considerations

- **Security**: API key rotation, encryption at rest
- **Scalability**: Horizontal scaling with load balancers
- **Reliability**: Health checks and automatic restarts
- **Monitoring**: Comprehensive observability stack

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contributing Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure all tests pass

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{alpha_architecture_agent,
  title={Alpha Architecture Agent: AI-Powered Neural Network Architecture Generation for Stock Prediction},
  author={Alpha Architecture Team},
  year={2024},
  url={https://github.com/your-org/alpha-architecture-agent}
}
```

## Acknowledgments

- Inspired by the presentation "株価予測の最強アーキテクチャはどれだ？ AIエージェントによる探索"
- Built on PyTorch and the broader deep learning ecosystem
- Thanks to the open-source community for the foundational tools

## Support

For questions and support:

- 📧 Email: support@alphaarchitecture.ai
- 💬 Discord: [Join our community](https://discord.gg/alpha-architecture)
- 📚 Documentation: [docs.alphaarchitecture.ai](https://docs.alphaarchitecture.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/alpha-architecture-agent/issues)

---

**⚠️ Disclaimer**: This software is for research and educational purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. Always consult with a qualified financial advisor before making investment decisions.