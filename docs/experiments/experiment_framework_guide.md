# AI Agent Architecture Validation Framework

## 🎯 Overview

This experimental framework validates the effectiveness of our AI Agent-based neural network architecture generation system for stock prediction. The framework implements a comprehensive 4-phase validation methodology using synthetic Japanese market data.

## 📁 Framework Components

### Core Modules

```
src/
├── experiments/
│   └── experiment_runner.py          # Main experimental framework
├── data/
│   └── synthetic_market.py            # Synthetic market data generation
├── agents/
│   └── architecture_agent.py          # AI agent for architecture generation
└── models/
    ├── domain_blocks.py               # Basic domain blocks
    └── domain_blocks_extended.py      # Extended block implementations
```

### Demo & Examples

```
examples/
├── demo_experiment_validation.py     # Interactive demo of framework
├── demo_architecture_generation.py   # Architecture generation demo
└── demo_synthetic_data.py            # Data generation demo
```

## 🔬 Experimental Phases

### Phase 1: Data Generation & Validation
- **Objective**: Create realistic synthetic Japanese market data
- **Components**: Multi-factor models, regime switching, GARCH volatility, jump diffusion
- **Validation**: Statistical properties, regime balance, price evolution
- **Output**: Validated market scenarios for testing

### Phase 2: Architecture Generation Testing  
- **Objective**: Validate AI agent's ability to generate diverse, viable architectures
- **Metrics**: Generation success rate (>90%), diversity scores, block compatibility
- **Output**: Suite of neural network architectures for evaluation

### Phase 3: Prediction Performance Evaluation
- **Objective**: Measure prediction performance of generated architectures
- **Metrics**: Sharpe ratio, win rate, max drawdown, direction accuracy
- **Target**: Individual strategies achieving Sharpe ratio >1.3

### Phase 4: Ensemble Strategy Testing
- **Objective**: Validate ensemble effectiveness for improved performance
- **Strategies**: Equal weight, Sharpe-weighted, diversity-weighted ensembles
- **Target**: Ensemble strategies achieving Sharpe ratio >2.0

## 🚀 Quick Start

### Basic Usage

```python
from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig

# Create experiment configuration
config = ExperimentConfig(
    experiment_name="my_validation",
    n_stocks=100,
    n_days=2016,        # 8 years
    n_architectures=50,
    target_individual_sharpe=1.3,
    target_ensemble_sharpe=2.0
)

# Run complete validation
runner = ExperimentRunner(config)
results = runner.run_full_experiment()

print(f"Overall success: {results['overall_success']}")
```

### Interactive Demo

```bash
python examples/demo_experiment_validation.py
```

Choose from:
1. Quick validation (all phases)
2. Data generation only
3. Architecture generation only  
4. Performance evaluation only
5. Run all demos

## ⚙️ Configuration Options

### ExperimentConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_stocks` | 100 | Number of stocks in synthetic market |
| `n_days` | 2016 | Trading days (8 years) |
| `n_features` | 20 | Features per stock |
| `n_architectures` | 50 | Architectures to generate and test |
| `input_shape` | (32, 252, 20) | Neural network input shape |
| `target_individual_sharpe` | 1.3 | Individual strategy target |
| `target_ensemble_sharpe` | 2.0 | Ensemble strategy target |

### Market Scenarios

```python
from src.data.synthetic_market import create_market_scenarios

scenarios = create_market_scenarios()
# Available: 'stable', 'volatile', 'trending', 'range_bound'
```

## 📊 Results & Metrics

### Success Criteria

The framework validates against these criteria:

✅ **Data Quality**: >80% quality score for generated market data  
✅ **Architecture Generation**: >90% success rate  
✅ **Individual Performance**: Best individual Sharpe ratio >1.3  
✅ **Ensemble Performance**: Best ensemble Sharpe ratio >2.0  
✅ **Risk Control**: Maximum drawdown <10%  

### Output Files

```
experiments/results/
├── final_experiment_report.json      # Complete results summary
├── phase1_market_data.npz            # Generated market data
├── phase2_architectures.json         # Generated architectures
├── phase3_performance.json           # Individual performance results
└── phase4_ensemble.json              # Ensemble strategy results
```

### Key Metrics

**Performance Metrics**:
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (worst loss period)
- Win Rate (percentage of profitable predictions)
- Direction Accuracy (up/down prediction accuracy)

**Architecture Metrics**:
- Generation Success Rate
- Diversity Score (block usage diversity)
- Complexity Score (architecture complexity)
- Compilation Success Rate

## 🏗️ Synthetic Market Data

### Multi-Layer Simulation

Our synthetic market includes:

1. **Factor Model**: 5-factor model (Market, Size, Value, Momentum, Quality)
2. **Regime Switching**: Bull/Bear/Sideways market states
3. **Volatility Modeling**: GARCH(1,1) with regime dependence
4. **Jump Diffusion**: Extreme event simulation
5. **Cross-Sectional Correlation**: Industry and market-wide correlations

### Realistic Properties

- **Japanese Market Characteristics**: Trading calendar, price ranges, volatility
- **Statistical Properties**: Fat tails, volatility clustering, mean reversion
- **Technical Indicators**: 20 features including moving averages, RSI, momentum
- **Regime Persistence**: Realistic market state durations

## 🤖 AI Architecture Agent

### Domain Block Categories

1. **Normalization** (8 blocks): BatchNorm, LayerNorm, RMSNorm, etc.
2. **Feature Extraction** (6 blocks): PCA, Fourier, Wavelet, etc.
3. **Mixing** (5 blocks): Time, Channel, Cross-Attention mixing
4. **Encoding** (6 blocks): Linear, MLP, Convolutional, Transformer
5. **Financial Domain** (6 blocks): Multi-timeframe, Lead-lag, Regime detection
6. **Prediction Heads** (5 blocks): Regression, Classification, Ranking

### Architecture Generation

- **LLM-Powered**: Uses GPT-4/Claude for intelligent combinations
- **Fallback System**: Random generation when LLM unavailable
- **Diversity Optimization**: Ensures varied architectural patterns
- **Compatibility Checking**: Validates shape compatibility

## 📈 Expected Results

Based on our research methodology, we expect:

### Individual Strategies
- **Top Performance**: Sharpe ratio 1.3-1.5
- **Average Performance**: Sharpe ratio 0.8-1.2
- **Success Rate**: 60-70% of strategies profitable
- **Drawdown Control**: <15% maximum drawdown

### Ensemble Strategies
- **Performance Boost**: 30-50% improvement over individuals
- **Target Achievement**: Sharpe ratio 2.0-2.5
- **Risk Reduction**: Lower drawdowns and volatility
- **Consistency**: More stable performance across market regimes

## 🔧 Advanced Usage

### Custom Market Scenarios

```python
from src.data.synthetic_market import MarketConfig, SyntheticMarketGenerator

# Create custom scenario
custom_config = MarketConfig(
    n_stocks=200,
    n_days=3024,  # 12 years
    n_features=30,
    factor_names=['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Profitability']
)

generator = SyntheticMarketGenerator(custom_config)
data = generator.generate_market_data(seed=123)
```

### Custom Architecture Constraints

```python
from src.agents.architecture_agent import ArchitectureAgent

agent = ArchitectureAgent()

# Generate with constraints
architectures = agent.generate_architecture_suite(
    input_shape=(64, 504, 25),
    num_architectures=100,
    max_blocks=10,
    required_categories=['financial_domain', 'sequence_models']
)
```

### Performance Analysis

```python
from src.experiments.experiment_runner import ArchitecturePerformanceEvaluator

evaluator = ArchitecturePerformanceEvaluator(config)

# Detailed evaluation
performance = evaluator.evaluate_architecture(architecture_spec, market_data)
print(f"Sharpe: {performance['sharpe_ratio']:.3f}")
print(f"Win Rate: {performance['win_rate']:.3f}")
```

## 🧪 Validation Methodology

### Statistical Validation
- **Bootstrap Resampling**: Confidence intervals for performance metrics
- **Cross-Validation**: Time-series aware validation splits
- **Regime Analysis**: Performance across different market conditions

### Robustness Testing
- **Noise Sensitivity**: Performance under different noise levels
- **Parameter Stability**: Sensitivity to hyperparameter changes
- **Out-of-Sample**: Forward validation on unseen data

### Comparative Analysis
- **Baseline Comparison**: Against simple strategies (buy-hold, momentum)
- **Benchmark Models**: LSTM, Transformer, traditional ML
- **Ensemble vs Individual**: Quantify ensemble benefits

## 📝 Research Applications

This framework enables research into:

- **Architecture Search**: Optimal neural network designs for finance
- **Domain Knowledge**: Effectiveness of financial domain blocks  
- **Ensemble Methods**: Advanced combination strategies
- **Market Regime Adaptation**: Architecture performance across regimes
- **Risk-Return Optimization**: Multi-objective architecture selection

## 🔍 Troubleshooting

### Common Issues

**Python Environment**: Framework requires Python 3.8+ with PyTorch
```bash
# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt
```

**Memory Issues**: Reduce experiment scale for limited memory
```python
config = ExperimentConfig(
    n_stocks=50,     # Reduce from 100
    n_days=1008,     # Reduce from 2016
    n_architectures=20  # Reduce from 50
)
```

**LLM API Issues**: Framework works with random generation fallback
```python
# Verify LLM initialization
from src.agents.architecture_agent import ArchitectureAgent
agent = ArchitectureAgent()  # Will use fallback if API unavailable
```

### Performance Optimization

**GPU Acceleration**: Ensure CUDA available for training
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Parallel Evaluation**: Evaluate multiple architectures concurrently
```python
# Enable parallel evaluation in future versions
config.parallel_evaluation = True
config.num_workers = 4
```

## 📚 References

1. **Architecture Search**: Neural Architecture Search for Financial Time Series
2. **Synthetic Data**: Realistic Market Simulation for Algorithmic Trading
3. **Ensemble Methods**: Deep Learning Ensembles for Financial Prediction
4. **Domain Knowledge**: Financial Domain Blocks for Neural Networks

## 🤝 Contributing

To extend the framework:

1. **Add Domain Blocks**: Implement new blocks in `src/models/blocks/`
2. **Create Market Models**: Add scenarios in `src/data/synthetic_market.py`
3. **Enhance Evaluation**: Extend metrics in `experiment_runner.py`
4. **Add Visualizations**: Create analysis tools in `examples/`

## 📄 License

This framework is part of the Alpha Architecture Agent research project.