#!/usr/bin/env python3
"""
Demo script for the comprehensive experimental validation framework.

This demonstrates how to run the AI agent architecture validation experiments
using synthetic market data generation and performance evaluation.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
    from data.synthetic_market import SyntheticMarketGenerator, MarketConfig
    from agents.architecture_agent import ArchitectureAgent
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"⚠ Import error: {e}")
    print("This is expected if dependencies are not available")


def demo_quick_validation():
    """Run a quick validation demo with minimal parameters."""
    print("=" * 70)
    print("ALPHA ARCHITECTURE AGENT - QUICK VALIDATION DEMO")
    print("=" * 70)
    
    # Create minimal configuration for demo
    config = ExperimentConfig(
        experiment_name="quick_validation_demo",
        output_dir="experiments/demo_results",
        
        # Minimal data for demo
        n_stocks=10,
        n_days=252,        # 1 year
        n_features=10,
        
        # Minimal architectures for demo
        n_architectures=5,
        input_shape=(16, 60, 10),  # smaller batch, shorter sequence
        
        # Adjusted targets for demo
        target_individual_sharpe=1.0,
        target_ensemble_sharpe=1.5
    )
    
    print(f"Configuration:")
    print(f"  Stocks: {config.n_stocks}")
    print(f"  Days: {config.n_days}")  
    print(f"  Features: {config.n_features}")
    print(f"  Architectures to test: {config.n_architectures}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    try:
        # Run individual phases for demonstration
        print(f"\n🔬 Running Phase 1: Data Generation...")
        phase1_result = runner.run_phase1_data_generation()
        print(f"   Status: {phase1_result.status}")
        
        if phase1_result.status == 'completed':
            print(f"\n🧬 Running Phase 2: Architecture Generation...")
            phase2_result = runner.run_phase2_architecture_generation()
            print(f"   Status: {phase2_result.status}")
            
            if phase2_result.status == 'completed':
                print(f"\n📊 Running Phase 3: Performance Evaluation...")
                phase3_result = runner.run_phase3_performance_evaluation()
                print(f"   Status: {phase3_result.status}")
                
                if phase3_result.status == 'completed':
                    print(f"\n🎯 Running Phase 4: Ensemble Testing...")
                    phase4_result = runner.run_phase4_ensemble_testing()
                    print(f"   Status: {phase4_result.status}")
        
        # Generate final report
        print(f"\n📋 Generating Final Report...")
        final_results = runner.generate_final_report()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"   Overall success: {final_results.get('overall_success', False)}")
        print(f"   Results saved to: {config.output_dir}")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None


def demo_data_generation_only():
    """Demo just the synthetic market data generation."""
    print("=" * 70)
    print("SYNTHETIC MARKET DATA GENERATION DEMO")
    print("=" * 70)
    
    try:
        # Create market configuration
        config = MarketConfig(
            n_stocks=20,
            n_days=504,     # 2 years
            n_features=15,
            start_date="2022-01-01"
        )
        
        print(f"Generating synthetic market data...")
        print(f"  {config.n_stocks} stocks")
        print(f"  {config.n_days} trading days")
        print(f"  {config.n_features} features per stock")
        
        # Generate data
        generator = SyntheticMarketGenerator(config)
        data = generator.generate_market_data(seed=42)
        
        print(f"\n✅ Data generation complete!")
        
        # Show basic statistics
        returns = data['returns']
        prices = data['prices']
        
        print(f"\nBasic Statistics:")
        print(f"  Returns shape: {returns.shape}")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Mean daily return: {np.mean(returns):.6f}")
        print(f"  Daily volatility: {np.std(returns):.4f}")
        print(f"  Price range: {np.min(prices):.2f} - {np.max(prices):.2f}")
        
        if 'regime_states' in data:
            regime_counts = np.bincount(data['regime_states'])
            print(f"  Regime distribution: {regime_counts}")
        
        # Show some technical indicators
        features = data['features']
        print(f"  Features shape: {features.shape}")
        print(f"  Feature range: {np.min(features):.3f} - {np.max(features):.3f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data generation demo failed: {e}")
        return None


def demo_architecture_generation_only():
    """Demo just the architecture generation system."""
    print("=" * 70) 
    print("ARCHITECTURE GENERATION DEMO")
    print("=" * 70)
    
    try:
        # Test domain blocks first
        from models.domain_blocks import get_domain_block_registry
        
        registry = get_domain_block_registry()
        print(f"Domain blocks available: {len(registry.get_all_blocks())}")
        
        categories = registry.get_categories()
        print(f"Categories: {', '.join(categories)}")
        
        for category in categories:
            blocks = registry.get_blocks_by_category(category)
            print(f"  {category}: {len(blocks)} blocks")
        
        # Test architecture generation
        print(f"\nTesting architecture generation...")
        
        try:
            agent = ArchitectureAgent()
            print("✓ LLM-based agent initialized")
        except Exception as e:
            print(f"⚠ LLM initialization failed: {e}")
            print("  Using random generation fallback")
            agent = ArchitectureAgent(generator=None)
        
        # Generate sample architectures
        input_shape = (32, 100, 15)  # batch, sequence, features
        n_architectures = 5
        
        print(f"Generating {n_architectures} architectures for input shape {input_shape}...")
        
        architectures = agent.generate_architecture_suite(
            input_shape=input_shape,
            num_architectures=n_architectures
        )
        
        print(f"\n✅ Generated {len(architectures)} architectures")
        
        # Show architecture details
        for i, arch in enumerate(architectures, 1):
            print(f"\nArchitecture {i}: {arch.name}")
            print(f"  ID: {arch.id}")
            print(f"  Blocks: {len(arch.blocks)}")
            print(f"  Complexity: {arch.complexity_score:.2f}")
            print(f"  Diversity: {arch.diversity_score:.2f}")
            
            print(f"  Block sequence:")
            for j, block_spec in enumerate(arch.blocks):
                block_name = block_spec['name']
                hyperparams = block_spec.get('hyperparameters', {})
                if hyperparams:
                    print(f"    {j+1}. {block_name} {hyperparams}")
                else:
                    print(f"    {j+1}. {block_name}")
        
        return architectures
        
    except Exception as e:
        print(f"❌ Architecture generation demo failed: {e}")
        return None


def demo_performance_evaluation():
    """Demo the performance evaluation system."""
    print("=" * 70)
    print("PERFORMANCE EVALUATION DEMO")
    print("=" * 70)
    
    try:
        # Generate simple test data
        print("Generating test market data...")
        config = MarketConfig(n_stocks=5, n_days=252, n_features=8)
        generator = SyntheticMarketGenerator(config)
        market_data = generator.generate_market_data(seed=42)
        
        # Create simple test architecture
        print("Creating test architecture...")
        test_architecture = {
            'id': 'test_arch_001',
            'name': 'simple_lstm_test',
            'blocks': [
                {'name': 'layer_norm', 'hyperparameters': {}},
                {'name': 'lstm', 'hyperparameters': {'hidden_size': 32}},
                {'name': 'regression_head', 'hyperparameters': {'output_size': 1}}
            ],
            'metadata': {'description': 'Simple LSTM for testing'}
        }
        
        # Evaluate performance
        print("Evaluating architecture performance...")
        from experiments.experiment_runner import ArchitecturePerformanceEvaluator, ExperimentConfig
        
        exp_config = ExperimentConfig(
            n_stocks=5,
            n_days=252,
            n_features=8,
            input_shape=(16, 60, 8)
        )
        
        evaluator = ArchitecturePerformanceEvaluator(exp_config)
        performance = evaluator.evaluate_architecture(test_architecture, market_data)
        
        print(f"\n✅ Performance evaluation complete!")
        
        if 'error' not in performance:
            print(f"Performance metrics:")
            print(f"  Sharpe ratio: {performance['sharpe_ratio']:.3f}")
            print(f"  Win rate: {performance['win_rate']:.3f}")
            print(f"  Max drawdown: {performance['max_drawdown']:.3f}")
            print(f"  Direction accuracy: {performance['direction_accuracy']:.3f}")
            print(f"  Mean return (annualized): {performance['mean_return']:.3f}")
            print(f"  Volatility (annualized): {performance['volatility']:.3f}")
        else:
            print(f"❌ Evaluation error: {performance['error']}")
        
        return performance
        
    except Exception as e:
        print(f"❌ Performance evaluation demo failed: {e}")
        return None


def main():
    """Main demo function with menu."""
    print("Alpha Architecture Agent - Demo Menu")
    print("=====================================")
    print()
    print("Available demos:")
    print("1. Quick validation (all phases)")
    print("2. Data generation only") 
    print("3. Architecture generation only")
    print("4. Performance evaluation only")
    print("5. Run all demos")
    print()
    
    choice = input("Select demo (1-5): ").strip()
    
    if choice == "1":
        return demo_quick_validation()
    elif choice == "2":
        return demo_data_generation_only()
    elif choice == "3":
        return demo_architecture_generation_only()
    elif choice == "4":
        return demo_performance_evaluation()
    elif choice == "5":
        print("Running all demos...\n")
        data = demo_data_generation_only()
        archs = demo_architecture_generation_only()
        perf = demo_performance_evaluation()
        validation = demo_quick_validation()
        return {'data': data, 'architectures': archs, 'performance': perf, 'validation': validation}
    else:
        print("Invalid choice. Running quick validation demo...")
        return demo_quick_validation()


if __name__ == "__main__":
    # Run with default choice if no interaction available
    print("Running data generation demo as default...")
    result = demo_data_generation_only()