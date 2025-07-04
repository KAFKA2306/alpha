#!/usr/bin/env python3
"""
Minimal test of the Alpha Architecture Agent framework.
Tests core functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path
import traceback

# Setup paths
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """Test if we can import our modules."""
    print("Testing module imports...")
    
    try:
        # Test basic scientific libraries
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        # Test domain blocks
        from models.domain_blocks import get_domain_block_registry
        registry = get_domain_block_registry()
        print(f"✓ Domain blocks: {len(registry.get_all_blocks())} blocks")
    except Exception as e:
        print(f"✗ Domain blocks failed: {e}")
        return False
    
    try:
        # Test synthetic data
        from data.synthetic_market import MarketConfig
        config = MarketConfig(n_stocks=2, n_days=10, n_features=3)
        print(f"✓ Market config: {config.n_stocks} stocks")
    except Exception as e:
        print(f"✗ Market config failed: {e}")
        return False
    
    return True

def test_domain_blocks():
    """Test domain block functionality."""
    print("\nTesting domain blocks...")
    
    try:
        from models.domain_blocks import get_domain_block_registry
        
        registry = get_domain_block_registry()
        blocks = registry.get_all_blocks()
        
        if not blocks:
            print("✗ No blocks available")
            return False
        
        # Test first block
        test_block = blocks[0]
        input_shape = (8, 20, 10)
        
        # Test output shape calculation
        output_shape = test_block.get_output_shape(input_shape)
        print(f"✓ Block '{test_block.name}': {input_shape} → {output_shape}")
        
        # Test hyperparameters
        hyperparams = test_block.get_hyperparameters()
        print(f"✓ Hyperparameters: {len(hyperparams) if hyperparams else 0} params")
        
        return True
        
    except Exception as e:
        print(f"✗ Domain block test failed: {e}")
        traceback.print_exc()
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    
    try:
        from data.synthetic_market import SyntheticMarketGenerator, MarketConfig
        
        # Minimal configuration
        config = MarketConfig(
            n_stocks=3,
            n_days=20,
            n_features=5
        )
        
        generator = SyntheticMarketGenerator(config)
        data = generator.generate_market_data(seed=42)
        
        # Check data structure
        required_keys = ['returns', 'prices', 'features']
        for key in required_keys:
            if key not in data:
                print(f"✗ Missing data key: {key}")
                return False
        
        returns_shape = data['returns'].shape
        features_shape = data['features'].shape
        
        print(f"✓ Returns shape: {returns_shape}")
        print(f"✓ Features shape: {features_shape}")
        
        # Basic validation
        if returns_shape != (config.n_days, config.n_stocks):
            print(f"✗ Wrong returns shape")
            return False
        
        if features_shape != (config.n_days, config.n_stocks, config.n_features):
            print(f"✗ Wrong features shape")
            return False
        
        print("✓ Data validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Synthetic data test failed: {e}")
        traceback.print_exc()
        return False

def test_architecture_generation():
    """Test architecture generation."""
    print("\nTesting architecture generation...")
    
    try:
        from agents.architecture_agent import ArchitectureAgent
        
        # Use fallback generator (no LLM required)
        agent = ArchitectureAgent(generator=None)
        
        input_shape = (16, 30, 8)
        architectures = agent.generate_architecture_suite(
            input_shape=input_shape,
            num_architectures=2
        )
        
        if not architectures:
            print("✗ No architectures generated")
            return False
        
        arch = architectures[0]
        print(f"✓ Generated architecture: '{arch.name}'")
        print(f"✓ Blocks: {len(arch.blocks)}")
        print(f"✓ Complexity: {arch.complexity_score:.2f}")
        
        # Check architecture structure
        if not hasattr(arch, 'blocks') or not arch.blocks:
            print("✗ Architecture has no blocks")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture generation failed: {e}")
        traceback.print_exc()
        return False

def test_experiment_config():
    """Test experiment configuration."""
    print("\nTesting experiment configuration...")
    
    try:
        from experiments.experiment_runner import ExperimentConfig
        
        config = ExperimentConfig(
            experiment_name="minimal_test",
            n_stocks=5,
            n_days=50,
            n_features=8,
            n_architectures=3
        )
        
        print(f"✓ Experiment config created")
        print(f"✓ Name: {config.experiment_name}")
        print(f"✓ Stocks: {config.n_stocks}")
        print(f"✓ Target Sharpe: {config.target_individual_sharpe}")
        
        return True
        
    except Exception as e:
        print(f"✗ Experiment config failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run minimal test suite."""
    print("=" * 60)
    print("ALPHA ARCHITECTURE AGENT - MINIMAL TEST")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Domain Blocks", test_domain_blocks),
        ("Synthetic Data", test_synthetic_data), 
        ("Architecture Generation", test_architecture_generation),
        ("Experiment Config", test_experiment_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"TEST: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<25} [{status}]")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Framework is ready.")
        print("\nNext steps:")
        print("1. Run full validation: python src/experiments/validation_utils.py")
        print("2. Run demo: python examples/demo_experiment_validation.py")
        print("3. Run experiments: from experiments.experiment_runner import ExperimentRunner")
    else:
        print(f"\n⚠ {total - passed} tests failed. Check dependencies and configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)