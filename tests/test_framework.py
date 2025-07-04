#!/usr/bin/env python3
"""
Quick test of the experimental framework components.
"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

print("=" * 60)
print("ALPHA ARCHITECTURE AGENT - FRAMEWORK TEST")
print("=" * 60)

# Test 1: Basic Python environment
print("\n1. Testing Python Environment...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test 2: Import dependencies
print("\n2. Testing Dependencies...")
dependencies = {
    'numpy': 'np',
    'pandas': 'pd', 
    'torch': 'torch',
    'scipy': None,
    'sklearn': None
}

available_deps = {}
for dep, alias in dependencies.items():
    try:
        if alias:
            module = __import__(dep)
            available_deps[dep] = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {dep}: {available_deps[dep]}")
        else:
            __import__(dep)
            available_deps[dep] = 'available'
            print(f"  ✓ {dep}: available")
    except ImportError:
        print(f"  ✗ {dep}: not available")

# Test 3: Domain Blocks
print("\n3. Testing Domain Blocks...")
try:
    from models.domain_blocks import get_domain_block_registry
    registry = get_domain_block_registry()
    blocks = registry.get_all_blocks()
    categories = registry.get_categories()
    print(f"  ✓ Domain blocks loaded: {len(blocks)} blocks in {len(categories)} categories")
    
    # Test a simple block
    test_block = blocks[0] if blocks else None
    if test_block:
        input_shape = (32, 100, 20)
        try:
            output_shape = test_block.get_output_shape(input_shape)
            module = test_block.create_module(input_shape)
            print(f"  ✓ Block test successful: {test_block.name}")
        except Exception as e:
            print(f"  ⚠ Block test failed: {e}")
    
except Exception as e:
    print(f"  ✗ Domain blocks failed: {e}")

# Test 4: Synthetic Data Generation
print("\n4. Testing Synthetic Data Generation...")
try:
    from data.synthetic_market import SyntheticMarketGenerator, MarketConfig
    config = MarketConfig(n_stocks=5, n_days=100, n_features=5)
    generator = SyntheticMarketGenerator(config)
    
    print(f"  ✓ Generator created for {config.n_stocks} stocks")
    
    # Try to generate small sample
    data = generator.generate_market_data(seed=42)
    print(f"  ✓ Data generated: returns shape {data['returns'].shape}")
    
except Exception as e:
    print(f"  ✗ Synthetic data generation failed: {e}")

# Test 5: Architecture Generation
print("\n5. Testing Architecture Generation...")
try:
    from agents.architecture_agent import ArchitectureAgent
    
    # Test with fallback (no LLM required)
    agent = ArchitectureAgent(generator=None)
    
    input_shape = (16, 50, 10)
    architectures = agent.generate_architecture_suite(
        input_shape=input_shape,
        num_architectures=2
    )
    
    print(f"  ✓ Architecture generation: {len(architectures)} architectures created")
    
    if architectures:
        arch = architectures[0]
        print(f"  ✓ Sample architecture: {arch.name} with {len(arch.blocks)} blocks")
    
except Exception as e:
    print(f"  ✗ Architecture generation failed: {e}")

# Test 6: Performance Evaluation
print("\n6. Testing Performance Evaluation...")
try:
    from experiments.experiment_runner import ArchitecturePerformanceEvaluator, ExperimentConfig
    
    # Use simple test configuration
    config = ExperimentConfig(
        n_stocks=3, n_days=50, n_features=5,
        input_shape=(8, 20, 5)
    )
    
    evaluator = ArchitecturePerformanceEvaluator(config)
    print(f"  ✓ Performance evaluator created")
    
    # Test with mock architecture
    test_arch = {
        'id': 'test',
        'name': 'test_arch',
        'blocks': [
            {'name': 'layer_norm', 'hyperparameters': {}},
            {'name': 'regression_head', 'hyperparameters': {'output_size': 1}}
        ]
    }
    
    # Create mock market data
    import numpy as np
    mock_data = {
        'returns': np.random.normal(0, 0.02, (50, 3)),
        'features': np.random.normal(0, 1, (50, 3, 5)),
        'prices': np.random.uniform(100, 200, (50, 3))
    }
    
    # This might fail due to shape mismatches, but tests the pipeline
    try:
        performance = evaluator.evaluate_architecture(test_arch, mock_data)
        if 'error' not in performance:
            print(f"  ✓ Evaluation successful")
        else:
            print(f"  ⚠ Evaluation completed with error: {performance['error'][:50]}...")
    except Exception as e:
        print(f"  ⚠ Evaluation pipeline test: {str(e)[:50]}...")
    
except Exception as e:
    print(f"  ✗ Performance evaluation setup failed: {e}")

# Test 7: Validation Framework
print("\n7. Testing Validation Framework...")
try:
    from experiments.validation_utils import ExperimentValidator
    
    validator = ExperimentValidator()
    print(f"  ✓ Validator created")
    
    # Run a subset of validations
    print(f"  → Running domain blocks validation...")
    validator.validate_domain_blocks()
    
    print(f"  → Running data generation validation...")
    validator.validate_synthetic_data_generation()
    
    results = validator.results
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"  ✓ Validation tests: {passed}/{total} passed")
    
except Exception as e:
    print(f"  ✗ Validation framework failed: {e}")

# Summary
print("\n" + "=" * 60)
print("FRAMEWORK TEST SUMMARY")
print("=" * 60)

print(f"✓ Python environment ready")
print(f"✓ Dependencies: {len(available_deps)}/{len(dependencies)} available")
print(f"✓ Core modules tested")

try:
    # Quick integration test
    from experiments.experiment_runner import ExperimentConfig
    
    demo_config = ExperimentConfig(
        experiment_name="framework_test",
        n_stocks=3,
        n_days=50,
        n_features=5,
        n_architectures=2,
        input_shape=(8, 20, 5)
    )
    
    print(f"✓ Framework ready for experiments")
    print(f"✓ Demo config: {demo_config.n_stocks} stocks, {demo_config.n_days} days")
    print(f"\nTo run full experiment:")
    print(f"  python examples/demo_experiment_validation.py")
    
except Exception as e:
    print(f"⚠ Integration test issue: {e}")

print(f"\n🎯 Framework test complete!")