#!/usr/bin/env python3
"""
Validation utilities for the experimental framework.

This module provides tools for validating the correctness and reliability
of the experimental components before running full experiments.
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic_market import SyntheticMarketGenerator, MarketConfig
from agents.architecture_agent import ArchitectureAgent
from models.domain_blocks import get_domain_block_registry
from experiments.experiment_runner import ArchitecturePerformanceEvaluator, ExperimentConfig

warnings.filterwarnings('ignore')


@dataclass
class ValidationResult:
    """Container for validation test results."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None


class ExperimentValidator:
    """Validates experimental framework components."""
    
    def __init__(self):
        self.results = []
        
    def run_all_validations(self) -> List[ValidationResult]:
        """Run comprehensive validation suite."""
        print("=" * 60)
        print("EXPERIMENTAL FRAMEWORK VALIDATION SUITE")
        print("=" * 60)
        
        self.results = []
        
        # Core component validations
        self.validate_domain_blocks()
        self.validate_synthetic_data_generation()
        self.validate_architecture_generation()
        self.validate_performance_evaluation()
        self.validate_data_pipeline()
        
        # Integration validations
        self.validate_end_to_end_pipeline()
        
        # Print summary
        self.print_validation_summary()
        
        return self.results
    
    def validate_domain_blocks(self):
        """Validate domain block system."""
        print("\n🧱 Validating Domain Blocks...")
        
        try:
            registry = get_domain_block_registry()
            
            # Check registry basics
            all_blocks = registry.get_all_blocks()
            categories = registry.get_categories()
            
            if len(all_blocks) < 10:
                self.results.append(ValidationResult(
                    "domain_blocks_count", False, 
                    f"Too few blocks registered: {len(all_blocks)}"
                ))
                return
            
            # Test each block
            test_input_shape = (32, 252, 20)
            block_test_results = {}
            
            for block in all_blocks:
                try:
                    # Test hyperparameter generation
                    hyperparams = block.get_hyperparameters()
                    
                    # Test output shape calculation
                    output_shape = block.get_output_shape(test_input_shape)
                    
                    # Test module creation (basic)
                    module = block.create_module(test_input_shape)
                    
                    block_test_results[block.name] = {
                        'hyperparams': len(hyperparams) if hyperparams else 0,
                        'output_shape': output_shape,
                        'module_created': module is not None
                    }
                    
                except Exception as e:
                    block_test_results[block.name] = {'error': str(e)}
            
            # Check results
            failed_blocks = [name for name, result in block_test_results.items() if 'error' in result]
            success_rate = (len(all_blocks) - len(failed_blocks)) / len(all_blocks)
            
            passed = success_rate >= 0.9  # 90% success rate required
            
            self.results.append(ValidationResult(
                "domain_blocks_functionality", passed,
                f"Block functionality test: {success_rate:.1%} success rate",
                {
                    'total_blocks': len(all_blocks),
                    'categories': len(categories),
                    'failed_blocks': failed_blocks,
                    'success_rate': success_rate
                }
            ))
            
            print(f"  ✓ {len(all_blocks)} blocks across {len(categories)} categories")
            print(f"  ✓ {success_rate:.1%} blocks passed functionality tests")
            if failed_blocks:
                print(f"  ⚠ Failed blocks: {', '.join(failed_blocks[:3])}{'...' if len(failed_blocks) > 3 else ''}")
                
        except Exception as e:
            self.results.append(ValidationResult(
                "domain_blocks_functionality", False, f"Domain blocks validation failed: {e}"
            ))
    
    def validate_synthetic_data_generation(self):
        """Validate synthetic market data generation."""
        print("\n📊 Validating Synthetic Data Generation...")
        
        try:
            # Test basic generation
            config = MarketConfig(n_stocks=10, n_days=252, n_features=10)
            generator = SyntheticMarketGenerator(config)
            
            data = generator.generate_market_data(seed=42)
            
            # Validate data structure
            required_keys = ['returns', 'prices', 'features', 'volatilities', 'regime_states']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                self.results.append(ValidationResult(
                    "synthetic_data_structure", False,
                    f"Missing data keys: {missing_keys}"
                ))
                return
            
            # Validate data properties
            returns = data['returns']
            prices = data['prices']
            features = data['features']
            
            # Check shapes
            expected_returns_shape = (config.n_days, config.n_stocks)
            expected_features_shape = (config.n_days, config.n_stocks, config.n_features)
            
            shape_valid = (
                returns.shape == expected_returns_shape and
                prices.shape == expected_returns_shape and
                features.shape == expected_features_shape
            )
            
            # Check statistical properties
            returns_mean = np.abs(np.mean(returns))
            returns_std = np.std(returns)
            price_ratio = np.max(prices) / np.min(prices)
            
            stats_valid = (
                returns_mean < 0.01 and  # Mean near zero
                0.05 < returns_std < 0.5 and  # Reasonable volatility
                1 < price_ratio < 100  # Reasonable price evolution
            )
            
            # Check regime balance
            regime_counts = np.bincount(data['regime_states'])
            regime_balance = 1.0 - np.std(regime_counts) / np.mean(regime_counts)
            regime_valid = regime_balance > 0.3  # Reasonable balance
            
            overall_valid = shape_valid and stats_valid and regime_valid
            
            self.results.append(ValidationResult(
                "synthetic_data_generation", overall_valid,
                f"Data generation validation: shapes={shape_valid}, stats={stats_valid}, regimes={regime_valid}",
                {
                    'returns_shape': returns.shape,
                    'features_shape': features.shape,
                    'returns_mean': returns_mean,
                    'returns_std': returns_std,
                    'price_ratio': price_ratio,
                    'regime_balance': regime_balance
                }
            ))
            
            print(f"  ✓ Generated data shape: {returns.shape}")
            print(f"  ✓ Returns statistics: mean={returns_mean:.6f}, std={returns_std:.4f}")
            print(f"  ✓ Regime balance: {regime_balance:.3f}")
            
        except Exception as e:
            self.results.append(ValidationResult(
                "synthetic_data_generation", False, f"Data generation failed: {e}"
            ))
    
    def validate_architecture_generation(self):
        """Validate architecture generation system."""
        print("\n🤖 Validating Architecture Generation...")
        
        try:
            # Test with fallback (no LLM dependency)
            agent = ArchitectureAgent(generator=None)
            
            input_shape = (32, 100, 15)
            n_architectures = 5
            
            architectures = agent.generate_architecture_suite(
                input_shape=input_shape,
                num_architectures=n_architectures
            )
            
            # Validate generation success
            generation_success = len(architectures) >= n_architectures * 0.8  # 80% success
            
            if not architectures:
                self.results.append(ValidationResult(
                    "architecture_generation", False, "No architectures generated"
                ))
                return
            
            # Validate architecture properties
            valid_architectures = 0
            total_blocks = 0
            unique_block_names = set()
            
            for arch in architectures:
                if hasattr(arch, 'blocks') and arch.blocks:
                    valid_architectures += 1
                    total_blocks += len(arch.blocks)
                    for block_spec in arch.blocks:
                        if 'name' in block_spec:
                            unique_block_names.add(block_spec['name'])
            
            diversity = len(unique_block_names) / max(total_blocks, 1)
            validity_rate = valid_architectures / len(architectures)
            
            passed = generation_success and validity_rate >= 0.8 and diversity >= 0.3
            
            self.results.append(ValidationResult(
                "architecture_generation", passed,
                f"Architecture generation: {len(architectures)} generated, {validity_rate:.1%} valid, {diversity:.2f} diversity",
                {
                    'architectures_generated': len(architectures),
                    'validity_rate': validity_rate,
                    'diversity_score': diversity,
                    'unique_blocks': len(unique_block_names),
                    'avg_blocks_per_arch': total_blocks / max(len(architectures), 1)
                }
            ))
            
            print(f"  ✓ Generated {len(architectures)} architectures")
            print(f"  ✓ Validity rate: {validity_rate:.1%}")
            print(f"  ✓ Block diversity: {diversity:.2f}")
            
        except Exception as e:
            self.results.append(ValidationResult(
                "architecture_generation", False, f"Architecture generation failed: {e}"
            ))
    
    def validate_performance_evaluation(self):
        """Validate performance evaluation system."""
        print("\n⚡ Validating Performance Evaluation...")
        
        try:
            # Create test data
            config = MarketConfig(n_stocks=5, n_days=252, n_features=8)
            generator = SyntheticMarketGenerator(config)
            market_data = generator.generate_market_data(seed=42)
            
            # Create test architecture
            test_architecture = {
                'id': 'test_validation',
                'name': 'validation_test_arch',
                'blocks': [
                    {'name': 'layer_norm', 'hyperparameters': {}},
                    {'name': 'lstm', 'hyperparameters': {'hidden_size': 16}},
                    {'name': 'regression_head', 'hyperparameters': {'output_size': 1}}
                ],
                'metadata': {}
            }
            
            # Test evaluation
            exp_config = ExperimentConfig(
                n_stocks=5, n_days=252, n_features=8,
                input_shape=(16, 60, 8)
            )
            
            evaluator = ArchitecturePerformanceEvaluator(exp_config)
            performance = evaluator.evaluate_architecture(test_architecture, market_data)
            
            # Validate results
            has_error = 'error' in performance
            has_required_metrics = all(metric in performance for metric in 
                                     ['sharpe_ratio', 'win_rate', 'max_drawdown'])
            
            if not has_error and has_required_metrics:
                # Check metric ranges
                sharpe_valid = -5 <= performance['sharpe_ratio'] <= 5
                win_rate_valid = 0 <= performance['win_rate'] <= 1
                drawdown_valid = 0 <= performance['max_drawdown'] <= 1
                
                metrics_valid = sharpe_valid and win_rate_valid and drawdown_valid
            else:
                metrics_valid = False
            
            passed = not has_error and has_required_metrics and metrics_valid
            
            self.results.append(ValidationResult(
                "performance_evaluation", passed,
                f"Performance evaluation: error={has_error}, metrics={has_required_metrics}, valid={metrics_valid}",
                {
                    'has_error': has_error,
                    'error_message': performance.get('error', None),
                    'has_required_metrics': has_required_metrics,
                    'metrics_valid': metrics_valid,
                    'sample_metrics': {k: v for k, v in performance.items() 
                                     if k in ['sharpe_ratio', 'win_rate', 'max_drawdown']}
                }
            ))
            
            if not has_error:
                print(f"  ✓ Evaluation completed successfully")
                print(f"  ✓ Sharpe ratio: {performance.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"  ✓ Win rate: {performance.get('win_rate', 'N/A'):.3f}")
            else:
                print(f"  ✗ Evaluation error: {performance.get('error', 'Unknown')}")
                
        except Exception as e:
            self.results.append(ValidationResult(
                "performance_evaluation", False, f"Performance evaluation failed: {e}"
            ))
    
    def validate_data_pipeline(self):
        """Validate data processing pipeline."""
        print("\n🔄 Validating Data Pipeline...")
        
        try:
            # Test data pipeline components
            config = MarketConfig(n_stocks=20, n_days=504, n_features=12)
            generator = SyntheticMarketGenerator(config)
            
            # Test multiple scenario generation
            scenarios = ['stable', 'volatile']
            scenario_data = {}
            
            for scenario in scenarios:
                data = generator.generate_market_data(seed=hash(scenario) % 1000)
                scenario_data[scenario] = data
            
            # Validate consistency across scenarios
            shapes_consistent = all(
                data['returns'].shape == scenario_data[scenarios[0]]['returns'].shape
                for data in scenario_data.values()
            )
            
            # Test data save/load functionality
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                generator.save_data(scenario_data['stable'], tmp.name)
                loaded_data = generator.load_data(tmp.name)
                
                # Check if loaded data matches
                save_load_works = np.allclose(
                    loaded_data['returns'], 
                    scenario_data['stable']['returns']
                )
            
            passed = shapes_consistent and save_load_works
            
            self.results.append(ValidationResult(
                "data_pipeline", passed,
                f"Data pipeline: consistency={shapes_consistent}, save_load={save_load_works}",
                {
                    'scenarios_tested': len(scenarios),
                    'shapes_consistent': shapes_consistent,
                    'save_load_works': save_load_works
                }
            ))
            
            print(f"  ✓ Tested {len(scenarios)} scenarios")
            print(f"  ✓ Shape consistency: {shapes_consistent}")
            print(f"  ✓ Save/load functionality: {save_load_works}")
            
        except Exception as e:
            self.results.append(ValidationResult(
                "data_pipeline", False, f"Data pipeline validation failed: {e}"
            ))
    
    def validate_end_to_end_pipeline(self):
        """Validate complete end-to-end pipeline."""
        print("\n🔗 Validating End-to-End Pipeline...")
        
        try:
            # Mini end-to-end test
            from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
            
            # Minimal config for testing
            config = ExperimentConfig(
                n_stocks=5,
                n_days=126,  # 6 months
                n_features=6,
                n_architectures=2,
                input_shape=(8, 30, 6)
            )
            
            runner = ExperimentRunner(config)
            
            # Test Phase 1 (Data Generation)
            try:
                phase1_result = runner.run_phase1_data_generation()
                phase1_success = phase1_result.status == 'completed'
            except Exception:
                phase1_success = False
            
            # Test Phase 2 (Architecture Generation) if Phase 1 succeeded
            phase2_success = False
            if phase1_success:
                try:
                    phase2_result = runner.run_phase2_architecture_generation()
                    phase2_success = phase2_result.status == 'completed'
                except Exception:
                    phase2_success = False
            
            passed = phase1_success and phase2_success
            
            self.results.append(ValidationResult(
                "end_to_end_pipeline", passed,
                f"End-to-end test: phase1={phase1_success}, phase2={phase2_success}",
                {
                    'phase1_success': phase1_success,
                    'phase2_success': phase2_success,
                    'full_pipeline_test': passed
                }
            ))
            
            print(f"  ✓ Phase 1 (Data): {phase1_success}")
            print(f"  ✓ Phase 2 (Architecture): {phase2_success}")
            print(f"  ✓ Pipeline integration: {passed}")
            
        except Exception as e:
            self.results.append(ValidationResult(
                "end_to_end_pipeline", False, f"End-to-end validation failed: {e}"
            ))
    
    def print_validation_summary(self):
        """Print summary of validation results."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success rate: {len(passed_tests)/len(self.results):.1%}")
        
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests:
                print(f"  - {test.test_name}: {test.message}")
        
        if passed_tests:
            print(f"\n✅ Passed tests:")
            for test in passed_tests:
                print(f"  - {test.test_name}: {test.message}")
        
        overall_success = len(failed_tests) == 0
        print(f"\n{'✅ ALL VALIDATIONS PASSED' if overall_success else '❌ SOME VALIDATIONS FAILED'}")
        
        return overall_success


def main():
    """Run validation suite."""
    validator = ExperimentValidator()
    results = validator.run_all_validations()
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()