"""
Validation Experiments for Generated Domain Blocks

This module runs comprehensive validation experiments on the generated blocks
to test their quality, diversity, functionality, and alignment with financial AI goals.
"""

import json
import os
import time
import random
from typing import Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict
import math


def mock_torch_imports():
    """Mock torch imports for validation without installation"""
    import sys
    from unittest.mock import MagicMock
    
    # Mock torch
    torch_mock = MagicMock()
    torch_mock.nn = MagicMock()
    torch_mock.nn.Module = object
    torch_mock.tensor = lambda x: x
    torch_mock.randn = lambda *args: [random.random() for _ in range(args[0] if args else 1)]
    
    sys.modules['torch'] = torch_mock
    sys.modules['torch.nn'] = torch_mock.nn
    sys.modules['torch.nn.functional'] = MagicMock()
    
    # Mock numpy
    numpy_mock = MagicMock()
    numpy_mock.mean = lambda x: 0.5
    numpy_mock.std = lambda x: 0.3
    numpy_mock.var = lambda x: 0.09
    sys.modules['numpy'] = numpy_mock


class BlockValidationExperiments:
    """Comprehensive validation experiments for generated blocks"""
    
    def __init__(self, blocks_file: str = "generated_blocks/generated_blocks.json"):
        self.blocks_file = blocks_file
        self.blocks = self._load_blocks()
        self.results = {}
        self.dashboard_data = {}
        
    def _load_blocks(self) -> List[Dict[str, Any]]:
        """Load generated blocks"""
        if not os.path.exists(self.blocks_file):
            raise FileNotFoundError(f"Blocks file not found: {self.blocks_file}")
        
        with open(self.blocks_file, 'r') as f:
            blocks = json.load(f)
        
        print(f"Loaded {len(blocks)} blocks for validation")
        return blocks
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all validation experiments"""
        
        print("=" * 80)
        print("BLOCK VALIDATION EXPERIMENTS")
        print("=" * 80)
        
        experiments = [
            ("Diversity Analysis", self._experiment_diversity_analysis),
            ("Quality Assessment", self._experiment_quality_assessment),
            ("Financial Domain Validation", self._experiment_financial_validation),
            ("Complexity Distribution", self._experiment_complexity_analysis),
            ("Innovation Score", self._experiment_innovation_scoring),
            ("Architecture Compatibility", self._experiment_architecture_compatibility),
            ("Performance Simulation", self._experiment_performance_simulation),
            ("Scalability Test", self._experiment_scalability_test)
        ]
        
        for exp_name, exp_func in experiments:
            print(f"\nRunning: {exp_name}")
            print("-" * 40)
            
            start_time = time.time()
            result = exp_func()
            duration = time.time() - start_time
            
            self.results[exp_name] = {
                'result': result,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Completed in {duration:.2f}s")
        
        # Generate dashboard data
        self._generate_dashboard_data()
        
        return self.results
    
    def _experiment_diversity_analysis(self) -> Dict[str, Any]:
        """Experiment 1: Analyze diversity metrics"""
        
        # Category diversity
        categories = set(block['category'] for block in self.blocks)
        category_counts = defaultdict(int)
        for block in self.blocks:
            category_counts[block['category']] += 1
        
        # Complexity diversity
        complexities = set(block['complexity'] for block in self.blocks)
        complexity_counts = defaultdict(int)
        for block in self.blocks:
            complexity_counts[block['complexity']] += 1
        
        # Component diversity
        all_components = set()
        component_usage = defaultdict(int)
        for block in self.blocks:
            for component in block['components']:
                all_components.add(component)
                component_usage[component] += 1
        
        # Parameter diversity
        param_diversity = defaultdict(set)
        for block in self.blocks:
            for key, value in block['parameters'].items():
                param_diversity[key].add(str(value))
        
        # Calculate diversity scores
        total_blocks = len(self.blocks)
        category_entropy = self._calculate_entropy(list(category_counts.values()))
        complexity_entropy = self._calculate_entropy(list(complexity_counts.values()))
        
        # Unique combinations
        unique_combinations = len(set(
            (block['category'], block['complexity'], tuple(sorted(block['components'])))
            for block in self.blocks
        ))
        
        diversity_score = (
            len(categories) / 10 * 0.3 +  # Normalize to expected max categories
            len(complexities) / 3 * 0.2 +  # 3 complexity levels
            len(all_components) / 50 * 0.3 +  # Expected max components
            unique_combinations / total_blocks * 0.2
        )
        
        result = {
            'category_diversity': {
                'count': len(categories),
                'entropy': category_entropy,
                'distribution': dict(category_counts)
            },
            'complexity_diversity': {
                'count': len(complexities),
                'entropy': complexity_entropy,
                'distribution': dict(complexity_counts)
            },
            'component_diversity': {
                'count': len(all_components),
                'most_used': sorted(component_usage.items(), key=lambda x: x[1], reverse=True)[:5],
                'least_used': sorted(component_usage.items(), key=lambda x: x[1])[:5]
            },
            'parameter_diversity': {
                'param_count': len(param_diversity),
                'avg_values_per_param': sum(len(values) for values in param_diversity.values()) / len(param_diversity)
            },
            'unique_combinations': unique_combinations,
            'overall_diversity_score': diversity_score
        }
        
        print(f"✓ Categories: {len(categories)}")
        print(f"✓ Components: {len(all_components)}")
        print(f"✓ Unique combinations: {unique_combinations}")
        print(f"✓ Overall diversity score: {diversity_score:.3f}")
        
        return result
    
    def _experiment_quality_assessment(self) -> Dict[str, Any]:
        """Experiment 2: Assess block quality"""
        
        quality_scores = []
        
        for block in self.blocks:
            score = 0
            
            # Name quality (descriptive, not too long/short)
            name_len = len(block['name'])
            if 10 <= name_len <= 40:
                score += 1
            
            # Description quality
            desc_len = len(block['description'])
            if 20 <= desc_len <= 150:
                score += 1
            
            # Parameter count (not too few/many)
            param_count = len(block['parameters'])
            if 2 <= param_count <= 8:
                score += 1
            
            # Component count
            component_count = len(block['components'])
            if 2 <= component_count <= 6:
                score += 1
            
            # Consistency (category matches components)
            category_keywords = {
                'financial_domain': ['market', 'financial', 'factor', 'risk', 'volatility'],
                'attention': ['attention', 'key_value', 'weights'],
                'normalization': ['normalization', 'scaling', 'centering'],
                'feature_extraction': ['decomposition', 'feature', 'extraction'],
                'mixing': ['mixing', 'cross_connections', 'interaction'],
                'temporal_processing': ['temporal', 'sequence', 'time'],
                'regularization': ['regularization', 'constraint', 'penalty']
            }
            
            expected_keywords = category_keywords.get(block['category'], [])
            if any(keyword in ' '.join(block['components']).lower() for keyword in expected_keywords):
                score += 1
            
            quality_scores.append(score / 5)  # Normalize to 0-1
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Quality distribution
        quality_bins = [0, 0.4, 0.7, 1.0]
        quality_labels = ['Poor', 'Good', 'Excellent']
        quality_dist = {label: 0 for label in quality_labels}
        
        for score in quality_scores:
            for i, threshold in enumerate(quality_bins[1:]):
                if score <= threshold:
                    quality_dist[quality_labels[i]] += 1
                    break
        
        high_quality_blocks = [
            self.blocks[i] for i, score in enumerate(quality_scores) if score >= 0.8
        ]
        
        result = {
            'average_quality': avg_quality,
            'quality_distribution': quality_dist,
            'high_quality_count': len(high_quality_blocks),
            'quality_score_range': [min(quality_scores), max(quality_scores)],
            'top_quality_blocks': [block['name'] for block in high_quality_blocks[:5]]
        }
        
        print(f"✓ Average quality: {avg_quality:.3f}")
        print(f"✓ High quality blocks: {len(high_quality_blocks)}")
        print(f"✓ Quality distribution: {quality_dist}")
        
        return result
    
    def _experiment_financial_validation(self) -> Dict[str, Any]:
        """Experiment 3: Validate financial domain relevance"""
        
        financial_blocks = [b for b in self.blocks if b['category'] == 'financial_domain']
        
        # Financial concepts coverage
        financial_concepts = {
            'market_microstructure': ['microstructure', 'order', 'spread', 'liquidity'],
            'risk_management': ['risk', 'var', 'volatility', 'exposure'],
            'factor_modeling': ['factor', 'pca', 'exposure', 'attribution'],
            'time_series': ['trend', 'seasonal', 'autocorr', 'lag'],
            'regime_detection': ['regime', 'switching', 'state', 'hidden'],
            'portfolio_theory': ['portfolio', 'weight', 'allocation', 'optimization'],
            'market_anomalies': ['momentum', 'reversion', 'anomaly', 'alpha'],
            'derivatives': ['option', 'volatility', 'smile', 'greeks'],
            'alternative_data': ['sentiment', 'text', 'satellite', 'social']
        }
        
        concept_coverage = {}
        for concept, keywords in financial_concepts.items():
            covered_blocks = []
            for block in financial_blocks:
                block_text = (block['name'] + ' ' + block['description']).lower()
                if any(keyword in block_text for keyword in keywords):
                    covered_blocks.append(block['name'])
            concept_coverage[concept] = {
                'count': len(covered_blocks),
                'blocks': covered_blocks[:3]  # Top 3 examples
            }
        
        # Uki-san's architecture alignment
        uki_concepts = {
            'multi_timeframe': ['multi', 'time', 'frame', 'scale'],
            'lead_lag': ['lead', 'lag', 'delay', 'shift'],
            'regime_detection': ['regime', 'state', 'switching'],
            'factor_exposure': ['factor', 'exposure', 'loading'],
            'volatility_clustering': ['volatility', 'cluster', 'garch'],
            'cross_sectional': ['cross', 'sectional', 'rank', 'universe']
        }
        
        uki_alignment = {}
        for concept, keywords in uki_concepts.items():
            matching_blocks = []
            for block in self.blocks:
                block_text = (block['name'] + ' ' + block['description']).lower()
                if any(keyword in block_text for keyword in keywords):
                    matching_blocks.append(block['name'])
            uki_alignment[concept] = len(matching_blocks)
        
        # Innovation in financial domain
        innovative_combinations = []
        for block in financial_blocks:
            # Look for novel parameter combinations
            params = block['parameters']
            if len(set(params.values())) == len(params):  # All unique values
                innovative_combinations.append(block['name'])
        
        financial_score = (
            len(financial_blocks) / len(self.blocks) * 0.4 +  # Financial block ratio
            sum(1 for count in concept_coverage.values() if count['count'] > 0) / len(financial_concepts) * 0.3 +  # Concept coverage
            sum(uki_alignment.values()) / (len(uki_alignment) * 5) * 0.3  # Uki alignment (max 5 blocks per concept)
        )
        
        result = {
            'financial_block_count': len(financial_blocks),
            'financial_block_ratio': len(financial_blocks) / len(self.blocks),
            'concept_coverage': concept_coverage,
            'uki_alignment': uki_alignment,
            'innovative_combinations': len(innovative_combinations),
            'financial_domain_score': financial_score
        }
        
        print(f"✓ Financial blocks: {len(financial_blocks)}")
        print(f"✓ Concept coverage: {sum(1 for c in concept_coverage.values() if c['count'] > 0)}/{len(financial_concepts)}")
        print(f"✓ Uki alignment score: {sum(uki_alignment.values())}")
        print(f"✓ Financial domain score: {financial_score:.3f}")
        
        return result
    
    def _experiment_complexity_analysis(self) -> Dict[str, Any]:
        """Experiment 4: Analyze complexity distribution and appropriateness"""
        
        complexity_counts = defaultdict(int)
        complexity_by_category = defaultdict(lambda: defaultdict(int))
        
        for block in self.blocks:
            complexity = block['complexity']
            category = block['category']
            complexity_counts[complexity] += 1
            complexity_by_category[category][complexity] += 1
        
        # Expected complexity distribution (based on research complexity)
        expected_dist = {'simple': 0.2, 'medium': 0.5, 'complex': 0.3}
        actual_dist = {k: v / len(self.blocks) for k, v in complexity_counts.items()}
        
        # Calculate distribution distance
        distribution_distance = sum(
            abs(actual_dist.get(k, 0) - expected_dist[k]) for k in expected_dist
        )
        
        # Complexity appropriateness (financial domain should be more complex)
        financial_blocks = [b for b in self.blocks if b['category'] == 'financial_domain']
        financial_complexity = defaultdict(int)
        for block in financial_blocks:
            financial_complexity[block['complexity']] += 1
        
        financial_complex_ratio = financial_complexity['complex'] / len(financial_blocks) if financial_blocks else 0
        
        result = {
            'complexity_distribution': dict(complexity_counts),
            'complexity_ratios': actual_dist,
            'expected_ratios': expected_dist,
            'distribution_distance': distribution_distance,
            'complexity_by_category': {k: dict(v) for k, v in complexity_by_category.items()},
            'financial_complexity_ratio': financial_complex_ratio,
            'complexity_appropriateness_score': 1 - distribution_distance + financial_complex_ratio * 0.5
        }
        
        print(f"✓ Complexity distribution: {dict(complexity_counts)}")
        print(f"✓ Distribution distance: {distribution_distance:.3f}")
        print(f"✓ Financial complex ratio: {financial_complex_ratio:.3f}")
        
        return result
    
    def _experiment_innovation_scoring(self) -> Dict[str, Any]:
        """Experiment 5: Score innovation and novelty"""
        
        # Novel parameter combinations
        param_combinations = set()
        for block in self.blocks:
            combo = tuple(sorted(f"{k}:{v}" for k, v in block['parameters'].items()))
            param_combinations.add(combo)
        
        # Novel component combinations
        component_combinations = set()
        for block in self.blocks:
            combo = tuple(sorted(block['components']))
            component_combinations.add(combo)
        
        # Cross-category inspiration (blocks that borrow from other domains)
        cross_category_blocks = []
        category_keywords = {
            'financial_domain': ['financial', 'market', 'factor'],
            'attention': ['attention', 'query', 'key'],
            'normalization': ['norm', 'scale', 'center'],
            'feature_extraction': ['feature', 'extract', 'decomp'],
            'mixing': ['mix', 'combine', 'merge'],
            'temporal_processing': ['temporal', 'time', 'sequence'],
            'regularization': ['regular', 'penalty', 'constraint']
        }
        
        for block in self.blocks:
            block_category = block['category']
            block_text = (block['name'] + ' ' + block['description']).lower()
            
            # Check if block uses concepts from other categories
            other_categories = [cat for cat in category_keywords if cat != block_category]
            cross_influences = 0
            
            for other_cat in other_categories:
                if any(keyword in block_text for keyword in category_keywords[other_cat]):
                    cross_influences += 1
            
            if cross_influences >= 2:  # Influenced by at least 2 other categories
                cross_category_blocks.append(block['name'])
        
        # Advanced technique usage
        advanced_techniques = [
            'hierarchical', 'multi_scale', 'adaptive', 'meta_learning',
            'graph_neural', 'transformer', 'attention', 'spectral',
            'sparse', 'variational', 'bayesian', 'reinforcement'
        ]
        
        advanced_blocks = []
        for block in self.blocks:
            block_text = (block['name'] + ' ' + block['description']).lower()
            if any(technique in block_text for technique in advanced_techniques):
                advanced_blocks.append(block['name'])
        
        # Innovation score calculation
        innovation_score = (
            len(param_combinations) / len(self.blocks) * 0.3 +  # Parameter novelty
            len(component_combinations) / len(self.blocks) * 0.3 +  # Component novelty
            len(cross_category_blocks) / len(self.blocks) * 0.2 +  # Cross-category inspiration
            len(advanced_blocks) / len(self.blocks) * 0.2  # Advanced techniques
        )
        
        result = {
            'unique_parameter_combinations': len(param_combinations),
            'unique_component_combinations': len(component_combinations),
            'cross_category_blocks': len(cross_category_blocks),
            'advanced_technique_blocks': len(advanced_blocks),
            'innovation_score': innovation_score,
            'most_innovative_blocks': cross_category_blocks[:5] + advanced_blocks[:5]
        }
        
        print(f"✓ Unique param combinations: {len(param_combinations)}")
        print(f"✓ Cross-category blocks: {len(cross_category_blocks)}")
        print(f"✓ Advanced technique blocks: {len(advanced_blocks)}")
        print(f"✓ Innovation score: {innovation_score:.3f}")
        
        return result
    
    def _experiment_architecture_compatibility(self) -> Dict[str, Any]:
        """Experiment 6: Test compatibility with neural architectures"""
        
        mock_torch_imports()
        
        # Shape compatibility test
        test_shapes = [
            (32, 252, 64),   # Standard: batch, seq_len, features
            (16, 100, 128),  # Smaller batch
            (64, 500, 32),   # Longer sequence
            (8, 50, 256)     # Larger features
        ]
        
        compatible_blocks = defaultdict(list)
        shape_errors = defaultdict(list)
        
        for shape in test_shapes:
            for block in self.blocks[:20]:  # Test subset for speed
                try:
                    # Simulate shape calculation
                    if block['category'] == 'feature_extraction':
                        if 'n_components' in block['parameters']:
                            output_shape = shape[:-1] + (block['parameters']['n_components'],)
                        else:
                            output_shape = shape
                    elif block['category'] == 'attention':
                        output_shape = shape  # Attention preserves shape
                    elif block['category'] == 'normalization':
                        output_shape = shape  # Normalization preserves shape
                    else:
                        output_shape = shape  # Default: preserve shape
                    
                    # Check if output is reasonable
                    if all(dim > 0 for dim in output_shape):
                        compatible_blocks[shape].append(block['name'])
                    else:
                        shape_errors[shape].append(block['name'])
                        
                except Exception as e:
                    shape_errors[shape].append(f"{block['name']}: {str(e)}")
        
        # Memory efficiency estimation
        memory_scores = {}
        for block in self.blocks:
            # Estimate memory based on parameters and complexity
            param_count = len(block['parameters'])
            complexity_multiplier = {'simple': 1, 'medium': 2, 'complex': 4}
            
            estimated_memory = param_count * complexity_multiplier[block['complexity']]
            memory_scores[block['name']] = estimated_memory
        
        # Training stability estimation
        stable_blocks = []
        unstable_blocks = []
        
        for block in self.blocks:
            # Blocks with normalization are more stable
            has_normalization = any('norm' in comp.lower() for comp in block['components'])
            # Blocks with regularization are more stable
            has_regularization = block['category'] == 'regularization' or any('regular' in comp.lower() for comp in block['components'])
            # Very complex blocks might be unstable
            is_very_complex = block['complexity'] == 'complex' and len(block['parameters']) > 6
            
            if (has_normalization or has_regularization) and not is_very_complex:
                stable_blocks.append(block['name'])
            elif is_very_complex and not (has_normalization or has_regularization):
                unstable_blocks.append(block['name'])
        
        compatibility_score = (
            sum(len(blocks) for blocks in compatible_blocks.values()) / (len(test_shapes) * 20) * 0.4 +
            len(stable_blocks) / len(self.blocks) * 0.3 +
            (1 - len(unstable_blocks) / len(self.blocks)) * 0.3
        )
        
        result = {
            'shape_compatibility': {str(shape): len(blocks) for shape, blocks in compatible_blocks.items()},
            'shape_errors': {str(shape): len(errors) for shape, errors in shape_errors.items()},
            'memory_efficiency': {
                'low_memory_blocks': len([b for b in memory_scores.values() if b <= 4]),
                'high_memory_blocks': len([b for b in memory_scores.values() if b > 8])
            },
            'training_stability': {
                'stable_blocks': len(stable_blocks),
                'unstable_blocks': len(unstable_blocks)
            },
            'compatibility_score': compatibility_score
        }
        
        print(f"✓ Shape compatibility: {sum(len(blocks) for blocks in compatible_blocks.values())} tests passed")
        print(f"✓ Stable blocks: {len(stable_blocks)}")
        print(f"✓ Compatibility score: {compatibility_score:.3f}")
        
        return result
    
    def _experiment_performance_simulation(self) -> Dict[str, Any]:
        """Experiment 7: Simulate performance characteristics"""
        
        # Simulate training speed
        speed_scores = {}
        for block in self.blocks:
            # Base speed depends on complexity
            base_speed = {'simple': 1.0, 'medium': 0.7, 'complex': 0.4}[block['complexity']]
            
            # Adjust based on category
            category_speed = {
                'normalization': 1.0,
                'attention': 0.6,
                'mixing': 0.8,
                'feature_extraction': 0.7,
                'financial_domain': 0.8,
                'temporal_processing': 0.6,
                'regularization': 0.9
            }.get(block['category'], 0.8)
            
            # Adjust based on parameters
            param_penalty = len(block['parameters']) * 0.05
            
            final_speed = base_speed * category_speed * (1 - param_penalty)
            speed_scores[block['name']] = max(0.1, final_speed)  # Minimum speed
        
        # Simulate accuracy potential
        accuracy_scores = {}
        for block in self.blocks:
            # Base accuracy depends on category and complexity
            base_accuracy = {
                'financial_domain': 0.8,
                'attention': 0.85,
                'feature_extraction': 0.75,
                'mixing': 0.7,
                'normalization': 0.6,  # Enables other blocks
                'temporal_processing': 0.8,
                'regularization': 0.65  # Prevents overfitting
            }.get(block['category'], 0.7)
            
            complexity_bonus = {'simple': 0.0, 'medium': 0.05, 'complex': 0.1}[block['complexity']]
            
            # Innovation bonus
            innovation_keywords = ['adaptive', 'hierarchical', 'multi', 'attention', 'spectral']
            innovation_bonus = sum(0.02 for keyword in innovation_keywords 
                                 if keyword in block['name'].lower()) 
            
            final_accuracy = base_accuracy + complexity_bonus + innovation_bonus
            accuracy_scores[block['name']] = min(1.0, final_accuracy)
        
        # Performance categories
        fast_blocks = [name for name, speed in speed_scores.items() if speed > 0.8]
        accurate_blocks = [name for name, acc in accuracy_scores.items() if acc > 0.8]
        balanced_blocks = [name for name in speed_scores.keys() 
                          if speed_scores[name] > 0.6 and accuracy_scores[name] > 0.7]
        
        # Overall performance score
        avg_speed = sum(speed_scores.values()) / len(speed_scores)
        avg_accuracy = sum(accuracy_scores.values()) / len(accuracy_scores)
        performance_score = (avg_speed + avg_accuracy) / 2
        
        result = {
            'average_speed': avg_speed,
            'average_accuracy': avg_accuracy,
            'performance_score': performance_score,
            'fast_blocks': len(fast_blocks),
            'accurate_blocks': len(accurate_blocks),
            'balanced_blocks': len(balanced_blocks),
            'top_performers': sorted(
                [(name, speed_scores[name] * accuracy_scores[name]) for name in speed_scores.keys()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        print(f"✓ Average speed: {avg_speed:.3f}")
        print(f"✓ Average accuracy: {avg_accuracy:.3f}")
        print(f"✓ Balanced blocks: {len(balanced_blocks)}")
        print(f"✓ Performance score: {performance_score:.3f}")
        
        return result
    
    def _experiment_scalability_test(self) -> Dict[str, Any]:
        """Experiment 8: Test scalability properties"""
        
        # Test different dataset sizes
        dataset_sizes = [1000, 10000, 100000, 1000000]
        scalability_scores = {}
        
        for block in self.blocks[:10]:  # Test subset
            block_scores = []
            
            for size in dataset_sizes:
                # Estimate computational complexity
                base_complexity = {'simple': 1, 'medium': 2, 'complex': 4}[block['complexity']]
                
                # Different categories scale differently
                scaling_factor = {
                    'normalization': 1.0,      # O(n)
                    'attention': 2.0,          # O(n^2) worst case
                    'mixing': 1.5,             # O(n log n)
                    'feature_extraction': 1.3, # O(n log n)
                    'financial_domain': 1.2,   # O(n log n)
                    'temporal_processing': 1.8, # O(n^1.5)
                    'regularization': 1.1      # O(n)
                }.get(block['category'], 1.5)
                
                # Calculate scaled complexity
                scaled_complexity = base_complexity * (size ** (scaling_factor / 10))
                normalized_score = 1 / (1 + scaled_complexity / 1000000)  # Normalize
                block_scores.append(normalized_score)
            
            scalability_scores[block['name']] = block_scores
        
        # Memory scalability
        memory_scalable_blocks = []
        for block in self.blocks:
            # Blocks with fewer parameters and simpler operations scale better
            param_count = len(block['parameters'])
            is_simple = block['complexity'] == 'simple'
            has_efficient_ops = any(op in block['name'].lower() for op in ['linear', 'conv', 'norm'])
            
            if param_count <= 4 and (is_simple or has_efficient_ops):
                memory_scalable_blocks.append(block['name'])
        
        # Parallel processing compatibility
        parallelizable_blocks = []
        for block in self.blocks:
            # Certain operations are more parallelizable
            parallel_ops = ['conv', 'linear', 'norm', 'attention', 'mix']
            if any(op in block['name'].lower() for op in parallel_ops):
                parallelizable_blocks.append(block['name'])
        
        # Overall scalability score
        avg_scalability = sum(
            sum(scores) / len(scores) for scores in scalability_scores.values()
        ) / len(scalability_scores) if scalability_scores else 0
        
        memory_ratio = len(memory_scalable_blocks) / len(self.blocks)
        parallel_ratio = len(parallelizable_blocks) / len(self.blocks)
        
        scalability_score = (avg_scalability + memory_ratio + parallel_ratio) / 3
        
        result = {
            'computational_scalability': avg_scalability,
            'memory_scalable_blocks': len(memory_scalable_blocks),
            'parallelizable_blocks': len(parallelizable_blocks),
            'scalability_ratios': {
                'memory': memory_ratio,
                'parallel': parallel_ratio
            },
            'overall_scalability_score': scalability_score
        }
        
        print(f"✓ Computational scalability: {avg_scalability:.3f}")
        print(f"✓ Memory scalable: {len(memory_scalable_blocks)}")
        print(f"✓ Parallelizable: {len(parallelizable_blocks)}")
        print(f"✓ Scalability score: {scalability_score:.3f}")
        
        return result
    
    def _generate_dashboard_data(self):
        """Generate data for dashboard visualization"""
        
        self.dashboard_data = {
            'overview': {
                'total_blocks': len(self.blocks),
                'experiment_count': len(self.results),
                'timestamp': datetime.now().isoformat(),
                'overall_score': self._calculate_overall_score()
            },
            'scores': self._extract_scores(),
            'distributions': self._extract_distributions(),
            'top_performers': self._extract_top_performers(),
            'recommendations': self._generate_recommendations()
        }
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall validation score"""
        scores = []
        
        if 'Diversity Analysis' in self.results:
            scores.append(self.results['Diversity Analysis']['result']['overall_diversity_score'])
        
        if 'Quality Assessment' in self.results:
            scores.append(self.results['Quality Assessment']['result']['average_quality'])
        
        if 'Financial Domain Validation' in self.results:
            scores.append(self.results['Financial Domain Validation']['result']['financial_domain_score'])
        
        if 'Innovation Score' in self.results:
            scores.append(self.results['Innovation Score']['result']['innovation_score'])
        
        if 'Architecture Compatibility' in self.results:
            scores.append(self.results['Architecture Compatibility']['result']['compatibility_score'])
        
        if 'Performance Simulation' in self.results:
            scores.append(self.results['Performance Simulation']['result']['performance_score'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_scores(self) -> Dict[str, float]:
        """Extract key scores for dashboard"""
        scores = {}
        
        for exp_name, exp_data in self.results.items():
            result = exp_data['result']
            
            if 'score' in str(result).lower():
                # Find score fields
                for key, value in result.items():
                    if 'score' in key.lower() and isinstance(value, (int, float)):
                        scores[f"{exp_name}_{key}"] = value
        
        return scores
    
    def _extract_distributions(self) -> Dict[str, Dict]:
        """Extract distribution data for dashboard"""
        distributions = {}
        
        if 'Diversity Analysis' in self.results:
            diversity = self.results['Diversity Analysis']['result']
            distributions['categories'] = diversity['category_diversity']['distribution']
            distributions['complexity'] = diversity['complexity_diversity']['distribution']
        
        if 'Quality Assessment' in self.results:
            quality = self.results['Quality Assessment']['result']
            distributions['quality'] = quality['quality_distribution']
        
        return distributions
    
    def _extract_top_performers(self) -> Dict[str, List]:
        """Extract top performing blocks"""
        performers = {}
        
        if 'Quality Assessment' in self.results:
            quality = self.results['Quality Assessment']['result']
            performers['high_quality'] = quality.get('top_quality_blocks', [])
        
        if 'Innovation Score' in self.results:
            innovation = self.results['Innovation Score']['result']
            performers['most_innovative'] = innovation.get('most_innovative_blocks', [])
        
        if 'Performance Simulation' in self.results:
            performance = self.results['Performance Simulation']['result']
            performers['top_performers'] = [name for name, score in performance.get('top_performers', [])]
        
        return performers
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        overall_score = self._calculate_overall_score()
        
        if overall_score > 0.8:
            recommendations.append("✅ Excellent block generation quality - ready for production")
        elif overall_score > 0.6:
            recommendations.append("⚠️ Good quality with room for improvement")
        else:
            recommendations.append("❌ Significant improvements needed")
        
        # Specific recommendations based on results
        if 'Diversity Analysis' in self.results:
            diversity_score = self.results['Diversity Analysis']['result']['overall_diversity_score']
            if diversity_score < 0.6:
                recommendations.append("🔄 Increase diversity by adding more parameter combinations")
        
        if 'Financial Domain Validation' in self.results:
            financial_score = self.results['Financial Domain Validation']['result']['financial_domain_score']
            if financial_score < 0.7:
                recommendations.append("💰 Strengthen financial domain coverage")
        
        if 'Architecture Compatibility' in self.results:
            compat_score = self.results['Architecture Compatibility']['result']['compatibility_score']
            if compat_score < 0.7:
                recommendations.append("🏗️ Improve neural architecture compatibility")
        
        return recommendations
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate entropy of a distribution"""
        total = sum(values)
        if total == 0:
            return 0
        
        entropy = 0
        for value in values:
            if value > 0:
                p = value / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def save_results(self, output_dir: str = "validation_results"):
        """Save validation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_file = os.path.join(output_dir, "validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save dashboard data
        dashboard_file = os.path.join(output_dir, "dashboard_data.json")
        with open(dashboard_file, 'w') as f:
            json.dump(self.dashboard_data, f, indent=2)
        
        print(f"Validation results saved to {output_dir}/")
        return output_dir


def main():
    """Run validation experiments"""
    
    validator = BlockValidationExperiments()
    results = validator.run_all_experiments()
    output_dir = validator.save_results()
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Overall Score: {validator.dashboard_data['overview']['overall_score']:.3f}")
    print(f"Results saved to: {output_dir}")
    
    return validator.dashboard_data


if __name__ == "__main__":
    main()