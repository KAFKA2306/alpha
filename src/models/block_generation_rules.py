"""
Block Generation Rules for Diverse Domain Block Creation

This module contains rules and templates for generating diverse new blocks
based on analysis of existing domain blocks and financial domain knowledge.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import random
import itertools


class BlockCategory(Enum):
    """Categories of domain blocks"""
    NORMALIZATION = "normalization"
    FEATURE_EXTRACTION = "feature_extraction"
    MIXING = "mixing"
    ENCODING = "encoding"
    FINANCIAL_DOMAIN = "financial_domain"
    PREDICTION_HEADS = "prediction_heads"
    SEQUENCE_MODELS = "sequence_models"
    ATTENTION = "attention"
    REGULARIZATION = "regularization"
    ACTIVATION = "activation"
    AGGREGATION = "aggregation"
    TEMPORAL_PROCESSING = "temporal_processing"
    CROSS_SECTIONAL = "cross_sectional"


@dataclass
class BlockTemplate:
    """Template for generating new blocks"""
    name_pattern: str
    category: BlockCategory
    description_pattern: str
    base_components: List[str]
    hyperparameters: Dict[str, List[Any]]
    complexity_level: str  # 'simple', 'medium', 'complex'
    input_requirements: Dict[str, Any]
    output_modifications: Dict[str, Any]


class BlockGenerationRules:
    """Rules for generating diverse domain blocks"""
    
    def __init__(self):
        self.templates = self._create_block_templates()
        self.component_library = self._create_component_library()
        self.diversity_metrics = self._create_diversity_metrics()
        
    def _create_block_templates(self) -> Dict[str, List[BlockTemplate]]:
        """Create templates for each block category"""
        templates = {
            BlockCategory.NORMALIZATION: [
                BlockTemplate(
                    name_pattern="{normalization_type}_{scope}_norm",
                    category=BlockCategory.NORMALIZATION,
                    description_pattern="{normalization_type} normalization applied to {scope} for {purpose}",
                    base_components=["normalization", "scaling", "centering"],
                    hyperparameters={
                        'normalization_type': ['adaptive', 'robust', 'spectral', 'quantile', 'running', 'exponential'],
                        'scope': ['channel', 'temporal', 'spatial', 'feature', 'cross_sectional'],
                        'purpose': ['stabilization', 'regularization', 'feature_enhancement', 'distribution_alignment']
                    },
                    complexity_level='medium',
                    input_requirements={'min_dims': 3, 'sequence_aware': True},
                    output_modifications={'shape_preserved': True}
                ),
                BlockTemplate(
                    name_pattern="{adaptive_type}_adaptive_norm",
                    category=BlockCategory.NORMALIZATION,
                    description_pattern="Adaptive normalization with {adaptive_type} adjustment mechanism",
                    base_components=["adaptive_scaling", "learned_parameters", "context_aware"],
                    hyperparameters={
                        'adaptive_type': ['meta_learning', 'attention_weighted', 'regime_specific', 'momentum_adjusted'],
                        'adaptation_rate': [0.01, 0.05, 0.1, 0.2],
                        'context_window': [5, 10, 20, 50]
                    },
                    complexity_level='complex',
                    input_requirements={'min_dims': 3, 'temporal_data': True},
                    output_modifications={'shape_preserved': True, 'learned_adaptation': True}
                )
            ],
            
            BlockCategory.FEATURE_EXTRACTION: [
                BlockTemplate(
                    name_pattern="{domain}_{method}_feature",
                    category=BlockCategory.FEATURE_EXTRACTION,
                    description_pattern="Extract {domain} features using {method} decomposition",
                    base_components=["decomposition", "feature_selection", "dimensionality_reduction"],
                    hyperparameters={
                        'domain': ['spectral', 'wavelet', 'statistical', 'geometric', 'topological'],
                        'method': ['basis_pursuit', 'sparse_coding', 'dictionary_learning', 'manifold_learning'],
                        'n_components': [8, 16, 32, 64, 128],
                        'sparsity_level': [0.1, 0.3, 0.5, 0.7]
                    },
                    complexity_level='complex',
                    input_requirements={'min_dims': 2, 'numeric_data': True},
                    output_modifications={'dimensionality_change': True}
                ),
                BlockTemplate(
                    name_pattern="{signal_type}_signal_decomp",
                    category=BlockCategory.FEATURE_EXTRACTION,
                    description_pattern="Decompose {signal_type} signals into interpretable components",
                    base_components=["signal_processing", "basis_functions", "component_analysis"],
                    hyperparameters={
                        'signal_type': ['trend', 'seasonal', 'cyclical', 'noise', 'volatility'],
                        'decomposition_method': ['emd', 'vmd', 'eemd', 'ceemdan', 'ssa'],
                        'n_modes': [3, 5, 7, 10],
                        'noise_threshold': [0.01, 0.05, 0.1]
                    },
                    complexity_level='complex',
                    input_requirements={'temporal_data': True, 'min_length': 50},
                    output_modifications={'multi_component_output': True}
                )
            ],
            
            BlockCategory.MIXING: [
                BlockTemplate(
                    name_pattern="{mixing_type}_{domain}_mixing",
                    category=BlockCategory.MIXING,
                    description_pattern="Mix information across {domain} using {mixing_type} mechanism",
                    base_components=["information_mixing", "cross_connections", "feature_interaction"],
                    hyperparameters={
                        'mixing_type': ['gated', 'attention', 'convolution', 'graph', 'spectral'],
                        'domain': ['temporal', 'cross_sectional', 'feature', 'multi_scale'],
                        'mixing_ratio': [0.2, 0.5, 0.8],
                        'interaction_depth': [1, 2, 3]
                    },
                    complexity_level='medium',
                    input_requirements={'min_dims': 3, 'mixing_compatible': True},
                    output_modifications={'feature_interaction': True}
                ),
                BlockTemplate(
                    name_pattern="{architecture}_cross_mixing",
                    category=BlockCategory.MIXING,
                    description_pattern="Cross-domain mixing using {architecture} architecture",
                    base_components=["cross_attention", "multi_head_interaction", "hierarchical_mixing"],
                    hyperparameters={
                        'architecture': ['transformer', 'graph_neural', 'capsule', 'tensor_network'],
                        'num_heads': [4, 8, 16],
                        'mixing_layers': [1, 2, 3, 4],
                        'attention_type': ['scaled_dot', 'additive', 'multiplicative']
                    },
                    complexity_level='complex',
                    input_requirements={'multi_modal': True, 'attention_compatible': True},
                    output_modifications={'cross_domain_features': True}
                )
            ],
            
            BlockCategory.FINANCIAL_DOMAIN: [
                BlockTemplate(
                    name_pattern="{market_property}_{analysis_type}",
                    category=BlockCategory.FINANCIAL_DOMAIN,
                    description_pattern="Analyze {market_property} using {analysis_type} methodology",
                    base_components=["market_microstructure", "price_dynamics", "risk_modeling"],
                    hyperparameters={
                        'market_property': ['momentum', 'mean_reversion', 'volatility_clustering', 'jump_diffusion', 'regime_switching'],
                        'analysis_type': ['fractal', 'multiscale', 'spectral', 'chaos', 'stochastic'],
                        'lookback_window': [5, 10, 20, 50, 100],
                        'sensitivity': [0.1, 0.5, 1.0, 2.0]
                    },
                    complexity_level='complex',
                    input_requirements={'financial_data': True, 'temporal_structure': True},
                    output_modifications={'market_feature_enhanced': True}
                ),
                BlockTemplate(
                    name_pattern="{factor_type}_factor_exposure",
                    category=BlockCategory.FINANCIAL_DOMAIN,
                    description_pattern="Calculate exposure to {factor_type} factors with dynamic weighting",
                    base_components=["factor_modeling", "risk_decomposition", "attribution_analysis"],
                    hyperparameters={
                        'factor_type': ['style', 'sector', 'macro', 'quality', 'sentiment', 'technical'],
                        'weighting_scheme': ['equal', 'volatility', 'risk_parity', 'momentum'],
                        'rebalancing_frequency': [1, 5, 10, 20],
                        'decay_factor': [0.9, 0.95, 0.99]
                    },
                    complexity_level='complex',
                    input_requirements={'cross_sectional_data': True, 'factor_data': True},
                    output_modifications={'factor_exposures': True}
                )
            ],
            
            BlockCategory.TEMPORAL_PROCESSING: [
                BlockTemplate(
                    name_pattern="{temporal_pattern}_{processing_type}",
                    category=BlockCategory.TEMPORAL_PROCESSING,
                    description_pattern="Process {temporal_pattern} patterns using {processing_type} methods",
                    base_components=["temporal_modeling", "sequence_analysis", "time_series_decomposition"],
                    hyperparameters={
                        'temporal_pattern': ['seasonal', 'trend', 'cyclical', 'irregular', 'structural_break'],
                        'processing_type': ['state_space', 'kalman_filter', 'particle_filter', 'hmm', 'changepoint'],
                        'model_order': [1, 2, 3, 5],
                        'smoothing_parameter': [0.1, 0.3, 0.5, 0.7]
                    },
                    complexity_level='complex',
                    input_requirements={'temporal_data': True, 'min_length': 30},
                    output_modifications={'temporal_features': True}
                )
            ],
            
            BlockCategory.ATTENTION: [
                BlockTemplate(
                    name_pattern="{attention_type}_{scope}_attention",
                    category=BlockCategory.ATTENTION,
                    description_pattern="{attention_type} attention mechanism for {scope} modeling",
                    base_components=["attention_weights", "key_value_query", "attention_pooling"],
                    hyperparameters={
                        'attention_type': ['multi_head', 'sparse', 'local', 'global', 'hierarchical', 'adaptive'],
                        'scope': ['temporal', 'cross_sectional', 'feature', 'multi_scale'],
                        'num_heads': [4, 8, 16, 32],
                        'attention_dropout': [0.1, 0.2, 0.3]
                    },
                    complexity_level='medium',
                    input_requirements={'attention_compatible': True, 'min_dims': 3},
                    output_modifications={'attention_weighted': True}
                )
            ],
            
            BlockCategory.REGULARIZATION: [
                BlockTemplate(
                    name_pattern="{regularization_type}_{target}_regularizer",
                    category=BlockCategory.REGULARIZATION,
                    description_pattern="Apply {regularization_type} regularization to {target}",
                    base_components=["regularization", "constraint_enforcement", "penalty_application"],
                    hyperparameters={
                        'regularization_type': ['adaptive', 'structured', 'group', 'elastic', 'nuclear'],
                        'target': ['weights', 'activations', 'gradients', 'features'],
                        'regularization_strength': [0.001, 0.01, 0.1, 1.0],
                        'adaptation_rate': [0.01, 0.1, 0.5]
                    },
                    complexity_level='medium',
                    input_requirements={'trainable_parameters': True},
                    output_modifications={'regularized_output': True}
                )
            ]
        }
        
        return templates
    
    def _create_component_library(self) -> Dict[str, List[Callable]]:
        """Create library of reusable components"""
        return {
            'activations': [
                lambda: nn.ReLU(),
                lambda: nn.GELU(),
                lambda: nn.Swish(),
                lambda: nn.Mish(),
                lambda: nn.ELU(),
                lambda: nn.SELU(),
                lambda: nn.Tanh(),
                lambda: nn.Sigmoid(),
                lambda: nn.Softplus(),
                lambda: nn.LeakyReLU(),
            ],
            'normalizations': [
                lambda dim: nn.LayerNorm(dim),
                lambda dim: nn.BatchNorm1d(dim),
                lambda dim: nn.GroupNorm(4, dim),
                lambda dim: nn.InstanceNorm1d(dim),
            ],
            'pooling': [
                lambda k: nn.AdaptiveAvgPool1d(k),
                lambda k: nn.AdaptiveMaxPool1d(k),
                lambda k: nn.AvgPool1d(k),
                lambda k: nn.MaxPool1d(k),
            ],
            'transformations': [
                lambda in_dim, out_dim: nn.Linear(in_dim, out_dim),
                lambda in_dim, out_dim: nn.Conv1d(in_dim, out_dim, 3, padding=1),
                lambda in_dim, out_dim: nn.Conv1d(in_dim, out_dim, 5, padding=2),
                lambda in_dim, out_dim: nn.Conv1d(in_dim, out_dim, 7, padding=3),
            ]
        }
    
    def _create_diversity_metrics(self) -> Dict[str, Callable]:
        """Create metrics for measuring block diversity"""
        return {
            'structural_diversity': self._calculate_structural_diversity,
            'functional_diversity': self._calculate_functional_diversity,
            'parameter_diversity': self._calculate_parameter_diversity,
            'computational_diversity': self._calculate_computational_diversity,
            'domain_diversity': self._calculate_domain_diversity
        }
    
    def generate_diverse_blocks(self, 
                              num_blocks: int = 100,
                              category_distribution: Optional[Dict[BlockCategory, float]] = None,
                              complexity_distribution: Optional[Dict[str, float]] = None,
                              min_diversity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Generate diverse blocks according to rules"""
        
        if category_distribution is None:
            category_distribution = self._get_default_category_distribution()
        
        if complexity_distribution is None:
            complexity_distribution = {'simple': 0.2, 'medium': 0.5, 'complex': 0.3}
        
        generated_blocks = []
        diversity_tracker = []
        
        for i in range(num_blocks):
            # Select category and complexity
            category = self._sample_category(category_distribution)
            complexity = self._sample_complexity(complexity_distribution)
            
            # Generate block
            block_spec = self._generate_single_block(category, complexity)
            
            # Check diversity
            if self._check_diversity(block_spec, diversity_tracker, min_diversity_threshold):
                generated_blocks.append(block_spec)
                diversity_tracker.append(block_spec)
            else:
                # Retry with different parameters
                for retry in range(3):
                    block_spec = self._generate_single_block(category, complexity)
                    if self._check_diversity(block_spec, diversity_tracker, min_diversity_threshold):
                        generated_blocks.append(block_spec)
                        diversity_tracker.append(block_spec)
                        break
        
        return generated_blocks
    
    def _get_default_category_distribution(self) -> Dict[BlockCategory, float]:
        """Get default distribution of block categories"""
        return {
            BlockCategory.NORMALIZATION: 0.15,
            BlockCategory.FEATURE_EXTRACTION: 0.20,
            BlockCategory.MIXING: 0.15,
            BlockCategory.FINANCIAL_DOMAIN: 0.25,
            BlockCategory.TEMPORAL_PROCESSING: 0.10,
            BlockCategory.ATTENTION: 0.10,
            BlockCategory.REGULARIZATION: 0.05
        }
    
    def _sample_category(self, distribution: Dict[BlockCategory, float]) -> BlockCategory:
        """Sample category according to distribution"""
        categories = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(categories, weights=weights)[0]
    
    def _sample_complexity(self, distribution: Dict[str, float]) -> str:
        """Sample complexity level according to distribution"""
        levels = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(levels, weights=weights)[0]
    
    def _generate_single_block(self, category: BlockCategory, complexity: str) -> Dict[str, Any]:
        """Generate a single block specification"""
        # Get templates for category
        templates = self.templates.get(category, [])
        if not templates:
            raise ValueError(f"No templates available for category {category}")
        
        # Filter by complexity
        suitable_templates = [t for t in templates if t.complexity_level == complexity]
        if not suitable_templates:
            suitable_templates = templates  # Fallback to all templates
        
        # Select template
        template = random.choice(suitable_templates)
        
        # Generate specific parameters
        block_params = {}
        for param, values in template.hyperparameters.items():
            if isinstance(values, list):
                block_params[param] = random.choice(values)
            else:
                block_params[param] = values
        
        # Generate name and description
        name = template.name_pattern.format(**block_params)
        description = template.description_pattern.format(**block_params)
        
        return {
            'name': name,
            'category': category.value,
            'description': description,
            'template': template,
            'parameters': block_params,
            'complexity': complexity,
            'components': template.base_components.copy()
        }
    
    def _check_diversity(self, 
                        new_block: Dict[str, Any], 
                        existing_blocks: List[Dict[str, Any]], 
                        threshold: float) -> bool:
        """Check if new block is diverse enough"""
        if not existing_blocks:
            return True
        
        diversity_scores = []
        for existing in existing_blocks:
            score = self._calculate_block_similarity(new_block, existing)
            diversity_scores.append(1 - score)  # Convert similarity to diversity
        
        avg_diversity = np.mean(diversity_scores)
        return avg_diversity >= threshold
    
    def _calculate_block_similarity(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> float:
        """Calculate similarity between two blocks"""
        similarities = []
        
        # Category similarity
        cat_sim = 1.0 if block1['category'] == block2['category'] else 0.0
        similarities.append(cat_sim * 0.3)
        
        # Component similarity
        comp1 = set(block1['components'])
        comp2 = set(block2['components'])
        comp_sim = len(comp1 & comp2) / len(comp1 | comp2) if comp1 | comp2 else 0.0
        similarities.append(comp_sim * 0.4)
        
        # Parameter similarity
        param_sim = self._calculate_parameter_similarity(block1['parameters'], block2['parameters'])
        similarities.append(param_sim * 0.3)
        
        return sum(similarities)
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate similarity between parameter sets"""
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = params1[key], params2[key]
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    similarities.append(1 - abs(val1 - val2) / max_val)
                else:
                    similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    # Diversity metric calculation methods
    def _calculate_structural_diversity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate structural diversity of blocks"""
        if len(blocks) <= 1:
            return 1.0
        
        # Compare architectural patterns
        patterns = []
        for block in blocks:
            pattern = {
                'category': block['category'],
                'complexity': block['complexity'],
                'components': tuple(sorted(block['components']))
            }
            patterns.append(pattern)
        
        unique_patterns = len(set(str(p) for p in patterns))
        return unique_patterns / len(patterns)
    
    def _calculate_functional_diversity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate functional diversity of blocks"""
        # Based on different parameter combinations
        if len(blocks) <= 1:
            return 1.0
        
        param_combinations = []
        for block in blocks:
            combo = tuple(sorted(f"{k}:{v}" for k, v in block['parameters'].items()))
            param_combinations.append(combo)
        
        unique_combinations = len(set(param_combinations))
        return unique_combinations / len(param_combinations)
    
    def _calculate_parameter_diversity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate parameter space diversity"""
        if len(blocks) <= 1:
            return 1.0
        
        # Calculate variance in parameter values
        all_params = {}
        for block in blocks:
            for k, v in block['parameters'].items():
                if k not in all_params:
                    all_params[k] = []
                all_params[k].append(v)
        
        diversities = []
        for param, values in all_params.items():
            if len(set(values)) > 1:
                diversities.append(1.0)
            else:
                diversities.append(0.0)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _calculate_computational_diversity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate computational complexity diversity"""
        if len(blocks) <= 1:
            return 1.0
        
        complexities = [block['complexity'] for block in blocks]
        unique_complexities = len(set(complexities))
        return unique_complexities / len(complexities)
    
    def _calculate_domain_diversity(self, blocks: List[Dict[str, Any]]) -> float:
        """Calculate domain knowledge diversity"""
        if len(blocks) <= 1:
            return 1.0
        
        categories = [block['category'] for block in blocks]
        unique_categories = len(set(categories))
        total_categories = len(BlockCategory)
        
        return unique_categories / total_categories


def create_block_generation_rules() -> BlockGenerationRules:
    """Factory function to create block generation rules"""
    return BlockGenerationRules()