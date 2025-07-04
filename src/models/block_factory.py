"""
Block Factory for Mass Production of Diverse Domain Blocks

This module implements the actual block generation system that creates
PyTorch modules based on the generation rules and templates.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from domain_blocks import DomainBlock
from block_generation_rules import BlockGenerationRules, BlockCategory, BlockTemplate
import random
import math
from collections import defaultdict
import itertools


class GeneratedDomainBlock(DomainBlock):
    """A dynamically generated domain block"""
    
    def __init__(self, block_spec: Dict[str, Any], module_factory: Callable):
        super().__init__(
            name=block_spec['name'],
            category=block_spec['category'],
            description=block_spec['description']
        )
        self.block_spec = block_spec
        self.module_factory = module_factory
        self.template = block_spec['template']
        self.parameters = block_spec['parameters']
        self.complexity = block_spec['complexity']
        self.components = block_spec['components']
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        """Create the PyTorch module for this generated block"""
        return self.module_factory(input_shape, self.parameters, **kwargs)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        """Calculate output shape based on block type"""
        return self._calculate_output_shape(input_shape, **kwargs)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for this block"""
        return self.parameters.copy()
    
    def _calculate_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        """Calculate output shape based on block specifications"""
        if self.template.output_modifications.get('shape_preserved', False):
            return input_shape
        elif self.template.output_modifications.get('dimensionality_change', False):
            # For feature extraction blocks that change dimensionality
            if 'n_components' in self.parameters:
                return input_shape[:-1] + (self.parameters['n_components'],)
            elif 'feature_expansion' in self.parameters:
                expansion_factor = self.parameters['feature_expansion']
                return input_shape[:-1] + (input_shape[-1] * expansion_factor,)
        elif self.template.output_modifications.get('multi_component_output', False):
            # For signal decomposition blocks
            n_components = self.parameters.get('n_modes', 3)
            return input_shape[:-1] + (input_shape[-1] * n_components,)
        elif self.template.output_modifications.get('factor_exposures', False):
            # For factor exposure blocks
            n_factors = self.parameters.get('n_factors', 5)
            return input_shape[:-1] + (input_shape[-1] + n_factors,)
        
        return input_shape


class BlockSimilarityTester:
    """Test similarity between blocks to ensure diversity"""
    
    def __init__(self):
        self.similarity_metrics = {
            'structural': self._structural_similarity,
            'functional': self._functional_similarity,
            'parameter': self._parameter_similarity,
            'output': self._output_similarity,
            'computational': self._computational_similarity
        }
    
    def calculate_similarity(self, block1: DomainBlock, block2: DomainBlock) -> Dict[str, float]:
        """Calculate various similarity metrics between two blocks"""
        similarities = {}
        
        for metric_name, metric_func in self.similarity_metrics.items():
            try:
                similarity = metric_func(block1, block2)
                similarities[metric_name] = similarity
            except Exception as e:
                similarities[metric_name] = 0.0
        
        # Calculate overall similarity
        similarities['overall'] = np.mean(list(similarities.values()))
        
        return similarities
    
    def _structural_similarity(self, block1: DomainBlock, block2: DomainBlock) -> float:
        """Calculate structural similarity based on architecture"""
        # Compare categories
        cat_sim = 1.0 if block1.category == block2.category else 0.0
        
        # Compare components if available
        comp_sim = 0.0
        if hasattr(block1, 'components') and hasattr(block2, 'components'):
            comp1 = set(block1.components)
            comp2 = set(block2.components)
            if comp1 or comp2:
                comp_sim = len(comp1 & comp2) / len(comp1 | comp2)
        
        return (cat_sim + comp_sim) / 2
    
    def _functional_similarity(self, block1: DomainBlock, block2: DomainBlock) -> float:
        """Calculate functional similarity based on parameters"""
        if not (hasattr(block1, 'parameters') and hasattr(block2, 'parameters')):
            return 0.0
        
        params1 = block1.parameters
        params2 = block2.parameters
        
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = params1[key], params2[key]
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2), 1e-8)
                similarities.append(1 - abs(val1 - val2) / max_val)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _parameter_similarity(self, block1: DomainBlock, block2: DomainBlock) -> float:
        """Calculate parameter space similarity"""
        hyperparams1 = block1.get_hyperparameters()
        hyperparams2 = block2.get_hyperparameters()
        
        all_keys = set(hyperparams1.keys()) | set(hyperparams2.keys())
        if not all_keys:
            return 0.0
        
        common_keys = set(hyperparams1.keys()) & set(hyperparams2.keys())
        jaccard_sim = len(common_keys) / len(all_keys)
        
        return jaccard_sim
    
    def _output_similarity(self, block1: DomainBlock, block2: DomainBlock) -> float:
        """Calculate output similarity based on shape transformations"""
        # Test with a standard input shape
        test_shape = (32, 252, 64)  # batch, seq_len, features
        
        try:
            output1 = block1.get_output_shape(test_shape)
            output2 = block2.get_output_shape(test_shape)
            
            if output1 == output2:
                return 1.0
            else:
                # Compare shape changes
                change1 = np.prod(output1) / np.prod(test_shape)
                change2 = np.prod(output2) / np.prod(test_shape)
                return 1 - abs(change1 - change2) / max(change1, change2, 1e-8)
        except:
            return 0.0
    
    def _computational_similarity(self, block1: DomainBlock, block2: DomainBlock) -> float:
        """Calculate computational complexity similarity"""
        if hasattr(block1, 'complexity') and hasattr(block2, 'complexity'):
            if block1.complexity == block2.complexity:
                return 1.0
            else:
                complexity_map = {'simple': 1, 'medium': 2, 'complex': 3}
                c1 = complexity_map.get(block1.complexity, 2)
                c2 = complexity_map.get(block2.complexity, 2)
                return 1 - abs(c1 - c2) / 2
        return 0.0
    
    def test_diversity_threshold(self, 
                               blocks: List[DomainBlock], 
                               threshold: float = 0.7) -> Tuple[bool, Dict[str, float]]:
        """Test if a set of blocks meets diversity threshold"""
        if len(blocks) <= 1:
            return True, {'overall_diversity': 1.0}
        
        similarities = []
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                sim = self.calculate_similarity(blocks[i], blocks[j])
                similarities.append(sim)
        
        # Calculate average similarities
        avg_similarities = defaultdict(list)
        for sim in similarities:
            for key, value in sim.items():
                avg_similarities[key].append(value)
        
        avg_similarities = {k: np.mean(v) for k, v in avg_similarities.items()}
        
        # Diversity is 1 - similarity
        diversity_scores = {k: 1 - v for k, v in avg_similarities.items()}
        
        meets_threshold = diversity_scores['overall'] >= threshold
        
        return meets_threshold, diversity_scores


class BlockFactory:
    """Factory for mass production of diverse domain blocks"""
    
    def __init__(self):
        self.generation_rules = BlockGenerationRules()
        self.similarity_tester = BlockSimilarityTester()
        self.module_factories = self._create_module_factories()
    
    def _create_module_factories(self) -> Dict[str, Callable]:
        """Create module factories for different block types"""
        return {
            'normalization': self._create_normalization_module,
            'feature_extraction': self._create_feature_extraction_module,
            'mixing': self._create_mixing_module,
            'financial_domain': self._create_financial_domain_module,
            'temporal_processing': self._create_temporal_processing_module,
            'attention': self._create_attention_module,
            'regularization': self._create_regularization_module,
            'encoding': self._create_encoding_module,
            'prediction_heads': self._create_prediction_head_module,
            'sequence_models': self._create_sequence_model_module,
            'activation': self._create_activation_module,
            'aggregation': self._create_aggregation_module,
            'cross_sectional': self._create_cross_sectional_module
        }
    
    def mass_produce_blocks(self, 
                          num_blocks: int = 200,
                          diversity_threshold: float = 0.6,
                          max_iterations: int = 1000) -> List[GeneratedDomainBlock]:
        """Mass produce diverse domain blocks"""
        
        print(f"Starting mass production of {num_blocks} blocks...")
        
        # Generate block specifications
        block_specs = self.generation_rules.generate_diverse_blocks(
            num_blocks=max_iterations,
            min_diversity_threshold=diversity_threshold
        )
        
        # Create actual blocks
        generated_blocks = []
        
        for i, spec in enumerate(block_specs):
            if len(generated_blocks) >= num_blocks:
                break
                
            try:
                # Get module factory
                factory = self.module_factories.get(spec['category'])
                if factory is None:
                    print(f"No factory for category: {spec['category']}")
                    continue
                
                # Create generated block
                block = GeneratedDomainBlock(spec, factory)
                
                # Test diversity against existing blocks
                if self._test_block_diversity(block, generated_blocks, diversity_threshold):
                    generated_blocks.append(block)
                    
                    if len(generated_blocks) % 20 == 0:
                        print(f"Generated {len(generated_blocks)} blocks...")
                
            except Exception as e:
                print(f"Error creating block {i}: {e}")
                continue
        
        print(f"Successfully generated {len(generated_blocks)} diverse blocks")
        
        # Final diversity test
        meets_threshold, diversity_scores = self.similarity_tester.test_diversity_threshold(
            generated_blocks, diversity_threshold
        )
        
        print(f"Final diversity scores: {diversity_scores}")
        print(f"Meets threshold ({diversity_threshold}): {meets_threshold}")
        
        return generated_blocks
    
    def _test_block_diversity(self, 
                            new_block: GeneratedDomainBlock, 
                            existing_blocks: List[GeneratedDomainBlock],
                            threshold: float) -> bool:
        """Test if new block is diverse enough"""
        if not existing_blocks:
            return True
        
        similarities = []
        for existing in existing_blocks:
            sim = self.similarity_tester.calculate_similarity(new_block, existing)
            similarities.append(sim['overall'])
        
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        return diversity >= threshold
    
    # Module factory methods
    def _create_normalization_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create normalization module"""
        normalization_type = params.get('normalization_type', 'adaptive')
        scope = params.get('scope', 'feature')
        
        class GeneratedNormalizationModule(nn.Module):
            def __init__(self, input_dim, normalization_type, scope):
                super().__init__()
                self.normalization_type = normalization_type
                self.scope = scope
                
                if normalization_type == 'adaptive':
                    self.norm = nn.LayerNorm(input_dim)
                    self.adaptation = nn.Linear(input_dim, input_dim)
                    self.gate = nn.Sigmoid()
                elif normalization_type == 'robust':
                    self.norm = nn.LayerNorm(input_dim)
                    self.outlier_detection = nn.Linear(input_dim, 1)
                elif normalization_type == 'spectral':
                    self.norm = nn.LayerNorm(input_dim)
                    self.spectral_scale = nn.Parameter(torch.ones(input_dim))
                elif normalization_type == 'quantile':
                    self.quantile_params = nn.Parameter(torch.randn(input_dim, 3))  # 25%, 50%, 75%
                else:
                    self.norm = nn.LayerNorm(input_dim)
            
            def forward(self, x):
                if self.normalization_type == 'adaptive':
                    normed = self.norm(x)
                    adaptation = self.gate(self.adaptation(x))
                    return normed * adaptation + x * (1 - adaptation)
                elif self.normalization_type == 'robust':
                    outlier_weights = torch.sigmoid(self.outlier_detection(x))
                    return self.norm(x) * outlier_weights + x * (1 - outlier_weights)
                elif self.normalization_type == 'spectral':
                    fft_x = torch.fft.fft(x, dim=-2)
                    scaled_fft = fft_x * self.spectral_scale
                    return torch.real(torch.fft.ifft(scaled_fft, dim=-2))
                elif self.normalization_type == 'quantile':
                    q25, q50, q75 = torch.chunk(self.quantile_params, 3, dim=-1)
                    x_norm = (x - q50) / (q75 - q25 + 1e-8)
                    return x_norm
                else:
                    return self.norm(x)
        
        return GeneratedNormalizationModule(input_shape[-1], normalization_type, scope)
    
    def _create_feature_extraction_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create feature extraction module"""
        domain = params.get('domain', 'spectral')
        method = params.get('method', 'basis_pursuit')
        n_components = params.get('n_components', 32)
        
        class GeneratedFeatureExtractionModule(nn.Module):
            def __init__(self, input_dim, domain, method, n_components):
                super().__init__()
                self.domain = domain
                self.method = method
                self.n_components = n_components
                
                if domain == 'spectral':
                    self.frequency_weights = nn.Parameter(torch.randn(input_dim, n_components))
                elif domain == 'wavelet':
                    self.wavelet_bank = nn.Parameter(torch.randn(n_components, input_dim))
                elif domain == 'statistical':
                    self.moment_extractors = nn.ModuleList([
                        nn.Linear(input_dim, n_components) for _ in range(4)  # mean, var, skew, kurtosis
                    ])
                elif domain == 'geometric':
                    self.geometric_transform = nn.Linear(input_dim, n_components)
                    self.manifold_proj = nn.Linear(n_components, input_dim)
                else:
                    self.projection = nn.Linear(input_dim, n_components)
                    self.reconstruction = nn.Linear(n_components, input_dim)
            
            def forward(self, x):
                if self.domain == 'spectral':
                    fft_x = torch.fft.fft(x, dim=-2)
                    freq_features = torch.matmul(torch.real(fft_x), self.frequency_weights)
                    return torch.cat([x, freq_features], dim=-1)
                elif self.domain == 'wavelet':
                    # Simplified wavelet transform
                    wavelet_features = torch.matmul(x, self.wavelet_bank.T)
                    return torch.cat([x, wavelet_features], dim=-1)
                elif self.domain == 'statistical':
                    # Statistical moments
                    moments = []
                    for i, extractor in enumerate(self.moment_extractors):
                        if i == 0:  # mean
                            moment = extractor(x.mean(dim=-2, keepdim=True).expand_as(x))
                        elif i == 1:  # variance
                            moment = extractor(x.var(dim=-2, keepdim=True).expand_as(x))
                        elif i == 2:  # skewness (approximation)
                            centered = x - x.mean(dim=-2, keepdim=True)
                            skew = (centered ** 3).mean(dim=-2, keepdim=True).expand_as(x)
                            moment = extractor(skew)
                        else:  # kurtosis (approximation)
                            centered = x - x.mean(dim=-2, keepdim=True)
                            kurt = (centered ** 4).mean(dim=-2, keepdim=True).expand_as(x)
                            moment = extractor(kurt)
                        moments.append(moment)
                    return torch.cat([x] + moments, dim=-1)
                elif self.domain == 'geometric':
                    # Geometric transformation
                    transformed = torch.tanh(self.geometric_transform(x))
                    reconstructed = self.manifold_proj(transformed)
                    return torch.cat([x, reconstructed], dim=-1)
                else:
                    # Standard projection
                    projected = self.projection(x)
                    reconstructed = self.reconstruction(projected)
                    return reconstructed
        
        return GeneratedFeatureExtractionModule(input_shape[-1], domain, method, n_components)
    
    def _create_mixing_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create mixing module"""
        mixing_type = params.get('mixing_type', 'gated')
        domain = params.get('domain', 'temporal')
        
        class GeneratedMixingModule(nn.Module):
            def __init__(self, input_dim, seq_len, mixing_type, domain):
                super().__init__()
                self.mixing_type = mixing_type
                self.domain = domain
                
                if mixing_type == 'gated':
                    self.gate = nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.Sigmoid()
                    )
                    self.transform = nn.Linear(input_dim, input_dim)
                elif mixing_type == 'attention':
                    self.attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
                elif mixing_type == 'convolution':
                    self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
                elif mixing_type == 'graph':
                    self.adjacency = nn.Parameter(torch.randn(seq_len, seq_len))
                    self.node_transform = nn.Linear(input_dim, input_dim)
                else:
                    self.mixing_weights = nn.Parameter(torch.randn(input_dim, input_dim))
            
            def forward(self, x):
                if self.mixing_type == 'gated':
                    gate_values = self.gate(x)
                    transformed = self.transform(x)
                    return x * gate_values + transformed * (1 - gate_values)
                elif self.mixing_type == 'attention':
                    attended, _ = self.attention(x, x, x)
                    return attended
                elif self.mixing_type == 'convolution':
                    # x: (batch, seq_len, features)
                    x_transposed = x.transpose(1, 2)  # (batch, features, seq_len)
                    conv_out = self.conv(x_transposed)
                    return conv_out.transpose(1, 2)
                elif self.mixing_type == 'graph':
                    # Graph convolution
                    adj_norm = torch.softmax(self.adjacency, dim=-1)
                    mixed = torch.matmul(adj_norm, x)
                    return self.node_transform(mixed)
                else:
                    return torch.matmul(x, self.mixing_weights)
        
        return GeneratedMixingModule(input_shape[-1], input_shape[-2], mixing_type, domain)
    
    def _create_financial_domain_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create financial domain module"""
        market_property = params.get('market_property', 'momentum')
        analysis_type = params.get('analysis_type', 'multiscale')
        lookback_window = params.get('lookback_window', 20)
        
        class GeneratedFinancialDomainModule(nn.Module):
            def __init__(self, input_dim, market_property, analysis_type, lookback_window):
                super().__init__()
                self.market_property = market_property
                self.analysis_type = analysis_type
                self.lookback_window = lookback_window
                
                if market_property == 'momentum':
                    self.momentum_weights = nn.Parameter(torch.randn(lookback_window))
                elif market_property == 'mean_reversion':
                    self.reversion_strength = nn.Parameter(torch.tensor(0.1))
                    self.mean_estimator = nn.Linear(input_dim, input_dim)
                elif market_property == 'volatility_clustering':
                    self.vol_model = nn.GRU(input_dim, input_dim, batch_first=True)
                elif market_property == 'regime_switching':
                    self.regime_detector = nn.Sequential(
                        nn.Linear(input_dim, 16),
                        nn.ReLU(),
                        nn.Linear(16, 3)  # 3 regimes
                    )
                    self.regime_transforms = nn.ModuleList([
                        nn.Linear(input_dim, input_dim) for _ in range(3)
                    ])
                
                self.output_transform = nn.Linear(input_dim, input_dim)
            
            def forward(self, x):
                if self.market_property == 'momentum':
                    # Weighted momentum calculation
                    weights = torch.softmax(self.momentum_weights, dim=0)
                    momentum = torch.zeros_like(x)
                    for i in range(min(self.lookback_window, x.shape[1])):
                        momentum += weights[i] * torch.roll(x, shifts=i, dims=1)
                    return self.output_transform(momentum)
                elif self.market_property == 'mean_reversion':
                    # Mean reversion calculation
                    estimated_mean = self.mean_estimator(x.mean(dim=1, keepdim=True))
                    reversion = x + self.reversion_strength * (estimated_mean - x)
                    return self.output_transform(reversion)
                elif self.market_property == 'volatility_clustering':
                    # Volatility clustering with GRU
                    vol_features, _ = self.vol_model(x)
                    return self.output_transform(vol_features)
                elif self.market_property == 'regime_switching':
                    # Regime switching
                    regime_probs = torch.softmax(self.regime_detector(x), dim=-1)
                    regime_outputs = []
                    for i, transform in enumerate(self.regime_transforms):
                        regime_outputs.append(transform(x) * regime_probs[..., i:i+1])
                    mixed_output = sum(regime_outputs)
                    return self.output_transform(mixed_output)
                else:
                    return self.output_transform(x)
        
        return GeneratedFinancialDomainModule(input_shape[-1], market_property, analysis_type, lookback_window)
    
    def _create_temporal_processing_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create temporal processing module"""
        temporal_pattern = params.get('temporal_pattern', 'seasonal')
        processing_type = params.get('processing_type', 'state_space')
        
        class GeneratedTemporalProcessingModule(nn.Module):
            def __init__(self, input_dim, temporal_pattern, processing_type):
                super().__init__()
                self.temporal_pattern = temporal_pattern
                self.processing_type = processing_type
                
                if processing_type == 'state_space':
                    self.state_transition = nn.Linear(input_dim, input_dim)
                    self.observation_model = nn.Linear(input_dim, input_dim)
                elif processing_type == 'kalman_filter':
                    self.prediction_model = nn.Linear(input_dim, input_dim)
                    self.noise_model = nn.Linear(input_dim, input_dim)
                elif processing_type == 'hmm':
                    self.hidden_states = nn.Parameter(torch.randn(5, input_dim))  # 5 hidden states
                    self.transition_probs = nn.Parameter(torch.randn(5, 5))
                    self.emission_model = nn.Linear(input_dim, 5)
                
                self.output_layer = nn.Linear(input_dim, input_dim)
            
            def forward(self, x):
                if self.processing_type == 'state_space':
                    # Simple state space model
                    state = torch.zeros_like(x[:, 0:1, :])
                    outputs = []
                    for t in range(x.shape[1]):
                        state = self.state_transition(state)
                        observation = self.observation_model(x[:, t:t+1, :])
                        state = state + observation
                        outputs.append(state)
                    return torch.cat(outputs, dim=1)
                elif self.processing_type == 'kalman_filter':
                    # Simplified Kalman filter
                    predictions = self.prediction_model(x)
                    noise = self.noise_model(x)
                    filtered = predictions + noise
                    return self.output_layer(filtered)
                elif self.processing_type == 'hmm':
                    # Hidden Markov Model
                    emission_probs = torch.softmax(self.emission_model(x), dim=-1)
                    state_weights = torch.softmax(self.transition_probs, dim=-1)
                    
                    # Weighted combination of hidden states
                    hidden_outputs = []
                    for t in range(x.shape[1]):
                        weighted_states = torch.matmul(emission_probs[:, t, :], self.hidden_states)
                        hidden_outputs.append(weighted_states.unsqueeze(1))
                    
                    return torch.cat(hidden_outputs, dim=1)
                else:
                    return self.output_layer(x)
        
        return GeneratedTemporalProcessingModule(input_shape[-1], temporal_pattern, processing_type)
    
    def _create_attention_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create attention module"""
        attention_type = params.get('attention_type', 'multi_head')
        num_heads = params.get('num_heads', 8)
        
        class GeneratedAttentionModule(nn.Module):
            def __init__(self, input_dim, attention_type, num_heads):
                super().__init__()
                self.attention_type = attention_type
                
                if attention_type == 'multi_head':
                    self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
                elif attention_type == 'sparse':
                    self.sparse_attention = nn.Linear(input_dim, input_dim)
                    self.sparsity_gate = nn.Linear(input_dim, 1)
                elif attention_type == 'local':
                    self.local_window = 5
                    self.local_attention = nn.Conv1d(input_dim, input_dim, kernel_size=self.local_window, padding=self.local_window//2)
                elif attention_type == 'hierarchical':
                    self.level1_attention = nn.MultiheadAttention(input_dim, num_heads//2, batch_first=True)
                    self.level2_attention = nn.MultiheadAttention(input_dim, num_heads//2, batch_first=True)
                    self.combine = nn.Linear(input_dim * 2, input_dim)
            
            def forward(self, x):
                if self.attention_type == 'multi_head':
                    attended, _ = self.attention(x, x, x)
                    return attended
                elif self.attention_type == 'sparse':
                    # Sparse attention with gating
                    attended = self.sparse_attention(x)
                    sparsity_weights = torch.sigmoid(self.sparsity_gate(x))
                    return attended * sparsity_weights
                elif self.attention_type == 'local':
                    # Local attention using convolution
                    x_transposed = x.transpose(1, 2)
                    local_attended = self.local_attention(x_transposed)
                    return local_attended.transpose(1, 2)
                elif self.attention_type == 'hierarchical':
                    # Hierarchical attention
                    level1, _ = self.level1_attention(x, x, x)
                    level2, _ = self.level2_attention(level1, level1, level1)
                    combined = torch.cat([level1, level2], dim=-1)
                    return self.combine(combined)
                else:
                    return x
        
        return GeneratedAttentionModule(input_shape[-1], attention_type, num_heads)
    
    def _create_regularization_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create regularization module"""
        regularization_type = params.get('regularization_type', 'adaptive')
        
        class GeneratedRegularizationModule(nn.Module):
            def __init__(self, input_dim, regularization_type):
                super().__init__()
                self.regularization_type = regularization_type
                
                if regularization_type == 'adaptive':
                    self.adaptation_network = nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, input_dim)
                    )
                elif regularization_type == 'structured':
                    self.structure_weights = nn.Parameter(torch.randn(input_dim, input_dim))
                elif regularization_type == 'group':
                    self.group_size = 4
                    self.group_weights = nn.Parameter(torch.randn(input_dim // self.group_size))
                
                self.output_layer = nn.Linear(input_dim, input_dim)
            
            def forward(self, x):
                if self.regularization_type == 'adaptive':
                    # Adaptive regularization
                    adaptation = torch.sigmoid(self.adaptation_network(x))
                    regularized = x * adaptation
                    return self.output_layer(regularized)
                elif self.regularization_type == 'structured':
                    # Structured regularization
                    structured = torch.matmul(x, self.structure_weights)
                    return self.output_layer(structured)
                elif self.regularization_type == 'group':
                    # Group regularization
                    batch_size, seq_len, features = x.shape
                    grouped = x.view(batch_size, seq_len, -1, self.group_size)
                    group_weights = self.group_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    weighted = grouped * group_weights
                    return self.output_layer(weighted.view(batch_size, seq_len, features))
                else:
                    return self.output_layer(x)
        
        return GeneratedRegularizationModule(input_shape[-1], regularization_type)
    
    # Additional factory methods (simplified for brevity)
    def _create_encoding_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create encoding module"""
        class SimpleEncodingModule(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(input_dim * 2, input_dim)
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        return SimpleEncodingModule(input_shape[-1])
    
    def _create_prediction_head_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create prediction head module"""
        output_size = params.get('output_size', 1)
        
        class PredictionHeadModule(nn.Module):
            def __init__(self, input_dim, output_size):
                super().__init__()
                self.head = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(input_dim // 2, output_size)
                )
            
            def forward(self, x):
                if len(x.shape) == 3:
                    x = x[:, -1, :]  # Take last time step
                return self.head(x)
        
        return PredictionHeadModule(input_shape[-1], output_size)
    
    def _create_sequence_model_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create sequence model module"""
        class SequenceModelModule(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
                self.norm = nn.LayerNorm(input_dim)
            
            def forward(self, x):
                output, _ = self.lstm(x)
                return self.norm(output)
        
        return SequenceModelModule(input_shape[-1])
    
    def _create_activation_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create activation module"""
        class ActivationModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.GELU()
            
            def forward(self, x):
                return self.activation(x)
        
        return ActivationModule()
    
    def _create_aggregation_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create aggregation module"""
        class AggregationModule(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.aggregator = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(input_dim, input_dim)
                )
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                x_transposed = x.transpose(1, 2)  # (batch, features, seq_len)
                aggregated = self.aggregator(x_transposed)
                return aggregated.unsqueeze(1).expand(-1, x.shape[1], -1)
        
        return AggregationModule(input_shape[-1])
    
    def _create_cross_sectional_module(self, input_shape: Tuple[int, ...], params: Dict[str, Any], **kwargs) -> nn.Module:
        """Create cross-sectional module"""
        class CrossSectionalModule(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.cross_transform = nn.Linear(input_dim, input_dim)
            
            def forward(self, x):
                return self.cross_transform(x)
        
        return CrossSectionalModule(input_shape[-1])


def create_block_factory() -> BlockFactory:
    """Factory function to create block factory"""
    return BlockFactory()