from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np


class DomainBlock(ABC):
    """Abstract base class for domain blocks."""
    
    def __init__(self, name: str, category: str, description: str):
        self.name = name
        self.category = category
        self.description = description
        self.input_shape = None
        self.output_shape = None
        
    @abstractmethod
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        """Create the PyTorch module for this domain block."""
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        """Calculate the output shape given input shape."""
        pass
    
    def validate_input_shape(self, input_shape: Tuple[int, ...]) -> bool:
        """Validate if the input shape is compatible with this block."""
        return len(input_shape) >= 2  # At least batch and feature dimensions
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for this block."""
        return {}


# Normalization Blocks
class BatchNormBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="batch_norm",
            category="normalization",
            description="Batch normalization for stabilizing training"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        num_features = input_shape[-1]
        return nn.BatchNorm1d(num_features)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape


class LayerNormBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="layer_norm",
            category="normalization",
            description="Layer normalization for transformer-like architectures"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        normalized_shape = input_shape[-1]
        return nn.LayerNorm(normalized_shape)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape


class AdaptiveInstanceNormBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="adaptive_instance_norm",
            category="normalization",
            description="Adaptive instance normalization with learnable parameters"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        num_features = input_shape[-1]
        return nn.InstanceNorm1d(num_features, affine=True)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape


class DemeanBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="demean",
            category="normalization",
            description="Subtract mean from input features"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        class DemeanModule(nn.Module):
            def forward(self, x):
                return x - x.mean(dim=-1, keepdim=True)
        return DemeanModule()
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape


# Feature Extraction Blocks
class PCABlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="pca",
            category="feature_extraction",
            description="Principal Component Analysis for dimensionality reduction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_components = kwargs.get('n_components', min(input_shape[-1], 32))
        
        class PCAModule(nn.Module):
            def __init__(self, input_dim, n_components):
                super().__init__()
                self.projection = nn.Linear(input_dim, n_components, bias=False)
                self.reconstruction = nn.Linear(n_components, input_dim, bias=False)
            
            def forward(self, x):
                # Project to principal components
                components = self.projection(x)
                # Reconstruct
                return self.reconstruction(components)
        
        return PCAModule(input_shape[-1], n_components)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'n_components': [8, 16, 32, 64]}


class FourierFeatureBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="fourier_features",
            category="feature_extraction",
            description="Extract Fourier features for frequency domain analysis"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_frequencies = kwargs.get('n_frequencies', 16)
        
        class FourierFeatureModule(nn.Module):
            def __init__(self, n_frequencies):
                super().__init__()
                self.n_frequencies = n_frequencies
            
            def forward(self, x):
                # Apply FFT and take top frequencies
                fft_x = torch.fft.fft(x, dim=-2)  # FFT along time dimension
                fft_magnitude = torch.abs(fft_x)
                
                # Take top frequencies
                top_freqs = fft_magnitude[:, :self.n_frequencies, :]
                
                # Flatten and concatenate with original
                freq_features = top_freqs.flatten(start_dim=1)
                return torch.cat([x.flatten(start_dim=1), freq_features], dim=-1)
        
        return FourierFeatureModule(n_frequencies)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_frequencies = kwargs.get('n_frequencies', 16)
        batch_size = input_shape[0]
        original_features = np.prod(input_shape[1:])
        freq_features = n_frequencies * input_shape[-1]
        return (batch_size, original_features + freq_features)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'n_frequencies': [8, 16, 32]}


# Mixing Blocks
class TimeMixingBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="time_mixing",
            category="mixing",
            description="Mix information across time dimension"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        mixing_type = kwargs.get('mixing_type', 'linear')
        
        if mixing_type == 'linear':
            class TimeMixingModule(nn.Module):
                def __init__(self, seq_len):
                    super().__init__()
                    self.mixing_weights = nn.Parameter(torch.randn(seq_len, seq_len))
                
                def forward(self, x):
                    # x: (batch, seq_len, features)
                    return torch.matmul(self.mixing_weights, x)
            
            return TimeMixingModule(input_shape[1])
        
        elif mixing_type == 'conv':
            return nn.Conv1d(input_shape[-1], input_shape[-1], kernel_size=3, padding=1)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'mixing_type': ['linear', 'conv']}


class ChannelMixingBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="channel_mixing",
            category="mixing",
            description="Mix information across feature channels"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dim = kwargs.get('hidden_dim', input_shape[-1] * 2)
        
        class ChannelMixingModule(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, input_dim)
                self.activation = nn.GELU()
            
            def forward(self, x):
                return self.fc2(self.activation(self.fc1(x)))
        
        return ChannelMixingModule(input_shape[-1], hidden_dim)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'hidden_dim': [64, 128, 256]}


# Financial Domain Blocks
class MultiTimeFrameBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="multi_time_frame",
            category="financial_domain",
            description="Extract features from multiple time frames"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7, 9])
        
        class MultiTimeFrameModule(nn.Module):
            def __init__(self, input_dim, kernel_sizes):
                super().__init__()
                self.convs = nn.ModuleList([
                    nn.Conv1d(input_dim, input_dim, kernel_size=k, padding=k//2)
                    for k in kernel_sizes
                ])
                self.combine = nn.Linear(input_dim * len(kernel_sizes), input_dim)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                x = x.transpose(1, 2)  # (batch, features, seq_len)
                
                conv_outputs = []
                for conv in self.convs:
                    conv_outputs.append(conv(x))
                
                # Concatenate and combine
                combined = torch.cat(conv_outputs, dim=1)
                combined = combined.transpose(1, 2)  # Back to (batch, seq_len, features)
                
                return self.combine(combined)
        
        return MultiTimeFrameModule(input_shape[-1], kernel_sizes)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'kernel_sizes': [
                [3, 5, 7],
                [3, 5, 7, 9],
                [2, 4, 8, 16]
            ]
        }


class LeadLagBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="lead_lag",
            category="financial_domain",
            description="Extract lead-lag relationships in time series"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        max_lag = kwargs.get('max_lag', 5)
        
        class LeadLagModule(nn.Module):
            def __init__(self, input_dim, max_lag):
                super().__init__()
                self.max_lag = max_lag
                self.lag_weights = nn.Parameter(torch.randn(max_lag * 2 + 1, input_dim))
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Create shifted versions
                shifted_x = []
                for lag in range(-self.max_lag, self.max_lag + 1):
                    if lag == 0:
                        shifted_x.append(x)
                    elif lag > 0:
                        # Lead: pad at beginning
                        padded = F.pad(x, (0, 0, lag, 0))
                        shifted_x.append(padded[:, :seq_len, :])
                    else:
                        # Lag: pad at end
                        padded = F.pad(x, (0, 0, 0, -lag))
                        shifted_x.append(padded[:, -seq_len:, :])
                
                # Weighted combination
                stacked = torch.stack(shifted_x, dim=0)  # (lags, batch, seq, features)
                weighted = torch.einsum('lbsf,lf->bsf', stacked, self.lag_weights)
                
                return weighted
        
        return LeadLagModule(input_shape[-1], max_lag)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'max_lag': [3, 5, 7, 10]}


class RegimeDetectionBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="regime_detection",
            category="financial_domain",
            description="Detect market regimes using clustering"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_regimes = kwargs.get('n_regimes', 3)
        
        class RegimeDetectionModule(nn.Module):
            def __init__(self, input_dim, n_regimes):
                super().__init__()
                self.n_regimes = n_regimes
                self.regime_centers = nn.Parameter(torch.randn(n_regimes, input_dim))
                self.regime_weights = nn.Parameter(torch.ones(n_regimes))
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Calculate distances to regime centers
                x_expanded = x.unsqueeze(2)  # (batch, seq, 1, features)
                centers_expanded = self.regime_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, regimes, features)
                
                distances = torch.norm(x_expanded - centers_expanded, dim=-1)  # (batch, seq, regimes)
                regime_probs = F.softmax(-distances, dim=-1)  # (batch, seq, regimes)
                
                # Weight by regime importance
                weighted_probs = regime_probs * self.regime_weights.unsqueeze(0).unsqueeze(0)
                
                # Combine with original features
                return torch.cat([x, weighted_probs], dim=-1)
        
        return RegimeDetectionModule(input_shape[-1], n_regimes)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_regimes = kwargs.get('n_regimes', 3)
        return input_shape[:-1] + (input_shape[-1] + n_regimes,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {'n_regimes': [2, 3, 4, 5]}


# Sequence Models
class LSTMBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="lstm",
            category="sequence_models",
            description="Long Short-Term Memory for sequence processing"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_size = kwargs.get('hidden_size', input_shape[-1])
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.1)
        
        class LSTMModule(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.layer_norm = nn.LayerNorm(hidden_size)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                return self.layer_norm(lstm_out)
        
        return LSTMModule(input_shape[-1], hidden_size, num_layers, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        hidden_size = kwargs.get('hidden_size', input_shape[-1])
        return input_shape[:-1] + (hidden_size,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3]
        }


class TransformerBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="transformer",
            category="sequence_models",
            description="Transformer encoder for sequence modeling"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        d_model = kwargs.get('d_model', input_shape[-1])
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.1)
        
        class TransformerModule(nn.Module):
            def __init__(self, d_model, nhead, num_layers, dropout):
                super().__init__()
                self.d_model = d_model
                self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = TransformerEncoder(encoder_layer, num_layers)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                seq_len = x.shape[1]
                
                # Add positional encoding
                x = x + self.pos_encoder[:seq_len, :].unsqueeze(0)
                
                # Apply transformer
                return self.transformer(x)
        
        return TransformerModule(d_model, nhead, num_layers, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        d_model = kwargs.get('d_model', input_shape[-1])
        return input_shape[:-1] + (d_model,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'd_model': [128, 256, 512],
            'nhead': [4, 8, 16],
            'num_layers': [2, 4, 6],
            'dropout': [0.1, 0.2]
        }


# Prediction Head Blocks
class RegressionHeadBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="regression_head",
            category="prediction_heads",
            description="Regression head for continuous prediction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        output_size = kwargs.get('output_size', 1)
        dropout = kwargs.get('dropout', 0.1)
        
        class RegressionHeadModule(nn.Module):
            def __init__(self, input_size, output_size, dropout):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(input_size, output_size)
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    x = x[:, -1, :]  # Take last time step
                
                x = self.dropout(x)
                return self.fc(x)
        
        return RegressionHeadModule(input_shape[-1], output_size, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        output_size = kwargs.get('output_size', 1)
        return (input_shape[0], output_size)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'output_size': [1, 3],  # 1 for regression, 3 for classification
            'dropout': [0.1, 0.2, 0.3]
        }


class ClassificationHeadBlock(DomainBlock):
    def __init__(self):
        super().__init__(
            name="classification_head",
            category="prediction_heads",
            description="Classification head for discrete prediction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        num_classes = kwargs.get('num_classes', 3)
        dropout = kwargs.get('dropout', 0.1)
        
        class ClassificationHeadModule(nn.Module):
            def __init__(self, input_size, num_classes, dropout):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(input_size, num_classes)
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    x = x[:, -1, :]  # Take last time step
                
                x = self.dropout(x)
                return self.fc(x)
        
        return ClassificationHeadModule(input_shape[-1], num_classes, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        num_classes = kwargs.get('num_classes', 3)
        return (input_shape[0], num_classes)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'num_classes': [3, 5, 7],
            'dropout': [0.1, 0.2, 0.3]
        }


# Domain Block Registry
class DomainBlockRegistry:
    """Registry for managing domain blocks."""
    
    def __init__(self):
        self._blocks = {}
        self._register_default_blocks()
    
    def _register_default_blocks(self):
        """Register all default domain blocks."""
        blocks = [
            # Normalization
            BatchNormBlock(),
            LayerNormBlock(),
            AdaptiveInstanceNormBlock(),
            DemeanBlock(),
            
            # Feature Extraction
            PCABlock(),
            FourierFeatureBlock(),
            
            # Mixing
            TimeMixingBlock(),
            ChannelMixingBlock(),
            
            # Financial Domain
            MultiTimeFrameBlock(),
            LeadLagBlock(),
            RegimeDetectionBlock(),
            
            # Sequence Models
            LSTMBlock(),
            TransformerBlock(),
            
            # Prediction Heads
            RegressionHeadBlock(),
            ClassificationHeadBlock(),
        ]
        
        for block in blocks:
            self.register_block(block)
    
    def register_block(self, block: DomainBlock):
        """Register a domain block."""
        self._blocks[block.name] = block
    
    def get_block(self, name: str) -> DomainBlock:
        """Get a domain block by name."""
        if name not in self._blocks:
            raise ValueError(f"Block '{name}' not found in registry")
        return self._blocks[name]
    
    def get_blocks_by_category(self, category: str) -> List[DomainBlock]:
        """Get all blocks in a specific category."""
        return [block for block in self._blocks.values() if block.category == category]
    
    def get_all_blocks(self) -> List[DomainBlock]:
        """Get all registered blocks."""
        return list(self._blocks.values())
    
    def get_block_names(self) -> List[str]:
        """Get all registered block names."""
        return list(self._blocks.keys())
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(set(block.category for block in self._blocks.values()))


# Global registry instance
registry = DomainBlockRegistry()


def get_domain_block_registry() -> DomainBlockRegistry:
    """Get the global domain block registry."""
    return registry