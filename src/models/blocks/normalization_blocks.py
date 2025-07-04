from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class BatchNormBlock(DomainBlock):
    """バッチ正規化ブロック - 訓練の安定化のため"""
    
    def __init__(self):
        super().__init__(
            name="batch_norm",
            category="normalization",
            description="Batch normalization for stabilizing training and reducing internal covariate shift"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        affine = kwargs.get('affine', True)
        track_running_stats = kwargs.get('track_running_stats', True)
        eps = kwargs.get('eps', 1e-5)
        momentum = kwargs.get('momentum', 0.1)
        
        class BatchNormModule(nn.Module):
            def __init__(self, num_features, affine, track_running_stats, eps, momentum):
                super().__init__()
                self.batch_norm = nn.BatchNorm1d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats
                )
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                # Reshape for BatchNorm1d: (batch * seq_len, features)
                x_reshaped = x.view(-1, features)
                x_normed = self.batch_norm(x_reshaped)
                return x_normed.view(batch_size, seq_len, features)
        
        num_features = input_shape[-1]
        return BatchNormModule(num_features, affine, track_running_stats, eps, momentum)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'affine': [True, False],
            'track_running_stats': [True, False],
            'eps': [1e-5, 1e-4, 1e-6],
            'momentum': [0.1, 0.01, 0.05, 0.2]
        }


class LayerNormBlock(DomainBlock):
    """レイヤー正規化ブロック - Transformer系アーキテクチャ用"""
    
    def __init__(self):
        super().__init__(
            name="layer_norm",
            category="normalization",
            description="Layer normalization for transformer-like architectures and stable training"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        eps = kwargs.get('eps', 1e-5)
        elementwise_affine = kwargs.get('elementwise_affine', True)
        
        normalized_shape = input_shape[-1]
        return nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'eps': [1e-5, 1e-4, 1e-6],
            'elementwise_affine': [True, False]
        }


class AdaptiveInstanceNormBlock(DomainBlock):
    """適応的インスタンス正規化ブロック - 学習可能なパラメータ付き"""
    
    def __init__(self):
        super().__init__(
            name="adaptive_instance_norm",
            category="normalization",
            description="Adaptive instance normalization with learnable parameters for style adaptation"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        eps = kwargs.get('eps', 1e-5)
        momentum = kwargs.get('momentum', 0.1)
        affine = kwargs.get('affine', True)
        track_running_stats = kwargs.get('track_running_stats', False)
        
        class AdaptiveInstanceNormModule(nn.Module):
            def __init__(self, num_features, eps, momentum, affine, track_running_stats):
                super().__init__()
                self.instance_norm = nn.InstanceNorm1d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track_running_stats
                )
                # Adaptive scaling parameters
                if affine:
                    self.adaptive_weight = nn.Parameter(torch.ones(num_features))
                    self.adaptive_bias = nn.Parameter(torch.zeros(num_features))
                else:
                    self.adaptive_weight = None
                    self.adaptive_bias = None
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Reshape for InstanceNorm1d: (batch, features, seq_len)
                x_transposed = x.transpose(1, 2)
                x_normed = self.instance_norm(x_transposed)
                
                # Apply adaptive parameters if available
                if self.adaptive_weight is not None:
                    x_normed = x_normed * self.adaptive_weight.unsqueeze(0).unsqueeze(-1)
                if self.adaptive_bias is not None:
                    x_normed = x_normed + self.adaptive_bias.unsqueeze(0).unsqueeze(-1)
                
                # Transpose back: (batch, seq_len, features)
                return x_normed.transpose(1, 2)
        
        num_features = input_shape[-1]
        return AdaptiveInstanceNormModule(num_features, eps, momentum, affine, track_running_stats)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'eps': [1e-5, 1e-4, 1e-6],
            'momentum': [0.1, 0.01, 0.05],
            'affine': [True, False],
            'track_running_stats': [True, False]
        }


class DemeanBlock(DomainBlock):
    """平均減算ブロック - 入力特徴量から平均を引く"""
    
    def __init__(self):
        super().__init__(
            name="demean",
            category="normalization",
            description="Subtract mean from input features to center the data"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        dim = kwargs.get('dim', -1)  # Which dimension to compute mean over
        keepdim = kwargs.get('keepdim', True)
        
        class DemeanModule(nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim
            
            def forward(self, x):
                mean = x.mean(dim=self.dim, keepdim=self.keepdim)
                return x - mean
        
        return DemeanModule(dim, keepdim)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'dim': [-1, -2, (-2, -1)],  # Feature dim, time dim, or both
            'keepdim': [True, False]
        }


class GroupNormBlock(DomainBlock):
    """グループ正規化ブロック - 特徴量をグループに分けて正規化"""
    
    def __init__(self):
        super().__init__(
            name="group_norm",
            category="normalization",
            description="Group normalization for better performance with small batch sizes"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        num_groups = kwargs.get('num_groups', 8)
        eps = kwargs.get('eps', 1e-5)
        affine = kwargs.get('affine', True)
        
        class GroupNormModule(nn.Module):
            def __init__(self, num_channels, num_groups, eps, affine):
                super().__init__()
                self.group_norm = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=num_channels,
                    eps=eps,
                    affine=affine
                )
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                # GroupNorm expects (batch, channels, *)
                batch_size, seq_len, features = x.shape
                
                # Reshape to (batch, features, seq_len)
                x_transposed = x.transpose(1, 2)
                x_normed = self.group_norm(x_transposed)
                
                # Transpose back
                return x_normed.transpose(1, 2)
        
        num_channels = input_shape[-1]
        # Ensure num_groups divides num_channels
        if num_channels % num_groups != 0:
            num_groups = min(num_groups, num_channels)
            for i in range(num_groups, 0, -1):
                if num_channels % i == 0:
                    num_groups = i
                    break
        
        return GroupNormModule(num_channels, num_groups, eps, affine)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'num_groups': [2, 4, 8, 16],
            'eps': [1e-5, 1e-4],
            'affine': [True, False]
        }


class RMSNormBlock(DomainBlock):
    """Root Mean Square正規化ブロック - より効率的な正規化手法"""
    
    def __init__(self):
        super().__init__(
            name="rms_norm",
            category="normalization",
            description="Root Mean Square normalization for efficient and effective normalization"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        eps = kwargs.get('eps', 1e-8)
        
        class RMSNormModule(nn.Module):
            def __init__(self, dim, eps):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                # Compute RMS normalization
                rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
                return self.weight * x / rms
        
        dim = input_shape[-1]
        return RMSNormModule(dim, eps)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'eps': [1e-8, 1e-6, 1e-5]
        }


class PowerNormBlock(DomainBlock):
    """Power正規化ブロック - べき乗ベースの正規化"""
    
    def __init__(self):
        super().__init__(
            name="power_norm",
            category="normalization",
            description="Power normalization using learnable power parameter"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        power = kwargs.get('power', 0.5)
        eps = kwargs.get('eps', 1e-8)
        learnable_power = kwargs.get('learnable_power', False)
        
        class PowerNormModule(nn.Module):
            def __init__(self, power, eps, learnable_power):
                super().__init__()
                self.eps = eps
                if learnable_power:
                    self.power = nn.Parameter(torch.tensor(power))
                else:
                    self.register_buffer('power', torch.tensor(power))
            
            def forward(self, x):
                # Compute power norm
                abs_x = torch.abs(x) + self.eps
                sign_x = torch.sign(x)
                
                # Apply power transformation
                power_norm = torch.pow(abs_x, self.power)
                
                # Restore sign
                return sign_x * power_norm
        
        return PowerNormModule(power, eps, learnable_power)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'power': [0.5, 0.25, 0.75, 1.0],
            'eps': [1e-8, 1e-6],
            'learnable_power': [True, False]
        }


class QuantileNormBlock(DomainBlock):
    """分位点正規化ブロック - 外れ値に対してロバストな正規化"""
    
    def __init__(self):
        super().__init__(
            name="quantile_norm",
            category="normalization",
            description="Quantile-based normalization robust to outliers"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        quantile_low = kwargs.get('quantile_low', 0.25)
        quantile_high = kwargs.get('quantile_high', 0.75)
        
        class QuantileNormModule(nn.Module):
            def __init__(self, quantile_low, quantile_high):
                super().__init__()
                self.quantile_low = quantile_low
                self.quantile_high = quantile_high
            
            def forward(self, x):
                # Compute quantiles along feature dimension
                q_low = torch.quantile(x, self.quantile_low, dim=-1, keepdim=True)
                q_high = torch.quantile(x, self.quantile_high, dim=-1, keepdim=True)
                
                # Normalize using interquartile range
                iqr = q_high - q_low
                iqr = torch.clamp(iqr, min=1e-8)  # Avoid division by zero
                
                return (x - q_low) / iqr
        
        return QuantileNormModule(quantile_low, quantile_high)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'quantile_low': [0.1, 0.25, 0.05],
            'quantile_high': [0.75, 0.9, 0.95]
        }