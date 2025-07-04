from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class LinearEncodingBlock(DomainBlock):
    """線形エンコーディングブロック - シンプルな線形変換"""
    
    def __init__(self):
        super().__init__(
            name="linear_encoding",
            category="encoding",
            description="Simple linear encoding with optional activation"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        output_dim = kwargs.get('output_dim', input_shape[-1])
        activation = kwargs.get('activation', 'none')
        dropout = kwargs.get('dropout', 0.0)
        bias = kwargs.get('bias', True)
        
        class LinearEncodingModule(nn.Module):
            def __init__(self, input_dim, output_dim, activation, dropout, bias):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim, bias=bias)
                self.activation = self._get_activation(activation)
                self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'tanh':
                    return nn.Tanh()
                elif activation == 'sigmoid':
                    return nn.Sigmoid()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU(0.2)
                elif activation == 'swish':
                    return nn.SiLU()
                else:
                    return nn.Identity()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                x = self.linear(x)
                x = self.activation(x)
                x = self.dropout(x)
                return x
        
        return LinearEncodingModule(input_shape[-1], output_dim, activation, dropout, bias)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        output_dim = kwargs.get('output_dim', input_shape[-1])
        return input_shape[:-1] + (output_dim,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'output_dim': [32, 64, 128, 256, 512],
            'activation': ['none', 'relu', 'gelu', 'tanh', 'swish'],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'bias': [True, False]
        }


class MLPEncodingBlock(DomainBlock):
    """多層パーセプトロンエンコーディングブロック"""
    
    def __init__(self):
        super().__init__(
            name="mlp_encoding",
            category="encoding",
            description="Multi-layer perceptron encoding with configurable depth"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dims = kwargs.get('hidden_dims', [128, 64])
        output_dim = kwargs.get('output_dim', input_shape[-1])
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', 0.1)
        batch_norm = kwargs.get('batch_norm', True)
        residual = kwargs.get('residual', False)
        
        class MLPEncodingModule(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout, batch_norm, residual):
                super().__init__()
                self.residual = residual and (input_dim == output_dim)
                
                layers = []
                prev_dim = input_dim
                
                # Hidden layers
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    
                    if batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dim))
                    
                    layers.append(self._get_activation(activation))
                    
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, output_dim))
                
                self.mlp = nn.Sequential(*layers)
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU(0.2)
                elif activation == 'elu':
                    return nn.ELU()
                elif activation == 'swish':
                    return nn.SiLU()
                elif activation == 'mish':
                    return nn.Mish()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Reshape for BatchNorm if needed
                x_flat = x.view(-1, features)  # (batch*seq_len, features)
                out_flat = self.mlp(x_flat)
                
                # Reshape back
                output_dim = out_flat.shape[-1]
                out = out_flat.view(batch_size, seq_len, output_dim)
                
                # Residual connection
                if self.residual:
                    out = out + x
                
                return out
        
        return MLPEncodingModule(input_shape[-1], hidden_dims, output_dim, activation, dropout, batch_norm, residual)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        output_dim = kwargs.get('output_dim', input_shape[-1])
        return input_shape[:-1] + (output_dim,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_dims': [[64], [128], [128, 64], [256, 128], [512, 256, 128]],
            'output_dim': [32, 64, 128, 256],
            'activation': ['relu', 'gelu', 'leaky_relu', 'elu', 'swish', 'mish'],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'batch_norm': [True, False],
            'residual': [True, False]
        }


class ConvolutionalEncodingBlock(DomainBlock):
    """畳み込みエンコーディングブロック - 1D CNNで時間パターンを抽出"""
    
    def __init__(self):
        super().__init__(
            name="convolutional_encoding",
            category="encoding",
            description="1D convolutional encoding for temporal pattern extraction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        out_channels = kwargs.get('out_channels', [32, 64])
        kernel_sizes = kwargs.get('kernel_sizes', [3, 5])
        stride = kwargs.get('stride', 1)
        dilation = kwargs.get('dilation', 1)
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', 0.1)
        pooling = kwargs.get('pooling', 'none')  # 'max', 'avg', 'none'
        
        class ConvolutionalEncodingModule(nn.Module):
            def __init__(self, input_dim, out_channels, kernel_sizes, stride, dilation, activation, dropout, pooling):
                super().__init__()
                self.pooling = pooling
                
                # Ensure kernel_sizes and out_channels have same length
                if len(kernel_sizes) != len(out_channels):
                    kernel_sizes = [kernel_sizes[0]] * len(out_channels)
                
                self.conv_layers = nn.ModuleList()
                in_channels = input_dim
                
                for out_ch, kernel_size in zip(out_channels, kernel_sizes):
                    padding = (kernel_size - 1) // 2 * dilation
                    
                    conv_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_ch,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation
                        ),
                        nn.BatchNorm1d(out_ch),
                        self._get_activation(activation),
                        nn.Dropout(dropout)
                    )
                    
                    self.conv_layers.append(conv_layer)
                    in_channels = out_ch
                
                # Pooling layer
                if pooling == 'max':
                    self.pool = nn.AdaptiveMaxPool1d(1)
                elif pooling == 'avg':
                    self.pool = nn.AdaptiveAvgPool1d(1)
                else:
                    self.pool = None
                
                self.final_dim = in_channels
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU(0.2)
                elif activation == 'elu':
                    return nn.ELU()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                # Conv1d expects (batch, features, seq_len)
                x = x.transpose(1, 2)  # (batch, features, seq_len)
                
                # Apply convolutional layers
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)
                
                # Apply pooling if specified
                if self.pool is not None:
                    x = self.pool(x)  # (batch, final_dim, 1)
                    return x.squeeze(-1)  # (batch, final_dim)
                else:
                    # Transpose back
                    return x.transpose(1, 2)  # (batch, seq_len, final_dim)
        
        return ConvolutionalEncodingModule(
            input_shape[-1], out_channels, kernel_sizes, stride, dilation, activation, dropout, pooling
        )
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        out_channels = kwargs.get('out_channels', [32, 64])
        pooling = kwargs.get('pooling', 'none')
        
        final_channels = out_channels[-1]
        
        if pooling in ['max', 'avg']:
            return (input_shape[0], final_channels)
        else:
            return (input_shape[0], input_shape[1], final_channels)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'out_channels': [[32], [64], [32, 64], [64, 128], [32, 64, 128]],
            'kernel_sizes': [[3], [5], [3, 5], [3, 5, 7], [7, 5, 3]],
            'stride': [1, 2],
            'dilation': [1, 2],
            'activation': ['relu', 'gelu', 'leaky_relu', 'elu'],
            'dropout': [0.0, 0.1, 0.2],
            'pooling': ['none', 'max', 'avg']
        }


class TransformerEncodingBlock(DomainBlock):
    """トランスフォーマーエンコーディングブロック"""
    
    def __init__(self):
        super().__init__(
            name="transformer_encoding",
            category="encoding",
            description="Transformer encoder for sequence encoding with self-attention"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        d_model = kwargs.get('d_model', input_shape[-1])
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 2)
        dim_feedforward = kwargs.get('dim_feedforward', d_model * 4)
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'relu')
        
        class TransformerEncodingModule(nn.Module):
            def __init__(self, seq_len, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, activation):
                super().__init__()
                
                # Input projection if needed
                if input_dim != d_model:
                    self.input_proj = nn.Linear(input_dim, d_model)
                else:
                    self.input_proj = nn.Identity()
                
                # Positional encoding
                self.pos_encoder = nn.Parameter(torch.randn(seq_len, d_model))
                
                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True
                )
                
                self.transformer_encoder = nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=num_layers
                )
                
                self.norm = nn.LayerNorm(d_model)
                self.d_model = d_model
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Project to d_model
                x = self.input_proj(x)
                
                # Add positional encoding
                pos_enc = self.pos_encoder[:seq_len, :].unsqueeze(0)
                x = x + pos_enc
                
                # Apply transformer encoder
                x = self.transformer_encoder(x)
                
                # Final normalization
                x = self.norm(x)
                
                return x
        
        return TransformerEncodingModule(
            input_shape[1], input_shape[-1], d_model, nhead, num_layers, dim_feedforward, dropout, activation
        )
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        d_model = kwargs.get('d_model', input_shape[-1])
        return input_shape[:-1] + (d_model,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'd_model': [64, 128, 256, 512],
            'nhead': [4, 8, 16],
            'num_layers': [1, 2, 3, 4],
            'dim_feedforward': [256, 512, 1024, 2048],
            'dropout': [0.0, 0.1, 0.2],
            'activation': ['relu', 'gelu']
        }


class ResidualEncodingBlock(DomainBlock):
    """残差接続エンコーディングブロック - ResNetスタイルのエンコーディング"""
    
    def __init__(self):
        super().__init__(
            name="residual_encoding",
            category="encoding",
            description="Residual encoding with skip connections for deep networks"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dim = kwargs.get('hidden_dim', input_shape[-1])
        num_blocks = kwargs.get('num_blocks', 2)
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', 0.1)
        
        class ResidualBlock(nn.Module):
            def __init__(self, dim, activation, dropout):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim)
                self.fc2 = nn.Linear(dim, dim)
                self.norm1 = nn.LayerNorm(dim)
                self.norm2 = nn.LayerNorm(dim)
                self.activation = self._get_activation(activation)
                self.dropout = nn.Dropout(dropout)
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'swish':
                    return nn.SiLU()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                residual = x
                
                x = self.norm1(x)
                x = self.fc1(x)
                x = self.activation(x)
                x = self.dropout(x)
                
                x = self.norm2(x)
                x = self.fc2(x)
                x = self.dropout(x)
                
                return x + residual
        
        class ResidualEncodingModule(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_blocks, activation, dropout):
                super().__init__()
                
                # Input projection if needed
                if input_dim != hidden_dim:
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                else:
                    self.input_proj = nn.Identity()
                
                # Residual blocks
                self.blocks = nn.ModuleList([
                    ResidualBlock(hidden_dim, activation, dropout)
                    for _ in range(num_blocks)
                ])
                
                self.final_norm = nn.LayerNorm(hidden_dim)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                x = self.input_proj(x)
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.final_norm(x)
                return x
        
        return ResidualEncodingModule(input_shape[-1], hidden_dim, num_blocks, activation, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        hidden_dim = kwargs.get('hidden_dim', input_shape[-1])
        return input_shape[:-1] + (hidden_dim,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_dim': [64, 128, 256, 512],
            'num_blocks': [1, 2, 3, 4],
            'activation': ['relu', 'gelu', 'swish'],
            'dropout': [0.0, 0.1, 0.2]
        }


class VariationalEncodingBlock(DomainBlock):
    """変分エンコーディングブロック - VAEスタイルの確率的エンコーディング"""
    
    def __init__(self):
        super().__init__(
            name="variational_encoding",
            category="encoding",
            description="Variational encoding with latent space regularization"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        latent_dim = kwargs.get('latent_dim', input_shape[-1] // 2)
        beta = kwargs.get('beta', 1.0)  # KL divergence weight
        activation = kwargs.get('activation', 'relu')
        
        class VariationalEncodingModule(nn.Module):
            def __init__(self, input_dim, latent_dim, beta, activation):
                super().__init__()
                self.latent_dim = latent_dim
                self.beta = beta
                
                # Encoder to latent parameters
                self.fc_mu = nn.Linear(input_dim, latent_dim)
                self.fc_logvar = nn.Linear(input_dim, latent_dim)
                
                # Decoder from latent
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, input_dim),
                    self._get_activation(activation)
                )
                
                self.training_loss = 0.0
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'tanh':
                    return nn.Tanh()
                else:
                    return nn.ReLU()
            
            def reparameterize(self, mu, logvar):
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return mu + eps * std
                else:
                    return mu
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Flatten for encoding
                x_flat = x.view(-1, features)
                
                # Encode to latent parameters
                mu = self.fc_mu(x_flat)
                logvar = self.fc_logvar(x_flat)
                
                # Sample from latent distribution
                z = self.reparameterize(mu, logvar)
                
                # Decode
                x_recon_flat = self.decoder(z)
                
                # Reshape back
                x_recon = x_recon_flat.view(batch_size, seq_len, features)
                
                # Compute KL divergence loss (for training)
                if self.training:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kl_loss = kl_loss / (batch_size * seq_len)  # Normalize
                    self.training_loss = self.beta * kl_loss
                
                return x_recon
        
        return VariationalEncodingModule(input_shape[-1], latent_dim, beta, activation)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape  # Reconstruction maintains input shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'latent_dim': [16, 32, 64, 128],
            'beta': [0.1, 0.5, 1.0, 2.0],
            'activation': ['relu', 'gelu', 'tanh']
        }