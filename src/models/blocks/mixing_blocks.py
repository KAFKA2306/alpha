from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class TimeMixingBlock(DomainBlock):
    """時間次元ミキシングブロック - 時間方向で情報を混合"""
    
    def __init__(self):
        super().__init__(
            name="time_mixing",
            category="mixing",
            description="Mix information across time dimension for temporal feature interaction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        mixing_type = kwargs.get('mixing_type', 'linear')
        dropout = kwargs.get('dropout', 0.1)
        
        class TimeMixingModule(nn.Module):
            def __init__(self, seq_len, mixing_type, dropout):
                super().__init__()
                self.seq_len = seq_len
                self.mixing_type = mixing_type
                self.dropout = nn.Dropout(dropout)
                
                if mixing_type == 'linear':
                    self.mixing_weights = nn.Parameter(torch.randn(seq_len, seq_len))
                    # Initialize as identity + small random
                    nn.init.eye_(self.mixing_weights)
                    self.mixing_weights.data += 0.1 * torch.randn_like(self.mixing_weights)
                    
                elif mixing_type == 'conv':
                    kernel_size = min(7, seq_len // 4)
                    self.conv1d = nn.Conv1d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=kernel_size,
                        padding=kernel_size//2,
                        bias=True
                    )
                    
                elif mixing_type == 'causal_conv':
                    kernel_size = min(7, seq_len // 4)
                    self.causal_conv = nn.Conv1d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=kernel_size,
                        padding=kernel_size-1,
                        bias=True
                    )
                    
                elif mixing_type == 'moving_average':
                    window_size = kwargs.get('window_size', 5)
                    self.window_size = min(window_size, seq_len)
                    self.weights = nn.Parameter(torch.ones(self.window_size) / self.window_size)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                if self.mixing_type == 'linear':
                    # Apply learned linear mixing across time
                    mixed = torch.matmul(self.mixing_weights[:seq_len, :seq_len], x)
                    return self.dropout(mixed)
                    
                elif self.mixing_type == 'conv':
                    # Apply 1D convolution across time for each feature
                    output = []
                    for f in range(features):
                        feature_data = x[:, :, f:f+1].transpose(1, 2)  # (batch, 1, seq_len)
                        conv_out = self.conv1d(feature_data).transpose(1, 2)  # (batch, seq_len, 1)
                        output.append(conv_out)
                    mixed = torch.cat(output, dim=-1)
                    return self.dropout(mixed)
                    
                elif self.mixing_type == 'causal_conv':
                    # Causal convolution (no future information)
                    output = []
                    for f in range(features):
                        feature_data = x[:, :, f:f+1].transpose(1, 2)
                        conv_out = self.causal_conv(feature_data)
                        # Remove padding from the end
                        conv_out = conv_out[:, :, :seq_len].transpose(1, 2)
                        output.append(conv_out)
                    mixed = torch.cat(output, dim=-1)
                    return self.dropout(mixed)
                    
                elif self.mixing_type == 'moving_average':
                    # Learnable moving average
                    padded_x = F.pad(x, (0, 0, self.window_size-1, 0))
                    output = []
                    for i in range(seq_len):
                        window = padded_x[:, i:i+self.window_size, :]
                        weighted = torch.sum(window * self.weights.view(1, -1, 1), dim=1)
                        output.append(weighted)
                    mixed = torch.stack(output, dim=1)
                    return self.dropout(mixed)
                
                else:
                    return x
        
        return TimeMixingModule(input_shape[1], mixing_type, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'mixing_type': ['linear', 'conv', 'causal_conv', 'moving_average'],
            'dropout': [0.0, 0.1, 0.2],
            'window_size': [3, 5, 7, 9]  # For moving_average
        }


class ChannelMixingBlock(DomainBlock):
    """チャンネルミキシングブロック - 特徴量次元で情報を混合"""
    
    def __init__(self):
        super().__init__(
            name="channel_mixing",
            category="mixing",
            description="Mix information across feature channels for feature interaction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dim = kwargs.get('hidden_dim', input_shape[-1] * 2)
        activation = kwargs.get('activation', 'gelu')
        dropout = kwargs.get('dropout', 0.1)
        mixing_type = kwargs.get('mixing_type', 'mlp')
        
        class ChannelMixingModule(nn.Module):
            def __init__(self, input_dim, hidden_dim, activation, dropout, mixing_type):
                super().__init__()
                self.mixing_type = mixing_type
                self.dropout = nn.Dropout(dropout)
                
                if mixing_type == 'mlp':
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, input_dim)
                    self.activation = self._get_activation(activation)
                    
                elif mixing_type == 'gated':
                    self.gate_fc = nn.Linear(input_dim, input_dim)
                    self.value_fc = nn.Linear(input_dim, input_dim)
                    
                elif mixing_type == 'expert':
                    num_experts = kwargs.get('num_experts', 4)
                    self.num_experts = num_experts
                    self.experts = nn.ModuleList([
                        nn.Linear(input_dim, input_dim) for _ in range(num_experts)
                    ])
                    self.gate = nn.Linear(input_dim, num_experts)
                    
                elif mixing_type == 'factorized':
                    rank = kwargs.get('rank', min(input_dim // 4, 64))
                    self.factor1 = nn.Linear(input_dim, rank, bias=False)
                    self.factor2 = nn.Linear(rank, input_dim, bias=False)
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'swish':
                    return nn.SiLU()
                elif activation == 'mish':
                    return nn.Mish()
                else:
                    return nn.GELU()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                
                if self.mixing_type == 'mlp':
                    # Standard MLP mixing
                    mixed = self.fc2(self.dropout(self.activation(self.fc1(x))))
                    return mixed + x  # Residual connection
                    
                elif self.mixing_type == 'gated':
                    # Gated mixing
                    gate = torch.sigmoid(self.gate_fc(x))
                    value = self.value_fc(x)
                    return gate * value + (1 - gate) * x
                    
                elif self.mixing_type == 'expert':
                    # Mixture of experts
                    gate_weights = F.softmax(self.gate(x), dim=-1)  # (batch, seq_len, num_experts)
                    
                    expert_outputs = []
                    for expert in self.experts:
                        expert_out = expert(x)  # (batch, seq_len, features)
                        expert_outputs.append(expert_out)
                    
                    expert_stack = torch.stack(expert_outputs, dim=-1)  # (batch, seq_len, features, num_experts)
                    mixed = torch.sum(expert_stack * gate_weights.unsqueeze(-2), dim=-1)
                    return mixed + x
                    
                elif self.mixing_type == 'factorized':
                    # Low-rank factorized mixing
                    compressed = self.factor1(x)
                    mixed = self.factor2(compressed)
                    return mixed + x
                
                else:
                    return x
        
        return ChannelMixingModule(input_shape[-1], hidden_dim, activation, dropout, mixing_type)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_dim': [64, 128, 256, 512],
            'activation': ['relu', 'gelu', 'swish', 'mish'],
            'dropout': [0.0, 0.1, 0.2],
            'mixing_type': ['mlp', 'gated', 'expert', 'factorized'],
            'num_experts': [2, 4, 8],  # For expert mixing
            'rank': [16, 32, 64]  # For factorized mixing
        }


class CrossAttentionMixingBlock(DomainBlock):
    """クロスアテンションミキシングブロック - 時間と特徴量のクロスアテンション"""
    
    def __init__(self):
        super().__init__(
            name="cross_attention_mixing",
            category="mixing",
            description="Cross-attention mixing between time and feature dimensions"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        d_model = kwargs.get('d_model', input_shape[-1])
        nhead = kwargs.get('nhead', 8)
        dropout = kwargs.get('dropout', 0.1)
        
        class CrossAttentionMixingModule(nn.Module):
            def __init__(self, seq_len, features, d_model, nhead, dropout):
                super().__init__()
                self.seq_len = seq_len
                self.features = features
                self.d_model = d_model
                
                # Project to d_model if necessary
                if features != d_model:
                    self.input_proj = nn.Linear(features, d_model)
                    self.output_proj = nn.Linear(d_model, features)
                else:
                    self.input_proj = nn.Identity()
                    self.output_proj = nn.Identity()
                
                # Time-to-feature attention
                self.time_to_feature_attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                
                # Feature-to-time attention
                self.feature_to_time_attn = nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                
                # Layer norms
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                
                # Feed forward
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Project to d_model
                x_proj = self.input_proj(x)  # (batch, seq_len, d_model)
                
                # Time-to-feature cross attention
                # Transpose for feature-wise attention
                x_transposed = x_proj.transpose(1, 2)  # (batch, features, d_model)
                
                # Query: features, Key&Value: time
                attn_out1, _ = self.time_to_feature_attn(
                    query=x_transposed,
                    key=x_proj,
                    value=x_proj
                )
                x_transposed = self.norm1(x_transposed + attn_out1)
                
                # Transpose back
                x_proj = x_transposed.transpose(1, 2)  # (batch, seq_len, d_model)
                
                # Feature-to-time cross attention
                # Query: time, Key&Value: features (transposed)
                attn_out2, _ = self.feature_to_time_attn(
                    query=x_proj,
                    key=x_transposed,
                    value=x_transposed
                )
                x_proj = self.norm2(x_proj + attn_out2)
                
                # Feed forward
                ff_out = self.ff(x_proj)
                x_proj = x_proj + ff_out
                
                # Project back to original dimension
                output = self.output_proj(x_proj)
                
                return output
        
        return CrossAttentionMixingModule(input_shape[1], input_shape[2], d_model, nhead, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'd_model': [64, 128, 256],
            'nhead': [4, 8, 16],
            'dropout': [0.0, 0.1, 0.2]
        }


class GatedMixingBlock(DomainBlock):
    """ゲーテッドミキシングブロック - ゲートメカニズムで情報を選択的に混合"""
    
    def __init__(self):
        super().__init__(
            name="gated_mixing",
            category="mixing",
            description="Gated mixing with learned gates for selective information flow"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        gate_type = kwargs.get('gate_type', 'sigmoid')
        mixing_mode = kwargs.get('mixing_mode', 'both')  # 'time', 'channel', 'both'
        dropout = kwargs.get('dropout', 0.1)
        
        class GatedMixingModule(nn.Module):
            def __init__(self, seq_len, features, gate_type, mixing_mode, dropout):
                super().__init__()
                self.gate_type = gate_type
                self.mixing_mode = mixing_mode
                self.dropout = nn.Dropout(dropout)
                
                if mixing_mode in ['time', 'both']:
                    self.time_gate = nn.Linear(features, features)
                    self.time_value = nn.Linear(features, features)
                
                if mixing_mode in ['channel', 'both']:
                    self.channel_gate = nn.Linear(features, features)
                    self.channel_value = nn.Linear(features, features)
                
                # Global gate for mixing modes
                if mixing_mode == 'both':
                    self.mode_gate = nn.Linear(features, 1)
            
            def apply_gate(self, gate_logits):
                if self.gate_type == 'sigmoid':
                    return torch.sigmoid(gate_logits)
                elif self.gate_type == 'tanh':
                    return torch.tanh(gate_logits)
                elif self.gate_type == 'glu':
                    # Gated Linear Unit
                    return torch.sigmoid(gate_logits)
                elif self.gate_type == 'swish':
                    return gate_logits * torch.sigmoid(gate_logits)
                else:
                    return torch.sigmoid(gate_logits)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                outputs = []
                
                if self.mixing_mode in ['time', 'both']:
                    # Time mixing with gates
                    time_gate = self.apply_gate(self.time_gate(x))
                    time_value = self.time_value(x)
                    time_mixed = time_gate * time_value + (1 - time_gate) * x
                    outputs.append(time_mixed)
                
                if self.mixing_mode in ['channel', 'both']:
                    # Channel mixing with gates
                    channel_gate = self.apply_gate(self.channel_gate(x))
                    channel_value = self.channel_value(x)
                    channel_mixed = channel_gate * channel_value + (1 - channel_gate) * x
                    outputs.append(channel_mixed)
                
                if len(outputs) == 1:
                    return self.dropout(outputs[0])
                elif len(outputs) == 2:
                    # Mix time and channel outputs
                    mode_gate = torch.sigmoid(self.mode_gate(x))  # (batch, seq_len, 1)
                    final_output = mode_gate * outputs[0] + (1 - mode_gate) * outputs[1]
                    return self.dropout(final_output)
                else:
                    return x
        
        return GatedMixingModule(input_shape[1], input_shape[2], gate_type, mixing_mode, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'gate_type': ['sigmoid', 'tanh', 'glu', 'swish'],
            'mixing_mode': ['time', 'channel', 'both'],
            'dropout': [0.0, 0.1, 0.2]
        }


class FourierMixingBlock(DomainBlock):
    """フーリエミキシングブロック - 周波数領域での情報混合"""
    
    def __init__(self):
        super().__init__(
            name="fourier_mixing",
            category="mixing",
            description="Fourier domain mixing for frequency-based feature interaction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        modes = kwargs.get('modes', 16)  # Number of Fourier modes
        dropout = kwargs.get('dropout', 0.1)
        
        class FourierMixingModule(nn.Module):
            def __init__(self, seq_len, features, modes, dropout):
                super().__init__()
                self.modes = min(modes, seq_len // 2)
                self.dropout = nn.Dropout(dropout)
                
                # Learnable Fourier weights
                self.fourier_weights = nn.Parameter(
                    torch.view_as_real(
                        torch.randn(features, features, self.modes, dtype=torch.cfloat)
                    )
                )
                
                self.norm = nn.LayerNorm(features)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Take FFT along time dimension
                x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, features)
                
                # Apply learnable Fourier mixing for low frequencies
                out_ft = torch.zeros_like(x_ft)
                
                # Convert Fourier weights to complex
                fourier_weights = torch.view_as_complex(self.fourier_weights)
                
                # Mix in Fourier domain
                for i in range(min(self.modes, x_ft.shape[1])):
                    # Apply frequency-specific mixing
                    out_ft[:, i, :] = torch.einsum('bf,ff->bf', x_ft[:, i, :], fourier_weights[:, :, i])
                
                # Copy high frequencies without mixing
                if x_ft.shape[1] > self.modes:
                    out_ft[:, self.modes:, :] = x_ft[:, self.modes:, :]
                
                # Inverse FFT
                x_mixed = torch.fft.irfft(out_ft, n=seq_len, dim=1)
                
                # Add residual and normalize
                x_mixed = self.norm(x_mixed + x)
                
                return self.dropout(x_mixed)
        
        return FourierMixingModule(input_shape[1], input_shape[2], modes, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'modes': [8, 16, 32, 64],
            'dropout': [0.0, 0.1, 0.2]
        }