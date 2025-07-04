from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class PCABlock(DomainBlock):
    """主成分分析ブロック - 次元削減と特徴抽出"""
    
    def __init__(self):
        super().__init__(
            name="pca",
            category="feature_extraction",
            description="Principal Component Analysis for dimensionality reduction and feature extraction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_components = kwargs.get('n_components', min(input_shape[-1], 32))
        whiten = kwargs.get('whiten', False)
        
        class PCAModule(nn.Module):
            def __init__(self, input_dim, n_components, whiten):
                super().__init__()
                self.input_dim = input_dim
                self.n_components = n_components
                self.whiten = whiten
                
                # Learnable PCA components
                self.components = nn.Parameter(torch.randn(input_dim, n_components))
                self.mean = nn.Parameter(torch.zeros(input_dim))
                
                if whiten:
                    self.explained_variance = nn.Parameter(torch.ones(n_components))
                
                # Reconstruction layer
                self.reconstruction = nn.Linear(n_components, input_dim, bias=False)
                
                # Initialize with orthogonal components
                nn.init.orthogonal_(self.components)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Center the data
                x_centered = x - self.mean.unsqueeze(0).unsqueeze(0)
                
                # Project to principal components
                # Reshape for matrix multiplication
                x_flat = x_centered.view(-1, features)  # (batch*seq_len, features)
                components = self.components / torch.norm(self.components, dim=0, keepdim=True)
                
                # Project
                projected = torch.matmul(x_flat, components)  # (batch*seq_len, n_components)
                
                # Apply whitening if enabled
                if self.whiten:
                    projected = projected / torch.sqrt(torch.clamp(self.explained_variance, min=1e-8))
                
                # Reconstruct
                reconstructed = self.reconstruction(projected)  # (batch*seq_len, features)
                
                # Reshape back
                return reconstructed.view(batch_size, seq_len, features)
        
        return PCAModule(input_shape[-1], n_components, whiten)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape  # Reconstruction maintains shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_components': [8, 16, 32, 64],
            'whiten': [True, False]
        }


class FourierFeatureBlock(DomainBlock):
    """フーリエ特徴抽出ブロック - 周波数領域解析"""
    
    def __init__(self):
        super().__init__(
            name="fourier_features",
            category="feature_extraction",
            description="Extract Fourier features for frequency domain analysis"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_frequencies = kwargs.get('n_frequencies', 16)
        include_phase = kwargs.get('include_phase', True)
        normalize = kwargs.get('normalize', True)
        
        class FourierFeatureModule(nn.Module):
            def __init__(self, n_frequencies, include_phase, normalize):
                super().__init__()
                self.n_frequencies = n_frequencies
                self.include_phase = include_phase
                self.normalize = normalize
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Apply FFT along time dimension
                fft_x = torch.fft.fft(x.transpose(1, 2))  # (batch, features, seq_len)
                
                # Extract magnitude and phase
                magnitude = torch.abs(fft_x)
                
                # Take top frequencies
                top_freqs_mag = magnitude[:, :, :self.n_frequencies]  # (batch, features, n_freq)
                
                freq_features = [top_freqs_mag]
                
                if self.include_phase:
                    phase = torch.angle(fft_x)
                    top_freqs_phase = phase[:, :, :self.n_frequencies]
                    freq_features.append(top_freqs_phase)
                
                # Concatenate frequency features
                freq_features = torch.cat(freq_features, dim=-1)  # (batch, features, total_freq_features)
                
                # Normalize if requested
                if self.normalize:
                    freq_features = F.normalize(freq_features, dim=-1)
                
                # Flatten and concatenate with original
                freq_flat = freq_features.flatten(start_dim=1)  # (batch, features * total_freq_features)
                x_flat = x.flatten(start_dim=1)  # (batch, seq_len * features)
                
                return torch.cat([x_flat, freq_flat], dim=-1)  # (batch, combined_features)
        
        return FourierFeatureModule(n_frequencies, include_phase, normalize)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_frequencies = kwargs.get('n_frequencies', 16)
        include_phase = kwargs.get('include_phase', True)
        
        batch_size = input_shape[0]
        seq_len, features = input_shape[1], input_shape[2]
        
        # Original features flattened
        original_flat = seq_len * features
        
        # Frequency features
        freq_multiplier = 2 if include_phase else 1
        freq_features = features * n_frequencies * freq_multiplier
        
        total_features = original_flat + freq_features
        return (batch_size, total_features)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_frequencies': [8, 16, 32, 64],
            'include_phase': [True, False],
            'normalize': [True, False]
        }


class WaveletFeatureBlock(DomainBlock):
    """ウェーブレット特徴抽出ブロック - 時間-周波数解析"""
    
    def __init__(self):
        super().__init__(
            name="wavelet_features",
            category="feature_extraction",
            description="Wavelet transform for time-frequency analysis"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_scales = kwargs.get('n_scales', 8)
        wavelet_type = kwargs.get('wavelet_type', 'morlet')
        
        class WaveletFeatureModule(nn.Module):
            def __init__(self, n_scales, wavelet_type):
                super().__init__()
                self.n_scales = n_scales
                self.wavelet_type = wavelet_type
                
                # Create learnable wavelet parameters
                if wavelet_type == 'morlet':
                    self.omega0 = nn.Parameter(torch.tensor(6.0))
                    self.scales = nn.Parameter(torch.logspace(0, 2, n_scales))
                elif wavelet_type == 'mexican_hat':
                    self.scales = nn.Parameter(torch.logspace(0, 2, n_scales))
            
            def morlet_wavelet(self, t, scale):
                """Morlet wavelet"""
                omega0 = self.omega0
                normalized_t = t / scale
                return (1.0 / torch.sqrt(scale)) * torch.exp(1j * omega0 * normalized_t) * torch.exp(-0.5 * normalized_t**2)
            
            def mexican_hat_wavelet(self, t, scale):
                """Mexican hat wavelet"""
                normalized_t = t / scale
                return (1.0 / torch.sqrt(scale)) * (1 - normalized_t**2) * torch.exp(-0.5 * normalized_t**2)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Create time vector
                t = torch.arange(seq_len, dtype=x.dtype, device=x.device)
                t = t.unsqueeze(0) - seq_len // 2  # Center at zero
                
                wavelet_features = []
                
                for scale in self.scales:
                    if self.wavelet_type == 'morlet':
                        wavelet = self.morlet_wavelet(t, scale)
                        # Convolve with real part
                        conv_real = F.conv1d(
                            x.transpose(1, 2),  # (batch, features, seq_len)
                            wavelet.real.unsqueeze(0).unsqueeze(0).expand(features, 1, -1),
                            groups=features,
                            padding=seq_len//2
                        )[:, :, :seq_len]
                        
                        # Convolve with imaginary part
                        conv_imag = F.conv1d(
                            x.transpose(1, 2),
                            wavelet.imag.unsqueeze(0).unsqueeze(0).expand(features, 1, -1),
                            groups=features,
                            padding=seq_len//2
                        )[:, :, :seq_len]
                        
                        # Magnitude
                        magnitude = torch.sqrt(conv_real**2 + conv_imag**2)
                        wavelet_features.append(magnitude)
                        
                    elif self.wavelet_type == 'mexican_hat':
                        wavelet = self.mexican_hat_wavelet(t, scale)
                        conv = F.conv1d(
                            x.transpose(1, 2),
                            wavelet.unsqueeze(0).unsqueeze(0).expand(features, 1, -1),
                            groups=features,
                            padding=seq_len//2
                        )[:, :, :seq_len]
                        wavelet_features.append(conv)
                
                # Concatenate all scales
                wavelet_features = torch.cat(wavelet_features, dim=1)  # (batch, features*n_scales, seq_len)
                
                # Reshape to match expected output
                return wavelet_features.transpose(1, 2)  # (batch, seq_len, features*n_scales)
        
        return WaveletFeatureModule(n_scales, wavelet_type)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_scales = kwargs.get('n_scales', 8)
        return (input_shape[0], input_shape[1], input_shape[2] * n_scales)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_scales': [4, 8, 16],
            'wavelet_type': ['morlet', 'mexican_hat']
        }


class StatisticalMomentsBlock(DomainBlock):
    """統計モーメント特徴抽出ブロック - 統計的特徴量を抽出"""
    
    def __init__(self):
        super().__init__(
            name="statistical_moments",
            category="feature_extraction",
            description="Extract statistical moments (mean, variance, skewness, kurtosis)"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        window_size = kwargs.get('window_size', 20)
        moments = kwargs.get('moments', ['mean', 'var', 'skew', 'kurt'])
        
        class StatisticalMomentsModule(nn.Module):
            def __init__(self, window_size, moments):
                super().__init__()
                self.window_size = window_size
                self.moments = moments
                self.n_moments = len(moments)
            
            def rolling_moments(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Pad for rolling window
                x_padded = F.pad(x, (0, 0, self.window_size-1, 0))
                
                moment_features = []
                
                for i in range(seq_len):
                    window = x_padded[:, i:i+self.window_size, :]  # (batch, window_size, features)
                    
                    window_moments = []
                    
                    if 'mean' in self.moments:
                        mean = torch.mean(window, dim=1)  # (batch, features)
                        window_moments.append(mean)
                    
                    if 'var' in self.moments:
                        var = torch.var(window, dim=1, unbiased=False)
                        window_moments.append(var)
                    
                    if 'skew' in self.moments:
                        mean = torch.mean(window, dim=1, keepdim=True)
                        centered = window - mean
                        m2 = torch.mean(centered**2, dim=1)
                        m3 = torch.mean(centered**3, dim=1)
                        skew = m3 / (torch.clamp(m2, min=1e-8)**(3/2))
                        window_moments.append(skew)
                    
                    if 'kurt' in self.moments:
                        mean = torch.mean(window, dim=1, keepdim=True)
                        centered = window - mean
                        m2 = torch.mean(centered**2, dim=1)
                        m4 = torch.mean(centered**4, dim=1)
                        kurt = m4 / torch.clamp(m2**2, min=1e-8) - 3
                        window_moments.append(kurt)
                    
                    if window_moments:
                        moment_features.append(torch.stack(window_moments, dim=-1))  # (batch, features, n_moments)
                
                return torch.stack(moment_features, dim=1)  # (batch, seq_len, features, n_moments)
            
            def forward(self, x):
                moments = self.rolling_moments(x)  # (batch, seq_len, features, n_moments)
                # Flatten the moments dimension
                batch_size, seq_len, features, n_moments = moments.shape
                return moments.view(batch_size, seq_len, features * n_moments)
        
        return StatisticalMomentsModule(window_size, moments)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        moments = kwargs.get('moments', ['mean', 'var', 'skew', 'kurt'])
        n_moments = len(moments)
        return (input_shape[0], input_shape[1], input_shape[2] * n_moments)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'window_size': [10, 20, 30, 50],
            'moments': [
                ['mean', 'var'],
                ['mean', 'var', 'skew'],
                ['mean', 'var', 'skew', 'kurt'],
                ['var', 'skew']
            ]
        }


class AutoEncoderFeatureBlock(DomainBlock):
    """オートエンコーダ特徴抽出ブロック - 非線形次元削減"""
    
    def __init__(self):
        super().__init__(
            name="autoencoder_features",
            category="feature_extraction",
            description="Autoencoder-based feature extraction with nonlinear dimensionality reduction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dim = kwargs.get('hidden_dim', input_shape[-1] // 2)
        activation = kwargs.get('activation', 'relu')
        dropout = kwargs.get('dropout', 0.1)
        
        class AutoEncoderFeatureModule(nn.Module):
            def __init__(self, input_dim, hidden_dim, activation, dropout):
                super().__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    self._get_activation(activation)
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    self._get_activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, input_dim)
                )
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'tanh':
                    return nn.Tanh()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Flatten for processing
                x_flat = x.view(-1, features)  # (batch*seq_len, features)
                
                # Encode
                encoded = self.encoder(x_flat)  # (batch*seq_len, hidden_dim)
                
                # Decode
                decoded = self.decoder(encoded)  # (batch*seq_len, features)
                
                # Reshape back
                return decoded.view(batch_size, seq_len, features)
        
        return AutoEncoderFeatureModule(input_shape[-1], hidden_dim, activation, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape  # Reconstruction maintains shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_dim': [16, 32, 64, 128],
            'activation': ['relu', 'gelu', 'tanh', 'leaky_relu'],
            'dropout': [0.0, 0.1, 0.2]
        }


class PolynomialFeatureBlock(DomainBlock):
    """多項式特徴抽出ブロック - 非線形特徴量を生成"""
    
    def __init__(self):
        super().__init__(
            name="polynomial_features",
            category="feature_extraction",
            description="Generate polynomial features for nonlinear relationships"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        degree = kwargs.get('degree', 2)
        interaction_only = kwargs.get('interaction_only', False)
        include_bias = kwargs.get('include_bias', False)
        
        class PolynomialFeatureModule(nn.Module):
            def __init__(self, degree, interaction_only, include_bias):
                super().__init__()
                self.degree = degree
                self.interaction_only = interaction_only
                self.include_bias = include_bias
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                poly_features = [x]  # Start with original features
                
                if self.include_bias:
                    bias = torch.ones_like(x[:, :, :1])  # (batch, seq_len, 1)
                    poly_features.append(bias)
                
                # Generate polynomial features
                for d in range(2, self.degree + 1):
                    if self.interaction_only:
                        # Only interaction terms, no pure powers
                        if d == 2:
                            # Pairwise interactions
                            for i in range(features):
                                for j in range(i+1, features):
                                    interaction = x[:, :, i:i+1] * x[:, :, j:j+1]
                                    poly_features.append(interaction)
                    else:
                        # Include pure powers
                        power_features = torch.pow(x, d)
                        poly_features.append(power_features)
                
                return torch.cat(poly_features, dim=-1)
        
        return PolynomialFeatureModule(degree, interaction_only, include_bias)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        degree = kwargs.get('degree', 2)
        interaction_only = kwargs.get('interaction_only', False)
        include_bias = kwargs.get('include_bias', False)
        
        features = input_shape[-1]
        output_features = features  # Original features
        
        if include_bias:
            output_features += 1
        
        if interaction_only:
            if degree >= 2:
                # Pairwise interactions: C(n,2)
                output_features += features * (features - 1) // 2
        else:
            # All polynomial terms up to degree
            for d in range(2, degree + 1):
                output_features += features  # Pure powers
        
        return (input_shape[0], input_shape[1], output_features)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'degree': [2, 3, 4],
            'interaction_only': [True, False],
            'include_bias': [True, False]
        }