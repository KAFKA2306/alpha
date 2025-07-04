from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class MultiTimeFrameBlock(DomainBlock):
    """マルチタイムフレームブロック - 複数の時間枠から特徴を抽出"""
    
    def __init__(self):
        super().__init__(
            name="multi_time_frame",
            category="financial_domain",
            description="Extract features from multiple time frames for multi-scale analysis"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7, 9, 21])  # Different time scales
        aggregation = kwargs.get('aggregation', 'concat')  # 'concat', 'attention', 'gated'
        normalize = kwargs.get('normalize', True)
        
        class MultiTimeFrameModule(nn.Module):
            def __init__(self, input_dim, kernel_sizes, aggregation, normalize):
                super().__init__()
                self.kernel_sizes = kernel_sizes
                self.aggregation = aggregation
                self.normalize = normalize
                
                # Convolutions for different time frames
                self.convs = nn.ModuleList()
                for k in kernel_sizes:
                    padding = k // 2
                    conv = nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=input_dim,
                        kernel_size=k,
                        padding=padding,
                        groups=1  # Share across features
                    )
                    self.convs.append(conv)
                
                # Aggregation layers
                if aggregation == 'attention':
                    self.attention = nn.MultiheadAttention(
                        embed_dim=input_dim,
                        num_heads=4,
                        batch_first=True
                    )
                elif aggregation == 'gated':
                    total_features = input_dim * len(kernel_sizes)
                    self.gate = nn.Sequential(
                        nn.Linear(total_features, total_features),
                        nn.Sigmoid()
                    )
                    self.combine = nn.Linear(total_features, input_dim)
                elif aggregation == 'concat':
                    total_features = input_dim * len(kernel_sizes)
                    self.combine = nn.Linear(total_features, input_dim)
                
                # Normalization
                if normalize:
                    self.norm = nn.LayerNorm(input_dim)
                else:
                    self.norm = nn.Identity()
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Transpose for conv1d: (batch, features, seq_len)
                x_conv = x.transpose(1, 2)
                
                # Apply convolutions for different time frames
                conv_outputs = []
                for conv in self.convs:
                    conv_out = conv(x_conv)  # (batch, features, seq_len)
                    conv_outputs.append(conv_out)
                
                if self.aggregation == 'attention':
                    # Use attention to combine different time frames
                    # Stack along feature dimension
                    stacked = torch.stack(conv_outputs, dim=1)  # (batch, num_scales, features, seq_len)
                    batch_size, num_scales, features, seq_len = stacked.shape
                    
                    # Reshape for attention
                    stacked_flat = stacked.view(batch_size, num_scales * features, seq_len)
                    stacked_flat = stacked_flat.transpose(1, 2)  # (batch, seq_len, num_scales * features)
                    
                    # Self-attention
                    attended, _ = self.attention(stacked_flat, stacked_flat, stacked_flat)
                    
                    # Average over time scales
                    attended = attended.view(batch_size, seq_len, num_scales, features)
                    output = attended.mean(dim=2)  # (batch, seq_len, features)
                    
                elif self.aggregation == 'gated':
                    # Gated combination
                    concatenated = torch.cat(conv_outputs, dim=1)  # (batch, total_features, seq_len)
                    concatenated = concatenated.transpose(1, 2)  # (batch, seq_len, total_features)
                    
                    # Apply gate
                    gate_weights = self.gate(concatenated)
                    gated = concatenated * gate_weights
                    
                    # Combine
                    output = self.combine(gated)
                    
                else:  # concat
                    # Simple concatenation and linear combination
                    concatenated = torch.cat(conv_outputs, dim=1)  # (batch, total_features, seq_len)
                    concatenated = concatenated.transpose(1, 2)  # (batch, seq_len, total_features)
                    output = self.combine(concatenated)
                
                # Normalize and add residual
                output = self.norm(output + x)
                
                return output
        
        return MultiTimeFrameModule(input_shape[-1], kernel_sizes, aggregation, normalize)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape  # Output shape same as input
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'kernel_sizes': [
                [3, 5, 7],
                [3, 5, 7, 9],
                [3, 5, 7, 9, 15],
                [5, 10, 20],
                [2, 4, 8, 16]
            ],
            'aggregation': ['concat', 'attention', 'gated'],
            'normalize': [True, False]
        }


class LeadLagBlock(DomainBlock):
    """リードラグブロック - 時系列の先行・遅行関係を抽出"""
    
    def __init__(self):
        super().__init__(
            name="lead_lag",
            category="financial_domain",
            description="Extract lead-lag relationships between time series"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        max_lag = kwargs.get('max_lag', 5)
        learnable_weights = kwargs.get('learnable_weights', True)
        causal_only = kwargs.get('causal_only', False)  # Only use past information
        
        class LeadLagModule(nn.Module):
            def __init__(self, input_dim, max_lag, learnable_weights, causal_only):
                super().__init__()
                self.max_lag = max_lag
                self.causal_only = causal_only
                
                if causal_only:
                    # Only lags (past information)
                    self.total_lags = max_lag
                    lag_range = range(1, max_lag + 1)
                else:
                    # Both leads and lags
                    self.total_lags = max_lag * 2 + 1
                    lag_range = range(-max_lag, max_lag + 1)
                
                self.lag_range = list(lag_range)
                
                if learnable_weights:
                    # Learnable weights for each lag
                    self.lag_weights = nn.Parameter(
                        torch.ones(len(self.lag_range), input_dim) / len(self.lag_range)
                    )
                    # Cross-series lead-lag weights
                    self.cross_weights = nn.Parameter(
                        torch.eye(input_dim).unsqueeze(0).repeat(len(self.lag_range), 1, 1) * 0.1
                    )
                else:
                    # Fixed exponential decay weights
                    weights = torch.exp(-torch.arange(len(self.lag_range)).float() * 0.1)
                    weights = weights / weights.sum()
                    self.register_buffer('lag_weights', weights.unsqueeze(1).repeat(1, input_dim))
                    
                    # No cross-series weights
                    self.cross_weights = None
                
                self.norm = nn.LayerNorm(input_dim)
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Create shifted versions
                shifted_series = []
                for lag in self.lag_range:
                    if lag == 0:
                        shifted = x
                    elif lag > 0:
                        # Lag (use past values)
                        shifted = F.pad(x, (0, 0, lag, 0))[:, :seq_len, :]
                    else:
                        # Lead (use future values) - only if not causal_only
                        shifted = F.pad(x, (0, 0, 0, -lag))[:, -seq_len:, :]
                    
                    shifted_series.append(shifted)
                
                # Stack all shifted series
                stacked = torch.stack(shifted_series, dim=0)  # (num_lags, batch, seq_len, features)
                
                # Apply lag weights
                weighted_series = []
                for i, shifted in enumerate(shifted_series):
                    if self.cross_weights is not None:
                        # Apply cross-series weights
                        cross_weighted = torch.matmul(shifted, self.cross_weights[i])  # (batch, seq_len, features)
                        lag_weighted = cross_weighted * self.lag_weights[i].unsqueeze(0).unsqueeze(0)
                    else:
                        lag_weighted = shifted * self.lag_weights[i].unsqueeze(0).unsqueeze(0)
                    
                    weighted_series.append(lag_weighted)
                
                # Combine weighted series
                combined = torch.sum(torch.stack(weighted_series, dim=0), dim=0)
                
                # Normalize and add residual
                output = self.norm(combined + x)
                
                return output
        
        return LeadLagModule(input_shape[-1], max_lag, learnable_weights, causal_only)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return input_shape
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'max_lag': [3, 5, 7, 10, 15],
            'learnable_weights': [True, False],
            'causal_only': [True, False]
        }


class RegimeDetectionBlock(DomainBlock):
    """レジーム検出ブロック - 市場レジームを検出し特徴量として使用"""
    
    def __init__(self):
        super().__init__(
            name="regime_detection",
            category="financial_domain",
            description="Detect market regimes using clustering-based approach"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_regimes = kwargs.get('n_regimes', 3)  # Bull, Bear, Sideways
        window_size = kwargs.get('window_size', 20)
        features_for_regime = kwargs.get('features_for_regime', ['volatility', 'trend', 'momentum'])
        
        class RegimeDetectionModule(nn.Module):
            def __init__(self, input_dim, n_regimes, window_size):
                super().__init__()
                self.n_regimes = n_regimes
                self.window_size = window_size
                
                # Regime centers (learnable)
                self.regime_centers = nn.Parameter(
                    torch.randn(n_regimes, input_dim)
                )
                
                # Regime transition probabilities
                self.transition_probs = nn.Parameter(
                    torch.eye(n_regimes) * 0.8 + torch.ones(n_regimes, n_regimes) * 0.2 / n_regimes
                )
                
                # Feature extraction for regime detection
                self.volatility_extractor = nn.Conv1d(1, 1, kernel_size=window_size, padding=window_size//2)
                self.trend_extractor = nn.Linear(window_size, 1)
                self.momentum_extractor = nn.Conv1d(1, 1, kernel_size=5, padding=2)
                
                # Regime classifier
                self.regime_classifier = nn.Sequential(
                    nn.Linear(input_dim + 3, n_regimes * 2),  # +3 for vol, trend, momentum
                    nn.ReLU(),
                    nn.Linear(n_regimes * 2, n_regimes),
                    nn.Softmax(dim=-1)
                )
                
                # Previous regime for transition smoothing
                self.register_buffer('prev_regime_probs', torch.ones(1, n_regimes) / n_regimes)
            
            def extract_market_features(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                market_features = []
                
                for i in range(features):
                    feature_series = x[:, :, i]  # (batch, seq_len)
                    
                    # Volatility (rolling std)
                    feature_padded = F.pad(feature_series.unsqueeze(1), (self.window_size//2, self.window_size//2))
                    volatility = self.volatility_extractor(feature_padded).squeeze(1)  # (batch, seq_len)
                    
                    # Trend (linear regression slope over window)
                    unfolded = F.unfold(
                        feature_series.unsqueeze(1).unsqueeze(1),
                        kernel_size=(1, self.window_size),
                        padding=(0, self.window_size//2)
                    )  # (batch, window_size, seq_len)
                    unfolded = unfolded.transpose(1, 2)  # (batch, seq_len, window_size)
                    trend = self.trend_extractor(unfolded).squeeze(-1)  # (batch, seq_len)
                    
                    # Momentum (rate of change)
                    momentum = self.momentum_extractor(feature_series.unsqueeze(1)).squeeze(1)
                    
                    # Combine features
                    combined = torch.stack([volatility, trend, momentum], dim=-1)  # (batch, seq_len, 3)
                    market_features.append(combined)
                
                # Average across all features
                market_features = torch.stack(market_features, dim=-1).mean(dim=-1)  # (batch, seq_len, 3)
                
                return market_features
            
            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size, seq_len, features = x.shape
                
                # Extract market regime features
                market_features = self.extract_market_features(x)  # (batch, seq_len, 3)
                
                # Combine with original features for regime detection
                combined_features = torch.cat([x, market_features], dim=-1)  # (batch, seq_len, features+3)
                
                # Detect regimes
                regime_probs = self.regime_classifier(combined_features)  # (batch, seq_len, n_regimes)
                
                # Apply transition smoothing (simple Markov model)
                if self.training:
                    # Update previous regime for next iteration
                    self.prev_regime_probs = regime_probs[:, -1:, :].detach().mean(dim=0, keepdim=True)
                
                # Soft regime assignment
                regime_weights = regime_probs.unsqueeze(-2)  # (batch, seq_len, 1, n_regimes)
                regime_centers_expanded = self.regime_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, n_regimes, features)
                
                # Weighted combination of regime centers
                regime_features = torch.sum(
                    regime_weights * regime_centers_expanded, dim=-2
                )  # (batch, seq_len, features)
                
                # Combine original features with regime features
                output = torch.cat([
                    x,
                    regime_features,
                    regime_probs  # Include regime probabilities as features
                ], dim=-1)
                
                return output
        
        return RegimeDetectionModule(input_shape[-1], n_regimes, window_size)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_regimes = kwargs.get('n_regimes', 3)
        # Original features + regime features + regime probabilities
        output_features = input_shape[-1] * 2 + n_regimes
        return input_shape[:-1] + (output_features,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_regimes': [2, 3, 4, 5],
            'window_size': [10, 20, 30, 50]
        }


class FactorExposureBlock(DomainBlock):
    """ファクターエクスポージャーブロック - リスクファクターへのエクスポージャーを算出"""
    
    def __init__(self):
        super().__init__(
            name="factor_exposure",
            category="financial_domain",
            description="Calculate exposure to common risk factors (market, size, value, etc.)"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        n_factors = kwargs.get('n_factors', 5)  # Market, Size, Value, Momentum, Quality
        rolling_window = kwargs.get('rolling_window', 60)
        factor_names = kwargs.get('factor_names', ['market', 'size', 'value', 'momentum', 'quality'])
        
        class FactorExposureModule(nn.Module):
            def __init__(self, input_dim, n_factors, rolling_window):
                super().__init__()
                self.n_factors = n_factors
                self.rolling_window = rolling_window
                
                # Factor loadings (learnable)
                self.factor_loadings = nn.Parameter(
                    torch.randn(input_dim, n_factors) * 0.1
                )
                
                # Factor returns (learnable time series)
                self.factor_returns = nn.Parameter(
                    torch.randn(rolling_window, n_factors) * 0.01
                )
                
                # Beta calculation network
                self.beta_network = nn.Sequential(
                    nn.Linear(rolling_window, rolling_window // 2),
                    nn.ReLU(),
                    nn.Linear(rolling_window // 2, n_factors),
                    nn.Tanh()  # Beta can be negative
                )
                
                # Residual calculation
                self.residual_network = nn.Linear(input_dim + n_factors, input_dim)
                
                # Exposure normalization
                self.exposure_norm = nn.LayerNorm(n_factors)
            
            def rolling_regression(self, returns, factor_returns):
                """
                Compute rolling factor exposures using simple linear regression
                """
                batch_size, seq_len, features = returns.shape
                
                # Pad returns for rolling window
                padded_returns = F.pad(returns, (0, 0, self.rolling_window-1, 0))
                
                exposures = []
                
                for t in range(seq_len):
                    # Get window of returns
                    window_returns = padded_returns[:, t:t+self.rolling_window, :]  # (batch, window, features)
                    
                    # Reshape for regression
                    window_flat = window_returns.view(batch_size, -1)  # (batch, window*features)
                    
                    # Compute exposures using neural network
                    exposure_t = self.beta_network(window_flat)  # (batch, n_factors)
                    exposures.append(exposure_t)
                
                return torch.stack(exposures, dim=1)  # (batch, seq_len, n_factors)
            
            def forward(self, x):
                # x: (batch, seq_len, features) - assume these are returns
                batch_size, seq_len, features = x.shape
                
                # Compute factor exposures over time
                exposures = self.rolling_regression(x, self.factor_returns)  # (batch, seq_len, n_factors)
                
                # Normalize exposures
                exposures = self.exposure_norm(exposures)
                
                # Compute factor-explained component
                factor_component = torch.matmul(
                    exposures.unsqueeze(-2),  # (batch, seq_len, 1, n_factors)
                    self.factor_loadings.T.unsqueeze(0).unsqueeze(0)  # (1, 1, n_factors, features)
                ).squeeze(-2)  # (batch, seq_len, features)
                
                # Compute residual (idiosyncratic) component
                combined_input = torch.cat([x, exposures], dim=-1)
                residual_component = self.residual_network(combined_input)
                
                # Output: original + exposures + factor component + residual
                output = torch.cat([
                    x,  # Original features
                    exposures,  # Factor exposures
                    factor_component,  # Factor-explained returns
                    residual_component  # Residual returns
                ], dim=-1)
                
                return output
        
        return FactorExposureModule(input_shape[-1], n_factors, rolling_window)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        n_factors = kwargs.get('n_factors', 5)
        # Original + exposures + factor component + residual
        output_features = input_shape[-1] * 3 + n_factors
        return input_shape[:-1] + (output_features,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'n_factors': [3, 5, 7, 10],
            'rolling_window': [30, 60, 120, 252]
        }


class VolatilityClusteringBlock(DomainBlock):
    """ボラティリティクラスタリングブロック - GARCHモデル風のボラティリティモデリング"""
    
    def __init__(self):
        super().__init__(
            name="volatility_clustering",
            category="financial_domain",
            description="Model volatility clustering using GARCH-like neural networks"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        garch_order = kwargs.get('garch_order', (1, 1))  # (p, q) for GARCH(p,q)
        vol_activation = kwargs.get('vol_activation', 'softplus')  # Ensure positive volatility
        
        class VolatilityClusteringModule(nn.Module):
            def __init__(self, input_dim, garch_order):
                super().__init__()
                self.p, self.q = garch_order
                self.input_dim = input_dim
                
                # GARCH parameters (learnable)
                self.alpha0 = nn.Parameter(torch.ones(input_dim) * 0.01)  # Constant
                self.alpha = nn.Parameter(torch.ones(self.q, input_dim) * 0.1)  # ARCH terms
                self.beta = nn.Parameter(torch.ones(self.p, input_dim) * 0.8)  # GARCH terms
                
                # Neural network for volatility prediction
                self.vol_network = nn.Sequential(
                    nn.Linear(input_dim * (self.p + self.q + 1), input_dim * 2),
                    nn.ReLU(),
                    nn.Linear(input_dim * 2, input_dim),
                    nn.Softplus()  # Ensure positive volatility
                )
                
                # Initialize volatility and squared returns history
                self.register_buffer('vol_history', torch.ones(self.p, input_dim) * 0.01)
                self.register_buffer('ret_sq_history', torch.ones(self.q, input_dim) * 0.01)
            
            def forward(self, x):
                # x: (batch, seq_len, features) - assume these are returns
                batch_size, seq_len, features = x.shape
                
                volatilities = []
                conditional_vars = []
                
                for t in range(seq_len):
                    current_returns = x[:, t, :]  # (batch, features)
                    
                    # GARCH equation: sigma^2_t = alpha0 + sum(alpha_i * r^2_{t-i}) + sum(beta_j * sigma^2_{t-j})
                    
                    # Constant term
                    vol_pred = self.alpha0.unsqueeze(0).repeat(batch_size, 1)
                    
                    # ARCH terms (lagged squared returns)
                    for i in range(self.q):
                        if t == 0:
                            # Use historical values for first timestep
                            lagged_sq_ret = self.ret_sq_history[i].unsqueeze(0).repeat(batch_size, 1)
                        else:
                            lag_idx = max(0, t - i - 1)
                            lagged_sq_ret = x[:, lag_idx, :] ** 2
                        
                        vol_pred += self.alpha[i] * lagged_sq_ret
                    
                    # GARCH terms (lagged conditional variances)
                    for j in range(self.p):
                        if t == 0:
                            # Use historical values for first timestep
                            lagged_vol = self.vol_history[j].unsqueeze(0).repeat(batch_size, 1)
                        else:
                            lag_idx = max(0, t - j - 1)
                            lagged_vol = volatilities[lag_idx] if lag_idx < len(volatilities) else vol_pred
                        
                        vol_pred += self.beta[j] * lagged_vol
                    
                    # Neural network enhancement
                    # Prepare input features
                    nn_input_list = [current_returns]
                    
                    # Add lagged squared returns
                    for i in range(self.q):
                        if t == 0:
                            lagged_sq_ret = self.ret_sq_history[i].unsqueeze(0).repeat(batch_size, 1)
                        else:
                            lag_idx = max(0, t - i - 1)
                            lagged_sq_ret = x[:, lag_idx, :] ** 2
                        nn_input_list.append(lagged_sq_ret)
                    
                    # Add lagged volatilities
                    for j in range(self.p):
                        if t == 0:
                            lagged_vol = self.vol_history[j].unsqueeze(0).repeat(batch_size, 1)
                        else:
                            lag_idx = max(0, t - j - 1)
                            lagged_vol = volatilities[lag_idx] if lag_idx < len(volatilities) else vol_pred
                        nn_input_list.append(lagged_vol)
                    
                    nn_input = torch.cat(nn_input_list, dim=-1)  # (batch, input_dim * (p+q+1))
                    nn_vol = self.vol_network(nn_input)  # (batch, input_dim)
                    
                    # Combine GARCH and neural network predictions
                    final_vol = 0.7 * vol_pred + 0.3 * nn_vol
                    final_vol = torch.clamp(final_vol, min=1e-6)  # Ensure positive
                    
                    volatilities.append(final_vol)
                    
                    # Conditional variance (volatility squared)
                    conditional_vars.append(final_vol)
                
                # Stack volatilities
                vol_tensor = torch.stack(volatilities, dim=1)  # (batch, seq_len, features)
                
                # Standardized residuals
                standardized_residuals = x / torch.sqrt(vol_tensor)
                
                # Log volatility (often more stable)
                log_vol = torch.log(vol_tensor + 1e-8)
                
                # Volatility regime (high/low)
                vol_regime = (vol_tensor > vol_tensor.mean(dim=1, keepdim=True)).float()
                
                # Combine all volatility-related features
                output = torch.cat([
                    x,  # Original returns
                    vol_tensor,  # Predicted volatility
                    log_vol,  # Log volatility
                    standardized_residuals,  # Standardized residuals
                    vol_regime  # Volatility regime
                ], dim=-1)
                
                return output
        
        return VolatilityClusteringModule(input_shape[-1], garch_order)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        # Original + volatility + log_vol + standardized + regime
        output_features = input_shape[-1] * 5
        return input_shape[:-1] + (output_features,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'garch_order': [(1, 1), (1, 2), (2, 1), (2, 2)],
            'vol_activation': ['softplus', 'elu', 'relu']
        }


class CrossSectionalBlock(DomainBlock):
    """クロスセクショナルブロック - 銘柄間の相対的関係をモデル化"""
    
    def __init__(self):
        super().__init__(
            name="cross_sectional",
            category="financial_domain",
            description="Model cross-sectional relationships between stocks"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        ranking_method = kwargs.get('ranking_method', 'percentile')  # 'percentile', 'zscore', 'decile'
        industry_groups = kwargs.get('industry_groups', 10)
        
        class CrossSectionalModule(nn.Module):
            def __init__(self, batch_size, features, ranking_method, industry_groups):
                super().__init__()
                self.ranking_method = ranking_method
                self.industry_groups = industry_groups
                self.batch_size = batch_size  # Number of stocks
                
                # Industry classification network
                self.industry_classifier = nn.Sequential(
                    nn.Linear(features, features * 2),
                    nn.ReLU(),
                    nn.Linear(features * 2, industry_groups),
                    nn.Softmax(dim=-1)
                )
                
                # Cross-sectional attention
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=features,
                    num_heads=4,
                    batch_first=True
                )
                
                # Ranking network
                self.ranking_network = nn.Sequential(
                    nn.Linear(features + industry_groups, features),
                    nn.ReLU(),
                    nn.Linear(features, 1),
                    nn.Tanh()  # Ranking score
                )
            
            def compute_cross_sectional_ranks(self, x):
                """
                Compute cross-sectional ranks for each time step
                """
                batch_size, seq_len, features = x.shape
                
                ranks = []
                
                for t in range(seq_len):
                    cross_section = x[:, t, :]  # (batch_size, features) - all stocks at time t
                    
                    if self.ranking_method == 'percentile':
                        # Percentile ranks
                        ranked = torch.zeros_like(cross_section)
                        for f in range(features):
                            feature_values = cross_section[:, f]
                            sorted_indices = torch.argsort(feature_values)
                            percentiles = torch.arange(batch_size, dtype=torch.float, device=x.device) / (batch_size - 1)
                            ranked[sorted_indices, f] = percentiles
                    
                    elif self.ranking_method == 'zscore':
                        # Z-score normalization
                        mean = cross_section.mean(dim=0, keepdim=True)
                        std = cross_section.std(dim=0, keepdim=True)
                        ranked = (cross_section - mean) / (std + 1e-8)
                    
                    elif self.ranking_method == 'decile':
                        # Decile ranks
                        ranked = torch.zeros_like(cross_section)
                        for f in range(features):
                            feature_values = cross_section[:, f]
                            quantiles = torch.quantile(feature_values, torch.arange(11, dtype=torch.float, device=x.device) / 10)
                            deciles = torch.searchsorted(quantiles, feature_values).float() / 10
                            ranked[:, f] = deciles
                    
                    ranks.append(ranked)
                
                return torch.stack(ranks, dim=1)  # (batch, seq_len, features)
            
            def forward(self, x):
                # x: (batch, seq_len, features) where batch represents different stocks
                batch_size, seq_len, features = x.shape
                
                # Compute cross-sectional ranks
                ranks = self.compute_cross_sectional_ranks(x)
                
                # Industry classification (based on current features)
                industry_probs = self.industry_classifier(x)  # (batch, seq_len, industry_groups)
                
                # Cross-sectional attention (stocks attending to other stocks)
                attended_features = []
                for t in range(seq_len):
                    cross_section = x[:, t, :].unsqueeze(0)  # (1, batch_size, features)
                    attended, _ = self.cross_attention(
                        cross_section, cross_section, cross_section
                    )
                    attended_features.append(attended.squeeze(0))  # (batch_size, features)
                
                attended_tensor = torch.stack(attended_features, dim=1)  # (batch, seq_len, features)
                
                # Compute relative ranking scores
                ranking_input = torch.cat([x, industry_probs], dim=-1)
                ranking_scores = self.ranking_network(ranking_input)  # (batch, seq_len, 1)
                
                # Industry-relative features
                industry_means = []
                for t in range(seq_len):
                    # Weighted average by industry probabilities
                    industry_weights = industry_probs[:, t, :].unsqueeze(-1)  # (batch, industry_groups, 1)
                    features_expanded = x[:, t, :].unsqueeze(1).repeat(1, self.industry_groups, 1)  # (batch, industry_groups, features)
                    
                    # Industry-weighted features
                    industry_weighted = industry_weights * features_expanded  # (batch, industry_groups, features)
                    industry_mean = industry_weighted.sum(dim=1)  # (batch, features)
                    industry_means.append(industry_mean)
                
                industry_relative = x - torch.stack(industry_means, dim=1)  # (batch, seq_len, features)
                
                # Combine all cross-sectional features
                output = torch.cat([
                    x,  # Original features
                    ranks,  # Cross-sectional ranks
                    attended_tensor,  # Cross-attention features
                    industry_probs,  # Industry probabilities
                    ranking_scores,  # Ranking scores
                    industry_relative  # Industry-relative features
                ], dim=-1)
                
                return output
        
        return CrossSectionalModule(input_shape[0], input_shape[-1], ranking_method, industry_groups)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        industry_groups = kwargs.get('industry_groups', 10)
        # Original + ranks + attended + industry_probs + ranking_scores + industry_relative
        output_features = input_shape[-1] * 4 + industry_groups + 1
        return input_shape[:-1] + (output_features,)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'ranking_method': ['percentile', 'zscore', 'decile'],
            'industry_groups': [5, 10, 15, 20]
        }