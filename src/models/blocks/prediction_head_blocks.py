from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..domain_blocks import DomainBlock


class RegressionHeadBlock(DomainBlock):
    """回帰予測ヘッドブロック - 連続値予測用"""
    
    def __init__(self):
        super().__init__(
            name="regression_head",
            category="prediction_heads",
            description="Regression head for continuous value prediction with multiple outputs"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        output_size = kwargs.get('output_size', 1)
        hidden_dims = kwargs.get('hidden_dims', [])  # Empty means direct linear
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'relu')
        use_batch_norm = kwargs.get('use_batch_norm', False)
        output_activation = kwargs.get('output_activation', 'none')
        
        class RegressionHeadModule(nn.Module):
            def __init__(self, input_size, output_size, hidden_dims, dropout, activation, use_batch_norm, output_activation):
                super().__init__()
                
                layers = []
                prev_dim = input_size
                
                # Hidden layers
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    
                    if use_batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_dim))
                    
                    layers.append(self._get_activation(activation))
                    
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    
                    prev_dim = hidden_dim
                
                # Output layer
                layers.append(nn.Linear(prev_dim, output_size))
                
                # Output activation
                if output_activation != 'none':
                    layers.append(self._get_activation(output_activation))
                
                self.network = nn.Sequential(*layers)
            
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
                elif activation == 'none':
                    return nn.Identity()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    # Take last timestep for sequence input
                    x = x[:, -1, :]
                
                return self.network(x)
        
        return RegressionHeadModule(
            input_shape[-1], output_size, hidden_dims, dropout, activation, use_batch_norm, output_activation
        )
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        output_size = kwargs.get('output_size', 1)
        return (input_shape[0], output_size)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'output_size': [1, 3, 5],  # 1 for single prediction, 3 for classification, 5 for quintiles
            'hidden_dims': [[], [64], [128], [64, 32], [128, 64]],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'activation': ['relu', 'gelu', 'tanh', 'leaky_relu', 'swish'],
            'use_batch_norm': [True, False],
            'output_activation': ['none', 'tanh', 'sigmoid']
        }


class ClassificationHeadBlock(DomainBlock):
    """分類予測ヘッドブロック - 離散値予測用"""
    
    def __init__(self):
        super().__init__(
            name="classification_head",
            category="prediction_heads",
            description="Classification head for discrete prediction with class probabilities"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        num_classes = kwargs.get('num_classes', 3)  # Up, Down, Sideways
        hidden_dims = kwargs.get('hidden_dims', [64])
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'relu')
        label_smoothing = kwargs.get('label_smoothing', 0.0)
        
        class ClassificationHeadModule(nn.Module):
            def __init__(self, input_size, num_classes, hidden_dims, dropout, activation, label_smoothing):
                super().__init__()
                self.num_classes = num_classes
                self.label_smoothing = label_smoothing
                
                layers = []
                prev_dim = input_size
                
                # Hidden layers
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(self._get_activation(activation))
                    
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    
                    prev_dim = hidden_dim
                
                # Output layer (logits)
                layers.append(nn.Linear(prev_dim, num_classes))
                
                self.network = nn.Sequential(*layers)
                
                # Temperature scaling for calibration
                self.temperature = nn.Parameter(torch.ones(1))
            
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
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    # Take last timestep for sequence input
                    x = x[:, -1, :]
                
                # Get logits
                logits = self.network(x)
                
                # Apply temperature scaling
                calibrated_logits = logits / self.temperature
                
                # Return probabilities
                probs = F.softmax(calibrated_logits, dim=-1)
                
                return probs
            
            def get_logits(self, x):
                """Get raw logits without softmax"""
                if len(x.shape) == 3:
                    x = x[:, -1, :]
                return self.network(x)
        
        return ClassificationHeadModule(
            input_shape[-1], num_classes, hidden_dims, dropout, activation, label_smoothing
        )
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        num_classes = kwargs.get('num_classes', 3)
        return (input_shape[0], num_classes)
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'num_classes': [2, 3, 5, 7, 10],  # Binary, ternary, quintiles, etc.
            'hidden_dims': [[], [32], [64], [128], [64, 32], [128, 64]],
            'dropout': [0.0, 0.1, 0.2, 0.3],
            'activation': ['relu', 'gelu', 'leaky_relu', 'elu', 'swish'],
            'label_smoothing': [0.0, 0.05, 0.1]
        }


class RankingHeadBlock(DomainBlock):
    """ランキング予測ヘッドブロック - 順位予測用"""
    
    def __init__(self):
        super().__init__(
            name="ranking_head",
            category="prediction_heads",
            description="Ranking head for relative ordering prediction"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        hidden_dims = kwargs.get('hidden_dims', [64])
        dropout = kwargs.get('dropout', 0.1)
        activation = kwargs.get('activation', 'relu')
        ranking_loss = kwargs.get('ranking_loss', 'listwise')  # 'pointwise', 'pairwise', 'listwise'
        
        class RankingHeadModule(nn.Module):
            def __init__(self, input_size, hidden_dims, dropout, activation, ranking_loss):
                super().__init__()
                self.ranking_loss = ranking_loss
                
                layers = []
                prev_dim = input_size
                
                # Hidden layers
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(self._get_activation(activation))
                    
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    
                    prev_dim = hidden_dim
                
                # Output layer (single score for ranking)
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
            
            def _get_activation(self, activation):
                if activation == 'relu':
                    return nn.ReLU()
                elif activation == 'gelu':
                    return nn.GELU()
                elif activation == 'tanh':
                    return nn.Tanh()
                elif activation == 'leaky_relu':
                    return nn.LeakyReLU(0.2)
                elif activation == 'swish':
                    return nn.SiLU()
                else:
                    return nn.ReLU()
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    # Take last timestep for sequence input
                    x = x[:, -1, :]
                
                # Get ranking scores
                scores = self.network(x)  # (batch, 1)
                
                return scores.squeeze(-1)  # (batch,)
        
        return RankingHeadModule(input_shape[-1], hidden_dims, dropout, activation, ranking_loss)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Tuple[int, ...]:
        return (input_shape[0],)  # Single ranking score per sample
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'hidden_dims': [[], [32], [64], [128], [64, 32]],
            'dropout': [0.0, 0.1, 0.2],
            'activation': ['relu', 'gelu', 'tanh', 'leaky_relu', 'swish'],
            'ranking_loss': ['pointwise', 'pairwise', 'listwise']
        }


class MultiTaskHeadBlock(DomainBlock):
    """マルチタスク予測ヘッドブロック - 複数のタスクを同時に予測"""
    
    def __init__(self):
        super().__init__(
            name="multi_task_head",
            category="prediction_heads",
            description="Multi-task head for simultaneous prediction of multiple targets"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        tasks = kwargs.get('tasks', {
            'return_pred': {'type': 'regression', 'output_size': 1},
            'direction': {'type': 'classification', 'num_classes': 3},
            'volatility': {'type': 'regression', 'output_size': 1}
        })
        shared_hidden_dims = kwargs.get('shared_hidden_dims', [128, 64])
        task_hidden_dims = kwargs.get('task_hidden_dims', [32])
        dropout = kwargs.get('dropout', 0.1)
        
        class MultiTaskHeadModule(nn.Module):
            def __init__(self, input_size, tasks, shared_hidden_dims, task_hidden_dims, dropout):
                super().__init__()
                self.tasks = tasks
                
                # Shared layers
                shared_layers = []
                prev_dim = input_size
                
                for hidden_dim in shared_hidden_dims:
                    shared_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                self.shared_network = nn.Sequential(*shared_layers)
                shared_output_dim = prev_dim
                
                # Task-specific heads
                self.task_heads = nn.ModuleDict()
                
                for task_name, task_config in tasks.items():
                    task_layers = []
                    task_prev_dim = shared_output_dim
                    
                    # Task-specific hidden layers
                    for hidden_dim in task_hidden_dims:
                        task_layers.extend([
                            nn.Linear(task_prev_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        task_prev_dim = hidden_dim
                    
                    # Task-specific output layer
                    if task_config['type'] == 'regression':
                        output_size = task_config['output_size']
                        task_layers.append(nn.Linear(task_prev_dim, output_size))
                    elif task_config['type'] == 'classification':
                        num_classes = task_config['num_classes']
                        task_layers.extend([
                            nn.Linear(task_prev_dim, num_classes),
                            nn.Softmax(dim=-1)
                        ])
                    
                    self.task_heads[task_name] = nn.Sequential(*task_layers)
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    # Take last timestep for sequence input
                    x = x[:, -1, :]
                
                # Shared feature extraction
                shared_features = self.shared_network(x)
                
                # Task-specific predictions
                outputs = {}
                for task_name, task_head in self.task_heads.items():
                    outputs[task_name] = task_head(shared_features)
                
                return outputs
        
        return MultiTaskHeadModule(input_shape[-1], tasks, shared_hidden_dims, task_hidden_dims, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Dict[str, Tuple[int, ...]]:
        tasks = kwargs.get('tasks', {
            'return_pred': {'type': 'regression', 'output_size': 1},
            'direction': {'type': 'classification', 'num_classes': 3},
            'volatility': {'type': 'regression', 'output_size': 1}
        })
        
        output_shapes = {}
        for task_name, task_config in tasks.items():
            if task_config['type'] == 'regression':
                output_shapes[task_name] = (input_shape[0], task_config['output_size'])
            elif task_config['type'] == 'classification':
                output_shapes[task_name] = (input_shape[0], task_config['num_classes'])
        
        return output_shapes
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'shared_hidden_dims': [[64], [128], [128, 64], [256, 128], [128, 64, 32]],
            'task_hidden_dims': [[], [16], [32], [64], [32, 16]],
            'dropout': [0.0, 0.1, 0.2],
            'tasks': [
                {'return_pred': {'type': 'regression', 'output_size': 1}},
                {
                    'return_pred': {'type': 'regression', 'output_size': 1},
                    'direction': {'type': 'classification', 'num_classes': 3}
                },
                {
                    'return_pred': {'type': 'regression', 'output_size': 1},
                    'direction': {'type': 'classification', 'num_classes': 3},
                    'volatility': {'type': 'regression', 'output_size': 1}
                }
            ]
        }


class DistributionHeadBlock(DomainBlock):
    """分布予測ヘッドブロック - 確率分布を予測"""
    
    def __init__(self):
        super().__init__(
            name="distribution_head",
            category="prediction_heads",
            description="Distribution head for probabilistic predictions"
        )
    
    def create_module(self, input_shape: Tuple[int, ...], **kwargs) -> nn.Module:
        distribution_type = kwargs.get('distribution_type', 'normal')  # 'normal', 'laplace', 'mixture'
        n_components = kwargs.get('n_components', 1)  # For mixture models
        hidden_dims = kwargs.get('hidden_dims', [64])
        dropout = kwargs.get('dropout', 0.1)
        
        class DistributionHeadModule(nn.Module):
            def __init__(self, input_size, distribution_type, n_components, hidden_dims, dropout):
                super().__init__()
                self.distribution_type = distribution_type
                self.n_components = n_components
                
                # Shared feature extraction
                shared_layers = []
                prev_dim = input_size
                
                for hidden_dim in hidden_dims:
                    shared_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                self.shared_network = nn.Sequential(*shared_layers)
                
                # Distribution parameter heads
                if distribution_type == 'normal':
                    if n_components == 1:
                        # Single Gaussian
                        self.mean_head = nn.Linear(prev_dim, 1)
                        self.std_head = nn.Sequential(
                            nn.Linear(prev_dim, 1),
                            nn.Softplus()  # Ensure positive
                        )
                    else:
                        # Gaussian mixture
                        self.mean_head = nn.Linear(prev_dim, n_components)
                        self.std_head = nn.Sequential(
                            nn.Linear(prev_dim, n_components),
                            nn.Softplus()
                        )
                        self.weight_head = nn.Sequential(
                            nn.Linear(prev_dim, n_components),
                            nn.Softmax(dim=-1)
                        )
                
                elif distribution_type == 'laplace':
                    # Laplace distribution
                    self.location_head = nn.Linear(prev_dim, n_components)
                    self.scale_head = nn.Sequential(
                        nn.Linear(prev_dim, n_components),
                        nn.Softplus()
                    )
                    if n_components > 1:
                        self.weight_head = nn.Sequential(
                            nn.Linear(prev_dim, n_components),
                            nn.Softmax(dim=-1)
                        )
            
            def forward(self, x):
                # x: (batch, seq_len, features) or (batch, features)
                if len(x.shape) == 3:
                    # Take last timestep for sequence input
                    x = x[:, -1, :]
                
                # Extract shared features
                features = self.shared_network(x)
                
                # Predict distribution parameters
                if self.distribution_type == 'normal':
                    mean = self.mean_head(features)
                    std = self.std_head(features)
                    
                    if self.n_components == 1:
                        return {'mean': mean, 'std': std}
                    else:
                        weight = self.weight_head(features)
                        return {'mean': mean, 'std': std, 'weight': weight}
                
                elif self.distribution_type == 'laplace':
                    location = self.location_head(features)
                    scale = self.scale_head(features)
                    
                    if self.n_components == 1:
                        return {'location': location, 'scale': scale}
                    else:
                        weight = self.weight_head(features)
                        return {'location': location, 'scale': scale, 'weight': weight}
            
            def sample(self, params, n_samples=1):
                """Sample from the predicted distribution"""
                if self.distribution_type == 'normal':
                    if self.n_components == 1:
                        dist = torch.distributions.Normal(params['mean'], params['std'])
                        return dist.sample((n_samples,))
                    else:
                        # Mixture of Gaussians
                        mix = torch.distributions.Categorical(params['weight'])
                        comp = torch.distributions.Normal(params['mean'], params['std'])
                        mixture = torch.distributions.MixtureSameFamily(mix, comp)
                        return mixture.sample((n_samples,))
                
                elif self.distribution_type == 'laplace':
                    if self.n_components == 1:
                        dist = torch.distributions.Laplace(params['location'], params['scale'])
                        return dist.sample((n_samples,))
                    else:
                        # Mixture of Laplace
                        mix = torch.distributions.Categorical(params['weight'])
                        comp = torch.distributions.Laplace(params['location'], params['scale'])
                        mixture = torch.distributions.MixtureSameFamily(mix, comp)
                        return mixture.sample((n_samples,))
        
        return DistributionHeadModule(input_shape[-1], distribution_type, n_components, hidden_dims, dropout)
    
    def get_output_shape(self, input_shape: Tuple[int, ...], **kwargs) -> Dict[str, Tuple[int, ...]]:
        distribution_type = kwargs.get('distribution_type', 'normal')
        n_components = kwargs.get('n_components', 1)
        
        if distribution_type == 'normal':
            if n_components == 1:
                return {
                    'mean': (input_shape[0], 1),
                    'std': (input_shape[0], 1)
                }
            else:
                return {
                    'mean': (input_shape[0], n_components),
                    'std': (input_shape[0], n_components),
                    'weight': (input_shape[0], n_components)
                }
        elif distribution_type == 'laplace':
            if n_components == 1:
                return {
                    'location': (input_shape[0], 1),
                    'scale': (input_shape[0], 1)
                }
            else:
                return {
                    'location': (input_shape[0], n_components),
                    'scale': (input_shape[0], n_components),
                    'weight': (input_shape[0], n_components)
                }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'distribution_type': ['normal', 'laplace'],
            'n_components': [1, 2, 3, 5],
            'hidden_dims': [[32], [64], [128], [64, 32]],
            'dropout': [0.0, 0.1, 0.2]
        }