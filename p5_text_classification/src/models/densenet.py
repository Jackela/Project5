"""
DenseNet model implementation for 1D text classification.
Based on "Densely Connected Convolutional Networks" (Huang et al., 2017).
Adapted for character-level text classification with 1D convolutions.
Follows SOLID principles with modular, well-documented components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class _DenseLayer(nn.Module):
    """
    A single dense layer in the DenseNet architecture implementing the bottleneck design.
    
    Architecture: BN-ReLU-Conv1d(1x1) -> BN-ReLU-Conv1d(3x3)
    The 1x1 convolution (bottleneck) produces 4 * growth_rate channels to reduce
    computational complexity, followed by a 3x3 convolution that outputs growth_rate channels.
    
    Args:
        num_input_features (int): Number of input channels from previous layers
        growth_rate (int): Number of new feature maps (k) produced by this layer
        bn_size (int): Multiplicative factor for bottleneck layer (default: 4)
        drop_rate (float): Dropout probability (default: 0.0)
    """
    
    def __init__(
        self, 
        num_input_features: int, 
        growth_rate: int, 
        bn_size: int = 4, 
        drop_rate: float = 0.0
    ):
        super(_DenseLayer, self).__init__()
        
        self.drop_rate = drop_rate
        
        # Bottleneck layer: BN-ReLU-Conv1d(1x1)
        # Reduces feature maps to bn_size * growth_rate for efficiency
        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.conv1 = nn.Conv1d(
            num_input_features, 
            bn_size * growth_rate,
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        
        # Main convolution: BN-ReLU-Conv1d(3x3)
        # Produces growth_rate new feature maps
        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.conv2 = nn.Conv1d(
            bn_size * growth_rate, 
            growth_rate,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dense layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_input_features, sequence_length)
            
        Returns:
            Concatenated tensor of input and new features
            Shape: (batch_size, num_input_features + growth_rate, sequence_length)
        """
        # Bottleneck transformation: BN-ReLU-Conv1d(1x1)
        bottleneck_output = self.conv1(F.relu(self.norm1(x)))
        
        # Main transformation: BN-ReLU-Conv1d(3x3)
        new_features = self.conv2(F.relu(self.norm2(bottleneck_output)))
        
        # Apply dropout if specified
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        # Dense connection: concatenate input with new features along channel dimension
        return torch.cat([x, new_features], dim=1)


class _DenseBlock(nn.Module):
    """
    A dense block consisting of multiple dense layers.
    
    Each layer receives feature maps from all preceding layers within the block,
    implementing the dense connectivity pattern that gives DenseNet its name.
    
    Args:
        num_layers (int): Number of dense layers in this block
        num_input_features (int): Number of input channels to the block
        bn_size (int): Multiplicative factor for bottleneck layers
        growth_rate (int): Number of new feature maps per layer
        drop_rate (float): Dropout probability
    """
    
    def __init__(
        self, 
        num_layers: int, 
        num_input_features: int, 
        bn_size: int, 
        growth_rate: int, 
        drop_rate: float = 0.0
    ):
        super(_DenseBlock, self).__init__()
        
        # Create dense layers sequentially
        # Each layer takes as input all feature maps from previous layers
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,  # Input grows with each layer
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dense block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with accumulated feature maps from all layers
        """
        # Sequentially apply each dense layer
        # Each layer receives concatenated outputs from all previous layers
        for name, layer in self.named_children():
            x = layer(x)
        return x


class _Transition(nn.Module):
    """
    Transition layer between dense blocks.
    
    Performs compression to reduce the number of feature maps and spatial dimensions.
    Architecture: BN-ReLU-Conv1d(1x1)-AvgPool1d
    
    Args:
        num_input_features (int): Number of input channels
        num_output_features (int): Number of output channels after compression
    """
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super(_Transition, self).__init__()
        
        # Compression layer: BN-ReLU-Conv1d(1x1)
        self.norm = nn.BatchNorm1d(num_input_features)
        self.conv = nn.Conv1d(
            num_input_features, 
            num_output_features, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        
        # Downsampling: Average pooling to reduce sequence length
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transition layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Compressed and downsampled tensor
        """
        x = self.conv(F.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    """
    DenseNet architecture for 1D text classification.
    
    Implements the DenseNet-BC (Bottleneck and Compression) variant adapted for
    character-level text classification. The network consists of:
    1. Initial convolution and pooling
    2. Multiple dense blocks separated by transition layers
    3. Global average pooling and classification head
    
    Args:
        num_classes (int): Number of output classes
        vocab_size (int): Size of the character vocabulary
        growth_rate (int): Number of new feature maps per layer (k parameter)
        block_config (tuple): Number of layers in each dense block
        num_init_features (int): Number of feature maps after initial convolution
        bn_size (int): Multiplicative factor for bottleneck layers
        compression (float): Compression factor theta (0 < theta <= 1)
        drop_rate (float): Dropout probability
        embedding_dim (int): Dimension of character embeddings
    """
    
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (4, 4, 4),
        num_init_features: int = 24,
        bn_size: int = 4,
        compression: float = 0.5,
        drop_rate: float = 0.0,
        embedding_dim: int = 16,
        **kwargs
    ):
        super(DenseNet, self).__init__()
        
        # Validate compression factor
        if not (0 < compression <= 1):
            raise ValueError(f"Compression must be in (0, 1], got {compression}")
        
        # Store configuration
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initial convolution: maps embeddings to initial feature space
        self.initial_conv = nn.Conv1d(
            embedding_dim, 
            num_init_features, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.initial_norm = nn.BatchNorm1d(num_init_features)
        self.initial_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks and transition layers
        self.features = nn.Sequential()
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            
            # Update feature count after dense block
            num_features += num_layers * growth_rate
            
            # Add transition layer (except after the last block)
            if i != len(block_config) - 1:
                # Apply compression
                num_output_features = int(num_features * compression)
                transition = _Transition(num_features, num_output_features)
                self.features.add_module(f'transition{i + 1}', transition)
                num_features = num_output_features
        
        # Final batch normalization
        self.final_norm = nn.BatchNorm1d(num_features)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using best practices for DenseNet."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DenseNet.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, sequence_length)
        
        # 1. Character embedding
        x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        
        # Transpose for 1D convolution: (batch_size, embedding_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # 2. Initial convolution and pooling
        x = self.initial_pool(F.relu(self.initial_norm(self.initial_conv(x))))
        
        # 3. Dense blocks and transitions
        x = self.features(x)
        
        # 4. Final normalization and activation
        x = F.relu(self.final_norm(x))
        
        # 5. Global average pooling
        x = self.global_pool(x)  # (batch_size, num_features, 1)
        x = x.view(x.size(0), -1)  # (batch_size, num_features)
        
        # 6. Classification
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x
    
    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get total and trainable parameter counts for the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


# Main class alias for convenient importing
DenseNet = DenseNet 