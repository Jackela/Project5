"""
VDCNN model implementation - refactored version.
Based on the verified implementation from src/vdcnn.py.
Follows the Single Responsibility Principle (SRP) and Open/Closed Principle (OCP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


class ConvBlock(nn.Module):
    """
    A building block for the VDCNN model, consisting of two convolutional layers.
    Each convolutional layer is followed by Batch Normalization and a ReLU activation.
    This follows the standard design for deep convolutional networks, promoting stable training
    and robust feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        use_shortcut (bool): If True, adds a residual shortcut connection. Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool = False):
        super(ConvBlock, self).__init__()
        self.use_shortcut = use_shortcut

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to allow for deeper networks by mitigating vanishing gradients.
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # Store the original input for the shortcut connection.
        residual = x
        
        # First convolution -> batch norm -> activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        
        # Second convolution -> batch norm
        out = self.conv2(out)
        out = self.bn2(out)

        # If using a shortcut, add the original (potentially transformed) input.
        if self.use_shortcut:
            residual = self.shortcut(residual)
            out += residual
        
        # Final activation after the block.
        out = nn.functional.relu(out)
        return out


class VDCNN(nn.Module):
    """
    Implementation of a Very Deep Convolutional Network (VDCNN) for character-level text classification.
    Reference: "Very Deep Convolutional Networks for Text Classification" (Conneau et al., 2017).

    The architecture uses a series of convolutional blocks to learn hierarchical features from
    character embeddings, followed by k-max pooling and a multi-layer perceptron for classification.

    Args:
        num_classes (int): The number of target classes for classification.
        vocab_size (int): The total number of unique characters in the vocabulary.
        embedding_dim (int): The dimensionality of the character embeddings.
        depth (int): The depth of the network, must be one of {9, 17, 29, 49}.
        use_shortcut (bool): Whether to use shortcut connections in ConvBlocks. Defaults to False.
        k (int): The value of 'k' for k-max pooling. Defaults to 8.
    """
    def __init__(self, num_classes: int, vocab_size: int, embedding_dim: int = 16, depth: int = 9, use_shortcut: bool = False, k: int = 8, **kwargs):
        super(VDCNN, self).__init__()
        
        # Configuration mapping depth to the number of ConvBlocks in each stage.
        depth_configs = {
            9: [1, 1, 1, 1],
            17: [2, 2, 2, 2],
            29: [5, 5, 2, 2],
            49: [8, 8, 5, 3]
        }
        if depth not in depth_configs:
            raise ValueError(f"Unsupported depth: {depth}. Choose from {list(depth_configs.keys())}")
            
        num_conv_blocks = depth_configs[depth]
        self.k = k

        # --- Architecture Layers ---

        # 1. Character Embedding Layer: Maps character IDs to dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. First Convolutional Layer: A temporal convolution to start feature extraction.
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)

        # 3. Convolutional Stages: A series of ConvBlocks and down-sampling layers.
        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_conv_blocks[0], use_shortcut),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(64, 128, num_conv_blocks[1], use_shortcut),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(128, 256, num_conv_blocks[2], use_shortcut),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            self._make_layer(256, 512, num_conv_blocks[3], use_shortcut)
            # No final max-pooling before k-max pooling
        )

        # 4. Final Classifier: A multi-layer perceptron to make the final prediction.
        self.classifier = nn.Sequential(
            nn.Linear(512 * self.k, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, use_shortcut: bool):
        """Helper function to create a stack of ConvBlocks."""
        layers = []
        # The first block might handle a change in channel dimensions.
        layers.append(ConvBlock(in_channels, out_channels, use_shortcut))
        # Subsequent blocks have the same input and output channels.
        for _ in range(1, num_blocks):
            layers.append(ConvBlock(out_channels, out_channels, use_shortcut))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length)
        
        # 1. Embedding
        x = self.embedding(x)  # Shape: (batch_size, sequence_length, embedding_dim)
        
        # Transpose for 1D convolution: (batch_size, embedding_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # 2. First Convolution
        x = self.conv1(x) # Shape: (batch_size, 64, sequence_length)
        
        # 3. Convolutional Stages
        x = self.layers(x) # Shape: (batch_size, 512, reduced_sequence_length)
        
        # 4. K-Max Pooling
        # Select the top 'k' features along the sequence dimension.
        # This captures the most important features regardless of their position.
        x = x.topk(self.k, dim=2)[0] # Shape: (batch_size, 512, k)
        
        # 5. Flatten for Classifier
        x = x.view(x.size(0), -1) # Shape: (batch_size, 512 * k)
        
        # 6. Classification
        x = self.classifier(x) # Shape: (batch_size, num_classes)
        
        return x 