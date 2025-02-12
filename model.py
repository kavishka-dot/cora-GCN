import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGCN(nn.Module):
    """
    Implements a simple 2-layer Graph Convolutional Network (GCN).

    Args:
        in_features (int): Number of input features per node.
        hidden_features (int): Number of hidden layer neurons.
        out_features (int): Number of output classes.

    Forward Pass:
        - First GCN layer: Uses a learnable weight matrix W1.
        - ReLU Activation.
        - Second GCN layer: Uses a learnable weight matrix W2.

    Returns:
        torch.Tensor: The output logits for each node.
    """
    def __init__(self, in_features, hidden_features, out_features):
        super(BasicGCN, self).__init__()
        self.W1 = nn.Parameter(torch.randn(in_features, hidden_features) * 0.01)
        self.W2 = nn.Parameter(torch.randn(hidden_features, out_features) * 0.01)

    def forward(self, x, adj):
        x = adj @ x @ self.W1
        x = F.relu(x)  # Activation function
        x = adj @ x @ self.W2
        return x

