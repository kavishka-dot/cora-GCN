import torch
from dataset import load_data
from model import BasicGCN
from utils import normalize_adjacency
from train import train

# Load dataset
data, in_features, num_classes = load_data()

# Compute normalized adjacency matrix
adj_norm = normalize_adjacency(data.edge_index, data.num_nodes)

# Define model
model = BasicGCN(in_features=in_features, hidden_features=16, out_features=num_classes)

# Train model
train(model, data, adj_norm, epochs=200)
