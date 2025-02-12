import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

def load_data():
    """
    Loads the Cora dataset for graph-based learning.

    The Cora dataset is a citation network where:
    - Nodes represent research papers.
    - Edges represent citation relationships between papers.
    - Each node has a feature vector representing the paper's text.
    - The goal is to classify papers into one of seven classes.

    Returns:
        data (torch_geometric.data.Data): The dataset containing node features, edges, labels, and masks.
        num_features (int): Number of input features per node.
        num_classes (int): Number of unique classes in the dataset.
    """
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    return dataset[0], dataset.num_node_features, dataset.num_classes
