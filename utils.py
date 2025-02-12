import torch
import matplotlib.pyplot as plt

def normalize_adjacency(edge_index, num_nodes):
    """
    Computes the normalized adjacency matrix using the formula:
    A_norm = D^(-1/2) * (A + I) * D^(-1/2)

    This ensures that the adjacency matrix is symmetric and prevents exploding gradients.

    Args:
        edge_index (torch.Tensor): The edge index tensor (sparse representation of the adjacency matrix).
        num_nodes (int): Total number of nodes in the graph.

    Returns:
        torch.Tensor: The normalized adjacency matrix in dense format.
    """
    edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)  # Assign weight of 1 to all edges
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

    # Convert sparse adjacency matrix to dense and add self-loops
    adj_dense = adj.to_dense() + torch.eye(num_nodes, device=edge_index.device)

    # Compute degree matrix D
    deg = adj_dense.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0  # Handle division by zero

    # Compute normalized adjacency matrix
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj_dense @ D_inv_sqrt  # Apply normalization

    return adj_norm

def compute_accuracy(model, data, adj, mask):
    """
    Computes accuracy of the model on a given mask (train/val/test).

    Args:
        model (nn.Module): The trained GCN model.
        data (torch_geometric.data.Data): The dataset containing node features and labels.
        adj (torch.Tensor): The normalized adjacency matrix.
        mask (torch.Tensor): Boolean mask indicating which nodes to evaluate.

    Returns:
        float: Accuracy of the model on the selected mask.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, adj)
        predictions = out.argmax(dim=1)
        correct = (predictions[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return accuracy

def plot_metrics(train_losses, test_accuracies):
    """
    Plots training loss and test accuracy over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        test_accuracies (list): List of test accuracies per epoch.
    """
    epochs = range(len(train_losses))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color='tab:blue')
    ax2.plot(epochs, test_accuracies, color='tab:blue', linestyle='dashed', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.title("Training Loss & Test Accuracy")
    plt.show()
