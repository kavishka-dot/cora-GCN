import torch.optim as optim
import torch.nn as nn
from utils import compute_accuracy, plot_metrics

def train(model, data, adj, epochs=200, lr=0.01, weight_decay=5e-4):
    """
    Trains the GCN model and plots the loss and accuracy.

    Args:
        model (nn.Module): The GCN model.
        data (torch_geometric.data.Data): The dataset (Cora).
        adj (torch.Tensor): The normalized adjacency matrix.
        epochs (int, optional): Number of training epochs. Default is 200.
        lr (float, optional): Learning rate. Default is 0.01.
        weight_decay (float, optional): Weight decay (L2 regularization). Default is 5e-4.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, adj)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        test_accuracies.append(compute_accuracy(model, data, adj, data.test_mask))

        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}, Loss: {loss.item():.4f}, Test Accuracy: {test_accuracies[-1]:.4f}')

    # Plot training loss and test accuracy
    plot_metrics(train_losses, test_accuracies)
