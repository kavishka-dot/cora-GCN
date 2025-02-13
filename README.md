# Graph Convolutional Network (GCN) for Research Paper Classification
![Cora2](https://github.com/user-attachments/assets/47adcb60-5177-4d48-872e-a0212c75b3e8)

## 📌 Project Overview
This project implements a **Graph Convolutional Network (GCN) from scratch** using PyTorch and PyTorch Geometric (PyG). The model is trained on the **Cora citation dataset**, which is widely used for node classification tasks in graph learning.

## 🚀 Features
- **Custom GCN implementation** without using built-in PyG layers
- **Cora dataset** loading and preprocessing
- **Adjacency matrix normalization** for stable training
- **Training pipeline with loss and accuracy tracking**
- **Visualization of training loss and test accuracy**

---
## 📂 Project Structure
```
GCN_Project/
│── main.py              # Main script to train and evaluate the model
│── model.py             # Defines the GCN model
│── utils.py             # Utility functions (normalization, accuracy, plotting)
│── dataset.py           # Loads the dataset
│── train.py             # Training function
│── requirements.txt     # Dependencies for your project
│── README.md            # Project description
```

---
## 📊 About the Cora Dataset
The **Cora dataset** is a citation network dataset used for semi-supervised node classification.
- **Nodes** represent research papers.
- **Edges** represent citation relationships.
- **Each node** has a feature vector (bag-of-words representation of the paper text).
- **Task**: Predict the category of each research paper.

📌 **Dataset Details:**
- **Number of Nodes**: 2,708
- **Number of Edges**: 5,429
- **Number of Features per Node**: 1,433
- **Number of Classes**: 7

---
## ⚙️ Installation
To set up the project, first install the required dependencies:

```sh
# Clone the repository
git clone https://github.com/kavishka-dot/cora-GCN.git
cd GCN_Project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

---
## 🏗️ How It Works
### 1️⃣ Load Dataset
```python
from dataset import load_data

data, in_features, num_classes = load_data()
```
### 2️⃣ Normalize Adjacency Matrix
```python
from utils import normalize_adjacency

adj_norm = normalize_adjacency(data.edge_index, data.num_nodes)
```
### 3️⃣ Define and Train GCN Model
```python
from model import BasicGCN
from train import train

model = BasicGCN(in_features=in_features, hidden_features=16, out_features=num_classes)
train(model, data, adj_norm, epochs=200)
```

---
## 📊 Training & Results
During training, the model prints the loss and test accuracy every 10 epochs.

Example output:
```
Epoch  0, Loss: 1.9456, Test Accuracy: 0.5620
Epoch 10, Loss: 1.1023, Test Accuracy: 0.7150
Epoch 20, Loss: 0.8154, Test Accuracy: 0.7652
...
Epoch 190, Loss: 0.2751, Test Accuracy: 0.8123
```

### 📈 Visualization
Once training is complete, the **loss and test accuracy plots** are generated.

```python
from utils import plot_metrics
plot_metrics(train_losses, test_accuracies)
```

---
## 🔥 Future Improvements
- Add **Graph Attention Networks (GAT)** for comparison.
- Extend to **other datasets** like PubMed and Citeseer.
- Implement **hyperparameter tuning**.

---
## 🛠️ Requirements
Ensure you have the following packages installed:
```txt
torch
torch_geometric
numpy
matplotlib
networkx
```

## 🤝 Contributing
Feel free to contribute by submitting a pull request or opening an issue! 😊



