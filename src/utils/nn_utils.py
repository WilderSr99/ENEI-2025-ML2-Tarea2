"""
nn_utils.py
Funciones auxiliares para entrenamiento y evaluación de redes neuronales (PyTorch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(model, train_loader, test_loader, epochs, lr, device):
    """
    Entrena un modelo PyTorch y retorna métricas.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    history = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Época {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc = evaluate_model(model, test_loader, device)
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(acc)

        print(f"Época {epoch+1}/{epochs} — Pérdida: {avg_loss:.4f}, Acc test: {acc:.4f}")

    return history


def evaluate_model(model, loader, device):
    """Calcula accuracy sobre un loader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def plot_training_curves(history, title="Curvas de entrenamiento"):
    """Grafica pérdida y accuracy."""
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Pérdida (train)", color="tab:red")
    ax1.plot(history["train_loss"], color="tab:red", label="Train loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (test)", color="tab:blue")
    ax2.plot(history["test_acc"], color="tab:blue", label="Test acc")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title(title)
    fig.tight_layout()
    plt.show()
