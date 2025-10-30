"""
pca_utils.py
Funciones para cálculo y visualización de PCA / Eigenfaces
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_mean(X):
    """Calcula el vector promedio μ de las imágenes."""
    return np.mean(X, axis=0)


def demean_data(X, mu):
    """Centra la matriz restando el promedio μ a cada fila."""
    return X - mu


def compute_pca(X_centered):
    """
    Calcula las componentes principales usando SVD.

    Returns
    -------
    V : np.ndarray
        Matriz de eigenfaces (componentes principales, columnas).
    S : np.ndarray
        Valores singulares.
    """
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T
    return V, S


def project_data(X_centered, V, r):
    """Proyecta X en el subespacio de las r primeras componentes."""
    Vr = V[:, :r]
    return np.dot(X_centered, Vr)


def reconstruct_data(F, V, mu, r):
    """Reconstruye X' a partir de sus proyecciones F y las r componentes."""
    Vr = V[:, :r]
    return np.dot(F, Vr.T) + mu


def plot_eigenfaces(V, num=10, img_shape=(50, 50)):
    """Muestra las primeras 'num' eigenfaces en escala de grises."""
    fig, axes = plt.subplots(1, num, figsize=(num * 1.2, 1.5))
    for i in range(num):
        ax = axes[i]
        face = np.real(V[:, i]).reshape(img_shape)
        ax.imshow(face, cmap="gray")
        ax.axis("off")
    plt.suptitle(f"Primeras {num} Eigenfaces")
    plt.show()


def frobenius_distance(X, X_rec):
    """Calcula la distancia Frobenius promedio."""
    diff = X - X_rec
    return np.mean(np.trace(np.dot(diff.T, diff))) / X.shape[0]


def plot_curve(x_vals, y_vals, xlabel, ylabel, title):
    """Gráfico genérico (para accuracy vs r o error vs r)."""
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
