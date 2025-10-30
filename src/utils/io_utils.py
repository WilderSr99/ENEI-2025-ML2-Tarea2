"""
io_utils.py
Funciones para lectura de datos e imágenes (Eigenfaces)
"""

import numpy as np
import imageio.v2 as imageio
from pathlib import Path


def load_faces(txt_path):
    """
    Lee rutas e etiquetas desde un archivo .txt y devuelve X (imágenes aplanadas) y y (etiquetas).
    Cada línea del txt debe tener: 'ruta_imagen etiqueta'

    Parameters
    ----------
    txt_path : str o Path
        Ruta al archivo de texto.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    txt_path = Path(txt_path)
    X, y = [], []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            img_path, label = parts[0], int(parts[1])
            img = imageio.imread(str(img_path))
            X.append(img.reshape(-1))
            y.append(label)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y


def show_image(flat_img, size=(50, 50), cmap="gray", title=None):
    """Muestra una imagen aplanada en escala de grises."""
    import matplotlib.pyplot as plt
    plt.imshow(flat_img.reshape(size), cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
