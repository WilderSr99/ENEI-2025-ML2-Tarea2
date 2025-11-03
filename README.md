# Assignment 2: Principal Component Analysis, Neural Networks
-------------------
2025-G1-910040-4-PEUCD-MACHINE LEARNING II
-------------------
## Integrantes del grupo

- *Buleje Ticse, Jean Carlos*
- *Rosales Chuco, Noel Ivan*
- *Sebastian Rios, Wilder Teddy*

## Estructura del proyecto

- **Parte I:** `src/ParteI_Eigenfaces.ipynb`  
  - Implementación del método **Eigenfaces for Face Recognition** usando **PCA**.  
  - Basado en 540 imágenes de entrenamiento (50×50 px) y 100 de prueba, listadas en `data/train.txt` y `data/test.txt`.  
  - Pasos realizados:
    - Cálculo del rostro promedio y centrado de los datos.  
    - Obtención de *eigenfaces* mediante **SVD** (`utils/pca_utils.py`).  
    - Proyección de los datos sobre los primeros *r* componentes principales.  
    - Clasificación con **Regresión Logística** sin intercepto.  
    - Reconstrucción y cálculo del **error Frobenius**.  
  - Resultados:
    - Con *r = 10*, accuracy test ≈ **0.59**  
    - Con *r = 130*, accuracy test ≈ **0.97**  
    - Error de reconstrucción decrece rápidamente hasta estabilizarse en *r ≈ 100*.  
  - Gráficos generados:  
    - `report/figuras/mean_face.png`  
    - `report/figuras/eigenfaces.png`  
    - `report/figuras/accuracy_vs_r.png`  
    - `report/figuras/error_reconstruccion.png`

- **Parte II:** `src/ParteII_NeuralNetworks.ipynb`  
  - Entrenamiento de redes neuronales en el dataset **MNIST** usando **PyTorch**.  
  - Modelos implementados:
    1. **MLP (Multilayer Perceptron)** - capas densas `784→256→128→10`.  
       - Accuracy final ≈ **0.9805**  
    2. **CNN básica** - 2 convoluciones + MaxPooling.  
       - Accuracy final ≈ **0.9869**  
    3. **CNN mejorada** - con **BatchNorm** y **Dropout**.  
       - Accuracy final ≈ **0.9924**  
  - Funciones auxiliares utilizadas:  
    - `utils/nn_utils.py` → Entrenamiento, evaluación y curvas.  
  - Curvas generadas:  
    - `report/figuras/curva_mlp.png`  
    - `report/figuras/curva_cnn_basica.png`  
    - `report/figuras/curva_cnn_mejorada.png`

---

## Resultados globales
- El **PCA** permitió reducir de 2500 a ~130 dimensiones conservando casi toda la información visual, logrando un **97% de acierto** en reconocimiento facial.  
- Las **CNNs** superaron a las redes densas (MLP) al aprovechar la estructura espacial de los píxeles.  
- La **CNN mejorada** alcanzó el mejor equilibrio entre precisión y generalización, con **≈99.2% de accuracy en test**.  
- Todos los resultados y gráficos se almacenan en la carpeta `report/`.

---

## Requisitos
Ver archivo `requirements.txt` para las librerías necesarias:
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `torch`, `torchvision`, `tqdm`, `imageio`, `seaborn`

---

## Configuración del entorno
```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno
source .venv/Scripts/activate        # o .venv\Scripts\Activate.ps1 en Windows

# Instalar dependencias
pip install -r requirements.txt
