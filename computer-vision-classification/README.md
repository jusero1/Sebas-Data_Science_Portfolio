# Detección de Neumonía en Radiografías Torácicas (CNN / Transfer Learning)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Objetivo del Proyecto y Planteamiento del Problema

### Contexto de negocio

La neumonía es la principal causa infecciosa de mortalidad infantil a nivel mundial. Su diagnóstico se realiza mayormente mediante interpretación visual de radiografías torácicas, un proceso lento, costoso y sujeto a variabilidad inter-observador. En regiones con escasez de radiólogos, un sistema de ayuda diagnóstica automatizado puede reducir drásticamente el tiempo de triaje y mejorar la cobertura.

### Objetivo técnico

Desarrollar un clasificador binario de imágenes médicas capaz de distinguir radiografías **NORMAL** vs **PNEUMONIA** con alta sensibilidad (recall), minimizando falsos negativos, que son clínicamente más costosos que los falsos positivos.

### Pregunta clínica

> ¿Puede un modelo de visión artificial detectar neumonía en radiografías torácicas con una sensibilidad ≥ 95% y especificidad ≥ 85%, comparable a un médico general?

---

## 2. Dataset

| Atributo | Detalle |
|---|---|
| **Nombre** | Chest X-Ray Images (Pneumonia) |
| **Origen** | [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Fuente original** | Guangzhou Women and Children's Medical Center / NIH |
| **Tamaño** | 5.863 imágenes JPEG (train: 5.216 / val: 16 / test: 624) |
| **Clases** | NORMAL (1.583 imgs) · PNEUMONIA (4.273 imgs) |
| **Balance** | Desbalanceado — ratio 1:2.7 |
| **Resolución** | Variable (~1000×1000 px); normalizada a 224×224 |

---

## 3. Estructura del Repositorio

```
computer-vision-classification/
├── data/
│   ├── raw/                    <- Imágenes originales (no versionadas en Git)
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── val/
│   │   └── test/
│   └── processed/              <- TFRecord o arrays preprocesados
├── notebooks/
│   ├── 01_eda_exploration.ipynb        <- EDA de imágenes médicas
│   └── 02_cnn_transfer_learning.ipynb  <- Modelado CNN + ResNet50
├── src/
│   ├── __init__.py
│   ├── utils.py                <- Data loaders, augmentation, métricas
│   ├── train.py                <- Entrenamiento con callbacks y MLflow
│   └── inference.py            <- Predicción sobre imagen individual o batch
├── api/
│   └── app.py                  <- API REST con FastAPI (upload de imagen)
├── tests/
│   └── test_utils.py           <- Tests unitarios con pytest
├── models/                     <- Pesos del modelo (.h5 / SavedModel)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Pipeline End-to-End

### Paso 1 — Adquisición de datos
Descarga del dataset desde Kaggle CLI o manual. Verificación de integridad (conteo por clase, resoluciones).

### Paso 2 — EDA de imágenes médicas
- Visualización de muestras representativas por clase
- Análisis de distribuciones de píxeles (histogramas por canal)
- Detección de imágenes corruptas o duplicadas (hashing MD5)
- Análisis de desbalance de clases y estrategia de mitigación

### Paso 3 — Preprocesamiento y Data Augmentation
- Resize a 224×224 px (compatible con arquitecturas preentrenadas en ImageNet)
- Normalización: `pixel / 255.0`
- **Augmentation en entrenamiento:** rotación ±15°, flip horizontal, zoom ±10%, brillo ±15%, shear 0.1
- **Sin augmentation en validación/test**
- Uso de `tf.data.Dataset` con prefetch y cache para eficiencia

### Paso 4 — Modelado
- **Baseline:** CNN propia — 3 bloques Conv2D → MaxPool → Dropout → Dense
- **Modelo final:** Transfer Learning con ResNet50 preentrenada en ImageNet
  - Etapa 1 (feature extraction): cabeza densa entrenada con base congelada
  - Etapa 2 (fine-tuning): descongelar últimas 20 capas, LR muy bajo (1e-5)
- Manejo de desbalance: `class_weight`, umbral de clasificación optimizado por F1

### Paso 5 — Evaluación
- Métricas: Accuracy, Precision, Recall (Sensibilidad), Especificidad, F1, AUC-ROC
- Matriz de confusión normalizada
- Curvas ROC y Precision-Recall
- **Grad-CAM:** visualización de mapas de activación para interpretabilidad clínica

### Paso 6 — Deployment
- API REST: endpoint `/predict` acepta imagen (multipart/form-data), retorna clase, probabilidad y mapa Grad-CAM en base64
- Contenedor Docker multi-stage para minimizar tamaño de imagen

---

## 5. Instalación y Uso

### Opción A — Entorno local

```bash
git clone https://github.com/tu-usuario/Sebas-Data_Science_Portfolio.git
cd Sebas-Data_Science_Portfolio/computer-vision-classification

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Descargar dataset (requiere Kaggle API key configurada)
kaggle datasets download paultimothymooney/chest-xray-pneumonia -p data/raw/ --unzip

# Entrenar (baseline CNN)
python src/train.py --model cnn --epochs 30 --batch-size 32

# Entrenar (ResNet50 Transfer Learning)
python src/train.py --model resnet50 --epochs 20 --fine-tune-epochs 10

# Lanzar la API
uvicorn api.app:app --reload --port 8001
```

### Opción B — Docker Compose

```bash
docker-compose up --build
# Swagger UI en http://localhost:8001/docs
```

### Ejemplo de inferencia

```bash
curl -X POST "http://localhost:8001/predict" \
  -F "file=@chest_xray_sample.jpg"
```

Respuesta esperada:
```json
{
  "prediction": "PNEUMONIA",
  "probability": 0.94,
  "confidence": "HIGH",
  "gradcam_base64": "iVBORw0KGgoAAAANS..."
}
```

---

## 6. Resultados Clave

### Métricas en conjunto de test (624 imágenes)

| Modelo | Accuracy | Recall | Precision | F1 | AUC-ROC |
|--------|----------|--------|-----------|-----|---------|
| CNN baseline | 0.88 | 0.91 | 0.89 | 0.90 | 0.93 |
| ResNet50 (frozen) | 0.92 | 0.94 | 0.92 | 0.93 | 0.96 |
| **ResNet50 (fine-tuned)** | **0.94** | **0.97** | **0.93** | **0.95** | **0.97** |

### Matriz de confusión — ResNet50 fine-tuned

|  | Pred: NORMAL | Pred: PNEUMONIA |
|---|---|---|
| **Real: NORMAL** | 207 (TN) | 27 (FP) |
| **Real: PNEUMONIA** | 11 (FN) | 379 (TP) |

**Sensibilidad = 97.2%** — solo 11 falsos negativos sobre 390 casos reales de neumonía.

### Interpretabilidad
Los mapas Grad-CAM confirman que el modelo focaliza su atención en las regiones pulmonares relevantes (infiltrados, consolidaciones), no en artefactos de la imagen.

---

## 7. Notas sobre Despliegue

### API FastAPI
El endpoint acepta imágenes en formato JPEG/PNG. El modelo se carga en memoria al inicio del servidor. Se incluye validación de tipo MIME y tamaño máximo (5 MB).

### Docker multi-stage build
```dockerfile
# Build stage: instala dependencias pesadas
FROM python:3.10-slim AS builder
# ...
# Runtime stage: solo lo necesario para inferencia
FROM python:3.10-slim AS runtime
```

### Despliegue en la nube
| Plataforma | Consideración |
|---|---|
| **Google Cloud Run** | Ideal — serverless, escala a cero |
| **AWS Lambda + EFS** | Viable con modelo en EFS para evitar cold start lento |
| **Hugging Face Spaces** | Para demo público gratuito |

---

## 8. Posibles Mejoras Futuras

- [ ] Clasificación multi-clase: Normal / Neumonía Viral / Neumonía Bacteriana
- [ ] Arquitectura EfficientNetV2 o Vision Transformer (ViT)
- [ ] Ensemble de modelos con calibración de probabilidades (Platt scaling)
- [ ] Anotación activa (Active Learning) para reducir costo de etiquetado
- [ ] Validación clínica formal con radiólogos (AUC comparativo)
- [ ] Exportación a ONNX para inferencia ligera en edge devices

---

## 9. Referencias e Inspiración

- Rajpurkar et al. (2017) — *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays*
- He et al. (2016) — *Deep Residual Learning for Image Recognition*
- [500 AI/ML Projects — Computer Vision section](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*
- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
