# Predicción de Consumo Eléctrico con LSTM/GRU

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Objetivo del Proyecto y Planteamiento del Problema

### Contexto de negocio

Las empresas distribuidoras de energía eléctrica requieren anticipar la demanda para optimizar la compra en mercados mayoristas, evitar sobrecargas en la red y reducir costos operativos. Un error de predicción del 5% en la demanda puede traducirse en pérdidas de miles de dólares diarios.

### Objetivo técnico

Construir un modelo de forecasting multivariante capaz de predecir el **consumo eléctrico residencial con un horizonte de 24 horas** (96 pasos de 15 minutos), usando redes neuronales recurrentes (LSTM y GRU) entrenadas sobre series históricas multivariantes.

### Pregunta de negocio

> ¿Con qué precisión podemos anticipar el consumo eléctrico minuto a minuto para las próximas 24 horas, dado el historial de los últimos 7 días?

---

## 2. Dataset

| Atributo | Detalle |
|---|---|
| **Nombre** | Individual Household Electric Power Consumption |
| **Origen** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption) |
| **Tamaño** | ~2.075.259 registros · 9 variables · resolución 1 minuto |
| **Período** | Diciembre 2006 – Noviembre 2010 (casi 4 años) |
| **Licencia** | Uso académico/investigación |

### Variables principales

| Variable | Descripción | Unidad |
|---|---|---|
| `Global_active_power` | Potencia activa global del hogar | kW |
| `Global_reactive_power` | Potencia reactiva | kW |
| `Voltage` | Voltaje de red | V |
| `Global_intensity` | Intensidad de corriente global | A |
| `Sub_metering_1/2/3` | Consumo por zona (cocina, lavandería, termostato) | Wh |

---

## 3. Estructura del Repositorio

```
time-series-forecast/
├── data/
│   ├── raw/                    <- Dataset original (no versionado en Git)
│   └── processed/              <- Series preprocesadas y normalizadas
├── notebooks/
│   ├── 01_eda_exploration.ipynb        <- Análisis exploratorio completo
│   └── 02_lstm_gru_modeling.ipynb      <- Modelado end-to-end
├── src/
│   ├── __init__.py
│   ├── utils.py                <- Funciones de preprocesamiento y ventanas
│   ├── train.py                <- Script de entrenamiento con MLflow
│   └── inference.py            <- Módulo de predicción
├── api/
│   └── app.py                  <- API REST con FastAPI
├── tests/
│   └── test_utils.py           <- Tests unitarios con pytest
├── models/                     <- Modelos serializados (.h5 / SavedModel)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Pipeline End-to-End

### Paso 1 — Adquisición de datos
Descarga automática desde UCI y almacenamiento en `data/raw/`. Manejo de valores faltantes (marcados como `?`) e indexación temporal.

### Paso 2 — EDA (Análisis Exploratorio)
- Distribuciones univariantes y correlaciones entre variables
- Descomposición estacional (tendencia + estacionalidad diaria/semanal)
- Detección de outliers con IQR y análisis de gaps temporales
- Visualización de autocorrelación (ACF/PACF)

### Paso 3 — Feature Engineering
- Remuestreo a frecuencia horaria (`resample('1H')`)
- Variables temporales: hora del día, día de la semana, mes, es_fin_de_semana
- Lags y ventanas rodantes (rolling mean 24h, 168h)
- Normalización MinMax por variable con scaler guardado en `models/`

### Paso 4 — Modelado
- **Baseline:** Media móvil y Naive (último valor conocido)
- **Modelo 1:** LSTM apilado (2 capas, 128 unidades + Dropout 0.2)
- **Modelo 2:** GRU bidireccional (64 unidades + BatchNorm)
- **Modelo 3:** CNN-LSTM híbrido (Conv1D → MaxPool → LSTM)
- Ventanas deslizantes de entrada 7 días → salida 24 horas
- Optimizador Adam, LR scheduler con reducción en plateau

### Paso 5 — Evaluación
- Métricas: MAE, RMSE, MAPE, R²
- Comparación vs baseline con test estadístico de Diebold-Mariano
- Visualización de predicciones por horizonte temporal

### Paso 6 — Deployment
- API REST con FastAPI: endpoint `/predict` acepta JSON con histórico y retorna forecast 24h
- Contenedor Docker listo para producción
- Tracking de experimentos con MLflow

---

## 5. Instalación y Uso

### Requisitos previos
- Python 3.10+
- Docker y Docker Compose (opcional, recomendado)

### Opción A — Entorno local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/Sebas-Data_Science_Portfolio.git
cd Sebas-Data_Science_Portfolio/time-series-forecast

# Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt

# Descargar dataset
python src/utils.py --download

# Entrenar el modelo
python src/train.py --model lstm --epochs 50 --window 168

# Lanzar la API localmente
uvicorn api.app:app --reload --port 8000
```

### Opción B — Docker Compose

```bash
docker-compose up --build
# API disponible en http://localhost:8000/docs
```

### Ejemplo de inferencia

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"historical_values": [3.2, 3.1, 2.8, ...], "horizon": 24}'
```

---

## 6. Resultados Clave

### Métricas en conjunto de test (año 2010)

| Modelo | MAE (kWh) | RMSE (kWh) | MAPE (%) | R² |
|--------|-----------|------------|----------|----|
| Naive baseline | 0.42 | 0.61 | 18.3% | 0.41 |
| Media móvil 24h | 0.38 | 0.54 | 16.1% | 0.52 |
| **LSTM apilado** | **0.18** | **0.26** | **7.4%** | **0.87** |
| GRU Bidireccional | 0.20 | 0.29 | 8.1% | 0.85 |
| CNN-LSTM | 0.19 | 0.27 | 7.8% | 0.86 |

### Impacto estimado de negocio

- Reducción del error de previsión del 18% al 7.4% → **ahorro estimado del 3-5% en costos de compra en mercado spot**
- El modelo LSTM supera al baseline en un **57% medido en MAE**

---

## 7. Notas sobre Despliegue

### API con FastAPI
El endpoint `/predict` acepta un array de valores históricos y retorna el forecast como JSON con intervalos de confianza (basados en dropout inference en tiempo de test).

### Docker
```bash
docker build -t ts-forecast-api .
docker run -p 8000:8000 ts-forecast-api
```

### Despliegue en la nube (opciones)
| Plataforma | Comando |
|---|---|
| **Railway** | `railway up` |
| **Google Cloud Run** | `gcloud run deploy` |
| **AWS ECS** | `aws ecs create-service` |

---

## 8. Posibles Mejoras Futuras

- [ ] Implementar Temporal Fusion Transformer (TFT) para interpretabilidad
- [ ] Añadir variables exógenas: temperatura, festividades, precio spot
- [ ] Pipeline de reentrenamiento automático con Airflow o Prefect
- [ ] Monitoreo de data drift con Evidently AI
- [ ] Soporte multi-hogar (arquitectura multi-task)

---

## 9. Referencias e Inspiración

- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- [500 AI/ML Projects — Time Series section](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- Lim et al. (2021) — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
- [UCI Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
- François Chollet — *Deep Learning with Python*, Cap. 6
