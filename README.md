# Sebastián — Data Science Portfolio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Sobre mí

Soy un **Data Scientist mid-level** con 2-4 años de experiencia diseñando soluciones de Machine Learning y Deep Learning para problemas reales de negocio. Me especializo en llevar proyectos desde la exploración de datos hasta el despliegue en producción, con énfasis en reproducibilidad, calidad de código y comunicación de resultados.

- Experiencia en industrias: **Finanzas, Salud, Energía**
- Apasionado por la intersección entre estadística rigurosa e ingeniería de software
- Contacto: [zebas.sr@gmail.com](mailto:zebas.sr@gmail.com) · [LinkedIn](#) · [GitHub](#)

---

## Habilidades Técnicas

### Lenguajes y entornos
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![SQL](https://img.shields.io/badge/-SQL-4479A1?logo=postgresql&logoColor=white&style=flat)
![Bash](https://img.shields.io/badge/-Bash-4EAA25?logo=gnubash&logoColor=white&style=flat)

### Machine Learning y Deep Learning
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=flat)
![Keras](https://img.shields.io/badge/-Keras-D00000?logo=keras&logoColor=white&style=flat)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)
![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600?style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-02A2AB?style=flat)

### Datos y visualización
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white&style=flat)

### MLOps y despliegue
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white&style=flat)
![Docker](https://img.shields.io/badge/-Docker-2496ED?logo=docker&logoColor=white&style=flat)
![MLflow](https://img.shields.io/badge/-MLflow-0194E2?logo=mlflow&logoColor=white&style=flat)
![Git](https://img.shields.io/badge/-Git-F05032?logo=git&logoColor=white&style=flat)

---

## Proyectos Destacados

| # | Proyecto | Dominio | Stack Principal | Métrica Clave |
|---|----------|---------|-----------------|---------------|
| 1 | [Predicción de Consumo Eléctrico con LSTM](./time-series-forecast/) | Series Temporales | TensorFlow · LSTM · FastAPI | RMSE: 0.18 kWh |
| 2 | [Detección de Neumonía por Rayos X](./computer-vision-classification/) | Computer Vision | TF · ResNet50 · Transfer Learning | AUC-ROC: 0.97 |
| 3 | [Credit Scoring y Segmentación de Clientes](./credit-scoring-risk/) | Finanzas / Riesgo | XGBoost · LightGBM · KMeans | Gini: 0.71 |

---

### Proyecto 1 — Predicción de Consumo Eléctrico (LSTM/GRU)

> Forecasting multivariante de series temporales con LSTM y GRU para anticipar el consumo eléctrico residencial en un horizonte de 24 horas.

**Dataset:** UCI Household Electric Power Consumption (~2 M registros, 9 variables)  
**Pipeline:** Ingesta → EDA → Feature Engineering → LSTM/GRU → Evaluación → API REST

[Ver proyecto completo →](./time-series-forecast/)

---

### Proyecto 2 — Detección de Neumonía por Radiografía (CNN / Transfer Learning)

> Clasificación binaria (Normal vs Neumonía) mediante CNN propia y Transfer Learning con ResNet50 sobre imágenes de rayos X torácicos.

**Dataset:** Chest X-Ray Images (Kaggle / NIH) — 5.863 imágenes  
**Pipeline:** Preprocesamiento → Data Augmentation → CNN baseline → Fine-tuning ResNet50 → API REST

[Ver proyecto completo →](./computer-vision-classification/)

---

### Proyecto 3 — Credit Scoring y Segmentación de Riesgo (XGBoost / LightGBM)

> Modelado de probabilidad de impago (PD) con Gradient Boosting y segmentación de cartera crediticia con clustering no supervisado.

**Dataset:** Give Me Some Credit (Kaggle) — 150.000 registros  
**Pipeline:** EDA → Feature Engineering → XGBoost/LightGBM → Scorecard → KMeans → API REST

[Ver proyecto completo →](./credit-scoring-risk/)

---

## Estructura del Repositorio

```
Sebas-Data_Science_Portfolio/
├── README.md                          <- Esta página
├── time-series-forecast/              <- Proyecto 1: Series Temporales
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   ├── api/
│   ├── tests/
│   ├── Dockerfile
│   └── docker-compose.yml
├── computer-vision-classification/    <- Proyecto 2: Computer Vision
│   ├── README.md
│   ├── notebooks/
│   ├── src/
│   ├── api/
│   ├── tests/
│   ├── Dockerfile
│   └── docker-compose.yml
└── credit-scoring-risk/               <- Proyecto 3: Credit Scoring
    ├── README.md
    ├── notebooks/
    ├── src/
    ├── api/
    ├── tests/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Filosofía de Trabajo

- **Reproducibilidad primero:** todos los entornos están contenedorizados con Docker.
- **Código limpio:** funciones modulares, type hints, logging estructurado y tests con pytest.
- **Orientado a negocio:** cada proyecto incluye métricas técnicas y de impacto de negocio.
- **Documentación como ciudadano de primera clase:** notebooks narrativos + README detallados.

---

## Licencia

Distribuido bajo licencia [MIT](LICENSE). Libre para uso educativo y referencia profesional.
