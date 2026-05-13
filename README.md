# Sebastián — Data Science Portfolio

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![CI](https://github.com/tu-usuario/Sebas-Data_Science_Portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/tu-usuario/Sebas-Data_Science_Portfolio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Sobre mí

Soy un **Data Scientist mid-level** con 2-4 años de experiencia diseñando soluciones de Machine Learning y Deep Learning para problemas reales de negocio. Me especializo en llevar proyectos desde la exploración de datos hasta el despliegue en producción, con énfasis en reproducibilidad, calidad de código y comunicación de resultados.

- Experiencia en industrias: **Fintech, Finanzas, Salud, Energía**
- Apasionado por la intersección entre estadística rigurosa, econometría e ingeniería de software
- Contacto: [juan.ssr@hotmail.com](mailto:juan.ssr@hotmail.com) · [LinkedIn](https://www.linkedin.com/in/juan-sebastian-segura-abi/) · [GitHub](https://github.com/jusero1)

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
![XGBoost](https://img.shields.io/badge/-XGBoost-FF6600?style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-02A2AB?style=flat)

### Datos y visualización
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=flat)
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

> Forecasting multivariante de series temporales para anticipar el consumo eléctrico residencial en un horizonte de 24 horas.

**Dataset:** UCI Household Electric Power Consumption (~2 M registros, 9 variables)
[Ver proyecto →](./time-series-forecast/)

---

### Proyecto 2 — Detección de Neumonía por Radiografía (CNN / Transfer Learning)

> Clasificación binaria (Normal vs Neumonía) con ResNet50 fine-tuned y Grad-CAM para interpretabilidad clínica.

**Dataset:** Chest X-Ray Images (Kaggle / NIH) — 5.863 imágenes
[Ver proyecto →](./computer-vision-classification/)

---

### Proyecto 3 — Credit Scoring y Segmentación de Riesgo (XGBoost / LightGBM)

> Modelado de probabilidad de impago con Gradient Boosting + scorecard FICO-like + segmentación KMeans.

**Dataset:** Give Me Some Credit (Kaggle) — 150.000 registros
[Ver proyecto →](./credit-scoring-risk/)

---

## Estructura del Repositorio

```
Sebas-Data_Science_Portfolio/
├── README.md
├── SECURITY.md                        <- Política de seguridad y reporte de vulnerabilidades
├── LICENSE
├── .gitignore                         <- Incluye .env, modelos, datos
├── .env.example                       <- Plantilla de variables de entorno (sin valores reales)
├── .pre-commit-config.yaml            <- Hooks: ruff, black, nbstripout, detect-secrets
├── pyproject.toml                     <- Configuración de ruff, black, pytest
├── .github/
│   └── workflows/
│       └── ci.yml                     <- CI: lint + tests + pip-audit + Docker build
├── time-series-forecast/
├── computer-vision-classification/
└── credit-scoring-risk/
    └── (cada proyecto contiene: src/ api/ tests/ notebooks/ Dockerfile docker-compose.yml)
```

---

## Inicio Rápido

### Requisitos previos
- Python 3.11+
- Docker y Docker Compose
- `pre-commit` (para desarrollo)

### Configuración del entorno local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/Sebas-Data_Science_Portfolio.git
cd Sebas-Data_Science_Portfolio

# 2. Configurar variables de entorno (nunca edites .env directamente en Git)
cp .env.example .env
# Edita .env con tus valores locales

# 3. Instalar hooks de pre-commit
pip install pre-commit
pre-commit install
# A partir de aquí, cada commit limpiará notebooks y verificará código automáticamente

# 4. Escanear dependencias en busca de CVEs (ejecutar antes de cada despliegue)
pip install pip-audit
pip-audit -r time-series-forecast/requirements.txt
```

### Levantar cualquier proyecto con Docker

```bash
cd time-series-forecast          # o computer-vision-classification / credit-scoring-risk
cp .env.example .env             # Configura variables locales
docker-compose up --build
# API disponible en http://localhost:8000/docs
```

---

## Seguridad

> Ver [SECURITY.md](SECURITY.md) para la política completa de seguridad y cómo reportar vulnerabilidades.

### Prácticas implementadas

| Área | Medida |
|------|--------|
| **Secretos** | Variables de entorno con `.env.example`; `.env` en `.gitignore` |
| **Dependencias** | Versiones fijadas exactamente; `pip-audit` en CI |
| **Contenedores** | `python:3.11-slim`, multi-stage, usuario sin privilegios (`uid=1000`) |
| **APIs** | CORS restrictivo, rate limiting (`slowapi`), cabeceras de seguridad HTTP |
| **Notebooks** | Outputs limpiados automáticamente con `nbstripout` (pre-commit) |
| **Código** | `ruff` + `black` + `detect-secrets` en pre-commit; CI/CD en GitHub Actions |

### Reportar una vulnerabilidad

Envía un correo a **zebas.sr@gmail.com** con el asunto `[SECURITY]`.
**No abras issues públicos** para vulnerabilidades de seguridad.

---

## Filosofía de Trabajo

- **Reproducibilidad primero:** entornos contenedorizados con Docker y dependencias fijadas.
- **Seguridad por defecto:** ningún secreto en el código, CORS restrictivo, usuarios no-root.
- **Código limpio:** type hints, logging estructurado, tests con pytest, pre-commit hooks.
- **Orientado a negocio:** métricas técnicas + impacto estimado en cada proyecto.

---

## Licencia

Distribuido bajo licencia [MIT](LICENSE). Libre para uso educativo y referencia profesional.
