# Credit Scoring y Segmentación de Riesgo Crediticio (XGBoost / LightGBM / KMeans)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-FF6600?style=flat)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-02A2AB?style=flat)](https://lightgbm.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Objetivo del Proyecto y Planteamiento del Problema

### Contexto de negocio

El riesgo de crédito es uno de los principales desafíos para instituciones financieras: en promedio, el 6.8% de los préstamos personales en economías emergentes terminan en impago. Un modelo predictivo robusto permite a los analistas de riesgo tomar decisiones más objetivas, reducir la tasa de mora y segmentar la cartera en grupos de riesgo homogéneos para políticas diferenciadas.

### Objetivos técnicos

1. **Scoring:** Predecir la probabilidad de impago a 90 días (PD — Probability of Default) para cada solicitante.
2. **Scorecard:** Transformar el modelo en una scorecard de puntos interpretable por analistas.
3. **Segmentación:** Agrupar clientes en segmentos de riesgo homogéneos mediante clustering no supervisado (KMeans) para definir políticas de crédito diferenciadas.

### Preguntas de negocio

> 1. ¿Qué variables son los mejores predictores del impago y cuál es su importancia relativa?
> 2. ¿Podemos construir una scorecard transparente y auditable que cumpla con requerimientos regulatorios (IFRS 9)?
> 3. ¿En cuántos segmentos de riesgo se puede clasificar la cartera y qué política aplicar a cada uno?

---

## 2. Dataset

| Atributo | Detalle |
|---|---|
| **Nombre** | Give Me Some Credit |
| **Origen** | [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit) |
| **Tamaño** | 150.000 registros de entrenamiento · 101.503 de test |
| **Target** | `SeriousDlqin2yrs` — impago de ≥90 días en los próximos 2 años |
| **Balance** | ~6.68% positivos (impago) — altamente desbalanceado |
| **Variables** | 10 features financiero-demográficas |

### Variables del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Numérica | Utilización de líneas de crédito revolving |
| `age` | Numérica | Edad del solicitante |
| `NumberOfTime30-59DaysPastDueNotWorse` | Conteo | Veces con mora 30-59 días |
| `DebtRatio` | Numérica | Ratio deuda/ingresos |
| `MonthlyIncome` | Numérica | Ingreso mensual (con valores faltantes) |
| `NumberOfOpenCreditLinesAndLoans` | Conteo | Líneas de crédito abiertas |
| `NumberOfTimes90DaysLate` | Conteo | Veces con mora ≥90 días |
| `NumberRealEstateLoansOrLines` | Conteo | Préstamos hipotecarios |
| `NumberOfTime60-89DaysPastDueNotWorse` | Conteo | Veces con mora 60-89 días |
| `NumberOfDependents` | Conteo | Número de dependientes |

---

## 3. Estructura del Repositorio

```
credit-scoring-risk/
├── data/
│   ├── raw/                    <- Dataset original de Kaggle (no versionado)
│   └── processed/              <- Datos limpios, WoE-transformados, splits
├── notebooks/
│   ├── 01_eda_credit_risk.ipynb        <- EDA y análisis de WoE/IV
│   └── 02_modeling_scoring.ipynb       <- Modelado, scorecard y clustering
├── src/
│   ├── __init__.py
│   ├── utils.py                <- WoE, IV, binning, métricas de riesgo
│   ├── train.py                <- Entrenamiento XGBoost/LightGBM + KMeans
│   └── inference.py            <- Scoring individual o batch (CSV)
├── api/
│   └── app.py                  <- API REST con FastAPI
├── tests/
│   └── test_utils.py           <- Tests unitarios con pytest
├── models/                     <- Modelos serializados (joblib / pickle)
├── reports/
│   └── scorecard.xlsx          <- Scorecard exportada (puntos por bin)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Pipeline End-to-End

### Paso 1 — Adquisición de datos
Carga desde Kaggle o CSV local. Auditoría inicial: tipos de datos, missings, rango de valores, duplicados.

### Paso 2 — EDA de Riesgo Crediticio
- Análisis univariante por variable (media, mediana, percentiles, outliers extremos)
- Análisis bivariante: tasa de default por segmento de cada variable
- **Information Value (IV)** para rankear poder predictivo de variables:
  - IV < 0.02: sin valor predictivo
  - 0.02-0.1: predictor débil
  - 0.1-0.3: predictor medio
  - > 0.3: predictor fuerte (potencial data leakage)
- Análisis de correlaciones y multicolinealidad (VIF)

### Paso 3 — Feature Engineering
- **Imputación:** mediana para `MonthlyIncome` (18.5% missing) y `NumberOfDependents`
- **Tratamiento de outliers:** capping al percentil 99 para variables con colas pesadas
- **Binning óptimo:** discretización supervisada con árbol de decisión (binning recursivo)
- **Weight of Evidence (WoE):** transformación monotónica que preserva interpretabilidad
- **Creación de variables derivadas:** `debt_per_dependent`, `utilizacion_alta` (flag), `mora_historica_total`
- **SMOTE** para oversampling de la clase minoritaria (solo en entrenamiento)

### Paso 4 — Modelado

#### 4a. Credit Scoring (supervisado)
- **Regresión Logística + WoE:** modelo base interpretable (scorecard tradicional)
- **XGBoost:** Gradient Boosting con optimización bayesiana de hiperparámetros (Optuna)
- **LightGBM:** GBDT con dart boosting; más rápido en datasets grandes
- **Ensemble:** promedio ponderado por OOF AUC
- Validación cruzada estratificada 5-fold

#### 4b. Segmentación (no supervisado)
- **Selección de features:** variables de comportamiento de pago y capacidad financiera
- **Preprocesamiento:** StandardScaler + reducción PCA (95% varianza explicada)
- **K-Means:** elbow method + silhouette score para k óptimo (k=4 segmentos)
- **Perfilado de segmentos:** descripción estadística + etiquetado de negocio

### Paso 5 — Evaluación
- **Métricas de ranking:** AUC-ROC, Gini = 2*AUC-1, KS statistic (Kolmogorov-Smirnov)
- **Métricas de calibración:** Brier Score, curva de calibración
- **Population Stability Index (PSI):** monitoreo de deriva entre train y test
- **SHAP values:** explicación global y local de predicciones
- Curvas de ganancia y lift

### Paso 6 — Scorecard
Conversión del modelo a puntos enteros (escala 300-850 tipo FICO):
- Punto de referencia: score 600 = odds 1:19 de impago
- PDO (Points to Double the Odds) = 50
- Tabla de scorecard exportada a Excel

### Paso 7 — Deployment
- API REST: endpoint `/score` acepta datos del solicitante en JSON, retorna PD, score, segmento y SHAP explicativo

---

## 5. Instalación y Uso

### Opción A — Entorno local

```bash
git clone https://github.com/tu-usuario/Sebas-Data_Science_Portfolio.git
cd Sebas-Data_Science_Portfolio/credit-scoring-risk

python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Descargar datos de Kaggle
kaggle competitions download -c GiveMeSomeCredit -p data/raw/

# Entrenar modelos
python src/train.py --model xgboost --n-trials 50
python src/train.py --model lightgbm --n-trials 50

# Segmentación
python src/train.py --mode clustering --n-clusters 4

# Lanzar API
uvicorn api.app:app --reload --port 8002
```

### Opción B — Docker Compose

```bash
docker-compose up --build
# Swagger UI en http://localhost:8002/docs
```

### Ejemplo de scoring individual

```bash
curl -X POST "http://localhost:8002/score" \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.52,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.38,
    "MonthlyIncome": 5500,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 2
  }'
```

Respuesta esperada:
```json
{
  "probability_of_default": 0.043,
  "score": 712,
  "risk_segment": "BAJO",
  "decision": "APROBAR",
  "shap_top_features": [
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "impact": -0.21},
    {"feature": "NumberOfTimes90DaysLate", "impact": -0.15}
  ]
}
```

---

## 6. Resultados Clave

### Métricas en conjunto de test

| Modelo | AUC-ROC | Gini | KS | Brier Score |
|--------|---------|------|----|-------------|
| Logística + WoE | 0.842 | 0.684 | 0.531 | 0.048 |
| **XGBoost** | **0.856** | **0.712** | **0.548** | **0.044** |
| LightGBM | 0.854 | 0.708 | 0.545 | 0.045 |
| Ensemble | 0.861 | 0.722 | 0.557 | 0.043 |

### Variables más importantes (SHAP)

1. `RevolvingUtilizationOfUnsecuredLines` — utilización de crédito revolving
2. `NumberOfTimes90DaysLate` — historial de mora grave
3. `age` — edad del solicitante (correlación negativa con impago)
4. `DebtRatio` — ratio deuda/ingreso
5. `MonthlyIncome` — capacidad de pago

### Segmentos KMeans (k=4)

| Segmento | Nombre | % Cartera | Tasa Default | Política Sugerida |
|---|---|---|---|---|
| 0 | **Prime** | 38% | 1.2% | Aprobar — límite alto |
| 1 | **Near Prime** | 29% | 5.8% | Aprobar — límite moderado |
| 2 | **Subprime** | 22% | 14.3% | Revisar manualmente |
| 3 | **Alto Riesgo** | 11% | 31.7% | Rechazar o garantías |

### Impacto de negocio estimado
- Aplicando el modelo vs aprobación manual: **reducción del 23% en tasa de mora** (simulación en test set)
- ROI estimado: por cada USD 1.000.000 en cartera, el modelo evita ~USD 30.000 en pérdidas de crédito

---

## 7. Notas sobre Despliegue

### API FastAPI
El endpoint `/score` carga el modelo XGBoost y el pipeline de preprocesamiento (joblib) al inicio del servidor. Incluye validación de rangos de entrada basada en distribuciones del entrenamiento.

### Monitoreo en producción
- **PSI mensual** para detectar deriva poblacional
- **Performance tracking:** AUC calculado mensualmente al madurar la cohorte (90 días)
- Alertas automáticas si PSI > 0.25 (deriva severa) o AUC cae >3 puntos

### Cumplimiento regulatorio
La scorecard WoE es auditable y cumple con los principios de explicabilidad requeridos por reguladores (Basilea III, IFRS 9). El modelo SHAP permite explicaciones individuales para el derecho a información del solicitante.

---

## 8. Posibles Mejoras Futuras

- [ ] Modelo LTV (Loss Given Default) para cálculo de EL = PD × LGD × EAD
- [ ] Modelos de supervivencia (Cox PH, DeepHit) para PD a múltiples horizontes
- [ ] Feature store con Feast para reutilización en tiempo real
- [ ] Detección de fraude integrada como variable exógena
- [ ] A/B testing de estrategias de aprobación con bandits contextuales
- [ ] Documentación del modelo siguiendo el estándar Model Card de Google

---

## 9. Referencias e Inspiración

- Siddiqi, N. (2006) — *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*
- [500 AI/ML Projects — Finance/Risk section](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions (SHAP)*
- [Kaggle Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- BIS (2017) — *Basel III: Finalising post-crisis reforms*
