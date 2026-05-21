# 🏦 Diccionario de Riesgo Crediticio y Estrategias de Recuperación
> Referencia técnica para analistas de cartera, modelos de scoring y gestión de cobranza.
> Contexto regulatorio: Colombia — Superintendencia Financiera, IFRS 9, Circular Básica Contable.

---

## 1. Métricas Fundamentales de Pérdida Esperada

### PD — Probabilidad de Default *(Probability of Default)*
Probabilidad de que un deudor incumpla sus obligaciones en un horizonte de tiempo determinado (generalmente 12 meses). Se estima mediante modelos de scoring (regresión logística, XGBoost) y es el insumo principal para calcular provisiones.

> **Clave técnica:** la PD no es un "sí/no" binario; es una tasa continua entre 0 y 1 que debe calibrarse periódicamente. Un modelo con buena discriminación (Gini alto) puede aun así tener PDs mal calibradas si no se aplica Platt Scaling o calibración isotónica.

**Tipos de PD según horizonte:**

| Tipo | Horizonte | Uso |
|---|---|---|
| PD 12m (IFRS 9 Stage 1) | 12 meses | Provisión para cartera sana |
| PD lifetime (IFRS 9 Stage 2/3) | Vida del crédito | Provisión para cartera deteriorada |
| PD punto en el tiempo | Ciclo económico actual | Scoring de originación |
| PD a través del ciclo | Promedio del ciclo | Capital regulatorio (Basilea) |

---

### LGD — Pérdida dado el Default *(Loss Given Default)*
Porcentaje de la exposición que el banco perdería si el cliente incumple, después de considerar recuperaciones y garantías.

**Fórmula:** `LGD = 1 − Tasa de Recuperación`

**Ejemplo:** Cliente debe $10.000.000 y se estima recuperar $2.000.000 → LGD = 80%.

> **Clave técnica:** la LGD varía por producto, tipo de garantía y altura de mora. En créditos de consumo sin garantía (libranzas, tarjetas) la LGD típica oscila entre 60% y 90%. En crédito hipotecario puede bajar a 20-30% dependiendo del LTV.

**Factores que reducen la LGD:**

- Garantías reales (hipoteca, prenda) bien valoradas y ejecutables.
- Velocidad de gestión jurídica (a mayor tiempo, mayor costo financiero de recuperación).
- Segmentación temprana del cliente para estrategias de curación vs. castigo.
- Tasa de descuento aplicada a los flujos de recuperación (valor presente de lo recuperado).

---

### EAD — Exposición al Default *(Exposure at Default)*
Monto total al que el banco está expuesto en el momento del incumplimiento.

> **Clave técnica:** en productos rotativos (tarjetas, cupos de crédito), la EAD puede ser mayor al saldo actual porque el cliente puede utilizar el cupo disponible antes de caer en default. Para estos casos se usa el **Factor de Conversión de Crédito (CCF o LEQ)**.

**Fórmula con CCF:**
`EAD = Saldo Actual + CCF × Cupo Disponible`

Un CCF de 0.5 sobre un cupo disponible de $5.000.000 agrega $2.500.000 a la exposición estimada.

---

### EL — Pérdida Esperada *(Expected Loss)*

```
EL = PD × LGD × EAD
```

Es el costo promedio esperado por riesgo de crédito y la base para calcular provisiones contables. Cualquier variación en PD, LGD o EAD afecta directamente la rentabilidad ajustada por riesgo (RAROC) y los requerimientos de capital.

**Ejemplo numérico:**

| Componente | Valor |
|---|---|
| PD | 5% |
| LGD | 70% |
| EAD | $10.000.000 |
| **EL** | **$350.000** |

---

## 2. Normativa y Provisiones (Contexto Colombiano)

### Provisiones e IFRS 9
Son los recursos que el banco separa para cubrir la pérdida esperada de su cartera. En Colombia la Superintendencia Financiera exige modelos de pérdida esperada bajo IFRS 9 o el estándar local (Capítulo II de la Circular Básica Contable).

**Tipos de provisión:**

- **Provisión individual:** calculada crédito a crédito con base en altura de mora y calificación de riesgo.
- **Provisión general:** porcentaje adicional sobre cartera sana como colchón ante deterioro no identificado.
- **Provisión dinámica (contracíclica):** acumulada en períodos de bonanza para utilizarse en períodos de estrés. Exigida históricamente por la SFC como estabilizador.

**Las tres etapas IFRS 9:**

| Stage | Condición | Base de provisión |
|---|---|---|
| Stage 1 | Sin deterioro significativo | PD 12 meses |
| Stage 2 | Deterioro significativo del riesgo | PD lifetime |
| Stage 3 | Default o evidencia objetiva de deterioro | PD lifetime + LGD efectiva |

> **Punto de transferencia Stage 1 → Stage 2:** la SFC acepta como criterio cuantitativo un incremento relativo de PD mayor al doble desde la originación, o mora mayor a 30 días como criterio presuntivo.

---

### Calificación de Cartera por Altura de Mora

| Categoría | Rango de mora | Riesgo | Provisión mínima orientativa |
|---|---|---|---|
| A — Normal | 0–29 días | Normal | 1% |
| B — Aceptable | 30–59 días | Aceptable | 3.2% |
| C — Apreciable | 60–89 días | Apreciable | 20% |
| D — Significativo | 90–119 días | Significativo | 50% |
| E — Incobrabilidad | 120+ días | Incobrabilidad | 100% |

> **Nota regulatoria:** los porcentajes anteriores son de referencia (modelo estándar). Los bancos con modelos internos aprobados por la SFC pueden usar porcentajes calibrados con datos propios, generalmente más precisos pero sujetos a validación periódica.

---

## 3. Análisis Dinámico de Cartera

### Cosechas — Vintage Analysis
Seguimiento longitudinal del comportamiento de créditos originados en el mismo período (mes o trimestre). Se grafica el porcentaje de cartera vencida (30+ dpd, 60+ dpd, 90+ dpd) a lo largo de su vida.

**Para qué sirve:**
- Detectar deterioro temprano en originaciones recientes vs. cosechas anteriores.
- Evaluar el impacto de cambios en política de crédito sobre la calidad futura de cartera.
- Estimar la curva de maduración esperada para provisionar anticipadamente.

**Señal de alerta:** si una cosecha reciente supera en más de 2 puntos porcentuales la mora de cosechas anteriores al mismo mes de vida, se dispara revisión de política de originación.

**Estructura típica de una tabla vintage (SQL/Python):**

```
Mes_Originación | Mes_Vida_1 | Mes_Vida_2 | ... | Mes_Vida_N
2024-01         |   0.8%     |   1.9%     | ... |   5.2%
2024-02         |   1.1%     |   2.4%     | ... |   6.1%
```

---

### Roll Rates — Matrices de Transición
Miden el porcentaje de clientes (o saldo) que migran de un estado de mora a otro en un período dado. Insumo crítico para pronóstico de flujos hacia el default y para dimensionamiento de cobranza.

**Matriz típica de transición mensual:**

| Desde \ Hasta | Al día | 1–30 | 30–60 | 60–90 | 90+ | Pagó total |
|---|---|---|---|---|---|---|
| Al día | 92% | 6% | 1.5% | 0.3% | 0.1% | 0.1% |
| 1–30 | 25% | 58% | 12% | 3% | 1% | 1% |
| 30–60 | 8% | 5% | 65% | 15% | 5% | 2% |
| 60–90 | 3% | 2% | 8% | 62% | 22% | 3% |
| 90+ | 1% | 1% | 2% | 6% | 88% | 2% |

**Interpretación estratégica:**
- **Forward roll rate:** tasa de migración hacia mora más alta → mide agravamiento.
- **Cure rate (backward roll):** tasa de retorno a corriente → mide efectividad de cobranza temprana.
- **Steady-state default rate:** si los roll rates son estables, se puede calcular la tasa de default a largo plazo como estado estacionario de la cadena de Markov.

> **Para equipos de cobranza:** un incremento de 3+ puntos porcentuales en el roll de 30–60 hacia 60–90 durante dos meses consecutivos es señal de que la estrategia de cobranza de mora media debe reforzarse.

---

### Curva de Maduración de Mora (DPD Vintage)
Evolución del porcentaje de cartera vencida promedio a medida que el crédito envejece. Permite estimar la **mora pico esperada** (generalmente se alcanza entre el mes 12 y 24 de vida del crédito en consumo) y comparar cosechas entre sí.

**Uso en provisiones:** con la curva de maduración y el saldo proyectado de la cartera, se puede estimar el stock de cartera vencida futura y constituir provisiones anticipadas.

---

## 4. Modelos Analíticos y Métricas de Evaluación

### Tipos de Scoring

| Tipo | Momento de uso | Variables clave | Output |
|---|---|---|---|
| Application Score | Originación | Buró, demográficas, ingresos | Aprobar / rechazar / condiciones |
| Behavioral Score | Gestión periódica | Transaccional, saldo, uso de cupo | Gestión proactiva, límites |
| Collection Score | Cobranza | DPD, historial de pagos, score previo | Prioridad y canal de gestión |
| Attrition Score | Retención | Actividad, quejas, uso de producto | Intervención preventiva |

---

### KS — Kolmogorov-Smirnov
Mide la separación máxima entre la distribución acumulada de scores de "buenos" (no default) y "malos" (default).

**Interpretación:**

| Rango KS | Interpretación |
|---|---|
| < 20% | Poder discriminatorio pobre |
| 20% – 30% | Aceptable |
| 30% – 40% | Bueno |
| 40% – 50% | Muy bueno |
| > 50% | Excelente (verificar sobreajuste) |

> **Uso operativo:** el KS identifica el punto de corte óptimo en el score donde la separación entre buenos y malos es máxima. Ese corte no siempre es el punto de decisión de negocio, pero sirve como referencia técnica.

---

### Gini / AUC-ROC

```
Gini = 2 × AUC − 1
```

Mide la capacidad del modelo para ordenar a los deudores de mejor a peor riesgo. En banca colombiana se reporta habitualmente el Gini en lugar del AUC.

**Tabla de referencia para cartera de consumo:**

| Gini | Calidad del modelo |
|---|---|
| < 30% | Inaceptable para producción |
| 30% – 45% | Mínimo regulatorio interno |
| 45% – 60% | Estándar de industria |
| 60% – 75% | Bueno, competitivo |
| > 75% | Excelente (revisar si hay data leakage) |

---

### Métricas de Clasificación (Matriz de Confusión)

```
                  Predicho: Malo    Predicho: Bueno
Real: Malo        TP (verdadero+)   FN (falso negativo)
Real: Bueno       FP (falso+)       TN (verdadero negativo)
```

| Métrica | Fórmula | Relevancia en riesgo |
|---|---|---|
| Precisión | TP / (TP + FP) | Evitar asfixiar clientes buenos en cobranza |
| Recall (Sensibilidad) | TP / (TP + FN) | Cubrir pérdidas — capturar todos los malos |
| Especificidad | TN / (TN + FP) | Aprobación de clientes buenos |
| F1-Score | 2 × (Prec × Rec) / (Prec + Rec) | Balance en datos desbalanceados |
| KS | max(CDF_malos − CDF_buenos) | Discriminación operativa |

> **Dilema de negocio:** maximizar recall captura más malos pero genera más falsos positivos (buenos rechazados o en cobranza innecesaria). El punto de corte del score debe optimizarse según el costo relativo de cada error, no solo la métrica técnica.

---

### PSI — Population Stability Index
Mide cuánto ha cambiado la distribución del score (o de una variable) entre la muestra de desarrollo y la producción actual.

```
PSI = Σ (% Actual − % Esperado) × ln(% Actual / % Esperado)
```

**Interpretación:**

| PSI | Acción recomendada |
|---|---|
| < 0.10 | Distribución estable — sin acción |
| 0.10 – 0.25 | Cambio moderado — investigar |
| > 0.25 | Deriva significativa — recalibrar o reentrenar el modelo |

> **En producción (Databricks / MLflow):** el PSI debe calcularse mensualmente por variable predictora (CSI — Characteristic Stability Index) y por score total. Un dashboard de monitoreo que dispare alertas automáticas cuando PSI > 0.15 es práctica estándar en equipos maduros de riesgo.

---

### Reject Inference
**El sesgo de selección más crítico en credit scoring.** Los modelos se entrenan con clientes aprobados (de quienes se observa el comportamiento), pero en producción también deben evaluar perfiles similares a los que históricamente se rechazaron, de quienes no se sabe cómo habrían pagado.

**Técnicas comunes:**

| Técnica | Descripción | Cuándo usar |
|---|---|---|
| Augmentation | Asignar PDs estimadas a rechazados y entrenar con todos | Tasa de rechazo < 30% |
| Parceling | Aprobar aleatoriamente una muestra de rechazados para observar comportamiento | Disponibilidad de cupo piloto |
| Fuzzy Augmentation | Variante probabilística del augmentation | Tasa de rechazo alta |
| Extrapolation | Extrapolar el modelo más allá del umbral de corte | Poblaciones con distribución continua |

> No documentar la estrategia de reject inference en el portafolio es uno de los errores más frecuentes que detecta un revisor especializado. La omisión sugiere que el Gini reportado puede estar sobreestimado.

---

## 5. Gestión de Recuperaciones y Cobranza

### Ciclo de Vida de la Mora y Estrategias por Tramo

| Tramo de mora | Nombre operativo | Estrategia principal | Canal |
|---|---|---|---|
| 1–30 dpd | Mora temprana | Recordatorio preventivo | SMS, WhatsApp, autogestión |
| 31–60 dpd | Mora media | Negociación activa, acuerdos de pago | Call center, app |
| 61–90 dpd | Mora grave | Reestructuración, quita de intereses | Asesor especializado |
| 91–180 dpd | Mora avanzada | Gestión externa, cobranza prejudicial | Agencia externa |
| 180+ dpd | Pre-castigo | Acuerdo total o inicio de proceso jurídico | Jurídico |
| Castigada | Castigo contable | Recuperación post-castigo | Gestor jurídico / venta de cartera |

---

### Tasa de Recuperación Acumulada
Porcentaje del monto vencido recuperado en un horizonte de tiempo después del inicio de la gestión.

```
Tasa de Recuperación = Σ Pagos Recibidos (t=1 a T) / Saldo Vencido Inicial
```

Se presenta por:
- Tramo de mora inicial (recuperación de cartera temprana vs. avanzada).
- Canal de cobranza (call center, jurídico, agencia, autogestión).
- Segmento de cliente (PD al momento de entrar en mora, producto, región).

> **Benchmark de industria en consumo sin garantía (Colombia):** recuperación a 12 meses de cartera 60–90 dpd oscila entre 35% y 55% dependiendo del canal y el perfil del cliente. Cartera 180+ dpd: 5%–20%.

---

### Cure Rate (Tasa de Curación)
Porcentaje de clientes en mora que regresan a corriente (al día) en un período dado. Es el indicador clave de efectividad de cobranza temprana y el complemento del roll rate.

```
Cure Rate (tramo X, mes M) = Clientes que estaban en tramo X en M-1 y están al día en M / Total en tramo X en M-1
```

**Niveles de referencia:**

| Tramo | Cure Rate esperado (consumo) |
|---|---|
| 1–30 dpd | 55% – 75% |
| 31–60 dpd | 25% – 40% |
| 61–90 dpd | 10% – 20% |
| 91–120 dpd | 5% – 12% |

---

### CEI — Collection Effectiveness Index
Indicador de eficiencia de cobranza: qué fracción de la deuda exigible fue recuperada en el período.

```
CEI = Pagos recibidos en el mes / Saldo vencido al inicio del mes
```

Versión ajustada (más precisa):

```
CEI_ajustado = Pagos / (Saldo vencido inicial + Nuevas entradas a mora − Castigos del período)
```

**Uso:** comparar la efectividad de distintos canales, agencias externas o estrategias de contacto. Un CEI superior en el canal digital vs. call center, controlando por perfil de cliente, justifica inversión en autogestión.

---

### Roll a Castigo — Write-off
Momento en que el banco reconoce contablemente que un crédito es irrecuperable y lo retira del balance. La altura de mora para castigo varía por política interna (generalmente 180 o 360 dpd en consumo, 360 en hipotecario).

> **Punto clave:** el castigo contable no implica abandono de la gestión de cobro. La cartera castigada sigue siendo gestionada (internamente o mediante venta a fondos de deuda), y las recuperaciones post-castigo se registran como ingresos extraordinarios. Modelar la curva de recuperación post-castigo es parte del cálculo de LGD.

---

### Segmentación Estratégica en Cobranza
Uso del collection score, la altura de mora y variables de comportamiento para asignar estrategias diferenciadas. El objetivo es maximizar la recuperación minimizando el costo de gestión.

**Matriz de segmentación típica:**

| Segmento | Score comportamental | Tramo de mora | Estrategia |
|---|---|---|---|
| 1 — Alta probabilidad de curación | Alto | 1–30 | SMS + recordatorio app, sin contacto humano |
| 2 — Curación posible con negociación | Medio | 31–60 | Call center + oferta de acuerdo de pago |
| 3 — Requiere reestructuración | Bajo | 61–90 | Asesor especializado + quita de mora |
| 4 — Gestión externa | Muy bajo | 90+ | Agencia externa + campaña de acuerdo total |
| 5 — Perfil jurídico | Muy bajo | 120+ | Proceso prejudicial / jurídico |

> **Buena práctica:** no asignar el segmento únicamente por mora. Un cliente con 45 dpd pero score comportamental alto tiene mayor probabilidad de curación que uno con 20 dpd y score bajo; tratarlos igual desperdicia presupuesto de cobranza.

---

### Estrategias de Reestructuración
Modificación de condiciones del crédito para viabilizar el pago. Son la principal herramienta para evitar que un cliente migre a castigo.

| Tipo | Descripción | Cuándo aplicar |
|---|---|---|
| Refinanciación | Extensión del plazo con reducción de cuota | Mora 30–60, buena voluntad de pago |
| Reestructuración | Cambio de condiciones (tasa, plazo, garantía) | Mora 60–90, cliente con capacidad reducida |
| Novación | Reemplazo del contrato original | Mora avanzada, acuerdo negociado |
| Quita | Condonación parcial de capital o intereses | Mora 90+, recuperación parcial preferible a castigo |
| Acuerdo de pago | Plan de pagos sin cambio contractual | Mora temprana, cliente con liquidez temporal baja |

> **Riesgo de reincidencia (re-default):** entre el 30% y 50% de los créditos reestructurados vuelven a entrar en mora en los 12 meses siguientes. Monitorear el comportamiento post-reestructuración con un behavioral score actualizado es crítico para evaluar la efectividad real de la estrategia.

---

### Venta de Cartera (Portfolio Sale)
Mecanismo de transferencia de cartera castigada o en mora avanzada a fondos de inversión o agencias especializadas, a cambio de un precio (generalmente 2%–15% del valor nominal según el tramo y el producto).

**Criterios para decidir venta vs. gestión interna:**

- Costo de gestión interna por peso recuperado vs. precio ofrecido.
- Capacidad operativa del equipo de cobranza jurídica.
- Impacto en el capital liberado (la venta mejora los indicadores de cartera vencida).
- Obligaciones regulatorias de reporte (la SFC monitorea la calidad de la cartera pre y post-venta).

---

## 6. Monitoreo de Modelos en Producción

### Ciclo Mínimo de Monitoreo

| Frecuencia | Indicador | Umbral de alerta |
|---|---|---|
| Mensual | PSI del score | > 0.15 |
| Mensual | CSI por variable | > 0.15 en variables top-10 |
| Trimestral | Gini en muestra reciente | Caída > 5 puntos vs. desarrollo |
| Trimestral | KS en muestra reciente | Caída > 5 puntos vs. desarrollo |
| Semestral | Calibración (Brier Score / curva de calibración) | Desviación sistemática en bandas de score |
| Anual | Backtesting completo y reentrenamiento | — |

---

### Champion / Challenger
Framework estándar de industria para validar modelos nuevos en producción sin reemplazar el modelo incumbente de golpe.

- **Champion:** modelo en producción actual con todo el volumen de decisiones.
- **Challenger:** modelo nuevo que recibe un porcentaje del tráfico (generalmente 5%–20%).
- Tras un período de observación (3–6 meses), se comparan métricas de discriminación, calibración y rentabilidad ajustada por riesgo.

> Si el challenger supera al champion con significancia estadística, se promueve a producción. Si no, se descarta o ajusta. Documentar este proceso es exigencia de los marcos de validación de modelos (SR 11-7 en EE.UU., equivalentes de la SFC en Colombia).

---

### Análisis de Vintage para Monitoreo de Modelos
Las cosechas también sirven para monitorear si el modelo mantiene su poder predictivo en originaciones recientes. Si la curva de mora de la cosecha 2024-Q3 se separa significativamente de la curva predicha por el modelo calibrado en 2023, es señal de deriva del modelo más allá del PSI del score.

---

## 7. Glosario Rápido de Referencia

| Término | Definición resumida |
|---|---|
| DPD | Days Past Due — días de mora |
| NPL | Non-Performing Loan — cartera vencida (generalmente 90+ dpd) |
| NPL Ratio | Saldo vencido 90+ / Saldo total de cartera |
| Coverage Ratio | Provisiones / Saldo vencido — qué tan cubierta está la mora |
| LTV | Loan-to-Value — relación préstamo / valor de la garantía |
| CCF | Credit Conversion Factor — factor de uso de cupos disponibles |
| PDO | Points to Double the Odds — escala estándar de scorecard |
| WOE | Weight of Evidence — transformación de variables para modelos logísticos |
| IV | Information Value — poder predictivo de una variable (> 0.3 = fuerte) |
| RAROC | Return on Risk-Adjusted Capital — rentabilidad ajustada por riesgo |
| CEI | Collection Effectiveness Index — eficiencia de cobranza |
| PSI | Population Stability Index — estabilidad de la distribución del score |
| CSI | Characteristic Stability Index — PSI por variable individual |
| Stage 1/2/3 | Clasificación IFRS 9 según nivel de deterioro crediticio |

---

*Documento de referencia interna · Actualizado 2026 · Contexto regulatorio: SFC Colombia / IFRS 9 / Basilea II-III*
