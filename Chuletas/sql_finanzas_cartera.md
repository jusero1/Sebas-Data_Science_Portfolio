# SQL Aplicado a Finanzas y Recuperación de Cartera

> Problemas prácticos con soluciones que cubren funciones de ventana, CTEs, joins complejos, manejo de fechas y optimización.

---

## Contexto del modelo de datos

Se asume el siguiente esquema base a lo largo de todos los ejercicios:

```
CLIENTES          (ID_CLIENTE, NOMBRE, SEGMENTO, CIUDAD)
CREDITOS          (ID_CREDITO, ID_CLIENTE, FECHA_DESEMBOLSO, MONTO_DESEMBOLSO, PLAZO_MESES, TASA)
PAGOS             (ID_PAGO, ID_CREDITO, FECHA_PAGO, MONTO_PAGADO)
SALDOS_MENSUALES  (ID_CREDITO, PERIODO, SALDO_CAPITAL, SALDO_VENCIDO, DPD)
GESTIONES         (ID_GESTION, ID_CREDITO, FECHA_GESTION, CANAL, RESULTADO, AGENTE)
CASTIGOS          (ID_CREDITO, FECHA_CASTIGO, MONTO_CASTIGADO)
```

---

## Problema 1 — Identificar clientes que migran de cartera vigente a vencida

### Enunciado

La gerencia de riesgo necesita un reporte mensual que muestre **qué clientes pasaron de tener DPD = 0 en un mes a DPD > 0 en el mes siguiente**. Este evento se conoce como *roll forward* y es el primer indicador de deterioro de cartera.

### Tablas involucradas

`SALDOS_MENSUALES`

### Solución

```sql
WITH SaldoConMesSiguiente AS (
    SELECT
        ID_CREDITO,
        PERIODO,
        DPD,
        -- Captura el DPD del mes inmediatamente anterior para cada crédito
        LAG(DPD, 1) OVER (
            PARTITION BY ID_CREDITO
            ORDER BY PERIODO
        ) AS DPD_MES_ANTERIOR
    FROM SALDOS_MENSUALES
),
MigracionVigVenc AS (
    SELECT
        ID_CREDITO,
        PERIODO                AS MES_DETERIORO,
        DPD_MES_ANTERIOR       AS DPD_ANTES,
        DPD                    AS DPD_DESPUES
    FROM SaldoConMesSiguiente
    WHERE DPD_MES_ANTERIOR = 0
      AND DPD > 0
)
SELECT
    c.ID_CLIENTE,
    c.NOMBRE,
    c.SEGMENTO,
    m.ID_CREDITO,
    m.MES_DETERIORO,
    m.DPD_ANTES,
    m.DPD_DESPUES
FROM MigracionVigVenc m
INNER JOIN CREDITOS cr ON cr.ID_CREDITO = m.ID_CREDITO
INNER JOIN CLIENTES  c  ON c.ID_CLIENTE  = cr.ID_CLIENTE
ORDER BY m.MES_DETERIORO DESC, c.SEGMENTO;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `LAG()` | Compara el DPD del periodo actual con el del mes anterior |
| CTE | Separa la lógica de captura del LAG y la lógica de filtrado |
| `INNER JOIN` entre hechos y dimensiones | Une saldos → créditos → clientes |

### Nota de optimización

> La columna `(ID_CREDITO, PERIODO)` debe tener un índice compuesto en `SALDOS_MENSUALES`. La función `LAG` trabaja sobre una ventana ordenada; sin índice, el motor ordena la tabla completa en memoria para cada partición, lo que en tablas de millones de filas puede multiplicar el tiempo de ejecución por 10x.

---

## Problema 2 — Calcular el saldo acumulado de pagos y el saldo pendiente real por crédito

### Enunciado

El área de cobranza necesita saber, **para cada crédito activo y en cada fecha de pago**, cuánto se ha pagado en total hasta ese momento y cuánto capital sigue pendiente. Esto permite detectar créditos donde los pagos nunca alcanzan el saldo.

### Tablas involucradas

`CREDITOS`, `PAGOS`

### Solución

```sql
WITH PagosAcumulados AS (
    SELECT
        p.ID_CREDITO,
        p.FECHA_PAGO,
        p.MONTO_PAGADO,
        -- Suma acumulada de pagos ordenados cronológicamente por crédito
        SUM(p.MONTO_PAGADO) OVER (
            PARTITION BY p.ID_CREDITO
            ORDER BY p.FECHA_PAGO
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS PAGADO_ACUMULADO
    FROM PAGOS p
),
SaldoPendiente AS (
    SELECT
        pa.ID_CREDITO,
        pa.FECHA_PAGO,
        pa.MONTO_PAGADO,
        pa.PAGADO_ACUMULADO,
        cr.MONTO_DESEMBOLSO,
        cr.MONTO_DESEMBOLSO - pa.PAGADO_ACUMULADO AS SALDO_PENDIENTE
    FROM PagosAcumulados pa
    INNER JOIN CREDITOS cr ON cr.ID_CREDITO = pa.ID_CREDITO
)
SELECT
    ID_CREDITO,
    FECHA_PAGO,
    MONTO_PAGADO,
    PAGADO_ACUMULADO,
    MONTO_DESEMBOLSO,
    SALDO_PENDIENTE,
    -- Alerta si el saldo pendiente supera el 80% del monto original después de 6 pagos
    CASE
        WHEN SALDO_PENDIENTE > MONTO_DESEMBOLSO * 0.80
         AND ROW_NUMBER() OVER (PARTITION BY ID_CREDITO ORDER BY FECHA_PAGO) >= 6
        THEN 'ALERTA: PAGOS INSUFICIENTES'
        ELSE 'OK'
    END AS ESTADO_PAGO
FROM SaldoPendiente
ORDER BY ID_CREDITO, FECHA_PAGO;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `SUM() OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)` | Saldo acumulado de pagos por crédito |
| `ROW_NUMBER()` | Cuenta el número de cuota para aplicar regla de alerta |
| CTE en cadena | Primera CTE calcula acumulado; segunda aplica la lógica de negocio |

---

## Problema 3 — Calcular DPD (Days Past Due) desde la fecha de vencimiento

### Enunciado

El área de riesgo necesita calcular los **días de mora reales** de cada crédito en función de su fecha de primer vencimiento impago. Si el cliente tiene múltiples cuotas vencidas, el DPD se mide desde la **más antigua sin pagar**.

### Tablas involucradas

`CREDITOS`, `PAGOS`

### Solución

```sql
WITH CuotasEsperadas AS (
    -- Genera la tabla de cuotas teóricas usando el plazo del crédito
    SELECT
        cr.ID_CREDITO,
        cr.ID_CLIENTE,
        cr.FECHA_DESEMBOLSO,
        cr.PLAZO_MESES,
        cr.MONTO_DESEMBOLSO / cr.PLAZO_MESES          AS CUOTA_MENSUAL,
        -- Fecha de vencimiento de cada cuota (mes 1, 2, ... N)
        DATEADD(MONTH, num.n, cr.FECHA_DESEMBOLSO)    AS FECHA_VENCIMIENTO,
        num.n                                          AS NRO_CUOTA
    FROM CREDITOS cr
    -- Tabla de números del 1 al 120 (10 años) para generar el calendario
    CROSS JOIN (
        SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
        FROM SALDOS_MENSUALES  -- tabla grande usada solo como generador de filas
    ) num
    WHERE num.n <= cr.PLAZO_MESES
),
CuotasPagadas AS (
    SELECT
        ID_CREDITO,
        COUNT(*)        AS TOTAL_PAGOS,
        MAX(FECHA_PAGO) AS ULTIMO_PAGO
    FROM PAGOS
    GROUP BY ID_CREDITO
),
PrimerImpago AS (
    SELECT
        ce.ID_CREDITO,
        MIN(ce.FECHA_VENCIMIENTO) AS FECHA_PRIMER_IMPAGO
    FROM CuotasEsperadas ce
    LEFT JOIN CuotasPagadas cp ON cp.ID_CREDITO = ce.ID_CREDITO
    -- Cuotas cuya fecha de vencimiento ya pasó y no tienen pago registrado suficiente
    WHERE ce.NRO_CUOTA > COALESCE(cp.TOTAL_PAGOS, 0)
      AND ce.FECHA_VENCIMIENTO < CAST(GETDATE() AS DATE)
    GROUP BY ce.ID_CREDITO
)
SELECT
    pi.ID_CREDITO,
    c.ID_CLIENTE,
    cl.NOMBRE,
    cl.SEGMENTO,
    pi.FECHA_PRIMER_IMPAGO,
    DATEDIFF(DAY, pi.FECHA_PRIMER_IMPAGO, CAST(GETDATE() AS DATE)) AS DPD_CALCULADO,
    CASE
        WHEN DATEDIFF(DAY, pi.FECHA_PRIMER_IMPAGO, CAST(GETDATE() AS DATE)) BETWEEN 1  AND 30  THEN 'Bucket 1 (1-30)'
        WHEN DATEDIFF(DAY, pi.FECHA_PRIMER_IMPAGO, CAST(GETDATE() AS DATE)) BETWEEN 31 AND 60  THEN 'Bucket 2 (31-60)'
        WHEN DATEDIFF(DAY, pi.FECHA_PRIMER_IMPAGO, CAST(GETDATE() AS DATE)) BETWEEN 61 AND 90  THEN 'Bucket 3 (61-90)'
        WHEN DATEDIFF(DAY, pi.FECHA_PRIMER_IMPAGO, CAST(GETDATE() AS DATE)) > 90              THEN 'Bucket 4 (>90)'
        ELSE 'Al día'
    END AS BUCKET_MORA
FROM PrimerImpago pi
INNER JOIN CREDITOS cr ON cr.ID_CREDITO = pi.ID_CREDITO
INNER JOIN CLIENTES cl ON cl.ID_CLIENTE = cr.ID_CLIENTE
ORDER BY DPD_CALCULADO DESC;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `DATEADD` / `DATEDIFF` | Generar calendario de cuotas y calcular DPD |
| `CROSS JOIN` con generador de filas | Expande cada crédito a su calendario completo |
| CTE en cascada (3 niveles) | Cuotas → Pagos agrupados → Primer impago |
| `CASE WHEN` para buckets | Clasificación estándar de mora |

---

## Problema 4 — Identificar la última gestión de cobro por crédito y su resultado

### Enunciado

El supervisor de cobranza necesita saber, **para cada crédito en mora**, cuál fue el último intento de contacto, qué canal se usó, qué agente lo gestionó y cuántos días han pasado desde esa última gestión. Los créditos sin gestión en más de 15 días deben marcarse como prioritarios.

### Tablas involucradas

`GESTIONES`, `CREDITOS`, `SALDOS_MENSUALES`

### Solución

```sql
WITH UltimaGestion AS (
    SELECT
        ID_CREDITO,
        FECHA_GESTION,
        CANAL,
        RESULTADO,
        AGENTE,
        -- Numera las gestiones de cada crédito de más reciente a más antigua
        ROW_NUMBER() OVER (
            PARTITION BY ID_CREDITO
            ORDER BY FECHA_GESTION DESC
        ) AS RN
    FROM GESTIONES
),
UltimaGestionPorCredito AS (
    SELECT
        ID_CREDITO,
        FECHA_GESTION  AS ULTIMA_GESTION,
        CANAL          AS ULTIMO_CANAL,
        RESULTADO      AS ULTIMO_RESULTADO,
        AGENTE         AS ULTIMO_AGENTE,
        DATEDIFF(DAY, FECHA_GESTION, CAST(GETDATE() AS DATE)) AS DIAS_SIN_GESTION
    FROM UltimaGestion
    WHERE RN = 1  -- Solo la gestión más reciente
),
CreditosEnMora AS (
    SELECT DISTINCT ID_CREDITO
    FROM SALDOS_MENSUALES
    WHERE PERIODO = FORMAT(DATEADD(MONTH, -1, GETDATE()), 'yyyyMM')  -- Mes anterior
      AND DPD > 0
)
SELECT
    cm.ID_CREDITO,
    cr.ID_CLIENTE,
    cl.NOMBRE,
    cl.SEGMENTO,
    sm.DPD,
    sm.SALDO_VENCIDO,
    ug.ULTIMA_GESTION,
    ug.ULTIMO_CANAL,
    ug.ULTIMO_RESULTADO,
    ug.ULTIMO_AGENTE,
    ug.DIAS_SIN_GESTION,
    CASE
        WHEN ug.ULTIMA_GESTION IS NULL          THEN 'SIN GESTIÓN'
        WHEN ug.DIAS_SIN_GESTION > 15           THEN 'PRIORITARIO'
        ELSE 'GESTIONADO'
    END AS ESTADO_GESTION
FROM CreditosEnMora cm
INNER JOIN CREDITOS cr         ON cr.ID_CREDITO  = cm.ID_CREDITO
INNER JOIN CLIENTES cl         ON cl.ID_CLIENTE   = cr.ID_CLIENTE
INNER JOIN SALDOS_MENSUALES sm ON sm.ID_CREDITO   = cm.ID_CREDITO
                               AND sm.PERIODO = FORMAT(DATEADD(MONTH, -1, GETDATE()), 'yyyyMM')
LEFT JOIN UltimaGestionPorCredito ug ON ug.ID_CREDITO = cm.ID_CREDITO
ORDER BY ug.DIAS_SIN_GESTION DESC NULLS FIRST;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY DESC)` | Selecciona únicamente la última gestión |
| `LEFT JOIN` | Preserva créditos en mora aunque no tengan gestión registrada |
| `FORMAT` + `DATEADD` | Construye el periodo del mes anterior dinámicamente |
| `NULLS FIRST` en `ORDER BY` | Sube al tope los créditos sin gestión alguna |

---

## Problema 5 — Calcular el PAR30 mensual por segmento (Vintage)

### Enunciado

El comité de riesgo quiere ver la evolución del **PAR30** (cartera vencida > 30 días / cartera total) **mes a mes y por segmento de cliente**, para detectar si alguna cohorte de originación se está deteriorando más rápido que las demás.

### Tablas involucradas

`SALDOS_MENSUALES`, `CREDITOS`, `CLIENTES`

### Solución

```sql
WITH BaseMensual AS (
    SELECT
        sm.PERIODO,
        cl.SEGMENTO,
        -- Año-mes de originación del crédito (cohorte / vintage)
        FORMAT(cr.FECHA_DESEMBOLSO, 'yyyy-MM') AS VINTAGE,
        sm.SALDO_CAPITAL,
        sm.SALDO_VENCIDO,
        sm.DPD
    FROM SALDOS_MENSUALES sm
    INNER JOIN CREDITOS cr ON cr.ID_CREDITO  = sm.ID_CREDITO
    INNER JOIN CLIENTES cl ON cl.ID_CLIENTE   = cr.ID_CLIENTE
),
AgregadoMensual AS (
    SELECT
        PERIODO,
        SEGMENTO,
        VINTAGE,
        SUM(SALDO_CAPITAL)                                   AS CARTERA_TOTAL,
        SUM(CASE WHEN DPD > 30 THEN SALDO_VENCIDO ELSE 0 END) AS CARTERA_PAR30,
        COUNT(DISTINCT CASE WHEN DPD > 30 THEN 1 END)       AS CREDITOS_EN_MORA
    FROM BaseMensual
    GROUP BY PERIODO, SEGMENTO, VINTAGE
),
PAR30Calculado AS (
    SELECT
        PERIODO,
        SEGMENTO,
        VINTAGE,
        CARTERA_TOTAL,
        CARTERA_PAR30,
        CREDITOS_EN_MORA,
        ROUND(CARTERA_PAR30 * 100.0 / NULLIF(CARTERA_TOTAL, 0), 2) AS PAR30_PCT,
        -- PAR30 del mes anterior para calcular la variación
        LAG(ROUND(CARTERA_PAR30 * 100.0 / NULLIF(CARTERA_TOTAL, 0), 2), 1) OVER (
            PARTITION BY SEGMENTO, VINTAGE
            ORDER BY PERIODO
        ) AS PAR30_MES_ANTERIOR
    FROM AgregadoMensual
)
SELECT
    PERIODO,
    SEGMENTO,
    VINTAGE,
    CARTERA_TOTAL,
    CARTERA_PAR30,
    PAR30_PCT,
    PAR30_MES_ANTERIOR,
    PAR30_PCT - PAR30_MES_ANTERIOR AS VARIACION_MES,
    CASE
        WHEN PAR30_PCT - PAR30_MES_ANTERIOR > 2  THEN '↑ DETERIORO'
        WHEN PAR30_PCT - PAR30_MES_ANTERIOR < -2 THEN '↓ MEJORA'
        ELSE '→ ESTABLE'
    END AS TENDENCIA
FROM PAR30Calculado
ORDER BY PERIODO DESC, SEGMENTO, PAR30_PCT DESC;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `SUM(CASE WHEN DPD > 30 ...)` | Suma condicional para separar cartera vencida |
| `LAG()` con doble `PARTITION BY` | Compara PAR30 por segmento y vintage simultáneamente |
| `NULLIF` | Evita división por cero cuando la cartera es cero |
| CTE en 3 niveles | Base → Agregado → Cálculo del KPI con variación |

---

## Problema 6 — Productividad de agentes: monto recuperado y ranking por canal

### Enunciado

El área de operaciones quiere un reporte de **productividad por agente** que muestre el monto total recuperado en el mes, el número de gestiones efectivas (resultado = 'PAGO'), el ranking dentro de su canal y la diferencia respecto al agente top de ese canal.

### Tablas involucradas

`GESTIONES`, `PAGOS`, `CREDITOS`

### Solución

```sql
WITH GestionesConPago AS (
    -- Une cada gestión con el pago registrado en los 3 días siguientes a la gestión
    SELECT
        g.ID_GESTION,
        g.ID_CREDITO,
        g.FECHA_GESTION,
        g.CANAL,
        g.AGENTE,
        g.RESULTADO,
        p.MONTO_PAGADO
    FROM GESTIONES g
    LEFT JOIN PAGOS p
        ON  p.ID_CREDITO  = g.ID_CREDITO
        AND p.FECHA_PAGO BETWEEN g.FECHA_GESTION AND DATEADD(DAY, 3, g.FECHA_GESTION)
    WHERE g.RESULTADO = 'PAGO'
      AND FORMAT(g.FECHA_GESTION, 'yyyy-MM') = FORMAT(GETDATE(), 'yyyy-MM')
),
ProductividadAgente AS (
    SELECT
        CANAL,
        AGENTE,
        COUNT(ID_GESTION)       AS GESTIONES_EFECTIVAS,
        SUM(MONTO_PAGADO)       AS MONTO_RECUPERADO,
        AVG(MONTO_PAGADO)       AS TICKET_PROMEDIO
    FROM GestionesConPago
    GROUP BY CANAL, AGENTE
),
RankingPorCanal AS (
    SELECT
        CANAL,
        AGENTE,
        GESTIONES_EFECTIVAS,
        MONTO_RECUPERADO,
        TICKET_PROMEDIO,
        -- Ranking dentro del canal por monto recuperado
        RANK() OVER (
            PARTITION BY CANAL
            ORDER BY MONTO_RECUPERADO DESC
        ) AS RANKING_CANAL,
        -- Monto del mejor agente del canal
        FIRST_VALUE(MONTO_RECUPERADO) OVER (
            PARTITION BY CANAL
            ORDER BY MONTO_RECUPERADO DESC
        ) AS MONTO_TOP_CANAL
    FROM ProductividadAgente
)
SELECT
    CANAL,
    RANKING_CANAL,
    AGENTE,
    GESTIONES_EFECTIVAS,
    MONTO_RECUPERADO,
    TICKET_PROMEDIO,
    MONTO_TOP_CANAL,
    MONTO_TOP_CANAL - MONTO_RECUPERADO AS BRECHA_VS_TOP,
    ROUND((MONTO_RECUPERADO * 100.0) / NULLIF(MONTO_TOP_CANAL, 0), 1) AS PCT_VS_TOP
FROM RankingPorCanal
ORDER BY CANAL, RANKING_CANAL;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `LEFT JOIN` con rango de fechas | Asocia el pago a la gestión que lo originó (ventana de 3 días) |
| `RANK()` | Permite empates, a diferencia de `ROW_NUMBER` |
| `FIRST_VALUE()` | Trae el valor del agente top sin necesidad de un self-join |
| `BETWEEN` con fechas | Atribuye el pago a la gestión más cercana |

---

## Problema 7 — Detección de promesas de pago rotas (PTP Broken)

### Enunciado

Cuando un cliente promete pagar (gestión con resultado = 'PROMESA'), se espera que el pago se registre en los próximos 5 días. El área de calidad necesita identificar **qué promesas se rompieron**, cuánto representan en monto y cuál es la tasa de cumplimiento por agente.

### Tablas involucradas

`GESTIONES`, `PAGOS`

### Solución

```sql
WITH Promesas AS (
    SELECT
        g.ID_GESTION,
        g.ID_CREDITO,
        g.FECHA_GESTION                              AS FECHA_PROMESA,
        DATEADD(DAY, 5, g.FECHA_GESTION)             AS FECHA_LIMITE_PAGO,
        g.AGENTE,
        g.CANAL
    FROM GESTIONES g
    WHERE g.RESULTADO = 'PROMESA'
),
PromesasConCumplimiento AS (
    SELECT
        pr.ID_GESTION,
        pr.ID_CREDITO,
        pr.FECHA_PROMESA,
        pr.FECHA_LIMITE_PAGO,
        pr.AGENTE,
        pr.CANAL,
        p.MONTO_PAGADO,
        p.FECHA_PAGO,
        CASE WHEN p.ID_PAGO IS NOT NULL THEN 1 ELSE 0 END AS CUMPLIDA
    FROM Promesas pr
    LEFT JOIN PAGOS p
        ON  p.ID_CREDITO = pr.ID_CREDITO
        AND p.FECHA_PAGO BETWEEN pr.FECHA_PROMESA AND pr.FECHA_LIMITE_PAGO
),
ResumenAgente AS (
    SELECT
        AGENTE,
        CANAL,
        COUNT(*)                          AS TOTAL_PROMESAS,
        SUM(CUMPLIDA)                     AS PROMESAS_CUMPLIDAS,
        COUNT(*) - SUM(CUMPLIDA)          AS PROMESAS_ROTAS,
        SUM(MONTO_PAGADO)                 AS MONTO_RECUPERADO,
        ROUND(SUM(CUMPLIDA) * 100.0
              / NULLIF(COUNT(*), 0), 2)   AS PTP_KEPT_RATE
    FROM PromesasConCumplimiento
    GROUP BY AGENTE, CANAL
)
SELECT
    AGENTE,
    CANAL,
    TOTAL_PROMESAS,
    PROMESAS_CUMPLIDAS,
    PROMESAS_ROTAS,
    MONTO_RECUPERADO,
    PTP_KEPT_RATE,
    -- Percentil del agente dentro de su canal
    PERCENT_RANK() OVER (
        PARTITION BY CANAL
        ORDER BY PTP_KEPT_RATE
    ) AS PERCENTIL_CANAL
FROM ResumenAgente
ORDER BY CANAL, PTP_KEPT_RATE DESC;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `DATEADD` | Define la ventana de cumplimiento de 5 días |
| `LEFT JOIN` + `CASE WHEN IS NOT NULL` | Detecta si existió un pago dentro de la ventana |
| `PERCENT_RANK()` | Ubica al agente en el percentil de su canal |
| `NULLIF` en división | Evita error cuando el agente no tiene promesas |

---

## Problema 8 — Conciliación entre reservas y facturación (basado en la consulta original)

### Enunciado extendido

Extendiendo el problema original de conciliación entre `RESERVAS` y `FACTURACION`, ahora se pide agregar:
- El **número de reserva** dentro del histórico de cada cliente (¿es su primera, segunda, tercera reserva?).
- El **acumulado de diferencias** por cliente.
- Solo mostrar clientes cuyo **acumulado de diferencias supere $200,000**.

### Solución

```sql
WITH ReservasFacturadas AS (
    SELECT
        r.ID_CLIENTE,
        r.ID_HABITACION,
        r.ID_RESERVA,
        r.FECHA_ENTRADA,
        r.MONTO_RESERVA,
        f.MONTO_FACTURADO,
        COALESCE(r.MONTO_RESERVA, 0) - COALESCE(f.MONTO_FACTURADO, 0) AS DIFERENCIA
    FROM RESERVAS r
    FULL OUTER JOIN FACTURACION f
        ON  r.ID_CLIENTE    = f.ID_CLIENTE
        AND r.ID_HABITACION = f.ID_HABITACION
        AND r.ID_RESERVA    = f.ID_RESERVA
        AND r.FECHA_ENTRADA = f.FECHA_ENTRADA
),
ReservasEnumeradas AS (
    SELECT
        *,
        -- Número de reserva histórica por cliente (1 = la más antigua)
        ROW_NUMBER() OVER (
            PARTITION BY ID_CLIENTE
            ORDER BY FECHA_ENTRADA
        ) AS NRO_RESERVA_CLIENTE,
        -- Diferencia acumulada por cliente ordenada cronológicamente
        SUM(ABS(DIFERENCIA)) OVER (
            PARTITION BY ID_CLIENTE
            ORDER BY FECHA_ENTRADA
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS DIFERENCIA_ACUMULADA
    FROM ReservasFacturadas
    WHERE ABS(DIFERENCIA) >= 50000
       OR MONTO_RESERVA    IS NULL
       OR MONTO_FACTURADO  IS NULL
),
ClientesConProblema AS (
    SELECT ID_CLIENTE
    FROM ReservasEnumeradas
    GROUP BY ID_CLIENTE
    HAVING SUM(ABS(DIFERENCIA)) > 200000
)
SELECT
    re.ID_CLIENTE,
    re.ID_RESERVA,
    re.FECHA_ENTRADA,
    re.NRO_RESERVA_CLIENTE,
    re.MONTO_RESERVA,
    re.MONTO_FACTURADO,
    re.DIFERENCIA,
    re.DIFERENCIA_ACUMULADA,
    CASE
        WHEN re.MONTO_RESERVA   IS NULL THEN 'SIN RESERVA'
        WHEN re.MONTO_FACTURADO IS NULL THEN 'SIN FACTURA'
        WHEN re.DIFERENCIA > 0          THEN 'SUBFACTURADO'
        ELSE 'SOBREFACTURADO'
    END AS TIPO_DISCREPANCIA
FROM ReservasEnumeradas re
INNER JOIN ClientesConProblema cp ON cp.ID_CLIENTE = re.ID_CLIENTE
ORDER BY re.ID_CLIENTE, re.FECHA_ENTRADA;
```

### Conceptos aplicados

| Técnica | Uso |
|---|---|
| `ROW_NUMBER()` | Enumera el historial de reservas por cliente |
| `SUM() OVER (ROWS BETWEEN UNBOUNDED PRECEDING...)` | Diferencia acumulada por cliente |
| `HAVING` en CTE | Filtra solo clientes con discrepancias altas |
| `FULL OUTER JOIN` | Conserva registros huérfanos en ambas tablas |

---

## Guía de optimización

### ¿Cuándo usar índices?

```sql
-- Índice compuesto recomendado para consultas de saldos mensuales
CREATE INDEX IX_SALDOS_CREDITO_PERIODO
    ON SALDOS_MENSUALES (ID_CREDITO, PERIODO)
    INCLUDE (SALDO_CAPITAL, SALDO_VENCIDO, DPD);

-- Justificación: las funciones de ventana PARTITION BY ID_CREDITO ORDER BY PERIODO
-- necesitan encontrar y ordenar filas del mismo crédito rápidamente.
-- Sin este índice, el motor hace un full scan + sort en memoria.
```

### ¿Por qué evitar `SELECT *`?

```sql
-- MAL: trae todas las columnas aunque solo necesites 3
SELECT * FROM SALDOS_MENSUALES WHERE DPD > 30;

-- BIEN: columnas explícitas → menos I/O, menor uso de red, el índice puede cubrir la consulta
SELECT ID_CREDITO, PERIODO, SALDO_VENCIDO
FROM SALDOS_MENSUALES
WHERE DPD > 30;
```

### ¿Cuándo usar particionamiento de tablas?

Las tablas de hechos como `SALDOS_MENSUALES`, `PAGOS` o `GESTIONES` crecen a millones de filas con el tiempo. Particionar por `PERIODO` o `YEAR(FECHA_PAGO)` permite que las consultas que filtran por mes solo lean las particiones relevantes (*partition pruning*), reduciendo el I/O en órdenes de magnitud.

```sql
-- Ejemplo conceptual de partición por año en SQL Server
CREATE PARTITION FUNCTION pf_anio (INT)
    AS RANGE RIGHT FOR VALUES (202201, 202301, 202401, 202501);
```

### Resumen de buenas prácticas

| Práctica | Razón |
|---|---|
| Filtrar en CTEs tempranas | Reduce el volumen de filas antes de los joins |
| Usar `INNER JOIN` cuando sea posible | Más eficiente que `LEFT JOIN` si la relación siempre existe |
| Evitar funciones sobre columnas en `WHERE` | `WHERE YEAR(FECHA) = 2024` impide usar el índice; mejor `WHERE FECHA BETWEEN '2024-01-01' AND '2024-12-31'` |
| Materializar CTEs pesadas como tablas temporales | Cuando una CTE se referencia más de una vez en la misma consulta |
| `NULLIF` en divisiones | Previene errores de división por cero sin necesidad de `CASE WHEN` |
