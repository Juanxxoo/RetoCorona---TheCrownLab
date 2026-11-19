# Cambios tras el feedback de la entrega intermedia (18-nov-2026)

## Orden de ejecución

```
01_carga_y_limpieza.py → 02_feature_engineering.py → 03_EDA.py → 04_evolucion_temporal.py
→ 05_segmentacion.py → 06_market_basket.py → 07_penetracion.py → 08_priorizacion.py
→ 09_evolucion_generacional.py → 10_validacion_k_recompra.py
```

`08` usa la salida de `07`. `09` y `10` dependen solo de `01`–`05`.

`07_penetracion.py` necesita el anexo del DANE en `data/`:
`PPED-AreaSexoEdadDep-2018-2050_VP.xlsx` (proyecciones departamentales por área, sexo y edad simple, actualizado el 30-jul-2025).
Fuente: https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Departamental/PPED-AreaSexoEdadDep-2018-2050_VP.xlsx

## 01_carga_y_limpieza.py

- **Fechas +24 años (Fix 0).** Enero de 2000 = enero de 2024. Se corrige al inicio, así que todo el pipeline usa meses reales (ene-2024 a jul-2026). Se elimina `00_revision_fechas.py`, que guardaba un parquet que ningún script usaba.
- **Errores de decimales en Valor (Fix 2).** Se esperaba que la relación Valor = (Cantidad / 2,65) × Precio se cumpliera en todas las filas, pero solo se cumple en el 91,4 %. En el 8,55 % (224.192 filas) **el campo errado es Valor**, al que le faltan 1, 2 o 3 ceros. Para verificarlo se comparó cada fila con la mediana del mismo producto: Cantidad y Precio dan ≈1,0 contra la referencia, y el Valor por unidad da 1/10, 1/100 o 1/1000. Se corrige multiplicando Valor por la potencia detectada, y se guarda el original en `Valor_original`.
  - Impacto: el valor total sube 4,5 %. El 17 % de los clientes cambia su valor total en más de 10 %. La tasa de error es estable entre meses (7–9 %) y entre negocios (8–13 %).
- **Etiqueta legible de categoría (Fix 2b).** Nueva columna `categoria_label` con el formato "NEGOCIO · CAT_xxxxx".

## 02_feature_engineering.py

- Solo se actualizó el comentario sobre la fecha de referencia. No hay cambios de lógica.

## 05_segmentacion.py

- **Bug corregido: los nombres de los segmentos dependían del número arbitrario que asigna KMeans.** Con el Valor corregido, KMeans reordenó los clusters y los nombres quedaron cruzados. Ahora cada cluster recibe su id según su perfil:
  - el de mayor valor mediano es el consolidado;
  - los que tienen recencia mediana mayor a 365 días son inactivos y los demás son activos;
  - dentro de cada grupo, el de menor valor es "bajo vínculo" o "reciente", y el de mayor valor es "vínculo intermedio" o "en desarrollo".

  Con los datos viejos, esta regla reproduce exactamente los nombres originales.
- Resultado con los datos corregidos: el 95,3 % de los clientes queda en el mismo segmento (ARI 0,88 en consumidores y 0,96 en empresas). Los tamaños cambian menos de 1,3 puntos porcentuales.

## 06_market_basket.py

- Sin cambios de código. Se mantienen los hallazgos: Cocinas→Baños en los consolidados (lift 2,8) y Servicios→Baños en los recientes (lift 2,7–3,0).

## 07_penetracion.py (nuevo)

Responde la pregunta central de Corona: dónde hay baja penetración y cómo cambia entre generaciones.

- **Métrica:** consumidores por cada 1.000 habitantes de la misma generación y departamento, con la población DANE de 2025.
- **Edades:** en la base van de 19 a 68 años (anonimizadas). La población DANE se recorta al mismo rango por generación: Gen Z 19–29, Millennial 30–45, Gen X 46–61 y Boomer 62–68, todas con edad a 2026.
- **Presencia física:** cada punto de venta le vende ~92 % (mediana) a un solo departamento, que se toma como su sede. Hay tiendas en 15 departamentos. Las oportunidades se separan en dos tipos:
  - **Activación:** hay tienda y la penetración es baja.
  - **Expansión:** no hay tienda.
- **Brecha:** clientes adicionales si el departamento llegara al percentil 75 de su generación, calculado solo entre departamentos con tienda.
- **Salidas** en `resultados/` (agregadas, sin datos de clientes):
  - `penetracion_dep_generacion.csv`
  - `penetracion_tabla_por_mil.csv`
  - `penetracion_oportunidades.csv`
  - `penetracion_evolucion_anual.csv`

### Hallazgos principales

| Generación | Clientes por cada 1.000 hab. | Cambio 2024→2025 (activos) |
|---|---|---|
| Gen Z | 2,9 | +24,6 % |
| Millennial | 16,3 | +7,6 % |
| Gen X | 22,0 | +2,2 % |
| Boomer | 2,9 | −4,1 % |

- Gen X y Millennials concentran la base. Gen Z es la generación más pequeña, pero la que más crece.
- **Activación:** ~149.000 clientes potenciales. Las mayores brechas son Antioquia (Gen X y Millennial), Córdoba, Norte de Santander, Bolívar, Tolima y Boyacá.
- **Expansión:** ~128.000 clientes potenciales. Las mayores brechas son Nariño, Cauca, Caldas, Meta y Huila. Esta cifra es optimista, porque se compara contra departamentos que sí tienen tienda.

## 08_priorizacion.py (nuevo)

Une penetración, segmentos, afinidades y contexto territorial en una tabla de **departamento × generación → acción**.

**Correcciones a los CSV externos.** Están documentadas en el código y los CSV crudos no se tocan:

- **Déficit cualitativo y cuantitativo invertidos.** Bogotá trae 1,6 % / 9,0 %, que son las cifras de la ECV 2021 al revés: cuantitativo 1,6 % y cualitativo 9,0 %. Además, a nivel nacional el cualitativo es ≈3 veces el cuantitativo.
- **Huila con 999 % de cobertura de energía.** Se deja vacío.

La base corregida queda en `resultados/base_territorial_corregida.csv`.

**Componentes del puntaje.** Cada componente se convierte a percentil y el puntaje se calcula por separado para activación y para expansión.

| Componente | Peso |
|---|---|
| Brecha de penetración | 40 % |
| Tamaño de la población | 20 % |
| m² licenciados por cada 1.000 habitantes | 20 % |
| Crecimiento de hogares | 10 % |
| Déficit cualitativo (el plus de sostenibilidad) | 10 % |

**Sensibilidad.** Se probaron 2.000 combinaciones aleatorias de pesos. La columna `pct_simulaciones_top10` dice qué tan robusta es cada oportunidad.

**Acción sugerida.** Para cada celda se cruzan dos cosas:

- el segmento **sobrerrepresentado** frente al promedio nacional de la generación, con su regla de afinidad de `06`;
- el negocio con menor penetración relativa en la celda.

**Hallazgos.**

- **Top de activación:** Antioquia (Gen X y Millennial), Norte de Santander, Cundinamarca, Bolívar y Córdoba.
- **Las más robustas:** Norte de Santander · Millennial (83,6 % de las simulaciones) y Cundinamarca (≈68 %).
- **Top de expansión:** Cauca, Meta, Huila y Nariño, con Pinturas como negocio de entrada.

## 09_evolucion_generacional.py (nuevo)

Responde cómo cambia el consumidor entre generaciones. Los años se comparan en el mismo periodo, enero–julio.

| Generación | Clientes activos 2024→2026 | Gasto 2024→2026 | Rasgo de su mezcla |
|---|---|---|---|
| Gen Z | +44 % | +69 % | Más Revestimientos (índice 107), menos Baños (82) |
| Millennial | +8 % | +23 % | Igual al promedio |
| Gen X | −4 % | +4 % | Algo más de Baños y Pinturas |
| Boomer | −21 % | −9 % | Baños (117), Pinturas (142) y Servicios de instalación (264) |

- La Gen Z pasó de 5,9 % a 8,3 % de los clientes nuevos.
- El ticket sube en todas las generaciones, entre 9 % y 16 %.

## 10_validacion_k_recompra.py (nuevo)

Responde la crítica del jurado sobre K=5 y K=4. Se segmenta con los datos hasta el 31-dic-2024 y se mide la recompra entre ene-2025 y jul-2026.

**Consumidores.** K=5 explica **45 % más** de la variación en recompra que K=2 (η² de 0,041 frente a 0,028). La recompra va de 15 % a 40 % entre segmentos, frente a 18 %–34 % con K=2. Pasar a K=6 casi no agrega nada.

**Empresas.** K=4 mejora la explicación en 12 % (η² de 0,075 frente a 0,067). La recompra va de 29 % a 71 %, frente a 33 %–61 % con K=2.

**Lectura honesta.** El η² absoluto es bajo porque la recompra individual es muy ruidosa. Aun así, los segmentos extra separan clientes que de verdad se comportan distinto, y eso justifica K=5 por negocio aunque el Silhouette prefiera K=2.

## PILOTO_KPIS.md (nuevo)

- **Piloto:** venta cruzada Cocinas → Baños en Antioquia, con grupo control aleatorio y solo para clientes con consentimiento.
- **Línea base:** 6,0 % de conversión a 60 días.
- **Muestra:** ≈415 clientes por grupo, suficiente para detectar un aumento de +5 puntos.
- **KPIs:** los del piloto y los del impacto del modelo en el negocio.
