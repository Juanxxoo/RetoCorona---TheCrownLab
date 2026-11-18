# Cambios tras el feedback de la entrega intermedia (18-nov-2026)

## Orden de ejecución

```
01_carga_y_limpieza.py → 02_feature_engineering.py → 03_EDA.py → 04_evolucion_temporal.py
→ 05_segmentacion.py → 06_market_basket.py → 07_penetracion.py
```

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
