# Reto Analytics Corona – Uniandes · The Crown Lab

Solución analítica para el reto **"Una nueva mirada al consumidor"** de Corona y la Universidad de los Andes, enfocada en **Almacenes Corona (división Retail)**.

Responde tres preguntas de negocio:

1. **¿Cómo cambia el consumidor entre generaciones?** Captación, gasto y mezcla de negocios de cada generación en el tiempo.
2. **¿Dónde hay baja penetración?** Clientes por cada 1.000 habitantes, por generación y departamento, con datos del DANE.
3. **¿Qué hacer y con quién?** Segmentos de clientes, afinidades de compra y un índice que prioriza departamento × generación con una acción sugerida.

---

## Estructura del repositorio

```
RetoCorona---TheCrownLab/
├── notebooks/            Código: 10 scripts que se corren en orden (01 → 10)
├── resultados/           Tablas de resultados (CSV) que genera el código; las usa Power BI
├── data/                 Bases de datos — NO se suben a GitHub (ver "Datos")
├── Busqueda fuentes externas corona - *.csv    Indicadores territoriales (DANE)
├── CAMBIOS.md            Cambios hechos después de la entrega intermedia
├── PILOTO_KPIS.md        Diseño del piloto de validación y KPIs del negocio
├── requirements.txt      Librerías de Python
└── README.md
```

Cada script empieza con un encabezado que dice **qué hace, qué lee, qué genera y cuál es el siguiente paso**. Además, está dividido en **partes** que explican cada bloque grande del código.

---

## Instalación

Requiere **Python 3.10 o superior**. En Windows (PowerShell), desde la carpeta del repositorio:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si PowerShell bloquea la activación del entorno, ejecuta una vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Datos

Las bases **no están en el repositorio** porque son información confidencial de Corona (el `.gitignore` excluye la carpeta `data/`). Antes de correr el código, copia estos archivos en `data/`:

| Archivo | Origen |
|---|---|
| `base_transaccional.csv` | Corona (base transaccional, separador `;`) |
| `maestro_clientes.csv` | Corona (maestro de clientes, separador `;`) |
| `PPED-AreaSexoEdadDep-2018-2050_VP.xlsx` | [DANE · Proyecciones de población departamental por área, sexo y edad](https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Departamental/PPED-AreaSexoEdadDep-2018-2050_VP.xlsx) |

Los dos CSV de fuentes externas de la raíz (`Busqueda fuentes externas corona - *.csv`) sí están en el repositorio: son indicadores públicos del DANE por departamento.

---

## Cómo correrlo

Desde la carpeta `notebooks/`, con el entorno virtual activado, corre los scripts **en orden**:

```powershell
cd notebooks
python 01_carga_y_limpieza.py
python 02_feature_engineering.py
python 03_EDA.py
python 04_evolucion_temporal.py
python 05_segmentacion.py
python 06_market_basket.py
python 07_penetracion.py
python 08_priorizacion.py
python 09_evolucion_generacional.py
python 10_validacion_k_recompra.py
```

También se pueden abrir en **VS Code** y correr celda por celda: cada `# %%` funciona como una celda de notebook (requiere la extensión de Jupyter).

Todo el flujo tarda **entre 5 y 10 minutos**. El más lento es el `02`.

### Qué hace cada script

| # | Script | Qué hace | Genera |
|---|---|---|---|
| 01 | `carga_y_limpieza` | Une las bases de Corona y corrige su calidad: fechas +24 años, clientes duplicados y errores de decimales en Valor | `data/dataset_limpio.parquet` |
| 02 | `feature_engineering` | Construye una fila por cliente: RFM, antigüedad, generación y perfil de canasta y canal | `data/features_clientes.parquet` |
| 03 | `EDA` | Análisis exploratorio que justifica las decisiones de los siguientes pasos | (solo en pantalla) |
| 04 | `evolucion_temporal` | Evolución mensual, comparación entre años y estacionalidad por categoría | (solo en pantalla) |
| 05 | `segmentacion` | KMeans separado para consumidores (5 segmentos) y empresas (4), con validación y perfil comercial | `data/features_clientes_segmentados.parquet` |
| 06 | `market_basket` | Afinidades de compra entre negocios y categorías, en general y por segmento | (solo en pantalla) |
| 07 | `penetracion` | Clientes por cada 1.000 habitantes por generación y departamento, presencia de tiendas y brecha | `resultados/penetracion_*.csv` |
| 08 | `priorizacion` | Índice de priorización con prueba de robustez y acción sugerida | `resultados/priorizacion_oportunidades.csv`, `resultados/base_territorial_corregida.csv` |
| 09 | `evolucion_generacional` | Captación, actividad, gasto y mezcla de negocios por generación en el tiempo | `resultados/generacion_*.csv` |
| 10 | `validacion_k_recompra` | Comprueba que K=5 y K=4 predicen mejor la recompra futura que K=2 | `resultados/validacion_k_*.csv` |

### Qué contiene cada resultado

| Archivo | Contenido |
|---|---|
| `penetracion_dep_generacion.csv` | Clientes, población, clientes por cada 1.000 habitantes, brecha y tiendas por departamento × generación |
| `penetracion_tabla_por_mil.csv` | Tabla resumida: departamentos en filas, generaciones en columnas |
| `penetracion_oportunidades.csv` | Departamentos y generaciones con brecha positiva, por tipo (activación o expansión) |
| `penetracion_evolucion_anual.csv` | Penetración 2024 vs 2025 por generación |
| `priorizacion_oportunidades.csv` | **Tabla principal**: puntaje, robustez, segmento, negocio a impulsar y acción sugerida |
| `base_territorial_corregida.csv` | Indicadores DANE por departamento, ya corregidos |
| `generacion_*.csv` | Clientes nuevos por mes, actividad trimestral, comparación enero–julio y mezcla de negocios por generación |
| `validacion_k_recompra.csv` / `validacion_k_detalle_segmentos.csv` | Poder predictivo de cada K y recompra futura por segmento |

---

## Prototipo (Power BI)

El tablero `Reto_Corona.pbix` presenta los resultados en tres páginas: **Radar de Crecimiento**, **Consumidor, generaciones e integralidad** y **Centro de Acción**. La guía de uso está en el **Manual de usuario**.

Para actualizar el tablero después de cambiar el código, corre de nuevo los scripts y, en Power BI, usa **Inicio → Actualizar**. Antes, revisa que la ruta de los datos apunte a la carpeta `resultados/` de tu copia del repositorio (*Transformar datos → Configuración de origen de datos*).

---

## Supuestos y limitaciones

- **Fechas:** la base viene corrida 24 años (enero de 2000 = enero de 2024). Se corrige en `01`, así que el periodo real va de enero de 2024 a julio de 2026.
- **Edad:** viene "a 2026" y va de 19 a 68 años (anonimizada). Por eso la población del DANE se recorta al mismo rango.
- **Presencia física:** la base no dice dónde queda cada tienda; se infiere por el departamento al que le vende al menos el 80 % de sus ventas.
- **Afinidades:** muestran qué se compra junto, no qué causa la compra. Se presentan como hipótesis que hay que validar con pilotos.
- **Consentimiento:** el análisis usa todos los clientes, pero las acciones de contacto aplican solo a quienes autorizaron el tratamiento de datos.

## Uso de inteligencia artificial

El uso de herramientas de IA durante el desarrollo está documentado en el informe técnico, como exige el reto. Todos los resultados se pueden reproducir corriendo este código.
