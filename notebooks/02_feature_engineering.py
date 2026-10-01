# %% [markdown]
# # Feature engineering — Reto Corona (Retail)
# Parte de `data/dataset_limpio.parquet` (una fila por transacción) y construye
# `data/features_clientes.parquet` (una fila por cliente) con las variables que
# van a alimentar el modelo de segmentación.

# %% Carga
import pandas as pd

df = pd.read_parquet("../data/dataset_limpio.parquet")
print(f"Transacciones: {df.shape}, Clientes únicos: {df['Cliente'].nunique()}")

# Corona confirmó que las fechas están anonimizadas pero las diferencias entre
# ellas son reales. Por eso usamos la fecha máxima DEL DATASET como referencia
# para "hoy" en vez de la fecha real de calendario -- restar contra hoy daría
# ~24 años de recencia para todo el mundo, sin sentido.
FECHA_REFERENCIA = df["Fecha Pedido"].max()
print(f"Fecha de referencia (máxima del dataset): {FECHA_REFERENCIA.date()}")

# %% RFM — Recencia, Frecuencia, Valor monetario
rfm = df.groupby("Cliente").agg(
    recencia_dias=("Fecha Pedido", lambda s: (FECHA_REFERENCIA - s.max()).days),
    frecuencia=("ID Pedido", "nunique"),
    valor_total=("Valor", "sum"),
).reset_index()

rfm["ticket_promedio"] = rfm["valor_total"] / rfm["frecuencia"]
print(rfm.describe())

# %% Antigüedad — desde Primera Fecha (Corona confirmó: sí es la primera compra real)
cli_unico = df.drop_duplicates(subset="Cliente")[
    ["Cliente", "Tipo Cliente", "Edad", "Genero", "Tratamiento Datos", "Primera Fecha"]
].copy()

cli_unico["antiguedad_dias"] = (FECHA_REFERENCIA - cli_unico["Primera Fecha"]).dt.days
print(cli_unico["antiguedad_dias"].describe())

# %% Generación — a partir de Edad (solo aplica a personas naturales, no a Empresa)
# Cortes aproximados de edad a 2026 según definiciones generacionales estándar;
# ajústenlos si encuentran una fuente más específica para su análisis.
# "Empresa" no tiene Edad (por diseño, es un cliente jurídico) -- se deja como
# categoría propia en vez de descartarla o forzarla a una generación.

import numpy as np
import pandas as pd

condiciones = [
    cli_unico["Tipo Cliente"] == "Empresa",
    cli_unico["Edad"].isna(),
    cli_unico["Edad"] <= 29,
    cli_unico["Edad"] <= 45,
    cli_unico["Edad"] <= 61
]

opciones = [
    "No aplica (Empresa)",
    "Edad no disponible",
    "Gen Z",
    "Millennial",
    "Gen X"
]

# np.select evalúa en orden y asigna 'Boomer' si no cumple ninguna anterior (> 61)
cli_unico["generacion"] = np.select(condiciones, opciones, default="Boomer")

print(cli_unico["generacion"].value_counts())

# %% Perfil de canasta y canal — a nivel de cliente
# Nota: esto recorre las 2.6M filas agrupadas por cliente, puede tardar un poco.
def moda(serie):
    m = serie.mode()
    return m.iloc[0] if not m.empty else pd.NA

perfil = df.groupby("Cliente").agg(
    n_negocios_distintos=("Negocio", "nunique"),
    n_categorias_distintas=("Categoria", "nunique"),
    negocio_dominante=("Negocio", moda),
    region_dominante=("Region", moda),
    cluster_dominante=("Cluster", moda),
).reset_index()

print(perfil.head())

# %% Consolidar — una sola tabla, una fila por cliente
features_clientes = (
    rfm.merge(cli_unico.drop(columns="Primera Fecha"), on="Cliente", how="left")
       .merge(perfil, on="Cliente", how="left")
)

print(f"\nTabla final: {features_clientes.shape}")
print(features_clientes.head())

# %% Checkpoint
features_clientes.to_parquet("../data/features_clientes.parquet", index=False)
print("Guardado en data/features_clientes.parquet")

# %% Verificación rápida del checkpoint
verificacion = pd.read_parquet("../data/features_clientes.parquet")
print(verificacion.shape)
print(verificacion.describe(include="all").T)
