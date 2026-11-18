from pathlib import Path
import pandas as pd

# Ruta del proyecto
ROOT = Path(__file__).resolve().parents[1]

# Cargar la base limpia que ya teníamos
df = pd.read_parquet(ROOT / "data" / "dataset_limpio.parquet")

# Asegurar que la fecha esté en formato fecha
df["Fecha Pedido"] = pd.to_datetime(df["Fecha Pedido"], errors="coerce")

print("Filas:", len(df))
print("Fechas nulas:", df["Fecha Pedido"].isna().sum())
print("Fecha mínima:", df["Fecha Pedido"].min())
print("Fecha máxima:", df["Fecha Pedido"].max())

print("\nRegistros por año:")
print(df["Fecha Pedido"].dt.year.value_counts().sort_index())

# Crear fecha corregida sumando 24 años
df["Fecha Pedido Real"] = df["Fecha Pedido"] + pd.DateOffset(years=24)

print("\n--- FECHAS CORREGIDAS ---")
print("Fecha real mínima:", df["Fecha Pedido Real"].min())
print("Fecha real máxima:", df["Fecha Pedido Real"].max())

print("\nRegistros por año real:")
print(df["Fecha Pedido Real"].dt.year.value_counts().sort_index())

# Verificar que mes y día no hayan cambiado
mismo_mes = (df["Fecha Pedido"].dt.month == df["Fecha Pedido Real"].dt.month).all()
mismo_dia = (df["Fecha Pedido"].dt.day == df["Fecha Pedido Real"].dt.day).all()

print("\n¿Se conserva el mes?:", mismo_mes)
print("¿Se conserva el día?:", mismo_dia)

# Guardar una nueva versión sin sobrescribir el dataset original
ruta_salida = ROOT / "data" / "dataset_fechas_corregidas.parquet"

df.to_parquet(ruta_salida, index=False)

print("\nGuardado en:", ruta_salida)
print("Columnas de fecha:")
print(df[["Fecha Pedido", "Fecha Pedido Real"]].head())

# Verificación final de cobertura temporal real
resumen_mensual = (
    df["Fecha Pedido Real"]
    .dt.to_period("M")
    .value_counts()
    .sort_index()
)

print("\n--- COBERTURA MENSUAL REAL ---")
print(resumen_mensual)

print("\nNúmero de meses distintos:", len(resumen_mensual))

print(
    "¿Hay fechas posteriores al 31-07-2026?:",
    (df["Fecha Pedido Real"] > pd.Timestamp("2026-07-31")).any()
)

print(
    "¿Hay fechas anteriores al 01-01-2024?:",
    (df["Fecha Pedido Real"] < pd.Timestamp("2024-01-01")).any()
)

print("\nVISTA DE LA BASE CORREGIDA")
print(df.head(10).to_string())