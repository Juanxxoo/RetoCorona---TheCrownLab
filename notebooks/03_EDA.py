# %% [markdown]
# # 03. Análisis exploratorio y descriptivo
#
# El objetivo del EDA es entender la estructura y el comportamiento de los datos
# para tomar decisiones sobre las variables que se utilizarán posteriormente
# en la segmentación, el análisis de afinidades y los análisis temporales.

# %% Librerías y carga de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("../data/dataset_limpio.parquet")
clientes = pd.read_parquet("../data/features_clientes.parquet")

print("BASE TRANSACCIONAL")
print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nBASE A NIVEL CLIENTE")
print("Clientes:", clientes.shape[0])
print("Variables:", clientes.shape[1])

# %% Cobertura general de la información

print("Periodo analizado:")
print("Desde:", df["Fecha Pedido"].min())
print("Hasta:", df["Fecha Pedido"].max())

print("\nClientes únicos:", df["Cliente"].nunique())
print("Pedidos únicos:", df["ID Pedido"].nunique())
print("Productos únicos:", df["Producto"].nunique())
print("Categorías únicas:", df["Categoria"].nunique())
print("Negocios únicos:", df["Negocio"].nunique())
print("Regiones:", df["Region"].nunique())

# %% Validación de la estructura de los pedidos

validacion_pedidos = (
    df.groupby("ID Pedido")
      .agg(
          clientes_distintos=("Cliente", "nunique"),
          fechas_distintas=("Fecha Pedido", "nunique")
      )
)

print(
    "Pedidos asociados a más de un cliente:",
    (validacion_pedidos["clientes_distintos"] > 1).sum()
)

print(
    "Pedidos asociados a más de una fecha:",
    (validacion_pedidos["fechas_distintas"] > 1).sum()
)

# %% Revisar pedidos que aparecen en más de una fecha

pedidos_varias_fechas = validacion_pedidos[
    validacion_pedidos["fechas_distintas"] > 1
].index

ejemplo_pedidos = (
    df[df["ID Pedido"].isin(pedidos_varias_fechas)]
    [["ID Pedido", "Cliente", "Fecha Pedido", "Producto", "Categoria"]]
    .sort_values(["ID Pedido", "Fecha Pedido"])
)

ejemplo_pedidos.head(30)

# %% Diferencia de días dentro de pedidos con varias fechas

rango_fechas_pedido = (
    df.groupby("ID Pedido")["Fecha Pedido"]
      .agg(["min", "max", "nunique"])
)

rango_fechas_pedido = rango_fechas_pedido[
    rango_fechas_pedido["nunique"] > 1
].copy()

rango_fechas_pedido["diferencia_dias"] = (
    rango_fechas_pedido["max"] - rango_fechas_pedido["min"]
).dt.days

print(rango_fechas_pedido["diferencia_dias"].describe())

print("\nDistribución de las diferencias más comunes:")
print(
    rango_fechas_pedido["diferencia_dias"]
    .value_counts()
    .sort_index()
    .head(20)
)

# %% Proporción de pedidos según separación entre fechas

rangos = {
    "1 día": (rango_fechas_pedido["diferencia_dias"] == 1).sum(),
    "2-3 días": rango_fechas_pedido["diferencia_dias"].between(2, 3).sum(),
    "4-7 días": rango_fechas_pedido["diferencia_dias"].between(4, 7).sum(),
    "8-15 días": rango_fechas_pedido["diferencia_dias"].between(8, 15).sum(),
    "16-30 días": rango_fechas_pedido["diferencia_dias"].between(16, 30).sum(),
    "Más de 30 días": (rango_fechas_pedido["diferencia_dias"] > 30).sum()
}

rangos_df = pd.DataFrame.from_dict(
    rangos, orient="index", columns=["pedidos"]
)

rangos_df["porcentaje"] = (
    rangos_df["pedidos"] / len(rango_fechas_pedido) * 100
).round(2)

rangos_df

# %% Tipo de cliente

tipo_cliente = (
    clientes["Tipo Cliente"]
    .value_counts(dropna=False)
    .to_frame("clientes")
)

tipo_cliente["porcentaje"] = (
    tipo_cliente["clientes"] / len(clientes) * 100
).round(2)

tipo_cliente

# %% Comparación de comportamiento por tipo de cliente

comparacion_tipo = (
    clientes.groupby("Tipo Cliente")
    [["recencia_dias", "frecuencia", "valor_total",
      "ticket_promedio", "n_negocios_distintos",
      "n_categorias_distintas"]]
    .median()
    .round(2)
)

comparacion_tipo

# %% Recurrencia según tipo de cliente

clientes_rec = clientes.copy()

clientes_rec["nivel_frecuencia"] = np.select(
    [
        clientes_rec["frecuencia"] == 1,
        clientes_rec["frecuencia"] == 2,
        clientes_rec["frecuencia"] >= 3
    ],
    [
        "1 pedido",
        "2 pedidos",
        "3 o más pedidos"
    ],
    default="Sin dato"
)

recurrencia_tipo = (
    pd.crosstab(
        clientes_rec["Tipo Cliente"],
        clientes_rec["nivel_frecuencia"],
        normalize="index"
    ) * 100
).round(2)

recurrencia_tipo

recurrencia_tipo

# %% Distribución de frecuencia de compra

clientes["grupo_frecuencia"] = pd.cut(
    clientes["frecuencia"],
    bins=[0, 1, 2, 5, 10, float("inf")],
    labels=[
        "1 pedido",
        "2 pedidos",
        "3-5 pedidos",
        "6-10 pedidos",
        "Más de 10 pedidos"
    ]
)

distribucion_frecuencia = (
    clientes["grupo_frecuencia"]
    .value_counts(sort=False)
    .to_frame("clientes")
)

distribucion_frecuencia["porcentaje"] = (
    distribucion_frecuencia["clientes"] / len(clientes) * 100
).round(2)

distribucion_frecuencia

# %% Distribución de recencia

print(clientes["recencia_dias"].describe())

# %% Distribución del valor total por cliente

print(clientes["valor_total"].describe())

# %% Percentiles altos del valor total

percentiles_valor = clientes["valor_total"].quantile(
    [0.90, 0.95, 0.99, 0.995, 0.999]
)

percentiles_valor

# %% Clientes con valor total extremo

umbral_extremo = clientes["valor_total"].quantile(0.999)

extremos_valor = clientes[
    clientes["valor_total"] >= umbral_extremo
]

print("Clientes extremos:", len(extremos_valor))

print("\nTipo de cliente entre los extremos:")
print(
    extremos_valor["Tipo Cliente"]
    .value_counts(dropna=False)
)

print("\nPorcentaje por tipo:")
print(
    (extremos_valor["Tipo Cliente"]
     .value_counts(normalize=True, dropna=False) * 100)
    .round(2)
)
# %% Comportamiento de los clientes con valor extremo

comparacion_extremos = (
    extremos_valor
    .groupby("Tipo Cliente")
    [["frecuencia", "ticket_promedio", "valor_total"]]
    .median()
    .round(2)
)

comparacion_extremos

# %% Distribución del ticket promedio

print(clientes["ticket_promedio"].describe())

# %% Diversidad de negocios por cliente

print(clientes["n_negocios_distintos"].value_counts().sort_index())

print("\nPorcentaje:")
print(
    (clientes["n_negocios_distintos"]
     .value_counts(normalize=True)
     .sort_index() * 100)
    .round(2)
)

# %% Diversidad de categorías por cliente

print(clientes["n_categorias_distintas"].describe())

print("\nDistribución:")
print(
    clientes["n_categorias_distintas"]
    .value_counts()
    .sort_index()
    .head(20)
)

# %% Correlaciones entre variables de comportamiento

variables_comportamiento = [
    "recencia_dias",
    "frecuencia",
    "valor_total",
    "ticket_promedio",
    "n_negocios_distintos",
    "n_categorias_distintas"
]

correlaciones = (
    clientes[variables_comportamiento]
    .corr(method="spearman")
    .round(2)
)

correlaciones

# %% Estructura de las canastas por pedido

estructura_canastas = (
    df.groupby("ID Pedido")
    .agg(
        n_lineas=("Producto", "size"),
        n_productos=("Producto", "nunique"),
        n_categorias=("Categoria", "nunique")
    )
)

estructura_canastas.describe()

# %% Proporción de canastas con múltiples productos y categorías

porcentaje_multi_producto = (
    (estructura_canastas["n_productos"] > 1).mean() * 100
)

porcentaje_multi_categoria = (
    (estructura_canastas["n_categorias"] > 1).mean() * 100
)

print(f"Pedidos con más de 1 producto: {porcentaje_multi_producto:.2f}%")
print(f"Pedidos con más de 1 categoría: {porcentaje_multi_categoria:.2f}%")

# %% Soporte de productos en los pedidos

soporte_productos = (
    df.groupby("Producto")["ID Pedido"]
    .nunique()
    .sort_values(ascending=False)
)

total_pedidos = df["ID Pedido"].nunique()

print("Productos con soporte >= 1%:",
      (soporte_productos / total_pedidos >= 0.01).sum())

print("Productos con soporte >= 0.5%:",
      (soporte_productos / total_pedidos >= 0.005).sum())

print("Productos con soporte >= 0.1%:",
      (soporte_productos / total_pedidos >= 0.001).sum())

print("Productos con soporte >= 0.05%:",
      (soporte_productos / total_pedidos >= 0.0005).sum())

# %% Soporte de categorías en los pedidos

soporte_categorias = (
    df.groupby("Categoria")["ID Pedido"]
    .nunique()
    .sort_values(ascending=False)
)

soporte_categorias_pct = (
    soporte_categorias / total_pedidos * 100
).round(2)

soporte_categorias_pct

# %% Distribución de pedidos por región

pedidos_region = (
    df.groupby("Region")["ID Pedido"]
    .nunique()
    .sort_values(ascending=False)
)

pedidos_region_pct = (
    pedidos_region / df["ID Pedido"].nunique() * 100
).round(2)

pedidos_region_pct

# %% Categorías principales por región

regiones_principales = [
    "BOGOTÁ",
    "VALLE",
    "ANTIOQUIA",
    "ATLÁNTICO",
    "CUNDINAMARCA",
    "SANTANDER"
]

mix_region_categoria = (
    df[df["Region"].isin(regiones_principales)]
    .groupby(["Region", "Categoria"])["ID Pedido"]
    .nunique()
    .groupby(level=0)
    .apply(lambda x: (x / x.sum() * 100).round(2))
)

for region in regiones_principales:
    print(f"\n{region}")
    print(
        mix_region_categoria
        .loc[region]
        .sort_values(ascending=False)
        .head(5)
    )
    
# %% Variación territorial por categoría

pedidos_por_region = (
    df[df["Region"].isin(regiones_principales)]
    .groupby("Region")["ID Pedido"]
    .nunique()
)

penetracion_region_categoria = (
    df[df["Region"].isin(regiones_principales)]
    .groupby(["Region", "Categoria"])["ID Pedido"]
    .nunique()
    .div(pedidos_por_region, level="Region")
    .mul(100)
    .unstack(fill_value=0)
)

variacion_categorias = pd.DataFrame({
    "min_region": penetracion_region_categoria.min(),
    "max_region": penetracion_region_categoria.max(),
})

variacion_categorias["rango_pp"] = (
    variacion_categorias["max_region"] -
    variacion_categorias["min_region"]
)

variacion_categorias.sort_values("rango_pp", ascending=False).head(10)

# %% Regiones con menor y mayor penetración por categoría

top_categorias_variables = (
    variacion_categorias
    .sort_values("rango_pp", ascending=False)
    .head(8)
    .index
)

for categoria in top_categorias_variables:
    serie = penetracion_region_categoria[categoria]

    print(f"\n{categoria}")
    print(f"Menor: {serie.idxmin()} -> {serie.min():.2f}%")
    print(f"Mayor: {serie.idxmax()} -> {serie.max():.2f}%")

# %% Valores cero en variables monetarias

for variable in ["valor_total", "ticket_promedio"]:
    n_ceros = (clientes[variable] == 0).sum()
    porcentaje = n_ceros / len(clientes) * 100

    print(
        f"{variable}: {n_ceros} clientes "
        f"({porcentaje:.4f}%)"
    )
# %%
