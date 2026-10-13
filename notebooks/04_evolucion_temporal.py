# %% Carga y evolución mensual general

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet("../data/dataset_limpio.parquet")

df["mes"] = df["Fecha Pedido"].dt.to_period("M")

evolucion_mensual = (
    df.groupby("mes")
    .agg(
        pedidos=("ID Pedido", "nunique"),
        clientes=("Cliente", "nunique"),
        valor_total=("Valor", "sum")
    )
    .reset_index()
)

print(evolucion_mensual.head())
print("\nÚltimos meses:")
print(evolucion_mensual.tail())

print("\nNúmero de meses:", len(evolucion_mensual))

# %% Evolución mensual de pedidos

plt.figure(figsize=(12, 5))

plt.plot(
    evolucion_mensual["mes"].astype(str),
    evolucion_mensual["pedidos"],
    marker="o"
)

plt.xticks(rotation=45)
plt.xlabel("Mes")
plt.ylabel("Número de pedidos")
plt.title("Evolución mensual de pedidos")
plt.tight_layout()
plt.show()

# %% Promedio de pedidos por mes del año

evolucion_mensual["numero_mes"] = evolucion_mensual["mes"].dt.month

pedidos_por_mes_del_ano = (
    evolucion_mensual
    .groupby("numero_mes")["pedidos"]
    .agg(["mean", "min", "max", "count"])
    .round(2)
)

pedidos_por_mes_del_ano

# %% Pedidos por mes y año

evolucion_mensual["anio"] = evolucion_mensual["mes"].dt.year

comparacion_anual = (
    evolucion_mensual
    .pivot(
        index="numero_mes",
        columns="anio",
        values="pedidos"
    )
)

comparacion_anual

# %% Comparación enero-julio entre años

comparacion_enero_julio = (
    evolucion_mensual[
        evolucion_mensual["numero_mes"] <= 7
    ]
    .groupby("anio")
    .agg(
        pedidos=("pedidos", "sum"),
        clientes_promedio=("clientes", "mean"),
        valor_total=("valor_total", "sum")
    )
    .round(2)
)

comparacion_enero_julio

# %% Valor total por mes del año

valor_por_mes_del_ano = (
    evolucion_mensual
    .groupby("numero_mes")["valor_total"]
    .agg(["mean", "min", "max", "count"])
    .round(2)
)

valor_por_mes_del_ano

# %% Estacionalidad por categoría corregida

# Crear variables temporales
df["mes_periodo"] = df["Fecha Pedido"].dt.to_period("M")
df["numero_mes"] = df["Fecha Pedido"].dt.month

# Total de pedidos de la base
total_pedidos = df["ID Pedido"].nunique()

# Soporte de cada categoría
soporte_categorias = (
    df.groupby("Categoria")["ID Pedido"]
    .nunique()
    .sort_values(ascending=False)
)

# Seleccionar categorías con soporte >= 1%
categorias_relevantes = soporte_categorias[
    soporte_categorias / total_pedidos >= 0.01
].index

print("Categorías analizadas:", len(categorias_relevantes))

# Pedidos de cada categoría en cada mes real
categoria_mes_real = (
    df[df["Categoria"].isin(categorias_relevantes)]
    .groupby(["Categoria", "mes_periodo", "numero_mes"])["ID Pedido"]
    .nunique()
    .reset_index(name="pedidos")
)

# Promedio de pedidos por mes del año
# Así evitamos que enero-julio pesen más por tener 3 años observados
categoria_mes_promedio = (
    categoria_mes_real
    .groupby(["Categoria", "numero_mes"])["pedidos"]
    .mean()
    .unstack()
)

# Normalizar cada categoría contra su propio promedio mensual
categoria_mes_rel = categoria_mes_promedio.div(
    categoria_mes_promedio.mean(axis=1),
    axis=0
)

# Medir amplitud estacional
variacion_estacional = pd.DataFrame({
    "min_relativo": categoria_mes_rel.min(axis=1),
    "max_relativo": categoria_mes_rel.max(axis=1)
})

variacion_estacional["amplitud"] = (
    variacion_estacional["max_relativo"]
    - variacion_estacional["min_relativo"]
)

# Mostrar categorías con mayor variación estacional
variacion_estacional = variacion_estacional.sort_values(
    "amplitud",
    ascending=False
)

variacion_estacional

# %% Meses pico y valle de las categorías más estacionales

top_estacionales = (
    variacion_estacional
    .sort_values("amplitud", ascending=False)
    .head(5)
    .index
)

for categoria in top_estacionales:
    serie = categoria_mes_rel.loc[categoria]

    print(f"\n{categoria}")
    print(f"Mes mínimo: {serie.idxmin()} -> {serie.min():.2f} veces su promedio")
    print(f"Mes máximo: {serie.idxmax()} -> {serie.max():.2f} veces su promedio")

# %% Tabla de perfiles por categoría

total_pedidos = df["ID Pedido"].nunique()

# Participación
participacion = (
    df.groupby("Categoria")["ID Pedido"]
    .nunique()
    .div(total_pedidos)
    .rename("participacion")
)

# Concentración regional (Herfindahl: suma de shares regionales al cuadrado)
# alto = concentrada en pocas regiones, bajo = repartida uniformemente
pedidos_region_cat = (
    df.groupby(["Categoria", "Region"])["ID Pedido"]
    .nunique()
)
share_regional = pedidos_region_cat / pedidos_region_cat.groupby("Categoria").transform("sum")
concentracion_regional = (share_regional ** 2).groupby("Categoria").sum().rename("concentracion_regional")

# Valor promedio por pedido
valor_promedio = (
    df.groupby("Categoria")["Valor"]
    .mean()
    .rename("valor_promedio")
)

# Amplitud estacional (ya la tienen de la fase 4)
# Nota: queda NaN para categorías con soporte < 1% (quedaron fuera de
# categorias_relevantes más arriba) — no es un error, no se calculó estacionalidad
# para esas por bajo volumen.
amplitud = variacion_estacional["amplitud"].rename("amplitud_estacional")

# Unir todo
perfil_categorias = pd.concat(
    [participacion, concentracion_regional, valor_promedio, amplitud],
    axis=1
).reset_index()

# %% Etiquetado por reglas (terciles)
# Fix: el corte de "tercio superior" es el cuantil 0.66, no el 1.0 (que es
# simplemente el máximo). Con el índice anterior casi todo caía en
# "Transaccional/Base" por defecto.

def etiquetar(row, cortes):
    etiquetas = []
    if row["participacion"] >= cortes["participacion"][1]:
        etiquetas.append("Ancla")
    if row["amplitud_estacional"] >= cortes["amplitud_estacional"][1]:
        etiquetas.append("Estacional")
    if row["concentracion_regional"] >= cortes["concentracion_regional"][1]:
        etiquetas.append("Regional/Nicho")
    if row["valor_promedio"] >= cortes["valor_promedio"][1]:
        etiquetas.append("Alto valor")
    return etiquetas if etiquetas else ["Transaccional/Base"]

cortes = {
    col: perfil_categorias[col].quantile([0.33, 0.66]).values
    for col in ["participacion", "amplitud_estacional", "concentracion_regional", "valor_promedio"]
}

perfil_categorias["perfil"] = perfil_categorias.apply(lambda r: etiquetar(r, cortes), axis=1)

perfil_categorias.sort_values("participacion", ascending=False)