# 09 - Evolución generacional
#
# Comparamos las generaciones en el tiempo: cuántos clientes nuevos llegan de cada
# una, cuántos compran cada trimestre, cuánto gastan y en qué negocios.
#
# Lee: data/dataset_limpio.parquet y data/features_clientes_segmentados.parquet
# Guarda: los archivos resultados/generacion_*.csv
# Después sigue el 10.

# %% [markdown]
# # Evolución generacional - Reto Corona (Retail)
# Pregunta de Corona: ¿cómo cambia el consumidor entre generaciones?
#
# Se compara, por generación y en el tiempo:
#   1. Clientes nuevos por mes (captación)
#   2. Clientes activos, gasto y ticket por trimestre
#   3. Mezcla de negocios (qué % del gasto va a cada negocio), por año
#   4. Canal: % de pedidos en el cluster de punto de venta principal
#
# Nota sobre "clientes nuevos": la base empieza en enero de 2024, así que los
# primeros meses mezclan clientes nuevos con clientes antiguos que aparecen por
# primera vez en la base. Por eso la captación se mide desde julio de 2024
# (6 meses de calentamiento). Los años se comparan en el mismo periodo
# (enero–julio) porque 2026 solo llega hasta julio.

# %% Carga
from pathlib import Path
import pandas as pd

DATA = Path("../data")
SALIDAS = Path("../resultados")
SALIDAS.mkdir(exist_ok=True)
ORDEN_GEN = ["Gen Z", "Millennial", "Gen X", "Boomer"]

df = pd.read_parquet(DATA / "dataset_limpio.parquet",
                     columns=["Cliente", "ID Pedido", "Fecha Pedido", "Negocio", "Valor"])
cli = pd.read_parquet(DATA / "features_clientes_segmentados.parquet",
                      columns=["Cliente", "Tipo Cliente", "generacion"])
cf = cli[(cli["Tipo Cliente"] == "Consumidor final") & cli["generacion"].isin(ORDEN_GEN)]
df = df.merge(cf[["Cliente", "generacion"]], on="Cliente")
df["mes"] = df["Fecha Pedido"].dt.to_period("M")
df["trimestre"] = df["Fecha Pedido"].dt.to_period("Q")
df["anio"] = df["Fecha Pedido"].dt.year
print(f"Líneas de consumidores con generación: {len(df):,}")

# %% 1. Captación: clientes nuevos por mes y generación
primera = df.groupby(["Cliente", "generacion"])["mes"].min().rename("mes_entrada").reset_index()
primera = primera[primera["mes_entrada"] >= pd.Period("2024-07", "M")]
nuevos = (primera.groupby(["mes_entrada", "generacion"]).size()
                 .unstack().reindex(columns=ORDEN_GEN).fillna(0))
mix_nuevos = nuevos.div(nuevos.sum(axis=1), axis=0) * 100

# Peso de cada generación entre los clientes nuevos: primeros 6 meses vs últimos 6
ini, fin = mix_nuevos.head(6).mean(), mix_nuevos.tail(6).mean()
captacion = pd.DataFrame({"% nuevos jul-dic 2024": ini, "% nuevos feb-jul 2026": fin,
                          "cambio_pp": fin - ini}).reindex(ORDEN_GEN)
print("\nCAPTACIÓN: PESO DE CADA GENERACIÓN ENTRE LOS CLIENTES NUEVOS")
print(captacion.round(1))

# %% 2. Actividad por trimestre (clientes activos, gasto, ticket)
trim = (df.groupby(["trimestre", "generacion"])
          .agg(activos=("Cliente", "nunique"), pedidos=("ID Pedido", "nunique"),
               valor=("Valor", "sum"))
          .reset_index())
trim["ticket"] = trim["valor"] / trim["pedidos"]
trim["valor_por_cliente"] = trim["valor"] / trim["activos"]

# Solo trimestres completos: la base llega hasta julio de 2026, así que 2026Q3
# tendría un solo mes y en una gráfica parecería una caída que no existe.
ultimo_mes = df["mes"].max()
completo = trim["trimestre"].map(lambda q: q.asfreq("M", "end") <= ultimo_mes)
print(f"Trimestres incompletos excluidos: {sorted(trim.loc[~completo, 'trimestre'].astype(str).unique())}")
trim = trim[completo].copy()

# %% Comparación interanual en el mismo periodo (enero–julio)
ene_jul = df[df["Fecha Pedido"].dt.month <= 7]
anual = (ene_jul.groupby(["anio", "generacion"])
                .agg(activos=("Cliente", "nunique"), pedidos=("ID Pedido", "nunique"),
                     valor=("Valor", "sum"))
                .reset_index())
anual["ticket"] = anual["valor"] / anual["pedidos"]
anual["valor_por_cliente"] = anual["valor"] / anual["activos"]
piv = anual.pivot(index="generacion", columns="anio",
                  values=["activos", "valor", "ticket"]).reindex(ORDEN_GEN)
for m in ["activos", "valor", "ticket"]:
    piv[(m, "var_24_26_%")] = (piv[(m, 2026)] / piv[(m, 2024)] - 1) * 100
print("\nENERO–JULIO: 2024 vs 2025 vs 2026 POR GENERACIÓN")
print(piv.round(0).to_string())

# %% 3. Mezcla de negocios por generación y año (% del gasto, enero–julio)
mezcla = (ene_jul.groupby(["generacion", "anio", "Negocio"], observed=True)["Valor"].sum()
                 .groupby(level=[0, 1]).transform(lambda s: s / s.sum() * 100)
                 .rename("pct_gasto").reset_index())
mz = mezcla.pivot_table(index=["generacion", "Negocio"], columns="anio",
                        values="pct_gasto").reindex(ORDEN_GEN, level=0)
mz["cambio_pp_24_26"] = mz[2026] - mz[2024]
print("\nMEZCLA DE NEGOCIOS (% DEL GASTO, ENERO–JULIO)")
print(mz.round(1).to_string())

# Qué distingue a cada generación: % del gasto vs. el promedio de todas
prom = (ene_jul.groupby("Negocio", observed=True)["Valor"].sum()
        / ene_jul["Valor"].sum() * 100)
gen_vs_prom = (ene_jul.groupby(["generacion", "Negocio"], observed=True)["Valor"].sum()
               .groupby(level=0).transform(lambda s: s / s.sum() * 100)
               .unstack().reindex(ORDEN_GEN))
indice_mezcla = gen_vs_prom.div(prom, axis=1) * 100
print("\nÍNDICE DE MEZCLA VS. PROMEDIO (100 = igual al promedio)")
print(indice_mezcla.round(0).to_string())

# %% Guardar
nuevos.to_csv(SALIDAS / "generacion_nuevos_por_mes.csv", encoding="utf-8-sig")
trim.to_csv(SALIDAS / "generacion_actividad_trimestral.csv", index=False, encoding="utf-8-sig")
piv.round(2).to_csv(SALIDAS / "generacion_comparacion_ene_jul.csv", encoding="utf-8-sig")
mz.round(2).to_csv(SALIDAS / "generacion_mezcla_negocios.csv", encoding="utf-8-sig")
indice_mezcla.round(1).to_csv(SALIDAS / "generacion_indice_mezcla.csv", encoding="utf-8-sig")
print(f"\nResultados guardados en {SALIDAS.resolve()}")

# %%
