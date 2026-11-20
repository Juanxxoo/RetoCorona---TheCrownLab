# %% [markdown]
# # Carga y limpieza inicial — Reto Corona (Retail)
# Cada celda está delimitada con `# %%` — en VSCode con la extensión de
# Jupyter, esto se comporta como un notebook aunque el archivo sea .py.
# Ajusta las rutas si tu carpeta `data/` está en otro lugar.

# %% Carga
import pandas as pd

TRANS_PATH = "../data/base_transaccional.csv"
CLI_PATH = "../data/maestro_clientes.csv"

trans = pd.read_csv(
    TRANS_PATH, sep=";", encoding="utf-8-sig", parse_dates=["Fecha Pedido"],
    dtype={
        "ID Pedido": str, "Cliente": str, "Asesor": str,
        "Region": "category", "Municipio": str, "Punto de Venta": str,
        "Cluster": "category", "Negocio": "category",
        "Categoria": "category", "Subategoria": "category",  # sic: así viene en el CSV, sin "c"
        "Producto": str,
    },
)

cli = pd.read_csv(
    CLI_PATH, sep=";", encoding="utf-8-sig", parse_dates=["Primera Fecha"],
    dtype={"Cliente": str, "Tipo Cliente": "category", "Genero": "category",
           "Tratamiento Datos": "category"},
)

print(f"Transacciones: {trans.shape}, Clientes (crudo, con duplicados): {cli.shape}")

# %% Fix 0 — Corrimiento de fechas (+24 años)
# Las fechas vienen corridas 24 años hacia atrás: enero de 2000 = enero de 2024
# (señalado en el feedback de la entrega intermedia). Se corrige aquí, al inicio,
# para que TODO el pipeline (Primera Fecha, RFM, evolución temporal, cruces con
# DANE) trabaje con meses reales. Las diferencias entre fechas no cambian, así
# que la recencia y la antigüedad quedan iguales.
# Nota: 2000 y 2024 son bisiestos, así que el 29-feb se conserva sin problema.
CORRIMIENTO_ANIOS = 24
trans["Fecha Pedido"] = trans["Fecha Pedido"] + pd.DateOffset(years=CORRIMIENTO_ANIOS)
cli["Primera Fecha"] = cli["Primera Fecha"] + pd.DateOffset(years=CORRIMIENTO_ANIOS)
print(f"Rango real de pedidos: {trans['Fecha Pedido'].min().date()} "
      f"a {trans['Fecha Pedido'].max().date()}")

# %% Fix 1 — Cliente no es llave única en maestro_clientes
# 60,158 clientes tienen 2 filas casi idénticas, solo Primera Fecha difiere.
# En vez de elegir arbitrariamente cuál de las dos fechas creerle, recalculamos
# Primera Fecha directamente desde la fuente más confiable: la fecha mínima real
# de compra en base_transaccional. Todo cliente de maestro tiene transacciones,
# así que esto no genera nulos.
# Confirmado con Corona: "Primera Fecha" representa la primera compra real.
cli_clean = cli.drop_duplicates(subset="Cliente", keep="first").reset_index(drop=True)

primera_compra_real = trans.groupby("Cliente")["Fecha Pedido"].min().rename("Primera Fecha_real")
cli_clean = (
    cli_clean.drop(columns="Primera Fecha")
             .merge(primera_compra_real, on="Cliente", how="left")
             .rename(columns={"Primera Fecha_real": "Primera Fecha"})
)
print(f"Clientes tras deduplicar: {cli_clean.shape[0]} "
      f"(se eliminaron {cli.shape[0] - cli_clean.shape[0]} filas duplicadas)")
print(f"Clientes sin Primera Fecha recalculada (deberían ser 0): "
      f"{cli_clean['Primera Fecha'].isna().sum()}")

# %% Fix 2 — Errores de decimales en Valor
# Relación de la anonimización: Valor = (Cantidad / 2,65) × Precio.
# Equivalente: Cantidad × Precio / Valor = 2,65. Se cumple en ~91,4 % de las líneas.
# En ~8,6 % el factor sale 26,5 / 265 / 2650 (errores de decimales).
#
# ¿Qué campo está mal? Se comparó cada fila con error contra la mediana del
# MISMO producto en las filas sanas:
#   - Cantidad / referencia ≈ 1,0  → Cantidad está bien
#   - Precio / referencia   ≈ 1,0  → Precio está bien
#   - Valor unitario / referencia ≈ 1/10, 1/100, 1/1000 → Valor está DESINFLADO
# Conclusión: el error está en Valor (le faltan 1, 2 o 3 ceros). Se corrige
# multiplicando Valor por la potencia detectada. Impacto: +4,5 % en el valor
# total; 17 % de los clientes cambia su valor total en más de 10 %.
# La tasa de error es estable entre meses (7–9 %) y negocios (8–13 %).

FACTOR_BASE = 2.65
POTENCIAS = [1, 10, 100, 1000]
TOLERANCIA = 0.02  # 2 % de error relativo

valido = (trans["Cantidad"].notna() & trans["Precio"].notna() & (trans["Valor"] > 0)
          & (trans["Cantidad"] > 0) & (trans["Precio"] > 0))

factor_crudo = trans.loc[valido, "Cantidad"] * trans.loc[valido, "Precio"] / trans.loc[valido, "Valor"]
errores = pd.DataFrame({p: (factor_crudo - FACTOR_BASE * p).abs() / (FACTOR_BASE * p)
                        for p in POTENCIAS})
potencia = errores.idxmin(axis=1).where(errores.min(axis=1) <= TOLERANCIA)  # NaN si no se ajusta a ninguno

print("Distribución del factor detectado (filas válidas):")
print(potencia.value_counts(normalize=True, dropna=False).round(4))

a_corregir = potencia[potencia > 1].index
trans["Valor_original"] = trans["Valor"]
trans["valor_corregido"] = False
trans.loc[a_corregir, "Valor"] = trans.loc[a_corregir, "Valor"] * potencia.loc[a_corregir]
trans.loc[a_corregir, "valor_corregido"] = True

# Verificación: tras corregir, las filas ajustadas cumplen el factor 2,65
ok = potencia.dropna().index
factor_post = trans.loc[ok, "Cantidad"] * trans.loc[ok, "Precio"] / trans.loc[ok, "Valor"]
print(f"\nFilas corregidas: {len(a_corregir):,} ({len(a_corregir)/valido.sum():.2%} de las válidas)")
print(f"Cumplen el factor 2,65 tras corregir: {((factor_post / FACTOR_BASE - 1).abs() <= TOLERANCIA).mean():.4%}")
print(f"Valor total: {trans['Valor_original'].sum():,.0f} → {trans['Valor'].sum():,.0f} "
      f"(+{trans['Valor'].sum() / trans['Valor_original'].sum() - 1:.1%})")
print(f"Filas que no se ajustan a ningún factor (sin tocar): {potencia.isna().sum():,}")
print(f"Filas fuera del diagnóstico (faltantes, ceros, negativos): {(~valido).sum():,}")

# Ninguna fila se elimina
trans_normales = trans.copy()

# %%
# Validación adicional: comprobar que el problema está en Valor

# Recuperamos el factor original, antes de corregir Valor
valido_original = (
    trans["Cantidad"].notna()
    & trans["Precio"].notna()
    & trans["Valor_original"].notna()
    & (trans["Cantidad"] > 0)
    & (trans["Precio"] > 0)
    & (trans["Valor_original"] > 0)
)

factor_original = (
    trans.loc[valido_original, "Cantidad"]
    * trans.loc[valido_original, "Precio"]
    / trans.loc[valido_original, "Valor_original"]
)

# Filas normales vs filas corregidas
idx_normales = factor_original[
    ((factor_original / 2.65 - 1).abs() <= 0.02)
].index

idx_error = trans.index[
    trans["valor_corregido"] == True
]

# Mediana de Cantidad y Precio por producto usando solo filas normales
referencia = (
    trans.loc[idx_normales]
    .groupby("Producto")[["Cantidad", "Precio"]]
    .median()
    .rename(columns={
        "Cantidad": "cantidad_mediana_producto",
        "Precio": "precio_mediano_producto"
    })
)

comparacion = (
    trans.loc[idx_error, ["Producto", "Cantidad", "Precio"]]
    .join(referencia, on="Producto")
)

comparacion["ratio_cantidad"] = (
    comparacion["Cantidad"]
    / comparacion["cantidad_mediana_producto"]
)

comparacion["ratio_precio"] = (
    comparacion["Precio"]
    / comparacion["precio_mediano_producto"]
)

print("Mediana ratio Cantidad:")
print(comparacion["ratio_cantidad"].median())

print("\nMediana ratio Precio:")
print(comparacion["ratio_precio"].median())

print("\nResumen ratio Precio:")
print(comparacion["ratio_precio"].describe())

# %% Fix 2b — Etiqueta legible de categoría (código + negocio)
# Los códigos de categoría solos no se entienden en el informe. Cada categoría
# pertenece a un único negocio (verificado en 06_market_basket), así que se crea
# una etiqueta "Negocio · Categoría" para usar en tablas y gráficas.
assert (trans_normales.groupby("Categoria", observed=True)["Negocio"].nunique() <= 1).all(), \
    "Hay categorías en más de un negocio: revisar antes de etiquetar"
trans_normales["categoria_label"] = (
    trans_normales["Negocio"].astype(str) + " · " + trans_normales["Categoria"].astype(str)
)

# %% Fix 3 — Tratamiento de datos
# Solo ~50% de los clientes consintió. Decisión del equipo: se trabaja con el
# universo completo de clientes (no se filtra) porque el análisis es agregado
# y no genera acciones a nivel de cliente individual. Si en una fase posterior
# (ej. piloto de recomendaciones) se llega a actuar sobre clientes puntuales,
# se debe retomar este filtro y cruzar contra "Tratamiento Datos" == "SÍ"
# antes de ejecutar cualquier acción sobre esos clientes específicos.
print(cli_clean["Tratamiento Datos"].value_counts(normalize=True))
cli_con_consentimiento = cli_clean[cli_clean["Tratamiento Datos"] == "SÍ"]
# Nota: cli_con_consentimiento se deja calculado por si se necesita más adelante,
# pero el join usa cli_clean (universo completo) — ver justificación arriba.

# %% Join — dataset limpio final
df = trans_normales.merge(cli_clean, on="Cliente", how="left")
print(f"Dataset final: {df.shape}")
print(f"Clientes sin match en maestro (deberían ser ~0): {df['Tipo Cliente'].isna().sum()}")

# %% Revisar cliente sin match en maestro
sin_match = df[df["Tipo Cliente"].isna()].copy()

print(sin_match)
print("\nClientes únicos sin match:", sin_match["Cliente"].nunique())

# %% Checkpoint — guarda en formato rápido de recargar (requiere pyarrow)
df.to_parquet("../data/dataset_limpio.parquet", index=False)
print("Guardado en data/dataset_limpio.parquet")

# %% Verificación rápida del checkpoint
verificacion = pd.read_parquet("../data/dataset_limpio.parquet")
print(verificacion.shape)
print(verificacion.head(10))
# %%