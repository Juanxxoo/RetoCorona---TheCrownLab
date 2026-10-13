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

# %% Fix 2 — Valor vs. Cantidad × Precio: no es un factor fijo, son múltiplos de 10
# Corona confirmó que Cantidad, Precio y Valor fueron anonimizados con factores distintos.
# Por eso no se exige que Cantidad × Precio = Valor en la base entregada.
# Este bloque se conserva como diagnóstico y NO se eliminan filas por esta diferencia.

CANDIDATOS_FACTOR = [2.65, 26.5, 265.0, 2650.0]

valido = (trans["Cantidad"].notna() & trans["Precio"].notna() & (trans["Valor"] > 0)
          & (trans["Cantidad"] > 0) & (trans["Precio"] > 0))

factor_crudo = (
    trans.loc[valido, "Cantidad"] * trans.loc[valido, "Precio"]
) / trans.loc[valido, "Valor"]

errores = pd.DataFrame({
    c: (factor_crudo - c).abs() / c
    for c in CANDIDATOS_FACTOR
})

trans["escala_detectada"] = pd.NA
trans["escala_valida"] = pd.NA

trans.loc[valido, "escala_detectada"] = (
    errores.idxmin(axis=1).astype(float).values
)

trans.loc[valido, "escala_valida"] = (
    errores.min(axis=1) <= 0.02
).values

print(trans.loc[valido, "escala_detectada"].value_counts())

n_atipicos = (trans["escala_valida"] == False).sum()

print(
    f"Filas que no siguen los factores observados: "
    f"{n_atipicos} ({n_atipicos/valido.sum():.3%})"
)

print(
    f"Filas fuera del diagnóstico "
    f"(faltantes, ceros, negativos o devoluciones): "
    f"{(~valido).sum()}"
)

trans_atipicos = trans[
    trans["escala_valida"] == False
].copy()

print(
    "Estas filas NO se eliminan del análisis "
    "por tratarse de datos anonimizados."
)

# Se conservan todas las transacciones originales
trans_normales = trans.copy()

print(
    f"\nDataset final: {len(trans_normales)} filas "
    f"({len(trans_normales)/len(trans):.2%} del total)"
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