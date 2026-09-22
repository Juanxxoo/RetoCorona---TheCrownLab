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
# PENDIENTE (ver bitácora): confirmar con Corona si "Primera Fecha" representa la
# primera compra o un evento distinto (ej. fecha de registro como cliente).
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
# El factor de empaque no es siempre 2.65: en ~235,000 filas, Cantidad o Precio
# quedaron con un dígito de más (factor real = 2.65, 26.5, 265 o 2650 según la fila).
# Es un problema de escala en esas dos columnas, NO en Valor -- por eso no se
# descartan filas de Valor, solo se marca cuáles tienen Cantidad/Precio en otra
# escala (útil si más adelante se usa Cantidad para análisis de unidades, ej. m²).
CANDIDATOS_FACTOR = [2.65, 26.5, 265.0, 2650.0]

valido = (trans["Cantidad"].notna() & trans["Precio"].notna() & (trans["Valor"] > 0)
          & (trans["Cantidad"] > 0) & (trans["Precio"] > 0))
factor_crudo = (trans.loc[valido, "Cantidad"] * trans.loc[valido, "Precio"]) / trans.loc[valido, "Valor"]
errores = pd.DataFrame({c: (factor_crudo - c).abs() / c for c in CANDIDATOS_FACTOR})

trans["escala_detectada"] = pd.NA
trans["escala_valida"] = pd.NA
trans.loc[valido, "escala_detectada"] = errores.idxmin(axis=1).astype(float).values
trans.loc[valido, "escala_valida"] = (errores.min(axis=1) <= 0.02).values

print(trans.loc[valido, "escala_detectada"].value_counts())
n_atipicos = (trans["escala_valida"] == False).sum()
print(f"Filas realmente sin explicar por ningún múltiplo (revisar aparte): "
      f"{n_atipicos} ({n_atipicos/valido.sum():.3%})")
print(f"Filas sin Cantidad/Precio para verificar (Valor se conserva igual): "
      f"{(~valido).sum()}")

trans_normales = trans[trans["escala_valida"] != False].copy()  # conserva válidas + no verificables
trans_atipicos = trans[trans["escala_valida"] == False].copy()
print(f"\nDataset final: {len(trans_normales)} filas ({len(trans_normales)/len(trans):.2%} del total)")

# %% Fix 3 — Tratamiento de datos
# Solo ~50% de los clientes consintió. Decisión de negocio/ética, no solo
# técnica — ajusten el filtro según lo que definan con Corona.
print(cli_clean["Tratamiento Datos"].value_counts(normalize=True))
cli_con_consentimiento = cli_clean[cli_clean["Tratamiento Datos"] == "SÍ"]

# %% Join — dataset limpio final
df = trans_normales.merge(cli_clean, on="Cliente", how="left")
print(f"Dataset final: {df.shape}")
print(f"Clientes sin match en maestro (deberían ser ~0): {df['Tipo Cliente'].isna().sum()}")

# %% Checkpoint — guarda en formato rápido de recargar (requiere pyarrow)
df.to_parquet("../data/dataset_limpio.parquet", index=False)
print("Guardado en data/dataset_limpio.parquet")

# %% Verificación rápida del checkpoint
verificacion = pd.read_parquet("../data/dataset_limpio.parquet")
print(verificacion.shape)
print(verificacion.head(10))
# %%
