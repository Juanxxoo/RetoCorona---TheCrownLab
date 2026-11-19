# %% [markdown]
# # Validación del número de segmentos con recompra futura — Reto Corona (Retail)
# Crítica del jurado: Silhouette y Davies-Bouldin favorecen K=2, pero usamos K=5
# (consumidores) y K=4 (empresas). ¿Los segmentos adicionales sirven para algo?
#
# Prueba fuera de tiempo:
#   - Se segmenta SOLO con lo que se sabía al 31-dic-2024 (mismas 4 variables y
#     mismo preprocesamiento de 05: log1p en frecuencia y valor, estandarización).
#   - Se mide qué hizo cada cliente DESPUÉS (ene-2025 a jul-2026):
#       ¿volvió a comprar? y ¿cuánto gastó?
#   - Para cada K se calcula qué parte de la variación del comportamiento futuro
#     explican los segmentos (eta², 0 a 1). Si pasar de K=2 a K=5 aumenta
#     claramente el eta², los segmentos extra separan clientes que de verdad se
#     comportan distinto en el futuro, y K=5 queda justificado por negocio.

# %% Carga
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = Path("../data")
SALIDAS = Path("../resultados")
SALIDAS.mkdir(exist_ok=True)

df = pd.read_parquet(DATA / "dataset_limpio.parquet",
                     columns=["Cliente", "ID Pedido", "Fecha Pedido", "Valor", "Categoria",
                              "Tipo Cliente"])
CORTE = pd.Timestamp("2024-12-31")
antes = df[df["Fecha Pedido"] <= CORTE]
despues = df[df["Fecha Pedido"] > CORTE]

# %% Variables al corte (mismas de 05)
X = antes.groupby("Cliente").agg(
    recencia_dias=("Fecha Pedido", lambda s: (CORTE - s.max()).days),
    frecuencia=("ID Pedido", "nunique"),
    valor_total=("Valor", "sum"),
    n_categorias_distintas=("Categoria", "nunique"),
    tipo=("Tipo Cliente", "first"),
)
# Resultado futuro
fut = despues.groupby("Cliente").agg(valor_futuro=("Valor", "sum"),
                                     pedidos_futuros=("ID Pedido", "nunique"))
X = X.join(fut).fillna({"valor_futuro": 0, "pedidos_futuros": 0})
X["recompra"] = (X["pedidos_futuros"] > 0).astype(int)
X["log_valor_futuro"] = np.log1p(X["valor_futuro"])
print(f"Clientes con compras en 2024: {len(X):,} | recompran después: {X['recompra'].mean():.1%}")

VARS = ["recencia_dias", "frecuencia", "valor_total", "n_categorias_distintas"]

def preparar(d):
    Z = d[VARS].copy()
    Z["frecuencia"] = np.log1p(Z["frecuencia"])
    Z["valor_total"] = np.log1p(Z["valor_total"])
    return StandardScaler().fit_transform(Z)

def eta2(y, grupos):
    """Proporción de la varianza de y explicada por los grupos (0 a 1)."""
    total = ((y - y.mean()) ** 2).sum()
    entre = y.groupby(grupos).apply(lambda g: len(g) * (g.mean() - y.mean()) ** 2).sum()
    return entre / total

# %% Evaluar K = 2 a 7 por tipo de cliente
K_ELEGIDO = {"Consumidor final": 5, "Empresa": 4}
filas, detalle = [], {}
for tipo in ["Consumidor final", "Empresa"]:
    d = X[X["tipo"] == tipo].copy()
    Z = preparar(d)
    for k in range(2, 8):
        etiquetas = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
        g = pd.Series(etiquetas, index=d.index)
        filas.append({
            "Tipo Cliente": tipo, "k": k,
            "eta2_recompra": eta2(d["recompra"], g),
            "eta2_log_valor_futuro": eta2(d["log_valor_futuro"], g),
            "rango_tasa_recompra_pp": (d.groupby(g)["recompra"].mean().max()
                                       - d.groupby(g)["recompra"].mean().min()) * 100,
        })
        if k in (2, K_ELEGIDO[tipo]):
            detalle[(tipo, k)] = (d.assign(segmento=g)
                                   .groupby("segmento")
                                   .agg(clientes=("recompra", "size"),
                                        recencia_mediana=("recencia_dias", "median"),
                                        valor_mediano_2024=("valor_total", "median"),
                                        tasa_recompra=("recompra", "mean"),
                                        valor_futuro_medio=("valor_futuro", "mean"))
                                   .sort_values("tasa_recompra", ascending=False))

res = pd.DataFrame(filas)
print("\nPODER PREDICTIVO DE LA SEGMENTACIÓN SOBRE EL COMPORTAMIENTO FUTURO")
print(res.round(4).to_string(index=False))

for (tipo, k), t in detalle.items():
    print(f"\n{tipo} — K={k}: recompra futura por segmento")
    t = t.copy()
    t["tasa_recompra"] = (t["tasa_recompra"] * 100).round(1)
    print(t.round(0).to_string())

# %% Conclusión automática
for tipo, k in K_ELEGIDO.items():
    r = res[res["Tipo Cliente"] == tipo].set_index("k")
    ganancia = r.loc[k, "eta2_recompra"] / r.loc[2, "eta2_recompra"] - 1
    print(f"\n{tipo}: K={k} explica {r.loc[k, 'eta2_recompra']:.1%} de la variación en recompra "
          f"vs. {r.loc[2, 'eta2_recompra']:.1%} con K=2 ({ganancia:+.0%}). "
          f"Valor futuro: {r.loc[k, 'eta2_log_valor_futuro']:.1%} vs. {r.loc[2, 'eta2_log_valor_futuro']:.1%}.")

# %% Guardar
res.to_csv(SALIDAS / "validacion_k_recompra.csv", index=False, encoding="utf-8-sig")
pd.concat({f"{t} K={k}": v for (t, k), v in detalle.items()}).to_csv(
    SALIDAS / "validacion_k_detalle_segmentos.csv", encoding="utf-8-sig")
print(f"\nResultados guardados en {SALIDAS.resolve()}")
