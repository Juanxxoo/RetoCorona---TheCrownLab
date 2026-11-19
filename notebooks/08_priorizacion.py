# %% [markdown]
# # Índice de priorización y tabla de acciones — Reto Corona (Retail)
# Une todo lo anterior en una sola respuesta: ¿dónde, con quién y con qué negocio
# debe actuar Corona primero?
#
# Unidad: departamento × generación (consumidores finales).
# Componentes del puntaje (cada uno se lleva a percentil 0–1 dentro de su tipo):
#   1. Brecha de penetración (clientes potenciales, de 07)       peso 0,40
#   2. Tamaño de la población de esa generación                  peso 0,20
#   3. Dinamismo de la construcción (m² licenciados por 1.000 hab) peso 0,20
#   4. Crecimiento de hogares (%)                                 peso 0,10
#   5. Déficit habitacional cualitativo (plus de sostenibilidad)  peso 0,10
# Los pesos se ponen a prueba con 2.000 combinaciones aleatorias: una oportunidad
# es robusta si queda en el top 10 en la mayoría de combinaciones.
#
# Acción sugerida: según el segmento sobrerrepresentado entre los clientes de
# esa celda y las reglas de afinidad de 06 (siempre hacia Baños), más el negocio
# con menor penetración relativa en esa celda.

# %% Carga
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

RAIZ = Path("..")
DATA = RAIZ / "data"
SALIDAS = RAIZ / "resultados"
SALIDAS.mkdir(exist_ok=True)

pen = pd.read_csv(SALIDAS / "penetracion_dep_generacion.csv")          # sale de 07
clientes = pd.read_parquet(DATA / "features_clientes_segmentados.parquet")
compras = pd.read_parquet(DATA / "dataset_limpio.parquet", columns=["Cliente", "Negocio"])

def normalizar(txt):
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return txt.upper().strip()

# Mismo mapeo que 07 (Región Corona / nombre externo → código DANE)
NOMBRE_A_DP = {
    "ANTIOQUIA": 5, "ATLANTICO": 8, "BOGOTA": 11, "BOLIVAR": 13, "BOYACA": 15,
    "CALDAS": 17, "CAQUETA": 18, "CAUCA": 19, "CESAR": 20, "CORDOBA": 23,
    "CUNDINAMARCA": 25, "CHOCO": 27, "HUILA": 41, "LA GUAJIRA": 44,
    "MAGDALENA": 47, "META": 50, "NARINO": 52, "NORTE SANTANDER": 54,
    "NORTE DE SANTANDER": 54, "QUINDIO": 63, "RISARALDA": 66, "SANTANDER": 68,
    "SUCRE": 70, "TOLIMA": 73, "VALLE": 76, "ARAUCA": 81, "CASANARE": 85,
    "PUTUMAYO": 86, "SAN ANDRES": 88, "SAN ANDRES Y PROVIDENCIA": 88,
    "AMAZONAS": 91, "GUAINIA": 94, "GUAVIARE": 95, "VAUPES": 97, "VICHADA": 99,
}

# %% Fuentes externas: lectura y correcciones documentadas
# Los CSV crudos NO se modifican; las correcciones quedan aquí.
def leer_externo(nombre):
    d = pd.read_csv(RAIZ / nombre, skiprows=1, decimal=",")
    d = d[d.iloc[:, 0].notna() & ~d.iloc[:, 0].astype(str).str.startswith(("(No", "Nota"))]
    d["dp"] = d.iloc[:, 0].map(lambda x: NOMBRE_A_DP[normalizar(x)])
    return d

nec = leer_externo("Busqueda fuentes externas corona - Necesidades sostenibles.csv")
cre = leer_externo("Busqueda fuentes externas corona - Crecimiento.csv")

# Corrección 1: déficit cualitativo y cuantitativo están invertidos en el CSV.
# Evidencia: Bogotá trae 1,6 % "cualitativo" y 9,0 % "cuantitativo", que son las
# cifras de la ECV 2021 de Bogotá pero al revés (cuantitativo 1,6 %, cualitativo
# 9,0 %); y a nivel nacional el cualitativo es ~3 veces el cuantitativo, no al revés.
# Las cifras corresponden a la ECV 2021 del DANE.
nec = nec.rename(columns={
    "Déficit habitacional cualitativo (%)": "deficit_cuantitativo",
    "Déficit habitacional cuantitativo (%)": "deficit_cualitativo",
})
# Corrección 2: Huila trae 999 % de cobertura de energía (error de digitación).
col_energia = "Cobertura energía eléctrica (%)"
nec.loc[nec[col_energia] > 100, col_energia] = np.nan

terr = (nec[["dp", "deficit_cualitativo", "deficit_cuantitativo"]]
        .merge(cre[["dp", "Área licenciada (m²)", "Crecimiento de hogares (%)"]], on="dp")
        .rename(columns={"Área licenciada (m²)": "area_licenciada_m2",
                         "Crecimiento de hogares (%)": "crecimiento_hogares"}))
pob_total = pen.groupby("dp")["poblacion"].sum()   # población 19–68 años (base comparable)
terr["m2_licenciados_por_mil_hab"] = terr["area_licenciada_m2"] / terr["dp"].map(pob_total) * 1000
terr.to_csv(SALIDAS / "base_territorial_corregida.csv", index=False, encoding="utf-8-sig")

# %% Segmento característico de cada celda (departamento × generación)
# El segmento más grande casi siempre es "recientes" (29 % de la base), así que no
# distingue una celda de otra. Se usa el segmento SOBRERREPRESENTADO: el de mayor
# cociente entre su peso en la celda y su peso nacional en esa generación.
cf = clientes[(clientes["Tipo Cliente"] == "Consumidor final")].copy()
cf["dp"] = cf["region_dominante"].map(lambda r: NOMBRE_A_DP.get(normalizar(r)))
cf = cf.dropna(subset=["dp"])
cf["dp"] = cf["dp"].astype(int)

MIN_CLIENTES_SEG = 200
peso_celda = (cf.groupby(["dp", "generacion"])["nombre_segmento"]
                .value_counts(normalize=True).rename("peso").reset_index())
peso_nac = (cf.groupby("generacion")["nombre_segmento"]
              .value_counts(normalize=True).rename("peso_nac").reset_index())
peso_celda = peso_celda.merge(peso_nac, on=["generacion", "nombre_segmento"])
peso_celda["sobrerrep"] = peso_celda["peso"] / peso_celda["peso_nac"]
seg_car = (peso_celda.sort_values("sobrerrep", ascending=False)
             .drop_duplicates(["dp", "generacion"])
             .set_index(["dp", "generacion"])
             .rename(columns={"nombre_segmento": "segmento_caracteristico",
                              "peso": "pct_segmento", "sobrerrep": "sobrerrep_segmento"})
             [["segmento_caracteristico", "pct_segmento", "sobrerrep_segmento"]])
n_celda = cf.groupby(["dp", "generacion"]).size()
seg_car.loc[n_celda[n_celda < MIN_CLIENTES_SEG].index.intersection(seg_car.index),
            "segmento_caracteristico"] = "Base insuficiente"

# %% Negocio con menor penetración relativa en cada celda
# % de clientes de la celda que compran cada negocio, frente al mismo % nacional
# de esa generación. El de menor cociente es el negocio "a impulsar".
cli_neg = compras.drop_duplicates().merge(cf[["Cliente", "dp", "generacion"]], on="Cliente")
base_celda = cf.groupby(["dp", "generacion"]).size().rename("n")
compra_celda = cli_neg.groupby(["dp", "generacion", "Negocio"], observed=True).size().rename("compran")
tasa = compra_celda.reset_index().merge(base_celda.reset_index(), on=["dp", "generacion"])
tasa["tasa"] = tasa["compran"] / tasa["n"]
nac = (cli_neg.groupby(["generacion", "Negocio"], observed=True).size()
       / cf.groupby("generacion").size()).rename("tasa_nacional").reset_index()
tasa = tasa.merge(nac, on=["generacion", "Negocio"])
tasa["indice_negocio"] = tasa["tasa"] / tasa["tasa_nacional"] * 100
# Solo negocios relevantes (Servicios es muy pequeño para usarlo como palanca)
tasa = tasa[tasa["Negocio"].astype(str) != "SERVICIOS"]
neg_impulsar = (tasa.sort_values("indice_negocio")
                    .drop_duplicates(["dp", "generacion"])
                    .set_index(["dp", "generacion"])[["Negocio", "indice_negocio"]]
                    .rename(columns={"Negocio": "negocio_a_impulsar",
                                     "indice_negocio": "indice_negocio_vs_nacional"}))

# %% Tabla base del índice
idx = (pen.merge(terr, on="dp", how="left")
          .set_index(["dp", "generacion"])
          .join(seg_car).join(neg_impulsar)
          .reset_index())

COMPONENTES = {
    "clientes_potenciales": 0.40,
    "poblacion": 0.20,
    "m2_licenciados_por_mil_hab": 0.20,
    "crecimiento_hogares": 0.10,
    "deficit_cualitativo": 0.10,
}

def puntuar(df, pesos):
    p = pd.Series(pesos) / sum(pesos.values())
    ranks = pd.DataFrame({c: df[c].rank(pct=True) for c in p.index})
    return (ranks * p).sum(axis=1) * 100

# Se puntúa por separado dentro de cada tipo de oportunidad (no tiene sentido
# competir una apertura de tienda contra una campaña en una ciudad con tienda).
idx = idx[idx["clientes_potenciales"] > 0].copy()
idx["puntaje"] = idx.groupby("tipo_oportunidad", group_keys=False).apply(
    lambda g: puntuar(g, COMPONENTES))

# %% Sensibilidad: 2.000 combinaciones aleatorias de pesos (Dirichlet)
rng = np.random.default_rng(42)
N_SIM, TOP = 2000, 10
veces_top = pd.Series(0, index=idx.index, dtype=float)
for _ in range(N_SIM):
    w = dict(zip(COMPONENTES, rng.dirichlet(np.ones(len(COMPONENTES)))))
    for tipo, g in idx.groupby("tipo_oportunidad"):
        top = puntuar(g, w).nlargest(TOP).index
        veces_top[top] += 1
idx["pct_simulaciones_top10"] = veces_top / N_SIM * 100

# %% Acción sugerida
# Reglas priorizadas de 06 por segmento (antecedente → consecuente):
REGLA_POR_SEGMENTO = {
    "Consumidores consolidados de alta vinculación":
        "Venta cruzada Cocinas → Baños (lift 2,8; 43 % de confianza)",
    "Consumidores activos en desarrollo":
        "Servicios de instalación → Baños (lift 3,8) y Cocinas → Baños",
    "Consumidores recientes de vínculo inicial":
        "Segunda compra: Servicios → Baños (lift 2,7–3,0; 42 % de confianza)",
    "Consumidores inactivos de vínculo intermedio":
        "Reactivación: Pinturas → Baños (lift 2,7)",
    "Consumidores inactivos de bajo vínculo":
        "Reactivación con oferta de entrada (bajo historial: sin regla fuerte)",
}

def accion(r):
    neg = str(r["negocio_a_impulsar"]).title()
    if r["tipo_oportunidad"] == "Expansión":
        return (f"Captar {r['generacion']} sin tienda física: canal digital/aliados; "
                f"entrada por {neg}")
    regla = REGLA_POR_SEGMENTO.get(r["segmento_caracteristico"], "Venta cruzada hacia Baños")
    return f"{regla}; impulsar {neg}"

idx["accion_sugerida"] = idx.apply(accion, axis=1)

COLS = ["tipo_oportunidad", "departamento", "generacion", "tiendas", "clientes",
        "por_mil", "p75_generacion", "clientes_potenciales", "m2_licenciados_por_mil_hab",
        "crecimiento_hogares", "deficit_cualitativo", "segmento_caracteristico",
        "pct_segmento", "sobrerrep_segmento", "negocio_a_impulsar", "indice_negocio_vs_nacional",
        "puntaje", "pct_simulaciones_top10", "accion_sugerida"]
ranking = idx.sort_values(["tipo_oportunidad", "puntaje"], ascending=[True, False])[COLS]

pd.set_option("display.width", 250)
for tipo in ["Activación", "Expansión"]:
    r = ranking[ranking["tipo_oportunidad"] == tipo].head(10)
    print(f"\nTOP 10 — {tipo.upper()}")
    print(r[["departamento", "generacion", "clientes_potenciales", "puntaje",
             "pct_simulaciones_top10", "segmento_caracteristico", "negocio_a_impulsar"]]
          .round(1).to_string(index=False))

print("\nOPORTUNIDADES ROBUSTAS (top 10 en ≥ 60 % de las combinaciones de pesos)")
print(ranking[ranking["pct_simulaciones_top10"] >= 60]
      [["tipo_oportunidad", "departamento", "generacion", "pct_simulaciones_top10",
        "accion_sugerida"]].round(1).to_string(index=False))

# %% Guardar
ranking.to_csv(SALIDAS / "priorizacion_oportunidades.csv", index=False, encoding="utf-8-sig")
print(f"\nGuardado en {(SALIDAS / 'priorizacion_oportunidades.csv').resolve()}")
