# 07 - Penetración por generación y departamento
#
# Calculamos cuántos clientes tiene Corona por cada 1.000 habitantes de cada
# generación en cada departamento, con las proyecciones de población del DANE.
# Con eso vemos dónde hay baja penetración y cuántos clientes se podrían ganar.
#
# Lee: data/features_clientes_segmentados.parquet, data/dataset_limpio.parquet
#      y data/PPED-AreaSexoEdadDep-2018-2050_VP.xlsx (DANE)
# Guarda: los archivos resultados/penetracion_*.csv
# Después sigue el 08.

# %% [markdown]
# # Penetración por generación y departamento - Reto Corona (Retail)
# Pregunta de Corona: ¿dónde hay baja penetración y cómo cambia entre generaciones?
#
# Métrica: clientes consumidores por cada 1.000 habitantes de la misma generación
# y departamento, con las proyecciones de población del DANE (base CNPV 2018).
#
# Decisiones:
# - Solo consumidores finales (las empresas no tienen edad ni población comparable).
# - El departamento del cliente es su región dominante de compra (99,5 % compra
#   en una sola región).
# - La edad en la base va de 19 a 68 años (anonimizada). Para comparar peras con
#   peras, la población DANE se recorta al MISMO rango de edades por generación.
# - Generación según edad en 2026 (mismos cortes de 02_feature_engineering):
#   Gen Z 19–29, Millennial 30–45, Gen X 46–61, Boomer 62–68.
# - Año de población: 2025 (mitad del periodo observado, ene-2024 a jul-2026).
#   Para la comparación anual (2024 vs 2025) cada año usa su propia población.
# - Presencia física: la base no trae la ubicación de los puntos de venta, pero
#   cada PdV le vende ~90 % (mediana) a un solo departamento; ese es su departamento
#   "sede" (solo la aceptamos si al menos el 80 % de sus ventas son de ahí).
#   Con eso se separan dos tipos de oportunidad:
#     * Activación: hay tienda y la penetración es baja -> acción comercial
#     * Expansión: no hay tienda -> decisión de canal (digital, aliados, apertura)
# - Brecha: clientes adicionales si el departamento llegara al percentil 75
#   de penetración de esa generación, calculado SOLO entre departamentos con
#   tienda (el referente justo: compararse con quien sí tiene presencia).

# %% Carga
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

DATA = Path("../data")
SALIDAS = Path("../resultados")
SALIDAS.mkdir(exist_ok=True)

ARCHIVO_DANE = DATA / "PPED-AreaSexoEdadDep-2018-2050_VP.xlsx"
ANIO_POBLACION = 2025
MIN_CLIENTES = 200     # debajo de esto la tasa es muy ruidosa: se reporta pero no define el P75

clientes = pd.read_parquet(DATA / "features_clientes_segmentados.parquet")
transacciones = pd.read_parquet(DATA / "dataset_limpio.parquet",
                                columns=["Cliente", "Fecha Pedido", "Region", "Punto de Venta"])

# %% Mapeo Región Corona -> código DANE (DIVIPOLA)
# Region de Corona = departamento, con Bogotá separada de Cundinamarca.
def normalizar(txt):
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return txt.upper().strip()

REGION_A_DP = {
    "ANTIOQUIA": 5, "ATLANTICO": 8, "BOGOTA": 11, "BOLIVAR": 13, "BOYACA": 15,
    "CALDAS": 17, "CAQUETA": 18, "CAUCA": 19, "CESAR": 20, "CORDOBA": 23,
    "CUNDINAMARCA": 25, "CHOCO": 27, "HUILA": 41, "LA GUAJIRA": 44,
    "MAGDALENA": 47, "META": 50, "NARINO": 52, "NORTE SANTANDER": 54,
    "QUINDIO": 63, "RISARALDA": 66, "SANTANDER": 68, "SUCRE": 70,
    "TOLIMA": 73, "VALLE": 76, "ARAUCA": 81, "CASANARE": 85, "PUTUMAYO": 86,
    "SAN ANDRES": 88, "AMAZONAS": 91, "GUAINIA": 94, "GUAVIARE": 95,
    "VAUPES": 97, "VICHADA": 99,
}

# %% Generaciones (rango de edad en 2026 -> año de nacimiento)
GENERACIONES = {           # edad en 2026 (incluye ambos extremos)
    "Gen Z": (19, 29),
    "Millennial": (30, 45),
    "Gen X": (46, 61),
    "Boomer": (62, 68),
}
ORDEN_GEN = list(GENERACIONES)

def generacion_por_edad(edad_2026):
    for gen, (lo, hi) in GENERACIONES.items():
        if lo <= edad_2026 <= hi:
            return gen
    return None


# %% Población DANE en formato largo: (dp, anio, edad, poblacion)
def cargar_poblacion_dane(ruta):
    """Lee el anexo departamental del DANE (actualizado el 30-jul-2025) y devuelve
    población total (ambos sexos, área total) por departamento, año y edad simple.
    Hoja 'PobDepartamentalxÁreaSexoEdad': encabezados en la fila 9; las 4 primeras
    columnas son DP, DPNOM, AÑO, ÁREA GEOGRÁFICA; las edades del total por sexos
    vienen como 'Total 0 años', 'Total 1 año', ..., 'Total 100 años y más'."""
    df = pd.read_excel(ruta, sheet_name="PobDepartamentalxÁreaSexoEdad", header=8)
    c = list(df.columns)
    df = df.rename(columns={c[0]: "dp", c[1]: "dpnom", c[2]: "anio", c[3]: "area"})
    df = df[df["area"] == "Total"].dropna(subset=["dp"])
    cols_edad = [col for col in df.columns
                 if isinstance(col, str) and col.startswith("Total ") and "año" in col]
    assert len(cols_edad) == 101, f"Se esperaban 101 edades (0 a 100+), hay {len(cols_edad)}"
    largo = df.melt(id_vars=["dp", "anio"], value_vars=cols_edad,
                    var_name="edad", value_name="poblacion")
    largo["edad"] = largo["edad"].str.extract(r"(\d+)", expand=False).astype(int)
    largo["dp"] = largo["dp"].astype(int)
    largo["anio"] = largo["anio"].astype(int)
    return largo[["dp", "anio", "edad", "poblacion"]]

pob = cargar_poblacion_dane(ARCHIVO_DANE)
print(f"Población DANE: {pob['dp'].nunique()} departamentos, años {pob['anio'].min()}–{pob['anio'].max()}")

def poblacion_generacion(anio):
    """Población por dp y generación en un año dado. La generación es una
    cohorte de nacimiento fija: en 2024 cada cohorte tiene 2 años menos que en 2026."""
    p = pob[pob["anio"] == anio].copy()
    p["edad_2026"] = p["edad"] + (2026 - anio)
    p["generacion"] = p["edad_2026"].map(generacion_por_edad)
    return (p.dropna(subset=["generacion"])
             .groupby(["dp", "generacion"], as_index=False)["poblacion"].sum())


# %%
# Contamos clientes por departamento y generación, y miramos en qué
# departamentos hay tienda.

# %% Clientes consumidores por departamento y generación
cf = clientes[(clientes["Tipo Cliente"] == "Consumidor final")
              & clientes["generacion"].isin(ORDEN_GEN)].copy()
cf["dp"] = cf["region_dominante"].map(lambda r: REGION_A_DP.get(normalizar(r)))
sin_region = cf["dp"].isna().sum()
print(f"Consumidores con generación: {len(cf):,} | sin región identificable (excluidos): "
      f"{sin_region:,} ({sin_region/len(cf):.1%})")
cf = cf.dropna(subset=["dp"])
cf["dp"] = cf["dp"].astype(int)

nombre_dp = (cf.groupby("dp")["region_dominante"].first()
               .map(lambda r: normalizar(r).title()))

# %% DIAGNÓSTICO DE REGIONES NO MAPEADAS

cf_revision = clientes[
    (clientes["Tipo Cliente"] == "Consumidor final")
    & clientes["generacion"].isin(ORDEN_GEN)
].copy()

cf_revision["region_normalizada"] = cf_revision["region_dominante"].map(normalizar)
cf_revision["dp_revision"] = cf_revision["region_normalizada"].map(REGION_A_DP)

print("REGIONES NO MAPEADAS:")
print(
    cf_revision.loc[
        cf_revision["dp_revision"].isna(),
        "region_normalizada"
    ]
    .value_counts(dropna=False)
    .to_string()
)

# %% Departamento sede de cada punto de venta

UMBRAL_SEDE_PDV = 0.80

ventas_pdv = (
    transacciones
    .assign(
        dp=transacciones["Region"].map(
            lambda r: REGION_A_DP.get(normalizar(r))
        )
    )
    .dropna(subset=["dp"])
    .groupby(["Punto de Venta", "dp"])
    .size()
    .rename("lineas")
    .reset_index()
)

ventas_pdv["participacion"] = (
    ventas_pdv["lineas"]
    / ventas_pdv.groupby("Punto de Venta")["lineas"].transform("sum")
)

# Departamento con mayor concentración para cada PdV
sede_pdv = (
    ventas_pdv
    .sort_values("participacion", ascending=False)
    .drop_duplicates("Punto de Venta")
)

# Solo aceptamos la sede si al menos 80% de sus líneas
# corresponden al mismo departamento.
sede_pdv_clara = sede_pdv[
    sede_pdv["participacion"] >= UMBRAL_SEDE_PDV
].copy()

print(
    f"Puntos de venta totales: {len(sede_pdv)} | "
    f"participación mediana del departamento dominante: "
    f"{sede_pdv['participacion'].median():.0%}"
)

print(
    f"PdV con sede territorial clara (>= {UMBRAL_SEDE_PDV:.0%}): "
    f"{len(sede_pdv_clara)} | "
    f"departamentos con presencia inferida: "
    f"{sede_pdv_clara['dp'].nunique()}"
)

tiendas_dp = (
    sede_pdv_clara
    .groupby("dp")
    .size()
    .rename("tiendas")
)

tiendas_dp.index = tiendas_dp.index.astype(int)
# %% DIAGNÓSTICO DE CLARIDAD DE SEDE POR PUNTO DE VENTA

print("Distribución de participación del departamento sede:")
print(sede_pdv["participacion"].describe().round(3))

print("\nPuntos de venta con sede menos clara:")
print("Participación < 80%:", (sede_pdv["participacion"] < 0.80).sum())
print("Participación < 70%:", (sede_pdv["participacion"] < 0.70).sum())
print("Participación < 60%:", (sede_pdv["participacion"] < 0.60).sum())

print("\n10 PdV con menor concentración territorial:")
print(
    sede_pdv[
        ["Punto de Venta", "dp", "participacion"]
    ]
    .sort_values("participacion")
    .head(10)
    .round(3)
    .to_string(index=False)
)

# %% SENSIBILIDAD DEL UMBRAL PARA ASIGNAR SEDE A UN PDV

for umbral in [0.70, 0.80, 0.90]:
    pdv_validos = sede_pdv[sede_pdv["participacion"] >= umbral]

    print(f"\nUMBRAL >= {umbral:.0%}")
    print("PdV con sede suficientemente clara:", len(pdv_validos))
    print("Departamentos con presencia inferida:", pdv_validos["dp"].nunique())

    print(
        sorted(
            pdv_validos["dp"].astype(int).unique().tolist()
        )
    )
    
# %%
# Calculamos la penetración y cuántos clientes faltan frente al percentil 75.

# %% Penetración (periodo completo, población 2025)
conteo = cf.groupby(["dp", "generacion"]).size().rename("clientes").reset_index()
pen = (poblacion_generacion(ANIO_POBLACION)
       .merge(conteo, on=["dp", "generacion"], how="left")
       .fillna({"clientes": 0}))
pen["departamento"] = pen["dp"].map(nombre_dp)
pen["por_mil"] = pen["clientes"] / pen["poblacion"] * 1000

nacional = pen.groupby("generacion")[["clientes", "poblacion"]].sum()
nacional["por_mil"] = nacional["clientes"] / nacional["poblacion"] * 1000
pen["indice_vs_nacional"] = pen["por_mil"] / pen["generacion"].map(nacional["por_mil"]) * 100

pen["tiendas"] = pen["dp"].map(tiendas_dp).fillna(0).astype(int)
pen["tipo_oportunidad"] = np.where(pen["tiendas"] > 0, "Activación", "Expansión")

# Brecha frente al P75 de los departamentos CON tienda y base suficiente
p75 = (pen[(pen["clientes"] >= MIN_CLIENTES) & (pen["tiendas"] > 0)]
       .groupby("generacion")["por_mil"].quantile(0.75))
pen["p75_generacion"] = pen["generacion"].map(p75)
pen["clientes_potenciales"] = (
    (pen["p75_generacion"] - pen["por_mil"]).clip(lower=0) * pen["poblacion"] / 1000
).round()
pen["base_suficiente"] = pen["clientes"] >= MIN_CLIENTES

print("\nPENETRACIÓN NACIONAL POR GENERACIÓN (clientes por cada 1.000 habitantes)")
print(nacional.reindex(ORDEN_GEN).round(2))

# %% Tabla departamento × generación
tabla = pen.pivot_table(index="departamento", columns="generacion",
                        values="por_mil").reindex(columns=ORDEN_GEN)
tabla["Total"] = (pen.groupby("departamento")["clientes"].sum()
                  / pen.groupby("departamento")["poblacion"].sum() * 1000)
tabla["Tiendas"] = pen.groupby("departamento")["tiendas"].first()
tabla = tabla.sort_values("Total", ascending=False)
print("\nCLIENTES POR CADA 1.000 HABITANTES — departamento × generación")
print(tabla.round(2).to_string())

# %% DIAGNÓSTICO DEL BENCHMARK P75

benchmark = pen[
    (pen["clientes"] >= MIN_CLIENTES)
    & (pen["tiendas"] > 0)
].copy()

print("DEPARTAMENTOS QUE DEFINEN EL P75 POR GENERACIÓN")
print(
    benchmark.groupby("generacion")
    .agg(
        departamentos=("dp", "nunique"),
        clientes_min=("clientes", "min"),
        clientes_mediana=("clientes", "median"),
        por_mil_min=("por_mil", "min"),
        por_mil_mediana=("por_mil", "median"),
        por_mil_max=("por_mil", "max")
    )
    .reindex(ORDEN_GEN)
    .round(2)
)

print("\nP75 UTILIZADO POR GENERACIÓN")
print(p75.reindex(ORDEN_GEN).round(2))

# %% VALIDAR CONFIABILIDAD DEL BENCHMARK P75

n_deptos_benchmark = (
    benchmark.groupby("generacion")["dp"]
    .nunique()
)

pen["n_deptos_benchmark"] = (
    pen["generacion"]
    .map(n_deptos_benchmark)
    .fillna(0)
    .astype(int)
)

pen["benchmark_confiable"] = (
    pen["n_deptos_benchmark"] >= 10
)

# Si el benchmark no es suficientemente robusto,
# no cuantificamos clientes potenciales.
pen.loc[
    ~pen["benchmark_confiable"],
    "clientes_potenciales"
] = np.nan

print("CONFIABILIDAD DEL BENCHMARK POR GENERACIÓN")
print(
    pen[
        ["generacion", "n_deptos_benchmark", "benchmark_confiable"]
    ]
    .drop_duplicates()
    .set_index("generacion")
    .reindex(ORDEN_GEN)
)

# %% Mayores oportunidades (brecha absoluta), separadas por tipo
oportunidades = (pen[pen["clientes_potenciales"] > 0]
                 .sort_values("clientes_potenciales", ascending=False)
                 [["tipo_oportunidad", "departamento", "generacion", "tiendas", "clientes",
                   "poblacion", "por_mil", "p75_generacion", "indice_vs_nacional",
                   "clientes_potenciales", "base_suficiente"]])
for tipo in ["Activación", "Expansión"]:
    o = oportunidades[oportunidades["tipo_oportunidad"] == tipo]
    print(f"\nTOP 10 — {tipo.upper()} (clientes adicionales si llegara al P75 de su generación)")
    print(o.head(10).round(2).to_string(index=False))
    print(f"Total clientes potenciales ({tipo}): {o['clientes_potenciales'].sum():,.0f}")

# %% CALIDAD DE LAS OPORTUNIDADES IDENTIFICADAS

resumen_calidad = (
    oportunidades
    .groupby(
        ["tipo_oportunidad", "generacion", "base_suficiente"],
        as_index=False
    )
    .agg(
        casos=("departamento", "count"),
        clientes_potenciales=("clientes_potenciales", "sum")
    )
)

print("RESUMEN DE OPORTUNIDADES SEGÚN CALIDAD DE BASE")
print(
    resumen_calidad
    .round(0)
    .to_string(index=False)
)

print("\nTOP EXPANSIÓN CON BASE SUFICIENTE")
print(
    oportunidades[
        (oportunidades["tipo_oportunidad"] == "Expansión")
        & (oportunidades["base_suficiente"])
    ]
    .head(10)
    [
        [
            "departamento",
            "generacion",
            "clientes",
            "poblacion",
            "por_mil",
            "indice_vs_nacional",
            "clientes_potenciales"
        ]
    ]
    .round(2)
    .to_string(index=False)
)

# %%
# Comparación 2024 vs 2025 y guardado.

# %% Cambio en el tiempo: clientes activos por año (2024 vs 2025)
# Un cliente cuenta en un año si compró al menos una vez ese año.
transacciones["anio"] = transacciones["Fecha Pedido"].dt.year
activos = (transacciones[transacciones["anio"].isin([2024, 2025])]
           .drop_duplicates(["Cliente", "anio"])
           .merge(cf[["Cliente", "dp", "generacion"]], on="Cliente"))
evol = []
for anio in (2024, 2025):
    a = activos[activos["anio"] == anio].groupby("generacion").size()
    p = poblacion_generacion(anio).groupby("generacion")["poblacion"].sum()
    evol.append(pd.DataFrame({"anio": anio, "clientes_activos": a,
                              "por_mil": a / p * 1000}))
evol = pd.concat(evol).reset_index().pivot(index="generacion", columns="anio",
                                            values=["clientes_activos", "por_mil"])
evol[("cambio_%", "")] = (evol[("por_mil", 2025)] / evol[("por_mil", 2024)] - 1) * 100
print("\nPENETRACIÓN ANUAL POR GENERACIÓN (clientes activos por cada 1.000 habitantes)")
print(evol.reindex(ORDEN_GEN).round(2).to_string())

# %% Guardar
pen.to_csv(SALIDAS / "penetracion_dep_generacion.csv", index=False, encoding="utf-8-sig")
tabla.round(3).to_csv(SALIDAS / "penetracion_tabla_por_mil.csv", encoding="utf-8-sig")
oportunidades.to_csv(SALIDAS / "penetracion_oportunidades.csv", index=False, encoding="utf-8-sig")
evol.round(3).to_csv(SALIDAS / "penetracion_evolucion_anual.csv", encoding="utf-8-sig")
print(f"\nResultados guardados en {SALIDAS.resolve()}")

# %%
