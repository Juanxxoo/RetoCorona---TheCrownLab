# %% [markdown]
# # Índice de priorización y tabla de acciones — Reto Corona (Retail)
# Une todo lo anterior en una sola respuesta: ¿dónde, con quién y con qué negocio
# debe actuar Corona primero?
#
# Unidad: departamento × generación (consumidores finales).
# Componentes del puntaje (cada uno se lleva a percentil 0–1 dentro de su tipo):
#   1. Brecha de penetración (clientes potenciales, de 07)          peso 0,40
#   2. Tamaño de la población de esa generación                     peso 0,20
#   3. Dinamismo de la construcción (m² licenciados por 1.000 hab) peso 0,20
#   4. Crecimiento de hogares (%)                                   peso 0,10
#   5. Déficit habitacional cualitativo                             peso 0,10
#
# Los pesos se ponen a prueba con 2.000 combinaciones aleatorias.
#
# Acción sugerida:
# según el segmento sobrerrepresentado entre los clientes de esa celda,
# las reglas de afinidad de 06 y el negocio con menor penetración relativa.


# %% Carga

from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd


RAIZ = Path("..")
DATA = RAIZ / "data"
SALIDAS = RAIZ / "resultados"

SALIDAS.mkdir(exist_ok=True)


pen = pd.read_csv(
    SALIDAS / "penetracion_dep_generacion.csv"
)

clientes = pd.read_parquet(
    DATA / "features_clientes_segmentados.parquet"
)

compras = pd.read_parquet(
    DATA / "dataset_limpio.parquet",
    columns=["Cliente", "Negocio"]
)


def normalizar(txt):
    txt = (
        unicodedata
        .normalize("NFKD", str(txt))
        .encode("ascii", "ignore")
        .decode()
    )

    return txt.upper().strip()


# Mismo mapeo que 07
NOMBRE_A_DP = {
    "ANTIOQUIA": 5,
    "ATLANTICO": 8,
    "BOGOTA": 11,
    "BOLIVAR": 13,
    "BOYACA": 15,
    "CALDAS": 17,
    "CAQUETA": 18,
    "CAUCA": 19,
    "CESAR": 20,
    "CORDOBA": 23,
    "CUNDINAMARCA": 25,
    "CHOCO": 27,
    "HUILA": 41,
    "LA GUAJIRA": 44,
    "MAGDALENA": 47,
    "META": 50,
    "NARINO": 52,
    "NORTE SANTANDER": 54,
    "NORTE DE SANTANDER": 54,
    "QUINDIO": 63,
    "RISARALDA": 66,
    "SANTANDER": 68,
    "SUCRE": 70,
    "TOLIMA": 73,
    "VALLE": 76,
    "ARAUCA": 81,
    "CASANARE": 85,
    "PUTUMAYO": 86,
    "SAN ANDRES": 88,
    "SAN ANDRES Y PROVIDENCIA": 88,
    "AMAZONAS": 91,
    "GUAINIA": 94,
    "GUAVIARE": 95,
    "VAUPES": 97,
    "VICHADA": 99,
}


# %% Fuentes externas: lectura y correcciones documentadas
# Los CSV crudos NO se modifican.

def leer_externo(nombre):

    d = pd.read_csv(
        RAIZ / nombre,
        skiprows=1,
        decimal=","
    )

    d = d[
        d.iloc[:, 0].notna()
        & ~d.iloc[:, 0].astype(str).str.startswith(
            ("(No", "Nota")
        )
    ]

    d["dp"] = d.iloc[:, 0].map(
        lambda x: NOMBRE_A_DP[
            normalizar(x)
        ]
    )

    return d


nec = leer_externo(
    "Busqueda fuentes externas corona - Necesidades sostenibles.csv"
)

cre = leer_externo(
    "Busqueda fuentes externas corona - Crecimiento.csv"
)


# Corrección 1:
# déficit cualitativo y cuantitativo están invertidos en el CSV.
nec = nec.rename(
    columns={
        "Déficit habitacional cualitativo (%)":
            "deficit_cuantitativo",

        "Déficit habitacional cuantitativo (%)":
            "deficit_cualitativo",
    }
)


# Corrección 2:
# Huila trae 999 % de cobertura de energía.
col_energia = "Cobertura energía eléctrica (%)"

nec.loc[
    nec[col_energia] > 100,
    col_energia
] = np.nan


terr = (
    nec[
        [
            "dp",
            "deficit_cualitativo",
            "deficit_cuantitativo"
        ]
    ]
    .merge(
        cre[
            [
                "dp",
                "Área licenciada (m²)",
                "Crecimiento de hogares (%)"
            ]
        ],
        on="dp"
    )
    .rename(
        columns={
            "Área licenciada (m²)":
                "area_licenciada_m2",

            "Crecimiento de hogares (%)":
                "crecimiento_hogares"
        }
    )
)


# Se mantiene EXACTAMENTE la metodología que ya tenía el 08.
pob_total = (
    pen
    .groupby("dp")["poblacion"]
    .sum()
)


terr["m2_licenciados_por_mil_hab"] = (
    terr["area_licenciada_m2"]
    / terr["dp"].map(pob_total)
    * 1000
)


terr.to_csv(
    SALIDAS / "base_territorial_corregida.csv",
    index=False,
    encoding="utf-8-sig"
)


# %% Segmento característico de cada celda
# departamento × generación

cf = clientes[
    clientes["Tipo Cliente"]
    == "Consumidor final"
].copy()


cf["dp"] = cf[
    "region_dominante"
].map(
    lambda r:
        NOMBRE_A_DP.get(
            normalizar(r)
        )
)


cf = cf.dropna(
    subset=["dp"]
)


cf["dp"] = cf[
    "dp"
].astype(int)


MIN_CLIENTES_SEG = 200


peso_celda = (
    cf
    .groupby(
        ["dp", "generacion"]
    )["nombre_segmento"]
    .value_counts(
        normalize=True
    )
    .rename("peso")
    .reset_index()
)


peso_nac = (
    cf
    .groupby(
        "generacion"
    )["nombre_segmento"]
    .value_counts(
        normalize=True
    )
    .rename("peso_nac")
    .reset_index()
)


peso_celda = peso_celda.merge(
    peso_nac,
    on=[
        "generacion",
        "nombre_segmento"
    ]
)


peso_celda["sobrerrep"] = (
    peso_celda["peso"]
    / peso_celda["peso_nac"]
)


seg_car = (
    peso_celda
    .sort_values(
        "sobrerrep",
        ascending=False
    )
    .drop_duplicates(
        [
            "dp",
            "generacion"
        ]
    )
    .set_index(
        [
            "dp",
            "generacion"
        ]
    )
    .rename(
        columns={
            "nombre_segmento":
                "segmento_caracteristico",

            "peso":
                "pct_segmento",

            "sobrerrep":
                "sobrerrep_segmento"
        }
    )
    [
        [
            "segmento_caracteristico",
            "pct_segmento",
            "sobrerrep_segmento"
        ]
    ]
)


n_celda = (
    cf
    .groupby(
        [
            "dp",
            "generacion"
        ]
    )
    .size()
)


seg_car.loc[
    n_celda[
        n_celda < MIN_CLIENTES_SEG
    ].index.intersection(
        seg_car.index
    ),
    "segmento_caracteristico"
] = "Base insuficiente"


# %% Negocio con menor penetración relativa
#
# CORRECCIÓN VALIDADA:
# las 4 opciones se construyen explícitamente para TODAS las celdas.
# Si un negocio tiene 0 compradores, debe entrar igualmente a la comparación.

negocios_relevantes = [
    "BAÑOS",
    "COCINAS",
    "PINTURAS E IMPERMEABILIZANTES",
    "REVESTIMIENTOS E INSTALACIÓN"
]


# Un cliente cuenta una sola vez por negocio.
cli_neg = (
    compras
    .drop_duplicates(
        [
            "Cliente",
            "Negocio"
        ]
    )
    .merge(
        cf[
            [
                "Cliente",
                "dp",
                "generacion"
            ]
        ],
        on="Cliente",
        how="inner"
    )
)


# Total de clientes de cada celda.
base_celda = (
    cf
    .groupby(
        [
            "dp",
            "generacion"
        ]
    )
    .size()
    .rename("n")
    .reset_index()
)


# Las 4 opciones para TODAS las celdas.
universo_negocios = (
    base_celda[
        [
            "dp",
            "generacion"
        ]
    ]
    .merge(
        pd.DataFrame(
            {
                "Negocio":
                    negocios_relevantes
            }
        ),
        how="cross"
    )
)


# Compradores realmente observados.
compra_celda = (
    cli_neg[
        cli_neg["Negocio"]
        .isin(
            negocios_relevantes
        )
    ]
    .groupby(
        [
            "dp",
            "generacion",
            "Negocio"
        ],
        observed=True
    )
    .size()
    .rename("compran")
    .reset_index()
)


tasa = (
    universo_negocios
    .merge(
        compra_celda,
        on=[
            "dp",
            "generacion",
            "Negocio"
        ],
        how="left"
    )
    .merge(
        base_celda,
        on=[
            "dp",
            "generacion"
        ],
        how="left"
    )
)


# Negocio no observado = 0 compradores.
tasa["compran"] = (
    tasa["compran"]
    .fillna(0)
    .astype(int)
)


tasa["tasa"] = (
    tasa["compran"]
    / tasa["n"]
)


# Tasa nacional por negocio y generación.
base_nacional = (
    cf
    .groupby(
        "generacion"
    )
    .size()
    .rename("n_nacional")
    .reset_index()
)


compra_nacional = (
    cli_neg[
        cli_neg["Negocio"]
        .isin(
            negocios_relevantes
        )
    ]
    .groupby(
        [
            "generacion",
            "Negocio"
        ],
        observed=True
    )
    .size()
    .rename(
        "compran_nacional"
    )
    .reset_index()
)


nac = (
    pd.DataFrame(
        [
            (g, n)
            for g in
            cf["generacion"]
            .dropna()
            .unique()

            for n in
            negocios_relevantes
        ],
        columns=[
            "generacion",
            "Negocio"
        ]
    )
    .merge(
        compra_nacional,
        on=[
            "generacion",
            "Negocio"
        ],
        how="left"
    )
    .merge(
        base_nacional,
        on="generacion",
        how="left"
    )
)


nac["compran_nacional"] = (
    nac["compran_nacional"]
    .fillna(0)
)


nac["tasa_nacional"] = (
    nac["compran_nacional"]
    / nac["n_nacional"]
)


tasa = tasa.merge(
    nac[
        [
            "generacion",
            "Negocio",
            "tasa_nacional"
        ]
    ],
    on=[
        "generacion",
        "Negocio"
    ],
    how="left"
)


tasa["indice_negocio"] = (
    tasa["tasa"]
    / tasa["tasa_nacional"]
    * 100
)


neg_impulsar = (
    tasa
    .sort_values(
        [
            "dp",
            "generacion",
            "indice_negocio",
            "Negocio"
        ]
    )
    .drop_duplicates(
        [
            "dp",
            "generacion"
        ]
    )
    .set_index(
        [
            "dp",
            "generacion"
        ]
    )
    [
        [
            "Negocio",
            "indice_negocio"
        ]
    ]
    .rename(
        columns={
            "Negocio":
                "negocio_a_impulsar",

            "indice_negocio":
                "indice_negocio_vs_nacional"
        }
    )
)


# Validación:
# todas las celdas deben comparar exactamente 4 negocios.
conteo_negocios = (
    tasa
    .groupby(
        [
            "dp",
            "generacion"
        ]
    )["Negocio"]
    .nunique()
)


print(
    "\nNEGOCIOS EVALUADOS POR CELDA"
)

print(
    conteo_negocios
    .value_counts()
    .sort_index()
)


# %% Tabla base del índice

idx = (
    pen
    .merge(
        terr,
        on="dp",
        how="left"
    )
    .set_index(
        [
            "dp",
            "generacion"
        ]
    )
    .join(
        seg_car
    )
    .join(
        neg_impulsar
    )
    .reset_index()
)


# CORRECCIÓN VALIDADA:
# distinguimos la calidad de la evidencia comportamental.
idx["calidad_base_comercial"] = np.select(
    [
        idx["clientes"] == 0,
        idx["base_suficiente"]
    ],
    [
        "Sin base de clientes",
        "Base suficiente"
    ],
    default="Exploratoria"
)


# Si no hay ningún cliente observado,
# no existe evidencia para escoger segmento ni negocio.
idx.loc[
    idx["clientes"] == 0,
    [
        "segmento_caracteristico",
        "negocio_a_impulsar",
        "indice_negocio_vs_nacional"
    ]
] = np.nan


print(
    "\nCALIDAD DE LA BASE COMERCIAL"
)

print(
    idx[
        "calidad_base_comercial"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nNEGOCIO A IMPULSAR AUSENTE "
    "CON CLIENTES > 0:"
)

print(
    idx[
        (idx["clientes"] > 0)
        & (
            idx[
                "negocio_a_impulsar"
            ].isna()
        )
    ].shape[0]
)


# %% Componentes del índice
#
# SE CONSERVAN LOS MISMOS QUE YA TENÍAMOS.

COMPONENTES = {
    "clientes_potenciales":
        0.40,

    "poblacion":
        0.20,

    "m2_licenciados_por_mil_hab":
        0.20,

    "crecimiento_hogares":
        0.10,

    "deficit_cualitativo":
        0.10,
}


def puntuar(df, pesos):

    p = pd.Series(
        pesos
    ) / sum(
        pesos.values()
    )

    ranks = pd.DataFrame(
        {
            c:
                df[c]
                .rank(
                    pct=True
                )

            for c
            in p.index
        }
    )

    return (
        ranks
        * p
    ).sum(
        axis=1
    ) * 100


# Se puntúa por separado dentro
# de cada tipo de oportunidad.
idx = idx[
    idx[
        "clientes_potenciales"
    ] > 0
].copy()


idx["puntaje"] = (
    idx
    .groupby(
        "tipo_oportunidad",
        group_keys=False
    )
    .apply(
        lambda g:
            puntuar(
                g,
                COMPONENTES
            )
    )
)


# %% Sensibilidad
# 2.000 combinaciones aleatorias de pesos

rng = np.random.default_rng(
    42
)

N_SIM = 2000
TOP = 10


veces_top = pd.Series(
    0,
    index=idx.index,
    dtype=float
)


for _ in range(
    N_SIM
):

    w = dict(
        zip(
            COMPONENTES,
            rng.dirichlet(
                np.ones(
                    len(
                        COMPONENTES
                    )
                )
            )
        )
    )

    for tipo, g in idx.groupby(
        "tipo_oportunidad"
    ):

        top = (
            puntuar(
                g,
                w
            )
            .nlargest(
                TOP
            )
            .index
        )

        veces_top[
            top
        ] += 1


idx[
    "pct_simulaciones_top10"
] = (
    veces_top
    / N_SIM
    * 100
)


# CORRECCIÓN VALIDADA:
# "robusta" ya no significa solamente estabilidad de pesos.
# También debe tener base comercial suficiente.
idx[
    "oportunidad_robusta"
] = (
    (
        idx[
            "pct_simulaciones_top10"
        ] >= 60
    )
    &
    (
        idx[
            "calidad_base_comercial"
        ]
        == "Base suficiente"
    )
)


# %% Acción sugerida
# Se CONSERVAN las reglas que ya teníamos de 06.

REGLA_POR_SEGMENTO = {

    "Consumidores consolidados de alta vinculación":
        "Venta cruzada Cocinas → Baños "
        "(lift 2,8; 43 % de confianza)",

    "Consumidores activos en desarrollo":
        "Servicios de instalación → Baños "
        "(lift 3,8) y Cocinas → Baños",

    "Consumidores recientes de vínculo inicial":
        "Segunda compra: Servicios → Baños "
        "(lift 2,7–3,0; 42 % de confianza)",

    "Consumidores inactivos de vínculo intermedio":
        "Reactivación: Pinturas → Baños "
        "(lift 2,7)",

    "Consumidores inactivos de bajo vínculo":
        "Reactivación con oferta de entrada "
        "(bajo historial: sin regla fuerte)",
}


def accion(r):

    # Sin clientes observados:
    # no se inventa una acción basada en comportamiento.
    if (
        r[
            "calidad_base_comercial"
        ]
        == "Sin base de clientes"
    ):

        return (
            "Validar demanda, canal y portafolio "
            "antes de definir el negocio de entrada"
        )


    neg = str(
        r[
            "negocio_a_impulsar"
        ]
    ).title()


    if (
        r[
            "tipo_oportunidad"
        ]
        == "Expansión"
    ):

        # Solo corregimos el lenguaje:
        # no conocemos tiendas físicas.
        if (
            r[
                "calidad_base_comercial"
            ]
            == "Exploratoria"
        ):

            return (
                f"Señal exploratoria para captar "
                f"{r['generacion']} en territorio "
                f"sin presencia comercial inferida; "
                f"evaluar {neg} como hipótesis de entrada"
            )


        return (
            f"Captar {r['generacion']} en territorio "
            f"sin presencia comercial inferida: "
            f"canal digital/aliados; "
            f"entrada por {neg}"
        )


    regla = REGLA_POR_SEGMENTO.get(
        r[
            "segmento_caracteristico"
        ],
        "Venta cruzada hacia Baños"
    )


    return (
        f"{regla}; "
        f"impulsar {neg}"
    )


idx[
    "accion_sugerida"
] = idx.apply(
    accion,
    axis=1
)


# %% Ranking final

COLS = [
    "tipo_oportunidad",
    "departamento",
    "generacion",
    "tiendas",
    "clientes",
    "por_mil",
    "p75_generacion",
    "clientes_potenciales",
    "m2_licenciados_por_mil_hab",
    "crecimiento_hogares",
    "deficit_cualitativo",
    "calidad_base_comercial",
    "segmento_caracteristico",
    "pct_segmento",
    "sobrerrep_segmento",
    "negocio_a_impulsar",
    "indice_negocio_vs_nacional",
    "puntaje",
    "pct_simulaciones_top10",
    "oportunidad_robusta",
    "accion_sugerida"
]


ranking = (
    idx
    .sort_values(
        [
            "tipo_oportunidad",
            "puntaje"
        ],
        ascending=[
            True,
            False
        ]
    )
    [
        COLS
    ]
)


# %% Mostrar resultados

pd.set_option(
    "display.width",
    250
)


for tipo in [
    "Activación",
    "Expansión"
]:

    r = (
        ranking[
            ranking[
                "tipo_oportunidad"
            ] == tipo
        ]
        .head(10)
    )

    print(
        f"\nTOP 10 — {tipo.upper()}"
    )

    print(
        r[
            [
                "departamento",
                "generacion",
                "clientes_potenciales",
                "puntaje",
                "pct_simulaciones_top10",
                "calidad_base_comercial",
                "segmento_caracteristico",
                "negocio_a_impulsar"
            ]
        ]
        .round(1)
        .to_string(
            index=False
        )
    )


# %% Oportunidades robustas

print(
    "\nOPORTUNIDADES ROBUSTAS "
    "(sensibilidad de pesos + base suficiente)"
)


print(
    ranking[
        ranking[
            "oportunidad_robusta"
        ]
    ]
    [
        [
            "tipo_oportunidad",
            "departamento",
            "generacion",
            "pct_simulaciones_top10",
            "accion_sugerida"
        ]
    ]
    .round(1)
    .to_string(
        index=False
    )
)


# %% Diagnósticos finales

print(
    "\nVALIDACIÓN — negocios comparados por celda"
)

print(
    tasa
    .groupby(
        [
            "dp",
            "generacion"
        ]
    )["Negocio"]
    .nunique()
    .value_counts()
    .sort_index()
)


print(
    "\nVALIDACIÓN — celdas donde "
    "el negocio elegido tiene 0 compradores"
)


seleccion = (
    tasa
    .merge(
        neg_impulsar.reset_index(),
        left_on=[
            "dp",
            "generacion",
            "Negocio"
        ],
        right_on=[
            "dp",
            "generacion",
            "negocio_a_impulsar"
        ],
        how="inner"
    )
)


print(
    (
        seleccion[
            "compran"
        ] == 0
    )
    .value_counts()
)


print(
    "\nVALIDACIÓN — calidad de base "
    "dentro del ranking"
)

print(
    ranking[
        "calidad_base_comercial"
    ]
    .value_counts()
    .to_string()
)


# %% Guardar

ranking.to_csv(
    SALIDAS
    / "priorizacion_oportunidades.csv",
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\nGuardado en "
    f"{(SALIDAS / 'priorizacion_oportunidades.csv').resolve()}"
)

# %%