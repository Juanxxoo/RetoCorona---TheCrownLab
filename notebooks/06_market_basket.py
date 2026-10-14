# %%
# MARKET BASKET / ANÁLISIS DE AFINIDADES
# Cargamos la base transaccional limpia y agregamos
# el segmento asignado a cada cliente.

import pandas as pd
import numpy as np

datos = pd.read_parquet("../data/dataset_limpio.parquet")

segmentos = pd.read_parquet(
    "../data/features_clientes_segmentados.parquet"
)[
    ["Cliente", "segmento_k5", "nombre_segmento"]
]

datos_mb = datos.merge(
    segmentos,
    on="Cliente",
    how="left"
)

print("Dimensiones:", datos_mb.shape)

print("\nFilas sin segmento:")
print(datos_mb["nombre_segmento"].isna().sum())

print("\nEjemplo:")
print(
    datos_mb[
        ["ID Pedido", "Cliente", "Categoria", "Producto", "nombre_segmento"]
    ].head()
)

# %%
# DIAGNÓSTICO PREVIO AL MARKET BASKET
# Antes de construir las canastas revisamos líneas con valores
# positivos, cero o negativos. No filtramos todavía.

print("CANTIDAD")
print("Negativas:", (datos_mb["Cantidad"] < 0).sum())
print("Cero:", (datos_mb["Cantidad"] == 0).sum())
print("Positivas:", (datos_mb["Cantidad"] > 0).sum())

print("\nVALOR")
print("Negativos:", (datos_mb["Valor"] < 0).sum())
print("Cero:", (datos_mb["Valor"] == 0).sum())
print("Positivos:", (datos_mb["Valor"] > 0).sum())

print("\nPRECIO")
print("Negativos:", (datos_mb["Precio"] < 0).sum())
print("Cero:", (datos_mb["Precio"] == 0).sum())
print("Positivos:", (datos_mb["Precio"] > 0).sum())


# %%
# REVISAMOS SI LOS CEROS Y FALTANTES COINCIDEN ENTRE VARIABLES

diagnostico_mb = pd.DataFrame({
    "cantidad_faltante": datos_mb["Cantidad"].isna(),
    "cantidad_cero": datos_mb["Cantidad"].eq(0),
    "precio_cero": datos_mb["Precio"].eq(0),
    "valor_cero": datos_mb["Valor"].eq(0)
})

print(diagnostico_mb.value_counts().head(15))

# %%
# CONSTRUCCIÓN DE CANASTAS A NIVEL CATEGORÍA
# Conservamos únicamente líneas con Valor positivo.
# Si una categoría aparece varias veces dentro del mismo pedido,
# la contamos una sola vez para el análisis de afinidades.

compras_mb = datos_mb[
    (datos_mb["Valor"] > 0)
    & datos_mb["ID Pedido"].notna()
    & datos_mb["Categoria"].notna()
].copy()

pedido_categoria = (
    compras_mb[
        ["ID Pedido", "Categoria"]
    ]
    .drop_duplicates()
)

print("Filas válidas para Market Basket:", len(compras_mb))
print("Pedidos únicos:", compras_mb["ID Pedido"].nunique())
print("Combinaciones únicas Pedido-Categoría:", len(pedido_categoria))

# Número de categorías distintas por pedido
categorias_por_pedido = (
    pedido_categoria
    .groupby("ID Pedido")["Categoria"]
    .nunique()
)

print("\nCategorías distintas por pedido:")
print(categorias_por_pedido.describe())

print(
    "\n% de pedidos con más de una categoría:",
    (categorias_por_pedido > 1).mean() * 100
)

# %%
# SOPORTE INDIVIDUAL DE CADA CATEGORÍA
# Support = porcentaje de pedidos en los que aparece una categoría.

total_pedidos = compras_mb["ID Pedido"].nunique()

soporte_categorias = (
    pedido_categoria
    .groupby("Categoria")["ID Pedido"]
    .nunique()
    .sort_values(ascending=False)
    .to_frame("pedidos")
)

soporte_categorias["support"] = (
    soporte_categorias["pedidos"] / total_pedidos
)

soporte_categorias["support_pct"] = (
    soporte_categorias["support"] * 100
)

print("SOPORTE DE LAS CATEGORÍAS")
print(soporte_categorias.round(4))

# %%
# MATRIZ DE CANASTAS: PEDIDO × CATEGORÍA
# 1 = la categoría apareció en el pedido
# 0 = la categoría no apareció

canastas_categorias = pd.crosstab(
    pedido_categoria["ID Pedido"],
    pedido_categoria["Categoria"]
).astype(bool)

print("Dimensiones de la matriz:", canastas_categorias.shape)

print("\nPrimeras filas:")
print(canastas_categorias.head())

print("\nNúmero de categorías:", canastas_categorias.shape[1])
print("Número de pedidos:", canastas_categorias.shape[0])

# %%
# NIVEL 1: AFINIDADES ENTRE NEGOCIOS
# Revisamos cuántos pedidos incluyen compras en más de un negocio de Corona.

compras_negocio = datos_mb[
    (datos_mb["Valor"] > 0)
    & datos_mb["ID Pedido"].notna()
    & datos_mb["Negocio"].notna()
].copy()

# Cada negocio cuenta una sola vez dentro de un mismo pedido
pedido_negocio = (
    compras_negocio[
        ["ID Pedido", "Negocio"]
    ]
    .drop_duplicates()
)

negocios_por_pedido = (
    pedido_negocio
    .groupby("ID Pedido")["Negocio"]
    .nunique()
)

print("Negocios distintos por pedido:")
print(negocios_por_pedido.describe())

print(
    "\n% de pedidos con más de un negocio:",
    (negocios_por_pedido > 1).mean() * 100
)

print("\nPedidos por negocio:")
print(
    pedido_negocio["Negocio"]
    .value_counts()
)

# %%
# MATRIZ DE CANASTAS: PEDIDO × NEGOCIO
# True = el negocio apareció en el pedido
# False = el negocio no apareció

canastas_negocio = pd.crosstab(
    pedido_negocio["ID Pedido"],
    pedido_negocio["Negocio"]
).astype(bool)

print("Dimensiones:", canastas_negocio.shape)

print("\nPrimeras filas:")
print(canastas_negocio.head())

print("\nNegocios:")
print(canastas_negocio.columns.tolist())

# %%
# AFINIDADES ENTRE NEGOCIOS
# Calculamos todas las combinaciones posibles entre los 5 negocios.
# No aplicamos todavía ningún umbral para no perder asociaciones interesantes.

from itertools import combinations
import pandas as pd

resultados_afinidad = []

total_pedidos = len(canastas_negocio)

for negocio_a, negocio_b in combinations(canastas_negocio.columns, 2):

    # Pedidos donde aparece cada negocio
    pedidos_a = canastas_negocio[negocio_a].sum()
    pedidos_b = canastas_negocio[negocio_b].sum()

    # Pedidos donde aparecen ambos
    pedidos_juntos = (
        canastas_negocio[negocio_a]
        & canastas_negocio[negocio_b]
    ).sum()

    # Support conjunto
    support = pedidos_juntos / total_pedidos

    # Confidence en ambas direcciones
    confianza_a_b = pedidos_juntos / pedidos_a
    confianza_b_a = pedidos_juntos / pedidos_b

    # Lift
    soporte_a = pedidos_a / total_pedidos
    soporte_b = pedidos_b / total_pedidos

    lift = support / (soporte_a * soporte_b)

    resultados_afinidad.append({
        "negocio_A": negocio_a,
        "negocio_B": negocio_b,
        "pedidos_juntos": pedidos_juntos,
        "support_pct": support * 100,
        "confidence_A_B_pct": confianza_a_b * 100,
        "confidence_B_A_pct": confianza_b_a * 100,
        "lift": lift
    })

afinidades_negocio = (
    pd.DataFrame(resultados_afinidad)
    .sort_values("lift", ascending=False)
)

print("AFINIDADES ENTRE NEGOCIOS")
print(afinidades_negocio.round(3).to_string(index=False))

# %%
# VERIFICAR RELACIÓN CATEGORÍA -> NEGOCIO
# Antes de buscar afinidades entre categorías de distintos negocios,
# comprobamos si cada categoría pertenece a un único negocio.

categoria_negocio = (
    compras_mb
    .groupby("Categoria")["Negocio"]
    .nunique()
    .sort_values(ascending=False)
)

print("Número de negocios asociados a cada categoría:")
print(categoria_negocio)

print(
    "\nCategorías asociadas a más de un negocio:",
    (categoria_negocio > 1).sum()
)

# %%
# MAPEO CATEGORÍA -> NEGOCIO
# Como verificamos que cada categoría pertenece a un único negocio,
# construimos una tabla de referencia para interpretar las afinidades.

mapa_categoria_negocio = (
    compras_mb[
        ["Categoria", "Negocio"]
    ]
    .dropna()
    .drop_duplicates()
    .sort_values(["Negocio", "Categoria"])
    .reset_index(drop=True)
)

print("MAPEO DE CATEGORÍAS A NEGOCIOS")
print(mapa_categoria_negocio.to_string(index=False))

# %%
# AFINIDADES ENTRE CATEGORÍAS DE NEGOCIOS DIFERENTES
# Buscamos asociaciones específicas que crucen negocios,
# evitando relaciones dentro del mismo negocio.

from itertools import combinations

# Diccionario: categoría -> negocio
dic_categoria_negocio = dict(
    zip(
        mapa_categoria_negocio["Categoria"],
        mapa_categoria_negocio["Negocio"]
    )
)

resultados_categorias = []

total_pedidos = len(canastas_categorias)

for cat_a, cat_b in combinations(canastas_categorias.columns, 2):

    negocio_a = dic_categoria_negocio[cat_a]
    negocio_b = dic_categoria_negocio[cat_b]

    # Solo analizamos categorías pertenecientes a negocios distintos
    if negocio_a == negocio_b:
        continue

    pedidos_a = canastas_categorias[cat_a].sum()
    pedidos_b = canastas_categorias[cat_b].sum()

    pedidos_juntos = (
        canastas_categorias[cat_a]
        & canastas_categorias[cat_b]
    ).sum()

    support_a = pedidos_a / total_pedidos
    support_b = pedidos_b / total_pedidos
    support_conjunto = pedidos_juntos / total_pedidos

    confianza_a_b = pedidos_juntos / pedidos_a
    confianza_b_a = pedidos_juntos / pedidos_b

    lift = (
        support_conjunto / (support_a * support_b)
        if support_a > 0 and support_b > 0
        else 0
    )

    resultados_categorias.append({
        "categoria_A": cat_a,
        "negocio_A": negocio_a,
        "categoria_B": cat_b,
        "negocio_B": negocio_b,
        "pedidos_juntos": pedidos_juntos,
        "support_pct": support_conjunto * 100,
        "confidence_A_B_pct": confianza_a_b * 100,
        "confidence_B_A_pct": confianza_b_a * 100,
        "lift": lift
    })

afinidades_categorias = pd.DataFrame(resultados_categorias)

print("Número de pares entre negocios distintos:", len(afinidades_categorias))

print("\nDISTRIBUCIÓN DEL NÚMERO DE PEDIDOS CONJUNTOS")
print(
    afinidades_categorias["pedidos_juntos"]
    .describe()
    .round(2)
)

print("\nPARES CON MAYOR LIFT")
print(
    afinidades_categorias
    .sort_values("lift", ascending=False)
    .head(20)
    .round(3)
    .to_string(index=False)
)

# %%
# AFINIDADES CON VOLUMEN SUFICIENTE
# Filtramos asociaciones positivas y con un número razonable
# de pedidos conjuntos para evitar lifts extremos por casos muy raros.

afinidades_relevantes = (
    afinidades_categorias[
        (afinidades_categorias["pedidos_juntos"] >= 500)
        & (afinidades_categorias["lift"] > 1)
    ]
    .sort_values(
        ["lift", "pedidos_juntos"],
        ascending=[False, False]
    )
)

print("AFINIDADES RELEVANTES ENTRE CATEGORÍAS")
print(
    afinidades_relevantes
    .round(3)
    .to_string(index=False)
)

# %%
# PEDIDOS POR SEGMENTO
# Antes de comparar afinidades entre segmentos,
# verificamos cuántos pedidos aporta cada grupo.

pedidos_segmento = (
    compras_mb[
        ["ID Pedido", "nombre_segmento"]
    ]
    .drop_duplicates()
    ["nombre_segmento"]
    .value_counts()
)

pedidos_segmento_pct = (
    pedidos_segmento / pedidos_segmento.sum() * 100
)

resumen_pedidos_segmento = pd.DataFrame({
    "pedidos": pedidos_segmento,
    "porcentaje": pedidos_segmento_pct
})

print("PEDIDOS POR SEGMENTO")
print(resumen_pedidos_segmento.round(2))

# %%
# AFINIDADES ENTRE CATEGORÍAS DENTRO DE CADA SEGMENTO
# Calculamos support, confidence y lift por separado para cada segmento.
# Así podremos detectar asociaciones que sean especialmente fuertes
# en determinados tipos de clientes.

resultados_segmentos = []

for segmento in compras_mb["nombre_segmento"].unique():

    # Pedidos correspondientes únicamente a este segmento
    datos_segmento = compras_mb[
        compras_mb["nombre_segmento"] == segmento
    ]

    pedido_categoria_segmento = (
        datos_segmento[
            ["ID Pedido", "Categoria"]
        ]
        .drop_duplicates()
    )

    # Matriz pedido × categoría del segmento
    canasta_segmento = pd.crosstab(
        pedido_categoria_segmento["ID Pedido"],
        pedido_categoria_segmento["Categoria"]
    ).astype(bool)

    # Aseguramos que estén las mismas 28 categorías
    canasta_segmento = canasta_segmento.reindex(
        columns=canastas_categorias.columns,
        fill_value=False
    )

    total_pedidos_segmento = len(canasta_segmento)

    for cat_a, cat_b in combinations(canasta_segmento.columns, 2):

        negocio_a = dic_categoria_negocio[cat_a]
        negocio_b = dic_categoria_negocio[cat_b]

        # Solo cruces entre negocios diferentes
        if negocio_a == negocio_b:
            continue

        pedidos_a = canasta_segmento[cat_a].sum()
        pedidos_b = canasta_segmento[cat_b].sum()

        pedidos_juntos = (
            canasta_segmento[cat_a]
            & canasta_segmento[cat_b]
        ).sum()

        support_a = pedidos_a / total_pedidos_segmento
        support_b = pedidos_b / total_pedidos_segmento
        support_conjunto = pedidos_juntos / total_pedidos_segmento

        confidence_a_b = (
            pedidos_juntos / pedidos_a
            if pedidos_a > 0 else 0
        )

        confidence_b_a = (
            pedidos_juntos / pedidos_b
            if pedidos_b > 0 else 0
        )

        lift = (
            support_conjunto / (support_a * support_b)
            if support_a > 0 and support_b > 0
            else 0
        )

        resultados_segmentos.append({
            "segmento": segmento,
            "categoria_A": cat_a,
            "negocio_A": negocio_a,
            "categoria_B": cat_b,
            "negocio_B": negocio_b,
            "pedidos_juntos": pedidos_juntos,
            "support_pct": support_conjunto * 100,
            "confidence_A_B_pct": confidence_a_b * 100,
            "confidence_B_A_pct": confidence_b_a * 100,
            "lift": lift
        })

afinidades_por_segmento = pd.DataFrame(resultados_segmentos)

print("Número total de asociaciones calculadas:")
print(len(afinidades_por_segmento))

print("\nDistribución de pedidos conjuntos por segmento:")
print(
    afinidades_por_segmento
    .groupby("segmento")["pedidos_juntos"]
    .describe()
    .round(2)
)

# %%
# REGLAS RELEVANTES POR SEGMENTO
# Usamos un criterio relativo:
# support >= 0.1% dentro del propio segmento
# y lift > 1 para conservar asociaciones positivas.

reglas_relevantes_segmento = afinidades_por_segmento[
    (afinidades_por_segmento["support_pct"] >= 0.1)
    & (afinidades_por_segmento["lift"] > 1)
].copy()

print("NÚMERO DE REGLAS RELEVANTES POR SEGMENTO")
print(
    reglas_relevantes_segmento
    .groupby("segmento")
    .size()
    .sort_values(ascending=False)
)

print("\nTOP 5 REGLAS POR SEGMENTO SEGÚN LIFT")

for segmento in reglas_relevantes_segmento["segmento"].unique():

    print("\n", segmento.upper())

    top = (
        reglas_relevantes_segmento[
            reglas_relevantes_segmento["segmento"] == segmento
        ]
        .sort_values(
            ["lift", "pedidos_juntos"],
            ascending=[False, False]
        )
        .head(5)
    )

    print(
        top[
            [
                "categoria_A",
                "negocio_A",
                "categoria_B",
                "negocio_B",
                "pedidos_juntos",
                "support_pct",
                "confidence_A_B_pct",
                "confidence_B_A_pct",
                "lift"
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

# %%
# COMPARACIÓN DE REGLAS CLAVE ENTRE SEGMENTOS
# Revisamos las mismas asociaciones en todos los segmentos
# para identificar dónde tienen mayor potencial.

reglas_clave = [
    ("CAT_000008", "CAT_000027"),  # Baños - Cocinas
    ("CAT_000015", "CAT_000027"),  # Baños - Cocinas
    ("CAT_000010", "CAT_000020"),  # Pinturas - Servicios
    ("CAT_000015", "CAT_000020")   # Baños - Servicios
]

comparacion_reglas = afinidades_por_segmento[
    afinidades_por_segmento.apply(
        lambda fila: (
            (fila["categoria_A"], fila["categoria_B"]) in reglas_clave
            or
            (fila["categoria_B"], fila["categoria_A"]) in reglas_clave
        ),
        axis=1
    )
].copy()

print(
    comparacion_reglas[
        [
            "segmento",
            "categoria_A",
            "categoria_B",
            "pedidos_juntos",
            "support_pct",
            "confidence_A_B_pct",
            "confidence_B_A_pct",
            "lift"
        ]
    ]
    .sort_values(["categoria_A", "categoria_B", "segmento"])
    .round(3)
    .to_string(index=False)
)

# %%
# REGLAS DIRECCIONALES POR SEGMENTO
# Convertimos cada pareja A-B en dos posibles recomendaciones:
# A -> B y B -> A.
#
# Esto es importante porque la confidence cambia según la dirección.

reglas_direccionales = []

for _, fila in reglas_relevantes_segmento.iterrows():

    # Regla A -> B
    reglas_direccionales.append({
        "segmento": fila["segmento"],
        "antecedente": fila["categoria_A"],
        "negocio_antecedente": fila["negocio_A"],
        "consecuente": fila["categoria_B"],
        "negocio_consecuente": fila["negocio_B"],
        "pedidos_juntos": fila["pedidos_juntos"],
        "support_pct": fila["support_pct"],
        "confidence_pct": fila["confidence_A_B_pct"],
        "lift": fila["lift"]
    })

    # Regla B -> A
    reglas_direccionales.append({
        "segmento": fila["segmento"],
        "antecedente": fila["categoria_B"],
        "negocio_antecedente": fila["negocio_B"],
        "consecuente": fila["categoria_A"],
        "negocio_consecuente": fila["negocio_A"],
        "pedidos_juntos": fila["pedidos_juntos"],
        "support_pct": fila["support_pct"],
        "confidence_pct": fila["confidence_B_A_pct"],
        "lift": fila["lift"]
    })

reglas_direccionales = pd.DataFrame(reglas_direccionales)

print("TOP 5 REGLAS DIRECCIONALES POR SEGMENTO")

for segmento in reglas_direccionales["segmento"].unique():

    print("\n", segmento.upper())

    top = (
        reglas_direccionales[
            reglas_direccionales["segmento"] == segmento
        ]
        .sort_values(
            ["confidence_pct", "lift", "pedidos_juntos"],
            ascending=False
        )
        .head(5)
    )

    print(
        top[
            [
                "antecedente",
                "negocio_antecedente",
                "consecuente",
                "negocio_consecuente",
                "pedidos_juntos",
                "support_pct",
                "confidence_pct",
                "lift"
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

# %%
# REGLAS DIRECCIONALES CON MAYOR POTENCIAL COMERCIAL
# Exigimos simultáneamente:
# - lift >= 1.5: asociación claramente superior a la esperada
# - confidence >= 10%: probabilidad útil en la dirección de recomendación
# - support >= 0.1%: presencia mínima dentro del segmento

reglas_candidatas = reglas_direccionales[
    (reglas_direccionales["lift"] >= 1.5)
    & (reglas_direccionales["confidence_pct"] >= 10)
    & (reglas_direccionales["support_pct"] >= 0.1)
].copy()

print("REGLAS CANDIDATAS POR SEGMENTO")

for segmento in reglas_candidatas["segmento"].unique():

    print("\n", segmento.upper())

    top = (
        reglas_candidatas[
            reglas_candidatas["segmento"] == segmento
        ]
        .sort_values(
            ["confidence_pct", "lift", "pedidos_juntos"],
            ascending=False
        )
        .head(10)
    )

    print(
        top[
            [
                "antecedente",
                "negocio_antecedente",
                "consecuente",
                "negocio_consecuente",
                "pedidos_juntos",
                "support_pct",
                "confidence_pct",
                "lift"
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

# %%
# TOP 3 REGLAS CANDIDATAS POR SEGMENTO
# Resumen final para interpretación y posterior uso en el prototipo.

top_reglas_segmento = (
    reglas_candidatas
    .sort_values(
        ["segmento", "confidence_pct", "lift", "pedidos_juntos"],
        ascending=[True, False, False, False]
    )
    .groupby("segmento")
    .head(3)
    .reset_index(drop=True)
)

print("TOP 3 REGLAS POR SEGMENTO")
print(
    top_reglas_segmento[
        [
            "segmento",
            "antecedente",
            "negocio_antecedente",
            "consecuente",
            "negocio_consecuente",
            "pedidos_juntos",
            "support_pct",
            "confidence_pct",
            "lift"
        ]
    ]
    .round(3)
    .to_string(index=False)
)
# %%
