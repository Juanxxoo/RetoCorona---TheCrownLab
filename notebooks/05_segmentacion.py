# %%
# SEGMENTACIÓN 1: COMPORTAMIENTO
# Hacemos la segmentación por separado para consumidores finales y empresas,
# porque el análisis de sensibilidad mostró que las empresas no seguían tan bien
# la misma estructura de los consumidores.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.decomposition import PCA


# %%
# CARGA DE DATOS Y VARIABLES DEL CLUSTERING

features = pd.read_parquet("../data/features_clientes.parquet")

variables_cluster = [
    "recencia_dias",
    "frecuencia",
    "valor_total",
    "n_categorias_distintas",
]

print("Tamaño de la base de clientes:", features.shape)
print("\nTipo de cliente:")
print(features["Tipo Cliente"].value_counts(dropna=False))


# %%
# FUNCIÓN DE PREPARACIÓN
# Frecuencia y valor_total pasan por log1p y después escalamos las 4 variables.
# El escalamiento se calcula por separado para consumidores y empresas.

def preparar_poblacion(df_tipo):
    X = df_tipo[variables_cluster].copy()

    X_transformado = X.copy()
    X_transformado["frecuencia"] = np.log1p(X_transformado["frecuencia"])
    X_transformado["valor_total"] = np.log1p(X_transformado["valor_total"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformado)

    return X, X_transformado, X_scaled, scaler


# Guardamos todo en un diccionario para no repetir el código dos veces.
poblaciones = {}

for tipo in ["Consumidor final", "Empresa"]:
    df_tipo = features[features["Tipo Cliente"] == tipo].copy()

    X, X_transformado, X_scaled, scaler = preparar_poblacion(df_tipo)

    poblaciones[tipo] = {
        "datos": df_tipo,
        "X": X,
        "X_transformado": X_transformado,
        "X_scaled": X_scaled,
        "scaler": scaler,
    }

    print(f"\n{tipo.upper()}")
    print("Clientes:", len(df_tipo))
    print(X.describe())


# %%
# EVALUACIÓN DEL NÚMERO DE CLUSTERS
# Probamos K de 2 a 7 para cada tipo de cliente.
# Silhouette: entre más alto, mejor separación.
# Davies-Bouldin: entre más bajo, mejor.

def evaluar_k(X_scaled, tipo):
    resultados = []

    for k in range(2, 8):
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        etiquetas = modelo.fit_predict(X_scaled)

        tamano_muestra = min(10000, len(X_scaled))

        silhouette = silhouette_score(
            X_scaled,
            etiquetas,
            sample_size=tamano_muestra,
            random_state=42,
        )

        davies = davies_bouldin_score(X_scaled, etiquetas)

        resultados.append(
            {
                "Tipo Cliente": tipo,
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": davies,
            }
        )

    return pd.DataFrame(resultados)


resultados_k = []

for tipo, info in poblaciones.items():
    resultados_tipo = evaluar_k(info["X_scaled"], tipo)
    resultados_k.append(resultados_tipo)

resultados_k = pd.concat(resultados_k, ignore_index=True)

print("RESULTADOS K POR TIPO DE CLIENTE")
print(resultados_k.round(4).to_string(index=False))


# %%
# GRÁFICAS DE SILHOUETTE
# Una gráfica para cada población para no mezclar escalas ni interpretaciones.

for tipo in ["Consumidor final", "Empresa"]:
    tabla = resultados_k[resultados_k["Tipo Cliente"] == tipo]

    plt.figure(figsize=(8, 5))
    plt.plot(tabla["k"], tabla["silhouette"], marker="o")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score - {tipo}")
    plt.xticks(tabla["k"])
    plt.grid(alpha=0.3)
    plt.show()


# %%
# PERFILES PARA CADA K
# Las métricas nos dicen qué tan separados quedan los grupos,
# pero también necesitamos ver si los perfiles tienen sentido.
# Por eso mostramos las medianas originales para K=2,...,7.

def perfiles_por_k(tipo, info):
    perfiles = []
    X_scaled = info["X_scaled"]
    df_tipo = info["datos"]

    for k in range(2, 8):
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        etiquetas = modelo.fit_predict(X_scaled)

        temporal = df_tipo.copy()
        temporal["cluster_prueba"] = etiquetas

        perfil = (
            temporal.groupby("cluster_prueba")
            .agg(
                clientes=("Cliente", "nunique"),
                recencia_mediana=("recencia_dias", "median"),
                frecuencia_mediana=("frecuencia", "median"),
                valor_mediano=("valor_total", "median"),
                categorias_medianas=("n_categorias_distintas", "median"),
                negocios_medianos=("n_negocios_distintos", "median"),
            )
            .reset_index()
        )

        perfil["porcentaje"] = perfil["clientes"] / perfil["clientes"].sum() * 100
        perfil["Tipo Cliente"] = tipo
        perfil["k"] = k

        perfiles.append(perfil)

    return pd.concat(perfiles, ignore_index=True)


perfiles_candidatos = []

for tipo, info in poblaciones.items():
    perfiles_candidatos.append(perfiles_por_k(tipo, info))

perfiles_candidatos = pd.concat(perfiles_candidatos, ignore_index=True)

for tipo in ["Consumidor final", "Empresa"]:
    print(f"\n\nPERFILES CANDIDATOS - {tipo.upper()}")

    tabla_tipo = perfiles_candidatos[
        perfiles_candidatos["Tipo Cliente"] == tipo
    ]

    for k in range(2, 8):
        print(f"\nK = {k}")
        print(
            tabla_tipo[tabla_tipo["k"] == k][
                [
                    "cluster_prueba",
                    "clientes",
                    "porcentaje",
                    "recencia_mediana",
                    "frecuencia_mediana",
                    "valor_mediano",
                    "categorias_medianas",
                    "negocios_medianos",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )


# %%
# K DE TRABAJO PARA CADA POBLACIÓN
# Dejamos los valores elegidos en un solo lugar para poder cambiarlos fácilmente si fuera necesario.
# Consumidores: usamos K=5 porque permite distinguir perfiles comerciales más granulares.
# Empresas: usamos K=4 porque mantiene una estructura interpretable y ofrece una buena alternativa
# entre separación matemática y utilidad comercial.

K_FINAL = {
    "Consumidor final": 5,
    "Empresa": 4,
}


# %%
# ENTRENAMIENTO FINAL POR TIPO DE CLIENTE
# Cada población tiene su propio scaler, sus propios centroides y su propio K.

features_segmentadas = features.copy()
features_segmentadas["segmento_tipo"] = pd.Series(pd.NA, index=features_segmentadas.index, dtype="Int64")
features_segmentadas["segmento_id"] = pd.NA

modelos_finales = {}

for tipo, info in poblaciones.items():
    k = K_FINAL[tipo]
    X_scaled = info["X_scaled"]
    indices = info["datos"].index

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    etiquetas = modelo.fit_predict(X_scaled)

    features_segmentadas.loc[indices, "segmento_tipo"] = etiquetas

    prefijo = "CF" if tipo == "Consumidor final" else "EMP"
    ids_segmento = [f"{prefijo}_{etiqueta}" for etiqueta in etiquetas]
    features_segmentadas.loc[indices, "segmento_id"] = ids_segmento

    modelos_finales[tipo] = {
        "modelo": modelo,
        "etiquetas": etiquetas,
        "X_scaled": X_scaled,
        "indices": indices,
    }


print("\nClientes por segmento final:")
print(features_segmentadas["segmento_id"].value_counts(dropna=False))


# %%
# PERFIL BÁSICO DE LOS SEGMENTOS FINALES
# Usamos los valores originales para que la interpretación sea directa.

resumen_segmentos = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby(["Tipo Cliente", "segmento_id"])
    .agg(
        clientes=("Cliente", "nunique"),
        recencia_mediana=("recencia_dias", "median"),
        frecuencia_mediana=("frecuencia", "median"),
        valor_mediano=("valor_total", "median"),
        categorias_medianas=("n_categorias_distintas", "median"),
        negocios_medianos=("n_negocios_distintos", "median"),
        antiguedad_mediana=("antiguedad_dias", "median"),
    )
    .reset_index()
)

resumen_segmentos["porcentaje_dentro_tipo"] = (
    resumen_segmentos.groupby("Tipo Cliente")["clientes"]
    .transform(lambda x: x / x.sum() * 100)
)

print("RESUMEN DE LOS SEGMENTOS FINALES")
print(resumen_segmentos.round(2).to_string(index=False))


# %%
# VALIDACIÓN 1: ESTABILIDAD
# Repetimos cada modelo con varias semillas para revisar si vuelve a encontrar
# la misma estructura.

semillas = [0, 1, 7, 21, 100]
resultados_estabilidad = []

for tipo, info in modelos_finales.items():
    k = K_FINAL[tipo]
    X_scaled = info["X_scaled"]
    etiquetas_referencia = info["etiquetas"]

    for semilla in semillas:
        modelo_prueba = KMeans(
            n_clusters=k,
            random_state=semilla,
            n_init=10,
        )

        nuevas_etiquetas = modelo_prueba.fit_predict(X_scaled)

        ari = adjusted_rand_score(
            etiquetas_referencia,
            nuevas_etiquetas,
        )

        resultados_estabilidad.append(
            {
                "Tipo Cliente": tipo,
                "k": k,
                "random_state": semilla,
                "ARI_vs_modelo_original": ari,
            }
        )

resultados_estabilidad = pd.DataFrame(resultados_estabilidad)

print("ESTABILIDAD DE LOS MODELOS FINALES")
print(resultados_estabilidad.round(4).to_string(index=False))


# %%
# VALIDACIÓN 2: SILHOUETTE POR SEGMENTO
# Calculamos el Silhouette individual sobre una muestra de cada población
# para ver qué grupos quedan más claros y cuáles se solapan más.

resumen_silhouette_final = []
rng = np.random.default_rng(42)

for tipo, info in modelos_finales.items():
    X_scaled = info["X_scaled"]
    etiquetas = info["etiquetas"]

    tamano_muestra = min(10000, len(X_scaled))
    indices_muestra = rng.choice(
        len(X_scaled),
        size=tamano_muestra,
        replace=False,
    )

    X_muestra = X_scaled[indices_muestra]
    etiquetas_muestra = etiquetas[indices_muestra]

    silhouette_individual = silhouette_samples(
        X_muestra,
        etiquetas_muestra,
    )

    temporal = pd.DataFrame(
        {
            "segmento_tipo": etiquetas_muestra,
            "silhouette": silhouette_individual,
        }
    )

    resumen = (
        temporal.groupby("segmento_tipo")["silhouette"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )

    resumen["Tipo Cliente"] = tipo
    resumen_silhouette_final.append(resumen)

resumen_silhouette_final = pd.concat(
    resumen_silhouette_final,
    ignore_index=True,
)

print("SILHOUETTE POR SEGMENTO FINAL")
print(resumen_silhouette_final.round(4).to_string(index=False))


# %%
# PCA PARA VISUALIZAR
# Los modelos se entrenaron con las 4 variables originales ya escaladas.
# PCA únicamente nos permite ver una aproximación en dos dimensiones.

for tipo, info in modelos_finales.items():
    X_scaled = info["X_scaled"]
    etiquetas = info["etiquetas"]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    rng = np.random.default_rng(42)
    tamano_muestra = min(10000, len(X_pca))
    indices_grafica = rng.choice(
        len(X_pca),
        size=tamano_muestra,
        replace=False,
    )

    X_pca_muestra = X_pca[indices_grafica]
    segmentos_muestra = etiquetas[indices_grafica]

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca_muestra[:, 0],
        X_pca_muestra[:, 1],
        c=segmentos_muestra,
        alpha=0.4,
        s=15,
    )

    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Visualización de segmentos - {tipo}")
    plt.legend(*scatter.legend_elements(), title="Segmento")
    plt.grid(alpha=0.2)
    plt.show()

    print(
        f"{tipo} - varianza explicada por los dos componentes:",
        round(pca.explained_variance_ratio_.sum(), 4),
    )


# %%
# CARGAMOS LA BASE TRANSACCIONAL PARA PERFILAR LOS SEGMENTOS

datos = pd.read_parquet("../data/dataset_limpio.parquet")


# %%
# PRESENCIA EN LOS NEGOCIOS DE CORONA
# Revisamos en qué negocios ha comprado cada segmento al menos una vez.

presencia_negocio = (
    datos[["Cliente", "Negocio"]]
    .dropna()
    .drop_duplicates()
    .merge(
        features_segmentadas[["Cliente", "segmento_id"]],
        on="Cliente",
        how="left",
    )
    .dropna(subset=["segmento_id"])
)

tabla_negocio = pd.crosstab(
    presencia_negocio["segmento_id"],
    presencia_negocio["Negocio"],
)

total_clientes_segmento = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby("segmento_id")["Cliente"]
    .nunique()
)

tabla_negocio_pct = tabla_negocio.div(
    total_clientes_segmento,
    axis=0,
) * 100

print("PRESENCIA EN CADA NEGOCIO POR SEGMENTO (%)")
print(tabla_negocio_pct.round(2))


# %%
# INTEGRALIDAD: NÚMERO DE NEGOCIOS DISTINTOS
# Esta variable no participó en la creación de los clusters.

integralidad_segmentos = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby("segmento_id")["n_negocios_distintos"]
    .agg(["mean", "median"])
)

print("NÚMERO DE NEGOCIOS DISTINTOS POR SEGMENTO")
print(integralidad_segmentos.round(2))


# %%
# PRESENCIA EN LOS CLUSTERS/CANALES DE CORONA
# Contamos todos los clusters en los que cada cliente ha comprado al menos una vez.

presencia_cluster = (
    datos[["Cliente", "Cluster"]]
    .dropna()
    .drop_duplicates()
    .merge(
        features_segmentadas[["Cliente", "segmento_id"]],
        on="Cliente",
        how="left",
    )
    .dropna(subset=["segmento_id"])
)

tabla_cluster = pd.crosstab(
    presencia_cluster["segmento_id"],
    presencia_cluster["Cluster"],
)

tabla_cluster_pct = tabla_cluster.div(
    total_clientes_segmento,
    axis=0,
) * 100

print("PRESENCIA EN CADA CLUSTER DE CORONA POR SEGMENTO (%)")
print(tabla_cluster_pct.round(2))


# %%
# REGIÓN DOMINANTE POR SEGMENTO
# Mostramos las 8 regiones con más clientes para que la tabla siga siendo legible.

top_regiones = (
    features_segmentadas["region_dominante"]
    .value_counts()
    .head(8)
    .index
)

perfil_region = pd.crosstab(
    features_segmentadas["segmento_id"],
    features_segmentadas["region_dominante"],
    normalize="index",
) * 100

perfil_region_top = perfil_region.reindex(columns=top_regiones)

print("REGIÓN DOMINANTE POR SEGMENTO (%)")
print(perfil_region_top.round(2))


# %%
# SEGMENTOS DENTRO DE CADA REGIÓN
# Esta es la lectura inversa: dentro de cada región, qué segmentos aparecen.

segmentos_por_region = pd.crosstab(
    features_segmentadas["region_dominante"],
    features_segmentadas["segmento_id"],
    normalize="index",
) * 100

segmentos_por_region_top = segmentos_por_region.reindex(index=top_regiones)

print("DISTRIBUCIÓN DE SEGMENTOS DENTRO DE CADA REGIÓN (%)")
print(segmentos_por_region_top.round(2))

# %%
# SEGMENTOS DENTRO DE CADA REGIÓN - CONSUMIDORES FINALES
# Calculamos los porcentajes únicamente dentro de consumidores,
# para comparar correctamente los cinco segmentos CF.

consumidores = features_segmentadas[
    features_segmentadas["Tipo Cliente"] == "Consumidor final"
].copy()

segmentos_region_consumidores = pd.crosstab(
    consumidores["region_dominante"],
    consumidores["segmento_id"],
    normalize="index"
) * 100

print("SEGMENTOS DE CONSUMIDORES DENTRO DE CADA REGIÓN (%)")
print(
    segmentos_region_consumidores
    .reindex(index=top_regiones)
    .round(2)
)


# %%
# SEGMENTOS DENTRO DE CADA REGIÓN - EMPRESAS
# Hacemos la misma lectura únicamente dentro de empresas.

empresas = features_segmentadas[
    features_segmentadas["Tipo Cliente"] == "Empresa"
].copy()

segmentos_region_empresas = pd.crosstab(
    empresas["region_dominante"],
    empresas["segmento_id"],
    normalize="index"
) * 100

print("SEGMENTOS DE EMPRESAS DENTRO DE CADA REGIÓN (%)")
print(
    segmentos_region_empresas
    .reindex(index=top_regiones)
    .round(2)
)


# %%
# GENERACIÓN Y GÉNERO - SOLO CONSUMIDORES FINALES
# En empresas estas variables no tienen la misma interpretación,
# por eso las revisamos únicamente en la población de consumidores.

consumidores_segmentados = features_segmentadas[
    features_segmentadas["Tipo Cliente"] == "Consumidor final"
].copy()

perfil_generacion = pd.crosstab(
    consumidores_segmentados["segmento_id"],
    consumidores_segmentados["generacion"],
    normalize="index",
) * 100

print("GENERACIÓN POR SEGMENTO - CONSUMIDORES FINALES (%)")
print(perfil_generacion.round(2))


perfil_genero = pd.crosstab(
    consumidores_segmentados["segmento_id"],
    consumidores_segmentados["Genero"],
    normalize="index",
) * 100

print("\nGÉNERO POR SEGMENTO - CONSUMIDORES FINALES (%)")
print(perfil_genero.round(2))


# %%
# AUTORIZACIÓN DE TRATAMIENTO DE DATOS
# No forma los clusters; sirve para revisar qué tan accionable es cada segmento.

perfil_tratamiento = pd.crosstab(
    features_segmentadas["segmento_id"],
    features_segmentadas["Tratamiento Datos"],
    normalize="index",
) * 100

print("TRATAMIENTO DE DATOS POR SEGMENTO (%)")
print(perfil_tratamiento.round(2))


# %%
# ANTIGÜEDAD Y DURACIÓN DE LA RELACIÓN OBSERVADA
# La antigüedad parte de la primera compra visible en la base,
# no necesariamente de la primera compra histórica del cliente con Corona.

features_segmentadas["duracion_relacion_dias"] = (
    features_segmentadas["antiguedad_dias"]
    - features_segmentadas["recencia_dias"]
)

perfil_antiguedad = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby("segmento_id")["antiguedad_dias"]
    .agg(["mean", "median"])
)

perfil_duracion = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby("segmento_id")["duracion_relacion_dias"]
    .agg(["mean", "median"])
)

print("ANTIGÜEDAD OBSERVADA POR SEGMENTO")
print(perfil_antiguedad.round(2))

print("\nDURACIÓN ENTRE PRIMERA Y ÚLTIMA COMPRA OBSERVADA")
print(perfil_duracion.round(2))


# %%
# TABLA RESUMEN FINAL

resumen_final = (
    features_segmentadas.dropna(subset=["segmento_id"])
    .groupby(["Tipo Cliente", "segmento_id"])
    .agg(
        clientes=("Cliente", "nunique"),
        recencia_mediana=("recencia_dias", "median"),
        frecuencia_mediana=("frecuencia", "median"),
        valor_mediano=("valor_total", "median"),
        categorias_medianas=("n_categorias_distintas", "median"),
        negocios_medianos=("n_negocios_distintos", "median"),
        antiguedad_mediana=("antiguedad_dias", "median"),
        duracion_mediana=("duracion_relacion_dias", "median"),
    )
    .reset_index()
)

resumen_final["porcentaje_dentro_tipo"] = (
    resumen_final.groupby("Tipo Cliente")["clientes"]
    .transform(lambda x: x / x.sum() * 100)
)

print("RESUMEN FINAL DE LOS SEGMENTOS")
print(resumen_final.round(2).to_string(index=False))


# %%
# NOMBRES DEFINITIVOS DE LOS SEGMENTOS
# Los nombres resumen el comportamiento principal observado en cada grupo.

nombres_segmentos = {
    "CF_0": "Consumidores consolidados de alta vinculación",
    "CF_1": "Consumidores recientes de vínculo inicial",
    "CF_2": "Consumidores inactivos de bajo vínculo",
    "CF_3": "Consumidores activos en desarrollo",
    "CF_4": "Consumidores inactivos de vínculo intermedio",

    "EMP_0": "Empresas inactivas de bajo vínculo",
    "EMP_1": "Empresas activas en desarrollo",
    "EMP_2": "Empresas recientes de vínculo inicial",
    "EMP_3": "Empresas consolidadas de alta vinculación"
}

features_segmentadas["nombre_segmento"] = (
    features_segmentadas["segmento_id"]
    .map(nombres_segmentos)
)

print(
    features_segmentadas["nombre_segmento"]
    .value_counts(dropna=False)
)


# %%
# GRÁFICA: TAMAÑO DE LOS SEGMENTOS
# El porcentaje se calcula dentro de cada tipo de cliente para que consumidores
# y empresas no se mezclen en una misma base porcentual.

for tipo in ["Consumidor final", "Empresa"]:
    tabla_tipo = features_segmentadas[
        features_segmentadas["Tipo Cliente"] == tipo
    ]

    tamano_segmentos = (
        tabla_tipo["nombre_segmento"]
        .value_counts(normalize=True)
        .mul(100)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    barras = plt.barh(
        tamano_segmentos.index,
        tamano_segmentos.values,
    )

    for barra in barras:
        ancho = barra.get_width()
        plt.text(
            ancho + 0.3,
            barra.get_y() + barra.get_height() / 2,
            f"{ancho:.1f}%",
            va="center",
        )

    plt.xlabel("Porcentaje de clientes")
    plt.ylabel("")
    plt.title(f"Distribución de segmentos - {tipo}")
    plt.xlim(0, tamano_segmentos.max() + 4)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()


# %%
# GUARDAR BASE DE CLIENTES CON LA SEGMENTACIÓN
# El cliente sin Tipo Cliente conocido se conserva en la base,
# pero queda sin segmento porque no sabemos en cuál de las dos poblaciones ubicarlo.

features_segmentadas.to_parquet(
    "../data/features_clientes_segmentados.parquet",
    index=False,
)

print("Archivo guardado correctamente.")
print(features_segmentadas.shape)
print(
    features_segmentadas[
        ["Cliente", "Tipo Cliente", "segmento_tipo", "segmento_id", "nombre_segmento"]
    ].head()
)
