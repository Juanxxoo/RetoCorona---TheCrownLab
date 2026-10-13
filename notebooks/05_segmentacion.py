# %%
#SEGMENTACIÓN 1: COMPORTAMIENTO
import pandas as pd
import numpy as np

features = pd.read_parquet("../data/features_clientes.parquet")

variables_cluster = [
    "recencia_dias",
    "frecuencia",
    "valor_total",
    "n_categorias_distintas"
]

X = features[variables_cluster].copy()

print(X.head())
print(X.describe())

# %%
#Transformar esas dos variables
X_transformado = X.copy()

X_transformado["frecuencia"] = np.log1p(X_transformado["frecuencia"])
X_transformado["valor_total"] = np.log1p(X_transformado["valor_total"])

print(X_transformado.describe())

# %%
#Variables comparables 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_transformado)

print(X_scaled[:5])

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Lista para guardar los resultados para cada número de clusters
resultados = []

# Probar diferentes cantidades de clusters: desde 2 hasta 7
for k in range(2, 8):

    # Crear modelo K-Means con k grupos
    # random_state=42 permite obtener los mismos resultados cada vez
    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    # Entrenar el modelo y obtener el cluster asignado a cada cliente
    etiquetas = modelo.fit_predict(X_scaled)
    
    # Silhouette Score:mide qué tan bien separado está cada cliente de los otros clusters
    # entre más alto, mejor
    silhouette = silhouette_score(
    X_scaled,
    etiquetas,
    sample_size=10000,
    random_state=42
)
    
    # Davies-Bouldin Index:mide qué tan parecidos o cercanos están los clusters entre sí
    # entre más bajo, mejor
    davies = davies_bouldin_score(X_scaled, etiquetas)
    
    # Guardar los resultados para compararlos después
    resultados.append({
        "k": k,
        "silhouette": silhouette,
        "davies_bouldin": davies
    })
    
resultados

# %%
# Entrenamos una primera segmentación con K = 2,
# ya que fue la opción con mejores métricas hasta ahora.

modelo_k2 = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Asignamos un cluster a cada cliente
features["segmento_k2"] = modelo_k2.fit_predict(X_scaled)

# Miramos cuántos clientes quedaron en cada grupo
print(features["segmento_k2"].value_counts())

print("\nPorcentaje de clientes por segmento:")
print(
    features["segmento_k2"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

# Perfil básico de los dos grupos usando los valores ORIGINALES,
# para poder interpretarlos fácilmente.
perfil_k2 = features.groupby("segmento_k2")[
    ["recencia_dias", "frecuencia", "valor_total", "n_categorias_distintas"]
].median()

print("\nPerfil mediano de cada segmento:")
print(perfil_k2)

# %%
# Probamos ahora K = 3 para ver si aparecen perfiles
# más detallados y útiles para el negocio.

modelo_k3 = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Asignamos un segmento a cada cliente
features["segmento_k3"] = modelo_k3.fit_predict(X_scaled)

# Cantidad y porcentaje de clientes por segmento
print("Cantidad de clientes:")
print(features["segmento_k3"].value_counts().sort_index())

print("\nPorcentaje de clientes:")
print(
    features["segmento_k3"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

# Perfil mediano de cada segmento
perfil_k3 = features.groupby("segmento_k3")[
    ["recencia_dias", "frecuencia", "valor_total", "n_categorias_distintas"]
].median()

print("\nPerfil mediano:")
print(perfil_k3)

# %%
# Probamos K = 4 para evaluar si aparecen segmentos
# adicionales que sean útiles e interpretables para Corona.

modelo_k4 = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Asignamos un segmento a cada cliente
features["segmento_k4"] = modelo_k4.fit_predict(X_scaled)

# Cantidad de clientes
print("Cantidad de clientes:")
print(features["segmento_k4"].value_counts().sort_index())

# Porcentaje de clientes
print("\nPorcentaje de clientes:")
print(
    features["segmento_k4"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

# Perfil mediano de cada segmento
perfil_k4 = features.groupby("segmento_k4")[
    ["recencia_dias", "frecuencia", "valor_total", "n_categorias_distintas"]
].median()

print("\nPerfil mediano:")
print(perfil_k4)

# %%
# Probamos K = 5 para evaluar si aparece un segmento adicional
# que aporte una diferencia clara frente a K = 4.

modelo_k5 = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Asignamos un segmento a cada cliente
features["segmento_k5"] = modelo_k5.fit_predict(X_scaled)

# Cantidad de clientes
print("Cantidad de clientes:")
print(features["segmento_k5"].value_counts().sort_index())

# Porcentaje de clientes
print("\nPorcentaje de clientes:")
print(
    features["segmento_k5"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

# Perfil mediano de cada segmento
perfil_k5 = features.groupby("segmento_k5")[
    ["recencia_dias", "frecuencia", "valor_total", "n_categorias_distintas"]
].median()

print("\nPerfil mediano:")
print(perfil_k5)

# %%
# PERFIL DE LOS SEGMENTOS:
# Revisamos qué tipo de cliente y qué cluster de Corona predominan dentro de cada uno de nuestros 5 segmentos.

# Porcentaje de Tipo Cliente dentro de cada segmento
perfil_tipo_cliente = pd.crosstab(
    features["segmento_k5"],
    features["Tipo Cliente"],
    normalize="index"
) * 100

print("TIPO DE CLIENTE POR SEGMENTO (%)")
print(perfil_tipo_cliente.round(2))


# Porcentaje de cluster dominante de Corona dentro de cada segmento
perfil_cluster_corona = pd.crosstab(
    features["segmento_k5"],
    features["cluster_dominante"],
    normalize="index"
) * 100

print("\nCLUSTER DE CORONA POR SEGMENTO (%)")
print(perfil_cluster_corona.round(2))

# %%
# PRESENCIA TOTAL POR CLUSTER/CANAL DE CORONA
# A diferencia de cluster_dominante, aquí contamos todos los clusters
# en los que cada cliente ha comprado al menos una vez.

datos = pd.read_parquet("../data/dataset_limpio.parquet")

# Nos quedamos con una sola combinación Cliente-Cluster
# para no contar muchas veces al mismo cliente por sus distintas transacciones
presencia_cluster = (
    datos[["Cliente", "Cluster"]]
    .dropna()
    .drop_duplicates()
)

# Agregamos el segmento K=5 que acabamos de construir
presencia_cluster = presencia_cluster.merge(
    features[["Cliente", "segmento_k5"]],
    on="Cliente",
    how="left"
)

# Calculamos qué porcentaje de clientes de cada segmento
# ha comprado al menos una vez en cada cluster de Corona
tabla_presencia = pd.crosstab(
    presencia_cluster["segmento_k5"],
    presencia_cluster["Cluster"]
)

# Dividimos por el TOTAL de clientes de cada segmento
total_clientes_segmento = features.groupby("segmento_k5")["Cliente"].nunique()

tabla_presencia_pct = (
    tabla_presencia
    .div(total_clientes_segmento, axis=0)
    * 100
)

print("PRESENCIA EN CADA CLUSTER DE CORONA POR SEGMENTO (%)")
print(tabla_presencia_pct.round(2))

# %%
# Verificamos cuántos registros y clientes existen realmente
# en cada cluster de Corona dentro de la base transaccional.

print("Número de registros por cluster:")
print(datos["Cluster"].value_counts(dropna=False))

print("\nNúmero de clientes únicos por cluster:")
print(
    datos.groupby("Cluster")["Cliente"]
    .nunique()
    .sort_values(ascending=False)
)

# %%
from sklearn.metrics import adjusted_rand_score

# VALIDACIÓN 1: ESTABILIDAD DE K = 5
# Queremos comprobar si K-Means encuentra grupos similares
# cuando cambia la inicialización aleatoria.

# Tomamos como referencia la segmentación K=5 que ya construimos
etiquetas_referencia = features["segmento_k5"].values

# Probamos varias semillas distintas
semillas = [0, 1, 7, 21, 100]

resultados_estabilidad = []

for semilla in semillas:

    # Entrenamos nuevamente K-Means con la misma cantidad de clusters
    # pero cambiando la inicialización aleatoria
    modelo_prueba = KMeans(
        n_clusters=5,
        random_state=semilla,
        n_init=10
    )

    nuevas_etiquetas = modelo_prueba.fit_predict(X_scaled)

    # Adjusted Rand Index (ARI):
    # compara si dos segmentaciones agrupan juntos a los mismos clientes.
    # 1 = prácticamente la misma segmentación
    # cerca de 0 = segmentaciones muy diferentes
    ari = adjusted_rand_score(
        etiquetas_referencia,
        nuevas_etiquetas
    )

    resultados_estabilidad.append({
        "random_state": semilla,
        "ARI_vs_modelo_original": ari
    })

pd.DataFrame(resultados_estabilidad)

# %%
from sklearn.metrics import silhouette_samples

# VALIDACIÓN 2: SILHOUETTE POR SEGMENTO
# Usamos una muestra de 10.000 clientes para evitar un cálculo demasiado pesado.

np.random.seed(42)

indices_muestra = np.random.choice(
    len(X_scaled),
    size=10000,
    replace=False
)

# Datos y etiquetas de la muestra
X_muestra = X_scaled[indices_muestra]
segmentos_muestra = features["segmento_k5"].values[indices_muestra]

# Calculamos el Silhouette de cada cliente de la muestra
silhouette_individual = silhouette_samples(
    X_muestra,
    segmentos_muestra
)

# Creamos una tabla para resumirlo por segmento
silhouette_por_segmento = pd.DataFrame({
    "segmento": segmentos_muestra,
    "silhouette": silhouette_individual
})

resumen_silhouette = silhouette_por_segmento.groupby("segmento")["silhouette"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(resumen_silhouette)

# %%
# VALIDACIÓN 3: EMPRESAS VS. CONSUMIDORES FINALES
# Queremos revisar si la estructura de K=5 se mantiene
# cuando analizamos cada tipo de cliente por separado.

resultados_tipo_cliente = []

for tipo in ["Consumidor final", "Empresa"]:

    # Identificamos qué clientes pertenecen a este tipo
    mascara = features["Tipo Cliente"] == tipo

    # Tomamos sus variables ya transformadas y escaladas
    X_tipo = X_scaled[mascara.values]

    # Segmentación original de esos mismos clientes
    etiquetas_globales = features.loc[
        mascara, "segmento_k5"
    ].values

    # Entrenamos K-Means solo con este tipo de cliente
    modelo_tipo = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    etiquetas_separadas = modelo_tipo.fit_predict(X_tipo)

    # Comparamos la segmentación separada con la segmentación general.
    # ARI = 1 significa que prácticamente forma los mismos grupos.
    ari = adjusted_rand_score(
        etiquetas_globales,
        etiquetas_separadas
    )

    resultados_tipo_cliente.append({
        "Tipo Cliente": tipo,
        "Número clientes": len(X_tipo),
        "ARI_vs_segmentacion_general": ari
    })

pd.DataFrame(resultados_tipo_cliente)

# %%
# VALIDACIÓN 3B:
# Evaluamos consumidores finales y empresas por separado
# para comprobar qué estructura de clusters aparece en cada población.

resultados_separados = []

for tipo in ["Consumidor final", "Empresa"]:

    # Seleccionamos únicamente los clientes de este tipo
    mascara = features["Tipo Cliente"] == tipo

    # Tomamos las mismas 4 variables que usamos en el modelo general
    X_tipo = X_transformado.loc[mascara].copy()

    # Escalamos dentro de esta población
    scaler_tipo = StandardScaler()
    X_tipo_scaled = scaler_tipo.fit_transform(X_tipo)

    # Probamos distintos números de clusters
    for k in range(2, 7):

        modelo_tipo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        etiquetas_tipo = modelo_tipo.fit_predict(X_tipo_scaled)

        # Silhouette sobre una muestra para que no sea demasiado pesado
        tamano_muestra = min(10000, len(X_tipo_scaled))

        silhouette = silhouette_score(
            X_tipo_scaled,
            etiquetas_tipo,
            sample_size=tamano_muestra,
            random_state=42
        )

        # Davies-Bouldin sobre todos los clientes del grupo
        davies = davies_bouldin_score(
            X_tipo_scaled,
            etiquetas_tipo
        )

        resultados_separados.append({
            "Tipo Cliente": tipo,
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies
        })

resultados_separados = pd.DataFrame(resultados_separados)

print(resultados_separados)

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# VISUALIZACIÓN DE LOS CLUSTERS
# PCA reduce las 4 variables del clustering a 2 dimensiones
# únicamente para poder representar gráficamente los segmentos.

pca = PCA(n_components=2)

# Reducimos los datos escalados de 4 dimensiones a 2
X_pca = pca.fit_transform(X_scaled)

# Tomamos una muestra de 10.000 clientes para que la gráfica sea legible
np.random.seed(42)

indices_grafica = np.random.choice(
    len(X_pca),
    size=10000,
    replace=False
)

X_pca_muestra = X_pca[indices_grafica]
segmentos_muestra = features["segmento_k5"].values[indices_grafica]

# Dibujamos los clientes según el segmento asignado por K-Means
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_pca_muestra[:, 0],
    X_pca_muestra[:, 1],
    c=segmentos_muestra,
    alpha=0.4,
    s=15
)

plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.title("Visualización de los cinco segmentos de clientes")

plt.legend(
    *scatter.legend_elements(),
    title="Segmento"
)

plt.grid(alpha=0.2)
plt.show()

# Cuánta información de las 4 variables logramos representar en estos 2 ejes
print(
    "Varianza explicada por los dos componentes:",
    pca.explained_variance_ratio_.sum()
)

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Convertimos los resultados obtenidos anteriormente en una tabla
resultados_df = pd.DataFrame(resultados)

# Gráfica de Silhouette Score
plt.figure(figsize=(8, 5))

plt.plot(
    resultados_df["k"],
    resultados_df["silhouette"],
    marker="o"
)

plt.xlabel("Número de clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Evaluación del número de clusters mediante Silhouette Score")
plt.xticks(resultados_df["k"])
plt.grid(alpha=0.3)

plt.show()

# %%
# Revisamos qué negocio de Corona predomina dentro de cada segmento.
# Estas variables NO participaron en la creación de los clusters:
# se usan después para entender mejor quién compone cada grupo.

perfil_negocio = pd.crosstab(
    features["segmento_k5"],
    features["negocio_dominante"],
    normalize="index"
) * 100

print("NEGOCIO DOMINANTE POR SEGMENTO (%)")
print(perfil_negocio.round(2))

# %%
# PRESENCIA TOTAL POR NEGOCIO
# Aquí revisamos en qué negocios ha comprado cada cliente al menos una vez,
# no solamente cuál es su negocio dominante.

presencia_negocio = (
    datos[["Cliente", "Negocio"]]
    .dropna()
    .drop_duplicates()
)

# Agregamos el segmento K=5 de cada cliente
presencia_negocio = presencia_negocio.merge(
    features[["Cliente", "segmento_k5"]],
    on="Cliente",
    how="left"
)

# Cantidad de clientes de cada segmento presentes en cada negocio
tabla_negocio = pd.crosstab(
    presencia_negocio["segmento_k5"],
    presencia_negocio["Negocio"]
)

# Lo convertimos a porcentaje sobre el total de clientes de cada segmento
total_clientes_segmento = features.groupby("segmento_k5")["Cliente"].nunique()

tabla_negocio_pct = (
    tabla_negocio
    .div(total_clientes_segmento, axis=0)
    * 100
)

print("PRESENCIA EN CADA NEGOCIO POR SEGMENTO (%)")
print(tabla_negocio_pct.round(2))

# %%
# Revisamos directamente cuántos negocios distintos compra
# cada segmento como medida adicional de integralidad.

integralidad_segmentos = features.groupby("segmento_k5")[
    "n_negocios_distintos"
].agg(
    ["mean", "median"]
)

print("NÚMERO DE NEGOCIOS DISTINTOS POR SEGMENTO")
print(integralidad_segmentos.round(2))

# %%
# REGIÓN POR SEGMENTO
# Revisamos si los cinco perfiles tienen diferencias territoriales.
# Mostramos únicamente las 8 regiones con más clientes para que la tabla
# sea fácil de interpretar.

# Identificamos las 8 regiones con mayor número de clientes
top_regiones = (
    features["region_dominante"]
    .value_counts()
    .head(8)
    .index
)

# Calculamos qué porcentaje de cada segmento pertenece a cada región
perfil_region = pd.crosstab(
    features["segmento_k5"],
    features["region_dominante"],
    normalize="index"
) * 100

# Nos quedamos solo con las regiones principales
perfil_region_top = perfil_region[top_regiones]

print("REGIÓN DOMINANTE POR SEGMENTO (%)")
print(perfil_region_top.round(2))

# %%
# SEGMENTOS DENTRO DE CADA REGIÓN
# Ahora cambiamos la pregunta:
# dentro de cada región, ¿qué porcentaje de clientes pertenece a cada segmento?

segmentos_por_region = pd.crosstab(
    features["region_dominante"],
    features["segmento_k5"],
    normalize="index"
) * 100

# Mostramos solo las 8 regiones principales
segmentos_por_region_top = segmentos_por_region.loc[top_regiones]

print("DISTRIBUCIÓN DE SEGMENTOS DENTRO DE CADA REGIÓN (%)")
print(segmentos_por_region_top.round(2))

# %%
# GENERACIÓN POR SEGMENTO
# La tabla responde:
# dentro de cada segmento, ¿qué composición generacional tienen sus clientes?

perfil_generacion = pd.crosstab(
    features["segmento_k5"],
    features["generacion"],
    normalize="index"
) * 100

print("GENERACIÓN POR SEGMENTO (%)")
print(perfil_generacion.round(2))

# %%
# GENERACIÓN DENTRO DE CONSUMIDORES FINALES
# Excluimos empresas para comparar generaciones sin que
# "No aplica (Empresa)" distorsione los porcentajes.

consumidores = features[
    features["Tipo Cliente"] == "Consumidor final"
].copy()

perfil_generacion_consumidores = pd.crosstab(
    consumidores["segmento_k5"],
    consumidores["generacion"],
    normalize="index"
) * 100

print("GENERACIÓN POR SEGMENTO - SOLO CONSUMIDORES FINALES (%)")
print(perfil_generacion_consumidores.round(2))

# %%
# GÉNERO POR SEGMENTO - SOLO CONSUMIDORES FINALES
# Revisamos si la composición por género cambia entre los segmentos.

perfil_genero = pd.crosstab(
    consumidores["segmento_k5"],
    consumidores["Genero"],
    normalize="index"
) * 100

print("GÉNERO POR SEGMENTO - SOLO CONSUMIDORES FINALES (%)")
print(perfil_genero.round(2))

# %%
# AUTORIZACIÓN DE TRATAMIENTO DE DATOS POR SEGMENTO
# Esto no forma los clusters.
# Se utiliza para saber qué proporción de cada segmento
# podría ser considerada en acciones directas de marketing.

perfil_tratamiento = pd.crosstab(
    features["segmento_k5"],
    features["Tratamiento Datos"],
    normalize="index"
) * 100

print("TRATAMIENTO DE DATOS POR SEGMENTO (%)")
print(perfil_tratamiento.round(2))

# %%
# ANTIGÜEDAD OBSERVADA POR SEGMENTO
# Revisamos hace cuánto apareció la primera compra observada de los clientes
# dentro del periodo disponible en la base.
# Importante: no representa necesariamente su primera compra histórica con Corona.

perfil_antiguedad = features.groupby("segmento_k5")[
    "antiguedad_dias"
].agg(
    ["mean", "median"]
)

print("ANTIGÜEDAD OBSERVADA POR SEGMENTO")
print(perfil_antiguedad.round(2))

# %%
# DURACIÓN DE LA RELACIÓN OBSERVADA
# Tiempo entre la primera y la última compra observada del cliente.

features["duracion_relacion_dias"] = (
    features["antiguedad_dias"] - features["recencia_dias"]
)

perfil_duracion = features.groupby("segmento_k5")[
    "duracion_relacion_dias"
].agg(["mean", "median"])

print("DURACIÓN ENTRE PRIMERA Y ÚLTIMA COMPRA OBSERVADA")
print(perfil_duracion.round(2))

# %%
# TABLA RESUMEN FINAL DE LOS 5 SEGMENTOS

resumen_segmentos = features.groupby("segmento_k5").agg(
    clientes=("Cliente", "nunique"),
    recencia_mediana=("recencia_dias", "median"),
    frecuencia_mediana=("frecuencia", "median"),
    valor_mediano=("valor_total", "median"),
    categorias_medianas=("n_categorias_distintas", "median"),
    negocios_medianos=("n_negocios_distintos", "median"),
    antiguedad_mediana=("antiguedad_dias", "median"),
    duracion_mediana=("duracion_relacion_dias", "median")
)

resumen_segmentos["porcentaje_clientes"] = (
    resumen_segmentos["clientes"] / resumen_segmentos["clientes"].sum() * 100
)

print("RESUMEN FINAL DE LOS SEGMENTOS")
print(resumen_segmentos.round(2))

# %%
# NOMBRES DEFINITIVOS DE LOS SEGMENTOS

nombres_segmentos = {
    0: "Inactivos de bajo vínculo",
    1: "Inactivos de vínculo intermedio",
    2: "Consolidados de alta vinculación",
    3: "Activos en desarrollo",
    4: "Recientes de vínculo inicial"
}

features["nombre_segmento"] = (
    features["segmento_k5"]
    .map(nombres_segmentos)
)

# Verificamos cantidad de clientes por nombre
print(
    features["nombre_segmento"]
    .value_counts()
)

# %%
# GRÁFICA: TAMAÑO DE LOS SEGMENTOS

import matplotlib.pyplot as plt

tamano_segmentos = (
    features["nombre_segmento"]
    .value_counts(normalize=True)
    .mul(100)
    .sort_values()
)

plt.figure(figsize=(10, 6))

barras = plt.barh(
    tamano_segmentos.index,
    tamano_segmentos.values
)

# Mostrar porcentaje al final de cada barra
for barra in barras:
    ancho = barra.get_width()
    plt.text(
        ancho + 0.3,
        barra.get_y() + barra.get_height()/2,
        f"{ancho:.1f}%",
        va="center"
    )

plt.xlabel("Porcentaje de clientes")
plt.ylabel("")
plt.title("Distribución de clientes por segmento")

plt.xlim(0, tamano_segmentos.max() + 4)
plt.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.show()

# %%
# GUARDAR BASE DE CLIENTES CON SEGMENTACIÓN FINAL

features.to_parquet(
    "../data/features_clientes_segmentados.parquet",
    index=False
)

print("Archivo guardado correctamente.")
print(features.shape)
print(features[["Cliente", "segmento_k5", "nombre_segmento"]].head())
# %%
