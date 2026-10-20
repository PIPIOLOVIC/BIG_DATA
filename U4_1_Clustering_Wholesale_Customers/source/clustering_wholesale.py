"""
Módulo de Clustering y Segmentación de Clientes (Wholesale Customers)
Práctica: U4_1_Clustering_Wholesale_Customers - Tópicos de Big Data

Este script implementa el pipeline completo de aprendizaje no supervisado:
1. Carga y validación del dataset (local o UCI Machine Learning Repository).
2. Transformación logarítmica (log1p) y estandarización (StandardScaler).
3. Evaluación de clusters (Inercia y Silhouette Score para k=2..8).
4. Modelado con K-Means (k=3) y perfilamiento de segmentos (medianas, perfiles relativos, cruce con Channel y Region).
5. Reducción dimensional con PCA (2 componentes).
6. Búsqueda de clientes similares mediante K-Nearest Neighbors (KNN).
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CSV = os.path.join(BASE_DIR, "data", "Wholesale customers data.csv")

VARIABLES_GASTO = [
    "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"
]


def cargar_datos(ruta_local=LOCAL_CSV, url=DATA_URL):
    """
    Carga el dataset desde archivo local o directamente desde la URL de UCI.
    """
    if os.path.exists(ruta_local):
        print(f"[INFO] Cargando dataset local desde: {ruta_local}")
        df = pd.read_csv(ruta_local)
    else:
        print(f"[INFO] Descargando dataset desde UCI: {url}")
        df = pd.read_csv(url)
    return df


def explorar_calidad(df):
    """
    Retorna un resumen de tipos de datos, nulos y duplicados.
    """
    resumen = pd.DataFrame({
        "tipo_dato": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "unicos": df.nunique()
    })
    duplicados = df.duplicated().sum()
    return resumen, duplicados


def preparar_datos(df, variables=VARIABLES_GASTO):
    """
    Aplica transformación log1p para mitigar asimetría de cola pesada
    y StandardScaler para igualar escalas de las variables de gasto.
    """
    X_original = df[variables].copy()
    X_log = np.log1p(X_original)
    
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X_log)
    X_preparado = pd.DataFrame(X_escalado, columns=variables, index=df.index)
    
    return X_original, X_preparado, escalador


def evaluar_numero_clusters(X_preparado, k_min=2, k_max=8, random_state=42):
    """
    Evalúa K-Means para un rango de valores de k calculando Inercia y Silhouette Score.
    """
    metricas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        etiquetas = modelo.fit_predict(X_preparado)
        metricas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silhouette": silhouette_score(X_preparado, etiquetas)
        })
    return pd.DataFrame(metricas)


def ajustar_kmeans(X_preparado, df_original, k=3, random_state=42):
    """
    Ajusta el modelo K-Means final con el valor seleccionado de k.
    Agrega la columna 'Cluster' al DataFrame de datos.
    """
    modelo = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    df_resultado = df_original.copy()
    df_resultado["Cluster"] = modelo.fit_predict(X_preparado)
    return modelo, df_resultado


def analizar_perfiles(df_con_cluster, variables=VARIABLES_GASTO):
    """
    Calcula:
    1. Conteo y porcentaje de clientes por cluster.
    2. Perfil mediano de gasto original.
    3. Perfil relativo normalizado.
    4. Cruce de canales (1: Horeca, 2: Retail) y regiones (1: Lisboa, 2: Oporto, 3: Otra).
    """
    # Conteo
    conteo = df_con_cluster["Cluster"].value_counts().sort_index().rename("clientes").to_frame()
    conteo["porcentaje"] = (conteo["clientes"] / len(df_con_cluster) * 100).round(1)
    
    # Mediana en escala monetaria original
    perfil_mediano = df_con_cluster.groupby("Cluster")[variables].median().round(0).astype(int)
    
    # Perfil relativo (mediana / media de gasto del grupo)
    perfil_relativo = perfil_mediano.div(perfil_mediano.mean(axis=1), axis=0).round(2)
    
    # Tablas cruzadas con variables de contexto
    canal_pct = pd.crosstab(df_con_cluster["Cluster"], df_con_cluster["Channel"], normalize="index").mul(100).round(1)
    region_pct = pd.crosstab(df_con_cluster["Cluster"], df_con_cluster["Region"], normalize="index").mul(100).round(1)
    
    return {
        "conteo": conteo,
        "perfil_mediano": perfil_mediano,
        "perfil_relativo": perfil_relativo,
        "canal_distribucion": canal_pct,
        "region_distribucion": region_pct
    }


def reducir_dimensiones_pca(X_preparado, n_componentes=2):
    """
    Aplica PCA sobre las variables escaladas para facilitar la visualización en 2D.
    """
    pca = PCA(n_components=n_componentes)
    coordenadas = pca.fit_transform(X_preparado)
    df_pca = pd.DataFrame(coordenadas, columns=[f"PC{i+1}" for i in range(n_componentes)], index=X_preparado.index)
    return pca, df_pca


def consultar_vecinos_knn(X_preparado, df_original, indice_cliente=0, n_vecinos=5):
    """
    Encuentra los clientes más similares a uno dado utilizando KNN (NearestNeighbors)
    sobre las variables preparadas y la distancia euclidiana.
    """
    buscador = NearestNeighbors(n_neighbors=n_vecinos + 1, metric="euclidean")
    buscador.fit(X_preparado)
    distancias, indices = buscador.kneighbors(X_preparado.iloc[[indice_cliente]])
    
    vecinos_df = df_original.iloc[indices[0][1:]].copy()
    vecinos_df["Distancia_Euclidiana"] = distancias[0][1:].round(3)
    return vecinos_df


def main():
    print("=" * 70)
    print("PRÁCTICA: SEGMENTACIÓN DE CLIENTES (WHOLESALE CUSTOMERS)")
    print("=" * 70)
    
    # 1. Cargar datos
    datos = cargar_datos()
    print(f"Dimensiones del dataset: {datos.shape[0]} filas x {datos.shape[1]} columnas\n")
    
    # 2. Calidad de datos
    resumen_cal, duplicados = explorar_calidad(datos)
    print("--- Calidad y Nulos ---")
    print(resumen_cal)
    print(f"Filas duplicadas: {duplicados}\n")
    
    # 3. Preparación de variables
    X_orig, X_prep, escalador = preparar_datos(datos)
    print("[INFO] Variables preparadas con log1p y StandardScaler:")
    print(X_prep.head(3).round(2), "\n")
    
    # 4. Evaluación de K (2 a 8)
    print("--- Evaluación de Número de Clusters (Inercia y Silhouette) ---")
    metricas_k = evaluar_numero_clusters(X_prep, k_min=2, k_max=8)
    print(metricas_k.to_string(index=False), "\n")
    
    # 5. Ajustar modelo final con k=3
    k_elegido = 3
    print(f"--- Ajuste de K-Means con k={k_elegido} ---")
    modelo_km, datos_cluster = ajustar_kmeans(X_prep, datos, k=k_elegido)
    perfiles = analizar_perfiles(datos_cluster)
    
    print("\n1. Distribución de clientes por cluster:")
    print(perfiles["conteo"])
    
    print("\n2. Perfil de gasto mediano por cluster (escala monetaria original):")
    print(perfiles["perfil_mediano"])
    
    print("\n3. Perfil de gasto relativo por cluster (mediana / media del grupo):")
    print(perfiles["perfil_relativo"])
    
    print("\n4. Proporción por Canal (% en fila; 1: Horeca, 2: Retail):")
    print(perfiles["canal_distribucion"])
    
    print("\n5. Proporción por Región (% en fila; 1: Lisboa, 2: Oporto, 3: Otra):")
    print(perfiles["region_distribucion"])
    
    # 6. Reducción PCA
    pca, df_pca = reducir_dimensiones_pca(X_prep, n_componentes=2)
    var_exp = pca.explained_variance_ratio_ * 100
    print(f"\n--- Reducción PCA (2 Componentes) ---")
    print(f"Varianza explicada: PC1 = {var_exp[0]:.2f}%, PC2 = {var_exp[1]:.2f}% (Total = {var_exp.sum():.2f}%)")
    
    # 7. Consulta KNN
    idx_consulta = 0
    print(f"\n--- Búsqueda de Clientes Similares con KNN (Consulta Cliente #{idx_consulta}) ---")
    vecinos = consultar_vecinos_knn(X_prep, datos_cluster, indice_cliente=idx_consulta, n_vecinos=5)
    print(f"Gasto del cliente #{idx_consulta} (Cluster {datos_cluster.loc[idx_consulta, 'Cluster']}):")
    print(datos_cluster.loc[[idx_consulta], VARIABLES_GASTO + ["Cluster", "Channel"]])
    print("\n5 Clientes más cercanos identificados por KNN:")
    print(vecinos[VARIABLES_GASTO + ["Cluster", "Channel", "Distancia_Euclidiana"]])
    
    print("\n" + "=" * 70)
    print("EJECUCIÓN DEL PIPELINE FINALIZADA CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    main()
