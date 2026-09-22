"""
Módulo de Modelos de Regresión para Proyecciones Demográficas
Práctica: U3_1_modelos_ml - Tópicos de Big Data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

DATA_URL = "https://ourworldindata.org/grapher/births-and-deaths-projected-to-2100.csv?v=1&csvType=full&useColumnShortNames=true"
LOCAL_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "births-and-deaths-projected-to-2100.csv")


def cargar_datos(ruta_local=LOCAL_CSV, url=DATA_URL):
    """
    Carga el dataset desde archivo local o desde la URL de Our World In Data.
    """
    if os.path.exists(ruta_local):
        print(f"Cargando dataset local desde: {ruta_local}")
        df = pd.read_csv(ruta_local)
    else:
        print(f"Descargando dataset desde: {url}")
        df = pd.read_csv(url, storage_options={"User-Agent": "Our World In Data data fetch/1.0"})
    
    # Renombrar columnas
    df = df.rename(columns={
        "entity": "Entity",
        "code": "Code",
        "year": "Year",
        "births__sex_all__age_all__variant_estimates": "births_estimates",
        "births__sex_all__age_all__variant_medium__projected": "births_projected",
        "deaths__sex_all__age_all__variant_estimates": "deaths_estimates",
        "deaths__sex_all__age_all__variant_medium__projected": "deaths_projected",
    })
    
    # Crear columnas unificadas
    df["births"] = df["births_estimates"].fillna(df["births_projected"])
    df["deaths"] = df["deaths_estimates"].fillna(df["deaths_projected"])
    return df


def preparar_datos_pais(df, pais="Mexico", anio_corte=2023):
    """
    Filtra los datos por país y separa los datos históricos y proyectados.
    """
    df_pais = df[df["Entity"] == pais].copy()
    df_historico = df_pais[df_pais["Year"] <= anio_corte].copy()
    df_proyectado = df_pais[df_pais["Year"] > anio_corte].copy()
    return df_historico, df_proyectado


def entrenar_modelos(X, y, X_futuro, grado_poly=2):
    """
    Entrena Regresión Lineal, Regresión Polinomial y Random Forest,
    y genera predicciones históricas y futuras.
    """
    # 1. Regresión Lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X, y)
    pred_lineal_hist = modelo_lineal.predict(X)
    pred_lineal_fut = modelo_lineal.predict(X_futuro)

    # 2. Regresión Polinomial
    poly = PolynomialFeatures(degree=grado_poly, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_futuro_poly = poly.transform(X_futuro)
    
    modelo_poly = LinearRegression()
    modelo_poly.fit(X_poly, y)
    pred_poly_hist = modelo_poly.predict(X_poly)
    pred_poly_fut = modelo_poly.predict(X_futuro_poly)

    # 3. Random Forest
    modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_rf.fit(X, y)
    pred_rf_hist = modelo_rf.predict(X)
    pred_rf_fut = modelo_rf.predict(X_futuro)

    return {
        "lineal": {"modelo": modelo_lineal, "hist": pred_lineal_hist, "fut": pred_lineal_fut},
        "polinomial": {"modelo": modelo_poly, "poly": poly, "hist": pred_poly_hist, "fut": pred_poly_fut},
        "rf": {"modelo": modelo_rf, "hist": pred_rf_hist, "fut": pred_rf_fut}
    }


def main():
    df = cargar_datos()
    df_hist, df_proy = preparar_datos_pais(df, "Mexico", 2023)
    
    X = df_hist[["Year"]]
    y_nacimientos = df_hist["births"]
    y_defunciones = df_hist["deaths"]
    
    anios_futuros = np.arange(2024, 2101).reshape(-1, 1)
    X_futuro = pd.DataFrame(anios_futuros, columns=["Year"])
    
    print("\n--- Entrenando modelos para Nacimientos ---")
    res_nac = entrenar_modelos(X, y_nacimientos, X_futuro)
    print(f"Nacimientos 2100 (Lineal):     {res_nac['lineal']['fut'][-1]:,.0f}")
    print(f"Nacimientos 2100 (Polinomial): {res_nac['polinomial']['fut'][-1]:,.0f}")
    print(f"Nacimientos 2100 (RF):         {res_nac['rf']['fut'][-1]:,.0f}")
    
    print("\n--- Entrenando modelos para Defunciones ---")
    res_def = entrenar_modelos(X, y_defunciones, X_futuro)
    print(f"Defunciones 2100 (Lineal):     {res_def['lineal']['fut'][-1]:,.0f}")
    print(f"Defunciones 2100 (Polinomial): {res_def['polinomial']['fut'][-1]:,.0f}")
    print(f"Defunciones 2100 (RF):         {res_def['rf']['fut'][-1]:,.0f}")


if __name__ == "__main__":
    main()
