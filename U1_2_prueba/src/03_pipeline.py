from pathlib import Path
import pandas as pd
import requests

# ============================================================
# 1. EXTRACCIÓN — Fetch the data
# ============================================================

url_datos = "https://ourworldindata.org/grapher/child-mortality.csv?v=1&csvType=full&useColumnShortNames=true"

df = pd.read_csv(
    url_datos,
    storage_options={
        "User-Agent": "Our World In Data data fetch/1.0"
    }
)

# ============================================================
# 2. EXTRACCIÓN — Fetch the metadata
# ============================================================

url_metadata = "https://ourworldindata.org/grapher/child-mortality.metadata.json?v=1&csvType=full&useColumnShortNames=true"

metadata = requests.get(url_metadata).json()

# ============================================================
# 3. EXPLORACIÓN INICIAL
# ============================================================

print("Dimensiones del DataFrame:")
print(df.shape)

print("\nColumnas:")
print(df.columns.tolist())

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores nulos:")
print(df.isnull().sum())

# ============================================================
# 4. LIMPIEZA
# ============================================================

# Eliminar registros que no tengan código de país
df = df.dropna(subset=["code"])

# Eliminar registros que no tengan tasa de mortalidad infantil
df = df.dropna(subset=["child_mortality_rate"])

# ============================================================
# 5. TRANSFORMACIÓN
# ============================================================

# Convertir el año a entero
df["Year"] = df["year"].astype(int)

# Crear una nueva columna con la tasa de mortalidad infantil
# redondeada a dos decimales
df["child_mortality_round"] = df["child_mortality_rate"].round(2)

# ============================================================
# 6. FILTRADO
# ============================================================

# Trabajaremos únicamente con datos a partir del año 2000
df = df[df["year"] >= 2000]

# ============================================================
# 7. ANÁLISIS
# ============================================================

# Promedio de mortalidad infantil por país
promedio_pais = (
    df.groupby("entity")["child_mortality_round"]
      .mean()
      .round(2)
      .sort_values(ascending=False)
)

print("\nPromedio de mortalidad infantil por país:")
print(promedio_pais.head(10))

# ============================================================
# 8. RESULTADO
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
carpeta_salida = BASE_DIR / "data" / "processed"
carpeta_salida.mkdir(parents=True, exist_ok=True)

archivo_salida = carpeta_salida / "promedio_mortalidad_infantil_por_pais.csv"

promedio_pais.to_csv(
    archivo_salida,
    header=["Average child mortality rate"]
)

print("\nPipeline ejecutado correctamente.")
print(f"Archivo generado: {archivo_salida}")