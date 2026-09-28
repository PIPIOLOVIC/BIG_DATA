# Práctica: U3_1_modelos_ml — Comparación de Modelos de Regresión

**Materia:** Tópicos de Big Data  
**Tema:** Introducción al aprendizaje supervisado mediante modelos de regresión  
**Herramientas:** Python · Jupyter Notebook · Pandas · NumPy · Matplotlib · Scikit-learn  

---

## 📁 Estructura del Proyecto

Conforme a las instrucciones de la práctica, la estructura de carpetas es la siguiente:

```text
U3_1_modelos_ml/
├── README.md
├── data/
│   └── births-and-deaths-projected-to-2100.csv       # Dataset demográfico histórico y proyecciones
├── notebook/
│   ├── births-and-deaths-projected-to-2100.csv       # Copia local de respaldo para ejecución sin conexión
│   ├── U3_1_modelos_regresion_demografia.ipynb      # Notebook completo (base de México + Partes 19-21 con los 2 ejercicios)
│   └── U3_1_ejercicios_extracciones.ipynb           # Notebook dedicado a los ejercicios de extracción (China y México >= 2000)
└── source/
    └── modelos_regresion.py                          # Pipeline modular en Python para México, China y México >= 2000
```

---

## 🎯 Objetivo de la Práctica y Ejercicios

Construir, comparar visualmente y evaluar tres modelos de aprendizaje supervisado (**Regresión Lineal Simple**, **Regresión Polinomial de Grado 2** y **Random Forest Regressor**) para proyectar **nacimientos** y **defunciones** hacia el año 2100:

1. **Práctica Base:** México con datos históricos completos (1950–2023).
2. **Ejercicio 1:** Extracción del país de **China** (1950–2023).
3. **Ejercicio 2:** Extracción de **México a partir del año 2000** (2000–2023).

---

## 📊 Hallazgos y Comparativa de Modelos

### 1. Resumen por Modelo y Experimento (Nacimientos hacia 2100)

| Experimento / Extracción | Referencia OWID (2100) | Regresión Lineal (2100) | Regresión Polinomial (2100) | Random Forest (2100) |
|---|---|---|---|---|
| **México Completo (1950–2023)** | ~1.10M | ~3.14M (Sube erróneamente) | ~ -4.40M (Colapso negativo) | ~2.05M (Meseta plana) |
| **Ejercicio 1: China (1950–2023)** | ~3.10M | ~352 mil (Fuerte descenso) | ~ -48.33M (Colapso severo) | ~9.49M (Meseta plana) |
| **Ejercicio 2: México (2000–2023)** | ~1.10M | **~934 mil (¡Casi idéntico a OWID!)** | **~905 mil (Estable)** | ~2.05M (Meseta plana) |

### 2. Hallazgo Clave del Ejercicio 2 (México post-2000 vs 1950)
- **1950–2023:** La tasa de natalidad creció de 1950 a 1990 y luego cayó, provocando que la Regresión Lineal aprendiera una pendiente positiva artificial ($+7,835$ nacimientos/año), proyectando más de 3.1 millones de nacimientos.
- **2000–2023:** Al delimitar la ventana histórica a la fase post-transición demográfica (a partir del 2000), la tendencia de natalidad es estrictamente decreciente ($-14,819$ nacimientos/año). Gracias a ello, la **Regresión Lineal simple converge notablemente con las proyecciones multivariadas de la ONU/OWID** ($1.67\text{M}$ vs $1.64\text{M}$ en 2050, y $934\text{K}$ vs $1.09\text{M}$ en 2100).

---

## 🚀 Instrucciones de Ejecución

### 1. Requisitos previos
Instalar dependencias necesarias:
```bash
pip install pandas numpy matplotlib scikit-learn jinja2
```

### 2. Ejecutar los Notebooks
Puedes abrir y visualizar directamente cualquiera de los dos notebooks pre-ejecutados:
- **Notebook dedicado a los ejercicios:**
  ```bash
  jupyter notebook notebook/U3_1_ejercicios_extracciones.ipynb
  ```
- **Notebook completo:**
  ```bash
  jupyter notebook notebook/U3_1_modelos_regresion_demografia.ipynb
  ```

### 3. Ejecutar el script modular
Ejecuta los análisis para México, China y México >= 2000 en la terminal:
```bash
python source/modelos_regresion.py
```
