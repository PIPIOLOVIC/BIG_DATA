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
│   └── U3_1_modelos_regresion_demografia.ipynb      # Notebook resuelto y ejecutado con todas las respuestas y visualizaciones
└── source/
    └── modelos_regresion.py                          # Pipeline modular en Python para entrenamiento y proyecciones
```

---

## 🎯 Objetivo de la Práctica

Construir y comparar visualmente tres modelos de aprendizaje supervisado para proyectar el número de **nacimientos** y **defunciones** en México hacia el año 2100 utilizando datos históricos (1950–2023):

1. **Regresión Lineal Simple** ($y = ax + b$)
2. **Regresión Polinomial de Grado 2** ($y = ax^2 + bx + c$)
3. **Random Forest Regressor** (Ensamble de 100 árboles de decisión)

---

## 📊 Hallazgos y Comparativa de Modelos

| Modelo | Comportamiento en Nacimientos (hacia 2100) | Comportamiento en Defunciones (hacia 2100) | Capacidad de Extrapolación |
|---|---|---|---|
| **Regresión Lineal** | Proyecta un aumento indefinido (~3.14M), ignorando la inflexión demográfica desde el 2000. | Proyecta un aumento lineal continuo (~981 mil), más conservador que OWID. | Fija y rígida; asume tasa de cambio constante. |
| **Regresión Polinomial (Grado 2)** | Se ajusta muy bien a los datos históricos, pero se desploma a valores negativos absurdos (~ -4.40M en 2100). | Se ajusta a la aceleración histórica y se dispara de forma agresiva (~3.51M en 2100). | Peligrosa y divergente fuera del intervalo de entrenamiento. |
| **Random Forest** | Excelente ajuste histórico, pero a partir de 2024 genera una línea horizontal plana (~2.05M). | Excelente ajuste histórico, pero a partir de 2024 se estanca en una meseta constante (~869 mil). | Nula; los árboles de decisión no extrapolan fuera de sus hojas observadas. |
| **OWID (ONU - Referencia)** | Descenso suave y paulatino hacia ~1.10M en 2100 basado en cohortes y fecundidad. | Incremento gradual por envejecimiento hacia ~1.79M en 2100. | Basada en dinámica demográfica multivariada. |

---

## 🚀 Instrucciones de Ejecución

### 1. Requisitos previos
Instalar dependencias necesarias:
```bash
pip install pandas numpy matplotlib scikit-learn jinja2
```

### 2. Ejecutar el Notebook
Abrir Jupyter Notebook o VS Code y ejecutar:
```bash
jupyter notebook notebook/U3_1_modelos_regresion_demografia.ipynb
```

### 3. Ejecutar el script modular
```bash
python source/modelos_regresion.py
```
