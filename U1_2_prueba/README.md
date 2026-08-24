# Práctica 1.1 — Probabilidad y Estadística
## Análisis de la evolución poblacional mundial

Este repositorio contiene la solución de la **Práctica 1.1 — Probabilidad y Estadística: Análisis de la evolución poblacional mundial**.

---

### Descripción del Proyecto
El propósito de esta práctica es familiarizarse con un conjunto de datos real sobre la evolución de la población mundial, comprobar el entorno de trabajo con Python, Pandas y Jupyter Notebook, y aplicar medidas estadísticas descriptivas básicas.

---

### Fuente de los Datos
- **Fuente original:** United Nations, *World Population Prospects (2024)*.
- **Procesamiento y publicación:** Our World in Data (OWID).
- **Cita bibliográfica:**
  > UN, World Population Prospects (2024) – processed by Our World in Data. “Annual change in population – UN WPP” [dataset]. United Nations, “World Population Prospects”; United Nations, “World Population Prospects - Interim Update” [original data].

---

### Estructura del Repositorio

```text
.
├── data/
│   └── raw/
│       ├── annual-population-growth/
│       │   ├── annual-population-growth.csv
│       │   ├── annual-population-growth.metadata.json
│       │   └── readme.md
│       └── births-and-deaths-projected-to-2100/
│           ├── births-and-deaths-projected-to-2100.csv
│           ├── births-and-deaths-projected-to-2100.metadata.json
│           └── readme.md
├── notebooks/
│   └── 01_evolucion_poblacional.ipynb
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

---

### Requisitos e Instalación

1. **Crear y activar el entorno virtual:**
   ```bash
   # En Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Abrir y ejecutar el notebook ubicado en [notebooks/01_evolucion_poblacional.ipynb](notebooks/01_evolucion_poblacional.ipynb).

---

### Actividades Desarrolladas
- **Actividad 1 — Identificación de la fuente:** Reconocimiento y citación formal de la fuente de datos y descripción de las variables.
- **Actividad 2 — Carga y exploración de datos:** Carga del archivo CSV, dimensiones, tipos de datos, valores faltantes y conteo de entidades.
- **Actividad 3 — Estadística descriptiva:** Filtrado de datos para el año histórico 2020 y cálculo de estadísticos (media, mediana, varianza, desviación estándar, mínimo y máximo).
- **Actividad 4 — Comprobación del entorno:** Verificación de las versiones de Python, Pandas y NumPy.
- **Resultados:** Resumen de los hallazgos y comprobación del flujo de trabajo reproducible.
