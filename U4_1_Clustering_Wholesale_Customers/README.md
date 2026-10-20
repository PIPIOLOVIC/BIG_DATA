# Práctica: U4_1_Clustering_Wholesale_Customers — Segmentación de Clientes con Aprendizaje No Supervisado

**Materia:** Tópicos de Big Data  
**Tema:** Segmentación de clientes mediante Clustering (K-Means, PCA y KNN)  
**Herramientas:** Python · Jupyter Notebook · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn  

---

## 📁 Estructura del Proyecto

Conforme a la estructura del repositorio de la materia, la organización de carpetas es la siguiente:

```text
U4_1_Clustering_Wholesale_Customers/
├── README.md
├── requirements.txt
├── .venv/                                          # Entorno virtual aislado de Python
├── data/
│   └── Wholesale customers data.csv               # Dataset descargado desde UCI Machine Learning Repository
├── notebook/
│   ├── Wholesale customers data.csv               # Copia local de respaldo para ejecución sin conexión
│   └── U4_1_Clustering_Wholesale_Customers.ipynb   # Notebook interactivo con exploración, visualizaciones y modelos
└── source/
    └── clustering_wholesale.py                    # Pipeline modular en Python para preprocesamiento, K-Means, PCA y KNN
```

---

## 🎯 Objetivo de la Práctica

El objetivo de esta práctica es implementar el ciclo completo de **aprendizaje no supervisado** para segmentar y perfilar clientes de un distribuidor mayorista en función de sus patrones anuales de gasto en diversas categorías de productos:

1. **Exploración y Calidad de Datos:** Identificar dimensiones, tipos de datos, valores nulos y duplicados sobre el dataset *Wholesale Customers* del repositorio UCI (440 clientes y 8 variables).
2. **Preprocesamiento:** Transformar variables con distribuciones de cola pesada mediante logaritmo natural ($\log(1+x)$) y estandarizar sus escalas con `StandardScaler`.
3. **Selección del Número Óptimo de Grupos ($k$):** Evaluar el compromiso entre reducción de inercia (método del codo) y cohesión/separación con el **Silhouette Score** para $k \in [2, 8]$.
4. **Perfilamiento de Negocio:** Caracterizar los clusters resultantes con medianas en la escala monetaria original y perfiles relativos normalizados, correlacionándolos con variables contextuales como el canal de compra (`Channel`: Horeca vs Retail) y la región geográfica (`Region`).
5. **Reducción de Dimensionalidad (PCA):** Proyectar las 6 dimensiones de gasto a 2 componentes principales para inspeccionar visualmente la separación de los clusters en el plano.
6. **Búsqueda de Similitud (KNN):** Emplear `NearestNeighbors` con métrica euclidiana para consultar clientes análogos y habilitar recomendaciones o benchmarking de clientes similares.

---

## 📊 Hallazgos y Perfilamiento de Clientes

### 1. Selección de $k$ (Inercia vs Silhouette)

| $k$ | Inercia | Silhouette Score | Interpretación |
|---|---|---|---|
| **2** | 1844.06 | 0.2903 | Buena separación global pero divide de forma excesivamente genérica los patrones de compra. |
| **3** | **1553.42** | **0.2594** | **Configuración óptima elegida:** balance idóneo entre cohesión matemática e interpretabilidad de negocio. |
| **4** | 1386.86 | 0.1885 | Descenso marcado en Silhouette; subgrupos fragmentados con solapamiento. |
| **5** | 1266.15 | 0.1916 | Baja diferenciación entre perfiles medios. |
| **6** | 1174.76 | 0.2009 | Mayor complejidad sin ganancia sustancial en significado comercial. |
| **7** | 1086.36 | 0.1958 | Sobre-segmentación de clientes. |
| **8** | 1028.49 | 0.1852 | Dispersión excesiva de grupos. |

### 2. Resumen de Perfiles por Cluster ($k = 3$)

| Cluster | Clientes (%) | Canal Dominante | Productos de Mayor Gasto (Mediana) | Productos de Menor Gasto (Mediana) | Tipo de Cliente / Negocio |
|---|---|---|---|---|---|
| **Cluster 0** | 80 (18.2%) | Retail (60.0%) | **Grocery** (10,502), **Detergents_Paper** (4,218), **Milk** (6,300) | **Frozen** (376), **Delicassen** (434), **Fresh** (1,492) | **Supermercados y tiendas de conveniencia:** alto consumo en abarrotes y productos para el hogar/limpieza, mínima demanda de perecederos frescos o congelados. |
| **Cluster 1** | 147 (33.4%) | Retail (61.9%) / Horeca (38.1%) | **Fresh** (12,126), **Grocery** (9,965), **Milk** (7,184), **Delicassen** (2,005) | Ninguno (volumen elevado y balanceado en todos los rubros) | **Grandes distribuidores / Minoristas premium:** clientes de alto presupuesto con cestas de compra diversificadas y gran rotación de productos frescos y delicatessen. |
| **Cluster 2** | 213 (48.4%) | **Horeca (98.6%)** | **Fresh** (9,612), **Frozen** (2,121) | **Detergents_Paper** (274), **Milk** (1,601), **Grocery** (2,155) | **Hoteles, Restaurantes y Cafeterías (HORECA):** compran casi exclusivamente ingredientes frescos y congelados para cocina diaria; adquieren insumos de papel y limpieza por otros canales. |

### 3. Reducción con PCA
- **PC1 (44.08% de varianza explicada):** Captura principalmente la intensidad de compra en categorías secas y de retail (`Grocery`, `Detergents_Paper`, `Milk`).
- **PC2 (27.19% de varianza explicada):** Captura el consumo de alimentos perecederos y de cocina (`Fresh`, `Frozen`, `Delicassen`).
- **Total acumulado:** 71.27% de la varianza total explicada en solo dos componentes, confirmando una clara separación espacial entre el segmento HORECA y los segmentos Retail en el plano 2D.

### 4. Consultas con Nearest Neighbors (KNN)
- Permite identificar de manera instantánea clientes "vecinos" con estructuras de gasto equivalentes en el espacio estandarizado.
- Por ejemplo, para el Cliente `#0` (Canal Retail, Cluster 1), los vecinos más próximos (Clientes `#14`, `#59`, `#180`) comparten proporciones similares entre frescos y abarrotes, lo que posibilita estrategias de surtido y precios personalizados.

---

## 🚀 Instrucciones de Ejecución

### 1. Activar el Entorno Virtual (`.venv`)
El proyecto incluye su propio entorno virtual aislado configurado con todas las dependencias necesarias:

- **Windows (PowerShell):**
  ```powershell
  cd U4_1_Clustering_Wholesale_Customers
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  cd U4_1_Clustering_Wholesale_Customers
  .\.venv\Scripts\activate.bat
  ```
- **Instalar o reinstalar requerimientos si fuera necesario:**
  ```bash
  pip install -r requirements.txt
  ```

### 2. Ejecutar el Notebook
Abre y ejecuta el notebook interactivo con sus visualizaciones (gráficas de distribución, curvas de inercia/silhouette, mapa de calor y scatter plot PCA):
```bash
jupyter notebook notebook/U4_1_Clustering_Wholesale_Customers.ipynb
```

### 3. Ejecutar el Script Modular
Ejecuta el pipeline completo desde la terminal para obtener todas las métricas, perfiles de clusters y pruebas de consulta KNN:
```bash
python source/clustering_wholesale.py
```
