# Insuficiencia Cardíaca – App Interactiva

Aplicación Streamlit para explorar el dataset de **Heart Failure Clinical Records**, realizar ingeniería de variables, entrenar modelos clásicos (Logistic Regression, Lasso, Ridge, ElasticNet, Random Forest) y visualizar resultados con gráficos interactivos.

## Requisitos

- Python 3.9+
- Dependencias principales (ver `requirements.txt`): `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `plotly`

## Estructura

- `streamlit_main.py`: aplicación interactiva (flujo completo del análisis).
- `heart_failure_clinical_records_dataset.csv`: dataset usado en el análisis.
- `Proyecto Final.pdf`: reporte detallado para descarga desde la app.
- `Proyecto Final.ipynb`: notebook original para descarga desde la app.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run streamlit_main.py
```

## Flujo en la app

1) **Inicio**: contexto del proyecto y enlaces a descargas.
2) **Lectura de Datos**: vista rápida del dataset y métricas básicas.
3) **Diccionario de datos**: descripción de variables.
4) **Exploración / Limpieza**: nulos, proporciones de la variable objetivo y frecuencias.
5) **Ingeniería de variables**: transformaciones clínicas (log, umbrales, interacciones, score de riesgo).
6) **Análisis Exploratorio**: estadísticas y **gráficos interactivos** (histograma/box combinado y dispersión con trendline por clase).
7) **Modelos**: selección de variables (SelectKBest), métricas de validación cruzada y nota sobre hiperparámetros.
8) **Resultados del modelo**: reporte de clasificación, ROC AUC y curva ROC del Random Forest.
9) **Recursos y Descargas**: botones para descargar el PDF y el notebook.
