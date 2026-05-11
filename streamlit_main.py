import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc


st.set_page_config(
    page_title="Heart Failure",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- DATOS Y PREPROCESADO ---
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")


@st.cache_data
def engineer_variables(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.copy()

    df_proc["log_creatinine_phosphokinase"] = np.log(df_proc["creatinine_phosphokinase"])
    df_proc["is_old"] = (df_proc["age"] > 60) * 1
    df_proc["low_ef"] = (df_proc["ejection_fraction"] < 35) * 1
    df_proc["high_creatinine"] = (df_proc["serum_creatinine"] > 1.2) * 1
    df_proc["low_platelets"] = (df_proc["platelets"] < 150) * 1
    df_proc["interaction_age_ef"] = df_proc["age"] * df_proc["ejection_fraction"]
    df_proc["hypertension_diabetes"] = (
        (df_proc["high_blood_pressure"] == 1) & (df_proc["diabetes"] == 1)
    ) * 1
    df_proc["risk_score"] = (
        df_proc["is_old"].astype(int)
        + df_proc["low_ef"].astype(int)
        + df_proc["high_creatinine"].astype(int)
        + df_proc["anaemia"].astype(int)
        + df_proc["smoking"].astype(int)
        + df_proc["diabetes"].astype(int)
        + df_proc["high_blood_pressure"].astype(int)
    )
    df_proc["platelets_to_creatinine_ratio"] = df_proc["platelets"] / (
        df_proc["serum_creatinine"] + 0.1
    )
    return df_proc


@st.cache_data
def get_file_content(path: str) -> bytes:
    file_path = Path(path)
    return file_path.read_bytes()


# --- ENTRENAMIENTO Y RESULTADOS ---
@st.cache_resource(show_spinner=False)
def run_full_pipeline(df_proc: pd.DataFrame):
    
    varc = [
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
        "time",
        "log_creatinine_phosphokinase",
        "is_old",
        "low_ef",
        "high_creatinine",
        "low_platelets",
        "interaction_age_ef",
        "hypertension_diabetes",
        "risk_score",
        "platelets_to_creatinine_ratio",
    ]
    target = ["DEATH_EVENT"]

    X = df_proc.drop(columns="DEATH_EVENT")
    y = df_proc["DEATH_EVENT"]

    # Selección de variables
    kb = SelectKBest(k=5, score_func=f_regression)
    kb.fit(X, y)
    ls_best = [x for x, y_flag in zip(X.columns, kb.get_support()) if y_flag]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=150)

    # Modelos base
    logreg = LogisticRegression()
    logreg_scores = cross_val_score(
        cv=4, estimator=logreg, X=X_train, y=y_train, n_jobs=-1, scoring="r2"
    )

    model_lasso = Lasso(alpha=0.001)
    lasso_scores = cross_val_score(
        estimator=model_lasso, X=X_train, y=y_train, cv=4, n_jobs=-1, scoring="r2"
    )

    model_ridge = Ridge(alpha=0.001)
    ridge_scores = cross_val_score(
        estimator=model_ridge, X=X_train, y=y_train, cv=4, n_jobs=-1, scoring="r2"
    )

    model_elastic = ElasticNet(alpha=0.001)
    elastic_scores = cross_val_score(
        estimator=model_elastic,
        X=X_train,
        y=y_train,
        cv=4,
        n_jobs=-1,
        scoring="r2",
    )

    model = RandomForestClassifier(random_state=42)
    rf_scores = cross_val_score(model, X=X_train, y=y_train, cv=4, scoring="roc_auc")

    # Hiperparametrización (manteniendo búsqueda aleatoria)
    param_dict = {
        "n_estimators": [x for x in range(100, 1500, 100)],
        "max_features": ["auto", "sqrt", "log2"],
        "criterion": ["gini", "entropy"],
        "class_weight": ["balanced", None],
        "min_samples_split": [x for x in range(2, 50, 2)],
        "min_samples_leaf": [x / 100 for x in range(5, 55, 5)],
    }

    search = RandomizedSearchCV(
        param_distributions=param_dict,
        cv=4,
        n_jobs=-1,
        scoring="accuracy",
        estimator=model,
        verbose=0,
        n_iter=10,
    )
    search.fit(X_train, y_train)
    best_search_score = search.best_score_

    # Entrenamiento final
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    clf_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc_curve = auc(fpr, tpr)

    return {
        "varc": varc,
        "target": target,
        "ls_best": ls_best,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_shape": y_train.shape,
        "y_test_shape": y_test.shape,
        "logreg_scores": logreg_scores,
        "lasso_scores": lasso_scores,
        "ridge_scores": ridge_scores,
        "elastic_scores": elastic_scores,
        "rf_scores": rf_scores,
        "best_search_score": best_search_score,
        "clf_report": clf_report,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc_curve": roc_auc_curve,
    }


# --- UI ---
def main():
    st.title("Predicciones de Insuficiencia Cardíaca")

    df = load_data()
    df_proc = engineer_variables(df)
    results = run_full_pipeline(df_proc)

    st.sidebar.header("Navegación")

    section = st.sidebar.radio(
        "Ir a",
        [
            "Inicio",
            "Lectura de Datos",
            "Diccionario de datos",
            "Exploración / Limpieza",
            "Ingeniería de variables",
            "Análisis Exploratorio",
            "Modelos",
            "Resultados del modelo",
            "Recursos y Descargas",
        ],
    )

    if section == "Inicio":
        st.subheader("Bienvenido")
        st.markdown(
            """
            Esta aplicación presenta una versión **interactiva** del análisis de insuficiencia cardíaca.
            El código y el desarrollo detallado se encuentran en los archivos descargables (.pdf y .ipynb).
            Usa el menú lateral para navegar por las secciones del flujo analítico original.
            """
        )

        st.subheader("Introducción")
        st.markdown(
            """
            La insuficiencia cardíaca (Heart Failure) es una de las condiciones de
            salud más importantes a nivel mundial debido a su alta prevalencia,
            mortalidad, carga económica y su impacto en la calidad de vida de los
            pacientes.

            Se estima que más de 64 millones de personas en el mundo viven con
            insuficiencia cardíaca (según datos de la Sociedad Europea de Cardiología
            y la OMS). Es una condición crónica y progresiva que lleva al deterioro
            funcional y puede terminar en muerte prematura si no se trata
            adecuadamente.

            Tiene una tasa de supervivencia a 5 años que oscila entre el 25% y el 50%,
            dependiendo del estado y tratamiento. En países desarrollados, representa
            entre 1% y 2% del gasto total en salud.

            La insuficiencia cardíaca se ha convertido en un foco prioritario para las
            guías clínicas internacionales, que recomiendan estrategias de prevención,
            diagnóstico temprano y manejo integral.

            El objetivo de este proyecto final es que mediante el dataset de Heart
            Failure podamos realizar predicciones y estimaciones sobre qué factores
            pueden estar relacionados a la muerte de personas con insuficiencia
            cardíaca.
            """
        )

    elif section == "Lectura de Datos":
        st.subheader("Lectura de Datos")
        st.write("Vista inicial del dataset:")
        st.dataframe(df.head())

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Filas", df.shape[0])
        with col_b:
            st.metric("Columnas", df.shape[1])

    elif section == "Diccionario de datos":
        st.subheader("Diccionario de datos")
        st.markdown(
            """
            **Objetivo:** Identificar factores asociados a la mortalidad en insuficiencia cardíaca.

            - **age**: Edad (años)
            - **anaemia**: Anemia (booleano)
            - **creatinine_phosphokinase**: Nivel de CPK en sangre (mcg/L)
            - **diabetes**: Diabetes (booleano)
            - **ejection_fraction**: Fracción de eyección (%)
            - **high_blood_pressure**: Hipertensión (booleano)
            - **platelets**: Plaquetas (kiloplaquetas/mL)
            - **serum_creatinine**: Creatinina sérica (mg/dL)
            - **serum_sodium**: Sodio sérico (mEq/L)
            - **sex**: Sexo (binario)
            - **smoking**: Fumador (booleano)
            - **time**: Seguimiento (días)
            - **DEATH_EVENT**: Evento de muerte durante seguimiento (booleano)
            """
        )

    elif section == "Exploración / Limpieza":
        st.subheader("Exploración / Inspección / Limpieza")
        na_counts = df.isna().sum()
        st.write("Valores faltantes (conteo y proporción):")
        na_df = pd.DataFrame({"nulos": na_counts, "proporción": na_counts / len(df)})
        st.dataframe(na_df)

        st.write("Distribución de la variable objetivo:")
        st.dataframe(
            pd.DataFrame({
                "conteo": df["DEATH_EVENT"].value_counts(),
                "porcentaje": df["DEATH_EVENT"].value_counts(normalize=True) * 100,
            })
        )

        st.markdown("#### Distribución de todas las variables")
        with st.expander("Ver frecuencias relativas"):
            for col in df.columns.tolist() + [["DEATH_EVENT"]]:
                if isinstance(col, list):
                    col_name = col[0]
                    data = df[col_name]
                else:
                    col_name = col
                    data = df[col_name]
                st.write(f"**{col_name}**")
                st.dataframe(data.value_counts(normalize=True).reset_index())

        st.info(
            "Los datos se observan limpios (sin nulos ni atípicos relevantes)"
        )

    elif section == "Ingeniería de variables":
        st.subheader("Ingeniería de variables")
        st.write("Se aplican las transformaciones definidas en el análisis:")
        st.code(
            """
            log_creatinine_phosphokinase = log(creatinine_phosphokinase)
            is_old = age > 60
            low_ef = ejection_fraction < 35
            high_creatinine = serum_creatinine > 1.2
            low_platelets = platelets < 150
            interaction_age_ef = age * ejection_fraction
            hypertension_diabetes = high_blood_pressure == 1 & diabetes == 1
            risk_score = suma de indicadores clínicos
            platelets_to_creatinine_ratio = platelets / (serum_creatinine + 0.1)
            """,
            language="python",
        )

        st.dataframe(df_proc.head())
        st.caption("Dtypes tras ingeniería:")
        st.dataframe(df_proc.dtypes.to_frame("dtype"))

    elif section == "Análisis Exploratorio":
        st.subheader("Análisis Exploratorio")
        st.write("Columnas presentes:")
        st.code(str(df_proc.columns.tolist()))

        st.write("Descripción estadística:")
        st.dataframe(df_proc.describe())

        st.write("Variables consideradas (varc):")
        st.code(str(results["varc"]))

        st.markdown("### Gráficos interactivos")
        numeric_cols = [
            c for c in df_proc.select_dtypes(include=[np.number]).columns if c != "DEATH_EVENT"
        ]

        if numeric_cols:
            col_a, col_b = st.columns(2)

            with col_a:
                hist_col = st.selectbox(
                    "Variable numérica para histograma",
                    options=numeric_cols,
                    index=0,
                )
                fig_hist = px.histogram(
                    df_proc,
                    x=hist_col,
                    color="DEATH_EVENT",
                    nbins=30,
                    barmode="overlay",
                    opacity=0.7,
                    marginal="box",
                    title=f"Distribución de {hist_col} por DEATH_EVENT",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_b:
                x_scatter = st.selectbox(
                    "Eje X (dispersión)",
                    options=numeric_cols,
                    index=0,
                )
                y_scatter_options = [c for c in numeric_cols if c != x_scatter]
                y_scatter = st.selectbox(
                    "Eje Y (dispersión)",
                    options=y_scatter_options or numeric_cols,
                    index=0,
                )
                fig_scatter = px.scatter(
                    df_proc,
                    x=x_scatter,
                    y=y_scatter,
                    color="DEATH_EVENT",
                    hover_data=["age", "ejection_fraction", "serum_creatinine", "time"],
                    trendline="ols",
                    title="Relaciones entre variables clínicas",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    elif section == "Modelos":
        st.subheader("Modelos")
        st.write("Selección de variables con SelectKBest (k=5, f_regression):")
        st.success(f"Features seleccionadas: {results['ls_best']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tamaño X_train", f"{results['X_train_shape']}")
            st.metric("Tamaño X_test", f"{results['X_test_shape']}")
        with col2:
            st.metric("Tamaño y_train", f"{results['y_train_shape']}")
            st.metric("Tamaño y_test", f"{results['y_test_shape']}")

        st.markdown("### Métricas de validación cruzada (r2 / AUC)")
        st.write(
            f"Regresión Logística (r2): media={np.mean(results['logreg_scores']):.4f}, std={np.std(results['logreg_scores']):.4f}"
        )
        st.write(
            f"Lasso (r2): media={np.mean(results['lasso_scores']):.4f}, std={np.std(results['lasso_scores']):.4f}"
        )
        st.write(
            f"Ridge (r2): media={np.mean(results['ridge_scores']):.4f}, std={np.std(results['ridge_scores']):.4f}"
        )
        st.write(
            f"ElasticNet (r2): media={np.mean(results['elastic_scores']):.4f}, std={np.std(results['elastic_scores']):.4f}"
        )
        st.write(
            f"Random Forest (roc_auc): media={np.mean(results['rf_scores']):.4f}, std={np.std(results['rf_scores']):.4f}"
        )

        st.markdown("### Hiperparametrización (RandomizedSearchCV)")
        st.info(
            f"Mejor score en búsqueda aleatoria (accuracy, 4-fold, n_iter=10): {results['best_search_score']:.4f}."
        )
        st.caption("No conviene realizar hiperparametrización adicional.")

    elif section == "Resultados del modelo":
        st.subheader("Resultados del modelo Random Forest")
        st.text("Clasificación:")
        st.code(results["clf_report"], language="text")
        st.metric("ROC AUC", f"{results['roc_auc']:.4f}")

        st.markdown("### Curva ROC")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(results["fpr"], results["tpr"], color="darkorange", lw=2, label=f"ROC curve (AUC = {results['roc_auc_curve']:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC - Modelo Random Forest")
        ax.legend(loc="lower right")
        ax.grid(False)
        st.pyplot(fig)

    elif section == "Recursos y Descargas":
        st.subheader("Recursos y Descargas")
        st.markdown(
            "Descarga el reporte y el notebook con el desarrollo completo del proyecto."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.info("Reporte detallado (PDF)")
            try:
                pdf_bytes = get_file_content("Proyecto Final.pdf")
                st.download_button(
                    label="📄 Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name="Analisis_Insuficiencia_Cardiaca.pdf",
                    mime="application/pdf",
                )
            except FileNotFoundError:
                st.error("Archivo PDF no encontrado en el repositorio.")

        with col2:
            st.info("Código fuente (Jupyter Notebook)")
            try:
                notebook_bytes = get_file_content("Proyecto Final.ipynb")
                st.download_button(
                    label="📓 Descargar Notebook .ipynb",
                    data=notebook_bytes,
                    file_name="Code_Heart_Failure.ipynb",
                    mime="application/x-ipynb+json",
                )
            except FileNotFoundError:
                st.error("Archivo .ipynb no encontrado en el repositorio.")

    st.sidebar.caption("Desarrollado por Juan Alexis Ramos Palacios")

if __name__ == "__main__":
    main()