#Uso de matplotlib y seaborn, retornando figuras compatibles con st.pyplot().
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_correlation_matrix(df: pd.DataFrame):
    """Genera un Heatmap de correlación interactivo."""
    # Solo calcular correlación sobre columnas numéricas
    num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] == 0:
        raise ValueError("No hay columnas numéricas para calcular la correlación.")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.tight_layout()
    return fig

def plot_distribution(df: pd.DataFrame, column: str):
    """Genera el histograma y KDE para una variable."""
    # Preparar la serie limpiando NaNs e infs
    series = pd.to_numeric(df[column], errors='coerce')
    series = series.replace([pd.NA, pd.NaT], np.nan).dropna()
    if series.empty:
        raise ValueError(f"La columna {column} no tiene valores numéricos válidos para graficar.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(series, kde=True, ax=ax, color='teal')
    ax.set_title(f'Distribución de {column}')
    plt.tight_layout()
    return fig