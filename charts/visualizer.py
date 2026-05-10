#Uso de matplotlib y seaborn, retornando figuras compatibles con st.pyplot().
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_matrix(df: pd.DataFrame):
    """Genera un Heatmap de correlación interactivo."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.tight_layout()
    return fig

def plot_distribution(df: pd.DataFrame, column: str):
    """Genera el histograma y KDE para una variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax, color='teal')
    ax.set_title(f'Distribución de {column}')
    return fig