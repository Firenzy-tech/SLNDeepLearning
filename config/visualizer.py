"""Utilidades de visualizacion para exploracion y evaluacion del proyecto."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    @staticmethod
    def plot_generic_scatter(df, x_col, y_col, hue_col, figsize=(10, 4)):
        """
        Genera un grafico de dispersion clasico con Matplotlib/Seaborn.

        Se usa como apoyo cuando se quiere una visualizacion rapida y estable,
        mientras que la UI principal favorece graficos interactivos con Plotly.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Se construye de forma dinamica para reutilizar la misma funcion con cualquier dataset.
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        
        ax.set_title(f'Relación: {x_col} y {y_col}')
        plt.tight_layout()
        
        return fig

    @staticmethod
    def plot_interactive_loss(history):
        """Muestra la evolucion de la perdida durante el entrenamiento."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', mode='lines+markers'))
        fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', mode='lines+markers'))
        fig.update_layout(
            title='Curva de Aprendizaje (Pérdida)',
            xaxis_title='Épocas',
            yaxis_title='Loss',
            template='plotly_white',
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def plot_interactive_corr(df):
        """Genera una matriz de correlacion interactiva con Plotly."""
        corr = df.corr()
        fig = px.imshow(
            corr, 
            text_auto='.2f', 
            aspect="auto", 
            color_continuous_scale='RdBu_r',
            title="Matriz de Correlación Interactiva"
        )
        return fig

    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Muestra la importancia de variables para modelos que la expongan."""
        importances = model.feature_importances_
        data = {"Feature": feature_names, "Importance": importances}
        df_importance = pd.DataFrame(data).sort_values(by="Importance", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_importance.head(15), x="Importance", y="Feature", ax=ax, palette="viridis")
        ax.set_title("Top 15 - Importancia de las Variables")
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None):
        """Construye la matriz de confusion usada para evaluar errores."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", ax=ax, values_format='d')
        ax.set_title("Matriz de Confusión")
        return fig

    @staticmethod
    def plot_distribution(df, column):
        """Muestra histograma y boxplot para revisar distribucion y outliers."""
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(8, 5))
        
        sns.boxplot(data=df, x=column, ax=ax_box, color="skyblue")
        sns.histplot(data=df, x=column, ax=ax_hist, kde=True, color="skyblue")
        
        ax_box.set(xlabel='')
        ax_hist.set_title(f"Distribución de: {column}")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_training_history(history):
        """Grafica precision y perdida si se prefiere una vista estandar."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title('Precisión'); ax1.legend()
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title('Pérdida'); ax2.legend()
        plt.tight_layout()
        return fig