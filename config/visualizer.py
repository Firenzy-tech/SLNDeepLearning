import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_generic_scatter(df, x_col, y_col, hue_col, figsize=(10, 4)):
        """
        Genera un gráfico de dispersión genérico basado en columnas específicas.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Replicamos la lógica solicitada de forma dinámica
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
        
        ax.set_title(f'Relación: {x_col} y {y_col}')
        plt.tight_layout()
        
        return fig