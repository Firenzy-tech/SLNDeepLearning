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
    series = series.dropna()
    if series.empty:
        raise ValueError(f"La columna {column} no tiene valores numéricos válidos para graficar.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(x=series, kde=True, ax=ax, color='teal')
    ax.set_title(f'Distribución de {column}')
    plt.tight_layout()
    return fig

def plot_relational_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    alpha: float = 0.5,
):
    """Genera un scatter relacional estilo seaborn relplot."""
    plot_df = df.copy()

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

    required_columns = [x_col, y_col]
    if size_col:
        plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors='coerce')
        required_columns.append(size_col)

    plot_df = plot_df.dropna(subset=required_columns)

    if plot_df.empty:
        raise ValueError("No hay suficientes datos numéricos válidos para generar la gráfica relacional.")

    g = sns.relplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        size=size_col,
        sizes=(40, 400) if size_col else None,
        alpha=alpha,
        palette='muted',
        height=6,
    )

    g.set_axis_labels(x_col, y_col)
    g.fig.suptitle(f"{y_col} vs {x_col}", y=1.02)
    return g.fig

def plot_categorical_swarm(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
):
    """Genera un swarmplot categórico estilo seaborn."""
    plot_df = df.copy()

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar la gráfica categórica.")

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.swarmplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, ax=ax)

    ax.set_xlabel(x_col)
    ax.set_ylabel("")
    ax.set_title(f"{y_col} por {x_col}")

    if hue_col:
        ax.legend(title=hue_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    return fig

def plot_scatter_semantics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    hue_order: list[str] | None = None,
):
    """Genera un scatterplot tipo seaborn con color y tamaño por variables."""
    plot_df = df.copy()

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

    required_columns = [x_col, y_col]
    if size_col:
        plot_df[size_col] = pd.to_numeric(plot_df[size_col], errors='coerce')
        required_columns.append(size_col)

    plot_df = plot_df.dropna(subset=required_columns)

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar el scatterplot.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(fig, left=True, bottom=True)

    scatter_kwargs = {
        "x": x_col,
        "y": y_col,
        "hue": hue_col,
        "size": size_col,
        "sizes": (1, 8) if size_col else None,
        "linewidth": 0,
        "data": plot_df,
        "ax": ax,
    }
    if hue_order:
        scatter_kwargs["hue_order"] = hue_order

    sns.scatterplot(**scatter_kwargs)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig

def plot_lineplot_semantics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    style_col: str | None = None,
):
    """Genera un lineplot estilo seaborn con hue y style."""
    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

    required_columns = [x_col, y_col]
    plot_df = plot_df.dropna(subset=required_columns)

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar el lineplot.")

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, style=style_col, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig

def plot_facet_histogram(
    df: pd.DataFrame,
    x_col: str,
    row_col: str | None = None,
    col_col: str | None = None,
    binwidth: float = 3,
):
    """Genera histogramas facetados estilo seaborn displot."""
    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')

    facet_columns = [column for column in [row_col, col_col] if column]
    plot_df = plot_df.dropna(subset=[x_col] + facet_columns)

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar los histogramas facetados.")

    sns.set_theme(style="darkgrid")
    g = sns.displot(
        data=plot_df,
        x=x_col,
        row=row_col,
        col=col_col,
        binwidth=binwidth,
        height=3,
        facet_kws=dict(margin_titles=True),
    )
    g.fig.suptitle(f"Distribución de {x_col}", y=1.02)
    return g.fig

def plot_facet_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
    size_col: str | None = None,
    col_col: str | None = None,
    size_order: list[str] | None = None,
):
    """Genera un relplot tipo line con facetas."""
    plot_df = df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

    required_columns = [x_col, y_col]
    if size_col:
        required_columns.append(size_col)
    if col_col:
        required_columns.append(col_col)
    if hue_col:
        required_columns.append(hue_col)

    plot_df = plot_df.dropna(subset=required_columns)

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar el lineplot facetado.")

    sns.set_theme(style="ticks")
    palette = sns.color_palette("rocket_r")
    g = sns.relplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        size=size_col,
        col=col_col,
        kind="line",
        palette=palette,
        height=5,
        aspect=.9,
        facet_kws=dict(sharex=False),
        size_order=size_order,
    )
    g.fig.suptitle(f"{y_col} vs {x_col}", y=1.02)
    return g.fig

def plot_grouped_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str | None = None,
):
    """Genera un boxplot agrupado estilo seaborn."""
    plot_df = df.copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[x_col, y_col])

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar el boxplot.")

    sns.set_theme(style="ticks", palette="pastel")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=x_col, y=y_col, hue=hue_col, data=plot_df, ax=ax)
    sns.despine(offset=10, trim=True)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    return fig

def plot_diagonal_correlation_matrix(df: pd.DataFrame):
    """Genera una matriz de correlación con triángulo inferior visible."""
    num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] == 0:
        raise ValueError("No hay columnas numéricas para calcular la correlación.")

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=.3,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        ax=ax,
    )
    plt.tight_layout()
    return fig

def plot_horizontal_barplot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    compare_col: str | None = None,
):
    """Genera un barplot horizontal estilo seaborn."""
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors='coerce')
    if compare_col:
        plot_df[compare_col] = pd.to_numeric(plot_df[compare_col], errors='coerce')

    required_columns = [category_col, value_col]
    if compare_col:
        required_columns.append(compare_col)

    plot_df = plot_df.dropna(subset=required_columns)

    if plot_df.empty:
        raise ValueError("No hay datos válidos para generar el barplot horizontal.")

    plot_df = plot_df.sort_values(value_col, ascending=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    if compare_col:
        sns.barplot(data=plot_df, x=value_col, y=category_col, hue=compare_col, ax=ax)
    else:
        sns.barplot(data=plot_df, x=value_col, y=category_col, ax=ax, color="b")
    sns.despine(left=True, bottom=True)
    ax.set_xlabel(value_col)
    ax.set_ylabel(category_col)
    plt.tight_layout()
    return fig