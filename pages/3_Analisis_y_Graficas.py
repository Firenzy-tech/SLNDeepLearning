import sys
import os
from typing import cast
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from charts.visualizer import (
    plot_correlation_matrix,
    plot_distribution,
    plot_relational_scatter,
    plot_categorical_swarm,
    plot_scatter_semantics,
    plot_lineplot_semantics,
    plot_facet_histogram,
    plot_facet_lineplot,
    plot_grouped_boxplot,
    plot_diagonal_correlation_matrix,
    plot_horizontal_barplot,
)

st.header("3. Visualización y Análisis Estadístico")

if st.session_state.get('clean_data') is not None:
    df = st.session_state['clean_data']

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Distribuciones", "Correlaciones", "Relacional", "Categórica", "Scatter avanzado", "Ejemplos Seaborn"])

    # Debug opcional para ver estado de la sesión y estructura del dataframe
    with st.expander("🔍 Debug (mostrar estado y estructura)", expanded=False):
        st.write("`st.session_state` keys:", list(st.session_state.keys()))
        st.write("Columnas:", df.columns.tolist())
        st.write("Tipos:")
        st.write(df.dtypes)
        st.write("Primeras filas:")
        st.dataframe(df.head(5))

    with tab1:
        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        if not cols_num:
            st.info("No se encontraron columnas numéricas para visualizar. Revisa el dataset o aplica transformaciones en Limpieza.")
        else:
            col_dist = st.selectbox("Selecciona variable numérica", cols_num)
            if col_dist:
                try:
                    fig = plot_distribution(df, col_dist)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al generar la distribución: {e}")

    with tab2:
        st.write("Matriz de Correlación de Pearson")
        try:
            fig_corr = plot_correlation_matrix(df)
            st.pyplot(fig_corr)
        except Exception as e:
            st.error(f"Error al generar la matriz de correlación: {e}")

    with tab3:
        st.write("Gráfica relacional tipo seaborn relplot")

        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        cols_all = df.columns.tolist()

        if len(cols_num) < 2:
            st.info("Se necesitan al menos dos columnas numéricas para esta gráfica.")
        else:
            x_col = cast(str, st.selectbox("Variable eje X", cols_num, index=0, key="relplot_x"))
            remaining_y = [col for col in cols_num if col != x_col]
            if not remaining_y:
                st.info("Selecciona otra columna numérica para el eje Y.")
            else:
                y_col = cast(str, st.selectbox("Variable eje Y", remaining_y, index=0, key="relplot_y"))

                hue_candidates = ["Ninguna"] + [col for col in cols_all if col not in {x_col, y_col}]
                hue_col = cast(str, st.selectbox("Agrupar por color (hue)", hue_candidates, index=0, key="relplot_hue"))

                size_candidates = ["Ninguna"] + [col for col in cols_num if col not in {x_col, y_col}]
                size_col = cast(str, st.selectbox("Escalar por tamaño", size_candidates, index=0, key="relplot_size"))

                try:
                    fig_rel = plot_relational_scatter(
                        df,
                        x_col=x_col,
                        y_col=y_col,
                        hue_col=None if hue_col == "Ninguna" else hue_col,
                        size_col=None if size_col == "Ninguna" else size_col,
                    )
                    st.pyplot(fig_rel)
                except Exception as e:
                    st.error(f"Error al generar la gráfica relacional: {e}")

    with tab4:
        st.write("Gráfica categórica tipo seaborn swarmplot")

        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

        if not cols_num or not cols_cat:
            st.info("Se necesitan al menos una columna numérica y una categórica para esta gráfica.")
        else:
            x_col = cast(str, st.selectbox("Variable numérica", cols_num, index=0, key="swarm_x"))
            y_col = cast(str, st.selectbox("Variable categórica", cols_cat, index=0, key="swarm_y"))

            hue_candidates = ["Ninguna"] + [col for col in cols_cat if col != y_col]
            hue_col = cast(str, st.selectbox("Agrupar por color (hue)", hue_candidates, index=0, key="swarm_hue"))

            try:
                fig_swarm = plot_categorical_swarm(
                    df,
                    x_col=x_col,
                    y_col=y_col,
                    hue_col=None if hue_col == "Ninguna" else hue_col,
                )
                st.pyplot(fig_swarm)
            except Exception as e:
                st.error(f"Error al generar la gráfica categórica: {e}")

    with tab5:
        st.write("Scatterplot con color y tamaño estilo seaborn")

        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        cols_all = df.columns.tolist()

        if len(cols_num) < 2:
            st.info("Se necesitan al menos dos columnas numéricas para esta gráfica.")
        else:
            x_col = cast(str, st.selectbox("Variable eje X", cols_num, index=0, key="scatter_x"))
            remaining_y = [col for col in cols_num if col != x_col]
            if not remaining_y:
                st.info("Selecciona otra columna numérica para el eje Y.")
            else:
                y_col = cast(str, st.selectbox("Variable eje Y", remaining_y, index=0, key="scatter_y"))

                hue_candidates = ["Ninguna"] + [col for col in cols_all if col not in {x_col, y_col}]
                hue_col = cast(str, st.selectbox("Color por variable", hue_candidates, index=0, key="scatter_hue"))

                size_candidates = ["Ninguna"] + [col for col in cols_num if col not in {x_col, y_col}]
                size_col = cast(str, st.selectbox("Tamaño por variable", size_candidates, index=0, key="scatter_size"))

                hue_order_input = st.text_input(
                    "Orden de hue opcional (separado por comas)",
                    value="",
                    key="scatter_hue_order",
                )
                hue_order = [item.strip() for item in hue_order_input.split(",") if item.strip()] or None

                try:
                    fig_scatter = plot_scatter_semantics(
                        df,
                        x_col=x_col,
                        y_col=y_col,
                        hue_col=None if hue_col == "Ninguna" else hue_col,
                        size_col=None if size_col == "Ninguna" else size_col,
                        hue_order=hue_order,
                    )
                    st.pyplot(fig_scatter)
                except Exception as e:
                    st.error(f"Error al generar el scatterplot: {e}")

    with tab6:
        example_tab1, example_tab2, example_tab3, example_tab4, example_tab5, example_tab6 = st.tabs([
            "Lineplot",
            "Histogramas facetados",
            "Lineplot facetado",
            "Boxplot agrupado",
            "Correlación diagonal",
            "Barplot horizontal",
        ])

        with example_tab1:
            st.caption("Basado en el ejemplo de series temporales con hue y style.")
            cols_num = df.select_dtypes(include=['number']).columns.tolist()
            cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

            if len(cols_num) < 2:
                st.info("Se necesitan al menos dos columnas numéricas.")
            else:
                x_col = cast(str, st.selectbox("Eje X", cols_num, key="ex_line_x"))
                remaining_y = [col for col in cols_num if col != x_col]
                y_col = cast(str, st.selectbox("Eje Y", remaining_y, key="ex_line_y"))
                hue_candidates = ["Ninguna"] + cols_cat
                style_candidates = ["Ninguna"] + cols_cat
                hue_col = cast(str, st.selectbox("Hue", hue_candidates, key="ex_line_hue"))
                style_col = cast(str, st.selectbox("Style", ["Ninguna"] + cols_cat, key="ex_line_style"))

                try:
                    fig_example = plot_lineplot_semantics(
                        df,
                        x_col=x_col,
                        y_col=y_col,
                        hue_col=None if hue_col == "Ninguna" else hue_col,
                        style_col=None if style_col == "Ninguna" else style_col,
                    )
                    st.pyplot(fig_example)
                except Exception as e:
                    st.error(f"Error al generar el lineplot: {e}")

        with example_tab2:
            st.caption("Basado en el ejemplo de histogramas facetados con `displot`.")
            cols_num = df.select_dtypes(include=['number']).columns.tolist()
            cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

            if not cols_num:
                st.info("Se necesita al menos una columna numérica.")
            else:
                x_col = cast(str, st.selectbox("Variable numérica", cols_num, key="ex_hist_x"))
                row_col = cast(str, st.selectbox("Facet fila", ["Ninguna"] + cols_cat, key="ex_hist_row"))
                col_col = cast(str, st.selectbox("Facet columna", ["Ninguna"] + [col for col in cols_cat if col != row_col], key="ex_hist_col"))
                binwidth = st.number_input("Bin width", min_value=0.1, value=3.0, step=0.5, key="ex_hist_bin")

                try:
                    fig_example = plot_facet_histogram(
                        df,
                        x_col=x_col,
                        row_col=None if row_col == "Ninguna" else row_col,
                        col_col=None if col_col == "Ninguna" else col_col,
                        binwidth=float(binwidth),
                    )
                    st.pyplot(fig_example)
                except Exception as e:
                    st.error(f"Error al generar los histogramas facetados: {e}")

        with example_tab3:
            st.caption("Basado en el ejemplo de `relplot(kind='line')` con facetas.")
            cols_num = df.select_dtypes(include=['number']).columns.tolist()
            cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

            if len(cols_num) < 2:
                st.info("Se necesitan al menos dos columnas numéricas.")
            else:
                x_col = cast(str, st.selectbox("Eje X", cols_num, key="ex_fline_x"))
                remaining_y = [col for col in cols_num if col != x_col]
                y_col = cast(str, st.selectbox("Eje Y", remaining_y, key="ex_fline_y"))
                hue_col = cast(str, st.selectbox("Hue", ["Ninguna"] + cols_cat, key="ex_fline_hue"))
                size_col = cast(str, st.selectbox("Size", ["Ninguna"] + cols_cat, key="ex_fline_size"))
                col_col = cast(str, st.selectbox("Facet columna", ["Ninguna"] + cols_cat, key="ex_fline_col"))

                try:
                    fig_example = plot_facet_lineplot(
                        df,
                        x_col=x_col,
                        y_col=y_col,
                        hue_col=None if hue_col == "Ninguna" else hue_col,
                        size_col=None if size_col == "Ninguna" else size_col,
                        col_col=None if col_col == "Ninguna" else col_col,
                    )
                    st.pyplot(fig_example)
                except Exception as e:
                    st.error(f"Error al generar el lineplot facetado: {e}")

        with example_tab4:
            st.caption("Basado en el ejemplo de boxplot agrupado.")
            cols_num = df.select_dtypes(include=['number']).columns.tolist()
            cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

            if not cols_num or not cols_cat:
                st.info("Se necesita al menos una columna numérica y una categórica.")
            else:
                x_col = cast(str, st.selectbox("Categoría principal", cols_cat, key="ex_box_x"))
                y_col = cast(str, st.selectbox("Valor numérico", cols_num, key="ex_box_y"))
                hue_col = cast(str, st.selectbox("Hue", ["Ninguna"] + [col for col in cols_cat if col != x_col], key="ex_box_hue"))

                try:
                    fig_example = plot_grouped_boxplot(
                        df,
                        x_col=x_col,
                        y_col=y_col,
                        hue_col=None if hue_col == "Ninguna" else hue_col,
                    )
                    st.pyplot(fig_example)
                except Exception as e:
                    st.error(f"Error al generar el boxplot: {e}")

        with example_tab5:
            st.caption("Basado en el ejemplo de matriz de correlación diagonal.")
            try:
                fig_example = plot_diagonal_correlation_matrix(df)
                st.pyplot(fig_example)
            except Exception as e:
                st.error(f"Error al generar la matriz de correlación diagonal: {e}")

        with example_tab6:
            st.caption("Basado en el ejemplo de barplot horizontal.")
            cols_num = df.select_dtypes(include=['number']).columns.tolist()
            cols_cat = df.select_dtypes(exclude=['number']).columns.tolist()

            if not cols_num or not cols_cat:
                st.info("Se necesita al menos una columna numérica y una categórica.")
            else:
                category_col = cast(str, st.selectbox("Categoría", cols_cat, key="ex_bar_cat"))
                value_col = cast(str, st.selectbox("Valor", cols_num, key="ex_bar_val"))
                compare_col = cast(str, st.selectbox("Comparar por", ["Ninguna"] + [col for col in cols_cat if col != category_col], key="ex_bar_cmp"))

                try:
                    fig_example = plot_horizontal_barplot(
                        df,
                        category_col=category_col,
                        value_col=value_col,
                        compare_col=None if compare_col == "Ninguna" else compare_col,
                    )
                    st.pyplot(fig_example)
                except Exception as e:
                    st.error(f"Error al generar el barplot horizontal: {e}")
else:
    st.warning("Por favor, carga un dataset en el paso 1.")