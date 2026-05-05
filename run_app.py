"""Lanzador compatible con PyInstaller para abrir la app de Streamlit."""

import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    """Obtiene la ruta absoluta para recursos, compatible con PyInstaller."""

    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)

if __name__ == "__main__":
    # Reescribimos argv para invocar Streamlit como si se hubiera lanzado desde consola.
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    
    # Ejecutamos el CLI de Streamlit.
    sys.exit(stcli.main())