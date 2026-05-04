import streamlit.web.cli as stcli
import os, sys

def resolve_path(path):
    """Obtiene la ruta absoluta para recursos, compatible con PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return os.path.join(os.path.abspath("."), path)

if __name__ == "__main__":
    # Configuramos los argumentos que normalmente irían en la terminal
    # 'run' + la ruta al archivo app.py
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--global.developmentMode=false",
    ]
    
    # Ejecutamos el CLI de streamlit
    sys.exit(stcli.main())