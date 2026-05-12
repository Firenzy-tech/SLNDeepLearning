"""
Script para cargar y validar variables de entorno.
Ejecuta esto una sola vez en la terminal para configurar tu API Key de Groq.
"""

import os
import sys
from pathlib import Path

def setup_groq_api():
    """Configura la API Key de Groq de forma segura."""
    
    print("=" * 60)
    print("🔧 Configuración de Groq API")
    print("=" * 60)
    
    # Verifica si ya está configurado
    if os.getenv("GROQ_API_KEY"):
        print("✅ GROQ_API_KEY ya está configurado en variables de entorno.")
        return True
    
    # Busca .env en el proyecto
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        if os.getenv("GROQ_API_KEY"):
            print("✅ GROQ_API_KEY cargado desde archivo .env")
            return True
    
    # Solicita al usuario que configure su API Key
    print("\n📝 No se encontró API Key configurada.")
    print("\nPasos para obtener tu API Key:")
    print("1. Ve a: https://console.groq.com")
    print("2. Crea una cuenta (gratis)")
    print("3. Copia tu API Key")
    print("\nOpciones de configuración:\n")
    
    choice = input("¿Deseas usar variables de entorno del sistema (S) o crear un archivo .env (E)?\nOpción (S/E): ").strip().upper()
    
    api_key = input("\n🔑 Pega tu API Key de Groq: ").strip()
    
    if not api_key:
        print("❌ API Key vacía. Configuración cancelada.")
        return False
    
    if choice == "S":
        print("\n⚠️  Para establecer la variable de entorno en Windows, ejecuta en PowerShell (como Admin):")
        print(f'setx GROQ_API_KEY "{api_key}"\n')
        print("Luego reinicia Streamlit: streamlit run app.py")
    elif choice == "E":
        with open(".env", "w") as f:
            f.write(f'GROQ_API_KEY="{api_key}"\n')
        print("✅ Archivo .env creado exitosamente.")
        print("💡 No olvides agregar .env a .gitignore para proteger tu API Key")
    else:
        print("❌ Opción no válida.")
        return False
    
    print("\n✅ Configuración completada. ¡Ya puedes usar diagnósticos con IA!")
    return True

if __name__ == "__main__":
    setup_groq_api()
