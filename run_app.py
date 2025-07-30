#!/usr/bin/env python3
"""
Script para ejecutar la aplicación de audio con FFT
"""

import subprocess
import sys
import os

def install_requirements():
    """Instala las dependencias necesarias"""
    print("🔧 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def run_streamlit_app():
    """Ejecuta la aplicación de Streamlit"""
    print("🚀 Iniciando aplicación de Streamlit...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando la aplicación: {e}")

def main():
    """Función principal"""
    print("🎵 Aplicación de Audio con Transformada de Fourier")
    print("=" * 50)
    
    # Verificar que existe app.py
    if not os.path.exists("app.py"):
        print("❌ Error: No se encuentra el archivo app.py")
        return
    
    # Verificar que existe requirements.txt
    if not os.path.exists("requirements.txt"):
        print("❌ Error: No se encuentra el archivo requirements.txt")
        return
    
    # Preguntar si instalar dependencias
    install = input("¿Instalar dependencias? (y/n): ").lower().strip()
    if install in ['y', 'yes', 'sí', 's']:
        if not install_requirements():
            return
    
    # Ejecutar aplicación
    print("\n📱 La aplicación se abrirá en tu navegador en http://localhost:8501")
    print("⚠️  Asegúrate de tener un micrófono conectado")
    print("🔴 Presiona Ctrl+C para cerrar la aplicación")
    print("-" * 50)
    
    run_streamlit_app()

if __name__ == "__main__":
    main()