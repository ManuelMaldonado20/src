import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <modulo>")
        print("Ejemplo: python main.py examen_p2")
        sys.exit(1)
    
    module_name = sys.argv[1]
    
    try:
        if module_name == "examen_p2":
            from examen_p2 import run_analysis
            run_analysis()
        else:
            print(f"Módulo '{module_name}' no reconocido")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error al importar el módulo: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main_2__":
    main_2()
