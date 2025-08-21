
import sys
import os

# Agregar src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) != 2:
        print("Uso: python main.py <nombre_modulo>")
        print("Ejemplo: python main.py examen_p1")
        sys.exit(1)
    
    module_name = sys.argv[1]
    
    try:
        # Importar dinámicamente el módulo especificado
        module = __import__(module_name)
        
        # Ejecutar la función principal del módulo
        if hasattr(module, 'run_analysis'):
            module.run_analysis()
        else:
            print(f"Error: El módulo {module_name} no tiene función run_analysis()")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error: No se pudo importar el módulo '{module_name}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
