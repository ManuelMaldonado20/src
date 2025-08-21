import matplotlib.pyplot as plt
import numpy as np

def continuous_plotter(x, y, title="Señal Continua", xlabel="x", ylabel="y", 
                      grid=True, highlight_points=None, figsize=(10, 6)):

    plt.figure(figsize=figsize)
    plt.plot(x, y, 'b-', linewidth=1.5, label='Señal')
    
    # Resaltar puntos específicos si se proporcionan
    if highlight_points is not None:
        x_pts, y_pts = highlight_points
        plt.plot(x_pts, y_pts, 'ro', markersize=8, label='Picos')
        
        # Anotar los primeros 3 picos más importantes
        for i, (xp, yp) in enumerate(zip(x_pts[:3], y_pts[:3])):
            plt.annotate(f'f={xp:.2f}Hz', 
                        xy=(xp, yp), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=10,
                        ha='left')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    if highlight_points is not None:
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def discrete_plotter(x, y, title="Señal Discreta", xlabel="x", ylabel="y", 
                    grid=True, figsize=(10, 6)):

    plt.figure(figsize=figsize)
    
    # Graficar puntos discretos
    plt.plot(x, y, 'ro', markersize=4, label='Muestras')
    
    # Agregar líneas verticales para mejor visualización
    for i in range(0, len(x), max(1, len(x)//50)):  # No más de 50 líneas
        plt.plot([x[i], x[i]], [0, y[i]], 'r-', alpha=0.3, linewidth=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def dual_plotter(x1, y1, x2, y2, title1="Señal 1", title2="Señal 2",
                xlabel="x", ylabel="y", figsize=(12, 8)):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Primer subplot
    ax1.plot(x1, y1, 'b-', linewidth=1.5)
    ax1.set_title(title1, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Segundo subplot
    ax2.plot(x2, y2, 'r-', linewidth=1.5)
    ax2.set_title(title2, fontsize=12, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel(ylabel, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def spectrum_plotter(frequencies, magnitude, phase=None, title="Espectro de Frecuencias",
                    figsize=(12, 8)):

    if phase is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Magnitud
        ax1.plot(frequencies, magnitude, 'b-', linewidth=1.5)
        ax1.set_title(f"{title} - Magnitud", fontweight='bold')
        ax1.set_ylabel("Magnitud", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Fase
        ax2.plot(frequencies, phase, 'r-', linewidth=1.5)
        ax2.set_title(f"{title} - Fase", fontweight='bold')
        ax2.set_xlabel("Frecuencia (Hz)", fontsize=10)
        ax2.set_ylabel("Fase (rad)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
    else:
        plt.figure(figsize=figsize)
        plt.plot(frequencies, magnitude, 'b-', linewidth=1.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Frecuencia (Hz)", fontsize=12)
        plt.ylabel("Magnitud", fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
