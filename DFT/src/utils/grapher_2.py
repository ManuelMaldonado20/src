import matplotlib.pyplot as plt
import numpy as np

def continuous_plotter(x, y, xlabel='x', ylabel='y', title='Gráfica Continua'):
 
    plt.plot(x, y, 'b-', linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def discrete_plotter(n, x, xlabel='n', ylabel='x[n]', title='Señal Discreta'):
 
    # Gráfica de puntos con líneas verticales (stem plot)
    markerline, stemlines, baseline = plt.stem(n, x, basefmt=' ')
    
    # Personalizar la apariencia
    plt.setp(markerline, 'markersize', 4, 'color', 'red')
    plt.setp(stemlines, 'color', 'blue', 'linewidth', 1)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def spectrum_plotter(freq, magnitude, xlabel='Frecuencia (Hz)', 
                    ylabel='Magnitud', title='Espectro de Frecuencia'):
 
    plt.plot(freq, magnitude, 'b-', linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, freq[-1])
    plt.tight_layout()

def comparison_plotter(x, y1, y2, labels=['Señal 1', 'Señal 2'], 
                      xlabel='x', ylabel='y', title='Comparación de Señales'):

    plt.plot(x, y1, 'b-', linewidth=1.5, label=labels[0], alpha=0.8)
    plt.plot(x, y2, 'r-', linewidth=1.5, label=labels[1], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
