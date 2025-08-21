
import numpy as np
import math
from utils.grapher import continuous_plotter, discrete_plotter

class DFTAnalyzer:
    
    def __init__(self, fm=0.5, fc=8.0, m=0.5):
        self.fm = fm  # Frecuencia de modulación
        self.fc = fc  # Frecuencia portadora
        self.m = m    # Índice de modulación
        
    def signal_function(self, t):
       
        envelope = 1 + self.m * np.cos(2 * np.pi * self.fm * t)
        carrier = np.sin(2 * np.pi * self.fc * t)
        return envelope * carrier
    
    def custom_dft(self, x):
    
        N = len(x)
        X = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for n in range(N):
                angle = -2 * np.pi * k * n / N
                X[k] += x[n] * (np.cos(angle) + 1j * np.sin(angle))
                
        return X
    
    def find_peaks(self, magnitude, frequencies, threshold_ratio=0.1):
       
        max_mag = np.max(magnitude)
        threshold = max_mag * threshold_ratio
        
        peaks = []
        for i in range(1, len(magnitude) - 1):
            if (magnitude[i] > magnitude[i-1] and 
                magnitude[i] > magnitude[i+1] and 
                magnitude[i] > threshold):
                peaks.append({
                    'frequency': frequencies[i],
                    'magnitude': magnitude[i],
                    'amplitude_rel': magnitude[i] / max_mag
                })
        
        # Ordenar por magnitud descendente
        peaks.sort(key=lambda x: x['magnitude'], reverse=True)
        return peaks
    
    def analyze_signal(self, duration=4.0, fs=64.0):
       
        print("=== ANÁLISIS DE SEÑAL MODULADA ===")
        print(f"Parámetros: fm={self.fm} Hz, fc={self.fc} Hz, m={self.m}")
        print(f"Duración: {duration} s, Fs: {fs} Hz")
        
        # Generar señal continua para visualización
        t_cont = np.linspace(0, duration, 1000)
        x_cont = self.signal_function(t_cont)
        
        # Generar señal muestreada
        N = int(fs * duration)
        t_discrete = np.linspace(0, duration, N)
        x_discrete = self.signal_function(t_discrete)
        
        print(f"Número de muestras: {N}")
        
        # Calcular resolución en frecuencia
        delta_f = fs / N
        print(f"Resolución en frecuencia: Δf = {delta_f:.3f} Hz")
        
        # Aplicar DFT personalizada
        print("\nCalculando DFT...")
        X = self.custom_dft(x_discrete)
        
        # Calcular magnitud y frecuencias
        magnitude = np.abs(X)
        frequencies = np.linspace(0, fs, N)
        
        # Usar solo la primera mitad (espectro positivo)
        half_N = N // 2
        magnitude_half = magnitude[:half_N]
        frequencies_half = frequencies[:half_N]
        
        # Encontrar picos espectrales
        peaks = self.find_peaks(magnitude_half, frequencies_half)
        
        print(f"\n=== PICOS ESPECTRALES DETECTADOS ===")
        for i, peak in enumerate(peaks[:8]):  # Mostrar top 8
            print(f"Pico {i+1}: f = {peak['frequency']:.3f} Hz, "
                  f"Magnitud = {peak['magnitude']:.2f}, "
                  f"Amplitud relativa = {peak['amplitude_rel']:.3f}")
        
        # Análisis teórico
        print(f"\n=== ANÁLISIS TEÓRICO ===")
        print(f"Frecuencias esperadas:")
        print(f"- Portadora: {self.fc} Hz")
        print(f"- Banda lateral inferior: {self.fc - self.fm} = {self.fc - self.fm} Hz")
        print(f"- Banda lateral superior: {self.fc + self.fm} = {self.fc + self.fm} Hz")
        
        # Graficar resultados
        self.plot_results(t_cont, x_cont, t_discrete, x_discrete, 
                         frequencies_half, magnitude_half, peaks)
        
        return {
            'time_cont': t_cont,
            'signal_cont': x_cont,
            'time_discrete': t_discrete,
            'signal_discrete': x_discrete,
            'frequencies': frequencies_half,
            'magnitude': magnitude_half,
            'peaks': peaks,
            'delta_f': delta_f
        }
    
    def plot_results(self, t_cont, x_cont, t_discrete, x_discrete, 
                    frequencies, magnitude, peaks):
        """Genera las gráficas de los resultados"""
        
        # Gráfica de señal continua
        continuous_plotter(
            t_cont, x_cont,
            title="Señal Modulada - Vista Continua",
            xlabel="Tiempo (s)",
            ylabel="Amplitud",
            grid=True
        )
        
        # Gráfica de señal muestreada
        discrete_plotter(
            t_discrete, x_discrete,
            title="Señal Modulada - Muestras Discretas",
            xlabel="Tiempo (s)",
            ylabel="Amplitud",
            grid=True
        )
        
        # Gráfica del espectro de frecuencias
        continuous_plotter(
            frequencies, magnitude,
            title="Espectro de Frecuencias (DFT)",
            xlabel="Frecuencia (Hz)",
            ylabel="Magnitud",
            grid=True
        )
        
        # Marcar picos principales
        if len(peaks) > 0:
            peak_freqs = [p['frequency'] for p in peaks[:5]]
            peak_mags = [p['magnitude'] for p in peaks[:5]]
            
            continuous_plotter(
                frequencies, magnitude,
                title="Espectro con Picos Identificados",
                xlabel="Frecuencia (Hz)",
                ylabel="Magnitud",
                grid=True,
                highlight_points=(peak_freqs, peak_mags)
            )

def run_analysis():
    """Función principal para ejecutar el análisis"""
    analyzer = DFTAnalyzer()
    results = analyzer.analyze_signal()
    
    print(f"\n=== RESUMEN ===")
    print(f"Análisis completado exitosamente")
    print(f"Se detectaron {len(results['peaks'])} picos espectrales")
    print(f"Resolución en frecuencia: {results['delta_f']:.3f} Hz")

if __name__ == "__main__":
    run_analysis()
