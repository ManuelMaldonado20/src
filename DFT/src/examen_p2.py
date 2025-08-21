import numpy as np
import matplotlib.pyplot as plt
from utils.grapher_2 import discrete_plotter

class DFTAnalyzer:
    
    def __init__(self, fs=256, duration=6):
    
        self.fs = fs
        self.ts = 1 / fs
        self.duration = duration
        self.N = int(fs * duration)
        self.n = np.arange(self.N)
        self.t = self.n * self.ts
        
        # Resolución en frecuencia
        self.delta_f = fs / self.N
        self.frequencies = np.arange(self.N) * self.delta_f
        
        print(f"Parámetros de la señal:")
        print(f"Frecuencia de muestreo: {self.fs} Hz")
        print(f"Periodo de muestreo: {self.ts:.6f} s")
        print(f"Duración: {self.duration} s")
        print(f"Número de muestras: {self.N}")
        print(f"Resolución en frecuencia: {self.delta_f:.3f} Hz")
    
    def dft(self, x):

        N = len(x)
        X = np.zeros(N, dtype=complex)
        
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
                
        return X
    
    def generate_clean_signal(self, f1=8, f2=20):

        x_clean = (np.sin(2 * np.pi * f1 * self.n * self.ts) + 
                   0.5 * np.sin(2 * np.pi * f2 * self.n * self.ts))
        
        print(f"\nSeñal generada:")
        print(f"x[n] = sin(2π·{f1}·n·ts) + 0.5·sin(2π·{f2}·n·ts)")
        
        return x_clean
    
    def add_noise(self, x_clean, noise_freq=50, noise_amplitude=0.3, random_noise=0.1):
 
        # Ruido tonal de frecuencia específica
        tonal_noise = noise_amplitude * np.sin(2 * np.pi * noise_freq * self.n * self.ts)
        
        # Ruido aleatorio blanco
        white_noise = random_noise * np.random.randn(self.N)
        
        # Señal con ruido
        x_noisy = x_clean + tonal_noise + white_noise
        
        print(f"\nRuido agregado:")
        print(f"Ruido tonal: {noise_amplitude}·sin(2π·{noise_freq}·n·ts)")
        print(f"Ruido blanco: amplitud {random_noise}")
        
        return x_noisy, tonal_noise, white_noise
    
    def analyze_spectrum(self, x, title="Señal"):
      
        # Calcular DFT
        X = self.dft(x)
        
        # Magnitud y fase
        magnitude = np.abs(X)
        phase = np.angle(X)
        
        # Solo mostrar frecuencias positivas (hasta Nyquist)
        nyquist_idx = self.N // 2
        freq_positive = self.frequencies[:nyquist_idx]
        mag_positive = magnitude[:nyquist_idx]
        
        # Encontrar picos principales
        peaks = self.find_peaks(mag_positive, threshold=0.1)
        
        print(f"\n=== Análisis de {title} ===")
        print(f"Picos detectados en frecuencias:")
        for peak_idx in peaks:
            freq = freq_positive[peak_idx]
            mag = mag_positive[peak_idx]
            print(f"  {freq:.2f} Hz (magnitud: {mag:.2f})")
        
        return X, magnitude, phase, freq_positive, mag_positive, peaks
    
    def find_peaks(self, magnitude, threshold=0.1):
       
        max_mag = np.max(magnitude)
        threshold_abs = threshold * max_mag
        peaks = []
        
        for i in range(1, len(magnitude) - 1):
            if (magnitude[i] > magnitude[i-1] and 
                magnitude[i] > magnitude[i+1] and 
                magnitude[i] > threshold_abs):
                peaks.append(i)
        
        return peaks
    
    def plot_analysis(self, x_clean, x_noisy, X_clean, X_noisy, 
                     freq_pos, mag_clean, mag_noisy, peaks_clean, peaks_noisy):
        """
        Genera todas las gráficas del análisis
        """
        # Configurar estilo de gráficas
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(15, 12))
        
        # Gráfica 1: Señal limpia en tiempo
        plt.subplot(3, 2, 1)
        discrete_plotter(self.t[:200], x_clean[:200], 
                        'Tiempo (s)', 'Amplitud', 'Señal Limpia (primeras 200 muestras)')
        
        # Gráfica 2: Señal con ruido en tiempo
        plt.subplot(3, 2, 2)
        discrete_plotter(self.t[:200], x_noisy[:200], 
                        'Tiempo (s)', 'Amplitud', 'Señal con Ruido (primeras 200 muestras)')
        
        # Gráfica 3: Espectro señal limpia
        plt.subplot(3, 2, 3)
        plt.plot(freq_pos, mag_clean, 'b-', linewidth=1.5, label='Magnitud')
        for peak in peaks_clean:
            plt.plot(freq_pos[peak], mag_clean[peak], 'ro', markersize=8)
            plt.annotate(f'{freq_pos[peak]:.1f} Hz', 
                        xy=(freq_pos[peak], mag_clean[peak]),
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.title('Espectro - Señal Limpia')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        
        # Gráfica 4: Espectro señal con ruido
        plt.subplot(3, 2, 4)
        plt.plot(freq_pos, mag_noisy, 'r-', linewidth=1.5, label='Magnitud')
        for peak in peaks_noisy:
            plt.plot(freq_pos[peak], mag_noisy[peak], 'ro', markersize=8)
            plt.annotate(f'{freq_pos[peak]:.1f} Hz', 
                        xy=(freq_pos[peak], mag_noisy[peak]),
                        xytext=(5, 5), textcoords='offset points')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.title('Espectro - Señal con Ruido')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        
        # Gráfica 5: Comparación de espectros (zoom)
        plt.subplot(3, 2, 5)
        plt.plot(freq_pos, mag_clean, 'b-', linewidth=2, label='Señal Limpia', alpha=0.8)
        plt.plot(freq_pos, mag_noisy, 'r-', linewidth=1, label='Señal con Ruido', alpha=0.7)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.title('Comparación de Espectros (0-60 Hz)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 60)
        
        # Gráfica 6: Espectro completo hasta Nyquist
        plt.subplot(3, 2, 6)
        plt.plot(freq_pos, mag_noisy, 'g-', linewidth=1, label='Señal con Ruido')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.title(f'Espectro Completo (0-{self.fs/2:.0f} Hz)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def run_analysis():
    """Función principal para ejecutar el análisis completo"""
    print("=== ANÁLISIS DE SEÑALES DISCRETAS CON DFT ===")
    print("Implementación propia de la Transformada de Fourier Discreta\n")
    
    # Inicializar analizador
    analyzer = DFTAnalyzer(fs=256, duration=6)
    
    # Generar señal limpia
    x_clean = analyzer.generate_clean_signal(f1=8, f2=20)
    
    # Añadir ruido
    x_noisy, tonal_noise, white_noise = analyzer.add_noise(
        x_clean, noise_freq=50, noise_amplitude=0.3, random_noise=0.1)
    
    # Analizar señal limpia
    X_clean, mag_clean_full, phase_clean, freq_pos, mag_clean, peaks_clean = \
        analyzer.analyze_spectrum(x_clean, "Señal Limpia")
    
    # Analizar señal con ruido
    X_noisy, mag_noisy_full, phase_noisy, _, mag_noisy, peaks_noisy = \
        analyzer.analyze_spectrum(x_noisy, "Señal con Ruido")
    
    # Generar gráficas
    analyzer.plot_analysis(x_clean, x_noisy, X_clean, X_noisy,
                          freq_pos, mag_clean, mag_noisy, peaks_clean, peaks_noisy)
    
    # Resumen final
    print(f"\n=== RESUMEN DEL ANÁLISIS ===")
    print(f"Resolución en frecuencia (Δf): {analyzer.delta_f:.3f} Hz")
    print(f"Frecuencia de Nyquist: {analyzer.fs/2:.0f} Hz")
    print(f"Número total de muestras: {analyzer.N}")
    print(f"Duración de la señal: {analyzer.duration} s")
    print(f"\nLa DFT fue calculada usando implementación propia (no FFT)")
    print("Análisis completado exitosamente.")

if __name__ == "__main__":
    run_analysis()
