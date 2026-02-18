import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Impede a criação de __pycache__
sys.dont_write_bytecode = True

class XRD_Surface_Scan_Process:
    def __init__(self, pkl_filename, beam_diameter_mm, overlap_ratio, noise_std_dev, scan_length_mm, recon_resolution_mm=0.05):
        self.pkl_filename = pkl_filename
        self.beam_diameter = beam_diameter_mm
        self.beam_radius = beam_diameter_mm / 2.0
        self.overlap = overlap_ratio
        self.noise_std = noise_std_dev
        self.scan_length = scan_length_mm
        self.resolution = recon_resolution_mm
        
        # Grid fino representando a superfície real (Ground Truth)
        self.x_fine = np.arange(0, self.scan_length, self.resolution)
        
        self.real_stress_profile = None
        self.measurement_points = None
        self.measured_values = None
        self.reconstructed_profile = None
        self.matrix_A = None

    def load_ground_truth(self):
        """Carrega o perfil de tensão superficial de um arquivo pickle."""
        if not os.path.exists(self.pkl_filename):
            raise FileNotFoundError(f"Arquivo {self.pkl_filename} não encontrado.")
            
        with open(self.pkl_filename, "rb") as f:
            spline_function = pickle.load(f)
            
        # Avalia a spline ao longo do eixo x da superfície
        self.real_stress_profile = spline_function(self.x_fine)
        return self.real_stress_profile

    def generate_measurement_grid(self):
        """Define onde o centro do feixe de RX irá tocar a superfície."""
        # O passo (step) é calculado com base no overlap (ex: 50% overlap = 0.5 * diâmetro)
        step_size = self.beam_diameter * (1 - self.overlap)
        self.measurement_points = np.arange(0, self.scan_length, step_size)
        return self.measurement_points

    def build_convolution_matrix(self):
        """
        Cria a matriz que mapeia o perfil real para a média integrada pelo feixe.
        Cada linha representa uma posição do feixe (janela móvel).
        """
        n_measurements = len(self.measurement_points)
        n_fine_points = len(self.x_fine)
        self.matrix_A = np.zeros((n_measurements, n_fine_points))
        
        for i, center in enumerate(self.measurement_points):
            # Define os limites da mancha do feixe na superfície
            window_start = center - self.beam_radius
            window_end = center + self.beam_radius
            
            # Identifica quais pontos do grid fino estão sob a mancha do feixe
            mask = (self.x_fine >= window_start) & (self.x_fine <= window_end)
            points_in_window = np.sum(mask)
            
            if points_in_window > 0:
                self.matrix_A[i, mask] = 1.0 / points_in_window

    def simulate_measurements(self):
        """Aplica a convolução e adiciona ruído gaussiano."""
        # O produto matricial simula a integração do feixe sobre a superfície
        clean_measurements = self.matrix_A @ self.real_stress_profile
        noise = np.random.normal(0, self.noise_std, len(clean_measurements))
        self.measured_values = clean_measurements + noise
        return self.measured_values

    def reconstruct_profile(self):
        """Realiza a deconvolução usando Mínimos Quadrados com regularização básica."""
        # rcond ajuda a lidar com a instabilidade da inversão (filtragem de ruído)
        x_reconstructed, _, _, _ = np.linalg.lstsq(
            self.matrix_A, 
            self.measured_values, 
            rcond=0.05 
        )
        self.reconstructed_profile = x_reconstructed
        return self.reconstructed_profile

    def run_full_process(self):
        self.load_ground_truth()
        self.generate_measurement_grid()
        self.build_convolution_matrix()
        self.simulate_measurements()
        self.reconstructed_profile = self.reconstruct_profile()

    def plot_results(self):
        plt.figure(figsize=(14, 6))
        
        # Perfil Real
        plt.plot(self.x_fine, self.real_stress_profile, 'k--', label='Perfil Real (Superfície)', alpha=0.7)
        
        # Medições (Barras horizontais representam o diâmetro do feixe)
        plt.errorbar(self.measurement_points, self.measured_values, 
                     xerr=self.beam_radius, yerr=self.noise_std, 
                     fmt='ro', capsize=0, alpha=0.5, label='Medição XRD (Abertura do Feixe)')
        
        # Reconstrução
        plt.plot(self.x_fine, self.reconstructed_profile, 'b-', linewidth=2, label='Deconvolução (Recuperado)')
        
        plt.title('Simulação de Varredura de Tensão Residual em Superfície')
        plt.xlabel('Posição na Superfície (mm)')
        plt.ylabel('Tensão (MPa)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    pkl_file = "rs_profile/martin_senai_rs_profile.pkl" 
    
    # Exemplo de uso: Varredura de 20mm na superfície
    try:
        scan = XRD_Surface_Scan_Process(
            pkl_filename=pkl_file,
            beam_diameter_mm=2.0,   # Feixe maior para ver o efeito de suavização
            overlap_ratio=0.75,     # 75% de sobreposição para maior densidade de dados
            noise_std_dev=10.0,     # Ruído de 10 MPa
            scan_length_mm=20.0
        )
        
        scan.run_full_process()
        scan.plot_results()
        
    except FileNotFoundError:
        print(f"Erro: Arquivo '{pkl_file}' não encontrado.")