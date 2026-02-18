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
        
        # --- DEFINIÇÃO DOS DOIS MUNDOS (GRIDS) ---
        
        # 1. Grid de Simulação (Ground Truth): Altíssima resolução para simular a realidade contínua
        # Usamos 0.005mm para garantir que a integral do feixe seja perfeita
        self.x_sim = np.arange(0, self.scan_length, 0.005) 
        
        # 2. Grid de Reconstrução (Target): A resolução que você escolheu para o resultado final
        self.x_recon = np.arange(0, self.scan_length, recon_resolution_mm)
        
        # Variáveis de Estado
        self.real_stress_profile = None   # O perfil real no grid x_sim
        self.measurement_centers = None   # Onde o robô parou
        self.measured_values = None       # O vetor 'b' (com ruído)
        self.reconstructed_profile = None # O vetor 'x' recuperado (no grid x_recon)
        
        # As Matrizes
        self.A_sim = None   # Conecta x_sim -> b
        self.A_recon = None # Conecta x_recon -> b

    def load_ground_truth(self):
        """Carrega a função spline e gera o perfil real no grid de SIMULAÇÃO."""
        if not os.path.exists(self.pkl_filename):
            raise FileNotFoundError(f"Arquivo {self.pkl_filename} não encontrado.")
            
        with open(self.pkl_filename, "rb") as f:
            spline_function = pickle.load(f)
        
        # Avalia apenas no grid de alta resolução (realidade)
        self.real_stress_profile = spline_function(self.x_sim)

    def generate_measurement_points(self):
        """Define os passos do robô."""
        step_size = self.beam_diameter * (1 - self.overlap)
        self.measurement_centers = np.arange(0, self.scan_length, step_size)

    def _build_generic_matrix(self, target_grid):
        """
        Método Genérico: Constrói uma matriz A que conecta o 'target_grid' às medições.
        Aplica a Regra de Simetria Reflexiva em X=0.
        """
        n_measurements = len(self.measurement_centers)
        n_target = len(target_grid)
        dx_target = target_grid[1] - target_grid[0]
        
        matrix = np.zeros((n_measurements, n_target))
        
        # Para otimizar, calculamos quantos pontos do grid cabem no feixe
        # Isso define o "peso base" de cada ponto na média
        points_per_beam = int(np.round(self.beam_diameter / dx_target))
        if points_per_beam < 1: points_per_beam = 1
        weight = 1.0 / points_per_beam

        for i, center in enumerate(self.measurement_centers):
            # Janela Física do Feixe
            x_start_phys = center - self.beam_radius
            
            # Geramos os pontos físicos teóricos que o feixe está "lendo"
            # Ex: se o feixe está em 0, lê de -0.25 a +0.25
            x_samples_phys = np.linspace(x_start_phys, x_start_phys + self.beam_diameter, points_per_beam)
            
            for x_p in x_samples_phys:
                # --- AQUI ESTÁ A MÁGICA DA SIMETRIA ---
                # Se o feixe lê -0.1mm, isso corresponde à tensão em +0.1mm na peça
                x_effective = abs(x_p)
                
                # Achamos qual nó do grid (índice j) está mais próximo dessa posição efetiva
                idx_j = int(np.round(x_effective / dx_target))
                
                # Se o índice cair dentro do nosso grid, somamos o peso
                if 0 <= idx_j < n_target:
                    matrix[i, idx_j] += weight
                    
        return matrix

    def run_simulation(self):
        """
        Passo 1: O Mundo Real
        Constrói A_sim e gera as medições com ruído.
        """
        print("1. Construindo Matriz de Simulação (Alta Resolução)...")
        self.A_sim = self._build_generic_matrix(self.x_sim)
        
        # Medição Perfeita = Matriz Simulação * Realidade Fina
        clean_measurements = self.A_sim @ self.real_stress_profile
        
        # Adiciona Ruído
        noise = np.random.normal(0, self.noise_std, len(clean_measurements))
        self.measured_values = clean_measurements + noise
        print(f"   -> Geradas {len(self.measured_values)} medições.")

    def run_reconstruction(self):
        """
        Passo 2: A Matemática Inversa
        Constrói A_recon (Resolução do Usuário) e inverte o sistema.
        """
        print(f"2. Construindo Matriz de Reconstrução (Resolução: {self.x_recon[1]}mm)...")
        self.A_recon = self._build_generic_matrix(self.x_recon)
        
        print("3. Resolvendo Sistema Inverso...")
        # Resolve: A_recon * x_recon = b_medido
        self.reconstructed_profile, _, _, _ = np.linalg.lstsq(
            self.A_recon, 
            self.measured_values, 
            rcond=0.05 # Filtro de regularização para estabilidade
        )

    def plot_comparison(self):
        plt.figure(figsize=(12, 7))
        
        # 1. Ground Truth (O perfil real que criamos no Step 1)
        plt.plot(self.x_sim, self.real_stress_profile, 'k--', linewidth=1.5, alpha=0.6, label='Realidade (Ground Truth)')
        
        # 2. Medições (O que o equipamento viu)
        plt.errorbar(self.measurement_centers, self.measured_values, 
                     xerr=self.beam_radius, yerr=self.noise_std, 
                     fmt='ro', capsize=0, alpha=0.4, label=f'Medição (Feixe {self.beam_diameter}mm)')
        
        # 3. Reconstrução (O resultado da matemática)
        plt.plot(self.x_recon, self.reconstructed_profile, 'b-', linewidth=2.5, label='Perfil Deconvolvido')
        
        plt.title(f'Deconvolução com Matrizes Independentes\nResolução Alvo: {self.x_recon[1]:.2f}mm | Overlap: {self.overlap*100:.0f}%')
        plt.xlabel('Distância (mm)')
        plt.ylabel('Tensão (MPa)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    # Substitua pelo caminho do seu arquivo .pkl gerado anteriormente
    pkl_file = "rs_profile/martin_senai_rs_profile.pkl" 
    
    try:
        # Configuração do Experimento Virtual
        experiment = XRD_Surface_Scan_Process(
            pkl_filename=pkl_file,
            beam_diameter_mm=0.5,    # Colimador largo (gera média forte)
            overlap_ratio=0.50,      # 60% de overlap
            noise_std_dev=0.0,      # Ruído realista
            scan_length_mm=3.5,      # Comprimento total
            recon_resolution_mm=0.1  # Quero um resultado a cada 0.1mm
        )
        
        experiment.load_ground_truth()
        experiment.generate_measurement_points()
        
        # Roda os dois processos separadamente
        experiment.run_simulation()
        experiment.run_reconstruction()
        
        experiment.plot_comparison()
        
    except FileNotFoundError:
        print("Erro: Arquivo pkl não encontrado. Rode o código gerador primeiro.")