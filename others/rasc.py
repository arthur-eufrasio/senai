import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt

# 1. Gerar Sinal Artificial (Similar ao seu de Tensão Residual)
x = np.linspace(0, 10, 200)
# Um pico fino e um vale largo
true_signal = -800 * np.exp(-(x - 2)**2 / 0.2) + 300 * np.exp(-(x - 7)**2 / 2.0)

# Adicionar Ruído
np.random.seed(42)
noise = np.random.normal(0, 100, size=len(x))
noisy_signal = true_signal + noise

# --- APLICAÇÃO DOS FILTROS ---

# A. Média Móvel (Convolução simples)
# Problema: Atraso de fase e achatamento de picos
window_size = 15
moving_avg = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='same')

# B. Filtro Savitzky-Golay (O "Rei" para este caso)
# window_length: Tamanho da janela (deve ser ímpar)
# polyorder: Grau do polinômio (2 ou 3 é bom para curvas suaves)
savgol = savgol_filter(noisy_signal, window_length=21, polyorder=3)

# C. Filtro de Mediana (Bom para spikes, mas deixa o sinal "quadrado")
median = medfilt(noisy_signal, kernel_size=15)

# --- PLOTAGEM ---
plt.figure(figsize=(12, 7))

plt.plot(x, noisy_signal, color='lightgray', label='Sinal Ruidoso (Raw Data)')
plt.plot(x, true_signal, 'k--', linewidth=2, label='Real (Ground Truth)')

plt.plot(x, moving_avg, 'm-', linewidth=2, label='Média Móvel (Achata o pico!)')
plt.plot(x, median, 'b-', linewidth=1.5, alpha=0.7, label='Mediana (Bom p/ spikes)')
plt.plot(x, savgol, 'g-', linewidth=3, label='Savitzky-Golay (Preserva o pico)')

plt.title('Batalha dos Filtros: Qual preserva melhor o pico de tensão?')
plt.xlabel('Profundidade (mm)')
plt.ylabel('Tensão (MPa)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()