import pickle
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar a função do arquivo
try:
    with open("rs_profile/martin_senai_rs_profile.pkl", "rb") as f:
        funcao_carregada = pickle.load(f)
    print("Função carregada com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Rode o código anterior primeiro.")

# 2. Usar a função normalmente
# Você pode passar um número ou um array numpy
valores_x = np.linspace(0, 4.0 * 1.75, 10000)
valores_y = funcao_carregada(valores_x)

# Teste rápido
print(f"Tensão em r=1.0mm: {funcao_carregada(1.0):.2f} MPa")

# Plotar para confirmar
plt.plot(valores_x, valores_y)
plt.title("Usando a Curva Carregada")
plt.xlabel("Distância Radial (mm)")
plt.ylabel("Tensão Residual (MPa)")
plt.grid(True, alpha=0.3)
plt.show()