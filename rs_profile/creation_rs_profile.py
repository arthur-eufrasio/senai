import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle

# --- 1. DEFINIÇÃO DA CURVA ORIGINAL (NORMALIZADA) ---
def curve_equation_original(r_norm, sigma_ref=1.0):
    """
    Modelo original normalizado para referência visual.
    """
    t1 = -0.38 * np.exp(-(r_norm - 0.0)**2 / (2 * 1.2**2)) 
    t2 = -0.75 * np.exp(-(r_norm - 1.05)**2 / (2 * 0.35**2))
    t3 = 0.12 * np.exp(-(r_norm - 2.6)**2 / (2 * 0.7**2))
    return sigma_ref * (t1 + t2 + t3)

# --- 2. PLOT INTERATIVO (MUNDO NORMALIZADO) ---
# Usamos o normalizado para facilitar o desenho (0 a 4)
r_norm_plot = np.linspace(0, 4.0, 300)
sigma_norm_plot = curve_equation_original(r_norm_plot)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r_norm_plot, sigma_norm_plot, 'b--', alpha=0.4, label='Referência (Normalizada)')
ax.set_title("PASSO 1: Clique para desenhar o perfil (Normalizado). ENTER para finalizar.")
ax.set_xlabel("Raio Normalizado (r / r_spot)")
ax.set_ylabel("Tensão Normalizada (sigma / sigma_ref)")
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.2, 0.5)
ax.set_xlim(0, 4.0)
ax.legend()

print(">>> INSTRUÇÕES:")
print("1. Clique no gráfico para criar a forma da curva.")
print("2. Pressione ENTER (ou botão do meio) quando terminar a seleção.")

# --- 3. COLETA DE PONTOS ---
points = plt.ginput(n=-1, timeout=0, show_clicks=True)
plt.close(fig) # Fecha a janela antiga para não confundir

if len(points) < 2:
    print("Erro: Você precisa selecionar pelo menos 2 pontos!")
else:
    # --- 4. ENTRADA DE DADOS FÍSICOS (CONVERSÃO) ---
    print("\n" + "="*40)
    print("   CONFIGURAÇÃO DE ESCALA REAL")
    print("="*40)
    
    try:
        r_spot_input = float(input("Digite o RAIO DO SPOT (r_spot) em mm [ex: 1.5]: "))
        sigma_ref_input = float(input("Digite a TENSÃO DE REFERÊNCIA (sigma_ref) em MPa [ex: -400]: "))
    except ValueError:
        print("Erro: Por favor, digite números válidos (use ponto para decimais).")
        exit()

    # Processamento dos pontos clicados
    points = np.array(points)
    order = np.argsort(points[:, 0])
    
    # --- CONVERSÃO: NORMALIZADO -> REAL ---
    # x_user (0 a 4) vira mm
    x_real_mm = points[order, 0] * r_spot_input
    
    # y_user (-1 a 0.5) vira MPa
    # Nota: Se sigma_ref for negativo (ex: -400 de compressão), 
    # certifique-se de que a lógica de sinal faz sentido para você.
    # Geralmente a curva normalizada é negativa e o sigma_ref é positivo (magnitude),
    # OU a curva é negativa e sigma_ref é escalar. 
    # Aqui multiplicamos direto: (-1.0 * 400 = -400 MPa) ou (-1.0 * -400 = +400?).
    # Assumindo que você desenhou picos negativos e sigma_ref é a MAGNITUDE (positivo):
    y_real_mpa = points[order, 1] * abs(sigma_ref_input)
    
    # Se quiser que sigma_ref seja o valor de pico negativo (ex: -800), remova o abs() acima.

    # --- 5. CRIAÇÃO DA SPLINE REAL ---
    # Esta função agora aceita mm e retorna MPa
    spline_real = CubicSpline(x_real_mm, y_real_mpa, bc_type='natural')

    # --- 6. PLOTAGEM DO RESULTADO FINAL (FÍSICO) ---
    r_final_mm = np.linspace(0, 4.0 * r_spot_input, 500)
    sigma_final_mpa = spline_real(r_final_mm)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Pontos que você clicou (convertidos)
    ax2.plot(x_real_mm, y_real_mpa, 'ro', label='Pontos Selecionados (Escalados)')
    
    # Curva suave final
    ax2.plot(r_final_mm, sigma_final_mpa, 'g-', linewidth=2, label='Perfil Final Real (Spline)')
    
    ax2.set_title(f"Perfil de Tensão Residual Real\n(Spot: {r_spot_input}mm | Ref: {sigma_ref_input} MPa)")
    ax2.set_xlabel("Distância Radial (mm)")
    ax2.set_ylabel("Tensão Residual (MPa)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Linha de zero para referência
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)

    # --- 7. SALVAR ARQUIVO ---
    nome_arquivo = "curva_real_calibrada.pkl"
    with open(nome_arquivo, "wb") as f:
        pickle.dump(spline_real, f)

    print(f"\n>>> SUCESSO!")
    print(f"A curva foi convertida e salva em '{nome_arquivo}'.")
    print(f"Para usar: carregue o arquivo e chame a função passando raio em mm.")
    print(f"Exemplo: tensao = funcao_carregada(0.5)  # Retorna MPa em 0.5mm")
    
    plt.show()