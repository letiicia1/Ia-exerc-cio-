import numpy as np

# Função sigmoide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Função para treinar a rede
def treinar_rede(X, y, camadas, taxa_aprendizado=0.5, epocas=10000, verbose=True):
    np.random.seed(42)

    # Inicializa pesos e biases
    pesos = []
    biases = []

    for i in range(len(camadas) - 1):
        pesos.append(np.random.uniform(-1, 1, (camadas[i], camadas[i + 1])))
        biases.append(np.random.uniform(-1, 1, (1, camadas[i + 1])))

    # Treinamento
    for epoca in range(epocas):
        # Forward
        a = [X]
        for i in range(len(pesos)):
            z = np.dot(a[-1], pesos[i]) + biases[i]
            a.append(sigmoid(z))

        # Backward
        erro = y - a[-1]
        deltas = [erro * sigmoid_deriv(a[-1])]

        for i in range(len(pesos) - 2, -1, -1):
            deltas.insert(0, deltas[0].dot(pesos[i + 1].T) * sigmoid_deriv(a[i + 1]))

        # Atualiza pesos e biases
        for i in range(len(pesos)):
            pesos[i] += a[i].T.dot(deltas[i]) * taxa_aprendizado
            biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * taxa_aprendizado

        # Exibe perda
        if verbose and epoca % (epocas // 5) == 0:
            loss = np.mean(np.square(erro))
            print(f"[Epoch {epoca}] Loss = {loss:.4f}")

    print(f"Treinamento finalizado com erro médio: {np.mean(np.square(erro)):.4f}")
    return pesos, biases

# Função para prever saídas após o treinamento
def prever(X, pesos, biases):
    a = X
    for i in range(len(pesos)):
        z = np.dot(a, pesos[i]) + biases[i]
        a = sigmoid(z)
    return a

# ============================================================
# 🔹 Parte 1 – Rede XOR
# ============================================================
print("\n=== Treinando Rede XOR ===")

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([[0],[1],[1],[0]])

camadas_xor = [2, 2, 1]

pesos_xor, biases_xor = treinar_rede(X_xor, y_xor, camadas_xor, taxa_aprendizado=0.5, epocas=10000)

# Resultados finais
print("\nResultados da Rede XOR:")
for i in range(len(X_xor)):
    saida = prever(X_xor[i].reshape(1, -1), pesos_xor, biases_xor)
    print(f"Entrada: {X_xor[i]} → Saída prevista: {saida[0][0]:.4f} | Esperado: {y_xor[i][0]}")

# ============================================================
# 🔹 Parte 2 – Rede de 7 Segmentos
# ============================================================
print("\n=== Treinando Rede de 7 Segmentos ===")

# Entradas (7 segmentos A-G)
X_seg = np.array([
    [1,1,1,1,1,1,0],  # 0
    [0,1,1,0,0,0,0],  # 1
    [1,1,0,1,1,0,1],  # 2
    [1,1,1,1,0,0,1],  # 3
    [0,1,1,0,0,1,1],  # 4
    [1,0,1,1,0,1,1],  # 5
    [1,0,1,1,1,1,1],  # 6
    [1,1,1,0,0,0,0],  # 7
    [1,1,1,1,1,1,1],  # 8
    [1,1,1,1,0,1,1]   # 9
])

# Saídas esperadas (binário de 0 a 9)
y_seg = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1]
])

camadas_seg = [7, 5, 4]

pesos_seg, biases_seg = treinar_rede(X_seg, y_seg, camadas_seg, taxa_aprendizado=0.3, epocas=20000)

# Resultados finais
print("\nResultados da Rede de 7 Segmentos:")
for i in range(len(X_seg)):
    saida = prever(X_seg[i].reshape(1, -1), pesos_seg, biases_seg)
    saida_bin = np.round(saida).astype(int)
    print(f"Dígito {i}: saída prevista {saida_bin.tolist()} | Esperado: {y_seg[i].tolist()}")
