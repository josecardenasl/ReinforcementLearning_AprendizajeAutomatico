"""Q-learning en una cuadrícula 4x4 (S = inicio, G = meta, X = trampa)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

N = 4
START, GOAL, TRAP = (0, 0), (3, 3), (1, 3)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # arriba, derecha, abajo, izquierda
ARROWS = ["↑", "→", "↓", "←"]

def step(s, a):
    r = min(max(s[0] + MOVES[a][0], 0), N - 1)
    c = min(max(s[1] + MOVES[a][1], 0), N - 1)
    s2 = (r, c)
    if s2 == GOAL: return s2, 10, True
    if s2 == TRAP: return s2, -10, True
    return s2, -1, False                       # costo por paso

Q = np.zeros((N, N, 4))
alpha, gamma, eps, episodes = 0.1, 0.9, 0.2, 500
returns = []

for ep in range(episodes):
    s, done, total = START, False, 0
    while not done:
        a = np.random.randint(4) if np.random.rand() < eps else np.argmax(Q[s])
        s2, r, done = step(s, a)
        Q[s][a] += alpha * (r + gamma * np.max(Q[s2]) * (not done) - Q[s][a])
        s, total = s2, total + r
    returns.append(total)

# Política aprendida
print("Política aprendida:")
for r in range(N):
    print(" ".join("G" if (r, c) == GOAL else "X" if (r, c) == TRAP
                    else ARROWS[np.argmax(Q[r, c])] for c in range(N)))

# Curva de aprendizaje
plt.plot(np.convolve(returns, np.ones(20) / 20, mode="valid"))
plt.xlabel("Episodio"); plt.ylabel("Retorno (media móvil)")
plt.title("Q-learning: curva de aprendizaje"); plt.grid(alpha=.3)
plt.savefig("curva_aprendizaje.png", dpi=150)
