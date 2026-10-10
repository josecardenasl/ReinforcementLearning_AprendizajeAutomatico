"""
Q-learning tabular para el juego del dinosaurio.
Uso:  python train.py
Genera: q_table.npy, curva_aprendizaje.png, resultados.json
"""
import json
import numpy as np
from dino_env import DinoEnv, forma_estado, n_acciones

episodios = 3000
alpha = 0.1 # learning rate
gamma = 0.95 # cuanto le importa el futuro
eps_inicial, eps_final, eps_decay = 1.0, 0.01, 0.998   # que tanto explora al azar
semilla = 0


def elegir_accion(q, estado, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_acciones)) # explora: accion al azar
    return int(np.argmax(q[estado]))


def entrenar():
    rng = np.random.default_rng(semilla)
    env = DinoEnv(semilla=semilla)
    q = np.zeros(forma_estado + (n_acciones,))  
    epsilon = eps_inicial
    puntajes = []

    for ep in range(episodios):
        estado = env.reset()
        terminado = False
        while not terminado:
            accion = elegir_accion(q, estado, epsilon, rng)
            sig_estado, recompensa, terminado = env.step(accion)

            # Actualizacion de Q-learning
            mejor_futuro = 0.0 if terminado else np.max(q[sig_estado])
            objetivo = recompensa + gamma * mejor_futuro
            q[estado + (accion,)] += alpha * (objetivo - q[estado + (accion,)])

            estado = sig_estado

        epsilon = max(eps_final, epsilon * eps_decay)
        puntajes.append(env.puntaje)
        if (ep + 1) % 500 == 0:
            print(f"Partida {ep+1} | epsilon {epsilon:.3f} | "
                  f"promedio ultimas 100: {np.mean(puntajes[-100:]):.1f}")
    return q, puntajes


def evaluar(politica, n_partidas=200, semilla_eval=123):
    """Juega sin explorar, con obstaculos nuevos, y devuelve los puntajes."""
    env = DinoEnv(semilla=semilla_eval)
    res = []
    for _ in range(n_partidas):
        estado, terminado = env.reset(), False
        while not terminado:
            estado, _, terminado = env.step(politica(estado))
        res.append(env.puntaje)
    return res


if __name__ == "__main__":
    # Entrenar y guardar la tabla
    q, puntajes = entrenar()
    np.save("q_table.npy", q)

    # Comparar contra un agente al azar
    rng = np.random.default_rng(1)
    azar = evaluar(lambda s: int(rng.integers(n_acciones)))
    agente = evaluar(lambda s: int(np.argmax(q[s])))
    print(f"\nAgente al azar : {np.mean(azar):.2f} obstaculos por partida (max {max(azar)})")
    print(f"Agente entrenado: {np.mean(agente):.2f} obstaculos por partida "
          f"(min {min(agente)}, max {max(agente)})")

    # Guardar los numeros
    with open("resultados.json", "w") as f:
        json.dump({
            "episodios": episodios, "alpha": alpha, "gamma": gamma,
            "azar_media": float(np.mean(azar)), "azar_max": int(max(azar)),
            "agente_media": float(np.mean(agente)), "agente_min": int(min(agente)),
            "agente_max": int(max(agente)),
            "media_cada_100": [float(np.mean(puntajes[i:i + 100])) for i in range(0, episodios, 100)],
        }, f, indent=1)

    # Grafica de la curva de aprendizaje
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        media = np.convolve(puntajes, np.ones(100) / 100, mode="valid")   # promedio de cada 100
        plt.figure(figsize=(8, 4))
        plt.plot(puntajes, alpha=0.25, label="puntaje por partida")
        plt.plot(range(99, episodios), media, label="media movil (100)")
        plt.xlabel("Partida"); plt.ylabel("Obstaculos superados"); plt.legend()
        plt.title("Curva de aprendizaje, Q-learning"); plt.tight_layout()
        plt.savefig("curva_aprendizaje.png", dpi=150)
    except ImportError:
        print("matplotlib no esta instalado: no se genero la grafica")