"""
Demo en la terminal.
Uso:  python play.py            -> juega el agente entrenado (necesita q_table.npy)
      python play.py --azar     -> juega un agente al azar, para comparar
      python play.py --humano   -> juegas tu: espacio o W salta, S se agacha
"""
import os
import sys
import time
import numpy as np
from dino_env import (DinoEnv, obstaculos, dino_x, dino_ancho, dino_alto, dino_alto_agachado,
                      n_acciones, NOMBRES_ACCIONES, nada, saltar, agacharse, pajaro)

ANCHO_PANTALLA, ESCALA_X, ESCALA_Y, FILAS = 78, 8, 12, 13  # 1 caracter = 8 px de ancho, 12 de alto

# Dibujos del dino, de arriba hacia abajo (solo decoracion: el choque usa la caja)
DINO_DE_PIE = ["   ▄██",
               "   ██▀",
               "▀████ ",
               " █  █ "]
DINO_AGACHADO = ["    ▄▄",
                 "▀█████",
                 " █  █ "]
FPS = 30


class DinoVisible(DinoEnv):
    """El mismo juego, pero dibuja cada cuadro en la terminal."""
    ultima_accion = "nada"

    def _cuadro(self):
        resultado = super()._cuadro()     # la logica del juego no cambia
        self._dibujar()
        time.sleep(1 / FPS)               # pausa para que se vea a velocidad humana
        return resultado

    def _dibujar(self):
        lienzo = [[" "] * ANCHO_PANTALLA for _ in range(FILAS)]

        def pintar(x0, x1, y0, y1, simbolo):
            for c in range(int(x0 // ESCALA_X), int(x1 // ESCALA_X) + 1):
                for f in range(int(y0 // ESCALA_Y), int((y1 - 1) // ESCALA_Y) + 1):
                    if 0 <= c < ANCHO_PANTALLA and 0 <= f < FILAS:
                        lienzo[FILAS - 1 - f][c] = simbolo

        # dino: se dibuja desde la fila de sus pies hacia arriba
        dibujo = DINO_AGACHADO if self.agachado else DINO_DE_PIE
        fila_pies, col_inicio = int(self.y // ESCALA_Y), dino_x // ESCALA_X
        for i, linea in enumerate(reversed(dibujo)):
            f = fila_pies + i
            for j, simbolo in enumerate(linea):
                if simbolo != " " and 0 <= f < FILAS:
                    lienzo[FILAS - 1 - f][col_inicio + j] = simbolo

        for x, tipo in self.obstaculos:
            ancho, y_inf, y_sup = obstaculos[tipo]
            if tipo == pajaro:
                pintar(x, x + ancho - 1, y_inf, y_sup, "░")       # zona que ocupa el pajaro
                pintar(x, x + ancho - 1, y_inf, y_inf + 1, "▼")   # su parte de abajo
            else:
                pintar(x, x + ancho - 1, y_inf, y_sup, "▓")

        texto = "\n".join("".join(fila) for fila in lienzo)
        estado = (f"Obstaculos superados: {self.puntaje}   velocidad: {self.vel:.1f}   "
                  f"accion: {self.ultima_accion}").ljust(ANCHO_PANTALLA)
        print("\033[H" + texto + "\n" + "▔" * ANCHO_PANTALLA + "\n" + estado, flush=True)


class Teclado:
    """Lee las teclas sin detener el juego (Windows, Mac y Linux)."""

    def __enter__(self):
        if os.name != "nt":
            import termios, tty
            self.fd = sys.stdin.fileno()
            self.config_original = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)            # leer tecla por tecla, sin esperar Enter
        return self

    def __exit__(self, *args):
        if os.name != "nt":
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.config_original)

    def teclas(self):
        """Devuelve las teclas presionadas desde la ultima vez que se pregunto."""
        leidas = ""
        if os.name == "nt":
            import msvcrt
            while msvcrt.kbhit():
                leidas += msvcrt.getwch()
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                leidas += sys.stdin.read(1)
        return leidas.lower()


def jugar_humano(env):
    agachado_restante = 0
    terminado = False
    env.reset()
    with Teclado() as teclado:
        while not terminado:
            t = teclado.teclas()
            if " " in t or "w" in t:
                accion, agachado_restante = saltar, 0
            else:
                if "s" in t:
                    agachado_restante = 15        # cada S lo mantiene agachado 15 cuadros
                accion = agacharse if agachado_restante > 0 else nada
                agachado_restante = max(0, agachado_restante - 1)
            env.ultima_accion = NOMBRES_ACCIONES[accion]
            _, _, terminado = env.step(accion)
            if accion == saltar:
                teclado.teclas()      # descarta lo que se presiono en el aire (evita saltos seguidos)


def jugar_maquina(env, al_azar):
    if not al_azar:
        if not os.path.exists("q_table.npy"):
            sys.exit("No existe q_table.npy. Primero entrena con: python train.py")
        Q = np.load("q_table.npy")
    rng = np.random.default_rng()
    estado, terminado = env.reset(), False
    while not terminado:
        accion = int(rng.integers(n_acciones)) if al_azar else int(np.argmax(Q[estado]))
        env.ultima_accion = NOMBRES_ACCIONES[accion]
        estado, _, terminado = env.step(accion)


if __name__ == "__main__":
    os.system("")                    # activa los codigos de pantalla en Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # para los simbolos del dibujo
    except Exception:
        pass
    print("\033[2J\033[H", end="")   # limpia la pantalla una sola vez
    env = DinoVisible()

    if "--humano" in sys.argv:
        print("Espacio o W: saltar   S: agacharse   Empieza en 3 segundos...")
        time.sleep(3)
        jugar_humano(env)
    else:
        jugar_maquina(env, al_azar="--azar" in sys.argv)

    if env.pasos >= env.max_pasos:
        print(f"\nLlego al tope de la partida sin chocar. Puntaje: {env.puntaje}")
    else:
        print(f"\nChoco. Fin de la partida. Puntaje: {env.puntaje}")