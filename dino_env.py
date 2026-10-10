import numpy as np

nada, saltar, agacharse = 0, 1, 2
n_acciones = 3
NOMBRES_ACCIONES = ["nada", "saltar", "agacharse"]

cactus_peq, cactus_grande, pajaro = 0, 1, 2

obstaculos = {
    cactus_peq: (20, 0, 35), #ancho, altura donde empieza, altura donde termina
    cactus_grande: (50, 0, 50),
    pajaro: (40, 30, 100)
}

dino_x, dino_ancho = 50, 40 # el dino no se mueve, se mueve el mundo!!
dino_alto, dino_alto_agachado = 44, 25
impulso_salto, gravedad = 12.0, 0.8
vel_inicial, vel_max, aceleracion = 6.0, 13.0, 0.002

# Las recompensas
r_obstaculo, r_choque, r_salto = 1.0, -10.0, -0.01

tam_bin_dist, n_bins_dist = 25, 21
n_bins_vel = 4
n_tipos = 3
forma_estado = (n_bins_dist, n_bins_vel, n_tipos) #distancia al obstaculo, velocidad, tipo de obstaculo, agachado

class DinoEnv:
    def __init__(self, max_pasos=5000, semilla=None):
        self.max_pasos = max_pasos
        self.rng = np.random.default_rng(semilla)

    def reset(self):
        self.y, self.vy = 0.0, 0.0 #altura y velocidad vertical
        self.agachado = False
        self.vel = vel_inicial
        self.pasos, self.puntaje = 0, 0
        self.obstaculos = []
        self._nuevo_obstaculo(x_base=600)
        self._nuevo_obstaculo()
        return self._estado()

    def _nuevo_obstaculo(self, x_base=None):
        if x_base is None:
            x_base = self.obstaculos[-1][0] + self.vel * 40 + self.rng.uniform(0, 300)
        tipo = int(self.rng.choice([cactus_peq, cactus_grande, pajaro], p=[0.4, 0.35, 0.25]))
        self.obstaculos.append([x_base, tipo])

    def _estado(self):
        x, tipo = self.obstaculos[0]
        dist = x - (dino_x + dino_ancho)
        bin_dist = int(np.clip(dist // tam_bin_dist, 0, n_bins_dist - 1))
        bin_vel = int(min((self.vel - vel_inicial) / (vel_max - vel_inicial) * n_bins_vel, n_bins_vel - 1))
        return (bin_dist, bin_vel, tipo)

    def _choca(self):
        x, tipo = self.obstaculos[0]
        ancho, y_inf, y_sup = obstaculos[tipo]
        alto = dino_alto_agachado if self.agachado else dino_alto
        solapa_x = x < dino_x + dino_ancho and x + ancho > dino_x
        solapa_y = self.y < y_sup and self.y + alto > y_inf
        return solapa_x and solapa_y

    def _cuadro(self):
    #Física del salto
        if self.y > 0 or self.vy > 0:
            self.y += self.vy
            self.vy -= gravedad
            if self.y <= 0:
                self.y, self.vy = 0.0, 0.0

        #El mundo se mueve hacia el dino y acelera
        for o in self.obstaculos:
            o[0] -= self.vel
        self.vel = min(self.vel + aceleracion, vel_max)
        self.pasos += 1

        # Chocó? Castigo y fin de la partida
        if self._choca():
            return r_choque, True

        # Superó el obstáculo? Premio, se borra y sale uno nuevo
        recompensa = 0.0
        x, tipo = self.obstaculos[0]
        if x + obstaculos[tipo][0] < dino_x:
            self.obstaculos.pop(0)
            self._nuevo_obstaculo()
            self.puntaje += 1
            recompensa = r_obstaculo
        return recompensa, self.pasos >= self.max_pasos

    def step(self, accion):
        recompensa = 0.0
        if accion == saltar:
            self.vy = impulso_salto
            recompensa += r_salto
        self.agachado = accion == agacharse

        r, terminado = self._cuadro()
        recompensa += r
        while self.y > 0 and not terminado: # sigue en el aire
            r, terminado = self._cuadro()
            recompensa += r
        return self._estado(), recompensa, terminado