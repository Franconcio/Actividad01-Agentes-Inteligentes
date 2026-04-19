"""
mi_agente.py — Agente Basado en Utilidad con A*
=================================================
Medida de utilidad : minimizar número de pasos.
Algoritmo          : A* sobre mapa construido al explorar.
Heurística         : distancia Manhattan (admisible).
Anti-bucle         : celdas visitadas tienen costo extra en A*.
Límites del mapa   : se aprenden de los bordes percibidos (None).
"""

from entorno import Agente
import heapq


class MiAgente(Agente):
    """
    Agente Basado en Utilidad.

    Función de utilidad:
        U(acción) = −f(n)   donde   f(n) = g(n) + h(n)
        g(n) = pasos acumulados + visitas_previas(n) × PENALIZACION
        h(n) = distancia Manhattan al objetivo

    El agente maximiza U eligiendo siempre la acción con menor f,
    lo que equivale a encontrar el camino con menos pasos evitando
    repetir celdas ya exploradas.
    """

    PENALIZACION = 5   # costo extra por cada visita previa a una celda

    def __init__(self):
        super().__init__(nombre="Agente Basado en Utilidad (A*)")
        self.mapa     = {}   # (r,c) -> 'libre' | 'pared' | 'meta'
        self.meta_pos = None # posición exacta de la meta
        self.plan     = []   # lista de posiciones a seguir
        self.visitas  = {}   # (r,c) -> cantidad de visitas
        # Límites conocidos del mapa (aprendidos de bordes None)
        self.min_r = 0
        self.min_c = 0
        self.max_r = 50   # se reduce al detectar bordes
        self.max_c = 50

    # ----------------------------------------------------------
    def al_iniciar(self):
        self.mapa     = {}
        self.meta_pos = None
        self.plan     = []
        self.visitas  = {}
        self.min_r    = 0
        self.min_c    = 0
        self.max_r    = 50
        self.max_c    = 50

    # ----------------------------------------------------------
    def decidir(self, percepcion: dict) -> str:
        pos = percepcion["posicion"]

        # 1. Actualizar mapa y aprender límites del grid
        self._actualizar(pos, percepcion)

        # 2. Meta adyacente → máxima utilidad, ir directo
        for d in self.ACCIONES:
            if percepcion[d] == "meta":
                return d

        # 3. Elegir objetivo para A*
        objetivo = self._objetivo(pos, percepcion)

        # 4. Invalidar plan si el siguiente paso es pared
        if self.plan:
            nxt = self.plan[0]
            d   = self._dir(pos, nxt)
            if not d or percepcion.get(d) == "pared":
                self.mapa[nxt] = "pared"
                self.plan = []

        # 5. Replanificar si hace falta
        if not self.plan:
            camino = self._astar(pos, objetivo)
            if camino:
                self.plan = camino   # pasos desde pos hasta objetivo

        # 6. Ejecutar primer paso del plan
        if self.plan:
            nxt = self.plan.pop(0)
            d   = self._dir(pos, nxt)
            if d and percepcion.get(d) not in ("pared", None):
                return d
            self.plan = []   # paso inválido → replanificar luego

        # 7. Fallback: celda accesible menos visitada
        return self._fallback(pos, percepcion)

    # ----------------------------------------------------------
    def _actualizar(self, pos, perc):
        """Registra la celda actual, sus vecinos y los límites del mapa."""
        r, c = pos
        self.mapa[pos]   = "libre"
        self.visitas[pos] = self.visitas.get(pos, 0) + 1
        # Expandir límites conocidos
        self.max_r = max(self.max_r, r)
        self.max_c = max(self.max_c, c)

        dirs = {
            "arriba":    (-1,  0),
            "abajo":     ( 1,  0),
            "izquierda": ( 0, -1),
            "derecha":   ( 0,  1),
        }
        for d, (dr, dc) in dirs.items():
            valor = perc[d]
            if valor is None:
                # Borde del mapa: fijar límite en esa dirección
                if d == "arriba":    self.min_r = max(self.min_r, r)
                if d == "abajo":     self.max_r = min(self.max_r, r)
                if d == "izquierda": self.min_c = max(self.min_c, c)
                if d == "derecha":   self.max_c = min(self.max_c, c)
                continue
            nb = (r + dr, c + dc)
            if self.mapa.get(nb) != "pared":   # no sobreescribir paredes
                self.mapa[nb] = valor
            if valor == "meta":
                self.meta_pos = nb

    # ----------------------------------------------------------
    def _objetivo(self, pos, perc):
        """Meta exacta si ya fue vista; si no, estimada con la brújula."""
        if self.meta_pos:
            return self.meta_pos

        vert, horiz = perc["direccion_meta"]
        r, c = pos
        # Proyectar 12 celdas en la dirección de la brújula,
        # respetando los límites conocidos
        mr = r + (12 if vert  == "abajo"     else
                 -12 if vert  == "arriba"    else 0)
        mc = c + (12 if horiz == "derecha"   else
                 -12 if horiz == "izquierda" else 0)
        # Clampear dentro de los límites conocidos
        mr = max(self.min_r, min(self.max_r, mr))
        mc = max(self.min_c, min(self.max_c, mc))
        return (mr, mc)

    # ----------------------------------------------------------
    def _en_rango(self, pos):
        """Verifica que una posición esté dentro del mapa conocido."""
        r, c = pos
        return self.min_r <= r <= self.max_r and self.min_c <= c <= self.max_c

    # ----------------------------------------------------------
    def _astar(self, inicio, meta):
        """
        A* con penalización anti-bucle.

        Costo real:
            g(n) = pasos_acumulados + visitas_previas(n) × PENALIZACION

        Celdas fuera del mapa conocido → ignoradas.
        Celdas desconocidas dentro del mapa → asumidas libres.

        Retorna lista de posiciones a visitar (SIN incluir 'inicio').
        """
        h0   = self._manhattan(inicio, meta)
        heap = [(h0, 0, inicio, [])]
        best = {inicio: 0}

        while heap:
            _, g, pos, camino = heapq.heappop(heap)

            if pos == meta:
                return camino   # camino ya excluye la posición de inicio

            if g > best.get(pos, float("inf")):
                continue

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (pos[0] + dr, pos[1] + dc)

                if not self._en_rango(nb):          # fuera del mapa
                    continue
                if self.mapa.get(nb) == "pared":    # bloqueado
                    continue

                pen  = self.visitas.get(nb, 0) * self.PENALIZACION
                ng   = g + 1 + pen
                if ng < best.get(nb, float("inf")):
                    best[nb] = ng
                    nf = ng + self._manhattan(nb, meta)
                    heapq.heappush(heap, (nf, ng, nb, camino + [nb]))

        return []   # sin camino

    # ----------------------------------------------------------
    def _fallback(self, pos, perc):
        """Elige la celda accesible con menos visitas previas."""
        dirs = {
            "arriba":    (-1,  0),
            "abajo":     ( 1,  0),
            "izquierda": ( 0, -1),
            "derecha":   ( 0,  1),
        }
        opciones = []
        for d, (dr, dc) in dirs.items():
            if perc[d] in ("libre", "meta"):
                nb = (pos[0] + dr, pos[1] + dc)
                opciones.append((self.visitas.get(nb, 0), d))
        if opciones:
            opciones.sort()
            return opciones[0][1]
        return "abajo"

    # ----------------------------------------------------------
    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _dir(self, desde, hacia):
        dr = hacia[0] - desde[0]
        dc = hacia[1] - desde[1]
        tabla = {(-1,0):"arriba", (1,0):"abajo", (0,-1):"izquierda", (0,1):"derecha"}
        return tabla.get((dr, dc))