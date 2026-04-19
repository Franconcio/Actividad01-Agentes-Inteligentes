"""
entorno.py — Entorno Grid World con animación.
╔══════════════════════════════════════════════════╗
║  NO MODIFICAR ESTE ARCHIVO                       ║
║  El estudiante solo necesita saber:              ║
║    - Qué recibe en 'percepcion'                  ║
║    - Qué debe retornar en 'decidir()'            ║
╚══════════════════════════════════════════════════╝
"""

import random
import copy
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation


# ─────────────────────────────────────────────────
#  Clase base del agente (interfaz pública)
# ─────────────────────────────────────────────────

class Agente:
    """
    Clase base que todo agente debe heredar.

    El estudiante solo necesita:
        1. Heredar de Agente
        2. Implementar el método decidir(percepcion) → str

    Percepción que recibe:
    ───────────────────────
    percepcion = {
        'posicion':    (fila, columna),      # Dónde estoy
        'arriba':      'libre'|'pared'|'meta'|None,  # Qué hay arriba
        'abajo':       'libre'|'pared'|'meta'|None,  # Qué hay abajo
        'izquierda':   'libre'|'pared'|'meta'|None,  # Qué hay a la izq
        'derecha':     'libre'|'pared'|'meta'|None,  # Qué hay a la der

        # OPCIONAL — brújula hacia la meta (no es percepción real,
        # es información global). Usarla o no es decisión del alumno.
        'direccion_meta': (vertical, horizontal),
            donde vertical  = 'arriba' | 'abajo'    | 'ninguna'
            y     horizontal = 'izquierda' | 'derecha' | 'ninguna'
    }

    Nota: None significa fuera del mapa (borde).

    Debe retornar:
    ──────────────
    'arriba', 'abajo', 'izquierda' o 'derecha'
    """

    ACCIONES = ['arriba', 'abajo', 'izquierda', 'derecha']

    DELTAS = {
        'arriba':    (-1,  0),
        'abajo':     ( 1,  0),
        'izquierda': ( 0, -1),
        'derecha':   ( 0,  1),
    }

    def __init__(self, nombre="Agente"):
        self.nombre = nombre

    def decidir(self, percepcion: dict) -> str:
        """Sobreescribir este método."""
        raise NotImplementedError("Debes implementar el método decidir()")

    def al_iniciar(self):
        """Opcional: se llama al comenzar la simulación."""
        pass


# ─────────────────────────────────────────────────
#  Grid World
# ─────────────────────────────────────────────────

class GridWorld:
    """Entorno de cuadrícula. Genera un mapa y ejecuta agentes."""

    def __init__(self, filas=10, columnas=10, semilla=42, porcentaje_paredes=0.20):
        self.filas = filas
        self.columnas = columnas
        self.semilla = semilla

        self.inicio = (0, 0)
        self.meta = (filas - 1, columnas - 1)

        # Generar mapa: 0 = libre, 1 = pared
        self.mapa = self._generar_mapa(semilla, porcentaje_paredes)

    def _generar_mapa(self, semilla, pct_paredes):
        rng = random.Random(semilla)
        mapa = [[0] * self.columnas for _ in range(self.filas)]

        # Proteger camino en L (fila 0 → columna final)
        protegidas = set()
        protegidas.add(self.inicio)
        protegidas.add(self.meta)
        for c in range(self.columnas):
            protegidas.add((0, c))
        for r in range(self.filas):
            protegidas.add((r, self.columnas - 1))

        # Colocar paredes
        celdas = [
            (r, c) for r in range(self.filas) for c in range(self.columnas)
            if (r, c) not in protegidas
        ]
        rng.shuffle(celdas)
        n_paredes = int(self.filas * self.columnas * pct_paredes)
        for r, c in celdas[:n_paredes]:
            mapa[r][c] = 1

        return mapa

    def _percepcion(self, pos):
        """Genera el diccionario de percepción para una posición."""
        r, c = pos
        p = {'posicion': pos}

        nombres = {
            'arriba':    (-1, 0),
            'abajo':     ( 1, 0),
            'izquierda': ( 0,-1),
            'derecha':   ( 0, 1),
        }

        for nombre, (dr, dc) in nombres.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.filas and 0 <= nc < self.columnas):
                p[nombre] = None
            elif (nr, nc) == self.meta:
                p[nombre] = 'meta'
            elif self.mapa[nr][nc] == 1:
                p[nombre] = 'pared'
            else:
                p[nombre] = 'libre'

        # Dirección general a la meta
        gr, gc = self.meta
        vert = 'arriba' if gr < r else ('abajo' if gr > r else 'ninguna')
        horiz = 'izquierda' if gc < c else ('derecha' if gc > c else 'ninguna')
        p['direccion_meta'] = (vert, horiz)

        return p

    # ─────────────────────────────────────────────
    #  Ejecución (sin animación, para métricas)
    # ─────────────────────────────────────────────

    def ejecutar(self, agente, max_pasos=300):
        """Ejecuta el agente y retorna resultados."""
        agente.al_iniciar()
        pos = self.inicio
        camino = [pos]
        visitadas = {pos}

        for paso in range(1, max_pasos + 1):
            percepcion = self._percepcion(pos)
            accion = agente.decidir(percepcion)

            if accion not in Agente.ACCIONES:
                accion = 'abajo'

            dr, dc = Agente.DELTAS[accion]
            nr, nc = pos[0] + dr, pos[1] + dc

            # Validar movimiento
            if (0 <= nr < self.filas and 0 <= nc < self.columnas
                    and self.mapa[nr][nc] != 1):
                pos = (nr, nc)

            camino.append(pos)
            visitadas.add(pos)

            if pos == self.meta:
                return {
                    'camino': camino, 'pasos': paso,
                    'llego': True, 'celdas_visitadas': len(visitadas)
                }

        return {
            'camino': camino, 'pasos': max_pasos,
            'llego': False, 'celdas_visitadas': len(visitadas)
        }

    # ─────────────────────────────────────────────
    #  Animación
    # ─────────────────────────────────────────────

    def animar(self, agente, max_pasos=300, velocidad=0.15):
        """
        Ejecuta el agente EN TIEMPO REAL con animación.

        Cada frame de la animación:
            1. Da la percepción al agente
            2. El agente decide
            3. Se ejecuta el movimiento
            4. Se dibuja el resultado

        Parámetros:
            agente    – instancia que hereda de Agente
            max_pasos – límite de pasos
            velocidad – segundos entre pasos (menor = más rápido)
        """
        agente.al_iniciar()

        # Estado mutable de la simulación
        estado = {
            'pos': self.inicio,
            'paso': 0,
            'huellas': {self.inicio},
            'llego': False,
            'terminado': False,
        }

        # Colores
        C_LIBRE  = '#16213e'
        C_PARED  = '#0f3460'
        C_INICIO = '#e94560'
        C_META   = '#53d769'
        C_AGENTE = '#f5c542'
        C_HUELLA = '#533483'
        C_BORDE  = '#1a1a2e'

        # Configurar figura
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        fig.suptitle(
            f'  {agente.nombre}',
            fontsize=14, fontweight='bold', color='#e0e0e0', y=0.96
        )

        # Header en consola
        print(f"\n{'═' * 60}")
        print(f"  Ejecutando: {agente.nombre}")
        print(f"  Mapa: {self.filas}×{self.columnas}  |  Inicio: {self.inicio}  |  Meta: {self.meta}")
        print(f"{'═' * 60}")
        print(f"  {'Paso':>7}  │  {'Movimiento':<40}  │  Estado")
        print(f"  {'─' * 7}──┼──{'─' * 40}──┼──{'─' * 15}")

        def paso_y_dibujar(frame):
            # ── Ejecutar un paso de la simulación ──
            if not estado['terminado']:
                estado['paso'] += 1

                percepcion = self._percepcion(estado['pos'])
                accion = agente.decidir(percepcion)

                if accion not in Agente.ACCIONES:
                    accion = 'abajo'

                pos_antes = estado['pos']
                dr, dc = Agente.DELTAS[accion]
                nr, nc = estado['pos'][0] + dr, estado['pos'][1] + dc

                movio = False
                if (0 <= nr < self.filas and 0 <= nc < self.columnas
                        and self.mapa[nr][nc] != 1):
                    estado['pos'] = (nr, nc)
                    movio = True

                estado['huellas'].add(estado['pos'])

                # ── Imprimir en consola ──
                simbolo = '→' if movio else '✗'
                destino = percepcion[accion] if accion in percepcion else '???'
                print(f"  Paso {estado['paso']:3d}  │  "
                      f"{pos_antes} ─{accion:>10s}─▶ {estado['pos']}  "
                      f"│  {simbolo} {'OK' if movio else f'bloqueado ({destino})'}")

                if estado['pos'] == self.meta:
                    estado['llego'] = True
                    estado['terminado'] = True
                    print(f"\n  ✓ ¡LLEGÓ A LA META en {estado['paso']} pasos!")
                elif estado['paso'] >= max_pasos:
                    estado['terminado'] = True
                    print(f"\n  ✗ Límite de {max_pasos} pasos alcanzado.")

            # ── Dibujar ──
            ax.clear()
            ax.set_xlim(-0.5, self.columnas - 0.5)
            ax.set_ylim(self.filas - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

            for r in range(self.filas):
                for c in range(self.columnas):
                    if (r, c) == self.inicio:
                        color = C_INICIO
                    elif (r, c) == self.meta:
                        color = C_META
                    elif self.mapa[r][c] == 1:
                        color = C_PARED
                    elif (r, c) in estado['huellas']:
                        color = C_HUELLA
                    else:
                        color = C_LIBRE

                    rect = mpatches.FancyBboxPatch(
                        (c - 0.45, r - 0.45), 0.9, 0.9,
                        boxstyle="round,pad=0.04",
                        facecolor=color, edgecolor=C_BORDE,
                        linewidth=0.8
                    )
                    ax.add_patch(rect)

            # Etiquetas A y B
            sr, sc = self.inicio
            ax.text(sc, sr, 'A', ha='center', va='center',
                    fontsize=12, color='white', fontweight='bold')
            mr, mc = self.meta
            ax.text(mc, mr, 'B', ha='center', va='center',
                    fontsize=12, color='white', fontweight='bold')

            # Agente
            pr, pc = estado['pos']
            circulo = plt.Circle((pc, pr), 0.3, color=C_AGENTE, zorder=10)
            ax.add_patch(circulo)

            # Info
            if estado['llego']:
                texto = f"✓ ¡LLEGÓ!   |   Paso {estado['paso']}   |   Posición: {estado['pos']}"
                color_txt = C_META
            elif estado['terminado']:
                texto = f"✗ No llegó   |   Paso {estado['paso']}   |   Posición: {estado['pos']}"
                color_txt = C_INICIO
            else:
                texto = f"Paso {estado['paso']}   |   Posición: {estado['pos']}"
                color_txt = '#aaaaaa'

            ax.text(0.5, -0.04, texto, transform=ax.transAxes,
                    ha='center', fontsize=11, color=color_txt,
                    fontfamily='monospace')

            # Detener la animación si terminó
            if estado['terminado']:
                anim.event_source.stop()

        # Generador de frames: se detiene cuando la simulación termina
        def gen_frames():
            paso = 0
            while paso < max_pasos:
                yield paso
                paso += 1
                if estado['terminado']:
                    return

        # Crear animación en tiempo real
        anim = FuncAnimation(
            fig, paso_y_dibujar,
            frames=gen_frames,
            interval=int(velocidad * 1000),
            repeat=False,
            save_count=max_pasos,
        )

        plt.subplots_adjust(bottom=0.08, top=0.92)
        plt.show()

        # Resumen en consola
        print(f"\n{'═' * 40}")
        print(f"  Agente: {agente.nombre}")
        print(f"  Pasos:  {estado['paso']}")
        print(f"  Llegó:  {'Sí ✓' if estado['llego'] else 'No ✗'}")
        print(f"  Celdas visitadas: {len(estado['huellas'])}")
        print(f"{'═' * 40}")

        return {
            'pasos': estado['paso'],
            'llego': estado['llego'],
            'celdas_visitadas': len(estado['huellas']),
        }

    # ─────────────────────────────────────────────
    #  Mostrar mapa en consola
    # ─────────────────────────────────────────────

    def mostrar_mapa(self):
        """Imprime el mapa en consola."""
        simbolos = {0: '·', 1: '█'}
        print(f"\nMapa {self.filas}×{self.columnas}  (A=inicio, B=meta)\n")
        for r in range(self.filas):
            fila = ''
            for c in range(self.columnas):
                if (r, c) == self.inicio:
                    fila += ' A'
                elif (r, c) == self.meta:
                    fila += ' B'
                else:
                    fila += ' ' + simbolos[self.mapa[r][c]]
            print(fila)
        print()