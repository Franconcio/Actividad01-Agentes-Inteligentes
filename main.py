"""
main.py — Ejecuta tu agente en el Grid World.

Uso:
    python main.py

Configuración:
    Modifica las variables de abajo para cambiar el mapa.
"""
import random

from entorno import GridWorld
from mi_agente import MiAgente

# ── Configuración del mapa ───────────────────────
FILAS     = 10       # Alto del mapa
COLUMNAS  = 10       # Ancho del mapa
SEMILLA = random.randint(0,10000)      # Randomizar para que sean mapas aleatorios
PAREDES   = 0.30     # Porcentaje de paredes (0.0 a 0.40)
VELOCIDAD = 0.15     # Segundos entre pasos (menor = más rápido)
MAX_PASOS = 50      # Máximo de pasos antes de rendirse
# ─────────────────────────────────────────────────

# Crear el mundo
mundo = GridWorld(
    filas=FILAS,
    columnas=COLUMNAS,
    semilla=SEMILLA,
    porcentaje_paredes=PAREDES,
)

# Mostrar mapa en consola
mundo.mostrar_mapa()

# Crear tu agente
agente = MiAgente()

# Ejecutar con animación
resultado = mundo.animar(agente, max_pasos=MAX_PASOS, velocidad=VELOCIDAD)