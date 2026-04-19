# Navegación de Agente — Grid World

## Objetivo

Programar un agente que navegue desde el punto **A** hasta el punto **B**
en un mapa con paredes. Verás en tiempo real cómo tu agente toma decisiones.

## Archivos

```
├── entorno.py     ← El mundo (NO MODIFICAR)
├── mi_agente.py   ← ✏️  TU AGENTE (editar aquí)
└── main.py        ← Ejecutor (configurar y correr)
```

## Cómo empezar

### 1. Abre `mi_agente.py`

Ahí encontrarás la clase `MiAgente` con el método `decidir()` vacío.
Tu trabajo es programar la lógica de decisión.

### 2. Lo que recibes

Cada turno tu agente recibe un diccionario `percepcion`:

```python
percepcion = {
    'posicion':       (3, 5),                    # Tu fila y columna
    'arriba':         'libre',                   # Qué hay arriba
    'abajo':          'pared',                   # Qué hay abajo
    'izquierda':      'libre',                   # Qué hay a la izquierda
    'derecha':        None,                      # None = fuera del mapa
    'direccion_meta': ('abajo', 'derecha'),      # Hacia dónde queda B
}
```

Valores posibles: `'libre'`, `'pared'`, `'meta'`, `None` (borde).

### 3. Lo que retornas

Una de estas cadenas: `'arriba'`, `'abajo'`, `'izquierda'`, `'derecha'`

Si retornas un movimiento inválido, tu agente se queda quieto.

### 4. Ejecutar

```bash
python main.py
```

Verás una animación paso a paso de tu agente navegando.

## Configuración

En `main.py` puedes cambiar:

| Variable   | Default | Qué hace                         |
|------------|---------|----------------------------------|
| `FILAS`    | 10      | Alto del mapa                    |
| `COLUMNAS` | 10      | Ancho del mapa                   |
| `SEMILLA`  | 42      | Cambiar = otro mapa              |
| `PAREDES`  | 0.20    | Porcentaje de paredes (0 a 0.40) |
| `VELOCIDAD`| 0.15    | Segundos entre pasos             |

## Leyenda del mapa

| Color    | Significado         |
|----------|---------------------|
| Rojo     | A — Punto de inicio |
| Verde    | B — Meta            |
| Azul osc.| Pared (bloqueada)   |
| Azul clr.| Libre (transitable) |
| Morado   | Ya visitada         |
| Amarillo | Tu agente           |

## Consejos

- Empieza simple: haz que tu agente al menos se mueva hacia la meta.
- Usa `percepcion['direccion_meta']` como guía general.
- Si tu agente se queda atrapado, necesitas lógica para rodear paredes.
- Puedes agregar atributos a tu clase (`self.memoria`, `self.pasos`, etc.)
  para que tu agente recuerde cosas.
- Prueba con diferentes semillas para verificar que tu agente es robusto.
