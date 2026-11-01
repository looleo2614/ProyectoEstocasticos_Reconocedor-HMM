"""Extracción geométrica autónoma 2D y esférica 3D.

K=8 (u opcionalmente 16) sectores ANGULARES. Además hay dos eventos
NO angulares: PEN_UP (levantamiento) y DOT (trazo degenerado/punto).
El tiempo se valida pero no se usa como denominador de velocidad: en el
JSON de Glifo existen marcas temporales repetidas.
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

LETTERS = tuple('abcdefghijklmnñopqrstuvwxyz')
K = 8
PEN_UP = K
DOT = K + 1
VOCAB_SIZE = K + 2
EPS = 1e-10


def validate_strokes(strokes: Sequence, dimensions: int = 2) -> None:
    if not isinstance(strokes, (list, tuple)) or not strokes:
        raise ValueError('Se requiere al menos un trazo')
    count = 0
    for stroke in strokes:
        if not isinstance(stroke, (list, tuple)) or not stroke:
            raise ValueError('Trazo vacío o mal formado')
        previous_t = -float('inf')
        for point in stroke:
            if not isinstance(point, (list, tuple)) or len(point) < dimensions:
                raise ValueError('Punto con dimensiones insuficientes')
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                raise ValueError('Coordenadas/tiempo no numéricos o no finitos')
            if dimensions == 2 and (not 0 <= point[0] <= 1000 or not 0 <= point[1] <= 1000):
                raise ValueError('Coordenadas 2D fuera de rango [0,1000]')
            if len(point) > dimensions:
                t = point[-1]
                if t < previous_t:
                    raise ValueError('Tiempos decrecientes dentro de un trazo')
                previous_t = t
            count += 1
    if count == 0:
        raise ValueError('Muestra sin puntos')


def quantize_direction(dx: float, dy: float, sectors: int = K) -> int:
    """0 = derecha, se incrementa antihorario. En pantalla se invierte y."""
    if sectors not in (8, 16):
        raise ValueError('Los sectores deben ser 8 o 16')
    angle = math.atan2(dy, dx) % (2 * math.pi)
    return int(math.floor(angle / (2 * math.pi / sectors) + .5)) % sectors


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample(points: np.ndarray, nsteps: int) -> np.ndarray:
    """Remuestreo uniforme en distancia de arco (no por tiempo)."""
    if len(points) < 2:
        return points
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths = np.r_[0.0, np.cumsum(ds)]
    valid = np.r_[True, np.diff(lengths) > EPS]
    if valid.sum() < 2:
        return points[:1]
    lengths = lengths[valid]
    coords = points[valid]
    targets = np.linspace(0, lengths[-1], max(2, nsteps + 1))
    out = np.column_stack([np.interp(targets, lengths, coords[:, k]) for k in range(points.shape[1])])
    return out


def extract_2d(strokes: Sequence, max_steps: int = 64, sectors: int = K, step_units: float | None = None) -> list[int]:
    """Discretiza direcciones 2D y eventos entre trazos.

    Coordenadas Glifo: x hacia derecha, y hacia abajo. Se cambia dy de signo
    para cuantizar con ángulos matemáticos convencionales.
    """
    if sectors not in (8, 16):
        raise ValueError('Solo K=8 o K=16')
    validate_strokes(strokes, 2)
    coords = [np.asarray([p[:2] for p in stroke], dtype=float) for stroke in strokes]
    lengths = [_length(p) for p in coords]
    total = sum(lengths)
    result = []
    for j, (points, length) in enumerate(zip(coords, lengths)):
        if j:
            result.append(sectors)  # PEN_UP
        if length < 4.0:
            result.append(sectors + 1)  # DOT
            continue
        count = max(1, round(length / step_units)) if step_units else max(1, round(max_steps * length / max(total, EPS)))
        simplified = _resample(points, count)
        for p, q in zip(simplified[:-1], simplified[1:]):
            dx, dy = q - p
            if math.hypot(dx, dy) > EPS:
                result.append(quantize_direction(float(dx), float(-dy), sectors))
    return result


def normalize_sphere(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p[:3], dtype=float)
    norm = np.linalg.norm(a)
    if norm < EPS:
        raise ValueError('El punto no puede ser el origen de la esfera')
    return a / norm


def to_spherical(p: Sequence[float]) -> tuple[float, float]:
    """(theta, phi) = (longitud, latitud), ambos en radianes."""
    x, y, z = normalize_sphere(p)
    return math.atan2(float(x), float(z)), math.asin(float(np.clip(y, -1, 1)))


def sphere_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([math.cos(phi) * math.sin(theta), math.sin(phi),
                     math.cos(phi) * math.cos(theta)])


def wrap_angle(delta: float) -> float:
    return (delta + math.pi) % (2 * math.pi) - math.pi


def tangent_deltas(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    th1, ph1 = to_spherical(a)
    th2, ph2 = to_spherical(b)
    return wrap_angle(th2 - th1) * math.cos((ph1 + ph2) * 0.5), ph2 - ph1


def extract_3d(strokes: Sequence, max_steps: int = 64, sectors: int = K, step_units: float | None = None) -> list[int]:
    """Dirección angular tangencial sobre una esfera, y eventos del lápiz."""
    if sectors not in (8, 16):
        raise ValueError('Solo K=8 o K=16')
    validate_strokes(strokes, 3)
    paths = [[normalize_sphere(p) for p in stroke] for stroke in strokes]
    lens = [sum(math.acos(float(np.clip(np.dot(a, b), -1, 1)))
                for a, b in zip(path[:-1], path[1:])) for path in paths]
    total = sum(lens)
    result: list[int] = []
    for j, (path, length) in enumerate(zip(paths, lens)):
        if j:
            result.append(sectors)
        if length < .005:
            result.append(sectors + 1)
            continue
        count = max(1, round(length / step_units)) if step_units else max(1, round(max_steps * length / max(total, EPS)))
        simplified = _resample(np.asarray(path), count)
        simplified = [normalize_sphere(p) for p in simplified]
        for p, q in zip(simplified[:-1], simplified[1:]):
            dx, dy = tangent_deltas(p, q)
            if math.hypot(dx, dy) > EPS:
                result.append(quantize_direction(dx, dy, sectors))
    return result


def map_2d_to_sphere(strokes: Sequence, angular_span: float = 0.9) -> list[list[list[float]]]:
    """Proyección sintética *de desarrollo*; NO equivale a captura 3D real.

    Conserva el orden del trazo y transforma las posiciones del lienzo en
    coordenadas sobre un parche esférico central.
    """
    result = []
    for stroke in strokes:
        out = []
        for p in stroke:
            theta = (float(p[0]) - 500.) / 500. * angular_span
            phi = (500. - float(p[1])) / 500. * angular_span
            out.append(sphere_from_angles(theta, phi).tolist())
        result.append(out)
    return result


def symbol_name(k: int, sectors: int = K) -> str:
    if k == sectors:
        return 'PEN_UP'
    if k == sectors + 1:
        return 'DOT'
    return f'DIRECCION_{k}'
