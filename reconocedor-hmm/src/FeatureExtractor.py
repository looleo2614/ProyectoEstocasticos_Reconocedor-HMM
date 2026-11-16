"""Extracción geométrica autónoma 2D y esférica 3D con contexto espacial.
Símbolos emitidos (para K=8 o K=16 sectores angulares):
  - 0 .. K-1      : direcciones angulares (compatibles con project_2d/project_3d)
  - K             : PEN_UP (levantamiento entre trazos)
  - K + 1         : DOT (trazo degenerado/punto)
  - K+2 .. K+10   : ZONE_0 .. ZONE_8 (cuadrícula espacial 3x3 normalizada)
  - K + 11        : ASPECT_TALL (alto / ancho > 1.75, p. ej. l, i, t, d)
  - K + 12        : ASPECT_MED_TALL (1.25 < alto / ancho <= 1.75)
  - K + 13        : ASPECT_SQUARE (0.75 <= alto / ancho <= 1.25)
  - K + 14        : ASPECT_WIDE (alto / ancho < 0.75, p. ej. w, m)
  - K + 15        : STROKE_1 (glifo de 1 trazo, p. ej. v, w, c, o)
  - K + 16        : STROKE_2 (glifo de 2 trazos, p. ej. x, t, i, j, ñ)
  - K + 17        : STROKE_3PLUS (glifo de 3 o más trazos)
  - K + 18        : TURN_VALLEY (giro de bajada a subida, p. ej. v, u, w)
  - K + 19        : TURN_PEAK (giro de subida a bajada, p. ej. w, m, n)"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

LETTERS = tuple('abcdefghijklmnñopqrstuvwxyz')
K = 8
PEN_UP = K
DOT = K + 1
ZONE_OFFSET = K + 2
ASPECT_TALL = K + 11
ASPECT_MED_TALL = K + 12
ASPECT_SQUARE = K + 13
ASPECT_WIDE = K + 14
STROKE_1 = K + 15
STROKE_2 = K + 16
STROKE_3PLUS = K + 17
TURN_VALLEY = K + 18
TURN_PEAK = K + 19
VOCAB_SIZE = K + 20
EPS = 1e-10


def vocab_size_for(sectors: int = K) -> int:
    if sectors not in (8, 16):
        raise ValueError('Los sectores deben ser 8 o 16')
    return sectors + 20


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
    # 0 = derecha, se incrementa antihorario. En pantalla se invierte y
    if sectors not in (8, 16):
        raise ValueError('Los sectores deben ser 8 o 16')
    angle = math.atan2(dy, dx) % (2 * math.pi)
    return int(math.floor(angle / (2 * math.pi / sectors) + .5)) % sectors


def _aspect_symbol(width: float, height: float, sectors: int = K) -> int:
    tall_sym = sectors + 11
    med_tall_sym = sectors + 12
    square_sym = sectors + 13
    wide_sym = sectors + 14
    if width < EPS and height < EPS:
        return square_sym
    ratio = height / max(width, EPS)
    if ratio > 1.75:
        return tall_sym
    if ratio > 1.25:
        return med_tall_sym
    if ratio < 0.75:
        return wide_sym
    return square_sym


def _stroke_count_symbol(num_strokes: int, sectors: int = K) -> int:
    if num_strokes <= 1:
        return sectors + 15
    if num_strokes == 2:
        return sectors + 16
    return sectors + 17


def _zone_symbol_2d(x: float, y: float, sectors: int = K) -> int:
    # cuadrícula 3x3 sobre espacio normalizado
    col = 0 if x < 367.0 else (1 if x <= 633.0 else 2)
    row = 0 if y < 367.0 else (1 if y <= 633.0 else 2)
    return (sectors + 2) + row * 3 + col


def _zone_symbol_3d(theta: float, phi: float, sectors: int = K, target_span: float = 1.44) -> int:
    # cuadrícula 3x3 sobre parche esférico normalizado
    thresh = target_span / 6.0
    col = 0 if theta < -thresh else (1 if theta <= thresh else 2)
    row = 0 if phi > thresh else (1 if phi >= -thresh else 2)
    return (sectors + 2) + row * 3 + col


def _length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def normalize_strokes_2d(strokes: Sequence, target_size: float = 800.0) -> list[np.ndarray]:
    validate_strokes(strokes, 2)
    coords = [
        np.asarray([p[:2] for p in stroke], dtype=float)
        for stroke in strokes
    ]
    all_points = np.vstack(coords)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)

    center = (min_xy + max_xy) / 2.0
    width_height = max_xy - min_xy
    max_dimension = float(np.max(width_height))
    scale = target_size / max_dimension if max_dimension > EPS else 1.0

    return [
        (points - center) * scale + 500.0
        for points in coords
    ]


def _resample(points: np.ndarray, nsteps: int) -> np.ndarray:
    # remuestreo uniforme en distancia de arco (no por tiempo)
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
    return np.column_stack([
        np.interp(targets, lengths, coords[:, k])
        for k in range(points.shape[1])
    ])


def extract_2d(
    strokes: Sequence,
    max_steps: int = 64,
    sectors: int = K,
    step_units: float | None = None,
) -> list[int]:
    # discretiza direcciones 2D, zonas espaciales 3x3, vértices y eventos entre trazos
    if sectors not in (8, 16):
        raise ValueError('Solo K=8 o K=16')
    validate_strokes(strokes, 2)

    raw_coords = [np.asarray([p[:2] for p in stroke], dtype=float) for stroke in strokes]
    all_raw = np.vstack(raw_coords)
    wh = all_raw.max(axis=0) - all_raw.min(axis=0)

    coords = normalize_strokes_2d(strokes)
    lengths = [_length(p) for p in coords]
    total = sum(lengths)

    valley_sym = sectors + 18
    peak_sym = sectors + 19

    result: list[int] = [
        _aspect_symbol(float(wh[0]), float(wh[1]), sectors),
        _stroke_count_symbol(len(coords), sectors),
    ]
    for j, (points, length) in enumerate(zip(coords, lengths)):
        if j:
            result.append(sectors)  # PEN_UP
        start_zone = _zone_symbol_2d(float(points[0, 0]), float(points[0, 1]), sectors)
        result.append(start_zone)
        if length < 4.0:
            result.append(sectors + 1)  # DOT
            continue

        count = (
            max(1, round(length / step_units))
            if step_units
            else max(1, round(max_steps * length / max(total, EPS)))
        )
        simplified = _resample(points, count)
        last_zone = start_zone
        v_run = 0.0
        v_trend = 0  # -1 bajando, +1 subiendo

        for p, q in zip(simplified[:-1], simplified[1:]):
            dx, dy = q - p
            if math.hypot(dx, dy) > EPS:
                dy_math = float(-dy)
                if dy_math < -2.5:
                    if v_trend == 1 and v_run >= 25.0:
                        result.append(peak_sym)
                        v_run = dy_math
                        v_trend = -1
                    else:
                        v_run = min(0.0, v_run) + dy_math
                        if v_run <= -25.0:
                            v_trend = -1
                elif dy_math > 2.5:
                    if v_trend == -1 and v_run <= -25.0:
                        result.append(valley_sym)
                        v_run = dy_math
                        v_trend = 1
                    else:
                        v_run = max(0.0, v_run) + dy_math
                        if v_run >= 25.0:
                            v_trend = 1

                result.append(quantize_direction(float(dx), dy_math, sectors))
                z = _zone_symbol_2d(float(q[0]), float(q[1]), sectors)
                if z != last_zone:
                    result.append(z)
                    last_zone = z
    return result


def normalize_sphere(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p[:3], dtype=float)
    norm = np.linalg.norm(a)
    if norm < EPS:
        raise ValueError('El punto no puede ser el origen de la esfera')
    return a / norm


def to_spherical(p: Sequence[float]) -> tuple[float, float]:
    #(theta, phi) = (longitud, latitud), ambos en radianes
    x, y, z = normalize_sphere(p)
    return math.atan2(float(x), float(z)), math.asin(float(np.clip(y, -1, 1)))


def sphere_from_angles(theta: float, phi: float) -> np.ndarray:
    return np.array([
        math.cos(phi) * math.sin(theta),
        math.sin(phi),
        math.cos(phi) * math.cos(theta),
    ])


def wrap_angle(delta: float) -> float:
    return (delta + math.pi) % (2 * math.pi) - math.pi


def tangent_deltas(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    th1, ph1 = to_spherical(a)
    th2, ph2 = to_spherical(b)
    return wrap_angle(th2 - th1) * math.cos((ph1 + ph2) * 0.5), ph2 - ph1


def normalize_strokes_3d(
    strokes: Sequence,
    target_span: float = 1.44,
) -> tuple[list[np.ndarray], float, float]:
    # centra los trazos 3D en el polo frontal (0,0,1) y normaliza su escala angular
    validate_strokes(strokes, 3)
    paths = [np.asarray([normalize_sphere(p) for p in stroke], dtype=float) for stroke in strokes]
    all_pts = np.vstack(paths)

    mean_vec = all_pts.sum(axis=0)
    norm_m = float(np.linalg.norm(mean_vec))
    center = mean_vec / norm_m if norm_m > EPS else np.array([0.0, 0.0, 1.0])

    world_up = np.array([0.0, 1.0, 0.0])
    up_proj = world_up - float(np.dot(world_up, center)) * center
    if float(np.linalg.norm(up_proj)) < 1e-4:
        alt_up = np.array([0.0, 0.0, -1.0])
        up_proj = alt_up - float(np.dot(alt_up, center)) * center
    ey = up_proj / float(np.linalg.norm(up_proj))
    ex = np.cross(ey, center)
    ex = ex / float(np.linalg.norm(ex))

    local_angles: list[np.ndarray] = []
    for path in paths:
        angs = []
        for p in path:
            local_p = np.array([
                float(np.dot(p, ex)),
                float(np.dot(p, ey)),
                float(np.dot(p, center)),
            ])
            angs.append(to_spherical(local_p))
        local_angles.append(np.asarray(angs, dtype=float))

    all_ang = np.vstack(local_angles)
    min_tp = all_ang.min(axis=0)
    max_tp = all_ang.max(axis=0)
    mid_tp = (min_tp + max_tp) / 2.0
    wh = max_tp - min_tp
    max_dim = float(np.max(wh))
    scale = target_span / max_dim if max_dim > EPS else 1.0

    norm_paths: list[np.ndarray] = []
    for angs in local_angles:
        scaled = (angs - mid_tp) * scale
        norm_paths.append(
            np.asarray([sphere_from_angles(float(t), float(p)) for t, p in scaled], dtype=float)
        )

    return norm_paths, float(wh[0]), float(wh[1])


def extract_3d(
    strokes: Sequence,
    max_steps: int = 64,
    sectors: int = K,
    step_units: float | None = None,
) -> list[int]:
    # dirección angular tangencial sobre esfera normalizada, zonas 3x3, vértices y eventos
    if sectors not in (8, 16):
        raise ValueError('Solo K=8 o K=16')
    validate_strokes(strokes, 3)

    target_span = 1.44
    paths, raw_w, raw_h = normalize_strokes_3d(strokes, target_span=target_span)
    lens = [
        sum(
            math.acos(float(np.clip(np.dot(a, b), -1, 1)))
            for a, b in zip(path[:-1], path[1:])
        )
        for path in paths
    ]
    total = sum(lens)

    valley_sym = sectors + 18
    peak_sym = sectors + 19

    result: list[int] = [
        _aspect_symbol(raw_w, raw_h, sectors),
        _stroke_count_symbol(len(paths), sectors),
    ]
    for j, (path, length) in enumerate(zip(paths, lens)):
        if j:
            result.append(sectors)  # PEN_UP
        th0, ph0 = to_spherical(path[0])
        start_zone = _zone_symbol_3d(th0, ph0, sectors, target_span=target_span)
        result.append(start_zone)
        if length < .005:
            result.append(sectors + 1)  # DOT
            continue

        count = (
            max(1, round(length / step_units))
            if step_units
            else max(1, round(max_steps * length / max(total, EPS)))
        )
        simplified = _resample(path, count)
        simplified = [normalize_sphere(p) for p in simplified]
        last_zone = start_zone
        v_run = 0.0
        v_trend = 0

        for p, q in zip(simplified[:-1], simplified[1:]):
            dx, dy = tangent_deltas(p, q)
            if math.hypot(dx, dy) > EPS:
                if dy < -0.0045:
                    if v_trend == 1 and v_run >= 0.045:
                        result.append(peak_sym)
                        v_run = dy
                        v_trend = -1
                    else:
                        v_run = min(0.0, v_run) + dy
                        if v_run <= -0.045:
                            v_trend = -1
                elif dy > 0.0045:
                    if v_trend == -1 and v_run <= -0.045:
                        result.append(valley_sym)
                        v_run = dy
                        v_trend = 1
                    else:
                        v_run = max(0.0, v_run) + dy
                        if v_run >= 0.045:
                            v_trend = 1

                result.append(quantize_direction(dx, dy, sectors))
                th_q, ph_q = to_spherical(q)
                z = _zone_symbol_3d(th_q, ph_q, sectors, target_span=target_span)
                if z != last_zone:
                    result.append(z)
                    last_zone = z
    return result


def map_2d_to_sphere(strokes: Sequence, angular_span: float = 0.9) -> list[list[list[float]]]:
    # proyección sintética *de desarrollo*; NO equivale a captura 3D real
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
    if sectors + 2 <= k <= sectors + 10:
        return f'ZONA_{k - (sectors + 2)}'
    if k == sectors + 11:
        return 'ASPECT_TALL'
    if k == sectors + 12:
        return 'ASPECT_MED_TALL'
    if k == sectors + 13:
        return 'ASPECT_SQUARE'
    if k == sectors + 14:
        return 'ASPECT_WIDE'
    if k == sectors + 15:
        return 'STROKE_1'
    if k == sectors + 16:
        return 'STROKE_2'
    if k == sectors + 17:
        return 'STROKE_3PLUS'
    if k == sectors + 18:
        return 'TURN_VALLEY'
    if k == sectors + 19:
        return 'TURN_PEAK'
    return f'DIRECCION_{k}'