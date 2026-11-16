# algoritmos HMM escritos desde cero. Única dependencia matemática: NumPy
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import random
from pathlib import Path
from typing import Iterable
import numpy as np

_EPS = 1.e-300


def logsumexp(values: np.ndarray) -> float:
    # suma logarítmica estable sin SciPy
    a = np.asarray(values, dtype=float)
    mx = float(np.max(a))
    if not np.isfinite(mx):
        return -math.inf
    return mx + math.log(float(np.exp(a - mx).sum()))


@dataclass
class HMM:
    A: np.ndarray
    B: np.ndarray
    pi: np.ndarray
    label: str = ''
    median_length: int = 50
    step_units: float = 10.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        self.pi = np.asarray(self.pi, dtype=float)
        self.validate()

    @property
    def states(self) -> int:
        return self.A.shape[0]

    @property
    def symbols(self) -> int:
        return self.B.shape[1]

    def validate(self) -> None:
        n = len(self.pi)
        if self.A.shape != (n, n) or self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError('Dimensiones incompatibles de A, B, pi')
        for arr in (self.A, self.B, self.pi):
            if not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError('Probabilidades negativas o no finitas')
        if not (np.allclose(self.A.sum(axis=1), 1.)
                and np.allclose(self.B.sum(axis=1), 1.)
                and np.isclose(self.pi.sum(), 1.)):
            raise ValueError('A, B y pi deben estar normalizadas')

    def _observations(self, obs: Iterable[int]) -> np.ndarray:
        o = np.asarray(list(obs), dtype=int)
        if o.ndim != 1 or not len(o) or (o < 0).any() or (o >= self.symbols).any():
            raise ValueError('Secuencia de observaciones vacía o inválida')
        return o

    def forward(self, obs: Iterable[int]) -> tuple[np.ndarray, np.ndarray, float]:
        o = self._observations(obs)
        T, n = len(o), self.states
        alpha = np.zeros((T, n), dtype=float)
        scales = np.empty(T, dtype=float)
        alpha[0] = self.pi * self.B[:, o[0]]
        if alpha[0].sum() == 0:
            raise ValueError('Observación inicial imposible bajo este HMM')
        scales[0] = float(alpha[0].sum())
        alpha[0] /= scales[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, o[t]]
            if alpha[t].sum() == 0:
                raise ValueError('Secuencia imposible bajo este HMM')
            scales[t] = float(alpha[t].sum())
            alpha[t] /= scales[t]
        return alpha, scales, float(np.log(scales).sum())

    def backward(self, obs: Iterable[int], scales: np.ndarray) -> np.ndarray:
        o = self._observations(obs)
        if len(scales) != len(o):
            raise ValueError('Escalas y observaciones deben tener igual longitud')
        beta = np.ones((len(o), self.states), dtype=float)
        for t in range(len(o)-2, -1, -1):
            beta[t] = self.A @ (self.B[:, o[t+1]] * beta[t+1])
            beta[t] /= max(float(scales[t+1]), _EPS)
        return beta

    def posterior(self, obs: Iterable[int]) -> np.ndarray:
        o = self._observations(obs)
        alpha, scales, _ = self.forward(o)
        beta = self.backward(o, scales)
        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)
        return gamma

    def viterbi(self, obs: Iterable[int]) -> tuple[list[int], float]:
        o = self._observations(obs)
        with np.errstate(divide='ignore'):
            ln_a, ln_b, ln_pi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = ln_pi + ln_b[:, o[0]]
        pointers = np.empty((len(o), self.states), dtype=np.int32)
        pointers[0] = -1
        for t in range(1, len(o)):
            scores = delta[:, None] + ln_a
            pointers[t] = np.argmax(scores, axis=0)
            delta = scores[pointers[t], np.arange(self.states)] + ln_b[:, o[t]]
        state = int(np.argmax(delta))
        score = float(delta[state])
        path = [state]
        for t in range(len(o)-1, 0, -1):
            state = int(pointers[t, state])
            path.append(state)
        path.reverse()
        return path, score

    def predict_symbols(self, obs: Iterable[int], horizon: int = 30,
                        stop_symbol: int | None = None) -> list[int]:
        # predicción iterativa mediante evolución natural de la cadena de Markov
        o = self._observations(obs)
        state = self.posterior(o)[-1]
        steps = min(max(0, int(self.median_length) - len(o)), horizon)
        future = []
        state_indices = np.arange(self.states, dtype=float)
        saturated_steps = 0

        for _ in range(steps):
            state = state @ self.A
            expected_state = float(state @ state_indices)
            if expected_state >= self.states - 1.15:
                saturated_steps += 1
                if saturated_steps > 2:
                    break

            emissions = state @ self.B
            symbol = int(np.argmax(emissions))
            if stop_symbol is not None and symbol >= stop_symbol:
                break
            future.append(symbol)
        return future

    def predict_direction_vectors(
        self,
        obs: Iterable[int],
        sectors: int = 8,
        horizon: int = 32,
        smoothing: float = 0.65,
    ) -> list[tuple[float, float]]:
        # calcula vectores unitarios esperados por paso evitando saltos de 45°
        o = self._observations(obs)
        state = self.posterior(o)[-1]
        steps = min(max(0, int(self.median_length) - len(o)), horizon)
        if steps <= 0:
            return []

        angles = np.arange(sectors, dtype=float) * (2.0 * math.pi / sectors)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        state_indices = np.arange(self.states, dtype=float)

        vectors: list[tuple[float, float]] = []
        prev_vx, prev_vy = 0.0, 0.0
        saturated_steps = 0

        for step_idx in range(steps):
            state = state @ self.A
            expected_state = float(state @ state_indices)
            if expected_state >= self.states - 1.15:
                saturated_steps += 1
                if saturated_steps > 2:
                    break

            emissions = state @ self.B
            dir_probs = emissions[:sectors]
            stop_prob = float(emissions[sectors:].sum()) if len(emissions) > sectors else 0.0
            if stop_prob > 0.45:
                break

            vx = float(dir_probs @ cos_a)
            vy = float(dir_probs @ sin_a)
            mag = math.hypot(vx, vy)
            if mag < 0.18:
                break

            vx /= mag
            vy /= mag
            if step_idx > 0:
                vx = smoothing * vx + (1.0 - smoothing) * prev_vx
                vy = smoothing * vy + (1.0 - smoothing) * prev_vy
                norm = math.hypot(vx, vy) or 1.0
                vx /= norm
                vy /= norm

            prev_vx, prev_vy = vx, vy
            vectors.append((vx, vy))

        return vectors

    def save(self, filename: str | Path) -> None:
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            json.dump({'label': self.label, 'A': self.A.tolist(),
                       'B': self.B.tolist(), 'pi': self.pi.tolist(),
                       'median_length': self.median_length,
                       'step_units': self.step_units,
                       'metadata': self.metadata}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filename: str | Path) -> 'HMM':
        with open(filename, encoding='utf-8') as f:
            return cls(**json.load(f))


def left_right_model(
    states: int,
    symbols: int,
    seed: int = 0,
    label: str = '',
    allow_skip: bool = True,
) -> HMM:
    if states < 2 or symbols < 2:
        raise ValueError('HMM requiere >=2 estados y >=2 símbolos')
    rng = random.Random(seed)
    A = np.zeros((states, states))
    for i in range(states - 1):
        if allow_skip and i < states - 2:
            A[i, i] = .68
            A[i, i + 1] = .24
            A[i, i + 2] = .08
        else:
            A[i, i] = .72
            A[i, i + 1] = .28
    A[-1, -1] = 1.
    B = np.array([[rng.gammavariate(3.0, 1.0) for _ in range(symbols)]
                  for _ in range(states)], dtype=float)
    B /= B.sum(axis=1, keepdims=True)
    pi = np.zeros(states)
    pi[0] = 1.
    return HMM(A, B, pi, label=label)


def segmental_init(model: HMM, obs_seq: list[np.ndarray], smoothing: float = 1.e-2) -> None:
    # inicializa B alineando uniformemente cada secuencia de entrenamiento a los N estados
    n, m = model.states, model.symbols
    accum_b = np.zeros((n, m), dtype=float)
    for obs in obs_seq:
        if len(obs) == 0:
            continue
        # estado 0 concentra los metadatos iniciales (aspecto y conteo de trazos)
        head = min(2, len(obs))
        for k in obs[:head]:
            accum_b[0, k] += 1.0
        tail = obs[head:] if len(obs) > head else obs
        chunks = np.array_split(tail, n)
        for state_idx, chunk in enumerate(chunks):
            for k in chunk:
                accum_b[state_idx, k] += 1.0
    model.B = accum_b + smoothing
    model.B /= model.B.sum(axis=1, keepdims=True)


def baum_welch(model: HMM, sequences: list[list[int]], epochs: int = 25,
               smoothing: float = 1.e-2, tolerance: float = 1.e-4,
               init_segmental: bool = True) -> list[float]:
    """Algoritmo EM Baum–Welch multisecuencia, sin bibliotecas HMM externas.
    Mantiene la topología inicial: transiciones inicialmente imposibles
    permanecen imposibles. Pseudocuentas de suavizado evitan emisiones 0.
    Cada iteración suma gamma y xi sobre secuencias completas. """
    obs_seq = [model._observations(seq) for seq in sequences if seq]
    if not obs_seq:
        raise ValueError('Sin secuencias de entrenamiento')
    if init_segmental:
        segmental_init(model, obs_seq, smoothing=smoothing)
    n, m = model.states, model.symbols
    mask = (model.A > 0).astype(float)
    history = []
    for epoch in range(epochs):
        accum_a = np.zeros((n, n), dtype=float)
        accum_b = np.zeros((n, m), dtype=float)
        pi_acc = np.zeros(n, dtype=float)
        total_loglik = 0.
        for obs in obs_seq:
            alpha, scales, ll = model.forward(obs)
            beta = model.backward(obs, scales)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)
            total_loglik += ll
            pi_acc += gamma[0]
            for t in range(len(obs)-1):
                xi = (alpha[t, :, None] * model.A *
                      (model.B[:, obs[t+1]] * beta[t+1])[None, :])
                xi /= max(float(xi.sum()), _EPS)
                accum_a += xi
            for k in np.unique(obs):
                accum_b[:, k] += gamma[obs == k].sum(axis=0)
        model.pi = np.zeros(n, dtype=float)
        model.pi[0] = 1.0
        for j in range(n):
            row = (accum_a[j] + smoothing * mask[j]) * mask[j]
            model.A[j] = row / max(float(row.sum()), _EPS)
        model.B = accum_b + smoothing
        model.B /= model.B.sum(axis=1, keepdims=True)
        history.append(float(total_loglik))
        if epoch > 1 and abs(history[-1] - history[-2]) < tolerance:
            break
    model.validate()
    return history


def classify(models: dict[str, HMM], obs: Iterable[int],
             priors: dict[str, float] | None = None,
             temperature: float = 1.0) -> list[tuple[str, float, float]]:
    # (letra, P(letra|O), log P(O|λ_letra)), ordenado descendente
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature debe ser positiva y finita')
    obs = list(obs)
    if not models:
        return []
    if not obs:
        return [(c, 1. / len(models), 0.) for c in sorted(models)]
    logs = []
    for character, model in models.items():
        ll = model.forward(obs)[2]
        prior = 1. / len(models) if priors is None else float(priors[character])
        logs.append((character, ll + math.log(max(prior, _EPS)), ll))
    lognorm = logsumexp(np.asarray([v[1] / temperature for v in logs]))
    results = [(ch, math.exp(score / temperature - lognorm), ll) for ch, score, ll in logs]
    return sorted(results, key=lambda x: x[1], reverse=True)


def project_2d(model: HMM, obs: list[int], cursor_xy: tuple[float, float],
               sectors: int = 8, horizon: int = 32) -> list[tuple[float, float]]:
    # reconstrucción continua desde el cursor usando direcciones esperadas
    if not obs:
        return []
    vectors = model.predict_direction_vectors(obs, sectors=sectors, horizon=horizon)
    px, py = float(cursor_xy[0]), float(cursor_xy[1])
    points = [(px, py)]
    for vx, vy in vectors:
        px += model.step_units * vx
        py -= model.step_units * vy  # coordenada gráfica +y hacia abajo
        points.append((px, py))
    return points if len(points) > 1 else []


def project_3d(model: HMM, obs: list[int], cursor_xyz: np.ndarray,
               sectors: int = 8, horizon: int = 32) -> list[np.ndarray]:
    # integración suavizada sobre la esfera unidad desde el último punto
    from src.FeatureExtractor import to_spherical, sphere_from_angles
    if not obs:
        return []
    vectors = model.predict_direction_vectors(obs, sectors=sectors, horizon=horizon)
    theta, phi = to_spherical(cursor_xyz)
    coords = [sphere_from_angles(theta, phi)]
    for vx, vy in vectors:
        delta_phi = model.step_units * vy
        delta_theta = model.step_units * vx / max(abs(math.cos(phi)), 0.15)
        phi = float(np.clip(phi + delta_phi, -math.pi / 2 + 1e-3, math.pi / 2 - 1e-3))
        theta += delta_theta
        coords.append(sphere_from_angles(theta, phi))
    return coords if len(coords) > 1 else []