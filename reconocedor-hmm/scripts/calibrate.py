"""Calibración offline con un conjunto de VALIDACIÓN separado del test.

No reutilizar el holdout final para ajustar temperatura: eso contamina la evaluación.
Formato: JSON lista de {'scores': {'a': log_likelihood, ...}, 'label': 'a'}.
"""
import argparse
import json
import math
from pathlib import Path
from src.HMMEngine import logsumexp
import numpy as np


def nll(rows, temperature):
    if temperature <= 0:
        raise ValueError('Temperatura positiva requerida')
    losses = []
    for row in rows:
        scores = row['scores']
        if row['label'] not in scores:
            raise ValueError('Etiqueta ausente en puntuaciones')
        vals = np.asarray(list(scores.values()), dtype=float) / temperature
        target = list(scores).index(row['label'])
        losses.append(float(logsumexp(vals) - vals[target]))
    if not losses:
        raise ValueError('Se necesitan muestras de validación')
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_json', type=Path)
    parser.add_argument('--output', type=Path, default=Path('calibration.json'))
    args = parser.parse_args()
    rows = json.loads(args.validation_json.read_text(encoding='utf-8'))
    # Búsqueda determinista sin bibliotecas de optimización.
    candidates = np.exp(np.linspace(math.log(0.3), math.log(50), 301))
    best = min(candidates, key=lambda t: nll(rows, float(t)))
    result = {'temperature': float(best), 'validation_nll': nll(rows, float(best)),
              'validation_samples': len(rows), 'warning': 'No evaluar en estas mismas muestras'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result)

if __name__ == '__main__':
    main()
