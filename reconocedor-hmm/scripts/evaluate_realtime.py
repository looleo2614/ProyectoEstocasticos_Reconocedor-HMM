"""Evaluación reproducible por prefijos y calibración sobre el holdout existente.

No entrena, no modifica los HMM y no utiliza muestras reservadas para ajustar
hiperparámetros. El modo 3D es EXCLUSIVAMENTE sintético.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from src.FeatureExtractor import LETTERS, extract_2d, extract_3d, map_2d_to_sphere
from src.HMMEngine import HMM, classify

ROOT = Path(__file__).resolve().parents[1]


def truncate_sample(sample, fraction):
    """Prefijo por cantidad de puntos, conservando límites entre trazos."""
    total = sum(len(s) for s in sample)
    keep = max(1, int(round(total * fraction)))
    result = []
    for stroke in sample:
        if keep <= 0:
            break
        taken = stroke[:keep]
        if taken:
            result.append(taken)
        keep -= len(taken)
    return result


def evaluate(mode='2d'):
    folder = ROOT / 'models' / mode
    split = json.loads((folder / 'split.json').read_text(encoding='utf-8'))
    data = json.loads((ROOT / 'data/raw/glifo-dataset-2026-11-02.json').read_text(encoding='utf-8'))
    models = {ch: HMM.load(folder / f'{ch}.json') for ch in LETTERS}
    sectors = {m.metadata.get('sectors', 8) for m in models.values()}
    steps = {m.step_units for m in models.values()}
    if len(sectors) != 1 or len(steps) != 1:
        raise ValueError('Modelos con configuraciones incompatibles')
    sectors, step = sectors.pop(), steps.pop()
    fractions = (.25, .50, .75, 1.0)
    totals = {f: 0 for f in fractions}
    hits = {f: 0 for f in fractions}
    confusion = Counter()
    confidences, correctness = [], []
    for letter in LETTERS:
        for index in split[letter]['test_indices']:
            sample = data[letter][index]
            for fraction in fractions:
                part = truncate_sample(sample, fraction)
                obs = (extract_2d(part, sectors=sectors, step_units=step)
                       if mode == '2d' else
                       extract_3d(map_2d_to_sphere(part), sectors=sectors, step_units=step))
                if not obs:
                    continue
                ranking = classify(models, obs)
                pred, probability, _ = ranking[0]
                totals[fraction] += 1
                hits[fraction] += (pred == letter)
                if fraction == 1.0:
                    if pred != letter:
                        confusion[(letter, pred)] += 1
                    confidences.append(probability)
                    correctness.append(int(pred == letter))
    bins = []
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        indexes = [i for i, p in enumerate(confidences) if lo <= p < hi or (hi == 1 and p == 1)]
        if indexes:
            bins.append({'interval': [round(float(lo), 2), round(float(hi), 2)],
                         'n': len(indexes),
                         'confianza_media': round(float(np.mean([confidences[i] for i in indexes])), 5),
                         'exactitud': round(float(np.mean([correctness[i] for i in indexes])), 5)})
    return {'mode': mode, 'synthetic_3d': mode == '3d',
            'split': 'holdout por muestra, no por autor',
            'prefixes': [{'fraction': f, 'correct': hits[f], 'total': totals[f],
                          'accuracy': round(hits[f]/totals[f], 6) if totals[f] else None}
                         for f in fractions],
            'confusions': [{'real': a, 'predicho': b, 'casos': n}
                           for (a, b), n in confusion.most_common()],
            'calibration_bins': bins,
            'warning': ('Los porcentajes son puntuaciones posteriores de modelos generativos; '
                        'NO equivalen a precisión calibrada. El 3D sintético NO valida la esfera real.')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    args = parser.parse_args()
    report = evaluate(args.mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
