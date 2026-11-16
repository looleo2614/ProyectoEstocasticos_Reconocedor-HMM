"""Evaluación de reconocimiento temprano por prefijo de trazo (25%, 50%, 75%, 100%).

Evalúa únicamente sobre el conjunto de prueba independiente definido en split.json.
Uso:
  python -m scripts.evaluate_prefixes --mode 2d
  python -m scripts.evaluate_prefixes --mode 3d
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.FeatureExtractor import LETTERS, extract_2d, extract_3d, map_2d_to_sphere
from src.HMMEngine import HMM, classify

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/raw/glifo-dataset-2026-11-04.json'
RATIOS = (0.25, 0.50, 0.75, 1.00)


def truncate_sample(sample: list, ratio: float) -> list:
    """Conserva el primer `ratio` de puntos respetando el orden de los trazos."""
    if ratio >= 1.0:
        return sample
    total_pts = sum(len(stroke) for stroke in sample)
    target = max(1, math.ceil(total_pts * ratio))
    truncated = []
    taken = 0
    for stroke in sample:
        if taken >= target:
            break
        k = min(len(stroke), target - taken)
        if k > 0:
            truncated.append(stroke[:k])
            taken += k
    return truncated


def main() -> None:
    parser = argparse.ArgumentParser(description='Evalúa precisión por prefijo en el split de test')
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--models-dir', type=Path, default=None)
    args = parser.parse_args()

    models_dir = args.models_dir or (ROOT / 'models' / args.mode)
    split_path = models_dir / 'split.json'
    if not split_path.is_file():
        raise FileNotFoundError(f'No se encontró {split_path}. Ejecute primero scripts.train.')

    data = json.loads(args.data.read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    models = {ch: HMM.load(models_dir / f'{ch}.json') for ch in LETTERS}

    sample_model = next(iter(models.values()))
    sectors = int(sample_model.metadata.get('sectors', 8))
    step_units = float(sample_model.step_units)

    temperature = 1.0
    cal_path = models_dir / 'calibration.json'
    if cal_path.is_file():
        temperature = float(json.loads(cal_path.read_text(encoding='utf-8')).get('temperature', 1.0))

    synthetic_3d = False
    eval_path = models_dir / 'evaluation.json'
    if eval_path.is_file():
        synthetic_3d = bool(json.loads(eval_path.read_text(encoding='utf-8')).get('synthetic_3d', False))

    summary = []
    per_letter_50: dict[str, list[bool]] = {ch: [] for ch in LETTERS}

    print(f'\nEVALUACIÓN POR PREFIJO ({args.mode.upper()}) — Split independiente (T={temperature:.2f})')
    print('-' * 74)
    print(f'{"Prefijo":<10} | {"Top-1":<16} | {"Top-3":<16} | {"Conf. media":<12} | {"Long. obs"}')
    print('-' * 74)

    for ratio in RATIOS:
        top1_ok = 0
        top3_ok = 0
        total = 0
        conf_sum = 0.0
        len_sum = 0

        for ch in LETTERS:
            test_indices = split[ch]['test_indices']
            for idx in test_indices:
                prefix_strokes = truncate_sample(data[ch][idx], ratio)
                if args.mode == '2d':
                    obs = extract_2d(prefix_strokes, sectors=sectors, step_units=step_units)
                else:
                    s3 = map_2d_to_sphere(prefix_strokes) if synthetic_3d else prefix_strokes
                    obs = extract_3d(s3, sectors=sectors, step_units=step_units)

                ranking = classify(models, obs, temperature=temperature)
                top_chars = [item[0] for item in ranking[:3]]
                is_top1 = (top_chars[0] == ch)
                is_top3 = (ch in top_chars)

                top1_ok += int(is_top1)
                top3_ok += int(is_top3)
                total += 1
                conf_sum += ranking[0][1]
                len_sum += len(obs)

                if math.isclose(ratio, 0.50):
                    per_letter_50[ch].append(is_top1)

        acc1 = 100.0 * top1_ok / max(total, 1)
        acc3 = 100.0 * top3_ok / max(total, 1)
        mean_conf = 100.0 * conf_sum / max(total, 1)
        mean_len = len_sum / max(total, 1)

        print(
            f'{int(ratio * 100):>6}%    | '
            f'{top1_ok:>3}/{total:<3} ({acc1:>5.2f}%) | '
            f'{top3_ok:>3}/{total:<3} ({acc3:>5.2f}%) | '
            f'{mean_conf:>9.2f}%   | '
            f'{mean_len:>6.1f}'
        )
        summary.append({
            'prefix_ratio': ratio,
            'top1_correct': top1_ok,
            'top3_correct': top3_ok,
            'total': total,
            'top1_accuracy': round(acc1, 2),
            'top3_accuracy': round(acc3, 2),
            'mean_top1_confidence': round(mean_conf, 2),
            'mean_obs_length': round(mean_len, 1),
        })

    print('-' * 74)

    letter_rates_50 = sorted(
        ((ch, 100.0 * sum(vals) / max(len(vals), 1)) for ch, vals in per_letter_50.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    best_5 = ', '.join(f'{c} ({r:.0f}%)' for c, r in letter_rates_50[:5])
    worst_5 = ', '.join(f'{c} ({r:.0f}%)' for c, r in letter_rates_50[-5:])
    print(f'Mayor discriminación al 50%: {best_5}')
    print(f'Mayor ambigüedad al 50%:     {worst_5}')

    out_report = models_dir / 'prefix_evaluation.json'
    out_report.write_text(
        json.dumps({'mode': args.mode, 'results': summary}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f'Reporte guardado en {out_report}\n')


if __name__ == '__main__':
    main()