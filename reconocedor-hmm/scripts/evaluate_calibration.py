import argparse
import json
import math
from pathlib import Path

from src.FeatureExtractor import LETTERS, extract_2d, extract_3d, map_2d_to_sphere
from src.HMMEngine import HMM, classify

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/raw/glifo-dataset-2026-11-04.json'
DEFAULT_MODELS = ROOT / 'models/2d'


def evaluate(models, test, temperature):
    correct = 0
    total = 0
    nll = 0.0
    brier = 0.0

    for true_char, sequences in test.items():
        for obs in sequences:
            ranking = classify(models, obs, temperature=temperature)
            probabilities = {row[0]: row[1] for row in ranking}

            predicted = ranking[0][0]
            correct += predicted == true_char
            total += 1

            p_true = max(probabilities.get(true_char, 0.0), 1e-15)
            nll -= math.log(p_true)

            brier += sum(
                (probabilities.get(char, 0.0) -
                 (1.0 if char == true_char else 0.0)) ** 2
                for char in LETTERS
            )

    return {
        'accuracy': correct / total if total else 0.0,
        'nll': nll / total if total else 0.0,
        'brier': brier / total if total else 0.0,
        'correct': correct,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    parser.add_argument('--models', type=Path, default=None)
    args = parser.parse_args()

    models_path = args.models or ROOT / 'models' / args.mode

    data = json.loads(args.data.read_text(encoding='utf-8'))
    split = json.loads(
        (models_path / 'split.json').read_text(encoding='utf-8')
    )

    calibration_path = models_path / 'calibration.json'
    if calibration_path.exists():
        calibration = json.loads(
            calibration_path.read_text(encoding='utf-8')
        )
        temperature = float(calibration['temperature'])
    else:
        temperature = 1.0

    models = {
        char: HMM.load(models_path / f'{char}.json')
        for char in LETTERS
    }

    test = {}
    for char in LETTERS:
        test[char] = []
        
        for index in split[char]['test_indices']:
            if args.mode == '2d':
                obs = extract_2d(
                    data[char][index],
                    sectors=8,
                    step_units=10.0
                )
            else:
                strokes = map_2d_to_sphere(data[char][index])
                obs = extract_3d(
                    strokes,
                    sectors=8,
                    step_units=0.018
                )

            if obs:
                test[char].append(obs)
    for char in LETTERS:
        test[char] = []

        for index in split[char]['test_indices']:
            obs = extract_2d(
                data[char][index],
                sectors=8,
                step_units=10.0
            )
            if obs:
                test[char].append(obs)

    original = evaluate(models, test, temperature=1.0)
    calibrated = evaluate(models, test, temperature=temperature)

    print(f'\nEvaluación sobre el mismo test independiente: {original["total"]} muestras')
    print(f'Temperatura calibrada: {temperature:.4f}\n')

    for name, result in [
        ('Original (T=1)', original),
        (f'Calibrado (T={temperature:.2f})', calibrated),
    ]:
        print(name)
        print(f'  Accuracy: {result["accuracy"] * 100:.2f}%')
        print(f'  NLL:      {result["nll"]:.4f}')
        print(f'  Brier:    {result["brier"]:.4f}')
        print()

    print('Comparación:')
    print(f'  Diferencia NLL:   {calibrated["nll"] - original["nll"]:+.4f}')
    print(f'  Diferencia Brier: {calibrated["brier"] - original["brier"]:+.4f}')


if __name__ == '__main__':
    main()