"""Evalúa el conjunto de prueba reservado por scripts.train, sin reentrenar.
Uso: py -m scripts.evaluate_holdout --mode 2d
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from src.FeatureExtractor import LETTERS, extract_2d, extract_3d, map_2d_to_sphere
from src.HMMEngine import HMM, classify

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    args = parser.parse_args()
    folder = ROOT / 'models' / args.mode
    split = json.loads((folder / 'split.json').read_text(encoding='utf-8'))
    dataset = json.loads((ROOT / 'data/raw/glifo-dataset-2026-10-31.json').read_text(encoding='utf-8'))
    models = {letter: HMM.load(folder / f'{letter}.json') for letter in LETTERS}
    correct = total = 0
    print('ADVERTENCIA: 3D es sintético, no mide generalización a escritura esférica real.' if args.mode == '3d' else 'Evaluación de partición reservada por muestra; escritores no separados.')
    print('Letra  Aciertos  Total  Confusiones principales')
    for letter in LETTERS:
        indices = split[letter]['test_indices']
        errors = Counter()
        hits = 0
        for index in indices:
            sample = dataset[letter][index]
            if args.mode == '2d':
                observations = extract_2d(sample, sectors=8, step_units=10.)
            else:
                observations = extract_3d(map_2d_to_sphere(sample), sectors=8, step_units=.018)
            predicted = classify(models, observations)[0][0]
            hits += predicted == letter
            if predicted != letter:
                errors[predicted] += 1
        correct += hits
        total += len(indices)
        print(f'{letter:>3} {hits:>8} {len(indices):>6}  {errors.most_common(4)}')
    print(f'Exactitud global: {correct}/{total} = {correct/total:.2%}' if total else 'Sin muestras de prueba')

if __name__ == '__main__':
    main()
