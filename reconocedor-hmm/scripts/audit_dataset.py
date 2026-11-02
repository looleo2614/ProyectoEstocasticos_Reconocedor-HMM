"""Chequeos básicos de dataset Glifo. Ejecutable en terminal."""
from pathlib import Path
import json
import collections
from src.FeatureExtractor import LETTERS, validate_strokes

ROOT = Path(__file__).resolve().parents[1]
file = ROOT / 'data/raw/glifo-dataset-2026-11-02.json'
source = json.loads(file.read_text(encoding='utf-8'))
assert set(source) == set(LETTERS)
print('Letra | Muestras | Trazos | Puntos | Invalidas')
print('-'*50)
for char in LETTERS:
    samples = source[char]
    invalid = 0
    for sample in samples:
        try:
            validate_strokes(sample)
        except ValueError:
            invalid += 1
    strokes = sum(len(s) for s in samples)
    points = sum(len(stroke) for s in samples for stroke in s)
    print(f'{char:>5} | {len(samples):>8} | {strokes:>6} | {points:>6} | {invalid:>8}')
