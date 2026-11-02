"""Diagnóstico reproducible, sin modificar modelos ni entrenar en la GUI.
Uso: python -m scripts.diagnose_letters
"""
import json
from collections import Counter
from pathlib import Path
from src.FeatureExtractor import extract_2d
from src.HMMEngine import HMM, classify

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'data/raw/glifo-dataset-2026-11-02.json').read_text(encoding='utf-8'))
models = {p.stem:HMM.load(p) for p in (ROOT/'models/2d').glob('?.json') if p.stem != 'evaluation'}
models.update({p.stem:HMM.load(p) for p in (ROOT/'models/2d').glob('ñ.json')})
for letter in ('i','l','j','q'):
    samples = data[letter]
    print(f'\n{letter}: {len(samples)} muestras, trazos {dict(Counter(len(s) for s in samples))}')
    correct = 0
    errors = Counter()
    for sample in samples:
        obs = extract_2d(sample, sectors=8, step_units=10.)
        ranking = classify(models, obs)
        pred = ranking[0][0]
        correct += pred == letter
        if pred != letter: errors[pred] += 1
    print(f'Coincidencias sobre dataset completo (incluye entrenamiento): {correct}/{len(samples)}')
    print('Confusiones:', dict(errors))
print('\nATENCIÓN: esta comprobación no mide generalización; use split de prueba para ello.')
