"""Entrenamiento OFFLINE. Nunca se invoca desde la GUI.

Uso:
 python -m scripts.train --mode 2d --epochs 8
 python -m scripts.train --mode 3d --synthetic-3d --epochs 8

El 3D sintético solo sirve como prototipo hasta disponer de capturas 3D
reales sobre la esfera. No presentarlo como experimento con datos 3D reales.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import time
from src.FeatureExtractor import LETTERS, K, extract_2d, extract_3d, map_2d_to_sphere
from src.HMMEngine import left_right_model, baum_welch, classify

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data/raw/glifo-dataset-2026-11-04.json'


def main() -> None:
    parser = argparse.ArgumentParser(description='Entrena 27 HMM nativos desde JSON de trazos')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--mode', choices=['2d', '3d'], default='2d')
    parser.add_argument('--synthetic-3d', action='store_true',
                        help='Mapea 2D a esfera artificialmente; solo para desarrollo')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--states', type=int, default=6)
    parser.add_argument('--sectors', type=int, choices=[8, 16], default=8)
    parser.add_argument('--test-ratio', type=float, default=.2)
    parser.add_argument('--validation-ratio', type=float, default=.2,
                    help='Proporción de las muestras restantes para validación')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    if args.mode == '3d' and not args.synthetic_3d:
        print('3D real: se espera JSON de capturas 3D en formato {letra: [muestra,...]}, '
              'con muestra=[[ [x,y,z],... ], ...].')
    if args.synthetic_3d and args.mode != '3d':
        parser.error('--synthetic-3d solo se usa con --mode 3d')
    if not 0 <= args.test_ratio < .5:
        parser.error('test-ratio debe estar en [0, 0.5)')
    if not 0 < args.validation_ratio < .5:
        parser.error('validation-ratio debe estar en (0, 0.5)')
    if args.epochs < 1 or args.states < 2:
        parser.error('epochs>=1; states>=2')
    output = args.out or ROOT / 'models' / args.mode
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.data.read_text(encoding='utf-8'))
    if set(data) != set(LETTERS):
        raise ValueError(f'Deben existir exactamente las 27 letras. Faltan {set(LETTERS)-set(data)}')
    rng = random.Random(603504)
    train, validation, test, provenance = {}, {}, {}, {}
    durations = {}
    for character in LETTERS:
        raw = data[character]
        pairs = []
        for idx, sample in enumerate(raw):
            try:
                if args.mode == '2d':
                    obs = extract_2d(sample, sectors=args.sectors, step_units=10.)
                else:
                    strokes = map_2d_to_sphere(sample) if args.synthetic_3d else sample
                    obs = extract_3d(strokes, sectors=args.sectors, step_units=.018)
                if not obs:
                    continue
                pairs.append((idx, obs))
            except (ValueError, TypeError, IndexError) as err:
                print(f'ADVERTENCIA: descartada {character}[{idx}]: {err}')
        if len(pairs) < 5:
            raise ValueError(f'Faltan muestras válidas para {character} (solo {len(pairs)})')
        rng.shuffle(pairs)
        n_test = max(1, int(len(pairs) * args.test_ratio)) if args.test_ratio > 0 else 0
        test_pairs = pairs[:n_test]
        remaining = pairs[n_test:]
        n_validation = max(1, int(len(remaining) * args.validation_ratio))
        validation_pairs = remaining[:n_validation]
        train_pairs = remaining[n_validation:]
        if len(train_pairs) < 2:
            raise ValueError(f'Muestras insuficientes para entrenar {character}')
        test[character] = [seq for _, seq in test_pairs]
        validation[character] = [seq for _, seq in validation_pairs]
        train[character] = [seq for _, seq in train_pairs]
        provenance[character] = {
            'train_indices': [i for i, _ in train_pairs],
            'validation_indices': [i for i, _ in validation_pairs],
            'test_indices': [i for i, _ in test_pairs]}
    (output / 'split.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    models = {}
    print(f'ENTRENAMIENTO {args.mode.upper()} — '
      f'{sum(map(len, train.values()))} entrenamiento; '
      f'{sum(map(len, validation.values()))} validación; '
      f'{sum(map(len, test.values()))} test independientes')
    start = time.perf_counter()
    for i, character in enumerate(LETTERS):
        samples = train[character]
        model = left_right_model(args.states, args.sectors + 2, seed=603504+i,
                                 label=character)
        model.median_length = int(statistics.median(len(s) for s in samples))
        model.step_units = 10. if args.mode == '2d' else .018
        model.metadata = {'training_mode': args.mode,
                          'source': 'PROYECCION_2D_A_ESFERA_SINTETICA' if args.synthetic_3d else 'TRAZOS_CAPTURADOS',
                          'train_size': len(samples),
                            'validation_size': len(validation[character]),
                            'test_size': len(test[character]),
                          'epochs_requested': args.epochs,
                          'sectors': args.sectors, 'states': args.states,
                          'normalization': '2d: y invertida; 3d: lat/lon radianes',
                          'note': '3D sintético es una prueba de integración, no muestras 3D reales' if args.synthetic_3d else ''}
        history = baum_welch(model, samples, epochs=args.epochs)
        model.metadata['train_loglikelihood_history'] = history
        model.save(output / f'{character}.json')
        models[character] = model
        print(f'{i+1:02}/27 {character}  {len(samples)} ejemplos | {len(history)} épocas | '
              f'LL={history[-1]:.1f}', flush=True)
    correct, total = 0, 0
    confusions = []
    validation_rows = []

    for ch in LETTERS:
        for seq in validation[ch]:
            scores = {
                character: float(model.forward(seq)[2])
                for character, model in models.items()
            }

            validation_rows.append({
                'scores': scores,
                'label': ch
            })

    validation_path = output / 'validation_scores.json'
    validation_path.write_text(
        json.dumps(validation_rows, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    for ch in LETTERS:
        for seq in test[ch]:
            ranking = classify(models, seq)
            predicted = ranking[0][0]
            correct += (predicted == ch)
            total += 1
            if predicted != ch:
                confusions.append({'true': ch, 'predicted': predicted, 'probability': round(ranking[0][1], 4)})
    report = {'dataset': args.data.name, 'mode': args.mode,
              'synthetic_3d': args.synthetic_3d,
              'train_size': sum(map(len, train.values())),
              'test_size': total, 'correct': correct,
              'accuracy': (correct/total if total else None),
              'confusions_first_40': confusions[:40],
              'elapsed_seconds': round(time.perf_counter()-start,2),
              'model_notes': ('Evaluación sobre transformaciones sintéticas del mismo dataset 2D, no prueba de generalización 3D real.'
                              if args.synthetic_3d else 'Sin identidad de autores: separación por muestra, posible sesgo de autor.')}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Finalizado: exactitud holdout={correct}/{total} ({100*correct/total:.2f}%)' if total else 'Sin muestras de prueba')
    print(f'Modelos en {output}; evaluación {output/"evaluation.json"}')


if __name__ == '__main__':
    main()
