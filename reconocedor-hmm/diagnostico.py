import json

with open('models/2d/evaluation.json', encoding='utf-8') as f:
    report = json.load(f)

print('Exactitud:', report['accuracy'])
print('Tamaño de prueba:', report['test_size'])

with open('models/2d/split.json', encoding='utf-8') as f:
    split = json.load(f)

for char in ('m', 'n', 'h'):
    print(
        char,
        'entrenamiento:', len(split[char]['train_indices']),
        'prueba:', len(split[char]['test_indices'])
    )