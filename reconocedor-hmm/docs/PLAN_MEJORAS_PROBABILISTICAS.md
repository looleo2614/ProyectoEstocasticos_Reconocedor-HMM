# Mejoras probabilísticas verificables — versión experimental

El PDF oficial tiene prioridad. No se incorporan librerías de HMM externas. No se modifica la GUI ni se entrena en ella.

## Implementado en esta versión

- `classify(..., temperature=1.0)` admite una temperatura **opcional**, preservando exactamente el cálculo anterior por defecto. No se aplica una temperatura inventada en la GUI.
- `scripts/calibrate.py` permite estimar temperatura exclusivamente desde **datos de validación** en formato de log-verosimilitudes por letra. La muestra de prueba final no puede usarse para ajustar temperatura.
- Forward detecta secuencias de verosimilitud exactamente nula, en lugar de fingir una verosimilitud minúscula. Las emisiones de modelos entrenados con suavizado deben ser positivas.
- Pruebas unitarias adicionales para los puntos anteriores.

## No solucionado todavía

- No se ha reentrenado ni mejorado la exactitud de clasificación `a/q`, `n/o` o `i`.
- No se ha entrenado con capturas esféricas reales.
- La proyección espacial sigue limitada a un trazo; no reconstruye saltos entre trazos.
- No hay calibración aplicada por defecto: primero se requiere un conjunto de validación separado.

## Próximos experimentos obligatorios antes de afirmar mejora

1. Partir de un mismo split fijo por autor, si el JSON conserva `autor`; si no, declarar separación por muestra.
2. Separar entrenamiento, validación y prueba. No elegir hiperparámetros usando el conjunto de prueba.
3. Comparar 8 y 16 sectores, 4/6/8 estados, inicializaciones múltiples y suavizado.
4. Evaluar exactitud completa, exactitud por prefijo, NLL, Brier y confusiones por letra.
5. Para 3D, recopilar datos con coordenadas espaciales reales y evaluar aparte de las conversiones sintéticas.
6. Para proyección multitrazo, conservar eventos de levantamiento y aprender distribuciones geométricas de reinicio con datos de entrenamiento.

**Precaución:** una distribución menos concentrada no equivale a una clasificación más correcta.
