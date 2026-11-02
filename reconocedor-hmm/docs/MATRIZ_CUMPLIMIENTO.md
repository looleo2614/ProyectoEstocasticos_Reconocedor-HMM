# Matriz de cumplimiento — Especificación oficial

**Fuente vinculante:** `enunciado-proyecto-final-reconocedor-hmm.pdf`, 5 páginas. **Nunca** se incorporan exigencias de apuntes informales que contradigan el PDF.

| Apartado oficial | Requisito / prohibición | Estado y evidencia del prototipo |
|---|---|---|
| 1 | Aplicación de escritorio y GUI 2D/3D | **Implementado como prototipo**; `src/GUI.py` |
| 2.1 | Menú con modos 2D / 3D y navegación | **Implementado** |
| 2.2 | Lienzo 2D, ratón/táctil | **Ratón implementado**; dispositivo táctil pendiente de prueba |
| 2.2 | Proyección transparente/punteada que se actualiza | **Prototipo implementado**; estimación acotada y primer trazo solamente hasta `PEN_UP` |
| 2.2 | Limpiar lienzo y reiniciar captura | **Implementado** |
| 2.2 | Panel ordenado por P(c|O), porcentaje e icono/referencia | **Implementado**; glifo estático en cada fila |
| 2.3 | Esfera 3D, ángulos, trazo, zoom, rotación, reorientación | **Prototipo implementado**; geometría real, pero entrenamiento con datos 3D reales pendiente |
| 2.3 | Ranking de 27 probabilidades y proyección sobre esfera | **Prototipo implementado**, depende de 3D sintético inicial |
| 3.1 | Prohibido hmmlearn / sklearn / PyTorch / TensorFlow / scipy.stats para HMM | **Cumplido en código del prototipo**; no se usan |
| 3.1 | Forward-Backward/Viterbi/probabilidades/proyección nativos | **Implementados** en `src/HMMEngine.py`; pruebas unitarias |
| 3.2 | HMM independiente λ=(A,B,π) por letra | **Implementado**; 27 JSON 2D y 27 sintéticos 3D |
| 3.3 | 27 letras minúsculas y ñ | **Implementado** |
| 3.4 | Chain code K=8 o 16, desplazamientos | **Implementado**; K=8 usado en los modelos entrenados |
| 3.4 | (x,y,z) → (θ,φ), variaciones esféricas | **Implementado**; pendiente validación con capturas 3D reales |
| 3.5 | Forward + ranking + pronóstico de símbolos + coordenadas | **Prototipo implementado**, proyección multimovimiento entre trazos pendiente |
| 4.1 | No modo de entrenamiento/calibración en GUI | **Cumplido** |
| 4.1 | Entrenamiento previo en scripts independientes | **Cumplido**; `scripts/train.py` con Baum–Welch nativo |
| 4.1 | Exportar/importar parámetros estructurados | **Cumplido**; `models/{2d,3d}/*.json` |
| 4.2 | Arquitectura limpia, POO, matemática/geometría/GUI separadas | **Implementado**; `HMMEngine.py`, `FeatureExtractor.py`, `GUI.py` |
| 4.3 | Bibliotecas sugeridas: Tkinter para GUI | **Cumplido**; Tkinter + Canvas con geometría nativa, NumPy |
| 5.1 | GitHub documentado y envío ZIP al docente | **ZIP listo para descarga**; publicar en GitHub/enviar institucional es tarea pendiente del equipo |
| 5.2 | Demostración en vivo 2D/3D y defensa de ecuaciones | **Material preparado**, pero prueba presencial y validación final pendientes |
| Indicación posterior atribuida al profesor | Botón de ayuda con referencia de dibujo | **Implementado** |

## Pendientes prioritarios antes de llamarlo entrega final

1. Sustituir/reentrenar los HMM 3D con un conjunto **real** de trazos sobre esfera, no con transformaciones de 2D.
2. Aprender/reconstruir correctamente el posicionamiento de trazos discontinuos después de un `PEN_UP` para autocompletado de letras multitrazo.
3. Ensayar reconocimiento parcial de letras en vivo, comparar hiperparámetros, medir latencia y errores, incluyendo diferentes autores.
4. Validar GUI en Windows y dispositivos táctiles y verificar referencia de fuentes y convenciones de escritura.
5. Publicar repositorio propio y preparar defensa matemática presencial.


## Revisión cuantitativa adicional — 01/11/2026

- GUI: validada visualmente por el usuario en Windows 11 (v0.11); no extrapolar a todos los equipos.
- Probabilidades iniciales: uniformes 1/27 cuando no hay observaciones,
  implementado en `HMMEngine.classify` y verificado en prueba automática.
- Evaluación por prefijos: implementada **offline** en `scripts/evaluate_realtime.py`.
- Calibración: **diagnosticada, no corregida**. Confianza elevada no implica certeza.
- Reconocimiento 3D real: **pendiente**; evaluación disponible solo sintética.
- Predicción espacial después de PEN_UP: **pendiente**.
- Evaluación por autor: **no realizable con el JSON actual sin identidad de escritor**.
- Comparación sistemática de hiperparámetros y reentrenamiento: **pendiente**.
