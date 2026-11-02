# Reconocedor HMM 2D/3D — Proyecto de Procesos Estocásticos

**Universidad de los Llanos — Ingeniería de Sistemas**  
**Especificación vinculante:** `enunciado-proyecto-final-reconocedor-hmm.pdf`, profesor Yerson Porras.

Prototipo ejecutable: modelos HMM discretos programados desde cero, 27 letras minúsculas (incluye ñ), entrenamiento offline independiente, GUI de escritorio, captura 2D y trazado sobre superficie esférica 3D, clasificación en tiempo real y predicción espacial aproximada.

> **Estado:** primera implementación funcional y experimental, **no entrega final validada**. El clasificador 2D ya fue entrenado con el JSON del curso; los modelos 3D incluidos proceden de una proyección **sintética** del JSON 2D y NO sustituyen datos 3D reales.

## 1. Instalar y abrir en Windows

Requisitos: **Python 3.10 o superior** (Tkinter incluido en instalaciones habituales de Python para Windows).

Abre PowerShell en esta carpeta y ejecuta:

```powershell
py -m pip install -r requirements.txt
py main.py
```

El programa carga automáticamente `models/2d/*.json` y `models/3d/*.json`, ya incluidos en este paquete. **No hay entrenamiento dentro de la GUI.**

Opcionalmente, para usar entorno virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

En Linux puede ser necesario instalar `python3-tk` por separado para que Tkinter esté disponible.

## 2. Utilizar las ventanas

- **Inicio:** seleccionar reconocimiento 2D o 3D. La ayuda está en el menú y en las vistas.
- **2D:** botón izquierdo presionado para dibujar; al soltar se registra un levantamiento de lápiz. Se muestra el ranking de 27 probabilidades posteriores y una trayectoria proyectada punteada.
- **3D:** botón izquierdo sobre esfera para dibujar; botón derecho arrastrado para **rotar**; rueda para **zoom**; botón *Reorientar esfera* restaura la vista.
- **Limpiar lienzo:** elimina el dibujo actual sin alterar la cámara.
- **Reiniciar captura:** limpia dibujo, tiempos, inferencia y restablece la cámara esférica.
- **Ayuda:** muestra una referencia estática de las 27 letras y convenciones adicionales de `a`, `q`, `i`, `j` y `ñ`.

**Importante:** la línea punteada es una estimación a partir de emisiones del HMM y longitud media aprendida; no es el trazo futuro exacto.

## 3. Estructura de archivos

```text
reconocedor-hmm/
  main.py                       # Arranque GUI
  requirements.txt             # NumPy: única dependencia externa necesaria
  src/
    HMMEngine.py               # λ=(A,B,π), Forward-Backward, Viterbi,
                               # Baum-Welch, clasificación, proyección
    FeatureExtractor.py        # Características 2D/3D, levantamiento y puntos
    GUI.py                     # Vistas, eventos y esfera 3D interactiva
  scripts/
    audit_dataset.py           # Auditoría del JSON
    train.py                   # Entrenamiento separado de la interfaz
  data/raw/
    glifo-dataset-2026-11-02.json
  models/2d/                  # 27 modelos reales entrenados con trazos 2D
    a.json ... z.json
    split.json, evaluation.json
  models/3d/                  # 27 modelos sintéticos PROVISIONALES
    a.json ... z.json
    split.json, evaluation.json
  tests/test_math.py           # Pruebas de fórmulas y extractor
  docs/
    DEMOSTRACION_MATEMATICA.md
    MATRIZ_CUMPLIMIENTO.md
    gui_2d_prototipo.png
    gui_3d_prototipo.png
```

## 4. Ejecutar pruebas matemáticas y auditoría

```powershell
py -m unittest discover -s tests -v
py -m scripts.audit_dataset
```

Se probaron 9 casos automáticos: Forward contra enumeración directa, Backward, posterior de estados, Viterbi por fuerza bruta, actualización Baum–Welch, probabilidades posteriores y discretización 2D/3D. **No son todavía pruebas exhaustivas del comportamiento interactivo en diversos dispositivos.**

## 5. Reentrenamiento — solo por terminal, nunca desde la GUI

```powershell
# HMM 2D con todos los grupos, 27 letras, K=8, N=6, seis épocas
py -m scripts.train --mode 2d --epochs 6

# 3D sintético de DESARROLLO, no válido como prueba de datos reales 3D
py -m scripts.train --mode 3d --synthetic-3d --epochs 6
```

Si el curso entrega trazos 3D reales, usa un JSON agrupado por letra donde cada muestra sea una lista de trazos y cada trazo una lista de puntos `[x,y,z]` o `[x,y,z,t]`, sobre la superficie esférica, y entrena con:

```powershell
py -m scripts.train --mode 3d --data data/raw/dataset-3d-real.json --epochs 8
```

No debe ejecutarse el modo 3D real con datos que contengan exclusivamente `[x,y,t]`: el tercer campo sería interpretado erróneamente como la coordenada `z`.

Los datos de prueba se separan por muestra antes del entrenamiento y los índices se guardan en `models/<modo>/split.json`. El archivo original agrupado no incluye autor: **no se puede garantizar separación por escritor** con esta exportación.

### Medidas iniciales verificadas

- Dataset 2D: **1609 muestras**, 27 letras, mínimo 40 por letra.
- Entrenamiento 2D con **1291 muestras**, prueba con **318** y **265 aciertos (83,33 %)**.
- Transformación esférica sintética: 1291 entrenamiento, 318 prueba, 264 aciertos (83,02 %). **No es evaluación de reconocimiento 3D real**.
- HMM de 6 estados, sectores de 45° (K=8), más dos símbolos de eventos no direccionales.
- Esta precisión es en muestras completas, no necesariamente en prefijos parciales ni escritura nueva.

## 6. Diseño matemático

Las 27 letras tienen modelos independientes `λ_c = (A_c, B_c, π_c)`.

- **Estados ocultos:** `S1...S6`, fases progresivas de la letra, topología izquierda-a-derecha. `π=(1,0,...,0)` inicializa en el primer estado. Transiciones restringidas a permanecer o avanzar un estado; el último es absorbente.
- **Observaciones:** `0...7` son 8 orientaciones angulares; el símbolo **8=PEN_UP** representa la interrupción entre trazos y el **9=DOT** un punto aislado. Los dos eventos adicionales no son sectores angulares y están justificados para la `i`, `j` y `ñ`.
- **Geometría 2D:** `Δx=x(t+1)-x(t)`, `Δy=y(t+1)-y(t)`; como el lienzo Glifo crece hacia abajo, se invierte `Δy` antes de aplicar `atan2`. Se remuestrea por distancia de arco (paso de 10 unidades normalizadas, no por tiempo).
- **Geometría 3D:** normalizar `(x,y,z)` al radio 1, transformar a longitud `θ=atan2(x,z)` y latitud `φ=asin(y)` y discretizar desplazamientos tangenciales `(cos(φ)Δθ, Δφ)`; se corrige discontinuidad en `±π`.
- **Forward-Backward:** recurrencias nativas con escalamiento para evitar underflow numérico.
- **Viterbi:** decodificación dinámica en log-probabilidades con retrotrazado de estados.
- **Baum-Welch:** EM con `γ` y `ξ` propios, pseudocuentas y topología fija.
- **Clasificación:** calcula logverosimilitud de cada letra y aplica Bayes con previas equiprobables. Las probabilidades posteriores suman 1 (aparte de error de redondeo).
- **Proyección:** pronostica el símbolo más probable iterativamente a partir de la posterior del estado, reconstruye segmentos con longitud de paso fijada por la discretización y limita su horizonte por la longitud mediana de las observaciones de entrenamiento. Detiene la visualización al encontrar `PEN_UP` porque el siguiente punto de contacto no está determinado geométricamente por un HMM de direcciones.

Véase `docs/DEMOSTRACION_MATEMATICA.md` para fórmulas y demostración de tablero.

## 7. Límites aún pendientes de resolver

1. **Datos 3D reales:** faltan muestras de escritura manual sobre la esfera y su evaluación independiente. Los modelos precargados 3D son exclusivamente sintéticos.
2. **Predicción completa multitrazo:** una emisión PEN_UP no determina por sí sola la posición donde comenzará el siguiente trazo. Para autocompletar también esa parte hace falta aprender un modelo adicional de desplazamiento entre trazos y validarlo, en lugar de dibujar un salto ficticio.
3. **Calidad del reconocedor:** 83,33 % en clasificación completa del holdout inicial. No hay medición aún de exactitud en prefijos, latencia interactiva real en Windows ni generalización por escritor.
4. **Navegación 3D:** el sistema renderiza una malla ortográfica de esfera y dibuja sobre ella, sin aceleración OpenGL. Se debe verificar rendimiento y compatibilidad táctil en equipos físicos.
5. **Parámetros de modelado:** número de estados, longitud de observación, 8 vs. 16 sectores y configuración de entrenamiento requieren validación y comparación experimental.
6. **Proyección futura exacta:** se ha implementado una aproximación probabilística razonable pero no necesariamente la reconstrucción completa de todos los segmentos restantes exigida en una interpretación fuerte del enunciado.

Estos límites están identificados explícitamente: el prototipo no se presenta como entrega completa ni como evidencia de que el modo 3D ya está validado.

## 8. Dependencias expresamente permitidas / prohibidas

**Usadas:** NumPy exclusivamente para álgebra de arreglos, Tkinter/ttk para GUI (está en la biblioteca estándar), JSON/math/pathlib y otros módulos estándar. La esfera se renderiza con código geométrico propio sobre Canvas; no requiere biblioteca estocástica externa.

**Prohibidas para la formulación/cálculo/resolución HMM:** `hmmlearn`, `scikit-learn`, `PyTorch`, `TensorFlow`, `scipy.stats`, o cualquier biblioteca equivalente que sustituya las implementaciones nativas.

No se debe añadir a la GUI un modo ni botón de entrenamiento/calibración. Revisa `docs/MATRIZ_CUMPLIMIENTO.md` para distinguir funcionalidad implementada, pruebas y asuntos pendientes.


## Auditoría adicional (01/11/2026)

Se añadió `scripts/evaluate_realtime.py`, que calcula exactitud por prefijos
(25 %, 50 %, 75 %, 100 % de los puntos), confusiones y grupos de calibración
utilizando exclusivamente los índices reservados en `models/<modo>/split.json`.

Ejecutar: `py -m scripts.evaluate_realtime --mode 2d` o `--mode 3d`.
No reentrena modelos ni corrige porcentajes artificialmente.

Resultados de la evaluación reservada (318 muestras):
- 2D: 110/318, 129/318, 200/318, 265/318 respectivamente.
- 3D **sintético**: 109/318, 132/318, 200/318, 264/318.
- Confusión más frecuente: `a` → `q` en 12 muestras de cada modalidad.
- Grupo de confianza [80 %,100 %] en 2D: 290 predicciones, confianza
  media 99,319 %, exactitud real 84,483 %. No está calibrado.

La GUI mantiene la v0.11 aprobada por el usuario. La auditoría NO soluciona
la generalización esférica real ni la proyección entre trazos. La evaluación
por autor requiere identificadores de escritor que el JSON no proporciona.


## v0.15 — Dataset actualizado (2026-11-02)

El dataset predeterminado es `data/raw/glifo-dataset-2026-11-02.json`. Los modelos incluidos son los originales de v0.14 y **no se reentrenaron** al empaquetar este ZIP. Para actualizar los modelos, desde la raíz del proyecto ejecute:

```cmd
py -m scripts.train --mode 2d --epochs 8
py -m scripts.train --mode 3d --synthetic-3d --epochs 8
```

El segundo comando entrena un modelo 3D **sintético**, no basado en capturas 3D reales. Ambos comandos sobrescriben los modelos de sus respectivas carpetas; guarde una copia antes si desea conservarlos.
