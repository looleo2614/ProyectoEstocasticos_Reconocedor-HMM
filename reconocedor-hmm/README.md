# Reconocedor HMM 2D/3D — Proyecto de Procesos Estocásticos

**Universidad de los Llanos — Ingeniería de Sistemas**  
**Especificación vinculante:** `enunciado-proyecto-final-reconocedor-hmm.pdf`, profesor Yerson Porras.

Prototipo ejecutable: modelos HMM discretos programados desde cero, 27 letras minúsculas (incluye `ñ`), entrenamiento offline independiente, GUI de escritorio, captura 2D y trazado sobre superficie esférica 3D, clasificación en tiempo real y predicción espacial aproximada.

> **Estado:** primera implementación funcional y experimental, **no entrega final validada**. El clasificador 2D ya fue entrenado con el JSON del curso; los modelos 3D incluidos proceden de una proyección **sintética** del JSON 2D y NO sustituyen datos 3D reales.

---

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

---

## 2. Utilizar las ventanas

- **Inicio:** seleccionar reconocimiento 2D o 3D. La ayuda está en el menú y en las vistas.
- **2D:** botón izquierdo presionado para dibujar; al soltar se registra un levantamiento de lápiz. Se muestra el ranking de 27 probabilidades posteriores y una trayectoria proyectada punteada.
- **3D:** botón izquierdo sobre esfera para dibujar; botón derecho arrastrado para **rotar**; rueda para **zoom**; botón *Reorientar esfera* restaura la vista.
- **Limpiar lienzo:** elimina el dibujo actual sin alterar la cámara.
- **Reiniciar captura:** limpia dibujo, tiempos, inferencia y restablece la cámara esférica.
- **Guía de Trazado de Caracteres (barra inferior y modal):** muestra las 27 letras minúsculas (`a–z`, `ñ`) con resaltado interactivo al pasar el cursor. Al hacer clic en una letra se abre una ventana modal con cuadrícula de referencia que desglosa cada trazo mediante un **código de color único** (círculo numerado, punto de inicio y flecha punteada direccional) y documenta las convenciones adicionales de `a`, `q`, `i`, `j` y `ñ`.

**Importante:** la línea punteada es una estimación a partir de emisiones del HMM y longitud media aprendida; no es el trazo futuro exacto.

---

## 3. Estructura de archivos

```text
reconocedor-hmm/
  main.py                       # Arranque GUI
  requirements.txt             # NumPy: única dependencia externa necesaria
  src/
    HMMEngine.py               # λ=(A,B,π), Forward-Backward, Viterbi,
                               # Baum-Welch, clasificación, proyección
    FeatureExtractor.py        # Características 2D/3D, levantamiento y puntos
    GUI.py                     # Vistas, eventos, esfera 3D y Guía de Trazado
  scripts/
    audit_dataset.py           # Auditoría del JSON
    train.py                   # Entrenamiento separado de la interfaz
    evaluate_realtime.py       # Evaluación por prefijos (25%, 50%, 75%, 100%)
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

---

## 4. Ejecutar pruebas matemáticas y auditoría

```powershell
py -m unittest discover -s tests -v
py -m scripts.audit_dataset
```

Se probaron **9 casos automáticos**: Forward contra enumeración directa, Backward, posterior de estados, Viterbi por fuerza bruta, actualización Baum–Welch, probabilidades posteriores y discretización 2D/3D. **No son todavía pruebas exhaustivas del comportamiento interactivo en diversos dispositivos.**

### Auditoría de exactitud por prefijos y calibración

El script `scripts/evaluate_realtime.py` calcula exactitud por prefijos (`25 %`, `50 %`, `75 %`, `100 %` de los puntos), matriz de confusiones y grupos de calibración utilizando exclusivamente los índices reservados en `models/<modo>/split.json` (no reentrena modelos ni corrige porcentajes artificialmente):

```powershell
py -m scripts.evaluate_realtime --mode 2d
py -m scripts.evaluate_realtime --mode 3d
```

---

## 5. Reentrenamiento — solo por terminal, nunca desde la GUI

> **Nota sobre v0.15 (`glifo-dataset-2026-11-02.json`):** El dataset predeterminado es `data/raw/glifo-dataset-2026-11-02.json`. Los modelos incluidos actualmente en `models/2d/` y `models/3d/` son los originales de **v0.14** y **no se reentrenaron** al empaquetar este ZIP. Ambos comandos sobrescriben los modelos de sus respectivas carpetas; guarde una copia antes si desea conservarlos.

```powershell
# HMM 2D con todos los grupos, 27 letras, K=8, N=6 (8 épocas sobre dataset v0.15)
py -m scripts.train --mode 2d --epochs 8

# 3D sintético de DESARROLLO, no válido como prueba de datos reales 3D
py -m scripts.train --mode 3d --synthetic-3d --epochs 8
```

Si el curso entrega trazos 3D reales, usa un JSON agrupado por letra donde cada muestra sea una lista de trazos y cada trazo una lista de puntos `[x,y,z]` o `[x,y,z,t]`, sobre la superficie esférica, y entrena con:

```powershell
py -m scripts.train --mode 3d --data data/raw/dataset-3d-real.json --epochs 8
```

**Advertencia:** No debe ejecutarse el modo 3D real con datos que contengan exclusivamente `[x,y,t]`: el tercer campo sería interpretado erróneamente como la coordenada `z`.

Los datos de prueba se separan por muestra antes del entrenamiento y los índices se guardan en `models/<modo>/split.json`. El archivo original agrupado no incluye autor: **no se puede garantizar separación por escritor** con esta exportación.

### Medidas iniciales verificadas (318 muestras de prueba)

- **Dataset 2D:** **1609 muestras**, 27 letras, mínimo 40 muestras por letra (`1291` entrenamiento / `318` prueba).
- **Configuración del modelo:** HMM de `N=6` estados, sectores de `45°` (`K=8`), más dos símbolos de eventos no direccionales.

| Modalidad | 25 % prefijo | 50 % prefijo | 75 % prefijo | 100 % (Muestra completa) |
| :--- | :---: | :---: | :---: | :---: |
| **2D (Datos reales)** | 110/318 (34,59 %) | 129/318 (40,57 %) | 200/318 (62,89 %) | **265/318 (83,33 %)** |
| **3D (Sintético — no real)** | 109/318 (34,28 %) | 132/318 (41,51 %) | 200/318 (62,89 %) | **264/318 (83,02 %)** |

- **Confusión más frecuente:** `a` $\rightarrow$ `q` en 12 muestras de cada modalidad.
- **Calibración en 2D:** Grupo de confianza `[80 %, 100 %]` con **290 predicciones**, confianza media **99,319 %** y exactitud real **84,483 %**. **No está calibrado.**
- Esta precisión en muestras completas no garantiza el mismo desempeño en prefijos parciales ni escritura nueva, ni soluciona la generalización esférica real o la proyección entre trazos.

---

## 6. Diseño matemático

Las 27 letras tienen modelos independientes $\lambda_c = (A_c, B_c, \pi_c)$.

- **Estados ocultos:** $S_1 \dots S_6$, fases progresivas de la letra, topología izquierda-a-derecha. $\pi=(1,0,\dots,0)$ inicializa en el primer estado. Transiciones restringidas a permanecer o avanzar un estado; el último es absorbente.
- **Observaciones:** `0...7` son 8 orientaciones angulares; el símbolo **`8=PEN_UP`** representa la interrupción entre trazos y el **`9=DOT`** un punto aislado. Los dos eventos adicionales no son sectores angulares y están justificados para la `i`, `j` y `ñ`.
- **Geometría 2D:** $\Delta x = x(t+1)-x(t)$, $\Delta y = y(t+1)-y(t)$; como el lienzo Glifo crece hacia abajo, se invierte $\Delta y$ antes de aplicar `atan2`. Se remuestrea por distancia de arco (paso de 10 unidades normalizadas, no por tiempo).
- **Geometría 3D:** normalizar $(x,y,z)$ al radio 1, transformar a longitud $\theta = \text{atan2}(x,z)$ y latitud $\phi = \arcsin(y)$ y discretizar desplazamientos tangenciales $(\cos(\phi)\Delta\theta, \Delta\phi)$; se corrige discontinuidad en $\pm\pi$.
- **Forward-Backward:** recurrencias nativas con escalamiento para evitar underflow numérico.
- **Viterbi:** decodificación dinámica en log-probabilidades con retrotrazado de estados.
- **Baum-Welch:** EM con $\gamma$ y $\xi$ propios, pseudocuentas y topología fija.
- **Clasificación:** calcula logverosimilitud de cada letra y aplica Bayes con previas equiprobables. Las probabilidades posteriores suman 1 (aparte de error de redondeo).
- **Proyección:** pronostica el símbolo más probable iterativamente a partir de la posterior del estado, reconstruye segmentos con longitud de paso fijada por la discretización y limita su horizonte por la longitud media de las observaciones de entrenamiento. Detiene la visualización al encontrar `PEN_UP` porque el siguiente punto de contacto no está determinado geométricamente por un HMM de direcciones.

Véase `docs/DEMOSTRACION_MATEMATICA.md` para fórmulas y demostración de tablero.

---

## 7. Dependencias expresamente permitidas / prohibidas

- **Usadas:** `NumPy` exclusivamente para álgebra de arreglos, `Tkinter`/`ttk` para GUI (está en la biblioteca estándar), `json`/`math`/`pathlib` y otros módulos estándar. La esfera se renderiza con código geométrico propio sobre Canvas; no requiere biblioteca estocástica externa.
- **Prohibidas para la formulación/cálculo/resolución HMM:** `hmmlearn`, `scikit-learn`, `PyTorch`, `TensorFlow`, `scipy.stats`, o cualquier biblioteca equivalente que sustituya las implementaciones nativas.

No se debe añadir a la GUI un modo ni botón de entrenamiento/calibración. Revisa `docs/MATRIZ_CUMPLIMIENTO.md` para distinguir funcionalidad implementada, pruebas y asuntos pendientes.