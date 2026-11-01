# Demostración matemática de los HMM nativos

**Especificación fuente:** `enunciado-proyecto-final-reconocedor-hmm.pdf`, secciones 3.1–3.5 y 5.2.

## 1. Modelo por carácter

Para cada carácter `c` se define un HMM discreto independiente
`λ_c=(A_c,B_c,π_c)`, donde:

- `A[i,j] = P(S_(t+1)=j | S_t=i)`.
- `B[j,k] = P(O_t=k | S_t=j)`.
- `π[i] = P(S_1=i)`.

Se exige `∑j A[i,j] = 1`, `∑k B[j,k] = 1` y `∑i π[i] = 1`.

Los estados ocultos representan fases del trazo. `K=8` sectores angulares cuantizados y 2 eventos adicionales de dibujo no angular (`PEN_UP`, `DOT`) representan las observaciones. Un ejemplo de izquierda a derecha tiene `A[i,i]>0`, `A[i,i+1]>0` y otras transiciones 0.

## 2. Forward: probabilidad total de una secuencia

`α_1(j)=π_j B_j(O_1)`

`α_t(j) = B_j(O_t) ∑i α_(t-1)(i) A_ij`

`P(O_1:T | λ) = ∑j α_T(j)`.

Para evitar subdesbordamiento numérico, cada vector `α_t` se divide por la suma de sus componentes `s_t`. Entonces:

`log P(O_1:T | λ) = ∑t log s_t`.

### Ejemplo a mano

Supón:

`π=[0.8,0.2]`, `A=[[0.7,0.3],[0.4,0.6]]`, `B=[[0.6,0.4],[0.3,0.7]]`.

Para observaciones `[0,1]`:

`α_1=(0.8*0.6, 0.2*0.3)=(0.48,0.06)`.

`α_2(1)=0.4*(0.48*0.7+0.06*0.4)=0.144`.

`α_2(2)=0.7*(0.48*0.3+0.06*0.6)=0.126`.

`P([0,1]|λ)=0.144+0.126=0.27`.

La implementación `HMM.forward()` devuelve la verosimilitud en logaritmos junto con `α` escaladas.

## 3. Backward

`β_T(i)=1`.

`β_t(i)=∑j A_ij B_j(O_(t+1)) β_(t+1)(j)`.

Si Forward está escalado, la implementación divide cada paso por `s_(t+1)`.

La probabilidad posterior de un estado es:

`γ_t(i)=α_t(i)β_t(i) / ∑j α_t(j)β_t(j)`.

## 4. Viterbi

En dominio logarítmico:

`δ_1(j)=log(π_j)+log(B_j(O_1))`.

`δ_t(j)=log(B_j(O_t))+max_i[δ_(t-1)(i)+log(A_ij)]`.

`ψ_t(j)=argmax_i[δ_(t-1)(i)+log(A_ij)]`.

Retrocediendo desde `argmax_j δ_T(j)` a través de `ψ` se obtiene la sucesión de **estados ocultos más probable**. Viterbi encuentra un camino, no suma todos los caminos como Forward.

## 5. Baum–Welch (EM), estimación de parámetros

En la etapa E:

`γ_t(i)=P(S_t=i|O,λ)`.

`ξ_t(i,j)=P(S_t=i,S_(t+1)=j|O,λ)`

`= α_t(i) A_ij B_j(O_(t+1)) β_(t+1)(j) / normalizador`.

Para entrenamiento multisecuencia, la etapa M combina las frecuencias esperadas de todas las muestras:

`A_ij' = Σ_m Σ_(t<T_m) ξ_t^m(i,j) / Σ_m Σ_(t<T_m) γ_t^m(i)`.

`B_j(k)' = Σ_m Σ_(t:O_t^m=k) γ_t^m(j) / Σ_m Σ_t γ_t^m(j)`.

`π_i' = (1/M) Σ_m γ_1^m(i)`.

El código aplica pseudocuentas pequeñas y mantiene la topología prohibiendo transiciones fuera de las fases secuenciales. Con inicio forzado en el estado 1, `π` queda `[1,0,...]` después de reestimarse.

## 6. Probabilidad posterior de cada letra

Con las 27 letras y previas uniformes:

`P(c|O) = P(O|λ_c) / Σ_(d∈C) P(O|λ_d)`.

En los cálculos se usa `log-sum-exp` implementado con NumPy y `math` para evitar subdesbordamiento, NO `scipy.stats`.

La clasificación con verosimilitud de la sección 3.5 coincide con el máximo posterior si las previas de los 27 caracteres son iguales.

## 7. Proyección futura

Se obtiene una distribución posterior sobre los estados `γ_T`. El paso siguiente predice:

`p(S_(T+1)) = γ_T A`.

`p(O_(T+1)=k|O_1:T) = Σ_j p(S_(T+1)=j) B_j(k)`.

Se toma el símbolo más probable y se actualiza la distribución de estados con la nueva emisión estimada. Para cada dirección angular elegida `k`, `θ_k=2πk/K`:

`x_(t+1)=x_t + d_c cos(θ_k)`

`y_(t+1)=y_t - d_c sin(θ_k)`  (eje vertical de pantalla hacia abajo).

`d_c` es la magnitud del paso fijado por la estrategia de discretización (10 unidades normalizadas en 2D; 0,018 radianes en 3D). Esta magnitud **no la estima el HMM**: es una hipótesis de reconstrucción que debe validarse. En 3D se integra en coordenadas esféricas:

`Δφ=d_c sin(θ_k)`

`Δθ=d_c cos(θ_k)/max(|cos φ|,ε)`

Se limita el número de pasos por la mediana de longitud de secuencia que se aprendió del carácter. Un símbolo `PEN_UP` interrumpe la proyección visible hasta que exista una nueva posición de contacto. Esto **no** resuelve por sí mismo el problema de localizar el comienzo del siguiente trazo.

## 8. Separación entrenamiento y evaluación

El script divide por letra las muestras en un conjunto de entrenamiento (80 %) y otro de prueba (20 %). Los parámetros se aprenden sin usar secuencias de prueba. Se guardan índices en `split.json` para reproducibilidad. El JSON entregado no incluye identificadores de autor, así que no puede probarse generalización por escritor independiente.
