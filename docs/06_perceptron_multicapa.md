# 06 --- Perceptron Multicapa y Retropropagacion (Backpropagation)

> Modulo: [`ce_rna/mlp.py`](../ce_rna/mlp.py)  
> Experimento: [08 (XOR y aproximacion de funciones)](../experimentos/exp08_mlp_xor_aproximacion.py)  
> Informe: [08](../resultados/reportes/08_mlp_xor_aproximacion.md)  
> Fuentes: `clases/claseRN01.md`, `clases/ICE-claseRN03.md` y Rumelhart, Hinton & Williams (1986).

---

## 1. El problema que resuelve

Los modelos de una sola capa (McCulloch-Pitts, Perceptron Simple y ADALINE) comparten una limitacion infranqueable:
solo pueden construir **hiperplanos lineales** de decision. En el caso del OR-Exclusivo (XOR), tal como se demostro
exhaustivamente en el Experimento 03, ninguna recta en el plano puede separar simultaneamente los cuatro patrones.

Tal como formula `ICE-claseRN03.md` (secciones *Red Multicapa* y *Convexidad*):

> *"El problema de separabilidad lineal o representatividad grafica para problemas mas complejos puede ser superado
> sumando mas capas a la red... Las capas ocultas siempre son no lineales. Se puede demostrar que si se construye
> una red multicapa con capas ocultas lineales, esta es equivalente a una red unicapa."*

Ademas, segun el **Teorema de Aproximacion Universal** (Cybenko 1989, Hornik 1991), una red feedforward con al menos
una capa oculta no lineal y salida lineal puede aproximar cualquier funcion continua definida sobre un compacto con
arbitraria precision.

---

## 2. Arquitectura de la red

Una red de $L$ capas neuronales ($l = 1, \dots, L$) con $m_0$ entradas sensoriales:

```
Capa 0 (Entrada)    Capa 1 (Oculta)         Capa L (Salida)
    [x1] -----------> (h1) -----------------> (y1)
    [x2] -----------> (h2)
     :                  :                       :
    [xm0] ----------> (hm1) ----------------> (ymL)
    [x0=1]            [h0=1] (sesgos)
```

Para cada capa $l$:
* Matriz de pesos sinapticos $W^{(l)}$ de dimension $m_l \times m_{l-1}$.
* Vector de sesgos $b^{(l)}$ de dimension $m_l$.
* Potencial post-sinaptico: $v^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$.
* Activacion no lineal: $a^{(l)} = \phi^{(l)}(v^{(l)})$, donde $a^{(0)} = x$.

---

## 3. Derivacion matematica de Backpropagation

El error cuadratico sobre un patron con salida real $a^{(L)}$ y objetivo deseado $d$ es:

$$E = \frac{1}{2} \sum_{k=1}^{m_L} (d_k - a_k^{(L)})^2$$

Buscamos actualizar los pesos en la direccion opuesta al gradiente:

$$\Delta W_{ji}^{(l)} = -\eta \frac{\partial E}{\partial W_{ji}^{(l)}}$$

Por regla de la cadena:

$$\frac{\partial E}{\partial W_{ji}^{(l)}} = \frac{\partial E}{\partial v_j^{(l)}} \frac{\partial v_j^{(l)}}{\partial W_{ji}^{(l)}} = -\delta_j^{(l)} a_i^{(l-1)}$$

donde se define el **gradiente local** como $\delta_j^{(l)} \equiv -\frac{\partial E}{\partial v_j^{(l)}}$.

### A. Capa de salida ($l = L$)

$$\delta_k^{(L)} = -\frac{\partial E}{\partial a_k^{(L)}} \frac{\partial a_k^{(L)}}{\partial v_k^{(L)}} = (d_k - a_k^{(L)}) \cdot {\phi^{(L)}}'(v_k^{(L)})$$

### B. Capas ocultas ($l = L-1, \dots, 1$)

Aplicando la regla de la cadena a traves de todas las neuronas de la capa siguiente $l+1$:

$$\delta_j^{(l)} = -\sum_k \frac{\partial E}{\partial v_k^{(l+1)}} \frac{\partial v_k^{(l+1)}}{\partial a_j^{(l)}} \frac{\partial a_j^{(l)}}{\partial v_j^{(l)}} = \left( \sum_k \delta_k^{(l+1)} W_{kj}^{(l+1)} \right) \cdot {\phi^{(l)}}'(v_j^{(l)})$$

La senal de error $\delta$ se **retropropaga** desde la salida hacia las capas iniciales multiplicada por la transpuesta
de la matriz de pesos sinapticos.

### C. Termino de Momento ($\alpha$)

Para evitar oscilaciones en valles estrechos y acelerar el paso a traves de mesetas (zonas donde $\phi' \approx 0$),
Rumelhart et al. (1986) introdujeron el termino de momento inercial $\alpha \in [0, 1)$:

$$\Delta W_{ji}^{(l)}(t) = \eta \cdot \delta_j^{(l)} a_i^{(l-1)} + \alpha \Delta W_{ji}^{(l)}(t-1)$$

---

## 4. Funciones de activacion y sus derivadas

| Problema | Capa oculta | Capa de salida | Justificacion |
|---|---|---|---|
| **XOR / Clasificacion** | Sigmoide $\frac{1}{1+e^{-v}}$ | Sigmoide $\frac{1}{1+e^{-v}}$ | Salidas discretas $\{0, 1\}$, derivada rapida $\phi(1-\phi)$, acotada. |
| **Aproximacion continua** | Sigmoide o Sigmoide Bipolar | Identidad $\phi(v)=v$ | Salida sin saturacion; permite alcanzar cualquier valor en $\mathbb{R}$. |

---

## 5. Uso como libreria

```python
import numpy as np
from ce_rna.mlp import PerceptronMulticapa

# 1. Resolucion del XOR con arquitectura 2-2-1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 1, 1, 0])

red = PerceptronMulticapa(
    arquitectura=(2, 2, 1),
    activacion_oculta="sigmoide",
    activacion_salida="sigmoide",
    razon_aprendizaje=1.0,
    momento=0.5,
    max_epocas=2500,
    tolerancia=1e-4,
    semilla=42,
)
red.entrenar(X, d)

print(red.predecir(X))   # ~ [0.017, 0.985, 0.985, 0.015]
print(red.clasificar(X)) # [0. 1. 1. 0.]

# 2. Aproximacion de f(x) = sin(x) con arquitectura 1-8-1 y salida lineal
x_datos = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y_datos = np.sin(x_datos).ravel()

aprox = PerceptronMulticapa(
    arquitectura=(1, 8, 1),
    activacion_oculta="sigmoide",
    activacion_salida="identidad",
    razon_aprendizaje=0.08,
    momento=0.7,
    modo="lote",
    max_epocas=3000,
    semilla=42,
)
aprox.entrenar(x_datos, y_datos)
print("ECM final:", aprox.historia_error[-1])
```

---

## 6. Verificacion numerica del gradiente

En [`pruebas/pruebas.py`](../pruebas/pruebas.py), la prueba `probar_mlp_gradiente_coincide_con_derivada_numerica`
compara el gradiente analitico $\partial E / \partial W$ obtenido por Backpropagation con la derivada numerica:

$$\frac{\partial E}{\partial W_{ji}} \approx \frac{E(W_{ji} + \epsilon) - E(W_{ji} - \epsilon)}{2\epsilon}$$

con $\epsilon = 10^{-6}$. La diferencia observada es menor a $2 \times 10^{-12}$, confirmando la exactitud analitica
del algoritmo implementado.
