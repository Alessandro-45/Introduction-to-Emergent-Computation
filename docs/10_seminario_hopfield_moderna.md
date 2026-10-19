# 10 --- Seminario de investigacion: *Hopfield Networks is All You Need*

> Actividad 6 de la guia de evaluacion: resumen escrito, implementacion propia del modelo
> y guion de la presentacion oral. Implementacion: [`ce_rna/hopfield_moderna.py`](../ce_rna/hopfield_moderna.py).
> Resultados: [experimento 13](../resultados/reportes/13_hopfield_moderna.md).

---

## 1. Referencia completa

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Adler, T., Gruber, L.,
Holzleitner, M., Pavlović, M., Sandve, G. K., Greiff, V., Kreil, D., Kopp, M.,
Klambauer, G., Brandstetter, J. y Hochreiter, S. (2021). *Hopfield Networks is All You
Need*. **International Conference on Learning Representations (ICLR 2021)**.
Preimpresion: arXiv:2008.02217. Codigo: https://github.com/ml-jku/hopfield-layers.

**Pertinencia de la fuente**: congreso de primer nivel en aprendizaje automatico, con
revision por pares abierta. El grupo es el Institute for Machine Learning de la
Universidad Johannes Kepler de Linz, dirigido por Sepp Hochreiter (coautor de las redes
LSTM), con colaboradores de la Universidad de Oslo y del IARAI. El articulo es la
referencia principal de las llamadas "redes de Hopfield modernas" y conecta el modelo de
memoria asociativa estudiado en la Actividad 5 con la arquitectura Transformer, que es la
base de los modelos de lenguaje actuales.

---

## 2. Resumen

La red de Hopfield de 1982 es una memoria asociativa: guarda patrones y los recupera a
partir de versiones incompletas o ruidosas. Sus limites son conocidos: estados binarios,
capacidad de solo ~0.14·d patrones para d neuronas y estados espurios. Los autores
proponen una red de Hopfield con **estados continuos** y una nueva funcion de energia
basada en el *log-sum-exp*. Demuestran que su regla de actualizacion converge, que puede
guardar un numero de patrones **exponencial** en la dimension, y que recupera un patron
con **una sola actualizacion** y un error exponencialmente pequeno. El resultado central
es que esa regla de actualizacion es **identica al mecanismo de atencion de los
Transformers**. Con esa equivalencia reinterpretan la atencion como una memoria
asociativa, analizan las cabezas de atencion de BERT y proponen capas de Hopfield
reutilizables en aprendizaje profundo, que prueban en clasificacion con instancias
multiples (incluida la clasificacion de repertorios inmunes), en pequenos conjuntos de
datos tabulares y en diseno de farmacos.

---

## 3. El problema

1. **Capacidad**: con la regla de Hebb, la red clasica de d neuronas solo recupera de
   forma fiable unos 0.138·d patrones aleatorios. Para las letras de 7x6 de la
   Actividad 5 son unos 5,8 patrones, y el experimento 11 muestra que incluso cuatro
   letras parecidas fallan.
2. **Estados binarios**: la red clasica trabaja con vectores ±1 y una dinamica de
   neurona a neurona. Eso no encaja con el aprendizaje profundo, que necesita
   operaciones continuas y derivables para entrenar con retropropagacion.
3. **Entender la atencion**: los Transformers usan softmax(QKᵀ/√d_k)V en todas sus
   capas sin una teoria clara de *que* hace esa operacion. El articulo da una respuesta:
   es un paso de recuperacion de una memoria asociativa.

---

## 4. El modelo

### Energia

Con X = (x₁, …, x_P) los patrones almacenados (columnas de una matriz d x P), ξ el estado
y M la mayor norma de un patron:

```
E(ξ) = −lse(β, Xᵀξ) + ½ ξᵀξ + β⁻¹ log P + ½ M²,      lse(β, z) = β⁻¹ log Σ_μ exp(β z_μ)
```

- −lse es un **maximo suave** de los solapamientos: la energia baja cuando el estado se
  parece mucho a *algun* patron.
- ½‖ξ‖² impide que el estado crezca sin limite.
- Los dos ultimos terminos son constantes y solo acotan la energia: 0 ≤ E ≤ 2M².

Linea de modelos que llevan hasta aqui: Hopfield (1982) usa una energia cuadratica en los
solapamientos; Krotov y Hopfield (2016), una potencia F(z) = zⁿ; Demircigil et al. (2017),
la exponencial con estados binarios. Ramsauer et al. (2021) la extienden a estados
continuos.

### Regla de actualizacion

Minimizando E con el procedimiento concavo-convexo (CCCP):

```
ξ_nuevo = X softmax(β Xᵀ ξ)
```

Tres pasos: comparar el estado con cada patron (Xᵀξ), convertir las similitudes en pesos
con softmax y promediar los patrones con esos pesos. **No hay pesos que entrenar**: la
memoria es la propia matriz de patrones.

### Resultados teoricos del articulo

- La energia no aumenta con la regla de actualizacion, y la iteracion converge a un
  punto estacionario de E.
- Si un patron esta bien separado de los demas (separacion Δ_μ = x_μᵀx_μ − max_{ν≠μ}
  x_μᵀx_ν grande frente a 1/β y al logaritmo del numero de patrones), **una
  actualizacion basta** para llegar cerca de el, con un error que decrece
  exponencialmente con β·Δ_μ.
- El numero de patrones aleatorios que se pueden almacenar y recuperar crece
  **exponencialmente** con la dimension d.
- Hay tres tipos de punto fijo segun β y la separacion: un **punto fijo global**
  (promedio de todos los patrones), **estados metaestables** (promedio de un grupo de
  patrones parecidos) y **puntos fijos cerca de un unico patron**.

### Equivalencia con la atencion

Con proyecciones aprendidas Q = R·W_Q, K = Y·W_K, V = K·W_V y β = 1/√d_k, aplicar la regla
de actualizacion a todas las consultas a la vez da

```
Z = softmax(Q Kᵀ / √d_k) V
```

la atencion de producto escalar de Vaswani et al. (2017). Sobre esta base los autores
proponen tres capas para redes profundas: `Hopfield` (asociacion general entre dos
conjuntos), `HopfieldPooling` (un patron de consulta aprendido que resume un conjunto) y
`HopfieldLayer` (patrones almacenados aprendidos, como una memoria de prototipos).

---

## 5. Datos y metodologia del articulo

- **Analisis teorico**: demostraciones de convergencia, capacidad y error de
  recuperacion (apendice extenso del articulo).
- **Analisis de BERT**: miden en un modelo BERT ya entrenado en cuantos elementos reparte
  cada cabeza de atencion su peso, y asi clasifican las cabezas segun el tipo de punto
  fijo en que operan.
- **Clasificacion con instancias multiples** (*multiple instance learning*): cada
  ejemplo es una bolsa de muchas instancias y solo la bolsa tiene etiqueta. Incluye la
  clasificacion de **repertorios inmunes**, con cientos de miles de secuencias por
  paciente, y conjuntos clasicos de MIL.
- **Conjuntos UCI pequenos**: decenas de problemas de clasificacion tabular con pocos
  datos, donde el aprendizaje profundo suele rendir mal.
- **Diseno de farmacos**: prediccion de propiedades de moleculas.

---

## 6. Resultados

**Del articulo** (segun su resumen):

1. En BERT, las cabezas de las primeras capas tienden a promediar globalmente y las de
   capas superiores promedian parcialmente, en estados metaestables.
2. Las capas de Hopfield mejoran el estado del arte en tres de cuatro problemas de
   instancias multiples considerados y en la clasificacion de repertorios inmunes.
3. En los conjuntos UCI pequenos, las capas de Hopfield obtienen el mejor resultado
   frente a distintos metodos de aprendizaje automatico.
4. Alcanzan el estado del arte en dos conjuntos de datos de diseno de farmacos.

> Antes de presentar, verificar en el PDF del articulo las cifras concretas que se
> quieran citar (tablas de resultados de cada aplicacion).

**De nuestra implementacion** ([experimento 13](../resultados/reportes/13_hopfield_moderna.md),
numpy puro, mismas letras de 7x6 que la Actividad 5):

| Prueba | Red clasica (Hebb) | Red moderna |
|---|---|---|
| Letras A-D con 8 px invertidos | 86 % | 100 % en **un** paso |
| Letras A-D con 16 px invertidos (38 %) | 33 % | 79 % (= vecino mas cercano, el optimo) |
| Alfabeto completo (26 letras), sin ruido | 0 % (ninguna es punto fijo) | 100 % |
| Alfabeto completo con 4 px invertidos | 0 % | 99 % |
| Patrones aleatorios recuperados (d = 42, 6 px de ruido) | 100 % solo hasta 4 patrones | 3000 al 100 %, y > 90 % con 100 000 (β = 2) |
| Energia a lo largo de la recuperacion | nunca sube | nunca sube (40 valores de β) |
| Diferencia con softmax(QKᵀ/√d_k)V | --- | ≤ 10⁻¹² (es la misma operacion) |

Ademas se reproducen los tres tipos de punto fijo: con las 26 letras y una B ruidosa,
β = 0.01 da la media de todas las letras, β = 0.2 un estado metaestable que mezcla B y R
y β = 1 la B limpia. Y se observa el limite que predice la teoria: con β = 0.5 y 100 000
patrones la red cae en estados metaestables, mientras que con β = 2 queda cerca del vecino mas
cercano. β debe crecer con el logaritmo del numero de patrones.

---

## 7. Ventajas, limitaciones y alcance

**Ventajas**
- Capacidad exponencial y recuperacion en un paso: una memoria asociativa mucho mas
  potente que la de 1982, sin estados espurios con patrones bien separados.
- Es continua y derivable: se puede insertar como capa en una red profunda y entrenar
  con retropropagacion.
- Da una interpretacion de la atencion de los Transformers y una forma de analizar que
  hace cada cabeza.
- Implementacion trivial: similitudes, softmax y promedio ponderado.

**Limitaciones**
- Guarda los patrones explicitamente: memoria y coste crecen con el numero de patrones P
  (la red clasica lo comprime todo en una matriz d x d de tamano fijo).
- β es un hiperparametro critico: demasiado pequeno da promedios en vez de recuerdos, y
  el valor adecuado depende del numero y la separacion de los patrones.
- Recuperar por producto escalar equivale a buscar el vecino mas cercano: con datos
  reales hace falta aprender proyecciones (W_Q, W_K, W_V) en las que esa similitud tenga
  sentido.
- Los resultados en aplicaciones dependen del ajuste de hiperparametros y se comparan
  con metodos concretos; las garantias teoricas son para patrones bien separados o
  aleatorios.

**Alcance**: el articulo popularizo la vision de los Transformers como memorias
asociativas y renovo el interes por las redes de Hopfield, que desde entonces se usan y
se estudian como componentes de arquitecturas profundas.

---

## 8. Relacion con el curso

| Concepto del curso | En el articulo |
|---|---|
| Red de Hopfield, memoria asociativa (act. 5) | el modelo de partida; misma idea, nueva energia |
| Funcion de energia de Liapunov (act. 5) | nueva energia con log-sum-exp; tambien nunca aumenta |
| Capacidad 0.138·d y estados espurios (exp. 11) | capacidad exponencial; puntos fijos metaestables |
| Regla de Hebb, una sola pasada (act. 5, exp. 07) | no hay pesos: la memoria son los propios patrones |
| Activaciones sigmoidales (clase 1) | softmax: generalizacion de la sigmoide a varias clases |
| Retropropagacion (act. 3) | las capas de Hopfield se entrenan con backprop dentro de redes profundas |
| Competencia del ganador en Kohonen (act. 4) | softmax con β grande es una competencia "el ganador se lo lleva todo" suave |

---

## 9. Conclusiones

- Cambiar la energia cuadratica por un log-sum-exp convierte la red de Hopfield en una
  memoria de capacidad exponencial que recupera en un solo paso.
- Su regla de actualizacion es exactamente la atencion de los Transformers: la memoria
  asociativa de 1982 y los modelos de lenguaje actuales estan conectados.
- β controla si la red promedia todo, recuerda grupos o recuerda un patron, y eso
  permite interpretar que hace cada cabeza de atencion.
- Nuestra implementacion en numpy confirma con las letras de la Actividad 5 la
  recuperacion en un paso, la capacidad muy superior, los tres tipos de punto fijo y la
  equivalencia con la atencion.

---

## 10. Guion de la presentacion oral (~15 min)

Diapositivas: [`presentaciones/03_seminario_hopfield_moderna.pptx`](../presentaciones/03_seminario_hopfield_moderna.pptx).

| # | Diapositiva | Contenido | Min |
|---|---|---|---|
| 1 | Titulo y referencia | articulo, autores, ICLR 2021, arXiv | 0.5 |
| 2 | Punto de partida | Hopfield 1982: memoria asociativa, energia, letras de la act. 5 | 1.5 |
| 3 | El problema | capacidad 0.138·d, estados binarios, la atencion sin teoria | 1.5 |
| 4 | La nueva energia | de z² a exp: Hopfield, Krotov-Hopfield, Demircigil, Ramsauer | 1.5 |
| 5 | Regla de actualizacion | ξ ← X softmax(βXᵀξ) en tres pasos | 1.5 |
| 6 | Hopfield = atencion | mapeo Q, K, V; verificacion numerica | 2 |
| 7 | Nuestra implementacion | letras recuperadas en un paso; ruido frente a la clasica | 1.5 |
| 8 | Capacidad | 26 letras y patrones aleatorios: clasica frente a moderna | 1.5 |
| 9 | Tres tipos de punto fijo | β: global, metaestable, unico; BERT | 1.5 |
| 10 | Aplicaciones del articulo | capas Hopfield, MIL, repertorios inmunes, UCI, farmacos | 1 |
| 11 | Ventajas y limitaciones | | 1 |
| 12 | Conclusiones y preguntas | resumen en 3 frases | 0.5 |

---

## Referencias

- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective
  computational abilities. *PNAS*, 79(8), 2554-2558.
- Amit, D. J., Gutfreund, H. y Sompolinsky, H. (1985). Storing infinite numbers of
  patterns in a spin-glass model of neural networks. *Physical Review Letters*, 55, 1530.
- Krotov, D. y Hopfield, J. J. (2016). Dense associative memory for pattern recognition.
  *NeurIPS 2016*.
- Demircigil, M., Heusel, J., Löwe, M., Upgang, S. y Vermet, F. (2017). On a model of
  associative memory with huge storage capacity. *Journal of Statistical Physics*, 168,
  288-299.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Ramsauer, H. et al. (2021). Hopfield Networks is All You Need. *ICLR 2021*.
  arXiv:2008.02217.
