# Redes Neuronales Artificiales desde cero

Implementacion **desde cero**, sin librerias de aprendizaje automatico, de los modelos
neuronales clasicos de la asignatura *Introduccion a la Computacion Emergente*
(Prof. Esteban Alvarez): la neurona de **McCulloch-Pitts** y las compuertas logicas
AND y OR, el **perceptron simple** de una capa, el **ADALINE** con la regla Delta y la
**regla de Hebb**.

Las unicas dependencias son `numpy`, `pandas` y `matplotlib`. Toda la maquinaria
neuronal --- propagacion, reglas de aprendizaje, criterios de parada, metricas y
diagramas de red --- esta escrita a mano y documentada linea a linea.

```bash
python3 -m pip install -r requirements.txt
python3 experimentos/ejecutar_todo.py     # regenera 37 figuras, 41 tablas y 8 informes
python3 pruebas/pruebas.py                # 29 pruebas, sin dependencias externas
```

---

## Que hay aqui

| Modelo | Ano | Que aporta | Modulo | Documentacion |
|---|---|---|---|---|
| **McCulloch-Pitts** | 1943 | Neurona binaria de umbral **fijo**; compuertas AND y OR; universalidad por composicion | [`mcculloch_pitts.py`](ce_rna/mcculloch_pitts.py) | [doc 01](docs/01_mcculloch_pitts.md) |
| **Regla de Hebb** | 1949 | Aprendizaje de **una sola pasada** por correlacion | [`hebb.py`](ce_rna/hebb.py) | [doc 04](docs/04_regla_hebb.md) |
| **Perceptron simple** | 1958 | Primera red **que aprende**; regla de correccion de error con teorema de convergencia | [`perceptron.py`](ce_rna/perceptron.py) | [doc 02](docs/02_perceptron_simple.md) |
| **ADALINE** | 1960 | Salida **real**; regla Delta = descenso del gradiente sobre el error cuadratico medio | [`adaline.py`](ce_rna/adaline.py) | [doc 03](docs/03_adaline.md) |

Los cuatro comparten la misma arquitectura unicapa; lo unico que cambia entre ellos es
**como se calcula `Δw`**, y de ahi salen todas las diferencias de comportamiento que
los experimentos miden.

---

## Los ocho experimentos

Cada experimento es un script ejecutable que produce figuras, tablas CSV y un informe
en Markdown **generado automaticamente** a partir de la misma ejecucion que produce
los numeros.

| # | Experimento | Pregunta | Informe |
|---|---|---|---|
| 00 | Funciones de activacion | ¿Cuales son derivables y por que importa? | [ver](resultados/reportes/00_activaciones.md) |
| 01 | McCulloch-Pitts: AND y OR | ¿Puede una neurona de umbral calcular AND, OR y XOR? | [ver](resultados/reportes/01_mcculloch_pitts.md) |
| 02 | Perceptron: AND y OR | ¿Puede una red aprender los pesos por si sola? | [ver](resultados/reportes/02_perceptron_and_or.md) |
| 03 | Perceptron: el XOR | ¿Por que falla, y de quien es la culpa? | [ver](resultados/reportes/03_perceptron_xor.md) |
| 04 | Perceptron: letras X y O | ¿Funciona sobre reconocimiento real con ruido? | [ver](resultados/reportes/04_perceptron_letras.md) |
| 05 | ADALINE: descodificador binario-decimal | ¿Puede aproximar una funcion de salida real? | [ver](resultados/reportes/05_adaline_decodificador.md) |
| 06 | Perceptron frente a ADALINE | ¿Que gana y que pierde cada uno? | [ver](resultados/reportes/06_perceptron_vs_adaline.md) |
| 07 | Regla de Hebb | ¿Hasta donde llega el aprendizaje de una sola pasada? | [ver](resultados/reportes/07_regla_hebb.md) |

---

## Algunos resultados

### Las compuertas AND y OR como neuronas de umbral

<p align="center">
  <img src="resultados/figuras/01_arquitectura_and.png" width="47%">
  <img src="resultados/figuras/01_frontera_and.png" width="47%">
</p>

Pesos `(1, 1)` y umbral `θ = 2` para el AND; `(2, 2)` y `θ = 2` para el OR. Ambas
tablas de verdad se reproducen exactamente, y ambas fronteras son rectas: los dos
problemas son **linealmente separables**.

### El XOR: donde se rompe la red unicapa

<p align="center">
  <img src="resultados/figuras/01_xor_no_separable.png" width="47%">
  <img src="resultados/figuras/01_red_xor.png" width="47%">
</p>

Una busqueda exhaustiva sobre **68 921 rectas** confirma que el maximo alcanzable
sobre el XOR es 3 patrones de 4. El algoritmo perceptronico nunca se detiene. La
solucion no es cambiar la regla de aprendizaje sino **anadir una capa**: componiendo
tres neuronas (`OR ∧ ¬AND`) el XOR sale exacto.

### El perceptron aprende, y encuentra una solucion entre infinitas

<p align="center">
  <img src="resultados/figuras/02_evolucion_and.png" width="47%">
  <img src="resultados/figuras/02_infinitas_soluciones.png" width="47%">
</p>

Cada correccion rota y desplaza el hiperplano. Con sesenta inicializaciones distintas
se obtienen sesenta fronteras validas distintas: *"o no existe ninguna solucion, o
existen infinitas"*.

### Lo que la red memoriza son plantillas

<p align="center">
  <img src="resultados/figuras/04_pesos_plantilla.png" width="62%">
</p>

Los pesos de la red de letras, vistos como imagen de 5x5. Un detalle que el informe
documenta y que contradice la intuicion: cada neurona no almacena su propia letra sino
la **anti-plantilla de su rival** --- porque el perceptron aprende lo primero que
resuelve el problema, no lo que uno espera encontrar.

### El ADALINE desciende por un paraboloide

<p align="center">
  <img src="resultados/figuras/05_contorno_error.png" width="47%">
  <img src="resultados/figuras/05_curva_ecm.png" width="47%">
</p>

Sobre el descodificador binario-decimal, la red recupera los pesos `(0, 4, 2, 1)` ---
los valores posicionales de los bits --- con un error cuadratico medio de `3e-17` y
R² = 1. La superficie de error es convexa: un unico minimo, sin minimos locales.

---

## Hallazgos que merecen destacarse

Los informes registran los resultados tal como salieron, incluidos los que contradicen
la version habitual de los manuales:

- **Con pesos iniciales nulos, la razon de aprendizaje del perceptron es irrelevante.**
  Todos los pesos quedan multiplicados por `a`, y la frontera `w·x = 0` no cambia.
- **Entrenar con ejemplos ruidosos empeora el reconocimiento de letras.** Entrenado
  solo con los prototipos limpios, el perceptron produce el filtro adaptado, que es
  optimo; el ruido lo desvia a otra solucion valida pero peor.
- **Sobre AND y OR, el ADALINE converge a la misma frontera que el perceptron.** La
  solucion de minimos cuadrados del AND bipolar es exactamente proporcional a la del
  perceptron: con cuatro patrones simetricos no hay ninguna diferencia que medir.
- **Sobre 60 patrones dispersos, la solucion de minimos cuadrados clasifica peor.**
  Tiene margen negativo mientras el perceptron separa todos los patrones: minimizar el
  error cuadratico **no** es lo mismo que separar clases.
- **La regla de Hebb falla en el AND binario** --- los aportes se cancelan y los pesos
  quedan en cero --- y funciona en el bipolar. La codificacion decide entre funcionar
  y no funcionar.

---

## Estructura

```
ce_rna/            El paquete: modelos, metricas, figuras e informes
experimentos/      Ocho scripts ejecutables, uno por estudio
pruebas/           29 pruebas de propiedades matematicas, sin pytest
docs/              Documentacion teorica y de la API
clases/            Material original de la asignatura
resultados/        Figuras, tablas CSV e informes generados
```

- [00 --- Fundamentos: el modelo de neurona y el vocabulario](docs/00_fundamentos.md)
- [01 --- McCulloch-Pitts y las compuertas logicas](docs/01_mcculloch_pitts.md)
- [02 --- El perceptron simple](docs/02_perceptron_simple.md)
- [03 --- ADALINE y la regla Delta](docs/03_adaline.md)
- [04 --- La regla de Hebb](docs/04_regla_hebb.md)
- [05 --- Guia de uso y referencia de la API](docs/05_guia_de_uso.md)

---

## Uso como libreria

```python
from ce_rna import datasets as ds
from ce_rna.perceptron import PerceptronSimple
from ce_rna.adaline import Adaline

# Perceptron sobre la compuerta AND
conjunto = ds.compuerta("AND", "bipolar")
red = PerceptronSimple(n_entradas=2, razon_aprendizaje=1.0).entrenar(conjunto.X, conjunto.d)
print(red.resumen())
print(red.predecir(conjunto.X))          # [-1. -1. -1.  1.]

# ADALINE sobre el descodificador de 3 bits
conjunto = ds.decodificador_binario(3)
red = Adaline(3, razon_aprendizaje=0.05, max_epocas=1000, tolerancia=1e-9)
red.entrenar(conjunto.X, conjunto.d)
print(red.w)                             # ~ [0. 4. 2. 1.]
```

---

## Verificacion

La bateria de [`pruebas/pruebas.py`](pruebas/pruebas.py) comprueba propiedades
**matematicas**, no solo que el codigo no reviente:

- las compuertas MCP reproducen sus tablas de verdad, y **ninguna** neurona de umbral
  resuelve el XOR;
- el perceptron converge en todo problema separable y no converge en el XOR;
- si el perceptron acierta, los pesos no cambian (paso 4 del algoritmo);
- la regla Delta alcanza la solucion exacta de minimos cuadrados;
- el gradiente implementado coincide con la derivada numerica del error (`1e-6`);
- las derivadas de las activaciones coinciden con sus diferencias finitas;
- la regla de Hebb falla en el AND binario y funciona en el bipolar.

```
29 de 29 pruebas superadas
```

---

## Creditos

Material teorico y ejercicios: **Prof. Esteban Alvarez**, asignatura *Introduccion a
la Computacion Emergente → Redes Neuronales Artificiales*. Las transcripciones de las
clases estan en [`clases/`](clases/) y todos los modulos citan la lamina concreta que
implementan.

Licencia [MIT](LICENSE).
