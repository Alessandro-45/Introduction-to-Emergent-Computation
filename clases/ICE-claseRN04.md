# **Aspectos de Computación Neuronal** **_Adaline_** 

**_Prof. Esteban Alvarez_** 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## PERCEPTRON 

El PERCEPTRON es un sistema de aprendizaje basado en ejemplos capaz de realizar tareas de clasificación. Sin embargo, existen un gran número de problemas abordables desde la perspectiva del aprendizaje basado en ejemplos que no se reducen a tareas de clasificación. La característica de clasificador del PERCEPTRON viene dada por la naturaleza de sus salidas . En la capa de salida las células codifican una función de salida en escalón F(s)=1 si s>0, F(s)= -1 en caso contrario, que transforma la suma ponderada de las entradas que es un valor real, en una salida binaria. _<u>Al ser binarias, sólo pueden codificar</u>_ <u>.</u> _<u>un conjunto discreto de estados</u>_ 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## PERCEPTRON 

Si las salidas fueran números reales, podrían codificar cualquier tipo de salida y se convertirían en sistemas de resolución de problemas generales (ej. Aproximar una función cualquiera F(xi)=yi ). Sin embargo, la naturaleza de la regla de aprendizaje del PERCEPTRON no permite producir salidas reales 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

Widrow y Hoff (1960) propusieron un sistema de aprendizaje que sí tuviera en cuenta el error producido, y diseñaron lo que denominaron ADAptive LInear Neuron, (ADALINE). Esta es una estructura prácticamente idéntica a la del perceptron. Es un elemento combinador adaptativo, que recibe un conjunto de entradas y las combina para producir una salida. El aprendizaje en este caso incluye la diferencia entre el valor real producido en la capa de salida para un patrón de entrada x<sup>p</sup> y el que debería haber producido dicho patrón (salida esperada d<sup>p</sup> ), que está en el conjunto de aprendizaje (| d<sup>p</sup> - x<sup>p</sup> |). A esta regla de aprendizaje se la conoce con el nombre de regla Delta. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

La diferencia con respecto a la regla de aprendizaje del PERCEPTRON es la manera de utilizar la salida, una diferencia fundamental entre ambos sistemas. El PERCEPTRON utiliza la salida de la función umbral para el aprendizaje; sin embargo, la regla Delta utiliza directamente la salida de la red, sin pasarla por ninguna función umbral. 

El objetivo es obtener un valor determinado y=d<sup>p</sup> cuando el conjunto de valores Xi<sup>p</sup> , i=1,...,n se introduce en la entrada. Por lo tanto el problema será obtener los valores de wi , i=1,...,n que permitan realizar lo anterior para un número arbitrario de patrones de entrada. 

No es posible conseguir una salida exacta, pero si minimizar la desviación cometida por la red, esto es , minimizar el error cometido por la red para la totalidad del conjunto de patrones de ejemplo.5 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

Habitualmente la medida de error global utilizada es el error cuadrático medio. 



La regla intentará minimizar este valor para todos los elementos del conjunto de patrones de aprendizaje. La manera de minimizar este error es recurrir a un proceso iterativo en el que se van  presentando los patrones uno a uno, y modificando los parámetros de la red, pesos de las conexiones, mediante **_la regla del descenso del_** . **_gradiente_** 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

La idea es realizar un cambio en cada peso proporcional a la derivada del error, medida en el patrón actual, respecto al peso: _Δ p w j_ =− _γ_<sup>∂</sup><sup>_E_</sup> _p_ ∂ _w j_ Para el calculo de la derivada anterior se utiliza la ∂ _A_ ∂ _<u>y</u> p p p_ regla de la cadena: ∂ _x_<sup>=∂</sup> ∂<sup>_A_</sup> _y_ ∂ _x_ ∂ _E_ ∂ _<u>y</u>_ =<sup>∂</sup><sup>_E_</sup> ∂ _w j_ ∂ _y_<sup>_p_</sup> ∂ _w j_ Al ser unidades lineales sin función de activación en la capa de salida , se cumple: _p_ ∂ _<u>y</u> p_ ∂ _E_ = _x j_ ∂ _w j_ ∂ _y_<sup>_p_=−(</sup><sup>_d p_−</sup><sup>_y p_)</sup> 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

que sustituyendo queda como sigue: _Δ p w j_ = _γ_ ( _d_<sup>_p_</sup> − _y_<sup>_p_</sup> ) _x j_ 

La regla Delta es una extensión de la regla del PERCEPTRON a valores de salida reales. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

- <u>:</u> 

- **<u>Procedimiento de aprendizaje definido por la regla Delta</u>** 

1. Inicializar los pesos de manera aleatoria. 

2. Introducir un patrón de entrada. 

3. Calcular la salida de la red, compararla con la deseada y obtener la diferencia (d<sup>p</sup> -y<sup>p</sup> ). 

4. Para todos los pesos , multiplicar dicha diferencia por la entrada correspondiente, y ponderarla por una tasa de aprendizaje . 

5. Modificar el peso restando del valor antiguo la cantidad obtenida en 4. 

6. Si no se ha cumplido el criterio de convergencia, regresar a 2; si se han acabado todos los patrones, empezar de nuevo a introducir patrones. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

### <u>Diferencias entre ambos modelos de Redes Neuronales artificiales:</u> 

1. En el PERCEPTRON la salida es binaria, en el ADALINE es real. 

2. En el PERCEPTRON la diferencia entre entrada y salida es 0 si ambas pertenecen a la misma categoría y ±1 si por el contrario pertenecen a categorías diferentes. En el ADALINE se calcula la diferencia real entre entrada salida. 

3. En el ADALINE existe una medida de cuánto se ha equivocado la red; en el PERCEPTRON sólo se determina si se ha equivocado o no. 

4. En el ADALINE hay una razón de aprendizaje () para regular cuánto va a afectar cada equivocación a la modificación de los pesos. Es siempre un valor entre 0 y 1 para ponderar el aprendizaje. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

• _<u>Descodificador de binario a decimal</u>_ Como ejemplo para ilustrar el funcionamiento del ADALINE se ha elegido un descodificador  de binario a decimal. El descodificador recibe una entrada en binario y produce como salida su valor en decimal. Este es un problema trivial, ya que existe una expresión analítica capaz de realizar la descodificación: 

_n d_ =∑ _i_ =1 2<sup>_i_</sup> _xi_ 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

Compruebe si una red de tipo ADALINE es capaz de aproximar la expresión por si sola, a partir de un conjunto de ejemplos de entrenamiento. 

- Elija un conjunto de patrones de entrenamiento de dimensión 3. 

- Como se trata de pasar de binario a decimal las entradas estarán en binario, y como el tamaño prefijado es tres el número máximo de entradas que se pueden generar es de 2<sup>3</sup> =8. 

- Inicializar los pesos de manera aleatoria de 0 a 1. 

- Refleje en pantalla de manera secuencial el número de iteración, pesos y el error por patrón. 

- Comentar los valores óptimos de los pesos. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## ADALINE 

_Limitaciones del modelo:_ El problema de este tipo de modelos, y la razón por la que fueron descartados como solución general de problemas, es que solo resuelven problemas en los que los ejemplos son linealmente separables en términos de clasificación, o en los que las salidas son funciones lineales de las entradas. El caso del descodificador es un ejemplo. Sin embargo, la mayoría de los problemas, no son separables linealmente. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

