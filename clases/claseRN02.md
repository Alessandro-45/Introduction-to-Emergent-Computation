

# **Aspectos de Computación Neuronal** **_Modelos Iniciales_** 

**_Prof. Esteban Alvarez_** 

Física Computacional II- Prof. E. Alvarez 



## **_Modelo Neuronal_** 

Aquí se desea introducir un modelo sencillo de la neurona, para construir redes, nuestro fin último es modelar correctamente el comportamiento global de toda la red. No se pretende modelar exactamente el comportamiento fisiológico de la neurona, sino más bien sus características más relevantes, que entran en juego en su interacción con toda la red. 

<u>.</u> Se presenta a continuación un _<u>esquema de neurona</u>_ 



<!-- Start of picture text -->
x1<br>(in))<br>wj1 ϕ (yj<br>v1j<br>xi y j<br>wji<br>v2j<br>wjn<br>xn<br>Física Computacional II- Prof. E. Alvarez<br><!-- End of picture text -->



## **_Modelo Neuronal_** 

Lo que hace cada peso sináptico es simplemente multiplicar a su entrada correspondiente y define la importancia relativa de cada entrada. Recordemos que en el _soma_ de la neurona biológica se _sumaban las entradas provenientes de todas las dendritas_ . Entonces _<u>es</u>_ <u>:</u> tenemos que _<u>la entrada total a la neurona y</u>_ _~~<u>j</u>~~_ 

_n_ ( _in_ ) _y j_ = ∑ _w ji xi i_ =1 

En donde el índice (in) denota “input” o entrada. Como mencionamos la neurona se activa si la entrada total supera un cierto umbral. Lo que para esto es aplicar una _función de activación_ ϕ sobre (in), que puede ser, por ejemplo, una función tipo escalón o yj sigmoidea, como la tangente hiperbólica. 

Física Computacional II- Prof. E. Alvarez 



## **_Modelo Neuronal_** 

Entonces tenemos que la señal de output o salida de la neurona yj es : ( _in_ ) _y_ =ϕ _y j_ ( _j_ ) 

Estas funciones evidentemente transmiten la idea de “disparar sobre un umbral”. Además, a veces se suele usar como función de activación una relación lineal, generalmente la función identidad. Esta se usa por lo general para neuronas de entrada a la red o sensores. Esto se debe a que evidentemente, lo que esperamos de un sensor es que indique precisamente lo que está percibiendo. 

si la función de activación de una neurona es lineal, decimos que es una _neurona lineal_ ,  en caso contrario, decimos que es una _neurona no lineal_ . Aquí a las neuronas lineales se las representa por un cuadrado, y a las no lineales por un círculo. 

Física Computacional II- Prof. E. Alvarez 



## **_El Comienzo:McCulloch-Pitts_** 

Este consiste en el primer modelo que se creó de red neural, el año 1943, antes que se construyeran los primeros computadores. McCulloch era un siquiatra y neuroanatomista y Pitts un matemático. su modelo tiene las siguientes características: 

- Las neuronas son del tipo binario. 

- Los umbrales y las sinapsis se mantienen fijas. 

- La función de activación es del tipo escalón. 

ellos demostraron que todas las funciones lógicas se pueden describir mediante combinaciones apropiadas de neuronas de este tipo, y que por lo tanto, se podía crear, en principio, una red capaz de resolver cualquier función computable. Además, el modelo sirve para describir algunos fenómenos biológicos sencillos. 

Física Computacional II- Prof. E. Alvarez 



## **_El Comienzo:McCulloch-Pitts_** 

De esta forma es posible describir algunas funciones lógicas como: **_Función And_** 



<!-- Start of picture text -->
x1 x2 y<br>1<br>1 1 1<br>x1<br>0 1 y 0<br>1 0 1 0<br>x2<br>0 0 0<br>in<br>El umbral de cada neurona no lineal es 2: o sea 0 si y < 2<br>y = 1 si y in ≥ 2<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 



## **_El Comienzo:McCulloch-Pitts_** 

#### **_Función Or_** 



<!-- Start of picture text -->
x1 x2 y<br>2<br>1 1 1<br>x1<br>0 1 y 1<br>1 0 2 1<br>x2<br>0 0 0<br>in<br>0 si y < 2<br>El umbral de cada neurona no lineal es 2: o sea y = 1 si y in ≥ 2<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 



### **_Modelos de Redes Neuronales Artificiales_** 

Los modelos de redes neuronales artificiales han sido estudiados por muchos años con la esperanza de llevar a cabo tareas o acciones semejantes a la del ser humano en el campo del habla y reconocimiento de imágenes. 

En vez de ejecutar un programa de instrucciones secuencialmente como en un computador de von Neumann, los modelos de redes neuronales exploran muchas hipótesis compartidas  simultáneamente, usando masivamente redes paralelas compuestas de muchos elementos computacionales conectados por medio de enlaces con pesos variables. <u>Los modelos de red neuronal son precisados por la topología de la red, características de los nodos, y entrenamientos o reglas de . aprendizaje</u> 

Física Computacional II- Prof. E. Alvarez 



## **_Omisiones importantes del modelo de McCulloch-Pitts_** 

- Neuronas reales **_<u>no son</u>_** en lo más mínimo unidades de umbral. Son mas bien unidades de respuesta continua. 

- Es de importancia la **_<u>relación no lineal</u>_** entre la entrada y la salida, constituye un rasgo universal en las neuronas. 

- El “computo” es más complejo, puede hasta llegar a ser una sumatoria <u>.</u> 

- no-lineal con apreciable **_<u>pre-procesamiento dendrítico</u>_** 

- Responden con una secuencia de pulsos, no con un solo nivel de salida <u>.</u> 

- representativo de la **_<u>razón de disparo</u>_** 

- No todas las neuronas responden con un retardo constante. t ->t+1. Tampoco son sincrónicas siguiendo un reloj central. 

- La cantidad de neurotransmisores segregados en una sinapsis puede variar de manera impredecible. Por lo que el modelaje podría involucrar una componente estocástica en la dinámica. 

Física Computacional II- Prof. E. Alvarez 



### **_Modelos de Redes Neuronales Artificiales_** 

Una clasificación de seis importantes redes neuronales que pueden ser usadas para la clasificación de patrones estáticos se presenta a continuación: 

##### Clasificación de las RNA para patrones fijos 



<!-- Start of picture text -->
Entrada binaria<br>Entrada continua<br>Supervisado No-supervisado Supervisado No-supervisado<br>Clasificador<br>Red de Red de Perceptrón Perceptrón  Kohonen<br>Carpenter/<br>Hopfield Hamming Multicapa (Mapa A-O)<br>Grossberg<br>Algoritmo<br>clasificador K-vecinos<br>Algoritmo  de<br>Clasificador  Gausiano mas cercano<br>de Clustering<br>Optimo<br>Clustering (K-medio)<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 



### **_Modelos de Redes Neuronales Artificiales_** 

Esta clasificación toma en cuenta en primer lugar la naturaleza del tipo de información a ser suministrada a la red : entrada binaria o entrada continua. En segundo lugar se considera si el entrenamiento es supervisado o no supervisado. 

Las redes con supervisión tales como la red de Hopfield y el perceptrón son usadas como memorias asociativas o como clasificadores. Estas redes son provistas con parte de la información que especifica la clase correcta para nuevos patrones de entrada durante el entrenamiento. Las redes entrenadas sin supervisión, tales como los mapas de Kohonen, son usados como vectores para formar cluster. Ninguna información relacionada a la clase correcta es dada a la red durante el entrenamiento. 

Física Computacional II- Prof. E. Alvarez 



## **_El perceptrón_** 

_<u>El perceptrón simple</u>_ (P.S.) es el más elemental de todos, dispone de una sola capa. Su modo de entrenamiento es del tipo supervisado. 

Esta red simple generó mucho interés en los años 60 por su capacidad para aprender a reconocer patrones sencillos: Un perceptrón, formado por varias neuronas lineales para recibir las entradas de la red y una neurona de salida, capaz de decidir cuando una entrada presentada a la red pertenece a una de las dos clases que es capaz de reconocer. 

Los dispositivos tipo (P.S.) consisten en un número fijo n de U.P.s C/U con m líneas de entrada por las cuales reciben patrones m-dimensionales C/U correlacionado con un patrón n-dimensional binario o bipolar (ej.). _<u>El dispositivo debe ser capaz de aprender la correlación entrada/salida.</u>_ 

Física Computacional II- Prof. E. Alvarez 



## **_El perceptrón_** 

En términos matemáticos los conjuntos de entrada/salida definen una relación y el perceptrón debe sintetizar una representación de la misma. 

Una técnica útil para analizar el comportamiento de este tipo de redes consiste en graficar las regiones de decisión creadas en el espacio multidimensional definido por las variables de entrada. Estas regiones especifican que valores de entrada corresponden a cada clase. 

Este modelo sólo es capaz de discriminar patrones muy sencillos, linealmente separables. _<u>La separabilidad lineal</u>_ limita a las redes tipo perceptrón simple a la resolución de problemas en los cuáles el conjunto de puntos (correspondientes a los valores de entrada) sean separables geométricamente. En el caso particular de dos entradas, la separación se lleva a cabo mediante una _<u>línea recta</u>_ <u>. Para tres entradas, la separación se realiza</u> mediante un _<u>plano</u>_ en el espacio tridimensional, y así sucesivamente hasta el caso de n entradas, en el cuál el espacio N-dimensional es dividido en un . _<u>hiperplano</u>_ 

Física Computacional II- Prof. E. Alvarez 





<!-- Start of picture text -->
El perceptrón<br>A<br>x0<br>A<br>A A<br>w0 A<br>B<br>A<br>B<br>y<br>entrada salida A B B<br>B<br>wN-1<br>xN-1<br>− w θ<br>0<br> N −1  +1 clase A x 1 = x 0 +<br>y = f ∑ i =0 wi x i −θ y = −1 clase B w 1 w 1<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 

