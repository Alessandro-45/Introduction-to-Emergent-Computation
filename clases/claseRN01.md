

# **Aspectos de Computación Neuronal** **_Conceptos básicos_** 

**_Prof. Esteban Alvarez_** 

Física Computacional II- Prof. E. Alvarez 



## Objetivos 

- Proporcionar una introducción a las Redes Neuronales Artificiales, presentando los conceptos fundamentales de este tipo de elementos de procesamiento de la información. 

- Hacer un estudio un poco detallado de los tipos de redes neuronales artificiales más utilizados en aplicaciones genéricas. 

- Abordar la aplicación de las redes neuronales artificiales como elementos base para tratar problemas genérico como clasificación o detección de patrones. 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

- Las **_RNA_** intentan modelar la estructura y funcionamiento de algunos componentes del sistema nervioso animal. 

- Aunque su similitud es relativa, sí existen elementos funcionales parecidos entre las RNA biológicas y formales, como veremos. 

- **<u>El sistema nervioso animal</u>** se divide entre sistema nervioso central (médula espinal y encéfalo) y periférico(ramificaciones nerviosas a los distintos órganos). 

- Dentro del sistema nervioso central, **<u>el encéfalo</u>** es el órgano fundamental de procesamiento de la información. 

- **<u>El encéfalo o cerebro</u>** está constituido por millones de neuronas (>10<sup>8</sup> ) de diferentes tamaños y formas. 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

- Estas neuronas se constituyen en redes de diferentes tipos que realizan diferentes funciones especializadas. 

- Esto dificulta extraordinariamente su modelización. 

- La mayor parte de las **<u>neuronas</u>** tiene tres estructuras diferenciadas: el cuerpo celular, las dendritas y el axón (elementos a destacar de su estructura histológica). 

- **_<u>Las dendritas</u>_** <u>, que son la vía de entrada de las señales que se</u> combinan en el <u>cuerpo de la neurona. De alguna manera la neurona</u> elabora una señal de salida a partir de ellas. 

- **_<u>El cuerpo celular</u>_** es el centro de síntesis de la neurona y procesa las señales o impulsos que le llegan de otras células generando un nuevo impulso si se dan ciertas condiciones. 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

- **_<u>El axón</u>_** actúa como canal transmisor de los impulsos generados y conecta con las dendritas de otras células a través de las sinapsis. 

- **_<u>La sinapsis</u>_** <u>, que son las unidades funcionales y estructuras elementales</u> que median entre las interacciones de las neuronas. En las terminaciones de la sinapsis se encuentran unas vesículas que contienen unas sustancias químicas llamadas neurotransmisores, que ayudan a la propagación de las señales electroquímicas de una neurona a otra. 

- La naturaleza de estas conexiones es química (neurotransmisores), la eficiencia o fuerza de la conexión se modifica a lo largo del tiempo y <u>.</u> 

- es cuando decimos que **_<u>el cerebro aprende</u>_** 

Física Computacional II- Prof. E. Alvarez 



_Esquema simplificado de la morfología de una neurona típica_ 

• Elementos a destacar de su estructura histológica Sinapsis 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

- Las neuronas en estado de inactividad presentan un potencial de reposo, sin embargo cuando reciben un determinado impulso de cierta intensidad que supera un determinado umbral, pasan a un potencial de acción o impulso nervioso que se propaga a otras neuronas. 

- Por este motivo se habla de **_<u>una ley</u>_** de todo o nada **_<u>del impulso nervioso.</u>_** 

- Hay que señalar que los impulsos que recibe una célula pueden ser excitatorios o inhibitorios, **<u>la suma total de todos estos impulsos</u>** es lo <u>.</u> 

- que **<u>determina el potencial de activación</u>** 

- La suma de potenciales puede ser debida a señales de distintas células (sumación espacial) o por impulsos recibidos en distintos instantes de tiempo (sumación temporal). 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

• <u>Simplificando:</u> **_la función básica de una célula es sumar entradas y producir una determinada salida si la suma es mayor que un umbral determinado._** 

- _<u>Al conjunto de neuronas y sus correspondientes sinapsis</u>_ lo llamamos **<u>redes de neuronas</u>** o **<u>redes neuronales</u>** <u>, a la simplificación teórica de</u> cada una de las <u>neuronas .</u> lo llamamos **<u>elemento de proceso</u>** 

Física Computacional II- Prof. E. Alvarez 



## Introducción a Redes Neuronales Artificiales 

En el ámbito del  aprendizaje el concepto de <u>plasticidad</u> está relacionado con la capacidad del cerebro frente a la adaptación de acuerdo a los estímulos exteriores. En un cerebro adulto la plasticidad se puede llevar a cabo por dos procedimientos: <u>creación de nuevas conexiones sinápticas entre las neuronas</u> y <u>la modificación de las ya existentes.</u> 

Así como la _plasticidad_ parece <u>la clave esencial en el funcionamiento de las neuronas como elemento de proceso de la información, dicho elemento se modeliza e intenta imitar en las Redes</u> Neuronales Artificiales. 

Física Computacional II- Prof. E. Alvarez 



#### Definiciones de Computación Neuronal 

- Una red neuronal es un procesador masivamente paralelo distribuido que es propenso por naturaleza a almacenar conocimiento experimental y hacerlo disponible para su uso (Simon Haykin/Neural Networks). 

Este mecanismo se parece al cerebro en dos aspectos: 

1. El conocimiento es adquirido por la red a través de un <u>.</u> proceso que se denomina _<u>aprendizaje</u>_ 

2. El conocimiento se almacena mediante la modificación de la fuerza o _<u>peso sináptico</u>_ de las distintas uniones entre neuronas. 

Física Computacional II- Prof. E. Alvarez 



#### Definiciones de Computación Neuronal 

Otra definición de Red Neuronal es la ofrecida por Ben J.A. Y P. Patrik/ An introduction to neural networks. “Una red neuronal es un _<u>modelo computacional</u>_ con un conjunto de propiedades específicas, como son la habilidad de adaptarse o aprender, generalizar u organizar la información, todo ello basado en un procesamiento eminentemente paralelo”. 

<u>:</u> _<u>Desde el punto de vista Ingenieril</u>_ Las redes neuronales artificiales son _<u>modelos matemáticos multiparamétricos no-lineales</u>_ capaces de inducir una correspondencia entre conjuntos de patrones de información (la relación estimulo-respuesta). 

El procesamiento de la información se realiza de manera tal de emular los sistemas neuronales biológicos. 

Física Computacional II- Prof. E. Alvarez 



**_Un modelo de neurona. El elemento de proceso._** 

Una _<u>neurona</u>_ es un _<u>elemento de procesamiento</u>_ de la información que juega un papel fundamental en la red neuronal. 

A continuación se presenta un modelo de neurona con sus tres elementos fundamentales: 



<!-- Start of picture text -->
Sefales de Pesos xXonl<br>Entrada Sinapticos<br>xy W i=94,<br>(wi) Funcion de Xy<br>Activacién Wi<br>x. .<br>2 (ui) rm We ot)<br>: O(-) Yk Xe Ye<br>. Salida<br>. Wig<br>Sumador<br>%,<br>Xp x p<br>Umbral<br>a) b)<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 



##### **_Un modelo de neurona. El elemento de proceso._** 

- Un conjunto de sinapsis o _<u>conexiones</u>_ <u>, cada una de ellas caracterizada</u> por su fuerza o _<u>peso</u>_ <u>. Así una señal de entrada xj tras pasar la conexión,</u> se habrá convertido en una señal xjwkj donde wkj es el peso o fuerza de la conexión con la entrada j-ésima de la neurona k. De acuerdo con el signo del peso wkj se tienen conexiones excitadoras cuando es positivo, y conexiones inhibidoras cuando es negativo. 

- Un _<u>sumador</u>_ Σ , que produce la suma ponderada de las entradas de acuerdo a los correspondientes pesos de las conexiones. 

- Una _<u>función de activación o transferencia</u>_ <u>, que tiene como misión</u> limitar la amplitud de la salida generada por la neurona. 

Física Computacional II- Prof. E. Alvarez 



##### **_Un modelo de neurona. El elemento de proceso._** 

También es habitual la inclusión en el modelo de un <u>umbral o polarización</u> representado por θk , cuya misión es controlar el nivel a partir del cual la neurona produce su salida. Este término θk es añadido a la suma ponderada, que posteriormente es transformada por la función de activación φ(). 

De acuerdo con este modelo, se puede describir el comportamiento de la neurona mediante las siguientes expresiones: 



<!-- Start of picture text -->
p<br>µ k = ∑ wkj x j<br>j =1<br>ν k = µ k −θ k<br>y k =φ(ν k )<br><!-- End of picture text -->

Física Computacional II- Prof. E. Alvarez 



##### **_Un modelo de neurona. El elemento de proceso._** 



La función de activación determina el nivel de activación de la neurona en términos de la actividad existente en sus entradas. 

Física Computacional II- Prof. E. Alvarez 



## _Algunas funciones de activación normalmente utilizadas_ 

Hay una infinidad de funciones para ser utilizadas como función de activación en el modelo propuesto, pero se pueden distinguir las siguientes clases: 

_F_ ( _x_ ) = θ( _x_ ) _Heaviside_ ( _escalon_ ) _F_ ( _x_ ) = 2θ( _x_ )−1 _Escalon bipolar_ 1 _F_ ( _x_ ) = 1+ _e_ − _x Sigmoide_ − − _x_ 1 _e F_ ( _x_ ) = 1+ _e_ − _x Sigmoide bipolar F x_ = _e_ − <u>(</u> _x_ 2−σµ2)2 _Gaussiana_ ( ) 

_<u>La elección de la función de activación depende fuertemente del algoritmo de</u>_ <u>aprendizaje que se vaya a utilizar. Cuando la función de</u> activación es de tipo escalón el elemento de proceso recibe el nombre de <u>.</u> _<u>perceptrón</u>_ 

Física Computacional II- Prof. E. Alvarez 



_Conceptos y definiciones sobre redes neuronales artificiales._ 

<u>.</u> Muchas veces se habla de _<u>arquitectura de una red neuronal</u>_ Este concepto se refiere básicamente a la manera en que se interconectan los distintos elementos de proceso que forman la red. Normalmente los elementos de proceso se organizan como una secuencia de _<u>capas</u>_ con un determinado patrón de interconexión entre los diferentes elementos de proceso que lo forman, y con un patrón de conexión entre los elementos de proceso de las distintas capas. 

_<u>Uno de los rasgos que puede ayudar a definir una capa</u>_ es el hecho de que todos los elementos de proceso que la forman usan la misma función de transferencia. 

Física Computacional II- Prof. E. Alvarez 



#### _Conceptos y definiciones sobre redes neuronales artificiales._ 

En muchas de las arquitecturas de redes neuronales se puede hacer la siguiente distinción entre las capas: 

- _<u>Capa de entrada</u>_ <u>: Es la capa que recibe los estímulos del entorno. No suele</u> tener asociado un mecanismo de aprendizaje, es decir, sus pesos se mantienen constantes, y su misión simplemente es la de distribuir dicha entrada al resto de los elementos de proceso que constituyen la red. 

- _<u>Capa de salida</u>_ <u>: Es la capa sobre la que se forman las salidas de la red.</u> 

- _<u>Capas ocultas</u>_ <u>: Son las demás capas que no son ni de entrada ni de salida.</u> 

Física Computacional II- Prof. E. Alvarez 



#### _Conceptos y definiciones sobre redes neuronales artificiales._ 

Una de las propiedades fundamentales de las redes neuronales es la capacidad de adaptarse al entorno, aprendiendo a proporcionar la respuesta se adecuada ante los estímulos que reciba de este entorno. Este <u>aprendizaje</u> plasma en la <u>modificación de los pesos</u> de las conexiones entre los distintos elementos que forman la red, de tal forma que memoriza los ejemplos de entrenamiento que le presentan. 

Existen muchos tipos de aprendizaje dependiendo del modo en que es realizado el ajuste de los pesos. En principio, los pesos pueden ser considerados parámetros libres, aunque es posible, si se conoce información acerca de la naturaleza del problema que se va a tratar, fijar restricciones a los valores iniciales de los pesos, o a los valores que puedan tomar a lo largo del aprendizaje. 

Física Computacional II- Prof. E. Alvarez 



#### _Conceptos y definiciones sobre redes neuronales artificiales._ 

_Un conjunto de reglas bien definidas que describen el método de adaptación o modificación de los pesos de acuerdo con el entorno en el que se encuentra sumergida la red recibe el nombre de_ **_<u>regla de aprendizaje</u>_** _<u>, y su</u> transcripción en forma de procedimiento se_ _<u>.</u> denomina_ **_<u>algoritmo de aprendizaje</u>_** 

Existe una _<u>relación</u>_ muy fuerte _<u>entre la arquitectura de una red</u>_ neuronal artificial _<u>y el o los algoritmos de aprendizaje</u>_ que puede usar, de tal modo que diferentes arquitecturas de redes neuronales requieren diferentes algoritmos de aprendizaje. 

Física Computacional II- Prof. E. Alvarez 



## _Paradigmas de aprendizaje_ 

Se denomina _<u>paradigma de aprendizaje</u>_ al modelo del entorno en el que la red trabaja. El paradigma de aprendizaje indica la forma en que el entorno influye en el proceso de aprendizaje. Así, el paradigma de aprendizaje puede ser: 

- _<u>Supervisado</u>_ <u>: Se presentan los conocimientos en forma de pares de</u> [entrada, salida deseada]. Si la entrada es distinta de la salida, se tendrá una red _<u>heteroasociativa</u>_ <u>; si la entrada es igual que la salida, se tratará</u> de una red _<u>autoasociativa</u>_ <u>.</u> 

- _<u>No Supervisado</u>_ <u>: Durante este proceso de aprendizaje a la red no se le</u> presenta la salida deseada. 

Física Computacional II- Prof. E. Alvarez 



## _Paradigmas de aprendizaje_ 

- _<u>Por refuerzo (entre los dos anteriores)</u>_ : El instructor o maestro exterior sólo indica cuando la salida es correcta o no, pero no indica en cuanto se diferencia de la salida buscada. Si se compara este paradigma con el supervisado, se observa que si bien el supervisado proporciona una información relativa a la dirección en la que se deben realizar los cambios en el sistema (ajuste de los pesos), en el caso de un aprendizaje por refuerzo no se tiene información acerca de la “dirección” del cambio, lo cual hace que su ámbito de aplicación sea mucho más reducido comparado con el modo supervisado, aunque presenta interés en la comunidad científica dedicada al estudio de las máquinas capaces de aprender. 

- _<u>Aprendizaje híbrido</u>_ <u>: Se trata de una combinación del aprendizaje supervisado</u> y del no supervisado. Parte de los pesos se ajustan por medio de un esquema de aprendizaje supervisado, y el resto se obtienen por medio de un aprendizaje no supervisado. 

Física Computacional II- Prof. E. Alvarez 



## _Aspectos importantes_ 

La teoría del aprendizaje mediante ejemplos conlleva tres aspectos muy importantes a tener en cuenta: determinar la capacidad de aprendizaje, la complejidad de los ejemplos utilizados y la complejidad computacional del proceso en sí: 

- _<u>La capacidad</u>_ <u>: Es un concepto relacionado con la cantidad de patrones que</u> pueden ser almacenados y qué funciones y contornos de decisión puede sintetizar una red neuronal artificial. 

- _<u>La complejidad de los ejemplos</u>_ : Determina el número de los patrones de aprendizaje necesitados para entrenar la red de tal manera que quede garantizado un determinado grado de generalización. 

- _<u>La complejidad computacional</u>_ : Se refiere al tiempo requerido para que el algoritmo de aprendizaje se aproxime a la solución usando los patrones de entrenamiento. Lo más corriente es que los algoritmos de aprendizaje sean computacionalmente muy complejos. 

Física Computacional II- Prof. E. Alvarez 



#### _Propiedades de las redes neuronales artificiales_ 

Las redes neuronales artificiales constituyen poderosos dispositivos de computo en gran parte por las siguientes propiedades : 

- Están constituidas por un conjunto de unidades de procesamiento bien sencillas que se comunican a traves de una vasta red de interconexiones con pesos sinápticos adaptables (variables). 

- Las memorias (información) se almacenan o representan en el patrón de pesos de las interconexiones de las U.P.s. El procesamiento de la información ocurre en base a la variación del patrón de actividades de las U.P.s. 

- Las redes neuronales no se programan, mas bien aprenden o se entrenan en la tarea a computar. 

Física Computacional II- Prof. E. Alvarez 



#### _Propiedades de las redes neuronales artificiales_ 

� Una RNA actúa naturalmente como una memoria asociativa. Es decir, inherentemente asocia entre si los patrones con los cuales se entrena, agrupando físicamente en su estructura patrones similares. Una RNA que opere como memoria asociativa accesa la información por contenido, en tal sentido es capaz de recuperar información a partir de estímulos incompletos, ruidosos o parcialmente erróneos(estímulos contaminados). 

- Una RNA tiene la capacidad de generalización, es decir es capaz de aprenderse las características de una categoría general de patrones basándose en una serie de ejemplos específicos de la categoría. 

- Son altamente tolerantes a las fallas de sus componentes. 

- Pueden poseer la propiedad de auto-organización, es decir algunas RNA son capaces de organizar y adaptar sus conexiones sinapticas sin necesidad de que se les provea durante el entrenamiento de instrucciones de que aprender. 

Física Computacional II- Prof. E. Alvarez 



### _Modelos iniciales de computación neuronal_ 

La creciente disponibilidad de poder de computo barato ha permitido la popularización del estudio de simulaciones de algunos modelos de computación neuronal en particular en lo que respecta a la capacidad de ciertas reglas de aprendizaje, así como la eficacia del computo. 

En lo que sigue repasaremos algunos modelos iniciales y variantes: 

- Perceptron (Rosenblatt 1958) 

- Adaline (Widrow 1962) 

- Matriz Memoria Asociativa (Willshaw 1969) 

Física Computacional II- Prof. E. Alvarez 

