# **Aspectos de Computación Neuronal** 

**_Problemas Linealmente Separables y Capas Neuronales_** 

**_Prof. Esteban Álvarez_** 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

### _Problemas Linealmente Separables_ 

A continuación se presenta una red simple, como la del And u Or, pero más general: 



_Figura:Función Lógica “Simple”_ 

En ella se ha añadido una _<u>neurona de inclinación</u>_ , en vez de un umbral. 

_y_<sup>(</sup><sup>_in_)</sup> _=w0 +w1 x_ 1 _+w2 x_ 2 

Es conocido que y<sup>(in)</sup> estará dada por: 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

### _Problemas Linealmente Separables_ 

y la respuesta, por: 



<u>Esto divide al plano formado por x</u> ~~<u>1</u>~~ <u>y x</u> ~~<u>2</u>~~ <u>en dos regiones: en una,</u> se tendrá que y=0 e y<sup>(in)</sup> <0, en la otra se tendrá que y=1 e y<sup>(in)</sup>  0. La <u>frontera</u> entre ambas está dada por la ecuación <u>lineal de la recta:</u> 

_<mark>w</mark>_ 0 _<mark>+w</mark> 1_ _<mark>x</mark>_ 1 _<mark>+w</mark> 2_ _<mark>x</mark>_ 2 <mark>= 0</mark> 

Veamos por ejemplo que ocurre con la función And. Tenemos las que y<sup>(in)</sup> =x1+x2-2, la frontera es x1+x2=2. Si superponemos respuestas que nos debe arrojar la red con el gráfico de las regiones, obtenemos la siguiente figura: 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

### _Problemas Linealmente Separables_ 



_Figura: Función And sobre el plano_ 

Si la entrada está en la región “Clase1” producirá una salida 1, si está en la región “Clase <u>0”, una salida 0.</u> Vemos que se pueden separar las entradas que deben producir una salida 1 de las que deben producir un salida 0 <u>por una línea recta.</u> _<u>Se dice entonces que el problema es linealmente separable.</u>_ 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

#### _- Problemas No Linealmente Separables_ 

Revisemos en cambio, como es la función Xor sobre el plano: 

Figura: Xor sobre el plano 



<!-- Start of picture text -->
1-1 0<br>0-ibs<br>0<br><!-- End of picture text -->

Mediante simple inspección podemos comprobar que efectivamente es imposible encontrar una línea recta que deje a un lado las entradas que deben producir 0, y al otro, las que deben producir 1. _<u>En este caso, decimos que el problema no es linealmente separable</u>_ . Por eso no nos basta con una red “sencilla” para resolver el Xor. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Separabilidad Lineal_ 

Lo que en realidad estamos haciendo es un caso muy sencillo del problema general de clasificación de patrones. Estamos clasificando las entradas en “Clase 1” o “Clase Verdadera” y “Clase 0” o “Clase Falsa”. 

_<u>El concepto de separabilidad</u>_ lineal se extiende de modo natural a entradas de más dimensiones. Las entradas que pertenecen a una clase y las que no pertenecen a esta simplemente tienen que poder separarse por el _<u>hiperplano</u> n_ en el espacio x de las entradas. ∑ _w ji xi_ = 0 _i=_ 0 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Separabilidad Lineal_ 



<!-- Start of picture text -->
Linealmente Separables: No Linealmente<br>Separables:<br>n<br>fii<br>pe wx, =0<br>i=0<br>¢<br>+<br>+<br>a<br>¢<br>¢<br>¢<br>¢<br>¢<br>¢<br>¢<br>¢of<br>iS<br>4<br>+<br>4<br>ca<br>o<br>¢<br>+<br>+<br>¢<br>Re Espaciox<br><!-- End of picture text -->

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Capas Neurales_ 

Cuando trabajamos con grandes cantidades de neuronas, es natural ordenar aquellas que tienen comportamientos similares en “ _<u>capas</u>_ ”, como se muestra en la figura: _<u>Cada capa es un vector de neuronas</u>_ . ~~—EE~~ _Figura : Red Unicapa_ 



<!-- Start of picture text -->
Wig<br>Wino)<br>Wy<br>P] Wit / P]<br>rT] \ r<br>S rT Wi; /\ g<br>EE §<br>oi ”n<br>rT] rT]<br>77 Wni LS .|<br>Vim \<br>Capa 0 Vv™fs Capa 1<br><!-- End of picture text -->

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Capas Neurales_ 

Se acostumbra no contabilizar la capa de entrada, por lo tanto se dice que la red mostrada anteriormente es “ _<u>Unicapa</u>_ ”. Las sinapsis obviamente están ordenadas en una matriz wji de n*(m+1). De nuestro análisis anterior, tenemos que una red unicapa sólo puede resolver problemas linealmente separables. En una red unicapa las neuronas de salida pueden ser lineales o no lineales. Pero es evidente que podemos seguir añadiendo capas como se muestra en la figura. 

_Figura: Red Multicapa_ 



<!-- Start of picture text -->
aah eet [ye “KY 3<br><= YE " ‘.<br><!-- End of picture text -->

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa: Paralelismo_ 

En una red multicapa, las <u>capas ocultas,</u> que en la figura anterior corresponde a la capa 2, <u>siempre son no lineales.</u> Se puede demostrar que si se construye una red multicapa con capas ocultas lineales, ésta es equivalente a una red unicapa. 

Podemos ver fácilmente _<u>la idea de paralelismo</u>_ al observar las capas de las redes. Cada neurona de una capa no necesita de las demás en su misma capa para trabajar, son capaces por lo tanto de trabajar simultáneamente. Esta cualidad se ha aprovechado al diseñar chips paralelos con la nueva tecnología <u>VLSI (Very Large Scale Integrated),</u> en donde se han implementado varios tipos de neuroredes. Una red multicapa es capaz de resolver problemas más complejos, pero su proceso de aprendizaje también es más complicado. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa_ 

El problema de separabilidad lineal o representatividad gráfica para problemas más complejos puede ser superado sumando más capas a la red(1960). Por ejemplo, las redes multicapa pueden llevar a cabo clasificaciones más generales, separando esos puntos o patrones de entrada en regiones convexas abiertas o cerradas. 

“Una _<u>región convexa</u>_ es una en la cual dos puntos arbitrarios pertenecientes a la región pueden ser unidos a partir de una línea recta que no abandona la región. Una región <u>cerrada es una región en la cual</u> todos los puntos están contenidos dentro de una frontera (ej. Un circulo). Una <u>región abierta</u> tiene algunos puntos que están fuera del contorno definido (ej. La región entre dos líneas paralelas ) ”. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa_ 

Para entender el problema de la convexidad, consideremos una red simple de dos capas con dos entradas (cada una dirigida a cada neurona de la primera capa), ambas alimentando una neurona en la segunda capa. 



<!-- Start of picture text -->
y<br>Capa 1<br>S1 w11 Capa 2<br>S1<br>w12<br>x<br>w21<br>S2<br>w22<br>S2<br><!-- End of picture text -->

_Figura: Región de decisión Convexa generada por un perceptrón de 2 capas_ 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa_ 

cada neurona en la capa 1 subdivide el plano x-y, una arrojando un valor de “1” para las entradas bajo la línea superior y la otra produciendo una salida de “1” para las entradas por encima de la línea inferior. La salida de la neurona de la capa 2 es uno “1” solo en la región en forma ce “V”. 

Similarmente, tres neuronas pueden ser usadas en la capa de entrada para subdividir el plano, generando una región en forma triangular. Finalmente incluyendo suficientes neuronas en la capa de entrada, <u>un polígono convexo de alguna forma deseada se puede</u> . <u>generar</u> 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa_ 

Las entradas no necesariamente tienen que ser binarias. Un vector de entradas continuas puede representar un punto en algún sitio del plano x-y. En este caso la red tiene la habilidad para subdividir el plano en regiones continuas mas que conjuntos de puntos discretos separados. 

Entonces, para separar regiones P y Q, todos los puntos en P deben estar dentro de un polígono convexo que no contenga los puntos de Q (o al contrario). 

Una red de tres capas es aún más general, su capacidad de clasificación esta limitada solo por el número de neuronas artificiales y los pesos. No hay convexibilidad obligada, las neuronas en la capa 3 ahora reciben como entrada un grupo de polígonos convexos, y la combinación lógica de estos no necesariamente es convexa. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## _Red Multicapa_ 

A continuación observamos un caso en el cual dos triángulos A y B son combinados por la función “A and not B”, arrojando una región no-convexa. 



<!-- Start of picture text -->
Capa 1 Capa 2<br>X1 Capa 3<br>X2<br>X3<br><!-- End of picture text -->

No todas las regiones de la capa 2 se interceptan, lo cual hace posible crear múltiples regiones, convexas y no convexas, produciendo una salida de “1” siempre que el vector de entrada este en alguna de ellas. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## **Aprendizaje o Entrenamiento** 

Es esencialmente el proceso en el que se adaptan las sinapsis, para que la red responda de un modo distinto a los estímulos del medio. En una Neurored, toda la información adquirida se guarda en el valor de cada peso sináptico. De hecho, las neuronas de la mayor parte de los seres vivos con sistema nervioso, desde un caracol hasta el hombre son esencialmente iguales. Lo que nos hace más inteligentes que un caracol es el número, organización y modo de cambio de las conexiones sinápticas. 

En la clase anterior citamos cuatro clase de entrenamiento entre las que destacan <u>el aprendizaje con Profesor o Supervisado</u> y <u>sin profesor o No Supervisado.</u> 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

##### **Aprendizaje Supervisado:Entrenamiento** 

El proceso es completamente análogo a enseñarle algo a un niño, digamos por ejemplo, a reconocer las vocales. Los pasos del proceso son los siguientes: 

1. El profesor dispone de un conjunto de N pares de entrenamiento (xi (n); dj(n) ) en donde xi(n) es la n-ésima entrada y dj(n) es la respuesta correcta a esa entrada. En nuestro ejemplo, significa que tenemos todas las vocales dibujadas en un papel ( xi (n) ) y que nosotros sabemos las respuestas correctas ( dj(n) ) a cada una de las figuras, los sonidos A,E,I,O,U. 

2. Introducimos una de las entradas xi (n) y esperamos que nuestra red nos responda. Sería como mostrarle al niño la letra A y preguntarle: “Dime, ¿ Qué letra es esta?” 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

##### **Aprendizaje Supervisado:Entrenamiento** 

3. La neurored responde mediante una salida Oj(n). Digamos, el niño nos respondió “Esa es una E”. 

4. Luego comparamos ambas señales, la respuesta deseada dj(n) y la respuesta de la red Oj(n), creando una señal de error, ej(n)= dj(n)Oj(n). “Mmm... El niño no está tan despierto como esperaba... ”. 

5. Luego, con la señal de error ej(n), corrijo las sinapsis de la red mediante algún algoritmo de los que se verá a continuación. “No hijo, esta no es una E, es una A...”. 

En general, pueden haber muchas secuencias completas de los N pares de entrenamiento, y el aprendizaje se detiene cuando la red responda correctamente a todos los pares de entrenamiento. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

##### **Aprendizaje Supervisado:Entrenamiento** 

En general, cuando adaptemos las sinapsis, la forma de hacerlo será mediante la siguiente ecuación: 

_<mark>w</mark> ji_ <mark>(</mark> _<mark>n+</mark>_ <mark>1)</mark> _<mark>=w</mark> ji_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> _<mark>+ew</mark> ji_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> 

en donde wji(n) son los pesos sinápticos con los que la red responderá al n-ésimo ejemplo. Esto equivale a no cambiar los pesos sinápticos en forma radical, sino que simplemente los variamos en una cantidad “pequeña” e*wji(n) con respecto a su estado anterior. Lo que diferencia a los algoritmos o reglas de aprendizaje, es básicamente como encontrar <u>e*w (n)</u> . El que hayan distintos algoritmos tiene cierta base <u>j</u> ~~<u>i</u>~~ biológica. Neuronas de distintas partes del cerebro aprenden de forma distinta también. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Regla de Hebb 

Esta es la más antigua y la más famosa de las reglas de aprendizaje, su base es completamente biológica. Fue encontrada por el neurofisiologo Hebb en 1949, quien descubrió que si dos neuronas a ambos lados de la sinapsis estaban activas (o inactivas) simultáneamente, entonces las sinapsis entre ellas se reforzaban, y si se activaban (o desactivaban) asincrónicamente, se debilitaban. Una forma de expresar esta idea de forma sencilla es la siguiente: 

_<mark>ew</mark> ji_ _<mark>=a y</mark> j_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> _<mark>x</mark> i_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> _<mark>a></mark>_ <mark>0</mark> 

donde las capas de neuronas xi y yj están distribuidas como se mostró anteriormente. A la constante de proporcionalidad “a” se le llama “ _<u>razón de aprendizaje</u>_ ”. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Regla de Hebb 

Para ver como funciona, supongamos que xi e yi son bipolares o antisimétricas, con a=1. Si xi e yi toman ambas simultaneamente el valor de 1 (o de -1), ewji(n) = a, y esa sinapsis se reforzará. En cambio, si una tomase el valor –1 y la otra el de 1, ewji(n) = -a, y esa sinapsis se debilitará. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Aprendizaje para redes unicapa _<u>Regla de aprendizaje perceptrónico.</u>_ <u>Objetivo y funcionamiento general: Esta regla de aprendizaje está</u> diseñada especialmente para el _reconocimiento de patrones_ , pero por ser una red unicapa, sólo se pueden usar patrones linealmente separables. 

**<u>El perceptrón</u>** nació como un primer intento de modelar la retina, en 1958, por Rosenblatt. El perceptrón es una red de una sola capa (ver figura de red unicapa). Las neuronas de salida no son lineales, con función de activación tipo escalón. A continuación utilizaremos funciones de activación bipolares o antisimétricas, como la siguiente: 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## El perceptrón 



Nótese que incluyó un punto neutro. A este punto se le suele llamar punto de indeterminación. Simplemente dice que la neurona no sabe que responder. 

Cada neurona de salida representa a una clase determinada, si una de ellas se activa con una entrada, significa que pertenece a esa clase, si está desactivada, que no pertenece. 

Veamos el siguiente ejemplo que nos permite clasificar imágenes de letras: 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## El perceptrón 

Digamos por ejemplo que tenemos una red que nos clasifica una entrada como X u O. Lo que queremos es que funcione como se muestra en la figura, en donde la neurona marcada con X reconoce la clase X, y la con O, a la clase O: 



<!-- Start of picture text -->
—_<br>: =p |: “Esuna X”<br>—_ 3<br>: —» -1:“NoesunaO”<br>—<br>—_—<br>: —> -1: “No esuna X”<br>>:<br>:— |: “Esuna0O”<br>—_<br><!-- End of picture text -->

_Figura : Funcionamiento de un Perceptrón_ 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Algoritmo Perceptrónico 

Veamos ahora como entrenar esta red que cuenta m0 y m1 número de neuronas de entrada y salida respectivamente. Además, existen N pares de entrenamiento (xi (n); dj(n) ). De esta forma el algoritmo es: _<u>Paso 0:</u>_ Inicializar las sinapsis de la red, se puede elegir wji=0 ó valores aleatorios. Se elige una razón de aprendizaje a comprendida 0<a<1. _<u>Paso 1:</u>_ Mientras la condición de parada del paso 5 sea falsa, realizar los pasos del 2 al 5. 

_<u>Paso2:</u>_ Para cada par de entrenamiento, (xi (n); dj(n) ) ; n=1,...,N, hacer los pasos del 3 y 4. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Algoritmo Perceptrónico 

Paso 3: j=1,...,m1 (barrido sobre las neuronas de salida). 





<!-- Start of picture text -->
m 0<br>y (  jin )( n )= ∑ w ji  ( n ) xi ( n )<br>i=  0<br><!-- End of picture text -->

Paso 4: Si yj!=dj(n), para algún j entre 1 y m1 , entonces _<mark>w</mark> ji_ <mark>(</mark> _<mark>n+</mark>_ <mark>1)</mark> _<mark>=w</mark> ji_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> _<mark>+a d</mark> j_ _<mark>x</mark> i_ <mark>(</mark> _<mark>n</mark>_ <mark>)</mark> 

Donde j=1,...,m1 ; i=0,...,m0 . En caso contrario 

_wji_ <mark>(</mark> _n+_ 1 <mark>)</mark> _=w ji_ <mark>(</mark> _n_ <mark>)</mark> 

Paso 5: Si los pesos sinápticos no cambian para cada patrón de entrenamiento durante la última vez que se realizó el paso 2, entonces parar, sino es así, continuar. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Algoritmo Perceptrónico 

Es bastante evidente que si un problema es linealmente separable, existen infinitos pesos sinápticos que servirán para solucionar el _n_ problema. Basta con multiplicar por una constante la ecuación ∑ _w ji xi_ = 0 _i=_ 0 y seguimos teniendo el mismo hiperplano de separación, aunque distintos pesos sinápticos. Además, generalmente, no es un solo hiperplano el que nos podría delimitar bien la frontera, sino que más bien hay infinitos, como se muestra en la figura siguiente: o sea , o no existe ninguna solución , o existen infinitas. 

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

## Algoritmo Perceptrónico: Infinitas Soluciones 



<!-- Start of picture text -->
r<br>.<br>.<br>.<br>ad . .<br>on ° ;<br>¢ o*<br>a ae<br>¢ ° at a= .<br>. 3° 6 ° .<br>a” gst ae<br>. oye B an<br>gt ee<br>ce ge?<br>aetins<br>eee<br>-+a<br>se eearoe% *<br>of Pale° ¢ .<br>a? .* Pa Pe.©<br>ote<br>Pe.<br>.. ° . .ae<br>.<br>¢<br>.<br>.<br><!-- End of picture text -->

Introducción a la Computación Emergente → Redes Neuronales Artificiales 

