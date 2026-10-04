# 08 --- Red de Hopfield y Reconocimiento de Imagenes

> Modulo: [`ce_rna/hopfield.py`](../ce_rna/hopfield.py)  
> Experimento: [10 (reconocimiento y tolerancia a ruido)](../experimentos/exp10_hopfield_imagenes.py)  
> Informe: [10](../resultados/reportes/10_hopfield_imagenes.md)  
> Fuentes: `clases/claseRN02.md`, Hopfield (1982) y Personnaz et al. (1985).

---

## 1. El problema que resuelve

En la computacion convencional basada en la arquitectura de Von Neumann, la informacion se almacena y recupera
accediendo a direcciones de memoria fijas. Si la direccion se desconoce o los datos sufren corrupcion, la recuperacion falla.

Por el contrario, la **memoria asociativa humana** es **direccionable por contenido**: somos capaces de recordar un
rostro completo a partir de una silueta difusa, o reconocer una palabra a partir de letras incompletas.

Tal como formula `claseRN02.md`:

> *"Las redes con supervision tales como la red de Hopfield y el perceptron son usadas como memorias asociativas
> o como clasificadores... son capaces de recuperar informacion a partir de estimulos incompletos, ruidosos o
> parcialmente erroneos."*

La **Red de Hopfield** es un modelo recurrente no lineal que resuelve este problema almacenando patrones como
**atractores estables** (minimos locales) de un paisaje energetico.

---

## 2. Arquitectura de la red

La red consta de $N$ neuronas con salidas bipolares $s_i \in \{-1, +1\}$ completamente interconectadas entre si:

```
    [s1] <====== W12 ======> [s2]
      ^                        ^
      ||  \                /  ||
      ||    \            /    ||
     W13     W14      W23    W24
      ||        \    /        ||
      v           \/          v
    [s3] <====== W34 ======> [s4]
```

Condiciones estructurales:
* **Simetria en las conexiones:** $W_{ij} = W_{ji}$ para todo $i, j$.
* **Ausencia de auto-conexiones:** $W_{ii} = 0$ (las neuronas no tienen bucles sobre si mismas).

---

## 3. Almacenamiento de Patrones

Para un conjunto de $P$ patrones $\xi^\mu \in \{-1, +1\}^N$, $\mu = 1, \dots, P$:

### A. Regla Hebbiana Clasica

$$W_{ij} = \frac{1}{N} \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu \quad (i \ne j), \qquad W_{ii} = 0$$

- **Ventaja:** Calculo puramente local en una sola pasada.
- **Limitacion:** Cuando los patrones presentan correlacion mutua (como letras del abecedario en retinas pequenas),
  los productos cruzados inducen **diafonia (*cross-talk*)**, desestabilizando los atractores.

### B. Regla de Proyeccion (Pseudoinversa)

$$W = X^T (X X^T)^{-1} X \quad \text{con } W_{ii} = 0$$

- **Ventaja:** Ortogonaliza el subespacio de memorias, garantizando que todos los patrones almacenados sean puntos fijos
  exactos (0 errores de recuperacion con entrada limpia), incluso cuando existe solapamiento tipografico significativo.

---

## 4. Dinamica de Recuperacion y Funcion de Energia

Dado un estado inicial $s(0)$ (imagen ruidosa):

1. **Actualizacion asincrona:** Se selecciona una neurona $i$ al azar y se actualiza:
   $$s_i(t+1) = \operatorname{sgn}\left( \sum_{j=1}^N W_{ij} s_j(t) \right)$$
2. **Funcion de Energia de Lyapunov:**
   $$E(\mathbf{s}) = -\frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N W_{ij} s_i s_j = -\frac{1}{2} \mathbf{s}^T W \mathbf{s}$$
3. **Teorema de convergencia:** En cada paso asincrono, el cambio de energia satisface:
   $$\Delta E = - \Delta s_i \left( \sum_{j=1}^N W_{ij} s_j \right) \le 0$$
   Como $E$ es monotona decreciente y el numero de estados $2^N$ es finito, la red converge inexorablemente a un minimo local.

---

## 5. Procesamiento de Imagenes Reales (Formato PNG)

El modulo [`ce_rna/hopfield.py`](../ce_rna/hopfield.py) implementa compatibilidad nativa con imagenes del sistema de archivos:
- **`guardar_imagen_png(patron, ruta, forma=(7, 6))`**: Guarda la matriz bipolar como archivo PNG real con escalado nitido.
- **`cargar_imagen_png(ruta, forma=(7, 6), umbral=128)`**: Abre cualquier imagen externa (PNG/JPG), la convierte a escala de grises, la redimensiona y la binariza en $\{-1, +1\}$.

---

## 6. Uso como libreria

```python
import numpy as np
from ce_rna import datasets as ds
from ce_rna.hopfield import RedHopfield, contaminar_patron, guardar_imagen_png

# 1. Cargar las 4 letras de 7x6 (42 pixeles)
conjunto = ds.letras_hopfield_7x6()

# 2. Instanciar y almacenar en la red de Hopfield
red = RedHopfield(n_neuronas=42, regla="proyeccion")
red.almacenar(conjunto.X, nombres=["A", "B", "C", "D"])

# 3. Contaminar la letra A con 6 pixeles de ruido (14.3%)
patron_a = ds.letra_7x6("A")
patron_ruidoso = contaminar_patron(patron_a, n_pixeles=6, semilla=42)

# 4. Recuperar el atractor estable mediante dinamica asincrona
recuperado, energia, conv = red.recuperar(patron_ruidoso, max_pasos=25, modo="asincrono")

print("Recuperacion exacta:", np.array_equal(recuperado, patron_a)) # True
print("Energia inicial:", energia[0], "-> Energia final:", energia[-1])

# 5. Exportar a imagen PNG fisica
guardar_imagen_png(recuperado, "letra_A_restaurada.png")
```
