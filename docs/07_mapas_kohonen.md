# 07 --- Mapas Autoorganizados de Kohonen (SOM)

> Modulo: [`ce_rna/kohonen.py`](../ce_rna/kohonen.py)  
> Experimento: [09 (clustering en el plano)](../experimentos/exp09_kohonen_clustering.py)  
> Informe: [09](../resultados/reportes/09_kohonen_clustering.md)  
> Fuentes: `clases/claseRN02.md` y Kohonen (1982).

---

## 1. El problema que resuelve

Hasta ahora, todos los modelos estudiados (McCulloch-Pitts, Perceptron, ADALINE y MLP) pertenecen a la categoria
del **aprendizaje supervisado**: requieren que cada vector de entrada este acompanado por una salida deseada $d$.

En el mundo real, la gran mayoria de los datos no estan etiquetados. Tal como introduce `claseRN02.md`:

> *"Las redes entrenadas sin supervision, tales como los mapas de Kohonen, son usados como vectores para formar cluster.
> Ninguna informacion relacionada a la clase correcta es dada a la red durante el entrenamiento."*

El **Mapa Autoorganizado de Kohonen (SOM)** resuelve dos problemas simultaneos:
1. **Cuantizacion vectorial (Agrupamiento / Clustering):** aproxima la densidad de probabilidad continua de los datos mediante un conjunto discreto de vectores prototipo $w_j \in \mathbb{R}^D$.
2. **Reduccion de dimensionalidad preservando topologia:** proyecta el espacio continuo multidimensional sobre una reticula discreta de baja dimension (usualmente 2D), de modo que patrones cercanos en el espacio de entrada se activen en neuronas vecinas en la reticula.

---

## 2. Arquitectura del mapa

El modelo consta de dos espacios coordinados:
1. **Espacio de entrada continuo ($\mathbb{R}^D$):** donde residen los datos $x = (x_1, \dots, x_D)$.
2. **Reticula discreta 2D:** una malla rectangular de dimensiones $K_x \times K_y$. Cada neurona $j$ tiene:
   - Una coordenada topologica fija $r_j = (\text{fila}, \text{columna})$ en la malla.
   - Un vector de pesos sinapticos ajustable $w_j \in \mathbb{R}^D$.

---

## 3. Dinamica del algoritmo de Kohonen

El entrenamiento transcurre a lo largo de $T$ epocas mediante tres fases biologicas:

### A. Fase de Competicion (BMU)
Para cada patron $x$, todas las neuronas compiten calculando su distancia euclidea:

$$j^* = \arg\min_j \|x - w_j\| = \arg\min_j \sqrt{\sum_{k=1}^D (x_k - w_{jk})^2}$$

La neurona $j^*$ con menor distancia se denomina **Unidad Mejor Coincidente** (*Best Matching Unit* o **BMU**).

### B. Fase de Cooperacion (Vecindad topologica)
La neurona ganadora influye sobre las neuronas vecinas en la reticula a traves de una funcion de vecindad gaussiana centrada en $j^*$:

$$h_{j, j^*}(t) = \exp\left( -\frac{\|r_j - r_{j^*}\|^2}{2\sigma(t)^2} \right)$$

donde $\|r_j - r_{j^*}\|$ es la distancia euclidea en coordenadas discretas de la reticula. El radio de vecindad $\sigma(t)$ decrece en el tiempo:

$$\sigma(t) = \sigma_0 \left(\frac{\sigma_f}{\sigma_0}\right)^{t / T}$$

Al inicio, $\sigma_0 \approx \max(K_x, K_y) / 2$ para organizar globalmente el mapa; hacia el final, $\sigma_f \approx 0.5$ para afinar unicamente la neurona ganadora.

### C. Fase de Adaptacion
Los vectores de pesos sinapticos se actualizan desplazandose hacia la entrada $x$:

$$w_j(t+1) = w_j(t) + \eta(t) \cdot h_{j, j^*}(t) \cdot (x - w_j(t))$$

con la razon de aprendizaje $\eta(t)$ decreciendo exponencialmente:

$$\eta(t) = \eta_0 \left(\frac{\eta_f}{\eta_0}\right)^{t / T}$$

---

## 4. Metricas y Diagnostico de Calidad

1. **Error de Cuantizacion ($QE$):** Distancia euclidea promedio entre cada dato y el prototipo de su BMU:
   $$QE = \frac{1}{N} \sum_{i=1}^N \|x_i - w_{j^*(x_i)}\|$$
   Mide la resolucion del mapa y que tan bien representan los prototipos a los datos.

2. **Error Topografico ($TE$):** Fraccion de datos para los cuales el primer y segundo BMU no son vecinos contiguos en la reticula:
   $$TE = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{dist\_malla}(j_1^*, j_2^*) > 1)$$
   Mide la preservacion de la topologia; un $TE \approx 0$ indica que la malla no esta torcida ni plegada.

3. **Matriz U (Unified Distance Matrix):**
   Para cada neurona $(r, c)$, calcula la distancia media a sus vecinos contiguos en la reticula:
   $$U(r, c) = \frac{1}{|V(r, c)|} \sum_{k \in V(r, c)} \|w_{(r, c)} - w_k\|$$
   - **Valles oscuros (baja distancia):** indican zonas de alta densidad de datos (centros de clusters).
   - **Crestas brillantes (alta distancia):** indican separacion o discontinuidades entre clusters.

---

## 5. Uso como libreria

```python
import numpy as np
from ce_rna import datasets as ds
from ce_rna.kohonen import MapaAutoorganizado

# 1. Generar distribucion de clusters en 2D
conjunto = ds.distribucion_clusters_2d(n_por_cluster=75, dispersion=0.38, semilla=42)

# 2. Instanciar y entrenar el SOM (aprendizaje no supervisado)
som = MapaAutoorganizado(
    forma_mapa=(6, 6),
    n_entradas=2,
    razon_aprendizaje_inicial=0.5,
    razon_aprendizaje_final=0.02,
    radio_inicial=3.0,
    radio_final=0.5,
    max_epocas=200,
    semilla=42,
)
som.entrenar(conjunto.X)

# 3. Evaluar calidad
print("Error de cuantizacion final:", som.error_cuantizacion(conjunto.X))
print("Error topografico final:   ", som.error_topografico(conjunto.X))

# 4. Obtener coordenadas BMU de nuevos patrones
nuevos_puntos = np.array([[2.1, 1.9], [-1.8, -2.2]])
bmus = som.propagar(nuevos_puntos)
print("Coordenadas en reticula:", bmus)
```
