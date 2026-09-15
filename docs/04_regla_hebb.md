# 04 --- La regla de Hebb

> Modulo: [`ce_rna/hebb.py`](../ce_rna/hebb.py)
> Experimento: [`experimentos/exp07_hebb.py`](../experimentos/exp07_hebb.py)
> Informe: [`resultados/reportes/07_regla_hebb.md`](../resultados/reportes/07_regla_hebb.md)
> Fuente: `clases/ICE-claseRN03.md`

---

## 1. El postulado de Hebb

Donald Hebb, neurofisiologo, formulo en 1949 la regla de aprendizaje mas antigua y
mas famosa, con base biologica directa:

> *"Si dos neuronas a ambos lados de la sinapsis estaban activas (o inactivas)
> simultaneamente, entonces las sinapsis entre ellas se reforzaban, y si se activaban
> (o desactivaban) asincronicamente, se debilitaban."*

Formalizado:

$$\Delta w_{ji} = a\, y_j(n)\, x_i(n), \qquad a > 0$$

Con `x_i` e `y_j` **bipolares** y `a = 1`, el comportamiento es exactamente el que
describe el postulado:

| `x_i` | `y_j` | `Δw_ji` | Efecto |
|---|---|---|---|
| +1 | +1 | +1 | se refuerza |
| −1 | −1 | +1 | se refuerza |
| +1 | −1 | −1 | se debilita |
| −1 | +1 | −1 | se debilita |

## 2. Version supervisada

En una red unicapa con maestro, se sustituye la salida producida `y_j` por la deseada
`d_j`:

$$w_{ji} = a \sum_{n=1}^{N} d_j(n)\, x_i(n)$$

Es decir, el peso final es la **correlacion acumulada** entre la entrada `i` y la
salida deseada `j`. En forma matricial, `W = a · Dᵀ X` --- una suma de productos
externos.

Propiedades que se derivan inmediatamente de esa formula:

1. **Aprendizaje de una sola pasada.** No hay epocas, ni comprobacion de errores, ni
   criterio de parada. Se presenta cada patron una vez y se ha terminado.
2. **`a` es irrelevante.** Escalar todos los pesos por una constante positiva no
   cambia la frontera `w·x = 0`. Verificado en
   `probar_hebb_independiente_de_la_razon`.
3. **El orden de presentacion no importa.** La suma es conmutativa, a diferencia del
   perceptron, donde el orden determina en que solucion se detiene.
4. **No hay realimentacion.** La regla nunca mira lo que la red responde. Puede fallar
   en silencio.

```python
for n in range(Xe.shape[0]):
    self.W += self.a * np.outer(d[n], Xe[n])     # una sola pasada, sin condiciones
```

---

## 3. El caso que falla: el AND en codificacion binaria

Este es el resultado mas instructivo del [experimento 07](../resultados/reportes/07_regla_hebb.md).

| patron | d | aporte a w1 (`d·x1`) | aporte a w2 (`d·x2`) | aporte a w0 (`d·1`) |
|---|---|---|---|---|
| (0,0) | −1 | 0 | 0 | −1 |
| (0,1) | −1 | 0 | −1 | −1 |
| (1,0) | −1 | −1 | 0 | −1 |
| (1,1) | +1 | +1 | +1 | +1 |
| **suma** | | **0** | **0** | **−2** |

Los pesos de entrada quedan **exactamente en cero**: la red responde `−1` ante
cualquier entrada, acertando 3 de 4 patrones por pura casualidad estadistica.

La causa es estructural: el incremento es proporcional a `x_i`, de modo que **todo
patron con `x_i = 0` es invisible para el peso `w_i`**. En el AND binario, los unicos
patrones con `x_i = 1` se reparten entre las dos clases de forma que sus aportes se
cancelan exactamente.

Con **codificacion bipolar** no hay entradas nulas: cada presentacion informa a todos
los pesos, con signo, y la regla resuelve AND y OR sin problema:

| codificacion | compuerta | w | exactitud |
|---|---|---|---|
| binaria | AND | (−2, 0, 0) | 75 % |
| binaria | OR | (2, 2, 2) | 75 % |
| bipolar | AND | (−2, 2, 2) | **100 %** |
| bipolar | OR | (2, 2, 2) | **100 %** |

Por eso la clase insiste en trabajar con valores *"bipolares o antisimetricos"* al
presentar la regla de Hebb. Es un detalle que parece cosmetico y decide entre
funcionar y no funcionar.

![Hebb AND bipolar](../resultados/figuras/07_hebb_and_bipolar.png)

---

## 4. Hebb como memoria asociativa

Con los prototipos de las letras X y O, la regla produce exactamente la **diferencia
de plantillas**:

$$w_{\text{neurona X}} = x^{(X)} - x^{(O)}, \qquad w_{\text{neurona O}} = x^{(O)} - x^{(X)}$$

![Pesos de Hebb sobre las letras](../resultados/figuras/07_hebb_letras_pesos.png)

El potencial `w·x` es entonces la correlacion de la entrada con esa diferencia, y la
red recupera la letra correcta a partir de retinas muy degradadas
([figura de tolerancia](../resultados/figuras/07_hebb_tolerancia.png)). Esto es una
**memoria asociativa** en el sentido de `claseRN01.md`:

> *"Una RNA que opere como memoria asociativa accede a la informacion por contenido;
> en tal sentido es capaz de recuperar informacion a partir de estimulos incompletos,
> ruidosos o parcialmente erroneos."*

El limite aparece cuando los patrones almacenados **no son ortogonales** entre si:
las correlaciones cruzadas se suman al peso y aparecen interferencias. La regla
perceptronica corrige esas interferencias por construccion --- porque comprueba sus
respuestas --- y la de Hebb no.

---

## 5. Comparacion de las tres reglas

| | Hebb | Perceptron | Delta (ADALINE) |
|---|---|---|---|
| `Δw_ji` | `a·d_j·x_i` | `a·d_j·x_i` si falla | `γ·(d_j − y_j)·x_i` |
| ¿Mira la salida producida? | **no** | si (umbralizada) | si (lineal) |
| Pasadas necesarias | 1 | hasta converger | hasta converger |
| Depende del orden | no | si | si (en modo estocastico) |
| Garantia | **ninguna** | converge si es separable | converge al minimo del ECM |
| Coste por patron | 1 producto externo | 1 comparacion + quiza 1 correccion | 1 correccion siempre |
| Falla en silencio | **si** | no | si (converge a solucion mala) |

La progresion historica --- Hebb (1949) → Rosenblatt (1958) → Widrow-Hoff (1960) ---
es exactamente la progresion de esta tabla: cada regla anade una forma de
realimentacion que la anterior no tenia.

---

## 6. API del modulo

```python
RedHebb(
    n_entradas,
    n_salidas=1,
    razon_aprendizaje=1.0,
    activacion=None,          # por omision, escalon bipolar
)
```

| Metodo | Devuelve |
|---|---|
| `.entrenar(X, d, verbose=False)` | el modelo (una sola pasada) |
| `.predecir(X)` | salidas bipolares |
| `.exactitud(X, d)` | fraccion de aciertos |
| `.recta_frontera(j)` | `(w0, w1, w2)` |
| `.resumen()` | pesos en texto |

### Ejemplo minimo

```python
from ce_rna.hebb import RedHebb
from ce_rna import datasets as ds

conjunto = ds.compuerta("AND", "bipolar")       # OJO: bipolar, no binaria
red = RedHebb(n_entradas=2).entrenar(conjunto.X, conjunto.d)

print(red.W)                                    # [[-2.  2.  2.]]
print(red.exactitud(conjunto.X, conjunto.d))    # 1.0
```
