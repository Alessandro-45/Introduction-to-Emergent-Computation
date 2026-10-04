"""
Bateria de pruebas del paquete `ce_rna`.

Se ejecuta sin dependencias externas (no hace falta `pytest`):

    python pruebas/pruebas.py

Cada prueba es una funcion `probar_*` que lanza `AssertionError` si algo no
cuadra.  El runner las descubre por nombre, las ejecuta todas e informa del
resultado.  Las pruebas verifican propiedades **matematicas** de los modelos, no
solo que el codigo no reviente:

* que las compuertas MCP reproduzcan sus tablas de verdad,
* que el perceptron converja en problemas separables y no lo haga en el XOR,
* que la regla Delta alcance la solucion de minimos cuadrados,
* que el gradiente implementado coincida con la derivada numerica del error,
* que las derivadas de las activaciones coincidan con sus diferencias finitas.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import numpy as np

from ce_rna import activaciones as act
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna.adaline import (Adaline, ecm_de_pesos, solucion_minimos_cuadrados)
from ce_rna.hebb import RedHebb
from ce_rna.hopfield import (RedHopfield, cargar_imagen_png, contaminar_patron,
                                guardar_imagen_png)
from ce_rna.kohonen import MapaAutoorganizado
from ce_rna.mcculloch_pitts import (NeuronaMCP, neurona_and, neurona_nand,
                                    neurona_nor, neurona_not, neurona_or,
                                    red_xor)
from ce_rna.mlp import PerceptronMulticapa
from ce_rna.perceptron import PerceptronSimple


# ---------------------------------------------------------------------------
# Activaciones
# ---------------------------------------------------------------------------

def probar_escalon():
    phi = act.Escalon(umbral=2.0)
    assert np.array_equal(phi([1.9, 2.0, 2.1]), [0.0, 1.0, 1.0])


def probar_escalon_bipolar_con_zona():
    phi = act.EscalonBipolarConZona(theta=0.5)
    assert np.array_equal(phi([-1.0, -0.5, 0.0, 0.5, 1.0]), [-1.0, 0.0, 0.0, 0.0, 1.0])


def probar_sigmoide_estable():
    """La sigmoide no debe desbordar para argumentos extremos."""
    valores = act.Sigmoide()([-1000.0, 0.0, 1000.0])
    assert np.all(np.isfinite(valores))
    assert np.isclose(valores[1], 0.5)


def probar_derivadas_numericas():
    """Cada activacion derivable debe coincidir con su diferencia finita central."""
    v = np.linspace(-3, 3, 41)
    h = 1e-6
    for phi in (act.Identidad(), act.Sigmoide(), act.SigmoideBipolar(), act.Gaussiana(0.5, 1.2)):
        numerica = (phi(v + h) - phi(v - h)) / (2 * h)
        assert np.allclose(phi.derivada(v), numerica, atol=1e-5), phi.nombre


# ---------------------------------------------------------------------------
# McCulloch-Pitts
# ---------------------------------------------------------------------------

def probar_compuertas_mcp():
    for nombre, neurona in (("AND", neurona_and(2)), ("OR", neurona_or(2)),
                            ("NAND", neurona_nand(2)), ("NOR", neurona_nor(2))):
        esperado = ds.compuerta(nombre).d
        obtenido = neurona.activar(ds.compuerta(nombre).X)
        assert np.array_equal(obtenido, esperado), nombre


def probar_not_mcp():
    assert np.array_equal(neurona_not().activar([[0.0], [1.0]]), [1.0, 0.0])


def probar_red_xor_mcp():
    esperado = ds.compuerta("XOR").d
    assert np.array_equal(red_xor().salida(ds.compuerta("XOR").X), esperado)


def probar_ninguna_mcp_resuelve_xor():
    """Ninguna neurona de umbral clasifica los 4 patrones del XOR."""
    X, d = ds.compuerta("XOR").X, ds.compuerta("XOR").d
    for w1 in np.linspace(-4, 4, 33):
        for w2 in np.linspace(-4, 4, 33):
            for theta in np.linspace(-4, 4, 33):
                if np.array_equal(NeuronaMCP([w1, w2], theta).activar(X), d):
                    raise AssertionError(f"encontrada solucion inesperada: {w1}, {w2}, {theta}")


def probar_red_mcp_rechaza_fuentes_invalidas():
    from ce_rna.mcculloch_pitts import RedMCP

    red = RedMCP(["a", "b"])
    try:
        red.agregar("y", neurona_and(2), ["a", "inexistente"])
    except KeyError:
        return
    raise AssertionError("se acepto una fuente no declarada")


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

def probar_perceptron_converge_en_and_or():
    for nombre in ("AND", "OR", "NAND", "NOR"):
        conjunto = ds.compuerta(nombre, "bipolar")
        modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
        modelo.entrenar(conjunto.X, conjunto.d)
        assert modelo.convergio, nombre
        assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) == 1.0, nombre


def probar_perceptron_no_converge_en_xor():
    conjunto = ds.compuerta("XOR", "bipolar")
    modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=300)
    modelo.entrenar(conjunto.X, conjunto.d)
    assert not modelo.convergio
    assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) < 1.0


def probar_perceptron_no_toca_pesos_si_acierta():
    """Paso 4: si y == d para todas las salidas, los pesos no cambian."""
    conjunto = ds.compuerta("AND", "bipolar")
    modelo = PerceptronSimple(2, 1, razon_aprendizaje=1.0, max_epocas=100)
    modelo.entrenar(conjunto.X, conjunto.d)
    W_convergido = modelo.W.copy()
    modelo.entrenar(conjunto.X, conjunto.d)   # segundo entrenamiento desde el estado final
    assert np.array_equal(modelo.W, W_convergido)
    assert modelo.epocas_usadas == 1


def probar_perceptron_multiclase():
    conjunto = ds.letras_xo(con_ruido=4, semilla=0)
    modelo = PerceptronSimple(25, 2, razon_aprendizaje=1.0, max_epocas=200)
    modelo.entrenar(conjunto.X, conjunto.d)
    assert modelo.convergio
    assert mt.exactitud(modelo.predecir(conjunto.X), conjunto.d) == 1.0


def probar_perceptron_valida_dimensiones():
    modelo = PerceptronSimple(2, 1)
    try:
        modelo.entrenar(np.zeros((4, 3)), np.zeros(4))
    except ValueError:
        return
    raise AssertionError("se acepto una matriz de entradas con dimension incorrecta")


def probar_margen_positivo_implica_clasificacion_correcta():
    conjunto = ds.compuerta("OR", "bipolar")
    modelo = PerceptronSimple(2, 1, max_epocas=100).entrenar(conjunto.X, conjunto.d)
    assert mt.margen_geometrico(modelo.W[0], conjunto.X, conjunto.d) > 0


# ---------------------------------------------------------------------------
# ADALINE
# ---------------------------------------------------------------------------

def probar_adaline_recupera_pesos_del_decodificador():
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=0.1, max_epocas=2000, tolerancia=1e-10,
                     inicializacion="aleatoria", semilla=0)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert np.allclose(modelo.w, ds.pesos_optimos_decodificador(3), atol=1e-5)
    assert modelo.ecm(conjunto.X, conjunto.d) < 1e-10


def probar_adaline_alcanza_minimos_cuadrados():
    """El descenso del gradiente converge al minimo global del paraboloide."""
    conjunto = ds.nubes_separables(n_por_clase=25, semilla=4)
    optimo = solucion_minimos_cuadrados(conjunto.X, conjunto.d)
    modelo = Adaline(2, razon_aprendizaje=0.05, modo="lote", max_epocas=20000,
                     tolerancia=1e-12, inicializacion="aleatoria", semilla=1)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert np.allclose(modelo.w, optimo, atol=1e-3)


def probar_gradiente_coincide_con_derivada_numerica():
    """El gradiente analitico -(X^T e)/N debe igualar la derivada numerica de E."""
    conjunto = ds.decodificador_binario(3)
    X, d = conjunto.X, conjunto.d
    w = np.array([0.3, -0.7, 1.4, 0.2])
    Xe = np.hstack([np.ones((X.shape[0], 1)), X])
    analitico = -(Xe.T @ (d - Xe @ w)) / X.shape[0]

    h = 1e-6
    numerico = np.empty_like(w)
    for i in range(w.size):
        mas, menos = w.copy(), w.copy()
        mas[i] += h
        menos[i] -= h
        numerico[i] = (ecm_de_pesos(mas, X, d) - ecm_de_pesos(menos, X, d)) / (2 * h)
    assert np.allclose(analitico, numerico, atol=1e-6)


def probar_adaline_diverge_con_razon_excesiva():
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=3.0, max_epocas=200, inicializacion="aleatoria", semilla=0)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    assert modelo.diverged


def probar_adaline_error_decrece_en_lote():
    """En modo por lotes y bajo la cota de estabilidad, E decrece monotonamente."""
    conjunto = ds.decodificador_binario(3)
    modelo = Adaline(3, razon_aprendizaje=0.2, modo="lote", max_epocas=300,
                     tolerancia=0.0, inicializacion="aleatoria", semilla=2)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
    errores = np.array([h.ecm for h in modelo.historial])
    assert np.all(np.diff(errores) <= 1e-12)


def probar_cota_estabilidad():
    conjunto = ds.decodificador_binario(3)
    cota = Adaline(3).cota_estabilidad(conjunto.X)
    assert 0 < cota < np.inf


# ---------------------------------------------------------------------------
# Hebb
# ---------------------------------------------------------------------------

def probar_hebb_funciona_en_bipolar():
    for nombre in ("AND", "OR"):
        conjunto = ds.compuerta(nombre, "bipolar")
        red = RedHebb(2, 1).entrenar(conjunto.X, conjunto.d)
        assert red.exactitud(conjunto.X, conjunto.d) == 1.0, nombre


def probar_hebb_falla_en_and_binario():
    """Los aportes se cancelan y los pesos de entrada quedan en cero."""
    conjunto = ds.compuerta("AND", "binaria")
    objetivo = 2 * conjunto.d - 1
    red = RedHebb(2, 1).entrenar(conjunto.X, objetivo)
    assert np.allclose(red.W[0, 1:], 0.0)
    assert red.exactitud(conjunto.X, objetivo) < 1.0


def probar_hebb_independiente_de_la_razon():
    """La razon de aprendizaje solo escala los pesos: la frontera no cambia."""
    conjunto = ds.compuerta("AND", "bipolar")
    W1 = RedHebb(2, 1, razon_aprendizaje=1.0).entrenar(conjunto.X, conjunto.d).W
    W2 = RedHebb(2, 1, razon_aprendizaje=0.25).entrenar(conjunto.X, conjunto.d).W
    assert np.allclose(W1 * 0.25, W2)


# ---------------------------------------------------------------------------
# Metricas y datos
# ---------------------------------------------------------------------------

def probar_matriz_confusion():
    M, etiquetas = mt.matriz_confusion([1, 1, -1, -1], [1, -1, -1, -1], etiquetas=[-1, 1])
    assert M.sum() == 4
    assert np.array_equal(etiquetas, [-1, 1])
    assert M[1, 1] == 1 and M[0, 0] == 2


def probar_r2_perfecto():
    assert np.isclose(mt.r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)


def probar_decodificador_es_lineal():
    conjunto = ds.decodificador_binario(4)
    w = ds.pesos_optimos_decodificador(4)
    Xe = np.hstack([np.ones((conjunto.X.shape[0], 1)), conjunto.X])
    assert np.allclose(Xe @ w, conjunto.d)


def probar_letras_son_bipolares():
    conjunto = ds.letras_xo(con_ruido=3, semilla=1)
    assert set(np.unique(conjunto.X).tolist()) <= {-1.0, 1.0}
    assert conjunto.X.shape[1] == 25


def probar_conjunto_valida_longitudes():
    try:
        ds.Conjunto(X=np.zeros((3, 2)), d=np.zeros(4), nombre="malo")
    except ValueError:
        return
    raise AssertionError("se acepto un conjunto con X y d de distinta longitud")


# ---------------------------------------------------------------------------
# Perceptron Multicapa y Backpropagation
# ---------------------------------------------------------------------------

def probar_mlp_gradiente_coincide_con_derivada_numerica():
    """El gradiente analitico de retropropagacion coincide con la derivada numerica."""
    red = PerceptronMulticapa((2, 3, 1), activacion_oculta="sigmoide", activacion_salida="sigmoide", semilla=7)
    X = np.array([[0.2, 0.8], [0.5, 0.1], [0.9, 0.4]])
    d = np.array([0.7, 0.3, 0.9])
    eps = 1e-6

    W = red.pesos[0]
    w_orig = W[0, 1]
    W[0, 1] = w_orig + eps
    e_pos = red.calcular_ecm(X, d)
    W[0, 1] = w_orig - eps
    e_neg = red.calcular_ecm(X, d)
    W[0, 1] = w_orig
    grad_num = (e_pos - e_neg) / (2 * eps)

    acts, pots = red.propagar_lote(X)
    deriv_out = red.act_salida.derivada(pots[-1])
    delta_out = (d.reshape(-1, 1) - acts[-1]) * deriv_out
    deriv_hid = red.act_oculta.derivada(pots[1])
    delta_hid = (delta_out @ red.pesos[1]) * deriv_hid
    grad_ana = -(1.0 / len(X)) * (delta_hid.T @ acts[0])[0, 1]

    assert np.isclose(grad_num, grad_ana, atol=1e-5), f"Num: {grad_num}, Ana: {grad_ana}"


def probar_mlp_resuelve_xor():
    """MLP 2-2-1 resuelve exactamente el XOR (exactitud 1.0 y ECM < 0.01)."""
    conjunto = ds.compuerta("XOR", "binaria")
    red = PerceptronMulticapa(
        arquitectura=(2, 2, 1),
        razon_aprendizaje=1.0,
        momento=0.5,
        max_epocas=2500,
        tolerancia=1e-4,
        semilla=42,
    )
    red.entrenar(conjunto.X, conjunto.d)
    preds = red.clasificar(conjunto.X)
    assert np.array_equal(preds, conjunto.d)
    assert red.calcular_ecm(conjunto.X, conjunto.d) < 0.01


def probar_mlp_aproxima_funcion():
    """MLP 1-8-1 con salida lineal aproxima una funcion continua con R2 > 0.95."""
    X = np.linspace(-np.pi, np.pi, 60).reshape(-1, 1)
    d = np.sin(X).ravel()
    red = PerceptronMulticapa(
        arquitectura=(1, 8, 1),
        activacion_oculta="sigmoide",
        activacion_salida="identidad",
        razon_aprendizaje=0.08,
        momento=0.7,
        modo="lote",
        max_epocas=3000,
        tolerancia=1e-5,
        semilla=42,
    )
    red.entrenar(X, d)
    preds = red.predecir(X)
    coef_r2 = mt.r2(preds, d)
    assert coef_r2 > 0.95, f"R2 insuficiente: {coef_r2}"


def probar_mlp_valida_argumentos():
    """El MLP valida arquitecturas cortas, neuronas invalidas y razones no positivas."""
    try:
        PerceptronMulticapa((2,))
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto arquitectura con menos de 2 capas")

    try:
        PerceptronMulticapa((2, 0, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto capa con 0 neuronas")

    try:
        PerceptronMulticapa((2, 2, 1), razon_aprendizaje=-0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto razon_aprendizaje negativa")


# ---------------------------------------------------------------------------
# Mapas Autoorganizados de Kohonen (SOM)
# ---------------------------------------------------------------------------

def probar_kohonen_error_cuantizacion_decrece():
    """El entrenamiento no supervisado reduce el error de cuantizacion promedio."""
    conjunto = ds.distribucion_clusters_2d(n_por_cluster=30, semilla=7)
    som = MapaAutoorganizado(forma_mapa=(4, 4), max_epocas=50, semilla=7)
    som.entrenar(conjunto.X)

    ec_inicial = som.historia_ec[0]
    ec_final = som.historia_ec[-1]
    assert ec_final < ec_inicial * 0.6, f"Inicial {ec_inicial}, Final {ec_final}"


def probar_kohonen_identifica_clusters():
    """El SOM mapea clusters distantes en neuronas separadas con alta pureza."""
    conjunto = ds.distribucion_clusters_2d(n_por_cluster=30, semilla=11)
    som = MapaAutoorganizado(forma_mapa=(6, 6), max_epocas=60, semilla=11)
    som.entrenar(conjunto.X)

    comp = som.composicion_clusters(conjunto.X, conjunto.d)
    puros = sum(max(cnts.values()) for cnts in comp.values())
    pureza = puros / len(conjunto.X)
    assert pureza >= 0.95, f"Pureza insuficiente: {pureza}"


def probar_kohonen_matriz_u_dimensiones():
    """La Matriz U tiene dimensiones iguales a la reticula y valores no negativos."""
    som = MapaAutoorganizado(forma_mapa=(5, 7), semilla=1)
    som.W = np.random.default_rng(1).uniform(-2, 2, size=(5, 7, 2))
    U = som.matriz_u()
    assert U.shape == (5, 7)
    assert np.all(U >= 0.0)


def probar_kohonen_valida_argumentos():
    """El SOM rechaza reticulas degeneradas o razones de aprendizaje invalidas."""
    try:
        MapaAutoorganizado(forma_mapa=(0, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto reticula con dimension cero")

    try:
        MapaAutoorganizado(razon_aprendizaje_inicial=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto razon inicial mayor que 1")


# ---------------------------------------------------------------------------
# Red de Hopfield (Memoria Asociativa)
# ---------------------------------------------------------------------------

def probar_hopfield_almacena_y_recupera_perfecto():
    """Los 4 patrones de 7x6 (A, B, C, D) son puntos fijos exactos con la regla de proyeccion."""
    conjunto = ds.letras_hopfield_7x6()
    red = RedHopfield(n_neuronas=42, regla="proyeccion", semilla=42)
    red.almacenar(conjunto.X, nombres=["A", "B", "C", "D"])

    for p in conjunto.X:
        rec, _, conv = red.recuperar(p, max_pasos=10, modo="sincrono")
        assert conv
        assert np.array_equal(rec, p)


def probar_hopfield_energia_decrece_o_constante():
    """La funcion de energia de Lyapunov decrece monotonamente en cada ciclo asincrono."""
    conjunto = ds.letras_hopfield_7x6()
    red = RedHopfield(n_neuronas=42, regla="proyeccion", semilla=123)
    red.almacenar(conjunto.X)

    patron_contaminado = contaminar_patron(conjunto.X[0], n_pixeles=6, semilla=123)
    _, historia_e, conv = red.recuperar(patron_contaminado, max_pasos=20, modo="asincrono")

    assert conv
    for t in range(len(historia_e) - 1):
        assert historia_e[t + 1] <= historia_e[t] + 1e-9, f"Energia subio: {historia_e[t]} -> {historia_e[t+1]}"


def probar_hopfield_corrige_ruido():
    """La red de Hopfield restaura la letra correcta con 4 pixeles de ruido."""
    conjunto = ds.letras_hopfield_7x6()
    red = RedHopfield(n_neuronas=42, regla="proyeccion", semilla=7)
    red.almacenar(conjunto.X, nombres=["A", "B", "C", "D"])

    for idx, p in enumerate(conjunto.X):
        ruidoso = contaminar_patron(p, n_pixeles=4, semilla=idx * 10 + 3)
        rec, _, _ = red.recuperar(ruidoso, max_pasos=25, modo="asincrono")
        assert np.array_equal(rec, p), f"Fallo al recuperar letra {idx}"


def probar_hopfield_io_imagenes_png():
    """Guardar y cargar una imagen PNG produce exactamente el mismo patron bipolar."""
    import tempfile
    patron = ds.letra_7x6("A")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        guardar_imagen_png(patron, tmp_path, forma=(7, 6))
        cargado = cargar_imagen_png(tmp_path, forma=(7, 6))
        assert np.array_equal(patron, cargado)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def probar_hopfield_valida_argumentos():
    """Hopfield rechaza patrones no bipolares o de dimension incompatible."""
    red = RedHopfield(n_neuronas=42)
    try:
        red.almacenar(np.zeros((2, 42)))  # no bipolar
    except ValueError:
        pass
    else:
        raise AssertionError("Se aceptaron patrones con ceros en Hopfield")

    try:
        red.almacenar(np.ones((2, 20)))  # dimension erronea
    except ValueError:
        pass
    else:
        raise AssertionError("Se acepto patron de dimension distinta a n_neuronas")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    pruebas = sorted(
        ((nombre, funcion) for nombre, funcion in globals().items()
         if nombre.startswith("probar_") and callable(funcion)),
        key=lambda par: par[0],
    )
    fallos = []
    print(f"Ejecutando {len(pruebas)} pruebas del paquete ce_rna\n")
    for nombre, funcion in pruebas:
        try:
            funcion()
            print(f"  [OK]     {nombre}")
        except AssertionError as error:
            fallos.append((nombre, f"AssertionError: {error}"))
            print(f"  [FALLO]  {nombre}: {error}")
        except Exception as error:  # pragma: no cover
            fallos.append((nombre, f"{type(error).__name__}: {error}"))
            print(f"  [ERROR]  {nombre}: {type(error).__name__}: {error}")

    print(f"\n{len(pruebas) - len(fallos)} de {len(pruebas)} pruebas superadas")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
