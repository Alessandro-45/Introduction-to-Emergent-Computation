"""
Experimento 13 --- Red de Hopfield moderna: "Hopfield Networks is All You Need".

Actividad 6 de la guia de evaluacion (seminario de investigacion).  Se
implementa y se pone a prueba el modelo central de Ramsauer et al. (2021) con
los mismos datos de la Actividad 5, para poder comparar directamente con la red
de Hopfield clasica.

Contenido
---------
1. Del modelo clasico al moderno: energia y regla de actualizacion.
2. Las mismas cuatro letras: recuperacion en una sola actualizacion.
3. Tolerancia al ruido: clasica (Hebb, pseudoinversa) frente a moderna.
4. Capacidad: el alfabeto completo y patrones aleatorios.
5. El papel de beta: punto fijo global, estados metaestables y patron unico.
6. Equivalencia con la atencion de los Transformers.

Ejecucion
---------
    python experimentos/exp13_hopfield_moderna.py
"""

from __future__ import annotations

import string

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import imagenes
from ce_rna import visual as vz
from ce_rna.hopfield import RedHopfield
from ce_rna.hopfield_moderna import RedHopfieldModerna, atencion, softmax
from ce_rna.reportes import Reporte, encabezado_experimento

FORMA = ds.FORMA_7X6
NOMBRES = list("ABCD")
ALFABETO = list(string.ascii_uppercase)
BETA = 0.5                # temperatura inversa de referencia (productos escalares en [-42, 42])
NIVELES_RUIDO = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21)
ENSAYOS = 200


def alfabeto_7x6() -> np.ndarray:
    """Las 26 mayusculas renderizadas con DejaVu Sans Bold y reducidas a 7x6.

    Se conserva la proporcion de cada letra (si no, la I estirada llenaria la retina) y se
    enciende un pixel con un 40 % de trazo en su bloque, para no perder trazos finos.
    """
    return np.array([imagenes.preprocesar(imagenes.renderizar_letra(c), forma=FORMA, conservar_proporcion=True,
                                          umbral_bloque=0.4) for c in ALFABETO])


def vecino_mas_cercano(x, patrones) -> int:
    """Indice del patron a menor distancia de Hamming (empates: el primero)."""
    return int(np.argmax(patrones @ x))


def evaluar_ruido(patrones, modelos: dict, niveles, ensayos) -> pd.DataFrame:
    """Tasa de recuperacion exacta por modelo y nivel de ruido (mismas consultas para todos)."""
    filas = []
    for k in niveles:
        aciertos = {nombre: 0 for nombre in modelos}
        aciertos["vecino mas cercano"] = 0
        total = 0
        for mu, patron in enumerate(patrones):
            for e in range(ensayos):
                x = ds.contaminar(patron, k, semilla=100_000 * k + 1000 * mu + e)
                for nombre, recuperar in modelos.items():
                    aciertos[nombre] += recuperar(x) == mu
                aciertos["vecino mas cercano"] += vecino_mas_cercano(x, patrones) == mu
                total += 1
        fila = {"pixeles_invertidos": k, "ruido_%": 100 * k / patrones.shape[1]}
        fila.update({f"{nombre}_%": 100 * a / total for nombre, a in aciertos.items()})
        filas.append(fila)
    return pd.DataFrame(filas)


def recuperador_clasico(red: RedHopfield):
    def recuperar(x):
        mu, tipo = red.identificar(red.recuperar(x).estado)
        return mu if tipo == "patron" else -1
    return recuperar


def recuperador_moderno(red: RedHopfieldModerna, pasos: int | None = None):
    def recuperar(x):
        if pasos is None:
            estado = red.recuperar(x).estado
        else:
            estado = x
            for _ in range(pasos):
                estado = red.actualizar(estado)
        mu, _ = red.identificar(estado)
        return mu
    return recuperar


def curvas(df: pd.DataFrame, columnas: dict, xlabel: str, titulo: str, x="pixeles_invertidos", logx=False,
           leyenda="lower left"):
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.4, 4.4))
    for k, (columna, (etiqueta, color, marcador, estilo_linea)) in enumerate(columnas.items()):
        ax.plot(df[x], df[columna], color=color, marker=marcador, markersize=5.5, lw=2.0,
                ls=estilo_linea, label=etiqueta)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recuperacion exacta (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(titulo)
    ax.legend(loc=leyenda)
    return fig, ax


def main() -> None:
    encabezado_experimento("Experimento 13 - Red de Hopfield moderna (Ramsauer et al., 2021)")

    reporte = Reporte(
        titulo="Experimento 13 --- Red de Hopfield moderna: \"Hopfield Networks is All You Need\"",
        nombre_archivo="13_hopfield_moderna",
        resumen=(
            "Implementacion del modelo central de Ramsauer et al. (2021), *Hopfield Networks is All "
            "You Need* (ICLR 2021), articulo elegido para el seminario de la Actividad 6. La red de "
            "Hopfield moderna sustituye la energia cuadratica de 1982 por un *log-sum-exp* y su "
            "regla de actualizacion resulta ser la atencion de los Transformers. Se pone a prueba "
            "con las mismas letras de 7x6 pixeles de la Actividad 5 para comparar directamente con "
            "la red clasica: recuperacion en un paso, tolerancia al ruido, capacidad, papel de la "
            "temperatura inversa beta y equivalencia con la atencion. Todo esta escrito con numpy, "
            "sin librerias de aprendizaje automatico."
        ),
    )

    letras = ds.letras_abcd("gruesa").X
    d = letras.shape[1]

    # ------------------------------------------------------------------
    # 1. Del modelo clasico al moderno
    # ------------------------------------------------------------------
    reporte.seccion("1. Del modelo clasico al moderno")
    reporte.texto(
        "La energia de la red clasica con la regla de Hebb se puede escribir en funcion de los "
        "solapamientos del estado con cada patron:"
    )
    reporte.formula(r"E_{\text{clasica}}(\mathbf{s}) = -\tfrac{1}{2}\,\mathbf{s}^{\top}W\mathbf{s} "
                    r"= -\frac{1}{2N}\sum_{\mu=1}^{P} \bigl(\mathbf{x}_{\mu}^{\top}\mathbf{s}\bigr)^{2}")
    reporte.texto(
        "Cada patron atrae al estado con el **cuadrado** de su solapamiento. Krotov y Hopfield "
        "(2016) usaron potencias mayores F(z) = zⁿ y Demircigil et al. (2017) la exponencial: "
        "cuanto mas empinada es F, mas domina el patron mas parecido sobre la diafonia de los "
        "demas, y la capacidad pasa de crecer linealmente a crecer exponencialmente con la "
        "dimension. Ramsauer et al. (2021) generalizan a estados **continuos** con la energia"
    )
    reporte.formula(r"E(\boldsymbol{\xi}) = -\operatorname{lse}\bigl(\beta, X^{\top}\boldsymbol{\xi}\bigr) "
                    r"+ \tfrac{1}{2}\boldsymbol{\xi}^{\top}\boldsymbol{\xi} + \beta^{-1}\log P "
                    r"+ \tfrac{1}{2}M^{2}, \qquad "
                    r"\operatorname{lse}(\beta,\mathbf{z}) = \beta^{-1}\log\sum_{\mu} e^{\beta z_{\mu}}")
    reporte.texto(
        "con X = (x₁, …, x_P) los patrones en columnas y M la mayor norma de un patron. El "
        "*log-sum-exp* es un maximo suave de los solapamientos y el termino ½‖ξ‖² impide que el "
        "estado crezca sin limite. Minimizando E con el procedimiento concavo-convexo se obtiene "
        "la regla de actualizacion:"
    )
    reporte.formula(r"\boldsymbol{\xi}^{\text{nuevo}} = X\,\operatorname{softmax}\bigl(\beta X^{\top}\boldsymbol{\xi}\bigr)")
    reporte.lista([
        "**No hay pesos que entrenar**: la memoria es la propia matriz de patrones X (d x P). "
        "Anadir un patron no modifica a los demas.",
        "**Estados continuos**: el estado es un vector real, no un vector de ±1.",
        "**Actualizacion sincrona** de todas las componentes a la vez; la energia baja igualmente "
        "(no hace falta la dinamica asincrona de la red clasica).",
        "**beta** es una temperatura inversa: controla cuanto se concentra softmax en el patron "
        "mas parecido.",
    ])

    # ------------------------------------------------------------------
    # 2. Las mismas cuatro letras
    # ------------------------------------------------------------------
    reporte.seccion("2. Las mismas cuatro letras: recuperacion en una actualizacion")
    moderna = RedHopfieldModerna(beta=BETA).almacenar(letras)
    hebb = RedHopfield(d, regla="hebb").almacenar(letras)
    print(moderna.resumen())
    reporte.texto(
        f"Se almacenan las letras A, B, C y D (trazo grueso, bipolares, d = {d}) con beta = {BETA}. "
        "Con patrones bipolares el producto escalar es x·ξ = d − 2·(distancia de Hamming), asi "
        "que beta = 0.5 multiplica por e² ≈ 7.4 el peso relativo de un patron por cada pixel de "
        "diferencia. La separacion de cada patron, Δ_μ = x_μ·x_μ − max_{ν≠μ} x_μ·x_ν, es la "
        "magnitud que controla el teorema de recuperacion del articulo:"
    )
    reporte.tabla(pd.DataFrame({"letra": NOMBRES, "separacion_Delta": moderna.separacion(),
                                "beta*Delta": BETA * moderna.separacion()}),
                  "Separacion de los patrones almacenados", nombre_csv="13_separacion_abcd")

    k_ej = 8
    filas_fig, titulos_fig, filas_tab = [], [], []
    for mu, nombre in enumerate(NOMBRES):
        x = ds.contaminar(letras[mu], k_ej, semilla=40 + mu)
        res_h = hebb.recuperar(x, semilla=mu)
        uno = moderna.actualizar(x)
        res_m = moderna.recuperar(x)
        filas_fig += [letras[mu], x, res_h.estado, uno]
        titulos_fig += [f"{nombre} original", f"{k_ej} px invertidos", f"clasica ({res_h.iteraciones} barridos)",
                        "moderna (1 paso)"]
        filas_tab.append({
            "letra": nombre,
            "clasica_correcta": hebb.identificar(res_h.estado) == (mu, "patron"),
            "clasica_barridos": res_h.iteraciones,
            "moderna_1_paso_correcta": moderna.identificar(uno) == (mu, "patron"),
            "error_tras_1_paso": float(np.linalg.norm(uno - letras[mu])),
            "iteraciones_hasta_converger": res_m.iteraciones,
            "error_final": float(np.linalg.norm(res_m.estado - letras[mu])),
            "peso_del_patron_correcto": float(res_m.pesos[mu]),
        })
    fig, _ = vz.rejilla_retinas(filas_fig, titulos_fig, forma=FORMA, n_columnas=4,
                                sup_titulo=f"Recuperacion de letras con {k_ej} de 42 pixeles invertidos")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_recuperacion_letras.png"),
                   "Por filas: original, consulta ruidosa, red clasica (Hebb, dinamica asincrona) y red "
                   "moderna tras una sola actualizacion.")
    reporte.tabla(pd.DataFrame(filas_tab), "Una actualizacion basta", nombre_csv="13_un_paso")
    reporte.texto(
        "La red moderna devuelve el patron correcto tras **una sola actualizacion**, con un error "
        "de milesimas sobre una norma de √42 ≈ 6.5: es la recuperacion en un paso que predice el "
        "articulo para patrones bien separados, con un error que decrece exponencialmente con "
        "beta·Δ. Las iteraciones siguientes solo corrigen decimales, y el punto fijo final no es "
        "exactamente x_μ sino un punto muy cercano: los demas patrones conservan un peso de "
        "softmax minusculo pero no nulo. Por eso se cuenta como acierto que sgn(ξ) coincida con la "
        "letra."
    )

    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.0, 4.2))
    colores = [vz.AZUL, vz.NARANJA, vz.AQUA, vz.AMARILLO]
    for mu, nombre in enumerate(NOMBRES):
        for e, ls in zip(range(3), ("-", "--", ":")):
            x = ds.contaminar(letras[mu], 6 + 4 * e, semilla=500 + 10 * mu + e)
            r = RedHopfieldModerna(beta=0.15).almacenar(letras).recuperar(x)
            ax.plot(range(len(r.energia)), r.energia, color=colores[mu], ls=ls, lw=1.8,
                    marker="o", markersize=3.5, label=f"{nombre}" if e == 0 else None)
    ax.set_xlabel("actualizacion")
    ax.set_ylabel("energia E(ξ)")
    ax.set_title("La energia nunca aumenta (beta = 0.15)")
    ax.legend(title="letra (lineas: 6, 10, 14 px de ruido)", loc="upper right")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_energia.png"),
                   "Energia a lo largo de la recuperacion. Se usa beta = 0.15 para que la convergencia "
                   "dure varios pasos y se vea la curva; con beta = 0.5 cae en un paso.")

    # ------------------------------------------------------------------
    # 3. Tolerancia al ruido
    # ------------------------------------------------------------------
    reporte.seccion("3. Tolerancia al ruido: clasica frente a moderna")
    pinv = RedHopfield(d, regla="pseudoinversa").almacenar(letras)
    modelos = {
        "clasica Hebb": recuperador_clasico(hebb),
        "clasica pseudoinversa": recuperador_clasico(pinv),
        "moderna 1 paso": recuperador_moderno(moderna, pasos=1),
        "moderna": recuperador_moderno(moderna),
    }
    df_ruido = evaluar_ruido(letras, modelos, NIVELES_RUIDO, ENSAYOS)
    print(df_ruido.round(1).to_string(index=False))
    reporte.texto(
        f"Mismo protocolo que el experimento 11: para cada letra y cada nivel de ruido se invierten "
        f"k pixeles al azar ({ENSAYOS} consultas por letra y nivel, las mismas para todos los "
        "modelos) y se cuenta como acierto solo la recuperacion **exacta** de la letra. Como "
        "referencia se incluye el **vecino mas cercano** (la letra a menor distancia de Hamming), "
        "que es el mejor resultado posible para una memoria que solo ve la consulta."
    )
    reporte.tabla(df_ruido, "Recuperacion exacta (%) frente al ruido", nombre_csv="13_ruido_abcd")
    fig, _ = curvas(df_ruido, {
        "clasica Hebb_%": ("clasica, Hebb", vz.NARANJA, "s", "-"),
        "clasica pseudoinversa_%": ("clasica, pseudoinversa", vz.AMARILLO, "D", "-"),
        "moderna_%": ("moderna (beta = 0.5)", vz.AZUL, "o", "-"),
        "vecino mas cercano_%": ("vecino mas cercano (optimo)", vz.TINTA, None, ":"),
    }, "pixeles invertidos (de 42)", "Cuatro letras: recuperacion frente al ruido")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_ruido_abcd.png"),
                   "La red moderna coincide con el vecino mas cercano: falla solo cuando el ruido "
                   "acerca la consulta a otra letra.")
    fila16 = df_ruido[df_ruido["pixeles_invertidos"] == 16].iloc[0]
    reporte.texto(
        f"Con 16 pixeles invertidos (38 %) la red clasica de Hebb recupera el "
        f"{fila16['clasica Hebb_%']:.0f} % de las letras y la moderna el {fila16['moderna_%']:.0f} %, "
        f"igual que el vecino mas cercano ({fila16['vecino mas cercano_%']:.0f} %). La red moderna "
        "no tiene estados espurios ni inversos con estos patrones: con beta grande softmax elige "
        "practicamente el patron de mayor solapamiento, de modo que solo se equivoca cuando la "
        "propia consulta ya se parece mas a otra letra. Sus errores son los inevitables."
    )

    # ------------------------------------------------------------------
    # 4. Capacidad
    # ------------------------------------------------------------------
    reporte.seccion("4. Capacidad: el alfabeto completo y patrones aleatorios")
    alfabeto = alfabeto_7x6()
    fig, _ = vz.rejilla_retinas(list(alfabeto), ALFABETO, forma=FORMA, n_columnas=9,
                                sup_titulo="Las 26 mayusculas en 7x6 pixeles")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_alfabeto.png"),
                   "Mayusculas de DejaVu Sans Bold reducidas a 7x6 con `ce_rna.imagenes.preprocesar`, "
                   "conservando la proporcion de cada letra.")
    m26 = RedHopfieldModerna(beta=BETA).almacenar(alfabeto)
    G = alfabeto @ alfabeto.T
    np.fill_diagonal(G, -np.inf)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    reporte.texto(
        f"Las 26 letras son distintas, pero algunas estan muy cerca: la pareja mas parecida es "
        f"{ALFABETO[i]}-{ALFABETO[j]}, a solo {int((d - G[i, j]) / 2)} pixeles de distancia. La red "
        f"clasica con Hebb admite unos 0.138·42 ≈ 5.8 patrones aleatorios; 26 letras "
        "correlacionadas son casi cinco veces esa capacidad."
    )
    modelos26 = {
        "clasica Hebb": recuperador_clasico(RedHopfield(d, "hebb").almacenar(alfabeto)),
        "clasica pseudoinversa": recuperador_clasico(RedHopfield(d, "pseudoinversa").almacenar(alfabeto)),
        "moderna": recuperador_moderno(m26),
    }
    df26 = evaluar_ruido(alfabeto, modelos26, (0, 1, 2, 3, 4, 6), 40)
    print(df26.round(1).to_string(index=False))
    reporte.tabla(df26, "Alfabeto completo (26 patrones, d = 42): recuperacion exacta (%)",
                  nombre_csv="13_ruido_alfabeto")
    reporte.texto(
        "Con Hebb **ni siquiera las letras sin ruido son puntos fijos**: la diafonia de 25 "
        "memorias borra cualquier patron. La pseudoinversa las convierte en puntos fijos, pero su "
        "cuenca de atraccion es minima. La red moderna recupera practicamente todo mientras el "
        "ruido no supere la mitad de la distancia entre las letras mas parecidas."
    )

    tamanos = (2, 4, 6, 8, 10, 15, 20, 30, 50, 100, 300, 1000, 3000, 10_000, 30_000, 100_000)
    betas_cap = (BETA, 2.0)
    ruido_cap = 6
    rng = np.random.default_rng(0)
    filas_cap = []
    for P in tamanos:
        nombres = ["clasica Hebb", "clasica pseudoinversa", *(f"moderna beta={b:g}" for b in betas_cap),
                   "vecino mas cercano"]
        aciertos = dict.fromkeys(nombres, 0)
        total = 0
        repeticiones, n_consultas = (4, 25) if P <= 1000 else (1, 60)
        for _ in range(repeticiones):
            X = rng.choice([-1.0, 1.0], size=(P, d))
            modelos_cap = {f"moderna beta={b:g}": recuperador_moderno(RedHopfieldModerna(beta=b).almacenar(X))
                           for b in betas_cap}
            modelos_cap["vecino mas cercano"] = lambda x, X=X: vecino_mas_cercano(x, X)
            if P <= 60:  # con mas patrones que neuronas la red clasica ya no recupera nada
                modelos_cap["clasica Hebb"] = recuperador_clasico(RedHopfield(d, "hebb").almacenar(X))
                modelos_cap["clasica pseudoinversa"] = recuperador_clasico(
                    RedHopfield(d, "pseudoinversa").almacenar(X))
            for mu in rng.choice(P, size=min(P, n_consultas), replace=False):
                x = ds.contaminar(X[mu], ruido_cap, semilla=int(rng.integers(1 << 30)))
                for nombre, recuperar in modelos_cap.items():
                    aciertos[nombre] += recuperar(x) == mu
                total += 1
        filas_cap.append({"patrones_P": P, "P/d": P / d, **{f"{n}_%": 100 * a / total for n, a in aciertos.items()}})
    df_cap = pd.DataFrame(filas_cap)
    print(df_cap.round(1).to_string(index=False))
    reporte.texto(
        f"Para ver la capacidad sin depender de la forma de las letras se almacenan P patrones "
        f"bipolares **aleatorios** de d = 42 componentes, desde 2 hasta 100 000, y se consulta con "
        f"{ruido_cap} pixeles invertidos (≈ 14 %). Hasta P = 1000 se promedian 4 repeticiones de 25 "
        "consultas; por encima, una repeticion de 60. Para P > 60 la red clasica ya no recupera "
        "ningun patron y no se simula (aparece con 0 %)."
    )
    reporte.tabla(df_cap, "Capacidad con patrones aleatorios (d = 42)", nombre_csv="13_capacidad")
    fig, ax = curvas(df_cap, {
        "clasica Hebb_%": ("clasica, Hebb", vz.NARANJA, "s", "-"),
        "clasica pseudoinversa_%": ("clasica, pseudoinversa", vz.AMARILLO, "D", "-"),
        f"moderna beta={BETA:g}_%": (f"moderna, beta = {BETA:g}", vz.AZUL, "o", "-"),
        "moderna beta=2_%": ("moderna, beta = 2", vz.VIOLETA, "^", "-"),
        "vecino mas cercano_%": ("vecino mas cercano (optimo)", vz.TINTA, None, ":"),
    }, "patrones almacenados P (escala log)", "Capacidad: patrones aleatorios de 42 componentes",
        x="patrones_P", logx=True, leyenda="center right")
    ax.axvline(0.138 * d, color=vz.TINTA_2, ls=":", lw=1.2)
    ax.text(0.138 * d * 1.08, 50, "0.138·d", color=vz.TINTA_2, fontsize=8)
    ax.axvline(d, color=vz.TINTA_2, ls=":", lw=1.2)
    ax.text(d * 1.08, 40, "P = d", color=vz.TINTA_2, fontsize=8)
    reporte.figura(vz.guardar(fig, FIGURAS / "13_capacidad.png"),
                   "La red clasica se hunde cerca de 0.138·d; la moderna guarda miles de patrones en "
                   "las mismas 42 dimensiones mientras beta sea suficientemente grande.")

    def ultimo_100(columna):
        sub = df_cap[df_cap[columna] >= 99.0]["patrones_P"]
        return int(sub.max()) if len(sub) else 0

    reporte.texto(
        f"La red clasica con Hebb deja de recuperar todos los patrones a partir de P = "
        f"{int(df_cap[df_cap['clasica Hebb_%'] < 99]['patrones_P'].min())}, cerca de 0.138·d ≈ 5.8. "
        f"La red moderna con beta = {BETA:g} recupera al menos el 99 % hasta P = "
        f"{ultimo_100(f'moderna beta={BETA:g}_%'):,} y con beta = 2 hasta P = "
        f"{ultimo_100('moderna beta=2_%'):,}: miles de veces mas patrones que la clasica con las "
        "mismas 42 neuronas.".replace(",", " ")
    )
    reporte.lista([
        "**Con beta suficiente, el limite es la geometria**: la curva de beta = 2 se pega al vecino "
        "mas cercano. Cuando hay tantos patrones que otro queda a menos distancia de la consulta "
        "que el correcto, ningun metodo puede acertar.",
        f"**Con beta pequeno el limite llega antes**: con beta = {BETA:g} y muchos patrones, la suma "
        "de los pesos de miles de patrones \"bastante parecidos\" supera a la del correcto y la "
        "red cae en un estado metaestable (seccion 5). El teorema del articulo lo expresa asi: "
        "para recuperar con un paso, beta·Δ debe crecer como el logaritmo del numero de patrones.",
        "El articulo demuestra que, para patrones aleatorios sobre una esfera, el numero de "
        "patrones que se pueden almacenar y recuperar crece **exponencialmente** con la "
        "dimension d, frente al crecimiento lineal (0.138·d) de la red clasica.",
    ])

    # ------------------------------------------------------------------
    # 5. El papel de beta
    # ------------------------------------------------------------------
    reporte.seccion("5. El papel de beta: tres tipos de punto fijo")
    consulta = ds.contaminar(alfabeto[1], 4, semilla=3)   # una B con 4 pixeles invertidos
    betas = np.geomspace(0.001, 2.0, 40)
    filas_beta = []
    for b in betas:
        r = RedHopfieldModerna(beta=b).almacenar(alfabeto).recuperar(consulta, max_iteraciones=1000)
        red_b = RedHopfieldModerna(beta=b).almacenar(alfabeto)
        orden = np.argsort(r.pesos)[::-1]
        filas_beta.append({
            "beta": b, "tipo": red_b.tipo_punto_fijo(r.pesos),
            "patrones_efectivos": red_b.patrones_efectivos(r.pesos),
            "peso_de_B": float(r.pesos[1]),
            "mayor_peso": f"{ALFABETO[orden[0]]} ({r.pesos[orden[0]]:.2f})",
            "segundo": f"{ALFABETO[orden[1]]} ({r.pesos[orden[1]]:.2f})",
            "iteraciones": r.iteraciones,
            "energia_no_aumenta": bool(np.all(np.diff(r.energia) <= 1e-9)),
        })
    df_beta = pd.DataFrame(filas_beta)
    reporte.texto(
        "Se almacenan las 26 letras y se consulta con una B con 4 pixeles invertidos, variando "
        "beta entre 0.001 y 2. Para cada beta se itera hasta converger y se mira como reparte "
        "softmax el peso en el punto fijo. El **numero efectivo de patrones** 1/Σp²_μ vale 1 si "
        "todo el peso esta en un patron y 26 si se reparte por igual."
    )
    vz.estilo()
    fig, ax = vz.plt.subplots(figsize=(7.4, 4.4))
    ax.plot(df_beta["beta"], df_beta["patrones_efectivos"], color=vz.AZUL, lw=2.2,
            label="numero efectivo de patrones 1/Σp²")
    ax2 = ax.twinx()
    ax2.plot(df_beta["beta"], df_beta["peso_de_B"], color=vz.NARANJA, lw=2.0, ls="--", label="peso de la letra B")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_ylabel("peso de B en softmax", color=vz.NARANJA)
    ax2.grid(False)
    ax.set_xscale("log")
    ax.set_xlabel("beta (escala log)")
    ax.set_ylabel("patrones promediados", color=vz.AZUL)
    ax.set_title("De promediar todo a recordar un solo patron")
    for tipo, color in (("global", vz.GRIS), ("metaestable", "#f6d9a8"), ("unico", "#cfe8dc")):
        sub = df_beta[df_beta["tipo"] == tipo]["beta"]
        if len(sub):
            ax.axvspan(sub.min(), sub.max(), color=color, alpha=0.35, lw=0)
            ax.text(np.sqrt(sub.min() * sub.max()), 20.0, tipo, ha="center", fontsize=9, color=vz.TINTA)
    lineas = ax.get_legend_handles_labels()
    lineas2 = ax2.get_legend_handles_labels()
    ax.legend(lineas[0] + lineas2[0], lineas[1] + lineas2[1], loc="center left")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_regimenes_beta.png"),
                   "Bandas: punto fijo global (gris), estado metaestable (ambar) y patron unico (verde).")

    ejemplos_beta = (0.01, 0.2, 1.0)
    estados, titulos = [consulta], ["consulta: B con 4 px"]
    filas_ej = []
    for b in ejemplos_beta:
        red_b = RedHopfieldModerna(beta=b).almacenar(alfabeto)
        r = red_b.recuperar(consulta, max_iteraciones=1000)
        orden = np.argsort(r.pesos)[::-1][:4]
        estados.append(r.estado)
        titulos.append(f"beta = {b:g}: {red_b.tipo_punto_fijo(r.pesos)}")
        filas_ej.append({"beta": b, "tipo": red_b.tipo_punto_fijo(r.pesos),
                         "patrones_efectivos": red_b.patrones_efectivos(r.pesos),
                         "cuatro_mayores_pesos": ", ".join(f"{ALFABETO[t]} {r.pesos[t]:.2f}" for t in orden)})
    fig, _ = vz.rejilla_retinas(estados, titulos, forma=FORMA, n_columnas=4, divergente=True,
                                sup_titulo="Puntos fijos de la misma consulta para tres valores de beta")
    reporte.figura(vz.guardar(fig, FIGURAS / "13_puntos_fijos_beta.png"),
                   "Estados continuos: azul = +1 (trazo), naranja = −1 (fondo), blanco = 0 (indeciso).")
    reporte.tabla(pd.DataFrame(filas_ej), "Los tres tipos de punto fijo", nombre_csv="13_puntos_fijos_beta")
    reporte.tabla(df_beta, "Barrido completo de beta", nombre_csv="13_barrido_beta")
    reporte.lista([
        "**Punto fijo global** (beta pequeno): softmax reparte el peso casi por igual y el estado "
        "converge a la media de las 26 letras, la misma para cualquier consulta. La red no "
        "recuerda nada concreto.",
        "**Estado metaestable** (beta intermedio): el peso se reparte entre un grupo de letras "
        "parecidas y el estado es su promedio. El articulo interpreta estos estados como una "
        "memoria de *grupos*: recuperan lo que tienen en comun varios patrones.",
        "**Punto fijo de un patron** (beta grande): todo el peso va a la letra mas parecida y se "
        "recupera la B. Es el comportamiento de memoria asociativa de la Actividad 5.",
        f"En los {len(df_beta)} valores de beta la energia **nunca aumenta**: "
        f"{'se cumple en todos' if df_beta['energia_no_aumenta'].all() else 'FALLA en algun caso'}.",
    ])
    reporte.texto(
        "En los Transformers beta = 1/√d_k es fijo, pero las proyecciones aprendidas escalan los "
        "productos escalares, de modo que cada cabeza de atencion opera en uno de estos tres "
        "regimenes. El articulo analiza asi las cabezas de BERT: las de las primeras capas "
        "tienden a promediar globalmente y las de capas superiores promedian parcialmente, en "
        "estados metaestables."
    )

    # ------------------------------------------------------------------
    # 6. Equivalencia con la atencion
    # ------------------------------------------------------------------
    reporte.seccion("6. Equivalencia con la atencion de los Transformers")
    reporte.texto(
        "Si los estados (consultas) R y los patrones almacenados Y pasan por proyecciones "
        "lineales aprendidas, Q = R·W_Q, K = Y·W_K, V = K·W_V, y se toma beta = 1/√d_k, aplicar "
        "la regla de actualizacion a todas las consultas a la vez da"
    )
    reporte.formula(r"Z = \operatorname{softmax}\!\Bigl(\tfrac{1}{\sqrt{d_k}}\,Q K^{\top}\Bigr) V")
    reporte.texto("que es exactamente la atencion de producto escalar de Vaswani et al. (2017). Se "
                  "comprueba numericamente:")
    rng = np.random.default_rng(13)
    filas_att = []
    for n_q, n_k, d_k, d_v in ((4, 4, 42, 42), (5, 26, 16, 8), (10, 100, 64, 64)):
        R, Y = rng.normal(size=(n_q, 20)), rng.normal(size=(n_k, 20))
        W_Q, W_K, W_V = rng.normal(size=(20, d_k)), rng.normal(size=(20, d_k)), rng.normal(size=(d_k, d_v))
        Q, K = R @ W_Q, Y @ W_K
        V = K @ W_V
        Z_atencion = atencion(Q, K, V)
        # Hopfield: cada consulta es un estado xi; los patrones son las claves, y lo que se
        # devuelve es la combinacion de los valores asociados con los mismos pesos softmax
        red = RedHopfieldModerna(beta=1 / np.sqrt(d_k)).almacenar(K)
        Z_hopfield = np.array([red.pesos(q) @ V for q in Q])
        filas_att.append({"consultas": n_q, "patrones": n_k, "d_k": d_k, "d_v": d_v,
                          "max_diferencia_absoluta": float(np.max(np.abs(Z_atencion - Z_hopfield)))})
    # caso sin proyecciones: la atencion con Q = consultas ruidosas y K = V = letras es la actualizacion
    X_ruido = np.array([ds.contaminar(p, 8, semilla=70 + mu) for mu, p in enumerate(letras)])
    Z = atencion(X_ruido, letras, letras, beta=BETA)
    Z_h = np.array([moderna.actualizar(x) for x in X_ruido])
    filas_att.append({"consultas": 4, "patrones": 4, "d_k": 42, "d_v": 42,
                      "max_diferencia_absoluta": float(np.max(np.abs(Z - Z_h)))})
    df_att = pd.DataFrame(filas_att)
    df_att.insert(0, "caso", ["proyecciones aleatorias"] * 3 + ["letras ruidosas, K = V = letras"])
    reporte.tabla(df_att, "Atencion frente a actualizacion de Hopfield", nombre_csv="13_atencion")
    reporte.texto(
        "Las diferencias no pasan de 10⁻¹², el redondeo de punto flotante: son la **misma "
        "operacion**. La ultima fila lo hace sin proyecciones: una capa de atencion cuyas claves "
        "y valores son las cuatro letras recupera las letras limpias a partir de consultas con 8 "
        "pixeles invertidos, en una sola pasada."
    )
    pesos_att = softmax(X_ruido @ letras.T, BETA)
    fig, _ = vz.mapa_calor(pesos_att, titulo="Pesos de atencion: consultas ruidosas → letras",
                           etiquetas_filas=[f"{n} + ruido" for n in NOMBRES], etiquetas_columnas=NOMBRES,
                           etiqueta_barra="softmax(β Q Kᵀ)", figsize=(5.0, 4.0))
    reporte.figura(vz.guardar(fig, FIGURAS / "13_pesos_atencion.png"),
                   "Cada consulta pone casi todo su peso en su propia letra: la matriz de atencion es "
                   "la diagonal.")

    # ------------------------------------------------------------------
    # 7. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("7. Conclusiones")
    reporte.lista([
        "La red de Hopfield moderna cabe en tres lineas de numpy: similitudes, softmax y promedio "
        "ponderado. No tiene pesos que entrenar: la memoria son los propios patrones.",
        "Con las letras de la Actividad 5 recupera en **una sola actualizacion**, tolera tanto ruido "
        "como el vecino mas cercano (el optimo) y no produce estados espurios ni inversos.",
        "Su capacidad no esta limitada por la diafonia: guarda las 26 letras (Hebb no guarda ni "
        "una) y miles de patrones aleatorios en 42 dimensiones, siempre que beta crezca con el "
        "numero de patrones.",
        "beta decide si la red promedia todo, recuerda grupos (estados metaestables) o recuerda un "
        "patron; la energia baja en todos los casos.",
        "Su regla de actualizacion **es** la atencion de los Transformers, comprobado hasta el "
        "redondeo: el puente entre la memoria asociativa de 1982 y el aprendizaje profundo actual.",
        "Limitacion: guarda los patrones explicitamente, de modo que memoria y coste crecen con P "
        "(la red clasica comprime todo en una matriz d x d). Y en datos reales hay que aprender "
        "las proyecciones W_Q, W_K, W_V con retropropagacion, como hace el articulo.",
    ])

    ruta = reporte.escribir()
    print(f"\nInforme: {ruta}")


if __name__ == "__main__":
    main()
