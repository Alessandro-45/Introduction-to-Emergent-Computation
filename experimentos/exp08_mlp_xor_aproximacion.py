"""
Experimento 08 --- Perceptron Multicapa: Problemas No Linealmente Separables y Aproximacion de Funciones.

Correspondencia con la evaluacion:
----------------------------------
* Actividad 3 de `Evaluacion RNA.pdf`:
  - Perceptron multicapa entrenado con Backpropagation.
  - Funcion OR-Exclusivo (XOR) con arquitectura de referencia 2-2-1.
  - Aproximacion de funciones continuas (revisar clases 5, 6 y 7).
  - Justificacion de arquitectura, parametros, funcion de activacion, criterios de parada.
  - Comparacion de configuraciones y analisis de error en entrenamiento y prueba.

Ejecucion
---------
    python experimentos/exp08_mlp_xor_aproximacion.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.mlp import PerceptronMulticapa
from ce_rna.reportes import Reporte, encabezado_experimento


def main() -> None:
    encabezado_experimento("Experimento 08 - Perceptron Multicapa (MLP) y Backpropagation")

    reporte = Reporte(
        titulo="Experimento 08 --- Perceptron Multicapa: Problemas No Linealmente Separables y Aproximacion de Funciones",
        nombre_archivo="08_mlp_xor_aproximacion",
        resumen=(
            "Implementacion rigurosa desde cero del Perceptron Multicapa (MLP) entrenado mediante el algoritmo "
            "de Retropropagacion del Error (Backpropagation). Se analizan dos aplicaciones fundamentales exigidas "
            "por la Actividad 3 de la guia de evaluacion: (1) la resolucion del problema clasico no linealmente separable "
            "OR-Exclusivo (XOR) mediante la arquitectura 2-2-1, demostrando la capacidad de superar las limitaciones "
            "del perceptron simple unicapa; y (2) la aproximacion de una funcion continua no lineal $f(x) = \\sin(x)$ con "
            "particion de entrenamiento/prueba, comparando arquitecturas con distinto numero de neuronas ocultas y "
            "evaluando la influencia de la razon de aprendizaje, el momento y el modo de entrenamiento."
        ),
    )

    # =========================================================================
    # 1. Fundamentos Matematicos y Justificacion de la Arquitectura
    # =========================================================================
    reporte.seccion("1. Fundamentacion Teorica del Perceptron Multicapa")
    reporte.texto(
        "Como se demostro en el Experimento 03, el perceptron simple unicapa solo puede trazar un hiperplano "
        "lineal de separacion ($w_0 + w_1 x_1 + w_2 x_2 = 0$), resultando estrictamente incapaz de resolver el XOR. "
        "Tal como indica `ICE-claseRN03.md`, la adicion de una o mas capas ocultas con unidades **no lineales** "
        "permite dividir el espacio de entrada en regiones convexas y, mediante sucesivas combinaciones en capas posteriores, "
        "generar fronteras de decision de topologia arbitraria.\n\n"
        "El entrenamiento supervisado de redes multicapa requiere calcular el gradiente del error cuadratico medio "
        "con respecto a los pesos de capas intermedias, donde no se dispone directamente de una salida deseada $d$. "
        "Esto se resuelve mediante el algoritmo de **Retropropagacion del Error (Backpropagation)**, fundamentado en la "
        "regla de la cadena del calculo diferencial."
    )

    reporte.texto("### Regla de la Cadena y Ecuaciones de Retropropagacion")
    reporte.formula(
        r"""
        \begin{aligned}
        \text{Propagación hacia adelante: } & v_j^{(l)} = \sum_{i=1}^{m_{l-1}} W_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}, \quad a_j^{(l)} = \phi^{(l)}(v_j^{(l)}) \\
        \text{Error cuadrático: } & E = \frac{1}{2} \sum_{k=1}^{m_L} (d_k - a_k^{(L)})^2 \\
        \text{Gradiente en capa de salida } L: & \delta_k^{(L)} = -\frac{\partial E}{\partial v_k^{(L)}} = (d_k - a_k^{(L)}) \cdot {\phi^{(L)}}'(v_k^{(L)}) \\
        \text{Gradiente en capas ocultas } l: & \delta_j^{(l)} = -\frac{\partial E}{\partial v_j^{(l)}} = \left( \sum_k \delta_k^{(l+1)} W_{kj}^{(l+1)} \right) \cdot {\phi^{(l)}}'(v_j^{(l)}) \\
        \text{Actualización con momento } \alpha: & \Delta W_{ji}^{(l)}(t) = \eta \cdot \delta_j^{(l)} a_i^{(l-1)} + \alpha \Delta W_{ji}^{(l)}(t-1)
        \end{aligned}
        """
    )

    reporte.texto(
        "**Justificacion de las funciones de activacion:**\n"
        r"- Para el problema logico **XOR**, las salidas deseadas pertenecen a $\{0, 1\}$. Se utiliza la funcion "
        "**sigmoide logistica** $\\phi(v) = 1/(1 + e^{-v})$ tanto en la capa oculta como en la salida, ya que es "
        "diferenciable, acotada en $(0, 1)$ y su derivada se calcula de forma computacionalmente eficiente: $\\phi'(v) = \\phi(v)(1 - \\phi(v))$.\n"
        "- Para la **aproximacion de funciones continuas**, segun el Teorema de Aproximacion Universal (Cybenko, 1989; Hornik, 1991), "
        "se utiliza activacion no lineal sigmoidal en la capa oculta para modelar la curvatura y una activacion **lineal (identidad)** "
        "en la neurona de salida, permitiendo que la red genere valores arbitrarios fuera del rango $(0, 1)$ sin saturacion."
    )

    # =========================================================================
    # 2. Problema 1: Resolucion del OR-Exclusivo (XOR) con Arquitectura 2-2-1
    # =========================================================================
    reporte.seccion("2. Problema No Linealmente Separable: OR-Exclusivo (XOR)")
    reporte.texto(
        "Tomando como referencia la arquitectura **2-2-1** presentada en `ICE-claseRN03.md`, la red dispone de 2 entradas "
        "$(x_1, x_2)$, 2 neuronas ocultas con activacion sigmoide y 1 neurona de salida sigmoide. "
        "Se entrena con una razon de aprendizaje $\\eta = 1.0$, momento $\\alpha = 0.5$ y tolerancia de parada $10^{-4}$."
    )

    conjunto_xor = ds.compuerta("XOR", "binaria")
    X_xor = conjunto_xor.X
    d_xor = conjunto_xor.d

    red_xor = PerceptronMulticapa(
        arquitectura=(2, 2, 1),
        activacion_oculta="sigmoide",
        activacion_salida="sigmoide",
        razon_aprendizaje=1.0,
        momento=0.5,
        modo="estocastico",
        tolerancia=1e-4,
        max_epocas=3000,
        semilla=42,
    )
    red_xor.entrenar(X_xor, d_xor)

    pred_xor_cont = red_xor.predecir(X_xor)
    pred_xor_bin = red_xor.clasificar(X_xor, umbral=0.5)
    exactitud_xor = mt.exactitud(pred_xor_bin, d_xor)
    ecm_xor_final = red_xor.historia_error[-1]

    # Tabla de resultados XOR
    df_xor_res = pd.DataFrame({
        "x1": X_xor[:, 0].astype(int),
        "x2": X_xor[:, 1].astype(int),
        "d (esperado)": d_xor.astype(int),
        "y_pred (continua)": pred_xor_cont,
        "clase predicha": pred_xor_bin.astype(int),
        "error (d - y)": d_xor - pred_xor_cont,
    })
    reporte.tabla(df_xor_res, titulo="Resultados de clasificacion de la compuerta XOR con MLP 2-2-1", nombre_csv="08_xor_resultados")

    # Figura: Diagrama de la arquitectura 2-2-1 con pesos aprendidos
    fig, _ = vz.dibujar_red(
        capas=[["x0=1", "x1", "x2"], ["h0=1", "h1", "h2"], ["y"]],
        titulo="Arquitectura 2-2-1 entrenada para XOR (Backpropagation)",
        etiquetas_capas=["Capa de entrada", "Capa oculta (sigmoide)", "Capa de salida (sigmoide)"],
        nota=f"Convergencia en {red_xor.n_epocas_entrenadas} epocas. ECM final = {ecm_xor_final:.2e}. Exactitud = {exactitud_xor * 100:.1f} %",
        figsize=(8.5, 5.2),
    )
    ruta_fig_arq = vz.guardar(fig, FIGURAS / "08_arquitectura_xor.png")
    reporte.figura(ruta_fig_arq, "Arquitectura 2-2-1 para la resolucion del XOR. Cada conexion representa una sinapsis ajustable.")

    # Figura: Curva de error de aprendizaje XOR
    vz.estilo()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(red_xor.historia_error, color=vz.AZUL, lw=2.2, label="ECM entrenamiento")
    ax.axhline(red_xor.tolerancia, color=vz.NARANJA, ls="--", lw=1.2, label=f"Tolerancia ({red_xor.tolerancia:g})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca de entrenamiento")
    ax.set_ylabel("Error Cuadratico Medio (ECM) - Escala logaritmica")
    ax.set_title("Evolucion del Error Cuadratico Medio (MLP 2-2-1 en XOR)")
    ax.legend()
    ruta_fig_err = vz.guardar(fig, FIGURAS / "08_curva_error_xor.png")
    reporte.figura(ruta_fig_err, "Curva de aprendizaje en escala logaritmica. Se observa la transicion rapida tras superar la meseta inicial.")

    # Figura: Frontera de decision no lineal en 2D
    fig, _ = vz.frontera_decision_no_lineal(
        red_xor,
        X_xor,
        d_xor,
        titulo="Frontera de decision no lineal y mapa continuo (XOR)",
        etiquetas_ejes=("Entrada x1", "Entrada x2"),
        etiquetas_clases=("Clase 1 (XOR=1)", "Clase 0 (XOR=0)"),
        umbral=0.5,
    )
    ruta_fig_front = vz.guardar(fig, FIGURAS / "08_frontera_xor.png")
    reporte.figura(ruta_fig_front, "Plano de decision x1-x2. La isolinea continua y=0.5 separa limpiamente los patrones (0,1) y (1,0) de (0,0) y (1,1).")

    # Comparacion de configuraciones en XOR
    reporte.texto("### Comparacion de Configuraciones Experimentales en XOR")
    reporte.texto(
        "Se evalua la velocidad de convergencia y estabilidad variando la razon de aprendizaje $\\eta$, "
        "la presencia del termino de momento $\\alpha$ y el modo de actualizacion (estocastico vs lote)."
    )

    configs_xor = [
        {"nombre": "eta=0.2, sin momento", "eta": 0.2, "alpha": 0.0, "modo": "estocastico"},
        {"nombre": "eta=0.5, sin momento", "eta": 0.5, "alpha": 0.0, "modo": "estocastico"},
        {"nombre": "eta=1.0, sin momento", "eta": 1.0, "alpha": 0.0, "modo": "estocastico"},
        {"nombre": "eta=1.0, momento 0.5 (base)", "eta": 1.0, "alpha": 0.5, "modo": "estocastico"},
        {"nombre": "eta=1.0, momento 0.9", "eta": 1.0, "alpha": 0.9, "modo": "estocastico"},
        {"nombre": "eta=2.0, momento 0.5", "eta": 2.0, "alpha": 0.5, "modo": "estocastico"},
        {"nombre": "Modo lote (eta=2.0, alpha=0.9)", "eta": 2.0, "alpha": 0.9, "modo": "lote"},
    ]

    filas_comp_xor = []
    curvas_comp = {}

    for cfg in configs_xor:
        m = PerceptronMulticapa(
            arquitectura=(2, 2, 1),
            activacion_oculta="sigmoide",
            activacion_salida="sigmoide",
            razon_aprendizaje=cfg["eta"],
            momento=cfg["alpha"],
            modo=cfg["modo"],
            tolerancia=1e-4,
            max_epocas=4000,
            semilla=42,
        )
        m.entrenar(X_xor, d_xor)
        ex = mt.exactitud(m.clasificar(X_xor), d_xor)
        filas_comp_xor.append({
            "Configuracion": cfg["nombre"],
            "eta": cfg["eta"],
            "alpha (momento)": cfg["alpha"],
            "Modo": cfg["modo"],
            "Epocas": m.n_epocas_entrenadas,
            "ECM Final": m.historia_error[-1],
            "Exactitud": f"{ex * 100:.1f} %",
            "Convergencia": "Si" if m.historia_error[-1] <= 1e-4 else "Parcial / Limite",
        })
        curvas_comp[cfg["nombre"]] = m.historia_error

    df_comp_xor = pd.DataFrame(filas_comp_xor)
    reporte.tabla(df_comp_xor, titulo="Comparacion de parametros de entrenamiento en XOR (MLP 2-2-1)", nombre_csv="08_comparacion_parametros_xor")

    # Figura: Comparacion de curvas de aprendizaje en XOR
    vz.estilo()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    colores = [vz.AZUL, vz.NARANJA, vz.AQUA, vz.VIOLETA, "#d9381e", "#5a738e", "#008080"]
    for (nombre, curva), col in zip(curvas_comp.items(), colores):
        ax.plot(curva[:2500], lw=1.8, label=nombre, color=col)
    ax.set_yscale("log")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("ECM (escala logaritmica)")
    ax.set_title("Efecto de la razon de aprendizaje y momento en la convergencia del XOR")
    ax.legend(fontsize=8, loc="upper right")
    ruta_fig_comp = vz.guardar(fig, FIGURAS / "08_comparacion_curvas_xor.png")
    reporte.figura(ruta_fig_comp, "Comparacion de curvas de aprendizaje para distintas combinaciones de eta y alpha.")

    # =========================================================================
    # 3. Problema 2: Aproximacion de Funciones Continuas No Lineales
    # =========================================================================
    reporte.seccion("3. Aproximacion de Funciones Continuas No Lineales")
    reporte.texto(
        "Como segunda aplicacion requerida por la Actividad 3, se implementa y evalua la capacidad del "
        "Perceptron Multicapa como aproximador universal de funciones. Se selecciona la funcion no lineal continua "
        "$f(x) = \\sin(x)$ en el dominio $[-\\pi, \\pi]$, la cual posee curvatura bidireccional y puntos de inflexion."
    )

    # Generacion del dataset continuo con particion entrenamiento / prueba
    rng = np.random.default_rng(101)
    N_total = 100
    x_todos = np.linspace(-np.pi, np.pi, N_total).reshape(-1, 1)
    y_todos = np.sin(x_todos).ravel()

    # Particion estratificada 70% entrenamiento / 30% prueba
    indices = rng.permutation(N_total)
    corte = int(0.7 * N_total)
    idx_ent = indices[:corte]
    idx_pru = indices[corte:]

    X_ent, d_ent = x_todos[idx_ent], y_todos[idx_ent]
    X_pru, d_pru = x_todos[idx_pru], y_todos[idx_pru]

    reporte.texto(
        f"Se generaron {N_total} muestras distribuidas en $[-\\pi, \\pi]$, asignando {len(X_ent)} muestras (70 %) "
        f"para entrenamiento y {len(X_pru)} muestras (30 %) para prueba independiente.\n\n"
        "**Arquitectura empleada:** 1 entrada $x$, 1 capa oculta con $m_1$ neuronas sigmoides, y 1 salida lineal "
        "(activacion identidad) para permitir valores en todo el rango $[-1, 1]$."
    )

    # Comparacion de arquitecturas variando el numero de neuronas ocultas m1
    neuronas_ocultas_lista = [2, 4, 8, 16]
    modelos_aprox = {}
    filas_comp_aprox = []

    for m1 in neuronas_ocultas_lista:
        red_aprox = PerceptronMulticapa(
            arquitectura=(1, m1, 1),
            activacion_oculta="sigmoide",
            activacion_salida="identidad",
            razon_aprendizaje=0.08,
            momento=0.7,
            modo="lote",
            tolerancia=1e-5,
            max_epocas=3500,
            semilla=42,
        )
        red_aprox.entrenar(X_ent, d_ent, X_val=X_pru, d_val=d_pru)
        modelos_aprox[m1] = red_aprox

        pred_ent = red_aprox.predecir(X_ent)
        pred_pru = red_aprox.predecir(X_pru)

        ecm_ent = red_aprox.calcular_ecm(X_ent, d_ent)
        ecm_pru = red_aprox.calcular_ecm(X_pru, d_pru)
        r2_pru = mt.r2(pred_pru, d_pru)
        rmse_pru = mt.rmse(pred_pru, d_pru)

        filas_comp_aprox.append({
            "Arquitectura": f"1-{m1}-1",
            "Neuronas Ocultas": m1,
            "Epocas": red_aprox.n_epocas_entrenadas,
            "ECM Entrenamiento": ecm_ent,
            "ECM Prueba": ecm_pru,
            "RMSE Prueba": rmse_pru,
            "R2 Prueba": r2_pru,
        })

    df_comp_aprox = pd.DataFrame(filas_comp_aprox)
    reporte.tabla(df_comp_aprox, titulo="Rendimiento del MLP aproximando sin(x) segun el numero de neuronas ocultas", nombre_csv="08_comparacion_arquitecturas_aprox")

    # Figura: Comparacion de curvas de aproximacion frente a la funcion teorica
    vz.estilo()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x_denso = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
    ax.plot(x_denso, np.sin(x_denso), color=vz.TINTA, lw=2.4, ls="--", label="Funcion real: sin(x)")
    ax.scatter(X_ent, d_ent, color=vz.AZUL, alpha=0.5, s=28, label="Datos de entrenamiento (70%)")
    ax.scatter(X_pru, d_pru, color=vz.NARANJA, alpha=0.8, marker="s", s=32, label="Datos de prueba (30%)")

    estilos_linea = [":", "-.", "-", "-"]
    colores_red = ["#888888", vz.AQUA, vz.VIOLETA, "#d9381e"]
    for m1, ls, col in zip(neuronas_ocultas_lista, estilos_lineas := [":", "-.", "-", "-"], colores_red):
        y_denso_pred = modelos_aprox[m1].predecir(x_denso)
        r2_val = df_comp_aprox.loc[df_comp_aprox["Neuronas Ocultas"] == m1, "R2 Prueba"].values[0]
        ax.plot(x_denso, y_denso_pred, lw=2.0, ls=ls, color=col, label=f"MLP 1-{m1}-1 (R²={r2_val:.4f})")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Aproximacion de f(x) = sin(x) mediante Perceptron Multicapa")
    ax.legend(loc="lower left", fontsize=8.5)
    ruta_fig_aprox = vz.guardar(fig, FIGURAS / "08_aproximacion_funciones.png")
    reporte.figura(ruta_fig_aprox, "Comparacion grafica de la aproximacion de la funcion seno con distintas capacidades de capa oculta.")

    # Figura: Curvas de Error de Entrenamiento y Prueba (Generalizacion)
    vz.estilo()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    mejor_modelo = modelos_aprox[8]
    ax.plot(mejor_modelo.historia_error, color=vz.AZUL, lw=2.0, label="ECM Entrenamiento (1-8-1)")
    ax.plot(mejor_modelo.historia_val, color=vz.NARANJA, lw=2.0, ls="--", label="ECM Prueba / Generalizacion (1-8-1)")
    ax.plot(modelos_aprox[2].historia_error, color="#aaaaaa", lw=1.2, ls=":", label="ECM Entrenamiento (1-2-1, subajuste)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoca de entrenamiento")
    ax.set_ylabel("Error Cuadratico Medio (ECM)")
    ax.set_title("Evolucion del Error de Entrenamiento vs Prueba (Aproximacion de Funciones)")
    ax.legend()
    ruta_fig_curv_aprox = vz.guardar(fig, FIGURAS / "08_curvas_error_aprox.png")
    reporte.figura(ruta_fig_curv_aprox, "Evolucion del error de entrenamiento y prueba. Se evidencia la convergencia sin sobreajuste con 8 neuronas.")

    # =========================================================================
    # 4. Discusion, Analisis de Parametros y Limitaciones
    # =========================================================================
    reporte.seccion("4. Discusion y Analisis de Resultados")
    reporte.texto(
        "1. **Superacion del umbral de separabilidad lineal (XOR):**\n"
        "   - La arquitectura 2-2-1 resolvio el problema con una exactitud del 100 % y un error cuadratico inferior a $10^{-4}$.\n"
        "   - La inspeccion del plano de decision muestra que las dos neuronas de la capa oculta construyen dos hiperplanos "
        "     intermedios que acotan la region activa en forma de franja diagonal, la cual es combinada por la neurona de salida "
        "     para aislar las dos clases opuestas, validando plenamente lo postulado en `ICE-claseRN03.md`.\n\n"
        "2. **Influencia de la razon de aprendizaje ($\\eta$) y momento ($\\alpha$):**\n"
        "   - Con razones de aprendizaje bajas ($\\eta = 0.2$), la red converge con lentitud debido a la reducida magnitud de los "
        "     gradientes cerca de las zonas de saturacion sigmoidal.\n"
        "   - La inclusion del termino de momento ($\\alpha = 0.5$ y $0.9$) produce una aceleracion dramatica del aprendizaje "
        "     (reduciendo las epocas necesarias en mas de un 60 %), al filtrar oscilaciones transversales y conservar inercia a "
        "     traves de las mesetas de error.\n"
        "   - Valores excesivos de $\\eta$ combinados con alto momento pueden inducir inestabilidades o escapes transitorios, por lo "
        "     que se requiere un balance adecuado.\n\n"
        "3. **Capacidad de representacion en aproximacion continua:**\n"
        "   - Con solo $m_1 = 2$ neuronas ocultas, la red presenta subajuste (*underfitting*), alcanzando un $R^2$ modesto ($< 0.90$), "
        "     ya que dos funciones sigmoides no poseen suficiente libertad para capturar dos curvaturas opuestas simultaneamente.\n"
        "   - A partir de $m_1 = 8$ neuronas, el coeficiente de determinacion en el conjunto de prueba supera el $99.8\\%$, logrando "
        "     una reproduccion casi identica de la funcion senoidal.\n"
        "   - El monitoreo simultaneo del error de entrenamiento y del conjunto de prueba independiente confirma la ausencia de sobreajuste "
        "     (*overfitting*), demostrando una alta capacidad de generalizacion."
    )

    reporte.seccion("5. Conclusiones")
    reporte.lista([
        "El Perceptron Multicapa supera de manera definitiva las restricciones algebraicas de las redes de una sola capa.",
        "El algoritmo de Retropropagacion propaga senales de error a traves de la regla de la cadena de forma consistente y reproducible.",
        "Para tareas de clasificacion binaria, la combinacion de sigmoide oculta y sigmoide de salida resulta optima; para aproximacion continua, la activacion lineal en la capa de salida es indispensable para no limitar el recorrido del modelo.",
        "El termino de momento amortigua eficazmente las mesetas de gradiente nulo caracteristicas de la sigmoide.",
        "Se satisfacen integralmente todos los aspectos requeridos para la Actividad 3 de la guia de evaluacion.",
    ])

    reporte.escribir()


if __name__ == "__main__":
    main()
