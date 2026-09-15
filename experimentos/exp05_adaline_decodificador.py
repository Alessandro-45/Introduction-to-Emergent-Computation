"""
Experimento 05 --- ADALINE y la regla Delta: descodificador de binario a decimal.

Resuelve el ejercicio propuesto en `ICE-claseRN04.md`:

    "Compruebe si una red de tipo ADALINE es capaz de aproximar la expresion por
     si sola, a partir de un conjunto de ejemplos de entrenamiento.
     - Elija un conjunto de patrones de entrenamiento de dimension 3.
     - ... el numero maximo de entradas que se pueden generar es de 2^3 = 8.
     - Inicializar los pesos de manera aleatoria de 0 a 1.
     - Refleje en pantalla de manera secuencial el numero de iteracion, pesos y
       el error por patron.
     - Comentar los valores optimos de los pesos."

Contenido
---------
1. El problema y su solucion analitica.
2. Arquitectura: neurona **lineal** (sin funcion umbral en el aprendizaje).
3. Traza secuencial: iteracion, pesos y error por patron.
4. Curva de error cuadratico medio y convergencia a los pesos optimos.
5. La superficie de error: el paraboloide por el que desciende el gradiente.
6. Efecto de la razon de aprendizaje, incluida la divergencia.
7. Modo estocastico frente a modo por lotes.
8. Generalizacion a 4 y 5 bits.

Ejecucion
---------
    python experimentos/exp05_adaline_decodificador.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.adaline import Adaline, solucion_minimos_cuadrados, superficie_error
from ce_rna.reportes import Reporte, encabezado_experimento

GAMMA = 0.05
SEMILLA = 1


def main() -> None:
    encabezado_experimento("Experimento 05 - ADALINE: descodificador binario a decimal")

    reporte = Reporte(
        titulo="Experimento 05 --- ADALINE y la regla Delta: descodificador de binario a decimal",
        nombre_archivo="05_adaline_decodificador",
        resumen=(
            "Widrow y Hoff (1960) disenaron un sistema de aprendizaje *que si tiene en cuenta el "
            "error producido*. La diferencia con el perceptron es de una sola linea de codigo y "
            "cambia todo: la regla Delta aprende sobre la **salida lineal**, sin pasarla por la "
            "funcion umbral, de modo que el error es una magnitud real y derivable y el "
            "aprendizaje se convierte en un **descenso del gradiente** sobre el error cuadratico "
            "medio. El experimento resuelve el descodificador binario-decimal propuesto en la "
            "clase y verifica que la red recupera los pesos optimos con error practicamente nulo."
        ),
    )

    conjunto = ds.decodificador_binario(3)
    optimo = ds.pesos_optimos_decodificador(3)
    minimos_cuadrados = solucion_minimos_cuadrados(conjunto.X, conjunto.d)

    # ------------------------------------------------------------------
    # 1. El problema
    # ------------------------------------------------------------------
    reporte.seccion("1. El problema y su solucion analitica")
    reporte.tabla(conjunto.tabla(), "Conjunto de entrenamiento: los 8 patrones de 3 bits",
                  nombre_csv="05_conjunto")
    reporte.texto(
        "El descodificador recibe una entrada en binario y produce como salida su valor decimal. "
        "Existe una expresion analitica exacta, lo que convierte al problema en un banco de "
        "pruebas ideal: se sabe de antemano cual es la respuesta correcta y se puede medir "
        "exactamente cuanto se aproxima la red."
    )
    reporte.formula(r"d = \sum_{i=1}^{n} 2^{\,i-1} x_i \;\Longrightarrow\; "
                    r"w^{*} = (w_0, w_1, w_2, w_3) = (0,\; 4,\; 2,\; 1)")
    reporte.texto(
        "La salida deseada es una **funcion lineal** de las entradas, que es justamente la clase "
        "de problemas que un ADALINE puede resolver de forma exacta: *\"solo resuelven problemas "
        "en los que los ejemplos son linealmente separables en terminos de clasificacion, o en "
        "los que las salidas son funciones lineales de las entradas. El caso del descodificador "
        "es un ejemplo\"* (`ICE-claseRN04.md`)."
    )
    reporte.texto(
        "Nota sobre el enunciado: la formula de la clase, d = Σ 2^i·x_i con i de 1 a n, daria "
        "pesos (2, 4, 8). Aqui se usa la convencion estandar d = Σ 2^(i−1)·x_i, con la que el "
        "rango de salida es 0..7. El cambio solo afecta la escala de los pesos optimos, no la "
        "naturaleza del problema."
    )

    fig, _ = vz.dibujar_red(
        [["x0=1", "b2", "b1", "b0"], ["y"]],
        titulo="ADALINE: combinador lineal adaptativo de 3 entradas",
        pesos=[optimo.reshape(1, -1)],
        capas_lineales={0, 1},
        etiquetas_capas=["Capa 0: bits de entrada + sesgo", "Capa 1: neurona lineal"],
        nota="La neurona de salida es LINEAL (cuadrado): y = w·x, sin funcion umbral.",
    )
    ruta = vz.guardar(fig, FIGURAS / "05_arquitectura_adaline.png")
    reporte.figura(ruta, "Arquitectura con los pesos optimos teoricos (4, 2, 1) y sesgo 0.")

    # ------------------------------------------------------------------
    # 2. La regla Delta
    # ------------------------------------------------------------------
    reporte.seccion("2. La regla Delta")
    reporte.texto("El error que se minimiza es el cuadratico medio sobre todos los patrones:")
    reporte.formula(r"E = \frac{1}{N}\sum_{p} E^{p}, \qquad E^{p} = \tfrac{1}{2}\left(d^{p} - y^{p}\right)^{2}")
    reporte.texto("El cambio de cada peso es proporcional a la derivada del error respecto a ese peso:")
    reporte.formula(r"\Delta^{p} w_j = -\gamma \frac{\partial E^{p}}{\partial w_j} = "
                    r"-\gamma \frac{\partial E^{p}}{\partial y^{p}}\frac{\partial y^{p}}{\partial w_j} "
                    r"= \gamma\left(d^{p} - y^{p}\right) x_j")
    reporte.texto(
        "porque al ser una unidad lineal ∂y/∂w_j = x_j y ∂E/∂y = −(d − y). **La regla Delta es "
        "una extension de la regla del perceptron a valores de salida reales.** Las cuatro "
        "diferencias que enumera la clase:"
    )
    reporte.lista([
        "En el PERCEPTRON la salida es binaria; en el ADALINE es real.",
        "En el PERCEPTRON la diferencia entre salida deseada y obtenida es 0 o ±1; en el ADALINE "
        "se calcula la diferencia real.",
        "En el ADALINE existe una **medida de cuanto** se ha equivocado la red; en el PERCEPTRON "
        "solo se determina **si** se ha equivocado.",
        "En el ADALINE hay una razon de aprendizaje γ ∈ (0, 1) que regula cuanto afecta cada "
        "equivocacion a la modificacion de los pesos.",
    ])

    # ------------------------------------------------------------------
    # 3. Traza secuencial
    # ------------------------------------------------------------------
    reporte.seccion("3. Traza secuencial: iteracion, pesos y error por patron")
    modelo = Adaline(3, razon_aprendizaje=GAMMA, modo="estocastico", max_epocas=1000,
                     tolerancia=1e-9, inicializacion="aleatoria", semilla=SEMILLA)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_traza=True)
    traza = modelo.traza_df()

    print("\nPrimeras 16 presentaciones (2 epocas):")
    print(traza.head(16).to_string(index=False))
    print("\nUltimas 8 presentaciones:")
    print(traza.tail(8).to_string(index=False))
    print("\n" + modelo.resumen())

    reporte.texto(
        f"Pesos iniciales aleatorios en [0, 1] (semilla {SEMILLA}), γ = {GAMMA}, modo estocastico "
        "(los pesos se actualizan despues de **cada** patron)."
    )
    reporte.tabla(traza.head(16), "Primeras 16 presentaciones de patron (dos epocas completas)",
                  nombre_csv="05_traza_inicio")
    reporte.tabla(traza.tail(8), "Ultimas 8 presentaciones: el error ya es despreciable",
                  nombre_csv="05_traza_final")
    reporte.bloque(modelo.resumen())

    # ------------------------------------------------------------------
    # 4. Convergencia
    # ------------------------------------------------------------------
    reporte.seccion("4. Convergencia y valores optimos de los pesos")
    historial = modelo.historial_df()
    fig, _ = vz.curva_aprendizaje(
        historial["ecm"].to_numpy(),
        titulo="Descenso del error cuadratico medio",
        etiqueta_y="ECM  E(w)",
        escala_log=True,
    )
    ruta = vz.guardar(fig, FIGURAS / "05_curva_ecm.png")
    reporte.figura(ruta, "El ECM cae varios ordenes de magnitud: el descenso del gradiente "
                         "converge al minimo global del paraboloide de error.")

    fig, _ = vz.curvas_comparadas(
        {f"w{i}": np.array([h.w[i] for h in modelo.historial]) for i in range(1, 4)},
        titulo="Convergencia de los pesos a sus valores optimos (4, 2, 1)",
        etiqueta_y="valor del peso",
        escala_log=False,
    )
    ruta = vz.guardar(fig, FIGURAS / "05_convergencia_pesos.png")
    reporte.figura(ruta, "Los tres pesos convergen a 4, 2 y 1: la red ha descubierto por si sola "
                         "el valor posicional de cada bit.")

    df_pesos = pd.DataFrame({
        "peso": ["w0 (sesgo)", "w1 (bit 2)", "w2 (bit 1)", "w3 (bit 0)"],
        "optimo_teorico": optimo,
        "minimos_cuadrados": minimos_cuadrados,
        "adaline": modelo.w,
        "error_absoluto": np.abs(modelo.w - optimo),
    })
    print("\nPesos finales frente a los optimos:")
    print(df_pesos.to_string(index=False))
    reporte.tabla(df_pesos, "Pesos aprendidos frente a los valores optimos",
                  nombre_csv="05_pesos_finales")
    reporte.texto(
        f"**Comentario sobre los valores optimos** (consigna de la clase): la red converge a "
        f"w = ({', '.join(f'{v:.4f}' for v in modelo.w)}), que coincide con el optimo teorico "
        f"(0, 4, 2, 1) con un error maximo de {np.max(np.abs(modelo.w - optimo)):.2e}. Cada peso "
        "ha adoptado exactamente el **valor posicional** de su bit --- 4, 2 y 1 --- y el sesgo se "
        "ha anulado porque la funcion no tiene termino independiente. Es un caso poco habitual en "
        "el que los pesos de una red neuronal tienen una interpretacion semantica directa e "
        "inequivoca."
    )

    y = modelo.predecir(conjunto.X)
    df_pred = conjunto.tabla()
    df_pred["y_adaline"] = y
    df_pred["error"] = conjunto.d - y
    reporte.tabla(df_pred, "Salida de la red para los 8 patrones", nombre_csv="05_predicciones")
    reporte.lista([
        f"Error cuadratico medio final: **{modelo.ecm(conjunto.X, conjunto.d):.3e}**",
        f"RMSE: **{mt.rmse(y, conjunto.d):.3e}**",
        f"Coeficiente de determinacion R²: **{mt.r2(y, conjunto.d):.10f}**",
        f"Epocas necesarias: **{modelo.epocas_usadas}**",
    ])

    # ------------------------------------------------------------------
    # 5. Superficie de error
    # ------------------------------------------------------------------
    reporte.seccion("5. La superficie de error: por donde desciende el gradiente")
    Wi, Wj, E = superficie_error(conjunto.X, conjunto.d, indices=(1, 2),
                                 w_base=minimos_cuadrados, rango=4.5, n_puntos=90)
    fig, _ = vz.contorno_error(
        Wi, Wj, E, trayectoria=modelo.trayectoria_pesos, indices=(1, 2), optimo=minimos_cuadrados,
        titulo="Curvas de nivel de E(w1, w2) y trayectoria del aprendizaje",
    )
    ruta = vz.guardar(fig, FIGURAS / "05_contorno_error.png")
    reporte.figura(ruta, "Cortando la superficie de error en el plano (w1, w2) --- los demas pesos "
                         "fijados en su valor optimo --- las curvas de nivel son elipses "
                         "concentricas. La trayectoria desciende hasta el minimo global.")

    fig, _ = vz.superficie_error_3d(Wi, Wj, E, indices=(1, 2),
                                    titulo="Paraboloide de error del ADALINE")
    ruta = vz.guardar(fig, FIGURAS / "05_superficie_error.png")
    reporte.figura(ruta, "La superficie de error de una unidad lineal es un paraboloide convexo: "
                         "tiene un unico minimo y ningun minimo local donde quedar atrapado.")
    reporte.texto(
        "Esta es la ventaja estructural del ADALINE frente al perceptron: **existe una funcion "
        "objetivo**. El perceptron solo sabe si acierta o falla y se detiene en cualquier solucion "
        "consistente; el ADALINE mide cuanto se equivoca y siempre se dirige al mismo punto, el "
        "minimo global, con independencia de donde arranque."
    )

    # ------------------------------------------------------------------
    # 6. Razon de aprendizaje
    # ------------------------------------------------------------------
    reporte.seccion("6. Efecto de la razon de aprendizaje")
    cota = Adaline(3).cota_estabilidad(conjunto.X)
    filas = []
    series_ecm = {}
    for gamma in (0.005, 0.02, 0.05, 0.2, 0.5, 1.0):
        m = Adaline(3, razon_aprendizaje=gamma, modo="estocastico", max_epocas=400,
                    tolerancia=1e-9, inicializacion="aleatoria", semilla=SEMILLA)
        m.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
        ecm_final = m.ecm(conjunto.X, conjunto.d) if not m.diverged else float("inf")
        filas.append({
            "gamma": gamma,
            "epocas": m.epocas_usadas,
            "estado": "diverge" if m.diverged else ("converge" if m.convergio else "no converge (limite)"),
            "ecm_final": ecm_final,
            "error_max_pesos": float(np.max(np.abs(m.w - optimo))) if np.all(np.isfinite(m.w)) else float("inf"),
        })
        if gamma in (0.005, 0.05, 0.5):
            series_ecm[f"γ = {gamma}"] = np.array([h.ecm for h in m.historial])
    df_gamma = pd.DataFrame(filas)
    print(f"\nCota teorica de estabilidad (modo por lotes): gamma < 2/lambda_max = {cota:.4f}")
    print(df_gamma.to_string(index=False))

    fig, _ = vz.curvas_comparadas(
        series_ecm,
        titulo="El ECM segun la razon de aprendizaje",
        etiqueta_y="ECM  E(w)",
        escala_log=True,
    )
    ruta = vz.guardar(fig, FIGURAS / "05_gamma_comparacion.png")
    reporte.figura(ruta, "γ pequeno: descenso lento pero seguro. γ grande: descenso rapido. "
                         "Por encima de la cota de estabilidad, el error crece sin limite.")
    reporte.tabla(df_gamma, "Razon de aprendizaje frente a convergencia", nombre_csv="05_gamma")
    reporte.texto(
        f"La cota teorica de estabilidad para el modo por lotes es γ < 2/λ_max = **{cota:.4f}**, "
        "donde λ_max es el mayor autovalor de la matriz de correlacion de las entradas. Por "
        "debajo de ella el descenso del gradiente es una contraccion y el error decrece "
        "monotonamente; por encima, cada paso sobrepasa el minimo con amplitud creciente y el "
        "aprendizaje **diverge**. Es la diferencia mas importante en la practica frente al "
        "perceptron, cuya convergencia no depende del valor de a."
    )

    # ------------------------------------------------------------------
    # 7. Estocastico frente a lotes
    # ------------------------------------------------------------------
    reporte.seccion("7. Modo estocastico frente a modo por lotes")
    filas = []
    series_modo = {}
    for modo in ("estocastico", "lote"):
        m = Adaline(3, razon_aprendizaje=0.05, modo=modo, max_epocas=3000, tolerancia=1e-9,
                    inicializacion="aleatoria", semilla=SEMILLA)
        m.entrenar(conjunto.X, conjunto.d, registrar_pasos=False)
        filas.append({
            "modo": modo,
            "epocas": m.epocas_usadas,
            "actualizaciones_de_pesos": m.epocas_usadas * (conjunto.n_patrones if modo == "estocastico" else 1),
            "ecm_final": m.ecm(conjunto.X, conjunto.d),
            "error_max_pesos": float(np.max(np.abs(m.w - optimo))),
        })
        series_modo[modo] = np.array([h.ecm for h in m.historial])
    df_modo = pd.DataFrame(filas)
    print("\nModo estocastico frente a modo por lotes:")
    print(df_modo.to_string(index=False))

    fig, _ = vz.curvas_comparadas(
        series_modo, titulo="Convergencia por epoca: estocastico frente a lotes",
        etiqueta_y="ECM  E(w)", escala_log=True,
    )
    ruta = vz.guardar(fig, FIGURAS / "05_modo_comparacion.png")
    reporte.figura(ruta, "Por epoca, el modo estocastico avanza mucho mas rapido porque actualiza "
                         "los pesos 8 veces (una por patron) en lugar de una sola.")
    reporte.tabla(df_modo, "Comparacion de los dos regimenes de actualizacion",
                  nombre_csv="05_modos")
    reporte.texto(
        "El modo por lotes calcula el gradiente exacto de E y da un unico paso por epoca; el "
        "estocastico aproxima ese gradiente patron a patron, lo que introduce ruido pero produce "
        "N veces mas correcciones. El procedimiento descrito en la clase --- *\"introducir un "
        "patron de entrada... si no se ha cumplido el criterio de convergencia, regresar a 2\"* --- "
        "es el estocastico, que es tambien el que se usa en el resto del experimento."
    )

    # ------------------------------------------------------------------
    # 8. Generalizacion a mas bits
    # ------------------------------------------------------------------
    reporte.seccion("8. Generalizacion: 4 y 5 bits")
    filas = []
    for n_bits in (3, 4, 5):
        c = ds.decodificador_binario(n_bits)
        opt = ds.pesos_optimos_decodificador(n_bits)
        m = Adaline(n_bits, razon_aprendizaje=0.02, modo="estocastico", max_epocas=5000,
                    tolerancia=1e-9, inicializacion="aleatoria", semilla=SEMILLA)
        m.entrenar(c.X, c.d, registrar_pasos=False)
        filas.append({
            "bits": n_bits,
            "patrones": c.n_patrones,
            "epocas": m.epocas_usadas,
            "ecm_final": m.ecm(c.X, c.d),
            "R2": mt.r2(m.predecir(c.X), c.d),
            "pesos_aprendidos": np.array2string(m.w, precision=3, suppress_small=True),
            "pesos_optimos": np.array2string(opt, precision=0),
        })
    df_bits = pd.DataFrame(filas)
    print("\nGeneralizacion a mas bits:")
    print(df_bits.to_string(index=False))
    reporte.tabla(df_bits, "El mismo modelo con 3, 4 y 5 bits", nombre_csv="05_generalizacion_bits")
    reporte.texto(
        "El resultado se mantiene al crecer la dimension: la red recupera siempre los pesos "
        "posicionales 2^(n−1), ..., 2, 1 con R² = 1. Conviene notar que el numero de **epocas** "
        "*disminuye* al anadir bits, lo que a primera vista sorprende. La razon es que una epoca "
        "no es una cantidad fija de trabajo: con n bits hay 2^n patrones, de modo que el modo "
        "estocastico realiza 2^n correcciones por epoca. Contadas en actualizaciones de pesos "
        "--- que es la unidad de coste real --- el problema de 5 bits necesita mas trabajo, no "
        "menos."
    )

    # ------------------------------------------------------------------
    # 9. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("9. Conclusiones")
    reporte.lista([
        f"El ADALINE aprende el descodificador **de forma exacta**: ECM = "
        f"{modelo.ecm(conjunto.X, conjunto.d):.2e} y R² = {mt.r2(y, conjunto.d):.8f}.",
        "Los pesos convergen a (0, 4, 2, 1), es decir, a los valores posicionales de los bits: "
        "la red descubre sola la estructura del problema.",
        "La superficie de error es un paraboloide convexo con un unico minimo; el descenso del "
        "gradiente llega siempre al mismo punto, con independencia de la inicializacion.",
        f"La razon de aprendizaje es el parametro critico: por encima de γ ≈ {cota:.2f} el "
        "aprendizaje diverge. El perceptron no tiene este problema, pero tampoco tiene una "
        "funcion objetivo que minimizar.",
        "El modelo hereda la limitacion del perceptron --- solo resuelve problemas lineales --- "
        "pero dentro de ese ambito es cualitativamente superior: mide el error, lo minimiza y "
        "produce salidas reales en lugar de binarias.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
