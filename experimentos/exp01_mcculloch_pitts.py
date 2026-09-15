"""
Experimento 01 --- Modelo de McCulloch-Pitts: compuertas logicas AND y OR.

Objetivos
---------
1. Reproducir numericamente las dos laminas de `claseRN02.md` que definen las
   compuertas AND y OR con neuronas binarias de umbral fijo.
2. Verificar que la red responde exactamente la tabla de verdad esperada.
3. Mostrar la frontera de decision que cada compuerta induce en el plano x1-x2
   y comprobar que ambas son *linealmente separables*.
4. Demostrar --- por busqueda exhaustiva, no por analogia --- que **ninguna**
   neurona MCP aislada puede calcular el XOR, y construir a continuacion una red
   de tres neuronas que si lo hace (universalidad de McCulloch-Pitts).

Ejecucion
---------
    python experimentos/exp01_mcculloch_pitts.py
"""

from __future__ import annotations

import _ruta  # noqa: F401  (configura sys.path)

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import visual as vz
from ce_rna.mcculloch_pitts import (NeuronaMCP, neurona_and, neurona_nand,
                                    neurona_nor, neurona_not, neurona_or,
                                    red_xor)
from ce_rna.reportes import Reporte, encabezado_experimento


def busqueda_exhaustiva_xor(rango_pesos=range(-3, 4), rango_umbrales=range(-3, 4)) -> pd.DataFrame:
    """Recorre todas las neuronas MCP de dos entradas con pesos y umbral enteros.

    Para cada combinacion (w1, w2, theta) se cuenta cuantos de los cuatro
    patrones del XOR resuelve correctamente.  Si ninguna combinacion alcanza 4
    aciertos, queda demostrado --- dentro de la rejilla explorada --- que el XOR
    no es representable por una sola neurona de umbral.

    El resultado no depende de la finura de la rejilla: la demostracion formal es
    que el XOR no es linealmente separable, y una neurona MCP solo puede trazar
    una frontera lineal.  La busqueda sirve como verificacion empirica de ese
    argumento.
    """
    X = ds.compuerta("XOR").X
    d = ds.compuerta("XOR").d
    filas = []
    for w1 in rango_pesos:
        for w2 in rango_pesos:
            for theta in rango_umbrales:
                neurona = NeuronaMCP(pesos=[w1, w2], umbral=theta)
                aciertos = int(np.sum(neurona.activar(X) == d))
                filas.append({"w1": w1, "w2": w2, "theta": theta, "aciertos": aciertos})
    return pd.DataFrame(filas)


def main() -> None:
    encabezado_experimento("Experimento 01 - McCulloch-Pitts: compuertas AND y OR")

    reporte = Reporte(
        titulo="Experimento 01 --- McCulloch-Pitts: compuertas logicas AND y OR",
        nombre_archivo="01_mcculloch_pitts",
        resumen=(
            "Primer modelo formal de neurona (McCulloch y Pitts, 1943). Las neuronas son "
            "binarias, los umbrales y las sinapsis **se mantienen fijos** --- no hay "
            "aprendizaje --- y la funcion de activacion es un escalon. El experimento "
            "verifica las compuertas AND y OR tal como se presentan en `claseRN02.md`, "
            "dibuja sus regiones de decision y demuestra por busqueda exhaustiva que una "
            "sola neurona de umbral no puede calcular el XOR, para despues construirlo "
            "componiendo tres neuronas."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Modelo
    # ------------------------------------------------------------------
    reporte.seccion("1. El modelo de neurona")
    reporte.texto(
        "El potencial de la neurona es la suma ponderada de sus entradas y la salida "
        "aplica la *ley de todo o nada* del impulso nervioso:"
    )
    reporte.formula(r"y^{(in)} = \sum_{i=1}^{n} w_i x_i \qquad "
                    r"y = \varphi\left(y^{(in)}\right) = "
                    r"\begin{cases} 1 & \text{si } y^{(in)} \geq \theta \\ "
                    r"0 & \text{si } y^{(in)} < \theta \end{cases}")
    reporte.lista([
        "Las neuronas son del tipo **binario**.",
        "Los umbrales y las sinapsis **se mantienen fijos**: los pesos se disenan, no se aprenden.",
        "La funcion de activacion es del tipo **escalon**.",
        "Pesos positivos = conexiones excitadoras; pesos negativos = conexiones inhibidoras.",
    ])

    # ------------------------------------------------------------------
    # 2. Compuertas AND y OR
    # ------------------------------------------------------------------
    reporte.seccion("2. Compuertas AND y OR")
    compuertas = {
        "AND": neurona_and(2),
        "OR": neurona_or(2),
        "NOT": neurona_not(),
        "NAND": neurona_nand(2),
        "NOR": neurona_nor(2),
    }

    for nombre in ("AND", "OR"):
        neurona = compuertas[nombre]
        tabla = neurona.tabla_verdad()
        esperado = ds.compuerta(nombre).d
        correcto = bool(np.all(tabla["y"].to_numpy() == esperado))
        print(f"\n{neurona.describir()}")
        print(tabla.to_string(index=False))
        print(f"  -> coincide con la tabla de verdad de {nombre}: {correcto}")

        reporte.seccion(f"2.{1 if nombre == 'AND' else 2} Compuerta {nombre}", nivel=3)
        reporte.texto(
            f"Pesos `w = {np.array2string(neurona.pesos, precision=0)}`, "
            f"umbral `θ = {neurona.umbral:g}`. Regla de disparo: "
            f"`{neurona.describir().split(': ', 1)[1]}`."
        )
        reporte.tabla(tabla, f"Tabla de verdad calculada por la neurona {nombre}",
                      nombre_csv=f"01_tabla_{nombre.lower()}")
        reporte.texto(
            f"Verificacion automatica frente a la tabla de verdad de referencia: "
            f"**{'CORRECTA' if correcto else 'INCORRECTA'}**."
        )

        fig, _ = vz.dibujar_red(
            [["x1", "x2"], [nombre]],
            titulo=f"Compuerta {nombre} de McCulloch-Pitts  (θ = {neurona.umbral:g})",
            pesos=[neurona.pesos.reshape(1, -1)],
            etiquetas_capas=["Capa de entrada", "Neurona de umbral"],
            nota=f"y = 1 si  w·x ≥ {neurona.umbral:g}",
        )
        ruta = vz.guardar(fig, FIGURAS / f"01_arquitectura_{nombre.lower()}.png")
        reporte.figura(ruta, f"Arquitectura de la compuerta {nombre}.")

        # Frontera de decision: w1 x1 + w2 x2 - theta = 0
        w_ext = np.concatenate(([-neurona.umbral], neurona.pesos))
        conjunto = ds.compuerta(nombre)
        fig, _ = vz.frontera_decision(
            w_ext, conjunto.X, conjunto.d,
            titulo=f"Region de decision de la compuerta {nombre}",
            etiquetas_clases=("y = 1", "y = 0"),
            margen=0.6,
        )
        ruta = vz.guardar(fig, FIGURAS / f"01_frontera_{nombre.lower()}.png")
        reporte.figura(
            ruta,
            f"La recta {neurona.pesos[0]:g}·x1 + {neurona.pesos[1]:g}·x2 = {neurona.umbral:g} "
            f"separa las entradas que producen 1 de las que producen 0.",
        )

    # Resto de compuertas elementales, en una sola tabla
    reporte.seccion("2.3 Otras compuertas elementales", nivel=3)
    filas = []
    for nombre in ("NOT", "NAND", "NOR"):
        neurona = compuertas[nombre]
        tabla = neurona.tabla_verdad()
        filas.append({
            "compuerta": nombre,
            "pesos": np.array2string(neurona.pesos, precision=0),
            "umbral": neurona.umbral,
            "salidas": "".join(str(int(v)) for v in tabla["y"]),
        })
    df_otras = pd.DataFrame(filas)
    print("\nOtras compuertas elementales:")
    print(df_otras.to_string(index=False))
    reporte.tabla(df_otras, "Compuertas NOT, NAND y NOR con neuronas MCP",
                  nombre_csv="01_otras_compuertas")
    reporte.texto(
        "La columna `salidas` lista la respuesta de la neurona para las entradas en orden "
        "(0,0), (0,1), (1,0), (1,1) --- o (0), (1) en el caso del NOT. "
        "NAND y NOR se obtienen con **conexiones inhibidoras**: pesos negativos."
    )

    # ------------------------------------------------------------------
    # 3. El XOR no es representable por una sola neurona
    # ------------------------------------------------------------------
    reporte.seccion("3. Limite del modelo: el XOR")
    df_busqueda = busqueda_exhaustiva_xor()
    maximo = int(df_busqueda["aciertos"].max())
    n_total = len(df_busqueda)
    n_maximo = int((df_busqueda["aciertos"] == maximo).sum())
    print(f"\nBusqueda exhaustiva sobre {n_total} neuronas MCP (pesos y umbral enteros en [-3, 3]):")
    print(f"  aciertos maximos sobre los 4 patrones del XOR = {maximo}")
    print(f"  numero de configuraciones que alcanzan ese maximo = {n_maximo}")

    reporte.texto(
        f"Se recorrieron **{n_total} neuronas** distintas (todas las combinaciones de pesos "
        f"enteros w1, w2 y umbral θ en el rango [-3, 3]) evaluando los cuatro patrones del XOR. "
        f"El maximo numero de aciertos alcanzado es **{maximo} de 4**, logrado por "
        f"{n_maximo} configuraciones: ninguna neurona de umbral resuelve el XOR."
    )
    reporte.texto(
        "La razon es geometrica y no depende de la rejilla explorada: una neurona de umbral "
        "traza una **unica frontera lineal**, y el XOR no es linealmente separable "
        "(`ICE-claseRN03.md`: *\"es imposible encontrar una linea recta que deje a un lado las "
        "entradas que deben producir 0, y al otro, las que deben producir 1\"*)."
    )
    resumen_busqueda = (
        df_busqueda.groupby("aciertos").size().reset_index(name="n_configuraciones")
    )
    reporte.tabla(resumen_busqueda, "Distribucion de aciertos en la busqueda exhaustiva",
                  nombre_csv="01_busqueda_xor")

    conjunto_xor = ds.compuerta("XOR")
    fig, _ = vz.frontera_decision(
        np.array([-1.0, 1.0, 1.0]), conjunto_xor.X, conjunto_xor.d,
        titulo="XOR: ninguna recta separa las clases",
        etiquetas_clases=("y = 1", "y = 0"), margen=0.6,
    )
    ruta = vz.guardar(fig, FIGURAS / "01_xor_no_separable.png")
    reporte.figura(ruta, "Los patrones del XOR alternan en diagonal: no hay recta que los separe.")

    # ------------------------------------------------------------------
    # 4. XOR por composicion
    # ------------------------------------------------------------------
    reporte.seccion("4. XOR por composicion de neuronas")
    red = red_xor()
    tabla_xor = red.tabla_verdad()
    correcto = bool(np.all(tabla_xor["y"].to_numpy() == conjunto_xor.d))
    print("\nRed XOR compuesta por tres neuronas MCP:")
    print(tabla_xor.to_string(index=False))
    print(f"  -> reproduce el XOR: {correcto}")

    reporte.texto(
        "McCulloch y Pitts demostraron que *\"todas las funciones logicas se pueden describir "
        "mediante combinaciones apropiadas de neuronas de este tipo\"*. El XOR se obtiene como"
    )
    reporte.formula(r"\text{XOR}(x_1,x_2) = \text{OR}(x_1,x_2) \;\wedge\; \neg\,\text{AND}(x_1,x_2)")
    reporte.texto(
        "La neurona de salida implementa ese *AND con una entrada inhibidora* con pesos "
        "(+1, −2) y umbral 1: la conexion inhibidora procedente de `h_and` cancela la "
        "excitacion de `h_or` cuando ambas entradas valen 1."
    )
    reporte.tabla(tabla_xor, "Activaciones internas y salida de la red XOR",
                  nombre_csv="01_tabla_xor_red")
    reporte.texto(f"Verificacion frente a la tabla de verdad del XOR: **{'CORRECTA' if correcto else 'INCORRECTA'}**.")

    fig, _ = vz.dibujar_red_mcp(red, titulo="Red de tres neuronas MCP que calcula el XOR")
    ruta = vz.guardar(fig, FIGURAS / "01_red_xor.png")
    reporte.figura(ruta, "Composicion OR / AND / AND-inhibido. Se indica el umbral θ bajo cada neurona.")

    # ------------------------------------------------------------------
    # 5. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("5. Conclusiones")
    reporte.lista([
        "Las compuertas AND (pesos 1,1; θ=2) y OR (pesos 2,2; θ=2) de `claseRN02.md` "
        "reproducen **exactamente** sus tablas de verdad.",
        "Ambas son linealmente separables: una sola recta basta para dividir el plano.",
        f"Ninguna de las {n_total} neuronas MCP exploradas resuelve el XOR "
        f"(maximo {maximo}/4 aciertos): el modelo de una unidad esta limitado a "
        "problemas linealmente separables.",
        "Componiendo tres neuronas si se obtiene el XOR, lo que ilustra el resultado de "
        "universalidad del articulo de 1943.",
        "El modelo carece por completo de aprendizaje: los pesos se calculan a mano. "
        "Ese es exactamente el vacio que llenan el perceptron (experimento 02) y el "
        "ADALINE (experimento 05).",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
