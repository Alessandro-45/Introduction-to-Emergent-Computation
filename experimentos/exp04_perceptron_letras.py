"""
Experimento 04 --- Perceptron multiclase: reconocimiento de las letras X y O.

Reproduce la figura *"Funcionamiento de un Perceptron"* de `ICE-claseRN03.md`:
una red unicapa con m0 = 25 entradas (una retina de 5x5 pixeles) y m1 = 2
neuronas de salida, donde "cada neurona de salida representa a una clase
determinada; si una de ellas se activa con una entrada, significa que pertenece
a esa clase, si esta desactivada, que no pertenece".

Contenido
---------
1. Los prototipos de las letras y la arquitectura de la red.
2. Entrenamiento con los dos prototipos limpios.
3. Los pesos aprendidos vistos como plantillas de 5x5 (que "ve" cada neurona).
4. Tolerancia a estimulos contaminados: exactitud frente al numero de pixeles
   invertidos, promediada sobre cientos de ensayos aleatorios.
5. Efecto de entrenar con ejemplos ruidosos (capacidad de generalizacion).
6. Comparacion de las dos lecturas posibles del paso 4 del algoritmo.
7. Matriz de confusion y casos de indeterminacion.

Ejecucion
---------
    python experimentos/exp04_perceptron_letras.py
"""

from __future__ import annotations

import _ruta  # noqa: F401

import numpy as np
import pandas as pd

from _ruta import FIGURAS
from ce_rna import datasets as ds
from ce_rna import metricas as mt
from ce_rna import visual as vz
from ce_rna.perceptron import PerceptronSimple
from ce_rna.reportes import Reporte, encabezado_experimento

CLASES = ("X", "O")


def evaluar_ruido(modelo, n_pixeles, n_ensayos=400, semilla=0):
    """Exactitud media al invertir `n_pixeles` pixeles de cada prototipo."""
    rng = np.random.default_rng(semilla)
    aciertos = 0
    total = 0
    for _ in range(n_ensayos):
        for indice, nombre in enumerate(CLASES):
            patron = ds.contaminar(ds.letra(nombre), n_pixeles, semilla=int(rng.integers(1 << 30)))
            prediccion = int(modelo.clasificar(patron.reshape(1, -1))[0])
            aciertos += int(prediccion == indice)
            total += 1
    return aciertos / total


def main() -> None:
    encabezado_experimento("Experimento 04 - Perceptron: reconocimiento de letras X y O")

    reporte = Reporte(
        titulo="Experimento 04 --- Perceptron multiclase: reconocimiento de las letras X y O",
        nombre_archivo="04_perceptron_letras",
        resumen=(
            "Una red unicapa con 25 entradas y **dos** neuronas de salida, una por clase. Es el "
            "mismo algoritmo del experimento 02 aplicado a un problema de reconocimiento de "
            "patrones reales, y sirve para comprobar dos propiedades que `claseRN01.md` atribuye "
            "a las redes neuronales: que la memoria queda almacenada *en el patron de pesos* y "
            "que la red tolera *estimulos incompletos, ruidosos o parcialmente erroneos*."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Los patrones y la arquitectura
    # ------------------------------------------------------------------
    reporte.seccion("1. Los prototipos y la arquitectura")
    fig, _ = vz.rejilla_retinas(
        [ds.letra("X"), ds.letra("O")],
        ["Prototipo X", "Prototipo O"],
        n_columnas=2,
        sup_titulo="Retinas de 5x5 pixeles (azul = +1, claro = −1)",
    )
    ruta = vz.guardar(fig, FIGURAS / "04_prototipos.png")
    reporte.figura(ruta, "Los dos patrones de entrenamiento, codificados como vectores bipolares de 25 componentes.")

    fig, _ = vz.dibujar_red(
        [["x0=1", "x1", "x2", "⋮", "x25"], ["X", "O"]],
        titulo="Red unicapa: 25 entradas (retina 5x5) + inclinacion, 2 neuronas de salida",
        etiquetas_capas=["Capa 0: retina", "Capa 1: una neurona por clase"],
        nota="Esquema: se dibujan solo algunas de las 25 entradas. Matriz de sinapsis W de 2 x 26.",
    )
    ruta = vz.guardar(fig, FIGURAS / "04_arquitectura.png")
    reporte.figura(ruta, "Arquitectura del clasificador. La neurona 'X' responde +1 ante una X y −1 "
                         "en cualquier otro caso; la neurona 'O' hace lo propio con la O.")
    reporte.texto(
        "El objetivo de entrenamiento de cada patron es un **vector**: la X se codifica como "
        "d = (+1, −1) y la O como d = (−1, +1). Es un esquema *uno contra el resto*: cada neurona "
        "de salida es un perceptron independiente que aprende su propia frontera en el espacio de "
        "25 dimensiones."
    )

    # ------------------------------------------------------------------
    # 2. Entrenamiento con los prototipos limpios
    # ------------------------------------------------------------------
    reporte.seccion("2. Entrenamiento con los prototipos limpios")
    conjunto = ds.letras_xo(con_ruido=0)
    modelo = PerceptronSimple(25, 2, razon_aprendizaje=1.0, max_epocas=100)
    modelo.entrenar(conjunto.X, conjunto.d, registrar_traza=True)
    print("\n" + modelo.resumen())

    reporte.texto(
        f"Pesos iniciales nulos, a = 1. El algoritmo **converge en {modelo.epocas_usadas} epocas** "
        f"con {len(modelo.trayectoria_pesos) - 1} correcciones y clasifica correctamente los dos "
        "prototipos. Con solo dos patrones el problema es trivialmente separable: en un espacio de "
        "25 dimensiones, dos puntos distintos siempre pueden separarse por un hiperplano."
    )
    reporte.tabla(modelo.historial_df(), "Historial de entrenamiento", nombre_csv="04_historial_limpio")

    # ------------------------------------------------------------------
    # 3. Los pesos como plantilla
    # ------------------------------------------------------------------
    reporte.seccion("3. Que aprende cada neurona: los pesos como plantilla")
    fig, _ = vz.rejilla_retinas(
        [modelo.W[0, 1:], modelo.W[1, 1:]],
        [f"Neurona X  (w0 = {modelo.W[0,0]:.0f})",
         f"Neurona O  (w0 = {modelo.W[1,0]:.0f})"],
        n_columnas=2, divergente=True,
        sup_titulo="Matriz de sinapsis aprendida, vista como imagen de 5x5",
    )
    ruta = vz.guardar(fig, FIGURAS / "04_pesos_plantilla.png")
    reporte.figura(ruta, "Azul = peso positivo, naranja = peso negativo. Cada neurona ha memorizado "
                         "una plantilla de 5x5 con la que compara toda entrada que reciba.")

    X_ref, O_ref = ds.letra("X"), ds.letra("O")
    identidades = {
        "plantilla de la X": X_ref,
        "plantilla de la O": O_ref,
        "anti-plantilla de la X (−X)": -X_ref,
        "anti-plantilla de la O (−O)": -O_ref,
    }
    filas_id = []
    for j, nombre in enumerate(CLASES):
        w = modelo.W[j, 1:]
        coincidencia = next((k for k, v in identidades.items() if np.array_equal(w, v)), "ninguna exacta")
        filas_id.append({
            "neurona": nombre,
            "los pesos coinciden con": coincidencia,
            "w · x(X)": float(w @ X_ref),
            "w · x(O)": float(w @ O_ref),
        })
    df_identidad = pd.DataFrame(filas_id)
    print("\nIdentidad de los pesos aprendidos:")
    print(df_identidad.to_string(index=False))
    reporte.tabla(df_identidad, "Que plantilla almacena exactamente cada neurona",
                  nombre_csv="04_identidad_pesos")

    reporte.texto(
        "El resultado ilustra de forma muy directa la afirmacion de `claseRN01.md`: *\"las memorias "
        "se almacenan o representan en el patron de pesos de las interconexiones\"*. El potencial "
        "y_in = w·x no es otra cosa que la **correlacion** entre la entrada y la plantilla "
        "almacenada: la neurona dispara cuando la imagen se parece a lo que recuerda."
    )
    reporte.texto(
        "Ahora bien, la plantilla que aprende cada neurona **no es su propia letra**, sino la "
        "*anti-plantilla de su rival*: la neurona X almacena −O y la neurona O almacena −X. No es "
        "un error, es consecuencia del algoritmo. Partiendo de pesos nulos, la primera "
        "presentacion (la X) deja y_in = 0 → +1 en ambas neuronas; solo la neurona O se equivoca, "
        "y su correccion es w ← w + a·d·x = −x(X). Al presentar la O ocurre lo simetrico. Como "
        "solo hay dos clases y las dos plantillas son casi opuestas, −O funciona como "
        "discriminador de la X igual de bien que la propia X: la tabla anterior muestra que la "
        "neurona X responde con potencial +17 ante una X y −25 ante una O."
    )
    reporte.texto(
        "Este es un buen ejemplo de una leccion general: una red neuronal no aprende "
        "*representaciones bonitas*, aprende **lo primero que resuelve el problema**. Interpretar "
        "los pesos exige comprobar que se esta leyendo lo que realmente hay, no lo que se espera "
        "encontrar."
    )

    # ------------------------------------------------------------------
    # 4. Tolerancia al ruido
    # ------------------------------------------------------------------
    reporte.seccion("4. Tolerancia a estimulos contaminados")
    niveles = list(range(0, 13))
    exactitud_limpio = [evaluar_ruido(modelo, k, n_ensayos=400, semilla=100 + k) for k in niveles]

    conjunto_ruidoso = ds.letras_xo(con_ruido=15, prob_ruido=0.12, semilla=5)
    modelo_ruidoso = PerceptronSimple(25, 2, razon_aprendizaje=0.5, theta=0.5, max_epocas=300)
    modelo_ruidoso.entrenar(conjunto_ruidoso.X, conjunto_ruidoso.d)
    exactitud_ruidoso = [evaluar_ruido(modelo_ruidoso, k, n_ensayos=400, semilla=100 + k) for k in niveles]

    print(f"\nEntrenado con ruido: convergio={modelo_ruidoso.convergio} en "
          f"{modelo_ruidoso.epocas_usadas} epocas sobre {conjunto_ruidoso.n_patrones} patrones")

    df_ruido = pd.DataFrame({
        "pixeles_invertidos": niveles,
        "porcentaje_retina": [round(100 * k / 25, 1) for k in niveles],
        "exactitud_entrenado_limpio": exactitud_limpio,
        "exactitud_entrenado_con_ruido": exactitud_ruidoso,
    })
    print("\nTolerancia al ruido (400 ensayos por nivel y por letra):")
    print(df_ruido.to_string(index=False))

    fig, _ = vz.curvas_comparadas(
        {
            "entrenado solo con prototipos limpios": np.array(exactitud_limpio),
            "entrenado con 15 copias ruidosas por letra": np.array(exactitud_ruidoso),
        },
        titulo="Degradacion de la exactitud al contaminar la retina",
        etiqueta_y="exactitud",
        etiqueta_x="pixeles invertidos (de 25)",
        escala_log=False,
        x=niveles,
    )
    ruta = vz.guardar(fig, FIGURAS / "04_tolerancia_ruido.png")
    reporte.figura(ruta, "Exactitud media sobre 400 ensayos aleatorios por nivel de ruido y por letra.")
    reporte.tabla(df_ruido, "Exactitud frente al numero de pixeles invertidos",
                  nombre_csv="04_tolerancia_ruido")
    reporte.texto(
        "La red mantiene el 100 % de aciertos hasta un nivel de contaminacion considerable y a "
        "partir de ahi degrada **suavemente**, sin colapsar de golpe. Esto es la *degradacion "
        "elegante* caracteristica de las redes neuronales: como la decision depende de la "
        "correlacion global entre la entrada y la plantilla, ningun pixel individual es critico. "
        "En 25 pixeles, invertir 12 o 13 equivale a destruir la mitad de la imagen, momento en el "
        "que el patron deja de parecerse mas a su prototipo que a su contrario."
    )
    reporte.texto(
        "**Un resultado contraintuitivo que conviene subrayar**: entrenar con copias ruidosas "
        "*no mejora* la tolerancia, sino que la empeora. La explicacion esta en la naturaleza del "
        "algoritmo. Entrenado solo con los prototipos limpios y pesos iniciales nulos, la unica "
        "correccion que hace la regla perceptronica deja `w = d·x`, es decir, **exactamente la "
        "plantilla de la letra**: el filtro adaptado, que es la solucion optima para este "
        "problema. Al anadir patrones ruidosos, el perceptron se detiene en la primera frontera "
        "que separa ese conjunto concreto --- no en la mejor --- y esa frontera esta peor alineada "
        "con los prototipos. El perceptron **no optimiza nada**: solo busca una solucion "
        "consistente. Esta es, de nuevo, la carencia que motiva el ADALINE."
    )

    # Ejemplos visuales
    ejemplos, titulos = [], []
    for nombre in CLASES:
        for k in (0, 3, 6, 9):
            patron = ds.contaminar(ds.letra(nombre), k, semilla=42 + k)
            prediccion = CLASES[int(modelo.clasificar(patron.reshape(1, -1))[0])]
            ejemplos.append(patron)
            titulos.append(f"{nombre} con {k} fallos → '{prediccion}'")
    fig, _ = vz.rejilla_retinas(ejemplos, titulos, n_columnas=4,
                                sup_titulo="Reconocimiento de patrones contaminados")
    ruta = vz.guardar(fig, FIGURAS / "04_ejemplos_ruido.png")
    reporte.figura(ruta, "Ejemplos concretos de entradas degradadas y la clase que la red les asigna.")

    # ------------------------------------------------------------------
    # 5. Dos lecturas del paso 4
    # ------------------------------------------------------------------
    reporte.seccion("5. Las dos lecturas del paso 4 del algoritmo")
    filas = []
    for literal in (False, True):
        m = PerceptronSimple(25, 2, razon_aprendizaje=1.0, max_epocas=100,
                             actualizar_todas_las_salidas=literal)
        m.entrenar(conjunto_ruidoso.X, conjunto_ruidoso.d)
        filas.append({
            "variante": "literal: corregir todas las salidas" if literal
                        else "estandar: corregir solo la que falla",
            "epocas": m.epocas_usadas,
            "correcciones": len(m.trayectoria_pesos) - 1,
            "convergio": "si" if m.convergio else "no",
            "exactitud": mt.exactitud(m.predecir(conjunto_ruidoso.X), conjunto_ruidoso.d),
        })
    df_variantes = pd.DataFrame(filas)
    print("\nVariantes del paso 4:")
    print(df_variantes.to_string(index=False))
    reporte.texto(
        "El paso 4 de `ICE-claseRN03.md` dice: *\"Si yj ≠ dj(n), para **algun** j entre 1 y m1, "
        "entonces wji(n+1) = wji(n) + a·dj·xi(n), donde j = 1,...,m1\"*. Leido al pie de la letra, "
        "el error de una sola neurona obligaria a corregir **todas**. La formulacion habitual "
        "--- y la unica para la que vale el teorema de convergencia --- trata cada neurona de "
        "salida como un perceptron independiente y corrige solo la que se equivoca. El codigo "
        "implementa la segunda por omision y ofrece la primera con "
        "`actualizar_todas_las_salidas=True`:"
    )
    reporte.tabla(df_variantes, "Comparacion de ambas lecturas sobre el conjunto con ruido",
                  nombre_csv="04_variantes_paso4")

    # ------------------------------------------------------------------
    # 6. Matriz de confusion e indeterminacion
    # ------------------------------------------------------------------
    reporte.seccion("6. Matriz de confusion y respuestas indeterminadas")
    rng = np.random.default_rng(2024)
    X_prueba, d_prueba = [], []
    for indice, nombre in enumerate(CLASES):
        for _ in range(200):
            X_prueba.append(ds.contaminar(ds.letra(nombre), 6, semilla=int(rng.integers(1 << 30))))
            d_prueba.append(indice)
    X_prueba = np.array(X_prueba)
    d_prueba = np.array(d_prueba)

    predicciones = modelo.clasificar(X_prueba)
    M, etiquetas = mt.matriz_confusion(predicciones, d_prueba, etiquetas=[0, 1])
    df_conf = pd.DataFrame(M, columns=[f"predicho {CLASES[int(e)]}" for e in etiquetas])
    df_conf.insert(0, "real", [f"{CLASES[int(e)]}" for e in etiquetas])
    print("\nMatriz de confusion (400 patrones con 6 pixeles invertidos):")
    print(df_conf.to_string(index=False))
    reporte.tabla(df_conf, "Matriz de confusion sobre 400 patrones con 6 pixeles invertidos",
                  nombre_csv="04_matriz_confusion")

    # Salidas ambiguas con el modelo que usa punto de indeterminacion
    modelo_theta = PerceptronSimple(25, 2, razon_aprendizaje=1.0, theta=4.0, max_epocas=200)
    modelo_theta.entrenar(conjunto.X, conjunto.d)
    salidas = modelo_theta.predecir(X_prueba)
    ambiguos = int(np.sum(np.all(salidas == 0, axis=1) | (np.sum(salidas == 1, axis=1) != 1)))
    reporte.texto(
        f"Con un punto de indeterminacion θ = 4 la red puede ademas **abstenerse**: de los 400 "
        f"patrones degradados, {ambiguos} producen una respuesta ambigua (ninguna neurona activa, "
        "o mas de una). En un sistema real esa es informacion valiosa --- equivale a que la red "
        "diga *\"no se\"* en lugar de arriesgar una clasificacion --- y es la utilidad practica "
        "del punto neutro que introduce `ICE-claseRN03.md`."
    )

    # ------------------------------------------------------------------
    # 7. Conclusiones
    # ------------------------------------------------------------------
    reporte.seccion("7. Conclusiones")
    umbral_90 = next((k for k, e in zip(niveles, exactitud_limpio) if e < 0.9), None)
    reporte.lista([
        f"La red unicapa con dos neuronas de salida aprende los dos prototipos en "
        f"{modelo.epocas_usadas} epocas.",
        "La memoria de la red esta literalmente dibujada en su matriz de sinapsis, aunque no en "
        "la forma que uno esperaria: cada neurona almacena la **anti-plantilla de la clase "
        "rival**, que sobre dos clases es un discriminador equivalente.",
        f"La exactitud se mantiene por encima del 90 % hasta "
        f"{'los ' + str(umbral_90 - 1) + ' pixeles invertidos' if umbral_90 else 'el maximo nivel evaluado'}, "
        "es decir, con una fraccion importante de la imagen destruida.",
        "Entrenar con ejemplos ruidosos **no** mejora la tolerancia en este caso: la degrada. "
        "Con pesos iniciales nulos, entrenar con los prototipos limpios produce el filtro "
        "adaptado, que es optimo; el ruido solo desvia al algoritmo hacia otra solucion "
        "igualmente valida sobre el conjunto de entrenamiento pero peor alineada con las "
        "plantillas. El perceptron busca *una* solucion, no la mejor.",
        "El punto de indeterminacion permite que la red se abstenga ante entradas ambiguas.",
    ])

    ruta_reporte = reporte.escribir()
    print(f"\nInforme escrito en: {ruta_reporte.relative_to(_ruta.RAIZ)}")


if __name__ == "__main__":
    main()
