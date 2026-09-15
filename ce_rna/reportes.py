"""
Generacion automatica de informes en Markdown.

Cada experimento construye un objeto `Reporte`, le va anadiendo secciones,
tablas y figuras, y al final lo escribe en `resultados/reportes/`.  De este modo
los informes **no se redactan a mano**: se generan a partir de la misma
ejecucion que produce los numeros, con lo que nunca pueden quedar desfasados
respecto del codigo.

Las tablas se guardan ademas en `resultados/tablas/` como CSV, para que puedan
reutilizarse desde una hoja de calculo o desde otro script.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DIR_FIGURAS = RAIZ / "resultados" / "figuras"
DIR_TABLAS = RAIZ / "resultados" / "tablas"
DIR_REPORTES = RAIZ / "resultados" / "reportes"


def _df_a_markdown(df) -> str:
    """Convierte un DataFrame a tabla Markdown sin depender de `tabulate`."""
    columnas = [str(c) for c in df.columns]
    encabezado = "| " + " | ".join(columnas) + " |"
    separador = "|" + "|".join(["---"] * len(columnas)) + "|"
    filas = []
    for _, fila in df.iterrows():
        celdas = []
        for valor in fila:
            if isinstance(valor, (float, np.floating)):
                celdas.append(f"{valor:.6g}")
            else:
                celdas.append(str(valor))
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join([encabezado, separador, *filas])


class Reporte:
    """Acumulador de secciones Markdown para un experimento.

    Parameters
    ----------
    titulo:
        Titulo del informe (encabezado de nivel 1).
    nombre_archivo:
        Nombre base del `.md` que se escribira en `resultados/reportes/`.
    resumen:
        Parrafo introductorio que describe el objetivo del experimento.
    """

    def __init__(self, titulo: str, nombre_archivo: str, resumen: str = ""):
        self.titulo = titulo
        self.nombre_archivo = nombre_archivo
        self.partes: list[str] = []
        self.figuras: list[Path] = []
        self.tablas: list[Path] = []
        if resumen:
            self.partes.append(resumen.strip())

    # -- construccion ------------------------------------------------------
    def seccion(self, titulo: str, nivel: int = 2) -> "Reporte":
        self.partes.append(f"{'#' * nivel} {titulo}")
        return self

    def texto(self, texto: str) -> "Reporte":
        self.partes.append(texto.strip())
        return self

    def lista(self, elementos) -> "Reporte":
        self.partes.append("\n".join(f"- {e}" for e in elementos))
        return self

    def bloque(self, contenido: str, lenguaje: str = "text") -> "Reporte":
        self.partes.append(f"```{lenguaje}\n{contenido.rstrip()}\n```")
        return self

    def formula(self, latex: str) -> "Reporte":
        self.partes.append(f"$$\n{latex.strip()}\n$$")
        return self

    def tabla(self, df, titulo: str = "", nombre_csv: str | None = None) -> "Reporte":
        """Anade una tabla al informe y, si se indica, la guarda como CSV."""
        if titulo:
            self.partes.append(f"**{titulo}**")
        self.partes.append(_df_a_markdown(df))
        if nombre_csv:
            DIR_TABLAS.mkdir(parents=True, exist_ok=True)
            ruta = DIR_TABLAS / f"{nombre_csv}.csv"
            df.to_csv(ruta, index=False)
            self.tablas.append(ruta)
            self.partes.append(f"*Datos completos: [`{ruta.name}`](../tablas/{ruta.name})*")
        return self

    def figura(self, ruta, pie: str = "") -> "Reporte":
        """Inserta una figura ya guardada (ruta relativa al directorio de reportes)."""
        ruta = Path(ruta)
        self.figuras.append(ruta)
        relativa = f"../figuras/{ruta.name}"
        self.partes.append(f"![{pie or ruta.stem}]({relativa})")
        if pie:
            self.partes.append(f"*Figura: {pie}*")
        return self

    # -- escritura ---------------------------------------------------------
    def escribir(self) -> Path:
        DIR_REPORTES.mkdir(parents=True, exist_ok=True)
        ruta = DIR_REPORTES / f"{self.nombre_archivo}.md"
        marca = datetime.now().strftime("%Y-%m-%d %H:%M")
        cabecera = (
            f"# {self.titulo}\n\n"
            f"> Informe generado automaticamente por los scripts de `experimentos/` "
            f"el {marca}.\n> No editar a mano: se regenera con `python experimentos/ejecutar_todo.py`.\n"
        )
        cuerpo = "\n\n".join(self.partes)
        ruta.write_text(f"{cabecera}\n{cuerpo}\n", encoding="utf-8")
        return ruta


def encabezado_experimento(nombre: str) -> None:
    """Imprime un separador legible en la consola."""
    print()
    print("=" * 78)
    print(f"  {nombre}")
    print("=" * 78)
