"""Anade la raiz del repositorio a `sys.path` para poder importar `ce_rna`.

Se importa al principio de cada experimento, de modo que los scripts funcionan
tanto ejecutados desde la raiz (`python experimentos/exp01_...py`) como desde
el propio directorio `experimentos/`, sin necesidad de instalar el paquete.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

FIGURAS = RAIZ / "resultados" / "figuras"
TABLAS = RAIZ / "resultados" / "tablas"
REPORTES = RAIZ / "resultados" / "reportes"
for _d in (FIGURAS, TABLAS, REPORTES):
    _d.mkdir(parents=True, exist_ok=True)
