"""
Funciones de activacion phi(v) usadas por los modelos neuronales del curso.

Referencia teorica: `clases/claseRN01.md`, seccion
"Algunas funciones de activacion normalmente utilizadas".

    F(x) = theta(x)                  Heaviside (escalon)
    F(x) = 2*theta(x) - 1            Escalon bipolar
    F(x) = 1 / (1 + e^-x)            Sigmoide
    F(x) = (1 - e^-x)/(1 + e^-x)     Sigmoide bipolar
    F(x) = e^-((x-mu)^2 / (2 sigma^2))   Gaussiana

Todas las funciones estan implementadas desde cero sobre `numpy`: no se usa
ninguna libreria de aprendizaje automatico.  Cada activacion se expone como una
clase con `__call__` (la funcion phi) y `derivada` (dphi/dv), de modo que los
algoritmos de descenso de gradiente (ADALINE) puedan consultarla de manera
uniforme.

La derivada del escalon es cero en todo punto donde esta definida (e infinita en
el origen); por eso la regla perceptronica NO puede derivarse del gradiente y se
implementa como una regla de correccion de error discreta.  Esta es exactamente
la diferencia conceptual entre el PERCEPTRON y el ADALINE que se discute en
`clases/ICE-claseRN04.md`.
"""

from __future__ import annotations

import numpy as np


class FuncionActivacion:
    """Interfaz comun de todas las activaciones."""

    nombre = "abstracta"
    #: True si la activacion es derivable y utilizable por descenso de gradiente.
    derivable = False

    def __call__(self, v):  # pragma: no cover - interfaz
        raise NotImplementedError

    def derivada(self, v):  # pragma: no cover - interfaz
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Escalon(FuncionActivacion):
    """Heaviside: phi(v) = 1 si v >= umbral, 0 en caso contrario.

    Es la activacion del modelo de McCulloch-Pitts (`clases/claseRN02.md`):
    "la funcion basica de una celula es sumar entradas y producir una
    determinada salida si la suma es mayor que un umbral determinado".

    Parameters
    ----------
    umbral:
        Nivel a partir del cual la neurona dispara.  Se deja explicito porque en
        el modelo de McCulloch-Pitts el umbral no es un peso entrenable sino una
        propiedad fija de la neurona (umbral 2 para AND y OR de dos entradas).
    """

    nombre = "escalon"

    def __init__(self, umbral: float = 0.0):
        self.umbral = float(umbral)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        return np.where(v >= self.umbral, 1.0, 0.0)

    def derivada(self, v):
        v = np.asarray(v, dtype=float)
        return np.zeros_like(v)

    def __repr__(self) -> str:
        return f"Escalon(umbral={self.umbral})"


class EscalonBipolar(FuncionActivacion):
    """phi(v) = 2*theta(v) - 1, es decir +1 si v >= 0 y -1 si v < 0.

    Es la activacion usada por el perceptron de Rosenblatt tal como se presenta
    en `clases/ICE-claseRN03.md`: cada neurona de salida responde +1 ("pertenece
    a la clase") o -1 ("no pertenece a la clase").
    """

    nombre = "escalon_bipolar"

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        return np.where(v >= 0.0, 1.0, -1.0)

    def derivada(self, v):
        v = np.asarray(v, dtype=float)
        return np.zeros_like(v)


class EscalonBipolarConZona(FuncionActivacion):
    """Escalon bipolar con *punto de indeterminacion* de ancho 2*theta.

              +1   si  v >  theta
    phi(v) =   0   si -theta <= v <= theta      <- "la neurona no sabe que responder"
              -1   si  v < -theta

    Esta es la activacion que aparece dibujada en `clases/ICE-claseRN03.md`
    ("Notese que incluyo un punto neutro. A este punto se le suele llamar punto
    de indeterminacion").  Con theta = 0 se recupera `EscalonBipolar`.

    El punto neutro tiene una consecuencia practica importante: obliga al
    algoritmo perceptronico a separar las clases con un *margen*, porque un
    patron que cae dentro de la banda [-theta, theta] se considera erroneo
    aunque su signo sea el correcto.
    """

    nombre = "escalon_bipolar_zona"

    def __init__(self, theta: float = 0.0):
        self.theta = float(theta)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        salida = np.zeros_like(v)
        salida[v > self.theta] = 1.0
        salida[v < -self.theta] = -1.0
        return salida

    def derivada(self, v):
        v = np.asarray(v, dtype=float)
        return np.zeros_like(v)

    def __repr__(self) -> str:
        return f"EscalonBipolarConZona(theta={self.theta})"


class Identidad(FuncionActivacion):
    """phi(v) = v.  Activacion de las *neuronas lineales*.

    En `clases/claseRN02.md` se explica que las neuronas de entrada o sensores
    usan la identidad ("lo que esperamos de un sensor es que indique
    precisamente lo que esta percibiendo"), y en `clases/ICE-claseRN04.md` se
    usa para la salida del ADALINE: "al ser unidades lineales sin funcion de
    activacion en la capa de salida".
    """

    nombre = "identidad"
    derivable = True

    def __call__(self, v):
        return np.asarray(v, dtype=float)

    def derivada(self, v):
        v = np.asarray(v, dtype=float)
        return np.ones_like(v)


class Sigmoide(FuncionActivacion):
    """phi(v) = 1 / (1 + e^-v), con derivada phi*(1-phi).

    Implementada de forma numericamente estable para evitar desbordamiento de
    `exp` con argumentos muy negativos.
    """

    nombre = "sigmoide"
    derivable = True

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        salida = np.empty_like(v)
        pos = v >= 0
        salida[pos] = 1.0 / (1.0 + np.exp(-v[pos]))
        exp_v = np.exp(v[~pos])
        salida[~pos] = exp_v / (1.0 + exp_v)
        return salida

    def derivada(self, v):
        s = self(v)
        return s * (1.0 - s)


class SigmoideBipolar(FuncionActivacion):
    """phi(v) = (1 - e^-v) / (1 + e^-v) = tanh(v/2).  Recorrido (-1, 1)."""

    nombre = "sigmoide_bipolar"
    derivable = True

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        return np.tanh(v / 2.0)

    def derivada(self, v):
        s = self(v)
        return 0.5 * (1.0 - s ** 2)


class Gaussiana(FuncionActivacion):
    """phi(v) = exp(-(v - mu)^2 / (2 sigma^2)).

    No se usa en los modelos de este repositorio (es la activacion tipica de las
    redes de base radial), pero aparece en la lista de `clases/claseRN01.md` y se
    incluye por completitud del catalogo de activaciones del curso.
    """

    nombre = "gaussiana"
    derivable = True

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma debe ser positivo")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        return np.exp(-((v - self.mu) ** 2) / (2.0 * self.sigma ** 2))

    def derivada(self, v):
        v = np.asarray(v, dtype=float)
        return -((v - self.mu) / self.sigma ** 2) * self(v)

    def __repr__(self) -> str:
        return f"Gaussiana(mu={self.mu}, sigma={self.sigma})"


#: Catalogo por nombre, util para los reportes y para construir figuras.
CATALOGO = {
    "escalon": Escalon,
    "escalon_bipolar": EscalonBipolar,
    "escalon_bipolar_zona": EscalonBipolarConZona,
    "identidad": Identidad,
    "sigmoide": Sigmoide,
    "sigmoide_bipolar": SigmoideBipolar,
    "gaussiana": Gaussiana,
}


def obtener(nombre: str, **kwargs) -> FuncionActivacion:
    """Devuelve una instancia de activacion a partir de su nombre."""
    if nombre not in CATALOGO:
        disponibles = ", ".join(sorted(CATALOGO))
        raise KeyError(f"Activacion desconocida '{nombre}'. Disponibles: {disponibles}")
    return CATALOGO[nombre](**kwargs)
