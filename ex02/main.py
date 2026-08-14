from dataclasses import dataclass
from functools import reduce
from typing import NewType, cast

import numpy as np
from numpy.typing import NDArray

activação = np.tanh


# O Numpy desencoraja o uso do tipo np.matrix, então só
# iremos usar arrays e criar um NewType para distinguir
# um array que é um vetor de um array que é uma matriz.
Vetor = NewType("Vetor", NDArray[np.float64])
Matriz = NewType("Matriz", NDArray[np.float64])


@dataclass
class Camada:
    pesos: Matriz
    viés: Vetor

    def __call__(self, entrada: Vetor) -> Vetor:
        return cast(Vetor, activação(entrada @ self.pesos + self.viés))


@dataclass
class MLP:
    camadas: list[Camada]

    def __call__(self, entrada: Vetor) -> Vetor:
        return reduce(
            lambda ativação_anterior, camada: camada(ativação_anterior),
            self.camadas,
            entrada,
        )


def main():
    print("Hello from ex02!")


if __name__ == "__main__":
    main()
