import random
from dataclasses import dataclass
from typing import Literal

type Matriz[T] = list[list[T]]
type Vetor[T] = list[T]
type Binário = Literal[+1, -1]
type Classificacao = Binário

X: Binário = +1
_: Binário = -1
MAX_TENTATIVAS = 10000
GERADOR = random.Random(123456789)
PROB_FLIP = 2 / 25
NUM_EXEMPLOS_CLASSIFICACAO = 10
TAXA_APRENDIZADO = 0.01


EXEMPLOS_PUROS_Y: list[tuple[Classificacao, Matriz[Binário]]] = [
    (
        +1,
        [
            [X, _, _, _, X],
            [X, _, _, _, X],
            [_, X, X, X, _],
            [_, _, X, _, _],
            [_, _, X, _, _],
        ],
    ),
    (
        -1,
        [
            [_, _, X, _, _],
            [_, _, X, _, _],
            [_, X, X, X, _],
            [X, _, _, _, X],
            [X, _, _, _, X],
        ],
    ),
]


EXEMPLOS_Y: list[tuple[Classificacao, Vetor[Binário]]] = [
    (
        classificação,
        [
            -valor if GERADOR.random() < PROB_FLIP else valor
            for linha in matriz
            for valor in linha
        ],
    )
    for classificação, matriz in EXEMPLOS_PUROS_Y
    for _ in range(NUM_EXEMPLOS_CLASSIFICACAO)
]


def sinal(x: float) -> Binário:
    return +1 if x > 0 else -1


@dataclass
class Modelo:
    pesos: Vetor[float]
    vies: float

    def calcular_z(self, entrada: Vetor[Binário]) -> float:
        return (
            sum(
                peso * val_entrada
                for peso, val_entrada in zip(self.pesos, entrada, strict=True)
            )
            + self.vies
        )

    def inferir(self, entrada: Vetor[Binário]) -> Classificacao:
        return sinal(self.calcular_z(entrada))

    def aprender(
        self, exemplos: list[tuple[Classificacao, Vetor[Binário]]]
    ) -> "Modelo":
        modelo = self
        for classificação, entrada in exemplos:
            z = modelo.calcular_z(entrada)
            modelo = Modelo(
                pesos=[
                    peso_original - TAXA_APRENDIZADO * (z - classificação) * x
                    for peso_original, x in zip(modelo.pesos, entrada, strict=True)
                ],
                vies=modelo.vies - TAXA_APRENDIZADO * (z - classificação),
            )
        return modelo


def main():
    modelo = Modelo(pesos=[0] * 25, vies=0)
    tentativas = 0
    while (
        any(
            modelo.inferir(entrada) != classificacao
            for classificacao, entrada in EXEMPLOS_Y
        )
        and tentativas < MAX_TENTATIVAS
    ):
        modelo = modelo.aprender(EXEMPLOS_Y)
        tentativas += 1

    if tentativas == MAX_TENTATIVAS:
        print("Não foi possível encontrar um modelo que classifique todos os exemplos.")
        return
    print(modelo.pesos)


if __name__ == "__main__":
    main()
