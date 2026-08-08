import random
from dataclasses import dataclass
from itertools import batched
from typing import Literal

type Matriz[T] = list[list[T]]
type Vetor[T] = list[T]
type Pixel = Literal[+1, -1]
type Classificação = Literal[+1, -1]

X: Pixel = +1
_: Pixel = -1
EXEMPLOS_PUROS_Y: list[tuple[Classificação, Matriz[Pixel]]] = [
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


def gerar_exemplos_classificados(
    num_exemplos_por_classificao: int, gerador: random.Random, prob_flip: float
) -> list[tuple[Classificação, Vetor[Pixel]]]:
    return [
        (
            classificação,
            [
                -valor if gerador.random() < prob_flip else valor
                for linha in matriz
                for valor in linha
            ],
        )
        for classificação, matriz in EXEMPLOS_PUROS_Y
        for _ in range(num_exemplos_por_classificao)
    ]


def printar_imagem_e_classificação(
    vector: Vetor[Pixel], classificacao: Classificação
) -> None:
    print(f"Classificação: {classificacao}")
    for linha in range(5):
        for coluna in range(5):
            print("X" if vector[linha * 5 + coluna] == +1 else "_", end=" ")
        print()


def printar_alguns_exemplos(
    exemplos: list[tuple[Classificação, Vetor[Pixel]]],
    num_para_printar: int,
    gerador: random.Random,
) -> None:
    exemplos_aleatorios = gerador.sample(exemplos, num_para_printar)
    for classificacao, exemplo in exemplos_aleatorios:
        printar_imagem_e_classificação(exemplo, classificacao)


def sinal(x: float) -> Classificação:
    return +1 if x > 0 else -1


@dataclass
class Modelo:
    pesos: Vetor[float]
    vies: float

    def calcular_z(self, entrada: Vetor[Pixel]) -> float:
        return (
            sum(
                peso * val_entrada
                for peso, val_entrada in zip(self.pesos, entrada, strict=True)
            )
            + self.vies
        )

    def inferir(self, entrada: Vetor[Pixel]) -> Classificação:
        return sinal(self.calcular_z(entrada))

    def aprender(
        self,
        taxa_aprendizagem: float,
        exemplos: list[tuple[Classificação, Vetor[Pixel]]],
    ) -> "Modelo":
        modelo = self
        for classificação, entrada in exemplos:
            z = modelo.calcular_z(entrada)
            erro = z - classificação
            ajuste = taxa_aprendizagem * erro
            modelo = Modelo(
                pesos=[
                    peso_original - ajuste * x
                    for peso_original, x in zip(modelo.pesos, entrada, strict=True)
                ],
                vies=modelo.vies - ajuste,
            )
        return modelo

    def classificou_todos_corretamente(
        self, exemplos: list[tuple[Classificação, Vetor[Pixel]]]
    ) -> bool:
        return all(
            self.inferir(entrada) == classificacao
            for classificacao, entrada in exemplos
        )


def main():
    taxa_aprendizado = 0.01
    max_epocas = 10000
    gerador = random.Random(123456789)
    prob_flip = 2 / 25
    num_exemplos_por_classificacao = 10
    treinamento_y = gerar_exemplos_classificados(
        num_exemplos_por_classificacao, gerador, prob_flip
    )
    num_pixels = len(treinamento_y[0][1])
    modelo = Modelo(pesos=[0] * num_pixels, vies=0)
    for _ in range(max_epocas):
        modelo = modelo.aprender(taxa_aprendizado, treinamento_y)
        if modelo.classificou_todos_corretamente(treinamento_y):
            break
    else:
        print("Não foi possível encontrar um modelo que classifique todos os exemplos.")
        return
    exemplos_nunca_vistos_y = gerar_exemplos_classificados(10_000, gerador, prob_flip)
    acertos = sum(
        modelo.inferir(exemplo) == classificacao
        for classificacao, exemplo in exemplos_nunca_vistos_y
    )
    print(f"Acertos: {(100 * acertos / len(exemplos_nunca_vistos_y)):.2f}%")
    printar_alguns_exemplos(exemplos_nunca_vistos_y, 5, gerador)


if __name__ == "__main__":
    main()
