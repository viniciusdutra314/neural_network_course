from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, batched, chain, pairwise
from math import sqrt
from random import Random
from typing import NewType

import numpy as np
from numpy import float64
from numpy.typing import NDArray

# O Numpy desencoraja o uso do tipo np.matrix, então só
# iremos usar arrays e criar um NewType para distinguir
# um array que é um vetor de um array que é uma matriz.
Vetor = NewType("Vetor", NDArray[np.float64])
Matriz = NewType("Matriz", NDArray[np.float64])


def phi(potencial: NDArray[np.float64]) -> Vetor:
    return Vetor(1 / (1 + np.exp(-potencial)))


@dataclass(frozen=True)
class Camada:
    pesos: Matriz
    viés: Vetor


@dataclass(frozen=True)
class GradienteCamada:
    pesos: Matriz
    viés: Vetor


@dataclass(frozen=True)
class GradienteMLP:
    camadas: tuple[GradienteCamada, ...]


@dataclass(frozen=True)
class MLP:
    camadas: tuple[Camada, ...]


def inicializar_modelo(
    dimensões_camadas: tuple[int, ...],
    inicializar_pesos: Callable[[tuple[int, int], int], Matriz],
    inicializar_viés: Callable[[int, int], Vetor],
) -> MLP:
    return MLP(
        camadas=tuple(
            Camada(
                pesos=inicializar_pesos((entradas, saídas), l),
                viés=inicializar_viés(saídas, l),
            )
            for l, (entradas, saídas) in enumerate(pairwise(dimensões_camadas))
        )
    )


def propagar_camada(camada: Camada, entrada: Vetor) -> Vetor:
    return phi(entrada @ camada.pesos + camada.viés)


def backpropagate(
    modelo: MLP,
    entrada: Vetor,
    saida_esperada: Vetor,
) -> GradienteMLP:
    # Equações tiradas dos slides da disciplina
    L = len(modelo.camadas)
    y = tuple(
        accumulate(
            modelo.camadas,
            lambda y_anterior, camada: propagar_camada(camada, y_anterior),
            initial=entrada,
        )
    )
    t = saida_esperada
    # Camada de saída:
    delta_L = Vetor((t - y[L]) * y[L] * (1 - y[L]))

    deltas_reversos = accumulate(
        reversed(range(L - 1)),
        lambda delta_l_mais_1, l: Vetor(
            (delta_l_mais_1 @ modelo.camadas[l + 1].pesos.T)
            * y[l + 1]
            * (1 - y[l + 1])
        ),
        initial=delta_L,
    )

    deltas = reversed(tuple(deltas_reversos))

    # dE/dw_ji = -delta_j y_i
    # dE/db_j   = -delta_j
    return GradienteMLP(
        camadas=tuple(
            GradienteCamada(
                pesos=Matriz(-np.outer(y_i, delta_j)),
                viés=Vetor(-delta_j),
            )
            for y_i, delta_j in zip(
                y[:L],
                deltas,
                strict=True,
            )
        )
    )


def gradient_descent(
    modelo: MLP, gradiente: GradienteMLP, taxa_aprendizado: float64
) -> MLP:
    return MLP(
        camadas=tuple(
            Camada(
                pesos=Matriz(camada.pesos - taxa_aprendizado * gradiente_camada.pesos),
                viés=Vetor(camada.viés - taxa_aprendizado * gradiente_camada.viés),
            )
            for (gradiente_camada, camada) in zip(
                gradiente.camadas, modelo.camadas, strict=True
            )
        )
    )


def média_gradientes(
    gradientes: list[GradienteMLP],
) -> GradienteMLP:
    return GradienteMLP(
        camadas=tuple(
            GradienteCamada(
                pesos=Matriz(
                    np.mean(
                        [gradiente.camadas[i].pesos for gradiente in gradientes],
                        axis=0,
                    )
                ),
                viés=Vetor(
                    np.mean(
                        [gradiente.camadas[i].viés for gradiente in gradientes],
                        axis=0,
                    )
                ),
            )
            for i in range(len(gradientes[0].camadas))
        )
    )


def train(
    modelo: MLP,
    exemplos_de_treinamento: list[tuple[Vetor, Vetor]],
    taxa_aprendizado: float64,
    épocas: int,
    tamanho_batch: int,
    gerador_aleatório: Random,
) -> MLP:
    batches = chain.from_iterable(
        batched(
            gerador_aleatório.sample(
                exemplos_de_treinamento,
                k=len(exemplos_de_treinamento),
            ),
            tamanho_batch,
        )
        for _ in range(épocas)
    )

    return reduce(
        lambda modelo_anterior, batch: gradient_descent(
            modelo_anterior,
            média_gradientes(
                [
                    backpropagate(modelo_anterior, entrada, esperado)
                    for entrada, esperado in batch
                ]
            ),
            taxa_aprendizado,
        ),
        batches,
        modelo,
    )


def forward(modelo: MLP, entrada: Vetor) -> Vetor:
    return reduce(
        lambda ativação_anterior, camada: propagar_camada(camada, ativação_anterior),
        modelo.camadas,
        entrada,
    )


def main() -> None:


if __name__ == "__main__":
    main()
