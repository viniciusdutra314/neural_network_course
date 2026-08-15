from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, batched, chain, repeat
from typing import NewType, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray

# O Numpy desencoraja o uso do tipo np.matrix, então só
# iremos usar arrays e criar um NewType para distinguir
# um array que é um vetor de um array que é uma matriz.
Vetor = NewType("Vetor", NDArray[np.float64])
Matriz = NewType("Matriz", NDArray[np.float64])


phi = np.tanh


def phi_prime(ativação: Vetor) -> Vetor:
    return Vetor(1.0 - ativação**2)


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


def propagar_camada(camada: Camada, entrada: Vetor) -> Vetor:
    return cast(Vetor, phi(entrada @ camada.pesos + camada.viés))


def backpropagate(modelo: MLP, entrada: Vetor, saida_esperada: Vetor) -> GradienteMLP:
    # Equações tiradas de:
    # https://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
    L = len(modelo.camadas)

    # Propagação direta: a_l = phi(a_(l-1) W_l + b_l).
    ativações = tuple(
        accumulate(
            modelo.camadas,
            lambda ativação, camada: propagar_camada(camada, ativação),
            initial=entrada,
        )
    )

    # Delta da saída: delta_L = (a_L - y) phi_prime(a_L).
    ERRO = ativações[L] - saida_esperada
    delta_L = Vetor(ERRO * phi_prime(ativações[L]))

    # Camadas ocultas:
    # delta_l = (delta_(l+1) W_(l+1).T) phi_prime(a_l).
    deltas_reversos = accumulate(
        zip(
            reversed(modelo.camadas[1:L]),
            reversed(ativações[1:L]),
            strict=True,
        ),
        lambda delta, camada_e_ativação: Vetor(
            (delta @ camada_e_ativação[0].pesos.T) * phi_prime(camada_e_ativação[1])
        ),
        initial=delta_L,
    )
    deltas = reversed(tuple(deltas_reversos))

    return GradienteMLP(
        camadas=tuple(
            GradienteCamada(
                # dW_l = outer(a_(l-1), delta_l)
                pesos=Matriz(np.outer(ativação_anterior, delta)),
                # db_l = delta_l
                viés=delta,
            )
            for ativação_anterior, delta in zip(
                ativações[:L],
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
) -> MLP:
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
        batched(
            chain.from_iterable(repeat(exemplos_de_treinamento, épocas)),
            tamanho_batch,
        ),
        modelo,
    )


def forward(modelo: MLP, entrada: Vetor) -> Vetor:
    return reduce(
        lambda ativação_anterior, camada: propagar_camada(camada, ativação_anterior),
        modelo.camadas,
        entrada,
    )


def main():
    print("Hello from ex02!")


if __name__ == "__main__":
    main()
