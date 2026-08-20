import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, batched, chain, pairwise
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


def sigma(potencial: NDArray[np.float64]) -> Vetor:
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


def backpropagate(
    modelo: MLP,
    entrada: Vetor,
    saida_esperada: Vetor,
) -> GradienteMLP:
    # Equações tiradas dos slides da disciplina
    L = len(modelo.camadas)
    a = tuple(
        accumulate(
            modelo.camadas,
            lambda a_anterior, camada: sigma(a_anterior @ camada.pesos + camada.viés),
            initial=entrada,
        )
    )
    t = saida_esperada
    deltas = reversed(
        tuple(
            accumulate(
                reversed(range(L - 1)),
                lambda delta_l_mais_1, l: Vetor(
                    (delta_l_mais_1 @ modelo.camadas[l + 1].pesos.T)
                    * a[l + 1]
                    * (1 - a[l + 1])
                ),
                initial=Vetor((t - a[L]) * a[L] * (1 - a[L])),
            ),
        )
    )
    return GradienteMLP(
        camadas=tuple(
            GradienteCamada(
                pesos=Matriz(-np.outer(a_i, delta_j)),  # dE/dw_ji = -a_i delta_j
                viés=Vetor(-delta_j),  # dE/db_j   = -delta_j
            )
            for a_i, delta_j in zip(
                a[:L],
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
        lambda ativação_anterior, camada: sigma(
            ativação_anterior @ camada.pesos + camada.viés
        ),
        modelo.camadas,
        entrada,
    )


def main() -> None:
    seed = 123456789
    gerador_np = np.random.default_rng(seed)
    gerador_random = Random(seed)
    ### Exemplo XOR
    modelo: MLP = inicializar_modelo(
        (2, 2, 1),
        lambda dimensões, _: Matriz(
            gerador_np.standard_normal(dimensões) / math.sqrt(dimensões[0])
        ),
        lambda dimensão, _: Vetor(np.zeros(dimensão)),
    )
    exemplos_xor = [
        (Vetor(np.array([0.0, 0.0])), Vetor(np.array([0.0]))),
        (Vetor(np.array([0.0, 1.0])), Vetor(np.array([1.0]))),
        (Vetor(np.array([1.0, 0.0])), Vetor(np.array([1.0]))),
        (Vetor(np.array([1.0, 1.0])), Vetor(np.array([0.0]))),
    ]
    modelo = train(
        modelo,
        exemplos_xor,
        taxa_aprendizado=float64(1.0),
        épocas=10_000,
        tamanho_batch=4,
        gerador_aleatório=gerador_random,
    )
    print("Exemplos XOR")
    for entrada, esperado in exemplos_xor:
        saída = forward(modelo, entrada)
        classe = 1 if saída[0] >= 0.5 else 0
        rmse = np.sqrt(np.mean((esperado - saída) ** 2))
        print(
            f"{entrada=}, saida={saída[0]:.4f}, "
            f"esperado={esperado[0]:.4f}, {classe=}, {rmse=:.4f}"
        )

    # Exemplos autoassociador
    print("Exemplos autoassociador")
    for N, épocas in [(8, 10_000), (15, 10_000)]:
        N_log2 = math.ceil(math.log2(N))
        modelo = inicializar_modelo(
            (N, N_log2, N),
            lambda dimensões, _: Matriz(
                gerador_np.standard_normal(dimensões) / math.sqrt(dimensões[0])
            ),
            lambda dimensão, _: Vetor(np.zeros(dimensão)),
        )

        identidade = np.eye(N)
        exemplos = [(Vetor(linha.copy()), Vetor(linha.copy())) for linha in identidade]
        modelo = train(
            modelo,
            exemplos,
            taxa_aprendizado=float64(1.0),
            épocas=épocas,
            tamanho_batch=len(exemplos),
            gerador_aleatório=gerador_random,
        )

        print(f"\nAutoassociador Id({N}x{N}): {N} -> {N_log2} -> {N}")
        print("padrão\tbits reconstruídos\tRMSE".expandtabs(20))
        corretos = 0
        for índice, (entrada, esperado) in enumerate(exemplos):
            saída = forward(modelo, entrada)
            bits_reconstruídos = [i + 1 for i in range(N) if saída[i] >= 0.5]
            correto = bits_reconstruídos == [índice + 1]
            rmse = np.sqrt(np.mean((esperado - saída) ** 2))
            corretos += correto
            print(f"{índice + 1}\t{bits_reconstruídos}\t{rmse:.4f}".expandtabs(20))

        print(f"Padrões reconstruídos corretamente: {corretos}/{N}")


if __name__ == "__main__":
    main()
