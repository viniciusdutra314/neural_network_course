import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

WINE_DATASET = Path(__file__).resolve().parent / "datasets" / "wine" / "wine.csv"


class Dataset(NamedTuple):
    classificações: torch.Tensor
    atributos: torch.Tensor


class Partições(NamedTuple):
    treinamento: Dataset
    validação: Dataset
    teste: Dataset


def load_wine_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    classificações = torch.from_numpy(dataset[:, 0].astype(np.int64)) - 1
    atributos = torch.from_numpy(dataset[:, 1:])
    return Dataset(
        classificações, (atributos - atributos.mean(dim=0)) / atributos.std(dim=0)
    )


def divide_em_treinamento_validação_teste(
    dataset: Dataset,
    seed: int,
) -> Partições:
    quantidade = dataset.classificações.shape[0]
    quantidade_teste = round(quantidade * 0.2)
    quantidade_validação = round(quantidade * 0.1)
    quantidade_treinamento = quantidade - quantidade_teste - quantidade_validação
    gerador = torch.Generator().manual_seed(seed)
    índices = torch.randperm(quantidade, generator=gerador)

    índices_treino = índices[:quantidade_treinamento]
    índices_validação = índices[
        quantidade_treinamento : quantidade_treinamento + quantidade_validação
    ]
    índices_teste = índices[quantidade_treinamento + quantidade_validação :]

    return Partições(
        Dataset(
            dataset.classificações[índices_treino],
            dataset.atributos[índices_treino],
        ),
        Dataset(
            dataset.classificações[índices_validação],
            dataset.atributos[índices_validação],
        ),
        Dataset(
            dataset.classificações[índices_teste],
            dataset.atributos[índices_teste],
        ),
    )


def criar_modelo(
    neuronios_entrada: int,
    camadas_intermediárias: Iterable[int],
    neuronios_saida: int,
) -> torch.nn.Sequential:
    assert neuronios_entrada > 0, "É necessário pelo menos 1 um neurônio de entrada"
    assert all(camada > 0 for camada in camadas_intermediárias), (
        "Todas as camadas intermediárias precisam ter neurônios"
    )
    assert neuronios_saida > 0, "É necessário pelo menos 1 um neurônio de saida"
    dimensões = [neuronios_entrada, *camadas_intermediárias, neuronios_saida]
    camadas: list[torch.nn.Module] = []
    for index, (camada_ant, camada_pos) in enumerate(itertools.pairwise(dimensões)):
        camadas.append(torch.nn.Linear(camada_ant, camada_pos))
        if index != len(dimensões) - 2:
            camadas.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*camadas)


def treinar_modelo(
    modelo: torch.nn.Module,
    entrada: torch.Tensor,
    esperado: torch.Tensor,
    momentum: float,
    lr: float,
    num_ciclo: int,
) -> None:
    optimizer = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(num_ciclo):
        loss = loss_fn(modelo(entrada), esperado)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main() -> None:
    seed = 123456789
    dataset = load_wine_dataset(WINE_DATASET)
    partições = divide_em_treinamento_validação_teste(
        dataset,
        seed,
    )

    modelo = criar_modelo(partições.treinamento.atributos.shape[1], (3, 4), 3)
    treinar_modelo(
        modelo,
        partições.treinamento.atributos,
        partições.treinamento.classificações,
        momentum=0.1,
        lr=1e-2,
        num_ciclo=20,
    )

    with torch.no_grad():
        previsto = torch.argmax(modelo(partições.teste.atributos), dim=1)
        acertos = torch.sum(previsto == partições.teste.classificações)
        print(acertos)


if __name__ == "__main__":
    main()
