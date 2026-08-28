import csv
import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import torch


class Dataset(NamedTuple):
    classificações: torch.Tensor
    atributos: torch.Tensor


class Partições(NamedTuple):
    treinamento: Dataset
    validação: Dataset
    teste: Dataset


class Resultado(TypedDict):
    camadas: int
    ciclos: int
    taxa_aprendizado: float
    momentum: float
    acurácia_validação: float


def carregar_wine_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    classificações = torch.from_numpy(dataset[:, 0].astype(np.int64)) - 1
    atributos = torch.from_numpy(dataset[:, 1:])
    return Dataset(
        classificações, (atributos - atributos.mean(dim=0)) / atributos.std(dim=0)
    )


def carregar_music_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    atributos = torch.from_numpy(dataset[:, :-2])
    classificações = torch.from_numpy(dataset[:, -2:])
    return Dataset(
        classificações, ((atributos - atributos.mean(dim=0)) / atributos.std(dim=0))
    )


def particiona_em_treinamento_validação_teste(
    dataset: Dataset,
) -> Partições:
    quantidade = dataset.classificações.shape[0]
    quantidade_teste = round(quantidade * 0.2)
    quantidade_validação = round(quantidade * 0.1)
    quantidade_treinamento = quantidade - quantidade_teste - quantidade_validação
    índices = torch.randperm(quantidade)

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
    loss_fn: torch.nn.CrossEntropyLoss | torch.nn.MSELoss,
) -> None:
    optimizer = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    for _ in range(num_ciclo):
        loss = loss_fn(modelo(entrada), esperado)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def porcentagem_acertos(modelo: torch.nn.Module, dataset: Dataset) -> float:
    with torch.no_grad():
        inferência = torch.argmax(modelo(dataset.atributos), dim=1)
        return float(torch.sum(inferência == dataset.classificações) / len(inferência))


def treinos_com_diferentes_hiperparâmetros(
    partições: Partições,
    loss_fn: torch.nn.CrossEntropyLoss | torch.nn.MSELoss,
    /,
    nums_camadas: Iterable[int],
    nums_ciclos: Iterable[int],
    taxas_aprendizado: Iterable[float],
    momentums: Iterable[float],
) -> list[Resultado]:
    resultados: list[Resultado] = []
    neuronios_entrada = partições.treinamento.atributos.shape[1]
    neuronios_saida = int(partições.treinamento.classificações.unique().numel())
    configurações = itertools.product(
        nums_camadas, nums_ciclos, taxas_aprendizado, momentums
    )
    for num_camadas, num_ciclos, taxa_aprendizado, momentum in configurações:
        modelo = criar_modelo(
            neuronios_entrada,
            [neuronios_entrada] * num_camadas,
            neuronios_saida,
        )
        treinar_modelo(
            modelo,
            partições.treinamento.atributos,
            partições.treinamento.classificações,
            momentum=momentum,
            lr=taxa_aprendizado,
            num_ciclo=num_ciclos,
            loss_fn=loss_fn,
        )
        resultados.append(
            {
                "camadas": num_camadas,
                "ciclos": num_ciclos,
                "taxa_aprendizado": taxa_aprendizado,
                "momentum": momentum,
                "acurácia_validação": porcentagem_acertos(
                    modelo,
                    partições.validação,
                ),
            }
        )
    return resultados


def salvar_resultados(
    resultados: list[Resultado],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=resultados[0])
        escritor.writeheader()
        escritor.writerows(resultados)


def main() -> None:
    WINE_DATASET_DIR = (
        Path(__file__).resolve().parent / "datasets" / "wine" / "wine.txt"
    )
    WINE_TABLE = Path(__file__).resolve().parent / "relatorio" / "wine_tabela.csv"
    WINE_DATASET = carregar_wine_dataset(WINE_DATASET_DIR)
    MUSIC_DATASET_DIR = (
        Path(__file__).resolve().parent
        / "datasets"
        / "geographical_original_of_music"
        / "default_features_1059_tracks.txt"
    )
    MUSIC_TABLE = Path(__file__).resolve().parent / "relatorio" / "music_tabela.csv"
    MUSIC_DATASET = carregar_music_dataset(MUSIC_DATASET_DIR)

    CAMADAS = (1, 2, 3)
    CICLOS = (100, 200, 300)
    TAXAS_APRENDIZADO = (1e-3, 1e-2, 1e-1)
    MOMENTUMS = (0.0, 0.5, 0.9)

    for dataset, table_dir, loss_fn in [
        (WINE_DATASET, WINE_TABLE, torch.nn.CrossEntropyLoss()),
        (MUSIC_DATASET, MUSIC_TABLE, torch.nn.MSELoss()),
    ]:
        partições = particiona_em_treinamento_validação_teste(dataset)
        resultados = treinos_com_diferentes_hiperparâmetros(
            partições,
            loss_fn,
            nums_camadas=CAMADAS,
            nums_ciclos=CICLOS,
            taxas_aprendizado=TAXAS_APRENDIZADO,
            momentums=MOMENTUMS,
        )
        salvar_resultados(resultados, table_dir)


if __name__ == "__main__":
    main()
