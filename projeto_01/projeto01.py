import csv
import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np
import torch

type FunçãoPerda = torch.nn.CrossEntropyLoss | torch.nn.MSELoss
type FunçãoMétrica = Callable[[torch.nn.Module, Dataset], float]
SEED = 123456789

class Dataset(NamedTuple):
    alvos: torch.Tensor
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
    valor_validação: float


def carregar_wine_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    alvos = torch.from_numpy(dataset[:, 0].astype(np.int64)) - 1
    atributos = torch.from_numpy(dataset[:, 1:])
    return Dataset(alvos, atributos)


def carregar_music_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    atributos = torch.from_numpy(dataset[:, :-2])
    alvos = torch.from_numpy(dataset[:, -2:])
    return Dataset(alvos, atributos)


def particiona_em_treinamento_validação_teste(
    dataset: Dataset,
) -> Partições:
    quantidade = dataset.alvos.shape[0]
    quantidade_validação = round(quantidade * 0.1)
    quantidade_teste = round(quantidade * 0.2)
    quantidade_treinamento = quantidade - (quantidade_teste + quantidade_validação)
    índices_treinamento, índices_validação, índices_teste = torch.randperm(
        quantidade
    ).split((quantidade_treinamento, quantidade_validação, quantidade_teste))

    def selecionar(dataset: Dataset, índices: torch.Tensor) -> Dataset:
        return Dataset(dataset.alvos[índices], dataset.atributos[índices])

    return Partições(
        selecionar(dataset, índices_treinamento),
        selecionar(dataset, índices_validação),
        selecionar(dataset, índices_teste),
    )


def normalizar_atributos(partições: Partições) -> Partições:
    atributos_treinamento = partições.treinamento.atributos
    média = atributos_treinamento.mean(dim=0)
    desvio = atributos_treinamento.std(dim=0)
    desvio_sem_zeros = torch.where(
        desvio == 0,
        torch.ones_like(desvio),
        desvio,
    )

    def normalizar(dataset: Dataset) -> Dataset:
        atributos = (dataset.atributos - média) / desvio_sem_zeros
        return Dataset(dataset.alvos, atributos)

    return Partições(*(normalizar(dataset) for dataset in partições))


def criar_modelo(
    neuronios_entrada: int,
    camadas_intermediárias: Sequence[int],
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
    dataset: Dataset,
    momentum: float,
    lr: float,
    num_ciclo: int,
    loss_fn: FunçãoPerda,
) -> None:
    optimizer = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    for _ in range(num_ciclo):
        optimizer.zero_grad()
        loss = loss_fn(modelo(dataset.atributos), dataset.alvos)
        loss.backward()
        optimizer.step()


def porcentagem_acertos(modelo: torch.nn.Module, dataset: Dataset) -> float:
    with torch.no_grad():
        return float(
            (modelo(dataset.atributos).argmax(dim=1) == dataset.alvos)
            .float()
            .mean()
            .item()
        )


def erro_quadrático_médio(
    modelo: torch.nn.Module,
    dataset: Dataset,
) -> float:
    with torch.no_grad():
        return torch.nn.functional.mse_loss(
            modelo(dataset.atributos), dataset.alvos
        ).item()


def quantidade_neuronios_saida(alvos: torch.Tensor) -> int:
    alvos_são_classes = not alvos.is_floating_point()
    if alvos_são_classes:
        quantidade_classes = alvos.unique().numel()
        return int(quantidade_classes)

    regressão_multivariada = alvos.ndim == 2
    if regressão_multivariada:
        quantidade_variáveis_resposta = alvos.shape[1]
        return quantidade_variáveis_resposta

    return 1


def treinos_com_diferentes_hiperparâmetros(
    partições: Partições,
    loss_fn: FunçãoPerda,
    métrica: FunçãoMétrica,
    maximizar: bool,
    /,
    nums_camadas: Iterable[int],
    nums_ciclos: Iterable[int],
    taxas_aprendizado: Iterable[float],
    momentums: Iterable[float],
) -> tuple[list[Resultado], torch.nn.Module]:
    resultados: list[Resultado] = []
    melhor_modelo: torch.nn.Module | None = None
    melhor_valor = -math.inf if maximizar else math.inf
    neuronios_entrada = partições.treinamento.atributos.shape[1]
    neuronios_saida = quantidade_neuronios_saida(partições.treinamento.alvos)
    configurações = itertools.product(
        nums_camadas, nums_ciclos, taxas_aprendizado, momentums
    )
    for num_camadas, num_ciclos, taxa_aprendizado, momentum in configurações:
        _ = torch.manual_seed(SEED + num_camadas)
        modelo = criar_modelo(
            neuronios_entrada,
            [neuronios_entrada] * num_camadas,
            neuronios_saida,
        )
        treinar_modelo(
            modelo,
            partições.treinamento,
            momentum=momentum,
            lr=taxa_aprendizado,
            num_ciclo=num_ciclos,
            loss_fn=loss_fn,
        )
        valor_validação = métrica(modelo, partições.validação)
        resultados.append(
            {
                "camadas": num_camadas,
                "ciclos": num_ciclos,
                "taxa_aprendizado": taxa_aprendizado,
                "momentum": momentum,
                "valor_validação": valor_validação,
            }
        )
        if melhor_modelo is None or (
            valor_validação > melhor_valor
            if maximizar
            else valor_validação < melhor_valor
        ):
            melhor_modelo = modelo
            melhor_valor = valor_validação

    assert melhor_modelo is not None
    return resultados, melhor_modelo


def salvar_csv(
    resultados: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=resultados[0].keys(),
            lineterminator="\n",
        )
        escritor.writeheader()
        escritor.writerows(resultados)


def main() -> None:
    _ = torch.manual_seed(SEED)
    DIRETÓRIO = Path(__file__).resolve().parent
    WINE_RESULTS = DIRETÓRIO / "relatorio" / "wine_tabela.csv"
    WINE_DATASET = carregar_wine_dataset(DIRETÓRIO / "datasets" / "wine" / "wine.txt")
    MUSIC_RESULTS = DIRETÓRIO / "relatorio" / "music_tabela.csv"
    MUSIC_DATASET = carregar_music_dataset(
        DIRETÓRIO
        / "datasets"
        / "geographical_original_of_music"
        / "default_features_1059_tracks.txt"
    )

    CAMADAS = (1, 2, 3)
    CICLOS = (10, 50, 100)
    TAXAS_APRENDIZADO = (1e-3, 1e-2, 1e-1)
    MOMENTUMS = (0.0, 0.5, 0.9)

    configurações = [
        (
            "Wine",
            "Acurácia",
            WINE_DATASET,
            WINE_RESULTS,
            torch.nn.CrossEntropyLoss(),
            porcentagem_acertos,
            True,
        ),
        (
            "Música",
            "MSE",
            MUSIC_DATASET,
            MUSIC_RESULTS,
            torch.nn.MSELoss(),
            erro_quadrático_médio,
            False,
        ),
    ]
    resumo: list[dict[str, object]] = []
    for (
        nome,
        nome_métrica,
        dataset,
        table_dir,
        loss_fn,
        métrica,
        maximizar,
    ) in configurações:
        partições = normalizar_atributos(
            particiona_em_treinamento_validação_teste(dataset)
        )
        resultados, melhor_modelo = treinos_com_diferentes_hiperparâmetros(
            partições,
            loss_fn,
            métrica,
            maximizar,
            nums_camadas=CAMADAS,
            nums_ciclos=CICLOS,
            taxas_aprendizado=TAXAS_APRENDIZADO,
            momentums=MOMENTUMS,
        )
        salvar_csv(resultados, table_dir)
        melhor_resultado = (max if maximizar else min)(
            resultados,
            key=lambda resultado: resultado["valor_validação"],
        )
        referência: float | str = ""
        if nome == "Música":
            média_alvos = partições.treinamento.alvos.mean(dim=0)
            referência = torch.nn.functional.mse_loss(
                média_alvos.expand_as(partições.teste.alvos), partições.teste.alvos
            ).item()
        resumo.append(
            {
                "dataset": nome,
                "métrica": nome_métrica,
                "camadas": melhor_resultado["camadas"],
                "ciclos": melhor_resultado["ciclos"],
                "taxa_aprendizado": melhor_resultado["taxa_aprendizado"],
                "momentum": melhor_resultado["momentum"],
                "treinamento": métrica(melhor_modelo, partições.treinamento),
                "validação": melhor_resultado["valor_validação"],
                "teste": métrica(melhor_modelo, partições.teste),
                "referência": referência,
            }
        )
    salvar_csv(resumo, DIRETÓRIO / "relatorio" / "resumo.csv")


if __name__ == "__main__":
    main()
