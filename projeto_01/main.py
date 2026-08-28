import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import PercentFormatter


class Dataset(NamedTuple):
    classificações: torch.Tensor
    atributos: torch.Tensor


class Partições(NamedTuple):
    treinamento: Dataset
    validação: Dataset
    teste: Dataset


def carregar_wine_dataset(path: Path) -> Dataset:
    dataset = np.loadtxt(path, delimiter=",", dtype=np.float32)
    classificações = torch.from_numpy(dataset[:, 0].astype(np.int64)) - 1
    atributos = torch.from_numpy(dataset[:, 1:])
    return Dataset(
        classificações, (atributos - atributos.mean(dim=1)) / (atributos.std(dim=1))
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
) -> None:
    optimizer = torch.optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
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
    /,
    arquiteturas: Iterable[Iterable[int]],
    nums_ciclos: Iterable[int],
    taxas_aprendizado: Iterable[float],
    momentums: Iterable[float],
) -> pd.DataFrame:
    resultados: list[dict[str, object]] = []
    configurações = itertools.product(
        arquiteturas,
        nums_ciclos,
        taxas_aprendizado,
        momentums,
    )
    for arquitetura, num_ciclos, taxa_aprendizado, momentum in configurações:
        modelo = criar_modelo(
            partições.treinamento.atributos.shape[1],
            arquitetura,
            partições.treinamento.classificações.shape[1],
        )
        treinar_modelo(
            modelo,
            partições.treinamento.atributos,
            partições.treinamento.classificações,
            momentum=momentum,
            lr=taxa_aprendizado,
            num_ciclo=num_ciclos,
        )
        resultados.append(
            {
                "arquitetura": arquitetura,
                "ciclos": nums_ciclos,
                "taxa_aprendizado": taxa_aprendizado,
                "momentum": momentum,
                "acurácia_validação": porcentagem_acertos(
                    modelo,
                    partições.validação,
                ),
            }
        )
    return pd.DataFrame.from_records(resultados)


def criar_heatmap_wine(
    resultados: pd.DataFrame,
    path: Path,
) -> None:
    sns.set_theme()
    arquiteturas = tuple(reversed(ARQUITETURAS))
    momentums = tuple(reversed(MOMENTUMS))
    figura, eixos = plt.subplots(
        len(arquiteturas),
        len(CICLOS),
        figsize=(12, 10),
        constrained_layout=True,
        squeeze=False,
    )
    grupos = resultados.groupby(["arquitetura", "ciclos"], sort=False)

    for linha, arquitetura in enumerate(arquiteturas):
        for índice_ciclos, ciclos in enumerate(CICLOS):
            grupo = grupos.get_group((arquitetura, ciclos))
            tabela = grupo.pivot(
                index="momentum",
                columns="taxa_aprendizado",
                values="acurácia_validação",
            )
            tabela = tabela.reindex(
                index=momentums,
                columns=TAXAS_APRENDIZADO,
            )
            tabela.index = [str(valor) for valor in momentums]
            tabela.columns = [f"{taxa:.0e}" for taxa in TAXAS_APRENDIZADO]

            eixo = eixos[linha, índice_ciclos]
            sns.heatmap(
                tabela,
                ax=eixo,
                annot=True,
                fmt=".0%",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                cbar=False,
                square=True,
                linewidths=1,
                linecolor="white",
                xticklabels=linha == len(arquiteturas) - 1,
                yticklabels=índice_ciclos == 0,
            )

            eixo.set_title(str(ciclos) if linha == 0 else "")
            eixo.set_xlabel("")
            if índice_ciclos == 0:
                quantidade = len(arquitetura)
                camada = "camada" if quantidade == 1 else "camadas"
                eixo.set_ylabel(f"{quantidade} {camada}")
                eixo.tick_params(axis="y", labelrotation=0)
            else:
                eixo.set_ylabel("")

    _ = figura.add_artist(
        ConnectionPatch(
            xyA=(0, 1.12),
            xyB=(1, 1.12),
            coordsA=eixos[0, 0].transAxes,
            coordsB=eixos[0, -1].transAxes,
            arrowstyle="->",
            color="black",
            linewidth=1.5,
        )
    )
    eixos[0, 1].text(
        0.5,
        1.14,
        "Ciclos (entre mapas)",
        transform=eixos[0, 1].transAxes,
        ha="center",
        va="bottom",
    )
    figura.add_artist(
        ConnectionPatch(
            xyA=(1.15, 0),
            xyB=(1.15, 1),
            coordsA=eixos[-1, -1].transAxes,
            coordsB=eixos[0, -1].transAxes,
            arrowstyle="->",
            color="black",
            linewidth=1.5,
        )
    )
    eixos[1, 2].text(
        1.18,
        0.5,
        "Camadas intermediárias (entre mapas)",
        transform=eixos[1, 2].transAxes,
        ha="left",
        va="center",
        rotation=90,
    )
    figura.suptitle("Wine — acurácia de validação")
    figura.supxlabel("Taxa de aprendizado (dentro de cada mapa) →")
    figura.supylabel("Momentum (dentro de cada mapa) ↑")
    barra = figura.colorbar(eixos[0, 0].collections[0], ax=eixos)
    barra.set_label("Acurácia")
    barra.set_ticks([0, 0.5, 1])
    barra.ax.yaxis.set_major_formatter(PercentFormatter(1))
    figura.savefig(path)
    plt.close(figura)


def main() -> None:
    WINE_DATASET = Path(__file__).resolve().parent / "datasets" / "wine" / "wine.csv"
    WINE_HEATMAP = Path(__file__).resolve().parent / "relatorio" / "wine_heatmap.svg"

    ARQUITETURAS = ((13,), (13, 13), (13, 13, 13))
    CICLOS = (100, 500, 1000)
    TAXAS_APRENDIZADO = (1e-3, 1e-2, 1e-1)
    MOMENTUMS = (0.0, 0.5, 0.9)

    dataset = carregar_wine_dataset(WINE_DATASET)
    partições = particiona_em_treinamento_validação_teste(dataset)
    resultados = treinos_com_diferentes_hiperparâmetros(
        partições,
        arquiteturas=ARQUITETURAS,
        nums_ciclos=CICLOS,
        taxas_aprendizado=TAXAS_APRENDIZADO,
        momentums=MOMENTUMS,
    )
    criar_heatmap_wine(resultados, WINE_HEATMAP)


if __name__ == "__main__":
    main()
