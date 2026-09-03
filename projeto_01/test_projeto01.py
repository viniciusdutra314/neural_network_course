import unittest

import torch

from projeto01 import (
    Dataset,
    fixar_semente,
    particiona_em_treinamento_validação_teste,
    quantidade_neuronios_saida,
)


class TestQuantidadeNeuroniosSaida(unittest.TestCase):
    def test_classificação_usa_uma_saida_por_classe_observada(self) -> None:
        alvos = torch.tensor([3, 1, 3, 2], dtype=torch.int64)

        self.assertEqual(quantidade_neuronios_saida(alvos), 3)

    def test_regressão_escalar_usa_uma_saida(self) -> None:
        alvos = torch.tensor([0.2, 0.5, 0.7], dtype=torch.float32)

        self.assertEqual(quantidade_neuronios_saida(alvos), 1)

    def test_regressão_multivariada_usa_uma_saida_por_coluna(self) -> None:
        alvos = torch.tensor([[0.2, 0.5], [0.7, 0.9]], dtype=torch.float32)

        self.assertEqual(quantidade_neuronios_saida(alvos), 2)


class TestReprodutibilidade(unittest.TestCase):
    def test_mesma_semente_produz_a_mesma_partição(self) -> None:
        dataset = Dataset(
            alvos=torch.arange(10),
            atributos=torch.arange(10).unsqueeze(dim=1).float(),
        )

        fixar_semente()
        primeira_partição = particiona_em_treinamento_validação_teste(dataset)
        fixar_semente()
        segunda_partição = particiona_em_treinamento_validação_teste(dataset)

        for primeira, segunda in zip(primeira_partição, segunda_partição, strict=True):
            self.assertTrue(torch.equal(primeira.alvos, segunda.alvos))
            self.assertTrue(torch.equal(primeira.atributos, segunda.atributos))


if __name__ == "__main__":
    unittest.main()
