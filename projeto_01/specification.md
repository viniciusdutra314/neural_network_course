## Projeto 1

- Considere as seguintes bases de dados:

  - **Wine** — classificação https://archive.ics.uci.edu/dataset/109/wine
  - **Music Origin** — regressão / aproximação https://archive.ics.uci.edu/dataset/315/geographical+original+of+music.

- Para cada uma das bases, você deverá realizar experimentos usando o algoritmo **backpropagation com o termo momentum**.

- Divida as bases de dados em:

  - treinamento: **70%**;
  - validação: **10%**;
  - teste: **20%**.

- Usando os conjuntos de treino e validação, você deverá calibrar os parâmetros das redes neurais considerando **3 variações** dos seguintes parâmetros:

  - número de camadas intermediárias, por exemplo: 1, 2 e 3;
  - número de ciclos usados no treinamento, por exemplo: múltiplos de 5;
  - parâmetro de **momentum**, com valores menores que 1;
  - velocidade de aprendizado, por exemplo: `1e-2`, `1e-3` e `1e-4`.

- A calibração consiste em buscar um conjunto de parâmetros que maximize ou minimize uma métrica de sua escolha, por exemplo, os parâmetros que produzam a maior acurácia.

- Elabore um relatório completo, detalhando as melhores arquiteturas para cada base de dados.

- Crie uma tabela comparativa considerando:

  - arquiteturas da rede;
  - número de ciclos;
  - velocidades de aprendizado;
  - momentum.

### Problema de classificação

Mostre a **acurácia** obtida para os conjuntos de:

- treinamento;
- teste.

### Problema de aproximação

Mostre o **erro quadrático médio (MSE)** obtido nos conjuntos de:

- treinamento;
- teste.

## Implementação

A implementação deverá ser realizada em **Python**, e qualquer pré-processamento dos arquivos de entrada deverá estar contido no próprio código-fonte.

## Entrega

Anexar no Moodle, em um único arquivo compactado com extensão `.zip` ou `.rar`, intitulado:

```text
<seu_nome>_projeto1.zip
```

ou

```text
<seu_nome>_projeto1.rar
```

O arquivo deverá conter:

- código-fonte;
- relatório.
