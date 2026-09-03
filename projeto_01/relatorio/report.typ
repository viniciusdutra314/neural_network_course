#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "tabela_resultados.typ": tabela-resultados

#show: codly-init.with()
#codly(languages: codly-languages)

#let ink = rgb("#17324D")
#let accent = rgb("#2D6F8E")
#let muted = rgb("#607286")
#let pale = rgb("#EEF4F7")

#set page(
  paper: "a4",
  margin: (x: 1.45cm, y: 1.25cm),
  numbering: "1",
)
#set text(
  font: "New Computer Modern",
  size: 9.2pt,
  lang: "pt",
  region: "br",
)
#set par(justify: true, leading: 0.48em)
#set heading(numbering: none)
#show figure.caption: set text(size: 7.4pt, fill: muted)
#show heading.where(level: 2): it => block(
  width: 100%,
  sticky: true,
  above: 5pt,
  below: 3pt,
  inset: (x: 5pt, y: 2.5pt),
  fill: pale,
  stroke: (left: (paint: accent, thickness: 1.2pt)),
)[
  #text(
    font: "Red Hat Display",
    size: 10.5pt,
    weight: "semibold",
    fill: ink,
  )[#it.body]
]

#grid(
  columns: (0.58fr, 3fr),
  gutter: 12pt,
  align: horizon,
  [#image("logo_icmc.jpg", width: 100%)],
  [
    #text(font: "Red Hat Display", size: 16pt, weight: "bold", fill: ink)[
      Projeto 01 - Redes neurais multicamadas
    ]
    #v(2pt)
    #text(font: "Red Hat Text", size: 8.5pt, weight: "medium", fill: muted)[
      Vinícius Sousa Dutra · Número USP 13686257
    ]
  ],
)
#v(1pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1.1pt))

== Objetivo e dados

Foram treinadas redes _multilayer perceptron_ (MLP) para dois problemas: *classificação* das
178 amostras da base Wine, descritas por 13 atributos e três cultivares, e *regressão*
das duas coordenadas geográficas das 1.059 músicas da base Music Origin, descritas
por 68 atributos. Os dados foram obtidas do repositório UCI #footnote[
  https://archive.ics.uci.edu/dataset/109/wine e
  https://archive.ics.uci.edu/dataset/315/geographical+original+of+music.
], embaralhados aleatoriamente e separados em
treinamento (70%), teste (20%) e validação (10%).

== Modelo e treinamento
O programa é um arquivo Python (`projeto01.py`) dependendo apenas de NumPy e PyTorch.
Cada configuração foi criada com `torch.nn.Sequential`, feita de transformações lineares e
ativação sigmoide nas camadas intermediárias. Para não comprimir os dados, todas as camadas
intermediárias têm a largura da entrada. Por exemplo, com uma camada intermediária,
as arquiteturas são 13-13-3 para Wine e 68-68-2 para Music Origin.

Antes do treinamento, os atributos são padronizados com média zero e desvio padrão um, assim não há diferenças numéricas grandes nos atributos.

Na classificação, a classe prevista é a do neurônio de saída de maior ativação. Na regressão, as duas
saídas lineares representam latitude e longitude.

Para atualizar os pesos, a função custo $L$ foi a entropia cruzada para Wine (classificação) e erro quadrático médio para
Music Origin (regressão). Através de diferenciação automática, o PyTorch foi usado para calcular o $gradient L$ em
relação a cada paramêtro.

O treinamento usa todos os exemplos em cada ciclo para calcular o gradiente, os pesos são atualizados
pelo otimizador SGD (Stochastic Gradient Descent) com diferente valores de momentum e taxa de aprendizado.

== Resultados

Para cada base, foram avaliadas 81 configurações: três números de camadas intermediárias,
três números de ciclos, três taxas de aprendizado e três valores de momentum. Cada rede foi
treinada com o conjunto de treinamento e comparada pela métrica do conjunto de validação.


=== Melhores modelos

#let resumo = csv("resumo.csv", row-type: dictionary)
#let wine = resumo.find(linha => linha.at("dataset") == "Wine")
#let música = resumo.find(linha => linha.at("dataset") == "Música")
#let percentual(valor) = {
  str(calc.round(float(valor) * 10000) / 100).replace(".", ",") + "%"
}
#let decimal(valor) = {
  str(calc.round(float(valor) * 100) / 100).replace(".", ",")
}
#let taxa(valor) = {
  if valor == "0.001" { "1e-03" } else if valor == "0.01" { "1e-02" } else if valor == "0.1" { "1e-01" } else { valor }
}

#table(
  columns: (1.05fr, 0.9fr, 0.72fr, 0.72fr, 0.72fr, 0.9fr, 0.9fr, 0.9fr),
  align: center + horizon,
  inset: (x: 3pt, y: 3pt),
  stroke: white,
  table.header(
    ..("Base", "Cam.", "Ciclos", "Taxa", "Mom.", "Treino", "Validação", "Teste").map(título => table.cell(
      fill: pale,
    )[#strong[#título]]),
  ),
  [Wine],
  [#wine.at("camadas")],
  [#wine.at("ciclos")],
  [#taxa(wine.at("taxa_aprendizado"))],
  [#wine.at("momentum").replace(".", ",")],
  [#percentual(wine.at("treinamento"))],
  [#percentual(wine.at("validação"))],
  [#percentual(wine.at("teste"))],

  [Música],
  [#música.at("camadas")],
  [#música.at("ciclos")],
  [#taxa(música.at("taxa_aprendizado"))],
  [#música.at("momentum").replace(".", ",")],
  [#decimal(música.at("treinamento"))],
  [#decimal(música.at("validação"))],
  [#decimal(música.at("teste"))],
)

Para Wine, a melhor configuração foi a rede 13-13-3, com
#wine.at("ciclos") ciclos, taxa #taxa(wine.at("taxa_aprendizado")) e momentum
#wine.at("momentum").replace(".", ","). Ela obteve
#percentual(wine.at("treinamento")) no treinamento e
#percentual(wine.at("teste")) no teste.

#figure(
  block(width: 100%)[
    #set text(size: 5.8pt)
    #tabela-resultados("wine_tabela.csv", porcentagem: true)
  ],
  caption: [
    Acurácia de validação das 81 configurações de Wine. Tons mais escuros
    indicam maior acurácia.
  ],
) <tab:wine>
Para Music Origin, a melhor configuração foi a rede 68-68-2, com
#música.at("ciclos") ciclos, taxa #taxa(música.at("taxa_aprendizado")) e momentum
#música.at("momentum").replace(".", ","). Ela obteve MSE de
#decimal(música.at("treinamento")) no treinamento e
#decimal(música.at("teste")) no teste.

#figure(
  block(width: 100%)[
    #set text(size: 5.8pt)
    #tabela-resultados("music_tabela.csv", menor-melhor: true)
  ],
  caption: [
    MSE de validação das 81 configurações de Music Origin. Tons mais escuros
    indicam menor erro.
  ],
) <tab:música>

== Conclusão

A normalização dos atributos foi importante para os dois experimentos, antes eram necessários mais épocas do que só 100 para uma precisão parecida com a dos atributos normalizados. 

O momentum acelerou a convergência em Wine. Com 50 ciclos, os valores maiores de
momentum produziram as melhores acurácias, já com 100 ciclos, os modelos com momentum
já tinham alcançado 100% na validação.

Uma camada intermediária foi suficiente para as duas bases. As redes com duas ou
três camadas pioraram os resultados, isso pode indicar overfitting
ou que o número de exemplos não exige redes mais profundas.