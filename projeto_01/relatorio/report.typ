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

Na classificação, a classe prevista é a do neurônio de saída de maior ativação. Na regressão, as duas
saídas lineares representam latitude e longitude.

Para atualizar os pesos, a função custo $L$ foi a entropia cruzada para Wine (classificação) e erro quadrático médio para
Music Origin (regressão). Através de diferenciação automática, o PyTorch foi usado para calcular o $gradient L$ em
relação a cada paramêtro.

O treinamento usa todos os exemplos em cada ciclo para calcular o gradiente, os pesos são atualizados
pelo otimizador SGD (Stochastic Gradient Descent) com diferente valores de momentum e taxa de aprendizado.

== Resultados

Foram avaliadas as 81 combinações do produto cartesiano camadas_intermediária=[1, 2, 3]
$times$ ciclos=[10, 20, 40] $times$ taxas_aprendizado=[$10^(-3)$, $10^(-2)$, $10^(-1)$]
momentum=[0.0 , 0.5, 0,9].
Os modelos são treinados para cada configuração usando a partição de treino (70% do dataset total)
e a qualidade do modelo é testada pela partição de validação (10% do dataset), isso é feito
para selecionar a priori o que seria o melhor conjunto de hiperparâmetros.


=== Melhores Modelos

#let resumo = csv("resumo.csv", row-type: dictionary)
#let wine = resumo.find(linha => linha.at("dataset") == "Wine")
#let música = resumo.find(linha => linha.at("dataset") == "Música")
#let percentual(valor) = {
  str(calc.round(float(valor) * 10000) / 100).replace(".", ",") + "%"
}
#let decimal(valor) = {
  str(calc.round(float(valor) * 100) / 100).replace(".", ",")
}
#let raiz(valor) = decimal(calc.sqrt(float(valor)))
#let redução(valor, referência) = percentual(
  1 - float(valor) / float(referência),
)
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

Para Wine, a melhor rede foi 13-13-3, com #wine.at("ciclos") ciclos, taxa
#taxa(wine.at("taxa_aprendizado")) e momentum
#wine.at("momentum").replace(".", ",").
Ela obteve #percentual(wine.at("treinamento")) no treinamento e
#percentual(wine.at("teste")) no teste. Para Music Origin, a rede 68-68-2,
treinada por #música.at("ciclos") ciclos com taxa
#taxa(música.at("taxa_aprendizado")) e momentum
#música.at("momentum").replace(".", ","), atingiu MSE de
#decimal(música.at("treinamento")) no treinamento e #decimal(música.at("teste"))
no teste.

Com poucos ciclos, o efeito do momentum ficou mais evidente. Em Wine, o melhor
resultado com momentum 0,5 ou 0,9 foi 100% na validação, contra 72,22% com 0,0.
Em Music Origin, embora a melhor configuração isolada tenha usado momentum zero,
o MSE médio das 27 configurações caiu de 1758,71 para 1631,89 e 1584,89 ao variar
o momentum de 0,0 para 0,5 e 0,9, respectivamente.

O MSE geográfico é expresso em graus ao quadrado e combina latitude e longitude.
Na base, os desvios padrão dessas coordenadas são 18,45° e 50,40°; portanto, a
longitude domina o erro. O MSE de teste equivale a RMSE de
#raiz(música.at("teste"))° por coordenada. Como referência, prever sempre a média
do treinamento produz MSE #decimal(música.at("referência")) no teste: a rede o
reduziu em #redução(música.at("teste"), música.at("referência")). Embora ainda
grande, o erro é melhor que essa referência trivial e não corresponde diretamente
a uma distância em quilômetros.

#figure(
  block(width: 100%)[
    #set text(size: 5.8pt)
    #tabela-resultados("wine_tabela.csv", porcentagem: true)
  ],
  caption: [
    Acurácia de validação das 81 configurações de Wine. Tons mais claros,
    no extremo azul-claro da escala, indicam maior acurácia. Cada grupo de
    colunas representa uma taxa de aprendizado e as subcolunas, o momentum.
  ],
) <tab:wine>

#figure(
  block(width: 100%)[
    #set text(size: 5.8pt)
    #tabela-resultados("music_tabela.csv", menor-melhor: true)
  ],
  caption: [
    MSE de validação das 81 configurações de Music Origin. Tons mais claros,
    no extremo azul-claro da escala, indicam menor erro. Cada grupo de colunas
    representa uma taxa de aprendizado e as subcolunas, o momentum.
  ],
) <tab:música>

== Conclusão

As tabelas mostram que o aumento de profundidade piorou o desempenho,
provavelmente devido a overfitting de possuir muitos paramêtros.
Com poucos ciclos de treinamento, o momentum foi útil para acelerar a convergência
em grande parte das configurações.
