#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#import "tabela_wine.typ": tabela-wine

#show: codly-init.with()
#codly(
  languages: codly-languages,
)

#let ink = rgb("#17324D")
#let accent = rgb("#2D6F8E")
#let muted = rgb("#607286")
#let pale = rgb("#EEF4F7")

#set page(
  paper: "a4",
  margin: (x: 1.7cm, y: 1.4cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 10.2pt,
  lang: "pt",
  region: "br",
)
#set par(justify: true, leading: 0.52em)
#set heading(numbering: none)
#show figure.caption: set text(size: 8.2pt, fill: muted)
#show heading.where(level: 2): it => block(
  width: 100%,
  sticky: true,
  above: 7pt,
  below: 4pt,
  inset: (x: 6pt, y: 3pt),
  fill: pale,
  stroke: (left: (paint: accent, thickness: 1.2pt)),
)[
  #text(
    font: "Red Hat Display",
    size: 11.5pt,
    weight: "semibold",
    fill: ink,
  )[#it.body]
]

#grid(
  columns: (0.72fr, 3fr),
  gutter: 15pt,
  align: horizon,
  [#image("logo_icmc.jpg", width: 100%)],
  [
    #text(font: "Red Hat Display", size: 17pt, weight: "bold", fill: ink)[
      Projeto 01
    ]
    #v(4pt)
    #text(font: "Red Hat Text", size: 9pt, weight: "medium", fill: muted)[
      Vinícius Sousa Dutra · Número USP 13686257
    ]
  ],
)
#v(2pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1.1pt))

== Implementação
O código utiliza a biblioteca de redes neurais _PyTorch_, configurada para rodar somente utilizado CPU devido
aos datasets serem pequenos
== Dados
Os datasets estão guardados na pasta `datasets/` com os arquivos originais dos sites #footnote[
  https://archive.ics.uci.edu/dataset/109/wine e https://archive.ics.uci.edu/dataset/315/geographical+original+of+music.
], o pre-processamento é feito pelas funções `carregar_wine_dataset` e `carregar_music_dataset`, elas normalizam os atributos
para terem $mu=0$ e $std=1$, afim de evitar qualquer tipo de saturação dos neurônios. Mesmo que os datasets sejam conceitualmente
colunas de uma tabela, eles são convertidos para ser um objeto tensor `pytorch.Tensor`

== Otimização
O otimizador usado é o `torch.optim.SGD` ou seja o método de otimização é o _stocastic gradient descent_


== Resultados


== Wine

#figure(
  block(width: 100%)[
    #set text(size: 6.4pt)
    #tabela-wine("wine_tabela.csv")
  ],
  caption: [
    Acurácia de validação das configurações testadas para a base Wine.
  ],
) <tab:wine-configurações>

== Música
