#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#codly(languages: codly-languages)

#let ink = rgb("#17324D")
#let accent = rgb("#2D6F8E")
#let muted = rgb("#607286")
#let pale = rgb("#EEF4F7")

#set page(
  paper: "a4",
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "pt",
  region: "br",
)
#set par(justify: true)
#set heading(numbering: none)
#show heading.where(level: 2): it => block(
  width: 100%,
  sticky: true,
  inset: (x: 7pt, y: 4pt),
  fill: pale,
  stroke: (paint: accent, thickness: 1pt),
)[
  #text(font: "Red Hat Display", size: 13pt, weight: "semibold", fill: ink)[#it.body]
]
#show figure.caption: set text(size: 9pt, fill: muted)

#grid(
  columns: (1fr, 2fr),
  gutter: 17pt,
  align: horizon,
  [#image("logo_icmc.jpg", width: 100%)],
  [
    #v(4pt)
    #text(font: "Red Hat Display", size: 21pt, weight: "bold", fill: ink)[
      Classificação de símbolos #linebreak() com ADALINE (Exercício 01)
    ]
    #v(6pt)
    #text(font: "Red Hat Text", size: 10pt, weight: "medium", fill: muted)[
      Vinícius Sousa Dutra · Número USP 13686257
    ]
  ],
)
#line(length: 100%, stroke: (paint: accent, thickness: 1.2pt))

== Introdução

O objetivo deste exercício é implementar e treinar um modelo ADALINE capaz de distinguir a letra Y de sua versão invertida. Para isso, foram criados exemplos com ruído para o treinamento e um conjunto separado para avaliar o comportamento do modelo com imagens que não foram usadas no ajuste dos pesos.

== Dados e representação

Cada símbolo foi representado por uma matriz $5 times 5$, na qual cada posição assume o valor $+1$ ou $-1$. Antes de ser fornecida ao modelo, a matriz é linearizada em ordem _row-major_, formando um vetor com 25 entradas. O Y convencional recebeu o rótulo $+1$, enquanto o Y invertido recebeu o rótulo $-1$. Os dois padrões de referência estão apresentados na @code:padroes.

#codly-range(10, end: 33)
#figure(
  scale(
    62%,
    origin: top + left,
    reflow: true,
    raw(read("../main.py"), lang: "python", block: true),
  ),
  caption: [Representação do Y convencional e do Y invertido (em `main.py`).],
) <code:padroes>

O conjunto de treinamento possui 20 exemplos, sendo 10 de cada classe. Esses exemplos são produzidos a partir dos dois padrões de referência definidos em `main.py`, para simular ruído, cada pixel tem probabilidade $p = 2/25$ de ter seu sinal invertido (cerca de 2 pixels por imagem).

Para a avaliação, foram geradas com o mesmo nível de ruído, 20.000 imagens: 10.000 de cada classe.

== Modelo ADALINE

A ADALINE possui uma saída linear durante o aprendizado. Dada uma entrada $bold(x)$, o potencial do neurônio é calculado por

$ z = sum_i w_i x_i + b. $

Para realizar a classificação, aplica-se a função sinal `sgn` a esse valor e  mudança de pesos e víes é feita de forma a minimizar o erro quadrático $E = 1/2 (t-z)^2$. Dessa forma, os gradientes em relação aos pesos e ao viés são

$ (partial E) / (partial w_i) = (z-t)x_i quad "e" quad (partial E) / (partial b) = z-t. $

Com uma taxa de aprendizado $alpha$, a atualização é realizada pela descida do gradiente, de modo que

$ w_i arrow.l w_i - alpha (z-t)x_i quad "e" quad b arrow.l b - alpha (z-t). $

== Treinamento

Os pesos e o viés foram inicializados em zero, e foi utilizada uma taxa de aprendizado de $alpha = 0,01$. O treinamento
foi feito até que o modelo classifique corretamente todos os dados de treinamento ou até um número máximo de épocas.

#codly-range(135, end: 142)
#figure(
  scale(
    62%,
    origin: top + left,
    reflow: true,
    raw(read("../main.py"), lang: "python", block: true),
  ),
  caption: [Lógica de treinamento (em `main.py`).],
) <code:treinamento>


== Resultados

O modelo classificou corretamente os 20 exemplos de treinamento e, portanto, obteve 100% de acurácia nesse conjunto após uma única época.

No conjunto de teste, foram classificadas corretamente 19.997 das 20.000 imagens, o que corresponde a 99,98% de acurácia.

#figure(
  image("relatorio_print_resultado.jpeg", width: 30%),
  caption: [Saída do programa com a acurácia e alguns exemplos de teste.],
)
