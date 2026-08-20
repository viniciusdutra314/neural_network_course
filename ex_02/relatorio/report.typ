#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#codly(
  languages: codly-languages,
)

#let main-code = read("../main.py")

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
      MLP aplicada ao XOR e ao autoassociador #linebreak()
      (Exercício 02)
    ]
    #v(4pt)
    #text(font: "Red Hat Text", size: 9pt, weight: "medium", fill: muted)[
      Vinícius Sousa Dutra · Número USP 13686257
    ]
  ],
)
#v(2pt)
#line(length: 100%, stroke: (paint: accent, thickness: 1.1pt))

== Objetivo

O exercício consiste em implementar uma rede neural de múltiplas camadas (MLP) e aplicá-la a dois problemas.
O primeiro é o XOR, que não pode ser resolvido por um único neurônio pela sua natureza não linear.
O segundo é um autoassociador que recebe uma linha de uma matriz identidade, comprime esse padrão
em uma camada oculta e deve reconstruí-lo na saída.
Foram considerados os casos $"Id"(8 times 8)$ e $"Id"(15 times 15)$.

== Definições e notação

#[
  #set text(size: 8.3pt)
  #table(
    columns: (1.2fr, 4.8fr),
    align: (center, left),
    inset: (x: 4pt, y: 2pt),
    stroke: (x: 0.3pt + muted, y: 0.3pt + muted),
    fill: (_, row) => if row == 0 { pale },
    table.header([*Símbolo*], [*Definição*]),
    [$ell$ e $L$], [Índice de uma camada e número total de camadas.],
    [$x$ e $a^(ell)$], [Entrada ($a^(0) = x$) e ativação da camada $ell$.],

    [$W^(ell)$, $b^(ell)$ e $z^(ell)$],
    [Matriz de pesos, vetor de vieses e potencial
      $z^(ell) = a^(ell - 1) W^(ell) + b^(ell)$ antes da ativação.],

    [$sigma$, $t$ e $delta^(ell)$],
    [Sigmoide, saída esperada e $delta^(ell) = -(partial E) / (partial z^(ell))$,
      em que $E = 1 / 2 sum_j (t_j - a_j^(L))^2$],
  )
]

== Implementação

O código utiliza somente a biblioteca NumPy para implementar o MLP basicamente do zero, o paradigma
usado foi de programação funcional dada a natureza extremamente matemática do problema.

O programa representa cada camada pela dataclass `Camada`, que reúne a matriz de pesos e o vetor
de vieses, uma MLP é simplesmente uma sequência dessas camadas.

Na função `inicializar_modelo`, os pesos são amostrados de uma distribuição normal e divididos por
$sqrt(n_"in")$, onde $n_"in"$ é o número de entradas da camada, todos os vieses começam em zero.
A divisão por $sqrt(n_"in")$ faz com que os pesos tenham média zero e
variância $"Var"(W_(i j)) = 1 / n_"in"$.
Isso reduz a probabilidade de iniciar a sigmoide nas regiões saturadas, nas quais
os gradientes retropropagados se tornam muito pequenos.


A etapa forward da MLP é implementada pela função `forward`, conceitualmente a propagação da camada de
entrada até a saída é a composição de L funções $RR^(n_i) arrow RR^(n_j)$ em que $n_i$ é o número de números
na camada i e $n_j$ é o número de números na camada j.

#codly-range(180, end: 188)
#figure(
  align(center)[
    #block(width: 60%)[
      #set text(size: 7pt)
      #raw(main-code, lang: "python", block: true)
    ]
  ],
  caption: [Implementação de `forward` por composição],
)


Na etapa de backpropagation, é necessário calcular as derivadas parciais da função de custo em relação
a cada peso e viés. Para isso, é necessário calcular $delta^(ell)$, isso é separado em dois casos

Na camada final:

$ delta^(L) = (t - a^(L)) dot a^(L) dot (1 - a^(L)), $

e, em uma camada oculta, é

$
  delta^(ell) = (delta^(ell + 1) (W^(ell + 1))^T)
  dot a^(ell) dot (1 - a^(ell)).
$


Em `backpropagate`, `accumulate` é aplicado duas vezes: no sentido direto, produz a sequência de
$a^(ell)$; no sentido inverso, propaga os deltas $delta^(ell)$ entre as camadas.

Com eles é possível calcular os gradientes de cada camada e atualizar os parâmetros.

$
  (partial E)/(partial W_(i j))=-delta_j a_i^T "e " (partial E)/(partial b^(j))=-delta^(j)
$


#codly-range(69, end: 103)
#figure(
  align(center)[
    #block(width: 72%)[
      #set text(size: 6.5pt)
      #raw(main-code, lang: "python", block: true)
    ]
  ],
  caption: [Implementação da função backpropagate],
)

Em `train`, os exemplos são embaralhados a cada época, separados em lotes e processados por
backpropagation.
`média_gradientes` calcula a média do lote, gerando assim o gradiente médio.

== Resultados

Para o XOR, foram usados os quatro pares binários possíveis e a arquitetura é $2 arrow.r 2 arrow.r 1$.
Para os autoassociadores, cada linha de $"Id"(N times N)$ é simultaneamente entrada e alvo.
A camada intermediária contém $ceil(log_2 N)$ neurônios: as arquiteturas são
$8 arrow.r 3 arrow.r 8$ e $15 arrow.r 4 arrow.r 15$.

Em todos os casos, o treinamento usa $eta = 1$, 10.000 épocas e lotes de tamanho igual ao número de exemplos.
Uma saída é interpretada como 1 quando sua ativação é maior ou igual a $0,5$ e 0 caso contrário.

Como é visível na @fig:resultados, as redes conseguiram aprender os padrões desejados, com
um RMSE (root mean square error) na casa de $10^(-2)$, mesmo em um caso específico em que o RMSE é $10^(-1)$, isso não faz
tanta diferença pois a classificação é binária.

#figure(
  image("resultados.jpeg", width: 55%),
  caption: [Captura de tela da execução do código],
) <fig:resultados>
