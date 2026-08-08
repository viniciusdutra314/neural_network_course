#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#codly(languages: codly-languages)

#set text(size: 11pt, lang: "pt", region: "br")
#set par(justify: true)

#align(center, [Aluno: Vinícius Sousa Dutra, Número USP: 13686257])
#codly-range(11, end: 35)

#figure(
  scale(
    75%,
    origin: top + left,
    reflow: true,
    raw(
      read("main.py"),
      lang: "python",
      block: true,
    ),
  ),
  caption: [main.py],
) <code:y_referencia>


=== Dados de treinamento

O trabalho consiste em classificar se imagens da letra Y estão invertidas ou não. Cada imagem
foi modelada por uma matriz 5 x 5, onde cada elemento pode ter um de dois valores (+1 ou -1). As duas
imagens de referências estão no código @code:y_referencia. Para o treinamento, foi criado 10 variações dessas
imagens em que cada pixel tinha uma probabilidade $p$ de sofrer flip ($x arrow -x$).

Como o modelo de rede neural que estamos usando é de somente uma camada, as imagens bidimensionais
são linearizadas para um vetor unidimensional, tal linearização é feita seguindo a convenção _row-major_ popularizada
pela linguagem C.

=== Modelo

O tipo de rede neural usado é uma ADALINE, tal rede somente uma camada de neurônios de
entrada $arrow(x)$ e somente uma saida, a função de ativação é a função sinal  e a inferência
foi calculada aplicando a função sinal no produto escalar da imagem $arrow(x)$ pelos pesos $arrow(w)$ deslocada pelo viés $b$.
$ "classificação" = "sgn"(sum_i omega_i x_i + b) $

O aprendizado foi feito pela minimização da função do erro quadrático médio, comparando a ativação com o estimulo desejado $t$, a otimização foi feita
seguindo o algoritmo de descida do gradiente. O tamanho do "passo" da descida do gradiente é controlado pelo parâmetro $alpha$.
$ (partial E)/(partial w_i) = (t -sum_i omega_i x_i + b) x_i arrow Delta w_i = - alpha (partial E)/(partial w_i) $
$ (partial E)/(partial b) = (t -sum_i omega_i x_i + b) arrow Delta b = - alpha (partial E)/(partial b) $



=== Resultados (inferência)
Como está no @code:resultados, inicializamos o modelo com os pesos e viés nulo, aplicamos o algoritmo de aprendizado até que ele classifique corretamente todos os exemplos de treinamento, ou
até que um MAX_EPOCH seja atingido. Com o modelo treinado, usamos a inferência dele para classificar 10 mil imagens de teste nunca vistas antes.
#codly-range(133, end: 147)

#figure(
  scale(
    75%,
    origin: top + left,
    reflow: true,
    raw(
      read("main.py"),
      lang: "python",
      block: true,
    ),
  ),
  caption: [Resultados main.py],
) <code:resultados>

Os resultados se encontram na @img:resultado, no meu computador a taxa de acerto foi de 99.98%, isso pode variar pois uma seed aleatória não é necessariamente reprodutível em diferentes plataformas. Alguns
exemplos aleatoriamente amostrados de classificação se encontram logo em seguida, mesmo com uma taxa de ruído considerável o modelo consegue classificar corretamente a imagem.

#figure(
  image("relatorio_print_resultado.jpeg", width: 40%),
  caption: [Resultado da inferência],
) <img:resultado>
