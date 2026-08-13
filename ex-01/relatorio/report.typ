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
