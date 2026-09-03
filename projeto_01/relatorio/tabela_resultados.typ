#let tabela-resultados(path, porcentagem: false, menor-melhor: false) = {
  let resultados = csv(path, row-type: dictionary)
  let cabeçalho = rgb("#E9EEF5")

  let valores-únicos(chave) = {
    let valores = ()
    for linha in resultados {
      let valor = linha.at(chave)
      if not valores.contains(valor) {
        valores.push(valor)
      }
    }
    valores
  }

  let camadas = valores-únicos("camadas")
  let ciclos = valores-únicos("ciclos")
  let taxas = valores-únicos("taxa_aprendizado")
  let momentums = valores-únicos("momentum")
  let valores = resultados.map(linha => float(linha.at("valor_validação")))
  let mínimo = calc.min(..valores)
  let máximo = calc.max(..valores)

  let taxa-label(taxa) = if taxa == "0.001" {
    "1e-03"
  } else if taxa == "0.01" {
    "1e-02"
  } else if taxa == "0.1" {
    "1e-01"
  } else {
    taxa
  }

  let valor-validação(camada, ciclo, taxa, momentum) = {
    let linha = resultados.find(linha => (
      linha.at("camadas") == camada
        and linha.at("ciclos") == ciclo
        and linha.at("taxa_aprendizado") == taxa
        and linha.at("momentum") == momentum
    ))
    float(linha.at("valor_validação"))
  }

  let qualidade(valor) = if máximo == mínimo {
    1
  } else if menor-melhor {
    (máximo - valor) / (máximo - mínimo)
  } else {
    (valor - mínimo) / (máximo - mínimo)
  }

  let cor(valor) = if qualidade(valor) >= 0.75 {
    rgb("#123E73")
  } else if qualidade(valor) >= 0.5 {
    rgb("#2F75B5")
  } else if qualidade(valor) >= 0.25 {
    rgb("#78B3D5")
  } else {
    rgb("#EAF2F8")
  }

  let formatar(valor) = if porcentagem {
    [#(calc.round(valor * 100))%]
  } else {
    [#calc.round(valor)]
  }

  let células = ()
  for camada in camadas {
    for (índice, ciclo) in ciclos.enumerate() {
      if índice == 0 {
        células.push(table.cell(
          rowspan: ciclos.len(),
          fill: cabeçalho,
        )[
          #strong[#camada]
        ])
      }
      células.push(table.cell(fill: cabeçalho)[#strong[#ciclo]])

      for momentum in momentums {
        for taxa in taxas {
          let valor = valor-validação(camada, ciclo, taxa, momentum)
          células.push(table.cell(fill: cor(valor))[
            #text(fill: if qualidade(valor) >= 0.5 { white } else { black })[
              #formatar(valor)
            ]
          ])
        }
      }
    }
  }

  table(
    columns: (1.1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: (x: 1.8pt, y: 2pt),
    stroke: white,
    table.header(
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Cam.]],
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Ciclos]],
      ..momentums.map(momentum => table.cell(
        colspan: taxas.len(),
        fill: cabeçalho,
      )[
        #strong[Momentum #momentum.replace(".", ",")]
      ]),
      ..momentums
        .map(_ => taxas.map(taxa => table.cell(fill: cabeçalho)[
          #strong[#taxa-label(taxa)]
        ]))
        .flatten(),
    ),
    ..células,
  )
}
