#let tabela-wine(path) = {
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

  let cor(valor) = if valor >= 0.9 {
    rgb("#123E73")
  } else if valor >= 0.75 {
    rgb("#2F75B5")
  } else if valor >= 0.5 {
    rgb("#78B3D5")
  } else {
    rgb("#EAF2F8")
  }

  let texto(valor) = if valor >= 0.75 { white } else { black }
  let células = ()

  for camada in camadas {
    for (índice, ciclo) in ciclos.enumerate() {
      if índice == 0 {
        let quantidade = int(camada)
        células.push(table.cell(
          rowspan: ciclos.len(),
          fill: cabeçalho,
        )[
          #strong[#camada #if quantidade == 1 { [camada] } else { [camadas] }]
        ])
      }

      células.push(table.cell(fill: cabeçalho)[#strong[#ciclo]])

      for momentum in momentums {
        for taxa in taxas {
          let valor = valor-validação(camada, ciclo, taxa, momentum)
          células.push(table.cell(
            fill: cor(valor),
          )[
            #text(fill: texto(valor))[#(calc.round(valor * 100))%]
          ])
        }
      }
    }
  }

  table(
    columns: (1.15fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: center + horizon,
    inset: (x: 2pt, y: 2.4pt),
    stroke: white,
    table.header(
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Camadas]],
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Ciclos]],
      ..momentums.map(momentum => table.cell(
        colspan: taxas.len(),
        fill: cabeçalho,
      )[
        #strong[Momentum #momentum]
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
