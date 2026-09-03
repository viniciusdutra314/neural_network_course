#let tabela-resultados(path, porcentagem: false, menor-melhor: false) = {
  let resultados = csv(path, row-type: dictionary)
  let cabeçalho = rgb("#E9EEF5")

  let valores-únicos(chave) = resultados
    .map(linha => linha.at(chave))
    .dedup()

  let camadas = valores-únicos("camadas")
  let ciclos = valores-únicos("ciclos")
  let taxas = valores-únicos("taxa_aprendizado")
  let momentums = valores-únicos("momentum")
  let valores = resultados.map(linha => float(linha.at("valor_validação")))

  let taxa-bonitinha(taxa) = if taxa == "0.001" {
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

  let mínimo = calc.min(..valores)
  let máximo = calc.max(..valores)
  let qualidade(valor) = if máximo == mínimo {
    1
  } else if menor-melhor {
    (máximo - valor) / (máximo - mínimo)
  } else {
    (valor - mínimo) / (máximo - mínimo)
  }
  // Azul escuro representa o pior valor; azul claro, o melhor.
  let escala = gradient.linear( rgb("#91bffa"),rgb("#0B3C6F"))
  let cor(valor) = escala.sample(qualidade(valor) * 100%)

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

      for taxa in taxas {
        for momentum in momentums {
          let valor = valor-validação(camada, ciclo, taxa, momentum)
          células.push(table.cell(fill: cor(valor))[
            #text(fill: if qualidade(valor) >= 0.5 { black } else { white })[
              #formatar(valor)
            ]
          ])
        }
      }
    }
  }

  table(
    columns: (1fr,) * (2 + momentums.len() * taxas.len()),
    align:  horizon,
    stroke: white,
    table.header(
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Cam.]],
      table.cell(rowspan: 2, fill: cabeçalho)[#strong[Ciclos]],
      ..taxas.map(taxa => table.cell(
        colspan: momentums.len(),
        fill: cabeçalho,
      )[
        #strong[Taxa #taxa-bonitinha(taxa)]
      ]),
      ..taxas
        .map(_ => momentums.map(momentum => table.cell(fill: cabeçalho)[
          #strong[Momentum #momentum]
        ]))
        .flatten(),
    ),
    ..células
  )
}
