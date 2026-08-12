# Exercício 01

## Objetivo

Implementar e treinar um modelo **Adaline** para reconhecer dois símbolos:

- **Y**
- **Y invertido**

## Representação dos símbolos

- Representar cada símbolo por uma matriz contendo apenas `-1` e `+1`.
- Cada valor da matriz corresponde a **uma entrada** do exemplo usado pelo modelo.
- A matriz pode ter dimensão `5x5` ou maior.
- Os símbolos devem poder ser desenhados/visualizados graficamente a partir dessa representação.

Exemplo de representação `5x5` para **Y**:

```text
+1  -1  -1  -1  +1
+1  -1  -1  -1  +1
-1  +1  +1  +1  -1
-1  -1  +1  -1  -1
-1  -1  +1  -1  -1
```

## Dados

- Criar exemplos para **treinamento e teste**.
- Inserir **ruído arbitrário** nos exemplos.
- Todos os exemplos devem ser rotulados.
- Usar um rótulo para `Y` e o outro para `Y invertido`, por exemplo:
  - `-1` → Y
  - `+1` → Y invertido
- Criar **no mínimo 6 exemplos por classe**:
  - 6 de Y
  - 6 de Y invertido
  - **mínimo total: 12 exemplos**

## O programa deve

1. Gerar/carregar os exemplos de Y e Y invertido.
2. Adicionar ruído aos exemplos.
3. Treinar o modelo Adaline.
4. Avaliar o modelo nos conjuntos de **treinamento e teste**.
5. Mostrar os resultados obtidos.
6. Permitir visualizar graficamente os símbolos representados pelas matrizes.

## Entrega

Entregar **um único arquivo `.zip` ou `.rar`** com nome:

```text
<seu_nome>_exercicio1.zip
```

ou

```text
<seu_nome>_exercicio1.rar
```

O arquivo deve conter:

- **Código-fonte comentado**.
- **Exemplos criados para uso como entradas**.
- **Relatório de 1 a 2 páginas**, descrevendo:
  - o que foi feito;
  - os resultados no conjunto de treinamento;
  - os resultados no conjunto de teste.
