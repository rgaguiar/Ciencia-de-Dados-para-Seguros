# Módulo 00.01 — Python para Análise: pandas, numpy, matplotlib e seaborn

> **Curso:** Ciência de Dados e ML para Seguros  
> **Fase:** 1 — Fundamentos  
> **Módulo:** 00 — Nivelamento: Python e Estatística para Seguros | **Item:** 01 de 06  
> **Nível:** Iniciante — sem pré-requisitos  
> **Ferramentas:** Python 3.10+, pandas 2.x, numpy 1.26+, matplotlib 3.x, seaborn 0.13+

---

## Por que começamos aqui?

Toda análise em seguros começa com dados. Dados de apólices, sinistros, endossos, cancelamentos — volumes que nenhuma planilha do Excel suporta sem travar. Quando uma seguradora fecha o mês, o time de dados está lendo arquivos com milhões de linhas, cruzando tabelas e calculando indicadores que vão direto para o comitê de subscrição. Isso não se faz com fórmulas de célula.

Python se tornou a língua franca da análise de dados porque resolveu exatamente esse problema. Não porque é "moderno" — mas porque o ecossistema científico que surgiu em torno dele oferece a capacidade de um time de engenharia de dados na mão de um analista com um notebook. pandas, numpy, matplotlib e seaborn são as quatro ferramentas que tornam isso possível.

O pandas é o eixo central: representa dados tabulares da forma que um analista de seguros reconhece — linhas são registros (apólices, sinistros), colunas são atributos (prêmio, ramo, data de vigência). O numpy opera nos bastidores, executando cálculos matemáticos em velocidade que o Python puro nunca atingiria. matplotlib e seaborn transformam números em gráficos que o diretor consegue ler — e que você constrói em três linhas.

Este módulo não é uma introdução genérica a Python. É um tour pelas ferramentas com o olhar de quem vai trabalhar com carteiras de seguro — os exemplos são de apólices, os datasets têm prêmios e sinistros, e os exercícios pedem métricas que fazem sentido no setor.

Se você já conhece pandas, use este módulo para confirmar que domina os padrões que o curso vai exigir. Se está chegando agora, leia com calma — cada bloco de código tem algo que vai reaparecer no Módulo 01.

---

## Vocabulário

| Termo | Definição | Exemplo no curso |
|---|---|---|
| **DataFrame** | Estrutura tabular do pandas — linhas × colunas | Carteira de apólices com 500 mil registros |
| **Series** | Coluna única de um DataFrame | Coluna `premio` ou `sinistros` |
| **ndarray** | Array multidimensional do numpy | Vetor de prêmios para cálculo vetorizado |
| **dtype** | Tipo de dado de cada coluna | `float64` para prêmio, `object` para ramo |
| **axis** | Dimensão de operação: 0 = linhas, 1 = colunas | `df.mean(axis=0)` calcula a média por coluna |
| **Figure / Axes** | Estrutura do matplotlib: Figure é a tela, Axes é o gráfico | Um Figure pode ter múltiplos Axes (subplots) |

---

## Seção 1 — pandas: a carteira de seguros em forma de DataFrame

O pandas representa dados tabulares com o objeto `DataFrame`. Pense nele como uma planilha com superpoderes: filtragem condicional, agrupamentos, joins e transformações em uma linha de código.

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

carteira = pd.DataFrame({
    "apolice_id": range(1, n + 1),
    "ramo": np.random.choice(["auto", "residencial", "vida"], size=n, p=[0.5, 0.3, 0.2]),
    "premio": np.random.lognormal(mean=7.0, sigma=0.5, size=n),  # 💡 lognormal é mais realista para prêmios
    "exposicao": np.random.uniform(0.1, 1.0, size=n),
    "sinistros": np.random.poisson(lam=0.3, size=n),
})

print(carteira.shape)
print(carteira.dtypes)
print(carteira.head())
```

Os métodos mais usados no dia a dia:

```python
# visão geral estatística da carteira
print(carteira.describe())

# filtrar apenas apólices com sinistro
com_sinistro = carteira[carteira["sinistros"] > 0]
print(f"Apólices com sinistro: {len(com_sinistro)} ({len(com_sinistro) / n:.1%})")

# prêmio médio por ramo
premio_medio = carteira.groupby("ramo")["premio"].mean().round(2)
print(premio_medio)
```

> 📌 **Conexão com seguros:** `groupby` é a operação central da análise tarifária. Quase toda pergunta de sinistralidade começa com "por ramo", "por faixa etária", "por região" — e o groupby resolve isso em uma linha.

O pandas também facilita a criação de variáveis derivadas — essencial para calcular indicadores atuariais:

```python
carteira["frequencia"] = carteira["sinistros"] / carteira["exposicao"]

# ⚠️ exposicao zero gera divisão por infinito — trate antes se houver registros com exposicao = 0
carteira["loss_ratio"] = (carteira["sinistros"] * carteira["premio"].mean()) / carteira["premio"]

print(carteira[["apolice_id", "ramo", "frequencia", "loss_ratio"]].head(10))
```

### 🏋️ Exercício 1

1. Usando a `carteira` criada acima, calcule o **total de prêmio emitido** por ramo.
2. Filtre as apólices com `frequencia > 1.0` (mais de um sinistro por unidade de exposição) e mostre quantas existem em cada ramo.
3. Crie uma nova coluna chamada `faixa_premio` usando `pd.cut` com 4 faixas iguais e conte quantas apólices existem em cada faixa.

---

## Seção 2 — numpy: operações vetorizadas sobre dados de seguro

O numpy não aparece muito no código do dia a dia — mas está em todo lugar por baixo do pandas. Entender o que é um array e como ele opera é o que separa quem escreve código que trava de quem escreve código que escala.

```python
premios = np.array([1200.0, 3400.0, 800.0, 5600.0, 2100.0])
exposicoes = np.array([1.0, 0.75, 0.5, 1.0, 0.25])

# operações vetorizadas: sem loop, sem iteração manual
premios_anualizados = premios / exposicoes
print("Prêmios anualizados:", premios_anualizados.round(2))

print(f"Média:         {premios_anualizados.mean():.2f}")
print(f"Mediana:       {np.median(premios_anualizados):.2f}")
print(f"Desvio padrão: {premios_anualizados.std():.2f}")
```

> 💡 **Ponto crítico:** numpy opera em C por baixo dos panos. Um loop `for` em Python para calcular frequência em 1 milhão de apólices pode levar 10 segundos. A operação vetorizada faz o mesmo em milissegundos.

Funções numpy que aparecem com frequência em análise atuarial:

```python
np.random.seed(42)

sinistros_simulados = np.random.poisson(lam=0.3, size=10_000)

print(f"Total de sinistros: {sinistros_simulados.sum()}")
print(f"Apólices sem sinistro: {(sinistros_simulados == 0).sum()} ({(sinistros_simulados == 0).mean():.1%})")
print(f"Percentis [50, 75, 90, 99]: {np.percentile(sinistros_simulados, [50, 75, 90, 99])}")
```

### 🏋️ Exercício 2

1. Gere um array de 5.000 prêmios com distribuição lognormal (`mean=7.5, sigma=0.6, seed=42`). Calcule média, mediana e o percentil 95.
2. Crie um segundo array de exposições com distribuição uniforme entre 0.3 e 1.0. Calcule o **prêmio anualizado** (prêmio / exposição) e compare a média com o array original.
3. Simule 5.000 contagens de sinistro com Poisson (`lam=0.25`). Qual é a proporção de apólices com **2 ou mais sinistros**?

---

## Seção 3 — matplotlib e seaborn: de número a decisão

Gráfico em seguros não é decoração. É o que convence o subscritor a ajustar uma tarifa, o que mostra ao comitê que o modelo funciona, o que entrega ao regulador a evidência que ele pede. Saber construir um gráfico claro e rápido é tão importante quanto saber calcular a métrica.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(carteira["premio"], bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("Distribuição de Prêmios")
axes[0].set_xlabel("Prêmio (R$)")
axes[0].set_ylabel("Frequência")

premio_ramo = carteira.groupby("ramo")["premio"].mean().reset_index()
axes[1].bar(premio_ramo["ramo"], premio_ramo["premio"], color="coral", edgecolor="white")
axes[1].set_title("Prêmio Médio por Ramo")
axes[1].set_xlabel("Ramo")
axes[1].set_ylabel("Prêmio Médio (R$)")

plt.tight_layout()
plt.savefig("distribuicao_premios_por_ramo.png", dpi=150, bbox_inches="tight")
plt.show()
```

O seaborn facilita gráficos mais analíticos com menos código:

```python
fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(data=carteira, x="ramo", y="premio", ax=ax)
ax.set_title("Dispersão de Prêmios por Ramo")
ax.set_xlabel("Ramo")
ax.set_ylabel("Prêmio (R$)")

plt.tight_layout()
plt.savefig("boxplot_premios_ramo.png", dpi=150, bbox_inches="tight")
plt.show()
```

> ⚠️ **Atenção:** gráfico sem título, sem label de eixo e sem unidade volta com comentário. Sempre inclua os três — quem lê o relatório não conhece o dataset de cabeça.

### 🏋️ Exercício 3

1. Crie um **histograma** da coluna `frequencia` da carteira (use 30 bins). Inclua título, labels de eixo e salve como `histograma_frequencia.png`.
2. Usando seaborn, plote um **gráfico de barras** com o número total de sinistros por ramo. Inclua todos os elementos visuais obrigatórios.
3. **Desafio:** plote um **scatter plot** com `premio` no eixo X e `frequencia` no eixo Y, colorindo os pontos por `ramo`. Use `sns.scatterplot` com `hue="ramo"` e interprete visualmente o resultado.

---

## Resumo do módulo

| Conceito | O que aprendemos |
|---|---|
| **DataFrame** | Estrutura central do pandas para dados tabulares — carteiras, sinistros, apólices |
| **Series** | Coluna de um DataFrame — opera como array com índice |
| **groupby** | Agrupamento por categoria — base da análise por ramo, faixa, região |
| **ndarray** | Array numpy — motor dos cálculos vetorizados que escalam para milhões de registros |
| **Operações vetorizadas** | Frequência, loss ratio e anualização sem loop, em uma linha |
| **matplotlib** | Controle total sobre figuras e eixos — histogramas, barras, scatter |
| **seaborn** | Interface de alto nível — boxplot, violinplot, pairplot com menos código |
| **sns.set_theme** | Padrão visual do curso: `style="whitegrid", palette="muted"` |

---

## Próxima aula

No item 02, entramos em **Estatística Descritiva para Seguros**: média, mediana, variância, assimetria e curtose aplicados a dados de prêmio e sinistro. Vamos entender por que a distribuição de prêmios é assimétrica por construção e o que isso significa para a escolha do modelo.

---

## Referências e leitura complementar

- Documentação oficial do pandas: pandas.pydata.org/docs
- Guia do numpy para usuários: numpy.org/doc/stable/user
- Galeria de exemplos do seaborn: seaborn.pydata.org/tutorial
- McKinney, W. *Python for Data Analysis*, 3ª ed. — O'Reilly (livro de referência do pandas)
- VanderPlas, J. *Python Data Science Handbook* — disponível gratuitamente online

---

*Curso de Ciência de Dados e ML para Seguros — Módulo 00.01 v1.0*
