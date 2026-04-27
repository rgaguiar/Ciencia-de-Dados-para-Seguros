# Módulo 00.02 — Estatística Descritiva para Seguros

> **Curso:** Ciência de Dados e ML para Seguros  
> **Fase:** 1 — Fundamentos  
> **Módulo:** 00 — Nivelamento: Python e Estatística para Seguros | **Item:** 02 de 06  
> **Nível:** Iniciante-Intermediário  
> **Pré-requisito:** Módulo 00.01 — Python para Análise  
> **Ferramentas:** Python 3.10+, pandas, numpy, scipy, matplotlib, seaborn  
> **Dataset:** `dados/raw/carteira_sintetica.csv`

---

## Por que começamos aqui?

Uma seguradora que conhece apenas a média dos seus sinistros conhece muito pouco. A média diz onde está o centro — mas em seguros, o centro raramente é o problema. O problema está nas caudas: os sinistros grandes, raros e caros que nenhuma média antecipa. Uma carteira com sinistro médio de R$ 5.000 pode ser muito segura ou extremamente perigosa, dependendo de como esses sinistros se distribuem ao redor desse centro.

É por isso que a estatística descritiva em seguros vai além da média e do desvio padrão. O atuário precisa entender a **forma** da distribuição — se ela é simétrica ou assimétrica, se tem cauda pesada ou leve, se a mediana está próxima ou distante da média. Cada uma dessas propriedades tem implicação direta na escolha do modelo de precificação, no cálculo da provisão e na decisão de resseguro.

A assimetria positiva é uma propriedade estrutural dos dados de seguro, não uma anomalia. Prêmios, sinistros e severidade são sempre distribuídos com cauda à direita — há um piso natural em zero e nenhum teto para cima. Quem trata esses dados como se fossem normalmente distribuídos está cometendo um erro que vai se propagar por toda a cadeia de modelagem.

A curtose, por sua vez, governa o peso das caudas. Uma distribuição com curtose alta produz mais eventos extremos do que a normal sugere. Em resseguro, onde o negócio é exatamente cobrir esses extremos, ignorar a curtose significa subprecificar risco catastrófico.

Este módulo constrói o vocabulário estatístico que o curso inteiro vai usar. Cada conceito aqui vai reaparecer — na escolha da função de ligação do GLM, na avaliação de resíduos, na definição de métricas de modelo.

---

## Vocabulário

| Termo | Definição | Exemplo no curso |
|---|---|---|
| **Média** | Soma dos valores dividida pela quantidade | Prêmio médio da carteira |
| **Mediana** | Valor central que divide a distribuição ao meio | Mediana de sinistros — mais robusta que a média |
| **Variância** | Média dos quadrados dos desvios em relação à média | Variância alta = carteira heterogênea |
| **Desvio padrão** | Raiz quadrada da variância — mesma unidade dos dados | Desvio de R$ 3.200 no prêmio |
| **CV (Coef. de Variação)** | Desvio padrão / média — mede dispersão relativa | CV = 0.8 indica alta heterogeneidade |
| **Assimetria** | Mede o desvio da simetria — positiva = cauda à direita | Sinistros têm assimetria positiva por construção |
| **Curtose** | Mede o peso das caudas em relação à normal | Curtose alta = mais eventos extremos |

---

## Seção 1 — Medidas de posição: média e mediana

Média e mediana respondem à mesma pergunta — "onde está o centro?" — mas de formas muito diferentes. Em distribuições simétricas, as duas coincidem. Em distribuições assimétricas, divergem — e é exatamente essa divergência que diz algo importante sobre os dados.

```python
import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)

df = pd.read_csv("dados/raw/carteira_sintetica.csv")

media   = df["premio"].mean()
mediana = df["premio"].median()
moda    = df["premio"].mode()[0]

print(f"Média:   R$ {media:,.2f}")
print(f"Mediana: R$ {mediana:,.2f}")
print(f"Moda:    R$ {moda:,.2f}")
print(f"Razão média/mediana: {media/mediana:.3f}")
```

> 💡 **Ponto crítico:** quando média > mediana, a distribuição tem cauda à direita — alguns valores muito altos puxam a média para cima. Em seguros, a razão média/mediana é um diagnóstico rápido de assimetria. Valores acima de 1.2 indicam distorção significativa.

A média é sensível a outliers; a mediana, não. Em seguros, isso tem consequência direta na análise de rentabilidade:

```python
# compara média e mediana por ramo
posicao_ramo = df.groupby("ramo")["premio"].agg(
    media   = "mean",
    mediana = "median",
    razao   = lambda x: x.mean() / x.median(),
).round(2)

print(posicao_ramo)

# o mesmo para sinistros — onde a distorção costuma ser maior
posicao_sinistro = df.groupby("ramo")["valor_sinistro"].agg(
    media   = "mean",
    mediana = "median",
    razao   = lambda x: x.mean() / x.median(),
).round(2)

print(posicao_sinistro)
```

> 📌 **Conexão com seguros:** a mediana do valor de sinistro por ramo é usada em análise de frequência × severidade para segmentar a carteira. Ramos com razão média/mediana acima de 2.0 geralmente justificam o uso de distribuição lognormal na modelagem de severidade.

### 🏋️ Exercício 1

1. Calcule média e mediana de `premio` por `perfil` (`jovem`, `adulto`, `senior`). Em qual perfil a razão média/mediana é maior? O que isso sugere sobre a heterogeneidade desse grupo?
2. Filtre apenas apólices com `sinistros > 0` e calcule média e mediana de `valor_sinistro` por `regiao`. Qual região tem maior divergência entre as duas medidas?
3. Calcule a média **ponderada** do prêmio pela exposição: `(df["premio"] * df["exposicao"]).sum() / df["exposicao"].sum()`. Compare com a média simples. Por que a média ponderada é mais adequada para análise tarifária?

---

## Seção 2 — Dispersão: variância, desvio padrão e coeficiente de variação

Saber onde está o centro não é suficiente. Duas carteiras com o mesmo prêmio médio podem ter perfis de risco completamente diferentes se uma for homogênea e a outra, heterogênea. A dispersão mede essa heterogeneidade.

```python
variancia  = df["premio"].var()
desvio     = df["premio"].std()
cv         = desvio / df["premio"].mean()  # 💡 CV permite comparar dispersão entre variáveis de escalas diferentes

print(f"Variância:            {variancia:,.2f}")
print(f"Desvio padrão:        R$ {desvio:,.2f}")
print(f"Coeficiente de var.:  {cv:.3f} ({cv:.1%})")
```

O coeficiente de variação (CV) é especialmente útil para comparar dispersão entre ramos com escalas de prêmio muito diferentes:

```python
dispersao_ramo = df.groupby("ramo")["premio"].agg(
    media  = "mean",
    desvio = "std",
    cv     = lambda x: x.std() / x.mean(),
    p25    = lambda x: x.quantile(0.25),
    p75    = lambda x: x.quantile(0.75),
    iqr    = lambda x: x.quantile(0.75) - x.quantile(0.25),
).round(3)

print(dispersao_ramo)
```

> ⚠️ **Atenção:** variância e desvio padrão são sensíveis a outliers — um único sinistro catastrófico pode dobrar o desvio padrão da carteira. O IQR (intervalo interquartílico = P75 − P25) é uma medida de dispersão robusta que ignora os extremos e deve ser usado junto com o desvio padrão.

Visualizar a dispersão por segmento revela rapidamente onde a carteira é mais heterogênea:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.boxplot(data=df, x="ramo", y="premio", ax=axes[0])
axes[0].set_title("Dispersão de Prêmio por Ramo")
axes[0].set_xlabel("Ramo")
axes[0].set_ylabel("Prêmio (R$)")

sns.violinplot(data=df, x="perfil", y="premio", ax=axes[1], order=["jovem", "adulto", "senior"])
axes[1].set_title("Distribuição de Prêmio por Perfil")
axes[1].set_xlabel("Perfil")
axes[1].set_ylabel("Prêmio (R$)")

plt.tight_layout()
plt.savefig("dispersao_premio_ramo_perfil.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 🏋️ Exercício 2

1. Calcule variância, desvio padrão e CV de `valor_sinistro` por `ramo`. Qual ramo tem maior heterogeneidade relativa (CV mais alto)?
2. Calcule o IQR de `premio` por `perfil`. O perfil `jovem` tem IQR maior ou menor que `senior`? O que isso indica sobre a homogeneidade tarifária de cada grupo?
3. Plote um boxplot de `valor_sinistro` por `ramo` (filtrando apenas apólices com `sinistros > 0`). Salve como `boxplot_severidade_ramo.png`. Qual ramo apresenta mais outliers visíveis?

---

## Seção 3 — Forma da distribuição: assimetria e curtose

Média e desvio padrão descrevem o centro e a dispersão — mas nada dizem sobre a forma. Duas distribuições podem ter a mesma média e o mesmo desvio padrão e ainda assim serem completamente diferentes na prática. Assimetria e curtose capturam essas diferenças de forma.

```python
from scipy.stats import skew, kurtosis

assimetria = skew(df["premio"])
curtose    = kurtosis(df["premio"])  # curtose em excesso — normal = 0

print(f"Assimetria: {assimetria:.3f}")
print(f"Curtose:    {curtose:.3f}")

# compara entre variáveis-chave da carteira
for col in ["premio", "valor_sinistro", "exposicao"]:
    s = skew(df[col].dropna())
    k = kurtosis(df[col].dropna())
    print(f"{col:<20} assimetria: {s:6.3f}  curtose: {k:7.3f}")
```

> 💡 **Ponto crítico:** assimetria positiva (> 0) indica cauda à direita — há mais valores extremos acima da média do que abaixo. Prêmios e sinistros têm assimetria positiva por definição (não podem ser negativos). Modelos que assumem simetria — como regressão linear com erros normais — são inadequados para esses dados.

Curtose em excesso acima de zero indica caudas mais pesadas que a normal — mais eventos extremos do que se esperaria:

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# prêmio: assimetria e curtose visíveis
axes[0].hist(df["premio"], bins=60, color="steelblue", edgecolor="white", density=True)
axes[0].axvline(df["premio"].mean(),   color="red",    linestyle="--", label=f"Média: {df['premio'].mean():,.0f}")
axes[0].axvline(df["premio"].median(), color="orange", linestyle="--", label=f"Mediana: {df['premio'].median():,.0f}")
axes[0].set_title(f"Distribuição de Prêmio  |  assim={skew(df['premio']):.2f}  kurt={kurtosis(df['premio']):.2f}")
axes[0].set_xlabel("Prêmio (R$)")
axes[0].set_ylabel("Densidade")
axes[0].legend()

# log do prêmio: aproxima a normalidade
axes[1].hist(np.log(df["premio"]), bins=60, color="coral", edgecolor="white", density=True)
axes[1].set_title(f"log(Prêmio)  |  assim={skew(np.log(df['premio'])):.2f}  kurt={kurtosis(np.log(df['premio'])):.2f}")
axes[1].set_xlabel("log(Prêmio)")
axes[1].set_ylabel("Densidade")

plt.tight_layout()
plt.savefig("assimetria_curtose_premio.png", dpi=150, bbox_inches="tight")
plt.show()
```

> 📌 **Conexão com seguros:** a transformação logarítmica reduz assimetria e curtose porque comprime a cauda direita. É por isso que GLMs para severidade usam a família Gamma com função de ligação log — o modelo opera na escala log onde a distribuição é mais comportada, e os coeficientes são interpretados como multiplicadores na escala original.

### 🏋️ Exercício 3

1. Calcule assimetria e curtose de `valor_sinistro` para cada `ramo` separadamente. Qual ramo tem a cauda mais pesada (curtose mais alta)?
2. Aplique a transformação `log1p` em `valor_sinistro` e recalcule assimetria e curtose. A transformação reduziu a assimetria? Em quanto?
3. **Desafio:** crie um painel com 4 subplots (2×2) mostrando a distribuição de `premio` e `valor_sinistro` nas escalas original e log. Para cada gráfico, inclua no título os valores de assimetria e curtose. Salve como `painel_forma_distribuicao.png`.

---

## Resumo do módulo

| Conceito | O que aprendemos |
|---|---|
| **Média** | Sensível a outliers — puxada pelos valores extremos da cauda direita |
| **Mediana** | Robusta a outliers — melhor representante do "cliente típico" em seguros |
| **Razão média/mediana** | Diagnóstico rápido de assimetria — acima de 1.2 indica distorção significativa |
| **Variância / Desvio padrão** | Medem a dispersão absoluta — sensíveis a outliers |
| **CV (Coef. de Variação)** | Dispersão relativa — permite comparar ramos com escalas diferentes |
| **IQR** | Dispersão robusta — ignora os extremos, útil para carteiras heterogêneas |
| **Assimetria** | Positiva em prêmios e sinistros — propriedade estrutural, não anomalia |
| **Curtose** | Caudas pesadas = mais eventos extremos — relevante para resseguro e provisão |

---

## Próxima aula

O item 03 apresenta as **distribuições de probabilidade** usadas em seguros: normal, exponencial, Poisson, Pareto e lognormal. Vamos conectar as propriedades de assimetria e curtose que vimos aqui com a escolha da distribuição correta para modelar prêmios, frequência e severidade.

---

## Referências e leitura complementar

- Bowers, N.L. et al. *Actuarial Mathematics*, 2ª ed. — Society of Actuaries (capítulo 2)
- Klugman, S.A. et al. *Loss Models: From Data to Decisions*, 4ª ed. — Wiley (referência de distribuições para seguros)
- Documentação do scipy.stats: distribuições e testes estatísticos — docs.scipy.org/doc/scipy/reference/stats
- Documentação do pandas: métodos de estatística descritiva — pandas.pydata.org/docs/reference/frame

---

*Curso de Ciência de Dados e ML para Seguros — Módulo 00.02 v1.0*
