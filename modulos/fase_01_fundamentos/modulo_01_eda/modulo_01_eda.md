# Módulo 01 — Análise Exploratória de Dados (EDA) em Seguros

> **Curso:** Ciência de Dados e ML para Seguros  
> **Fase:** 1 — Fundamentos  
> **Módulo:** 01 — Análise Exploratória de Dados (EDA) em Seguros  
> **Nível:** Iniciante-Intermediário  
> **Pré-requisito:** Módulo 00 — Python para Análise  
> **Ferramentas:** Python 3.10+, pandas, numpy, matplotlib, seaborn  
> **Dataset:** `dados/raw/carteira_sintetica.csv`

---

## Por que começamos aqui?

Antes de qualquer modelo, existe uma carteira. E toda carteira esconde problemas que nenhum algoritmo resolve sozinho: apólices sem exposição registrada, sinistros com valor zero, campos de data preenchidos errado, concentrações de risco que distorcem qualquer média. Quem pula a EDA e vai direto ao modelo está construindo em cima de areia — e só descobre isso quando o modelo volta com resultados absurdos em produção.

A Análise Exploratória de Dados em seguros tem um propósito muito mais cirúrgico do que em outros domínios. Não se trata de "conhecer os dados" em sentido genérico — trata-se de responder perguntas específicas antes de qualquer modelagem: a exposição está correta? A frequência por segmento faz sentido atuarial? Há sazonalidade no comportamento de sinistros? Os valores extremos são erros de digitação ou riscos reais?

A exposição merece atenção especial. Em seguros, uma apólice vigente por seis meses contribui metade do risco de uma apólice anual. Ignorar a exposição e calcular frequência como "sinistros / número de apólices" é um erro clássico que distorce toda a análise tarifária. A EDA é o momento de verificar se a exposição está registrada corretamente e se os indicadores derivados dela fazem sentido.

A análise bivariada — sinistralidade por variável de risco — é a ponte entre a EDA e a modelagem. Ela revela quais variáveis têm poder discriminante antes de qualquer GLM ou gradient boosting. Uma variável que não mostra diferença de frequência entre seus níveis dificilmente vai contribuir no modelo. Identificar isso na EDA poupa tempo e evita overfitting por inclusão de variáveis irrelevantes.

Este módulo entrega a base do projeto integrador: a EDA completa da carteira sintética que vai alimentar todos os modelos das fases seguintes.

---

## Vocabulário

| Termo | Definição | Exemplo no curso |
|---|---|---|
| **Exposição** | Tempo de vigência de uma apólice em unidades de ano | Apólice vigente 6 meses = 0.5 de exposição |
| **Ano-apólice** | Unidade de exposição equivalente a uma apólice por um ano completo | Carteira com 10.000 ano-apólices |
| **Earned premium** | Prêmio proporcional ao período efetivamente decorrido | Prêmio de R$ 1.200 por 6 meses = R$ 600 earned |
| **Frequência** | Número de sinistros por unidade de exposição | 0.3 sinistros por ano-apólice |
| **Severidade** | Valor médio por sinistro | R$ 8.500 por sinistro |
| **Loss ratio** | Razão entre sinistros pagos e prêmio ganho | Sinistros / Prêmio earned = 0.65 (65%) |
| **Zero inflado** | Distribuição com excesso de zeros além do esperado por Poisson | 80% das apólices sem sinistro em risco de baixa freq. |
| **Outlier** | Observação muito distante da distribuição central | Sinistro de R$ 2 milhões em carteira de auto popular |

---

## Seção 1 — Diagnóstico da carteira: qualidade antes de tudo

O primeiro contato com qualquer dataset de seguros deve ser um diagnóstico sistemático — tipos de dado, completude, distribuição básica e anomalias visíveis. Problemas encontrados aqui precisam ser documentados e tratados antes de qualquer cálculo de indicador.

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# dataset sintético da carteira — gerado aqui enquanto o módulo 00 não foi concluído
carteira = pd.DataFrame({
    "apolice_id":    range(1, n + 1),
    "ramo":          np.random.choice(["auto", "residencial", "vida"], size=n, p=[0.5, 0.3, 0.2]),
    "regiao":        np.random.choice(["SE", "S", "NE", "N", "CO"], size=n, p=[0.4, 0.25, 0.2, 0.1, 0.05]),
    "perfil":        np.random.choice(["jovem", "adulto", "senior"], size=n, p=[0.25, 0.55, 0.20]),
    "premio":        np.random.lognormal(mean=7.2, sigma=0.6, size=n),  # 💡 lognormal reflete assimetria real de prêmios
    "exposicao":     np.random.uniform(0.1, 1.0, size=n),
    "sinistros":     np.random.poisson(lam=0.28, size=n),
    "valor_sinistro": np.random.lognormal(mean=8.5, sigma=0.8, size=n),
})

# introduz problemas reais de qualidade
idx_missing = np.random.choice(n, size=120, replace=False)
carteira.loc[idx_missing, "exposicao"] = np.nan

idx_outlier = np.random.choice(n, size=8, replace=False)
carteira.loc[idx_outlier, "valor_sinistro"] = carteira["valor_sinistro"] * 40

carteira.to_csv("dados/raw/carteira_sintetica.csv", index=False)
print(carteira.shape)
print(carteira.dtypes)
```

Com o dataset carregado, o diagnóstico segue três camadas: completude, distribuição e consistência lógica.

```python
df = pd.read_csv("dados/raw/carteira_sintetica.csv")

# completude — missing por coluna em percentual
missing = df.isnull().mean().mul(100).round(2)
print("Missing (%):\n", missing[missing > 0])

# distribuição básica das variáveis numéricas
print(df.describe().round(2))

# consistência lógica — exposição deve estar entre 0 e 1
fora_range = df[(df["exposicao"] < 0) | (df["exposicao"] > 1)]
print(f"Exposição fora do intervalo [0,1]: {len(fora_range)} registros")

# zeros inflados em sinistros
zero_pct = (df["sinistros"] == 0).mean()
print(f"Apólices sem sinistro: {zero_pct:.1%}")
```

> 📌 **Conexão com seguros:** a proporção de apólices sem sinistro é um indicador estrutural. Em auto popular, valores acima de 70% são normais. Se estiver abaixo de 50%, ou a carteira tem problemas de registro ou a frequência está anormalmente alta — ambos os casos pedem investigação antes de modelar.

Outliers em `valor_sinistro` precisam de tratamento cuidadoso — podem ser erros de digitação ou sinistros catastróficos legítimos:

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# filtra apenas apólices com sinistro para análise de severidade
com_sinistro = df[df["sinistros"] > 0].copy()

p95 = com_sinistro["valor_sinistro"].quantile(0.95)
p99 = com_sinistro["valor_sinistro"].quantile(0.99)
print(f"P95: R$ {p95:,.0f} | P99: R$ {p99:,.0f} | Máx: R$ {com_sinistro['valor_sinistro'].max():,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(com_sinistro["valor_sinistro"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Distribuição de Valor por Sinistro")
axes[0].set_xlabel("Valor (R$)")
axes[0].set_ylabel("Frequência")

axes[1].hist(np.log1p(com_sinistro["valor_sinistro"]), bins=50, color="coral", edgecolor="white")
axes[1].set_title("Distribuição de Valor por Sinistro (log)")
axes[1].set_xlabel("log(Valor + 1)")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.savefig("distribuicao_valor_sinistro.png", dpi=150, bbox_inches="tight")
plt.show()
```

> ⚠️ **Atenção:** nunca remova outliers de valor de sinistro sem validação. Um sinistro de R$ 2 milhões pode ser um erro de digitação ou um evento catastrófico real. A decisão de winsorizar, separar ou manter afeta diretamente a severidade modelada.

### 🏋️ Exercício 1

1. Calcule o percentual de missing para cada coluna e apresente apenas as colunas com missing > 0 em ordem decrescente.
2. Identifique as apólices onde `valor_sinistro` supera o percentil 99. Quantas são? Qual o ramo com maior concentração desses outliers?
3. Verifique se há apólices com `sinistros > 0` mas `valor_sinistro` igual a zero ou nulo. Esse padrão indica qual tipo de problema de qualidade?

---

## Seção 2 — Exposição e indicadores atuariais

Com a qualidade dos dados mapeada, o próximo passo é calcular os indicadores que vão aparecer em todo o restante do curso: frequência, severidade e loss ratio — sempre relativizados pela exposição.

```python
# trata missing de exposição antes de calcular indicadores
df["exposicao"] = df["exposicao"].fillna(df["exposicao"].median())

# ⚠️ exposição zero gera divisão por infinito — substitua antes de dividir
df["exposicao"] = df["exposicao"].replace(0, np.nan)

df["frequencia"]  = df["sinistros"] / df["exposicao"]
df["earned_prem"] = df["premio"] * df["exposicao"]  # 💡 prêmio ganho proporcional à exposição

# severidade só faz sentido onde houve sinistro
df["severidade"] = np.where(
    df["sinistros"] > 0,
    df["valor_sinistro"] / df["sinistros"],
    np.nan
)

df["loss_ratio"] = (df["sinistros"] * df["severidade"].fillna(0)) / df["earned_prem"]

print(df[["frequencia", "severidade", "loss_ratio"]].describe().round(3))
```

Indicadores por ramo revelam o perfil de risco de cada segmento:

```python
indicadores = df.groupby("ramo").agg(
    apolices        = ("apolice_id", "count"),
    exposicao_total = ("exposicao", "sum"),
    sinistros_total = ("sinistros", "sum"),
    earned_prem     = ("earned_prem", "sum"),
    valor_total     = ("valor_sinistro", "sum"),
).assign(
    frequencia  = lambda x: x["sinistros_total"] / x["exposicao_total"],
    severidade  = lambda x: x["valor_total"] / x["sinistros_total"],
    loss_ratio  = lambda x: x["valor_total"] / x["earned_prem"],
).round(3)

print(indicadores[["apolices", "frequencia", "severidade", "loss_ratio"]])
```

> 📌 **Conexão com seguros:** o loss ratio agregado por ramo é o primeiro diagnóstico de rentabilidade. Um ramo com loss ratio > 1.0 está gerando mais sinistros do que prêmio — ou a tarifa está errada, ou houve evento atípico, ou há seleção adversa na carteira.

### 🏋️ Exercício 2

1. Calcule frequência, severidade e loss ratio por **perfil** (`jovem`, `adulto`, `senior`). Qual perfil apresenta maior frequência? E maior severidade?
2. Calcule o **earned premium total** por região e compare com o valor total de sinistros. Quais regiões têm loss ratio acima de 0.7?
3. Filtre apenas o ramo `auto` e calcule os indicadores por combinação de `perfil` × `regiao` usando `groupby`. Identifique o segmento com maior loss ratio.

---

## Seção 3 — Análise bivariada, sazonalidade e feature engineering

A análise bivariada responde a pergunta que o modelo vai tentar responder: qual variável discrimina melhor o risco? Antes de qualquer GLM, a visualização da frequência por nível de cada variável já indica o poder preditivo esperado.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

variaveis = ["ramo", "perfil", "regiao"]
for ax, var in zip(axes, variaveis):
    ordem = df.groupby(var)["frequencia"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x=var, y="frequencia", order=ordem, ax=ax, errorbar="sd")
    ax.set_title(f"Frequência média por {var}")
    ax.set_xlabel(var.capitalize())
    ax.set_ylabel("Frequência (sinistros / exposição)")

plt.tight_layout()
plt.savefig("frequencia_por_variavel_risco.png", dpi=150, bbox_inches="tight")
plt.show()
```

Sazonalidade em séries de sinistros é um padrão estrutural em muitos ramos — chuvas intensas no verão para residencial, final de ano para auto. Para visualizar, precisamos de uma dimensão temporal:

```python
np.random.seed(42)

datas = pd.date_range(start="2022-01-01", end="2023-12-31", periods=n)
df["data_inicio"] = np.random.choice(datas, size=n)
df["mes"] = pd.to_datetime(df["data_inicio"]).dt.month

sinistros_mes = df.groupby("mes").agg(
    sinistros_total = ("sinistros", "sum"),
    exposicao_total = ("exposicao", "sum"),
).assign(frequencia = lambda x: x["sinistros_total"] / x["exposicao_total"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sinistros_mes.index, sinistros_mes["frequencia"], marker="o", color="steelblue")
ax.set_title("Frequência de Sinistros por Mês")
ax.set_xlabel("Mês")
ax.set_ylabel("Frequência")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"])

plt.tight_layout()
plt.savefig("sazonalidade_sinistros.png", dpi=150, bbox_inches="tight")
plt.show()
```

Feature engineering básico para dados de seguros — variáveis derivadas que o modelo vai precisar:

```python
df["log_premio"]        = np.log1p(df["premio"])           # 💡 log estabiliza a escala para modelos lineares
df["log_severidade"]    = np.log1p(df["severidade"])
df["premio_anualizado"] = df["premio"] / df["exposicao"]
df["trimestre"]         = pd.to_datetime(df["data_inicio"]).dt.quarter

# salva carteira processada com indicadores para uso nos próximos módulos
df.to_csv("dados/processed/carteira_eda.csv", index=False)
print(f"Carteira salva: {df.shape[0]} registros, {df.shape[1]} colunas")
print(df[["log_premio", "log_severidade", "premio_anualizado", "trimestre"]].describe().round(3))
```

> 🔗 **Próximos módulos:** `log_premio` e `log_severidade` vão aparecer como variáveis resposta no GLM Gamma do Módulo 02. `trimestre` será usado na análise de tendência do Módulo 06.

### 🏋️ Exercício 3

1. Plote um **heatmap de correlação** entre `premio`, `exposicao`, `frequencia`, `severidade` e `loss_ratio` usando `sns.heatmap`. Quais pares têm correlação acima de 0.3 em valor absoluto?
2. Crie uma variável `faixa_premio` com `pd.cut` (4 faixas) e calcule a frequência média por faixa. A frequência aumenta com o prêmio? O que isso indica sobre seleção de risco?
3. **Desafio:** construa um **triângulo de sinistros simplificado** — uma tabela pivô com `trimestre` nas linhas, `ramo` nas colunas e `frequencia` como valores. Use `df.pivot_table`. Identifique o trimestre e ramo com maior sinistralidade.

---

## Resumo do módulo

| Conceito | O que aprendemos |
|---|---|
| **Diagnóstico de qualidade** | Missing values, outliers e zeros inflados devem ser mapeados antes de qualquer cálculo |
| **Exposição** | Base de todo indicador atuarial — frequência e loss ratio sem exposição são métricas enganosas |
| **Earned premium** | Prêmio proporcional ao período vigente — `premio × exposicao` |
| **Frequência** | `sinistros / exposicao` — indicador central de risco de ocorrência |
| **Severidade** | `valor_sinistro / sinistros` — indica o custo médio quando o sinistro ocorre |
| **Loss ratio** | `valor_sinistros / earned_premium` — termômetro de rentabilidade por segmento |
| **Análise bivariada** | Frequência por variável de risco revela poder discriminante antes do modelo |
| **Feature engineering** | `log_premio`, `premio_anualizado`, `trimestre` — variáveis derivadas para os próximos módulos |

---

## Próxima aula

O Módulo 02 entra em **Modelos Lineares Generalizados (GLM)**. Com a EDA feita e a carteira sintética limpa, vamos modelar frequência com GLM Poisson e severidade com GLM Gamma — as duas peças do motor de precificação atuarial clássico.

---

## Referências e leitura complementar

- Frees, E.W. *Regression Modeling with Actuarial and Financial Applications* — Cambridge University Press
- de Jong, P. & Heller, G.Z. *Generalized Linear Models for Insurance Data* — Cambridge University Press
- Documentação do pandas: guia de groupby e agregações — pandas.pydata.org/docs/user_guide/groupby
- Documentação do seaborn: galeria de gráficos estatísticos — seaborn.pydata.org/tutorial

---

*Curso de Ciência de Dados e ML para Seguros — Módulo 01 v1.0*
