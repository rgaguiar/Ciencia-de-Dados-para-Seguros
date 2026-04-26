# STYLE GUIDE — Curso de Ciência de Dados e ML para Seguros

> Este arquivo define os padrões de escrita, código e estrutura de todas as aulas do curso.
> Sempre passe este arquivo como contexto ao gerar ou revisar módulos.

---

## 1. Identidade do curso

**Nome:** Ciência de Dados e Machine Learning para Seguros  
**Público:** Analistas de dados que querem se tornar engenheiros de IA/ML com foco no setor de seguros  
**Perfil da turma:** Misto — iniciantes em programação, analistas com Python básico e profissionais com experiência em BI  
**Tom:** Técnico mas acessível. Direto ao ponto. Sem condescendência, sem excesso de didatismo  
**Idioma:** Português brasileiro. Termos técnicos em inglês são mantidos quando não há tradução consagrada (ex: loss ratio, pure premium, overfitting)  
**Framework principal:** Nenhum — o foco é em Python científico puro (pandas, numpy, scipy, scikit-learn, statsmodels)

---

## 2. Estrutura obrigatória de cada aula

Todo arquivo `.md` de aula deve seguir exatamente esta ordem:

```
# Módulo XX — Título da Aula

> Metadados (curso, nível, pré-requisito, ferramentas)

---

## Por que começamos aqui?         ← narrativa introdutória (sem código)

---

## Vocabulário / Conceitos-chave   ← quando aplicável, antes do código

---

## Seção 1 — [Tema]               ← conteúdo técnico com código

### 🏋️ Exercício 1                 ← exercício após cada seção principal

## Seção 2 — [Tema]

### 🏋️ Exercício 2

## Seção N — [Tema]

### 🏋️ Exercício N (máx. 3 por aula)

---

## Resumo do módulo               ← tabela com conceito + o que aprendemos

---

## Próxima aula                   ← 2-3 linhas sobre o que vem a seguir

---

## Referências e leitura complementar

---

*Curso de Ciência de Dados e ML para Seguros — Módulo XX vX.X*
```

---

## 3. Narrativa introdutória

- Sempre abre com **por que este conteúdo existe** — qual problema ele resolve no contexto de seguros
- Nunca começa com "Nesta aula vamos aprender..." — essa frase é proibida
- Deve ter entre 3 e 6 parágrafos
- Pode usar analogias do setor (atuária, subscrição, sinistro) para ancorar conceitos novos
- Não contém código

**Exemplo de abertura ruim:**
> "Nesta aula vamos aprender sobre GLMs. GLM significa Modelo Linear Generalizado..."

**Exemplo de abertura boa:**
> "Quando uma seguradora precifica um risco, ela está essencialmente respondendo uma pergunta: quanto este segurado vai me custar? A resposta não vem de intuição — vem de um modelo. E o modelo que domina a precificação atuarial há décadas tem um nome: GLM..."

---

## 4. Blocos de código

### Padrão geral

- Linguagem: sempre `python` no fence do markdown
- Comentários: **apenas nos pontos críticos** — onde há uma decisão não óbvia ou um conceito que precisa de ancoragem
- Nunca comentar o óbvio: `# soma os valores` em cima de `sum(valores)` é ruído
- Máximo de 50 linhas por bloco. Blocos maiores devem ser quebrados em partes com texto entre eles
- Sempre incluir `print()` ou visualização ao final de blocos demonstrativos — o aluno precisa ver output

### Comentários críticos — formato

Use o padrão `# 💡` para comentários que explicam uma decisão importante:

```python
# lognormal é mais realista para prêmios — distribui assimetria positiva naturalmente
premio = np.random.lognormal(mean=7.0, sigma=0.5, size=n)
```

Use o padrão `# ⚠️` para alertas e armadilhas comuns:

```python
exposicao = df["exposicao"].replace(0, np.nan)  # ⚠️ divisão por zero gera infinito — trate antes
frequencia = df["sinistros"] / exposicao
```

### Variáveis e nomenclatura

Sempre use nomes em português para variáveis de negócio:

```python
# correto
premio, sinistro, exposicao, frequencia, severidade, loss_ratio

# evitar
x, y, val, tmp, data2, df_final2
```

Exceções permitidas: `df` para DataFrames genéricos, `X` e `y` em contexto de ML (convenção scikit-learn)

### Imports

Sempre declare imports no início do bloco quando são novos para o aluno. Quando já foram apresentados, não repita:

```python
# primeira vez que aparece na aula — declare
import pandas as pd
import numpy as np

# bloco posterior na mesma aula — não repita o import
carteira = pd.DataFrame({...})
```

### Seeds de aleatoriedade

Sempre use `np.random.seed(42)` ou `random_state=42` em código com aleatoriedade. Garante reprodutibilidade para o aluno.

---

## 5. Exercícios

- **Quantidade:** 2 a 3 por aula, distribuídos após cada seção principal — nunca agrupados no final
- **Dificuldade progressiva:** o primeiro exercício é de fixação (aplica o que acabou de ver), o último é de extensão (exige combinar conceitos)
- **Formato:** enunciado em lista numerada, sem gabarito no mesmo arquivo
- **Gabarito:** criado em arquivo separado `modulo_XX_gabarito.md` — nunca embutido na aula
- **Contexto:** sempre ambientado em seguros — nunca exercícios genéricos como "calcule a média de uma lista de números"
- **Ícone:** sempre use `### 🏋️ Exercício N` como header

**Exemplo de exercício ruim:**
> Calcule a média e o desvio padrão de uma lista de valores.

**Exemplo de exercício bom:**
> Usando a carteira `df_auto`, calcule a frequência de sinistros por faixa de prêmio (use `pd.cut` com 4 faixas) e identifique qual faixa tem maior sinistralidade esperada.

---

## 6. Tabelas de vocabulário

- Usadas quando o módulo introduz 5 ou mais termos técnicos novos
- Sempre com três colunas: `Termo | Definição | Exemplo`
- Posicionadas **antes do primeiro bloco de código** da aula
- Termos em negrito na primeira coluna

---

## 7. Callouts e destaques

Use blockquotes com emoji para destacar informações importantes:

```markdown
> 💡 **Ponto crítico:** explicação de uma decisão técnica importante

> ⚠️ **Atenção:** armadilha comum ou comportamento inesperado

> 📌 **Conexão com seguros:** como este conceito aparece na prática do setor

> 🔗 **Próximos módulos:** quando este conceito será aprofundado adiante
```

Limite: no máximo 2 callouts por seção. Excesso dilui o impacto.

---

## 8. Visualizações

- Sempre salvar com `plt.savefig("nome_descritivo.png", dpi=150, bbox_inches="tight")`
- Nome do arquivo deve descrever o conteúdo: `distribuicao_premios.png`, não `grafico1.png`
- Sempre incluir `plt.tight_layout()` antes do `savefig`
- Títulos, labels de eixo e legendas são obrigatórios — gráfico sem label é gráfico incompleto
- Paleta padrão: `sns.set_theme(style="whitegrid", palette="muted")` no início da seção de visualização

---

## 9. Dados e datasets

- **Dataset padrão do curso:** `carteira_sintetica.csv` — gerado no Módulo 00 e reutilizado ao longo do curso
- Dados externos reais (SUSEP, ANTT, DENATRAN) devem ser carregados de `dados/raw/`
- Dados processados salvos em `dados/processed/`
- Nunca hardcode de paths absolutos — sempre paths relativos à raiz do projeto
- Sempre incluir `encoding="latin-1"` ao ler CSVs da SUSEP (padrão do portal)

```python
# correto
df = pd.read_csv("dados/raw/susep_2023.csv", sep=";", encoding="latin-1")

# errado
df = pd.read_csv("C:/Users/rafael/Downloads/susep_2023.csv")
```

---

## 10. Resumo do módulo

Sempre uma tabela com duas colunas: `Conceito | O que aprendemos`  
Deve cobrir todos os tópicos principais da aula em no máximo 8 linhas.

---

## 11. Convenção de arquivos e commits

### Nomenclatura de arquivos

```
modulo_00_nivelamento.md
modulo_00_gabarito.md
modulo_01_eda.md
modulo_01_gabarito.md
modulo_02_glm.md
...
```

### Convenção de commits (Conventional Commits)

```
feat(modulo-XX): adiciona aula sobre [tema]
fix(modulo-XX): corrige [descrição do erro]
refactor(modulo-XX): melhora [seção] sem alterar conteúdo
docs(modulo-XX): atualiza referências e links
data: adiciona dataset [nome] em dados/raw
```

### Estrutura do repositório

```
curso-ds-seguros/
│
├── STYLE_GUIDE.md          ← este arquivo
├── README.md               ← apresentação do curso
├── requirements.txt        ← dependências do curso
│
├── dados/
│   ├── raw/                ← dados originais, nunca modificados
│   └── processed/          ← dados tratados
│
├── modulos/
│   ├── modulo_00_nivelamento.md
│   ├── modulo_00_gabarito.md
│   └── ...
│
└── notebooks/              ← versões .ipynb das aulas (opcional)
```

---

## 12. O que nunca fazer

- ❌ Começar aula com "Nesta aula vamos aprender..."
- ❌ Comentar o óbvio no código
- ❌ Exercícios sem contexto de seguros
- ❌ Blocos de código com mais de 50 linhas sem texto intermediário
- ❌ Gráficos sem título, label de eixo ou legenda
- ❌ Hardcode de paths absolutos
- ❌ Mais de 3 exercícios por aula
- ❌ Gabarito embutido na aula principal
- ❌ Importar a mesma biblioteca duas vezes na mesma aula
- ❌ Usar nomes genéricos de variáveis (`x`, `val`, `tmp`) para dados de negócio

---

*STYLE GUIDE v1.0 — Curso de Ciência de Dados e ML para Seguros*
