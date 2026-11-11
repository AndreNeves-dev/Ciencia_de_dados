# Mineração de Dados — Custos das Empresas de Transporte (IBGE/PAS)

Projeto da disciplina de Mineração de Dados.

**Equipe:** João Murilo, Levy Porfirio, Emerson Ribeiro e André Wilke

**Base:** IBGE, Pesquisa Anual de Serviços (PAS), Tabela 2653 — custos das empresas de transporte, serviços auxiliares aos transportes e correio, Brasil, 2007–2023, em mil reais.

## Sprint 2 — Pré-processamento

O script `preprocessamento.py` executa as 10 etapas descritas na seção 3.2 do relatório:

1. Leitura e padronização
2. Verificação de valores ausentes, duplicatas e valores negativos
3. Consistência contábil (total = soma dos componentes; Total geral = soma das atividades)
4. Remoção do Total geral (136 → 119 registros)
5. Identificação de valores discrepantes pelo critério IQR (sem remoção)
6. Transformação log(1 + x)
7. Atributos defasados em t−1 e criação dos alvos (119 → 105 registros)
8. One-hot encoding da atividade
9. Multicolinearidade (VIF)
10. Padronização deixada para o treino (Sprint 3)

### Como executar

```bash
pip install -r requirements.txt
python preprocessamento.py base_sprint1.csv
```

### Saídas (pasta `dados/`)

| Arquivo | Conteúdo |
|---|---|
| `base_limpa.csv` | 119 registros, sem o Total geral e com log(1 + x) |
| `base_modelagem.csv` | 105 registros, 15 preditores e os alvos `cresc` (regressão) e `tendencia` (classificação) |
| `relatorio_preprocessamento.txt` | Resultado de cada verificação |

## Sprint 3 — Algoritmos Individuais

O script `algoritmos_individuais.py` treina e avalia algoritmos individuais em duas tarefas:

- **Regressão:** prever o crescimento anual do custo total de cada atividade (`cresc`).
- **Classificação:** prever a tendência do custo no ano seguinte: queda (< 0%), alta moderada (0% a 15%) ou alta forte (> 15%).

**Protocolo:** treino 2009–2020 (84 registros), teste 2021–2023 (21 registros), validação cruzada temporal de janela expansiva com 5 dobras (2016 a 2020) e busca em grade com padronização z-score dentro do `Pipeline`.

**Algoritmos:** Regressão Ridge, KNN, Árvore de Decisão, SVR/SVM, MLP, Regressão Logística e Naive Bayes, comparados com modelos de referência (média, persistência e classe majoritária).

### Como executar

```bash
pip install -r requirements.txt
python preprocessamento.py base_sprint1.csv
python algoritmos_individuais.py dados/base_modelagem.csv
```

### Saídas (pasta `resultados/`)

| Arquivo | Conteúdo |
|---|---|
| `regressao_metricas.csv` | MAE, RMSE, MdAE, R² e tempos de cada modelo |
| `classificacao_metricas.csv` | Acurácia, precisão, recall, F1, AUC e tempos |
| `hiperparametros.csv` | Melhores hiperparâmetros encontrados |
| `previsoes_teste.csv` | Valores reais e previstos de 2021 a 2023 |
| `fig_*.png` | Figuras usadas no relatório |
