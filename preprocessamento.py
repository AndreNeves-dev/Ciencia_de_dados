"""
Sprint 2 - Pre-processamento
Base: IBGE / Pesquisa Anual de Servicos (PAS), Tabela 2653 - SIDRA
Custos das empresas de transporte, servicos auxiliares aos transportes e correio
Brasil, 2007-2023, valores em Mil Reais.

Entrada : CSV da Sprint 1 (136 linhas x 7 colunas)
Saidas  : dados/base_limpa.csv          -> 119 linhas, sem "Total geral", log aplicado
          dados/base_modelagem.csv      -> 105 linhas, com defasagens (t-1), one-hot e
                                           os alvos cresc (regressao) e tendencia
                                           (classificacao)
          dados/relatorio_preprocessamento.txt

Uso:
    python preprocessamento.py caminho/para/base_sprint1.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
COLUNAS = {
    "ano": "ano",
    "atividade": "atividade",
    "total_de_custos_mil_reais": "total",
    "mercadorias_consumo_reposicao_mil_reais": "mercadorias",
    "custo_mercadorias_revendidas_mil_reais": "revendidas",
    "combustiveis_lubrificantes_mil_reais": "combustiveis",
    "outros_custos_mil_reais": "outros",
}
COMPONENTES = ["mercadorias", "revendidas", "combustiveis", "outros"]
MONETARIAS = ["total"] + COMPONENTES
TOTAL_GERAL = "Total geral"
LIMITE_ALTA_FORTE = 15  # % de crescimento anual que separa alta moderada de alta forte
CLASSES = ["queda", "alta_moderada", "alta_forte"]
PASTA_SAIDA = Path("dados")

log = []


def registrar(msg: str) -> None:
    print(msg)
    log.append(msg)


# ---------------------------------------------------------------------------
# Etapa 1 - Leitura e padronizacao
# ---------------------------------------------------------------------------
def carregar(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    df = df.rename(columns=COLUNAS)
    faltando = set(COLUNAS.values()) - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")
    df = df[list(COLUNAS.values())]

    # O SIDRA usa "-" para zero absoluto; garante a conversao caso o CSV venha cru
    for c in MONETARIAS:
        df[c] = pd.to_numeric(df[c].replace({"-": 0, "..": np.nan, "...": np.nan}))
    df["ano"] = df["ano"].astype(int)
    df["atividade"] = df["atividade"].astype(str).str.strip()

    registrar(f"[1] Leitura: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


# ---------------------------------------------------------------------------
# Etapa 2 - Ausentes e duplicatas
# ---------------------------------------------------------------------------
def checar_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    nulos = int(df.isna().sum().sum())
    dup = int(df.duplicated(subset=["ano", "atividade"]).sum())
    negativos = int((df[MONETARIAS] < 0).sum().sum())
    registrar(f"[2] Valores ausentes: {nulos} | Duplicatas (ano, atividade): {dup} "
              f"| Valores negativos: {negativos}")
    if nulos or dup or negativos:
        raise ValueError("Base com problemas de qualidade - revisar antes de seguir.")
    return df


# ---------------------------------------------------------------------------
# Etapa 3 - Consistencia contabil
# ---------------------------------------------------------------------------
def checar_consistencia(df: pd.DataFrame) -> None:
    # (a) total = soma dos quatro componentes, linha a linha
    dif = (df["total"] - df[COMPONENTES].sum(axis=1)).abs()
    # tolerancia de 5 mil reais: a tabela do IBGE e arredondada por componente,
    # e na base real a maior diferenca entre total e soma e de 3 mil reais
    linhas_ok = int((dif <= 5).sum())
    registrar(f"[3a] Total = soma dos 4 componentes em {linhas_ok}/{len(df)} linhas "
              f"(maior diferenca: {dif.max():,.0f} mil R$)")

    # (b) Total geral = soma das 7 atividades, ano a ano
    ativ = df[df["atividade"] != TOTAL_GERAL].groupby("ano")[MONETARIAS].sum()
    tg = df[df["atividade"] == TOTAL_GERAL].set_index("ano")[MONETARIAS]
    anos_ok = int(((ativ - tg).abs() <= 5).all(axis=1).sum())
    registrar(f"[3b] Total geral = soma das atividades em {anos_ok}/{len(tg)} anos")


# ---------------------------------------------------------------------------
# Etapa 4 - Remocao do agregado "Total geral"
# ---------------------------------------------------------------------------
def remover_total_geral(df: pd.DataFrame) -> pd.DataFrame:
    antes = len(df)
    df = df[df["atividade"] != TOTAL_GERAL].reset_index(drop=True)
    registrar(f"[4] Removido '{TOTAL_GERAL}': {antes} -> {len(df)} linhas "
              f"({df['atividade'].nunique()} atividades x {df['ano'].nunique()} anos)")
    return df


# ---------------------------------------------------------------------------
# Etapa 5 - Valores discrepantes (IQR) - identificacao, sem remocao
# ---------------------------------------------------------------------------
def identificar_outliers(df: pd.DataFrame) -> None:
    registrar("[5] Outliers pelo criterio IQR (escala original):")
    for c in MONETARIAS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        lim = q3 + 1.5 * (q3 - q1)
        out = df[df[c] > lim]
        ativs = ", ".join(sorted(out["atividade"].unique())) or "-"
        registrar(f"     {c:<13} {len(out):>3} pontos acima de {lim:,.0f} | {ativs}")
    registrar("     Decisao: manter todos (sao valores reais, nao erros de coleta); "
              "a assimetria e tratada pela transformacao log.")


# ---------------------------------------------------------------------------
# Etapa 6 - Transformacao logaritmica
# ---------------------------------------------------------------------------
def aplicar_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in MONETARIAS:
        df[f"log_{c}"] = np.log1p(df[c])
    assim_antes = df[MONETARIAS].skew().round(2).to_dict()
    assim_depois = df[[f"log_{c}" for c in MONETARIAS]].skew().round(2).to_dict()
    registrar(f"[6] log1p aplicado. Assimetria antes: {assim_antes}")
    registrar(f"     Assimetria depois: {assim_depois}")
    return df


# ---------------------------------------------------------------------------
# Etapa 7 - Atributos defasados (evita vazamento do alvo)
# ---------------------------------------------------------------------------
def criar_defasagens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Como total = soma exata dos componentes do mesmo ano, usar os componentes
    do ano t para prever o total do ano t e uma identidade, nao uma previsao.
    Aqui o alvo e o total do ano t e os preditores sao informacoes de t-1.
    """
    df = df.sort_values(["atividade", "ano"]).copy()
    g = df.groupby("atividade")
    for c in MONETARIAS:
        df[f"log_{c}_lag1"] = g[f"log_{c}"].shift(1)
    df["part_combustiveis_lag1"] = g["combustiveis"].shift(1) / g["total"].shift(1)

    # Alvo de regressao: crescimento anual do custo total, em log
    # cresc_t = log(1+total_t) - log(1+total_t-1)  ~  variacao percentual
    df["cresc"] = df["log_total"] - df["log_total_lag1"]
    # Preditor: crescimento do ano anterior (t-2 -> t-1)
    df["cresc_lag1"] = df.groupby("atividade")["cresc"].shift(1)

    # Alvo de classificacao: tendencia do custo no ano t
    pct = np.expm1(df["cresc"]) * 100
    df["tendencia"] = pd.cut(pct, bins=[-np.inf, 0, LIMITE_ALTA_FORTE, np.inf],
                             labels=CLASSES).astype(object)
    df.loc[df["cresc"].isna(), "tendencia"] = np.nan

    antes = len(df)
    df = df.dropna(subset=["cresc_lag1"]).reset_index(drop=True)
    registrar(f"[7] Defasagens e alvos criados; dois primeiros anos de cada atividade "
              f"removidos: {antes} -> {len(df)} linhas")
    registrar(f"     Classes de tendencia (limites 0% e {LIMITE_ALTA_FORTE}%): "
              f"{df['tendencia'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Etapa 8 - Codificacao da variavel categorica
# ---------------------------------------------------------------------------
def codificar(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["atividade"], prefix="ativ", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    registrar(f"[8] One-hot em 'atividade': {dummies.shape[1]} colunas binarias")
    return df


# ---------------------------------------------------------------------------
# Etapa 9 - Multicolinearidade (VIF)
# ---------------------------------------------------------------------------
def vif(X: pd.DataFrame) -> pd.Series:
    X = X.astype(float)
    res = {}
    for col in X.columns:
        y = X[col].values
        outros = X.drop(columns=col).values
        A = np.column_stack([np.ones(len(X)), outros])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(res).round(1)


def checar_multicolinearidade(df: pd.DataFrame) -> None:
    cols = [f"log_{c}_lag1" for c in COMPONENTES]
    v = vif(df[cols])
    registrar("[9] VIF dos componentes defasados (log):")
    for k, val in v.items():
        registrar(f"     {k:<26} {val}")
    registrar("     VIF > 10 indica multicolinearidade forte -> usar regularizacao "
              "(Ridge/Lasso) nos modelos lineares; arvores nao sao afetadas.")


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------
def main(caminho: str) -> None:
    PASTA_SAIDA.mkdir(exist_ok=True)

    df = carregar(caminho)
    df = checar_qualidade(df)
    checar_consistencia(df)
    df = remover_total_geral(df)
    identificar_outliers(df)
    df = aplicar_log(df)

    df.to_csv(PASTA_SAIDA / "base_limpa.csv", index=False)

    df_mod = criar_defasagens(df)
    df_mod = codificar(df_mod)
    checar_multicolinearidade(df_mod)

    preditores = (["ano"]
                  + [f"log_{c}_lag1" for c in MONETARIAS]
                  + ["part_combustiveis_lag1", "cresc_lag1"]
                  + [c for c in df_mod.columns if c.startswith("ativ_")])
    saida = df_mod[["ano", "atividade"] + preditores[1:]
                   + ["cresc", "tendencia", "log_total", "total"]]
    saida.to_csv(PASTA_SAIDA / "base_modelagem.csv", index=False)

    registrar(f"[10] Base de modelagem: {saida.shape[0]} linhas | "
              f"{len(preditores)} preditores | alvos = cresc (regressao) "
              f"e tendencia (classificacao)")
    registrar("     Padronizacao (z-score) NAO aplicada aqui: sera ajustada so no "
              "conjunto de treino, dentro do Pipeline da Sprint 3, para evitar vazamento.")

    (PASTA_SAIDA / "relatorio_preprocessamento.txt").write_text("\n".join(log),
                                                                encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python preprocessamento.py caminho/para/base_sprint1.csv")
    main(sys.argv[1])
