"""
Sprint 3 - Algoritmos Individuais
Base: IBGE / PAS, Tabela 2653 (saida do preprocessamento.py)

Duas tarefas sobre a mesma base:
  - Regressao: prever o crescimento anual do custo total de cada atividade (cresc)
  - Classificacao: prever a tendencia do custo (queda, alta_moderada, alta_forte)

Protocolo:
  - Treino: 2009-2020 (84 registros) | Teste: 2021-2023 (21 registros)
  - Validacao cruzada temporal no treino (janela expansiva, 5 dobras: 2016 a 2020)
  - Busca em grade (GridSearchCV) com padronizacao z-score dentro do Pipeline

Uso:
    python algoritmos_individuais.py dados/base_modelagem.csv
Saidas em resultados/
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             make_scorer, mean_absolute_error, mean_squared_error,
                             median_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")  # avisos de convergencia/zero_division em dobras pequenas

SEMENTE = 42
ANO_INICIO_TESTE = 2021
ANOS_VALIDACAO = [2016, 2017, 2018, 2019, 2020]
CLASSES = ["queda", "alta_moderada", "alta_forte"]
SAIDA = Path("resultados")

# Paleta (series categoricas e tinta), fixa para todas as figuras
COR = {"azul": "#2a78d6", "laranja": "#eb6834", "aqua": "#1baf7a",
       "tinta": "#0b0b0b", "tinta2": "#52514e", "muted": "#898781",
       "grade": "#e1e0d9", "eixo": "#c3c2b7", "fundo": "#fcfcfb"}


# ---------------------------------------------------------------------------
# Dados e divisao
# ---------------------------------------------------------------------------
def carregar(caminho):
    df = pd.read_csv(caminho).sort_values(["ano", "atividade"]).reset_index(drop=True)
    preditores = (["ano", "log_total_lag1", "log_mercadorias_lag1",
                   "log_revendidas_lag1", "log_combustiveis_lag1", "log_outros_lag1",
                   "part_combustiveis_lag1", "cresc_lag1"]
                  + [c for c in df.columns if c.startswith("ativ_")])
    return df, preditores


def dobras_temporais(anos_treino):
    """Janela expansiva: cada dobra treina com todos os anos anteriores ao ano validado."""
    anos = np.asarray(anos_treino)
    return [(np.where(anos < a)[0], np.where(anos == a)[0]) for a in ANOS_VALIDACAO]


# ---------------------------------------------------------------------------
# Modelos e grades de hiperparametros
# ---------------------------------------------------------------------------
def pipe(modelo, escalar=True):
    passos = [("escala", StandardScaler())] if escalar else []
    return Pipeline(passos + [("modelo", modelo)])


MODELOS_REG = {
    "Média do treino (referência)": (pipe(DummyRegressor(strategy="mean"), False), {}),
    "Regressão Ridge": (pipe(Ridge()), {"modelo__alpha": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (pipe(KNeighborsRegressor()),
            {"modelo__n_neighbors": [3, 5, 7, 9, 11],
             "modelo__weights": ["uniform", "distance"]}),
    "Árvore de Decisão": (pipe(DecisionTreeRegressor(random_state=SEMENTE), False),
                          {"modelo__max_depth": [2, 3, 4, 5, None],
                           "modelo__min_samples_leaf": [1, 3, 5, 10]}),
    "SVR": (pipe(SVR(kernel="rbf")),
            {"modelo__C": [0.1, 1, 10], "modelo__epsilon": [0.01, 0.05, 0.1],
             "modelo__gamma": ["scale", 0.01, 0.1]}),
    "MLP": (pipe(MLPRegressor(max_iter=3000, random_state=SEMENTE)),
            {"modelo__hidden_layer_sizes": [(8,), (16,), (16, 8)],
             "modelo__alpha": [0.001, 0.01, 0.1, 1.0]}),
}

MODELOS_CLF = {
    "Classe majoritária (referência)": (pipe(DummyClassifier(strategy="most_frequent"), False), {}),
    "Regressão Logística": (pipe(LogisticRegression(max_iter=5000)),
                            {"modelo__C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (pipe(KNeighborsClassifier()),
            {"modelo__n_neighbors": [3, 5, 7, 9, 11],
             "modelo__weights": ["uniform", "distance"]}),
    "Árvore de Decisão": (pipe(DecisionTreeClassifier(random_state=SEMENTE), False),
                          {"modelo__max_depth": [2, 3, 4, 5, None],
                           "modelo__min_samples_leaf": [1, 3, 5, 10]}),
    "SVM": (pipe(SVC(kernel="rbf", probability=True, random_state=SEMENTE)),
            {"modelo__C": [0.1, 1, 10, 100], "modelo__gamma": ["scale", 0.01, 0.1]}),
    "Naive Bayes": (pipe(GaussianNB()),
                    {"modelo__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
    "MLP": (pipe(MLPClassifier(max_iter=3000, random_state=SEMENTE)),
            {"modelo__hidden_layer_sizes": [(8,), (16,), (16, 8)],
             "modelo__alpha": [0.001, 0.01, 0.1, 1.0]}),
}


# ---------------------------------------------------------------------------
# Metricas (regressao avaliada em pontos percentuais de crescimento)
# ---------------------------------------------------------------------------
def em_pct(cresc_log):
    return np.expm1(cresc_log) * 100


def metricas_reg(y_log, p_log):
    y, p = em_pct(y_log), em_pct(p_log)
    return {"MAE (p.p.)": mean_absolute_error(y, p),
            "RMSE (p.p.)": np.sqrt(mean_squared_error(y, p)),
            "MdAE (p.p.)": median_absolute_error(y, p),
            "R²": r2_score(y, p)}


def metricas_clf(y, pred, proba, classes_modelo):
    m = {"Acurácia": accuracy_score(y, pred),
         "Precisão (macro)": precision_score(y, pred, average="macro", zero_division=0),
         "Recall (macro)": recall_score(y, pred, average="macro", zero_division=0),
         "F1 (macro)": f1_score(y, pred, average="macro", zero_division=0)}
    try:
        ordem = sorted(CLASSES)  # o scikit-learn exige os rotulos em ordem alfabetica
        p = pd.DataFrame(proba, columns=classes_modelo).reindex(columns=ordem, fill_value=0)
        m["AUC (macro, OvR)"] = roc_auc_score(y, p.values, multi_class="ovr",
                                              average="macro", labels=ordem)
    except ValueError:
        m["AUC (macro, OvR)"] = np.nan
    return m


def mae_pct_score(y_log, p_log):
    return -mean_absolute_error(em_pct(y_log), em_pct(p_log))


# ---------------------------------------------------------------------------
# Execucao de uma tarefa
# ---------------------------------------------------------------------------
def rodar(modelos, X_tr, y_tr, X_te, y_te, cv, scoring, tarefa):
    linhas, hiper, preds = [], [], {}
    for nome, (modelo, grade) in modelos.items():
        # modelos de referencia tem grade vazia: uma unica configuracao avaliada
        busca = GridSearchCV(modelo, grade, cv=cv, scoring=scoring, refit=True)
        t0 = time.perf_counter()
        busca.fit(X_tr, y_tr)
        t_treino = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = busca.predict(X_te)
        t_pred = time.perf_counter() - t0

        if tarefa == "reg":
            m = metricas_reg(y_te, pred)
            val = -busca.best_score_
            m = {"MAE validação (p.p.)": val, **m}
        else:
            proba = busca.predict_proba(X_te)
            m = metricas_clf(y_te, pred, proba, busca.best_estimator_.classes_)
            m = {"F1 validação (macro)": busca.best_score_, **m}
        m.update({"Tempo busca+treino (s)": t_treino, "Tempo predição (ms)": t_pred * 1000,
                  "Combinações testadas": len(busca.cv_results_["params"])})
        linhas.append({"Modelo": nome, **m})
        hiper.append({"Tarefa": "Regressão" if tarefa == "reg" else "Classificação",
                      "Modelo": nome,
                      "Melhores hiperparâmetros": ", ".join(
                          f"{k.replace('modelo__', '')}={v}" for k, v in busca.best_params_.items())
                      or "(sem hiperparâmetros)"})
        preds[nome] = pred
        print(f"  {nome:<34} ok ({t_treino:.1f}s)")
    return pd.DataFrame(linhas), pd.DataFrame(hiper), preds


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def estilo(ax):
    ax.set_facecolor(COR["fundo"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(COR["eixo"])
    ax.tick_params(colors=COR["tinta2"], labelsize=9)
    ax.grid(axis="x", color=COR["grade"], linewidth=0.8)
    ax.set_axisbelow(True)


def fig_mae(tab, caminho):
    t = tab.sort_values("MAE (p.p.)", ascending=False)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(COR["fundo"])
    estilo(ax)
    h = 0.36
    ax.barh(y + h / 2, t["MAE validação (p.p.)"], height=h, color=COR["azul"],
            label="Validação (2016–2020)")
    ax.barh(y - h / 2, t["MAE (p.p.)"], height=h, color=COR["laranja"],
            label="Teste (2021–2023)")
    for yi, v in zip(y - h / 2, t["MAE (p.p.)"]):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=8, color=COR["tinta2"])
    ax.set_yticks(y, t["Modelo"], color=COR["tinta"])
    ax.set_xlabel("Erro absoluto médio do crescimento anual (pontos percentuais)",
                  color=COR["tinta2"], fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=COR["fundo"])
    plt.close(fig)


def fig_real_previsto(df_te, pred_log, nome, caminho):
    real, prev = em_pct(df_te["cresc"].values), em_pct(pred_log)
    fig, ax = plt.subplots(figsize=(6.4, 5), dpi=200)
    fig.patch.set_facecolor(COR["fundo"])
    estilo(ax)
    ax.grid(axis="y", color=COR["grade"], linewidth=0.8)
    lim = [min(real.min(), prev.min()) - 10, max(real.max(), prev.max()) + 10]
    ax.plot(lim, lim, color=COR["muted"], linewidth=1, linestyle="--", zorder=1)
    ax.text(lim[1], lim[1], " previsão perfeita", fontsize=8, color=COR["muted"],
            ha="right", va="bottom")
    cores = {2021: COR["azul"], 2022: COR["laranja"], 2023: COR["aqua"]}
    for ano, cor in cores.items():
        m = df_te["ano"].values == ano
        ax.scatter(real[m], prev[m], s=46, color=cor, edgecolor=COR["fundo"],
                   linewidth=1.5, label=str(ano), zorder=3)
    # rotula so os dois maiores erros, para nao sobrepor textos
    for i in np.argsort(-np.abs(real - prev))[:2]:
        rotulo = f'{df_te["atividade"].values[i].replace("Transporte ", "")} {df_te["ano"].values[i]}'
        ax.annotate(rotulo, (real[i], prev[i]), xytext=(-6, 10), textcoords="offset points",
                    fontsize=7.5, color=COR["tinta2"], ha="right")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Crescimento real (%)", color=COR["tinta2"], fontsize=9)
    ax.set_ylabel(f"Crescimento previsto – {nome} (%)", color=COR["tinta2"], fontsize=9)
    ax.legend(title="Ano", frameon=False, fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=COR["fundo"])
    plt.close(fig)


def fig_confusao(y, pred, nome, caminho):
    rotulos = {"queda": "Queda", "alta_moderada": "Alta moderada", "alta_forte": "Alta forte"}
    cm = confusion_matrix(y, pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    fig.patch.set_facecolor(COR["fundo"])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#86b6ef", "#1c5cab"])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=max(cm.max(), 1))
    for i in range(3):
        for j in range(3):
            cor = "#ffffff" if cm[i, j] > cm.max() * 0.6 else COR["tinta"]
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, color=cor)
    ax.set_xticks(range(3), [rotulos[c] for c in CLASSES], fontsize=9, color=COR["tinta"])
    ax.set_yticks(range(3), [rotulos[c] for c in CLASSES], fontsize=9, color=COR["tinta"])
    ax.set_xlabel(f"Classe prevista – {nome}", color=COR["tinta2"], fontsize=9)
    ax.set_ylabel("Classe real", color=COR["tinta2"], fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=COR["fundo"])
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(caminho):
    SAIDA.mkdir(exist_ok=True)
    df, preditores = carregar(caminho)
    tr = df[df["ano"] < ANO_INICIO_TESTE].reset_index(drop=True)
    te = df[df["ano"] >= ANO_INICIO_TESTE].reset_index(drop=True)
    cv = dobras_temporais(tr["ano"])
    print(f"Treino: {len(tr)} registros ({tr.ano.min()}-{tr.ano.max()}) | "
          f"Teste: {len(te)} registros ({te.ano.min()}-{te.ano.max()})")

    X_tr, X_te = tr[preditores].values, te[preditores].values

    print("Regressão (alvo: crescimento anual do custo total)")
    reg, hip_r, pred_r = rodar(MODELOS_REG, X_tr, tr["cresc"].values, X_te,
                               te["cresc"].values, cv, make_scorer(mae_pct_score), "reg")
    # Referencia de persistencia: crescimento de t igual ao de t-1 (sem treino)
    pers = te["cresc_lag1"].values
    m = metricas_reg(te["cresc"].values, pers)
    val_pers = np.mean([mean_absolute_error(em_pct(tr.loc[v, "cresc"]),
                                            em_pct(tr.loc[v, "cresc_lag1"])) for _, v in cv])
    reg = pd.concat([pd.DataFrame([{"Modelo": "Persistência (referência)",
                                    "MAE validação (p.p.)": val_pers, **m,
                                    "Tempo busca+treino (s)": 0.0, "Tempo predição (ms)": 0.0,
                                    "Combinações testadas": 0}]), reg], ignore_index=True)
    pred_r["Persistência (referência)"] = pers

    print("Classificação (alvo: tendência do custo)")
    f1 = make_scorer(f1_score, average="macro", zero_division=0)
    clf, hip_c, pred_c = rodar(MODELOS_CLF, X_tr, tr["tendencia"].values, X_te,
                               te["tendencia"].values, cv, f1, "clf")

    # Tabelas
    reg.round(3).to_csv(SAIDA / "regressao_metricas.csv", index=False)
    clf.round(3).to_csv(SAIDA / "classificacao_metricas.csv", index=False)
    pd.concat([hip_r, hip_c]).to_csv(SAIDA / "hiperparametros.csv", index=False)
    prev = te[["ano", "atividade", "cresc", "tendencia"]].copy()
    prev["cresc_real_%"] = em_pct(prev["cresc"]).round(1)
    for nome, p in pred_r.items():
        prev[f"prev_% | {nome}"] = em_pct(p).round(1)
    for nome, p in pred_c.items():
        prev[f"classe | {nome}"] = p
    prev.drop(columns="cresc").to_csv(SAIDA / "previsoes_teste.csv", index=False)

    # Figuras
    modelos_r = reg[~reg["Modelo"].str.contains("referência")]
    melhor_r = modelos_r.sort_values("MAE validação (p.p.)").iloc[0]["Modelo"]
    modelos_c = clf[~clf["Modelo"].str.contains("referência")]
    melhor_c = modelos_c.sort_values("F1 validação (macro)", ascending=False).iloc[0]["Modelo"]
    fig_mae(reg, SAIDA / "fig_mae_regressao.png")
    fig_real_previsto(te, pred_r[melhor_r], melhor_r, SAIDA / "fig_real_vs_previsto.png")
    fig_confusao(te["tendencia"].values, pred_c[melhor_c], melhor_c,
                 SAIDA / "fig_matriz_confusao.png")

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n== Regressão ==\n", reg.round(3).to_string(index=False))
    print("\n== Classificação ==\n", clf.round(3).to_string(index=False))
    print("\n== Hiperparâmetros ==\n", pd.concat([hip_r, hip_c]).to_string(index=False))
    print(f"\nMelhor na validação -> regressão: {melhor_r} | classificação: {melhor_c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python algoritmos_individuais.py dados/base_modelagem.csv")
    main(sys.argv[1])
