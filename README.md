
perfeito — aqui vai um script único que:

1. Treina um classificador com dados sintéticos (iguais aos que mostrei antes)


2. (Opcional) junta seus exemplos rotulados de um Excel/CSV para reforçar o treino


3. Pontua o seu Excel final (com MsgID e Body) e gera Top-K para revisão



Ele usa embeddings multilíngues e5 + Regressão Logística calibrada.

> deps:

pip install pandas numpy scikit-learn sentence-transformers openpyxl matplotlib



# -*- coding: utf-8 -*-
"""
Treina com (a) dados sintéticos + (b) seus exemplos rotulados (opcional),
e depois pontua seu Excel final (MsgID, Body).

Saídas:
- CSV com scores ordenados e Top-K
- (se houver rótulos no treino extra) métrica de validação (PR curve/AP impresso)

Uso típico:
python train_and_score.py \
  --score_xlsx MEU_EXCEL_FINAL.xlsx --score_sheet 0 \
  --output_dir saida_scores --topk 200

Com treino extra rotulado:
python train_and_score.py \
  --train_extra_xlsx MEUS_ROTULADOS.xlsx --train_sheet 0 \
  --score_xlsx MEU_EXCEL_FINAL.xlsx --score_sheet 0 \
  --output_dir saida_scores --topk 200
"""

import os, argparse, random, numpy as np, pandas as pd
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# -----------------------------
# Args
# -----------------------------
def get_args():
    ap = argparse.ArgumentParser(description="Treino (sintético + opcional extra) e scoring do Excel final.")
    # treino extra (opcional)
    ap.add_argument("--train_extra_xlsx", default=None, help="Excel/CSV com Body e Label (0/1) para acrescentar ao treino")
    ap.add_argument("--train_sheet", default=0, help="Sheet do treino extra (se .xlsx)")
    # arquivo a pontuar (obrigatório)
    ap.add_argument("--score_xlsx", required=True, help="Excel/CSV com MsgID e Body a serem pontuados")
    ap.add_argument("--score_sheet", default=0, help="Sheet do score (se .xlsx)")
    # config geral
    ap.add_argument("--output_dir", default="saida_scores", help="Pasta de saída")
    ap.add_argument("--model_name", default="intfloat/multilingual-e5-small",
                    help='Modelo de embeddings (ex.: "intfloat/multilingual-e5-small" ou "intfloat/multilingual-e5-base")')
    ap.add_argument("--topk", type=int, default=200, help="Top-K para exportar")
    ap.add_argument("--test_size", type=float, default=0.2, help="Validação interna quando houver treino extra")
    ap.add_argument("--random_state", type=int, default=42)
    return ap.parse_args()

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def clean_text(s):
    if pd.isna(s): return ""
    return str(s).strip()

def read_any_table(path, sheet=0):
    path = str(path)
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Arquivo deve ser .xlsx ou .csv")

def find_col(df, candidates: List[str]):
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        for c in df.columns:
            if c.lower() == name.lower(): return c
    return None

def embed_texts(model, texts: List[str], batch_size=64, prefix="passage: "):
    texts = [prefix + clean_text(t) for t in texts]
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

def plot_pr(y_true, scores, out_png):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AP={ap:.3f})")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    return ap

def precision_at_k(y_true, scores, k):
    order = np.argsort(scores)[::-1]; topk = order[:k]
    if k == 0: return 0.0
    return float(np.mean(np.array(y_true)[topk]))

# -----------------------------
# 1) Dados sintéticos (iguais ao exemplo anterior)
# -----------------------------
def synthetic_df():
    positivos = [
        "vamos discutir os números ainda não divulgados antes do anúncio oficial",
        "isso está em período de blackout, evitem compartilhar fora do time",
        "segue a planilha confidencial com a projeção do deal",
        "não mencione isso publicamente até o press release",
        "informação privilegiada: receita do trimestre vai surpreender",
        "NDA assinado, mas não repasse esses dados por e-mail",
        "evitem falar sobre a aquisição até o anúncio de M&A",
        "parece que o board aprovou antes de ser público, cuidado",
        "vazou o link do drive privado com dados de clientes",
        "não compartilhe credenciais, o token da API está no anexo",
        "podemos adiantar guidance extraoficial para o investidor X?",
        "janela de silêncio até sexta, sem comentar resultados",
        "não suba isso no slack, é sensível e não público ainda",
        "a due diligence descobriu números que não podem sair daqui",
        "isso não pode circular, é informação interna estratégica",
        "manda o excel com os dados antes da divulgação ao mercado",
        "isso fere a política de comunicação com investidores",
        "não encaminhe o e-mail com os KPIs ainda não publicados",
        "cuidado com o compartilhamento, está coberto por confidencialidade",
        "vamos tratar offline para evitar rastros, ok?"
    ]
    negativos_base = [
        "podemos marcar a reunião para terça-feira às 10h?",
        "segue a ata da reunião de ontem para revisão",
        "o contrato foi revisado pelo jurídico, tudo ok",
        "vamos consolidar os números públicos do trimestre",
        "segue o link do press release já publicado",
        "o cliente aprovou a proposta, prosseguir com o onboard",
        "precisamos da assinatura no documento até sexta",
        "reunião com o time de vendas às 15h",
        "apresentação do roadmap foi atualizada",
        "ajustei o orçamento conforme a planilha pública",
        "pauta do comitê: metas e resultados publicados",
        "precisamos alinhar a campanha de marketing",
        "o investidor pediu o relatório trimestral público",
        "o link do site oficial está no e-mail",
        "o NDA expirou e os dados já são públicos",
        "vamos revisar a documentação do produto",
        "o relatório está no drive público da empresa",
        "precisamos atualizar o FAQ no site",
        "confirmar sala para a reunião com parceiros",
        "segue a gravação do webinar aberto"
    ]
    random.seed(42)
    negativos = []
    for _ in range(800):
        base = random.choice(negativos_base)
        if random.random() < 0.2: base += " por favor confirmar."
        if random.random() < 0.1: base = "Re: " + base
        negativos.append(base)
    negativos_dificeis = [
        "o NDA foi encerrado, podemos tornar os dados públicos a partir de hoje",
        "antes do anúncio público, não tínhamos nada a comunicar; agora já saiu no site",
        "o link que vazou era falso, já corrigimos e não havia dados sensíveis",
        "blackout acabou ontem, podemos comentar os resultados públicos",
        "informação privilegiada não deve ser compartilhada; reforçar política com o time"
    ]
    df = pd.DataFrame({
        "Body": negativos + negativos_dificeis + positivos,
        "Label": [0]*len(negativos) + [0]*len(negativos_dificeis) + [1]*len(positivos)
    }).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

# -----------------------------
# 2) Treino (sintético + extra)
# -----------------------------
def train_model(model_name: str, extra_path: str = None, extra_sheet=0,
                test_size=0.2, rs=42, outdir="saida_scores"):
    ensure_dir(outdir)

    # Base sintética
    df_train = synthetic_df()

    # Anexa extra rotulado (se existir)
    if extra_path:
        df_extra = read_any_table(extra_path, sheet=extra_sheet)
        col_body = find_col(df_extra, ["Body", "body", "texto", "text", "message"])
        col_label = find_col(df_extra, ["Label", "label", "y"])
        if not col_body or not col_label:
            raise ValueError("Treino extra precisa ter colunas Body e Label (0/1).")
        df_extra = df_extra.rename(columns={col_body: "Body", col_label: "Label"})
        df_extra["Body"] = df_extra["Body"].apply(clean_text)
        df_extra = df_extra.dropna(subset=["Body", "Label"])
        # garantir 0/1
        df_extra["Label"] = df_extra["Label"].apply(lambda x: int(x))
        # concat
        df_train = pd.concat([df_train, df_extra], ignore_index=True)
        df_train = df_train.sample(frac=1.0, random_state=rs).reset_index(drop=True)

    # limpeza
    df_train["Body"] = df_train["Body"].apply(clean_text)
    df_train = df_train[~df_train["Body"].eq("")].reset_index(drop=True)

    # split p/ avaliação (se só sintético, ainda assim avaliamos)
    X_tr_text, X_va_text, y_tr, y_va = train_test_split(
        df_train["Body"], df_train["Label"],
        test_size=test_size, stratify=df_train["Label"], random_state=rs
    )

    # embeddings
    mdl = SentenceTransformer(model_name)
    Xtr = embed_texts(mdl, X_tr_text.tolist())
    Xva = embed_texts(mdl, X_va_text.tolist())

    # modelo calibrado
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=rs)
    lr.fit(Xtr, y_tr)
    cal = CalibratedClassifierCV(
        base_estimator=LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=rs),
        method="sigmoid", cv=3
    )
    cal.fit(Xtr, y_tr)

    # avaliação interna (informativa)
    prob_va = cal.predict_proba(Xva)[:, 1]
    ap = plot_pr(y_va, prob_va, os.path.join(outdir, "pr_curve_valid.png"))
    print(f"[Treino] Average Precision (validação): {ap:.3f}")
    for k in [5, 10, 20, 50, 100]:
        if k <= len(y_va):
            print(f"[Treino] Precision@{k}: {precision_at_k(list(y_va), list(prob_va), k):.3f}")

    return mdl, cal

# -----------------------------
# 3) Scoring do Excel final (MsgID, Body)
# -----------------------------
def score_excel(model, calibrator, score_path: str, score_sheet=0, outdir="saida_scores", topk=200):
    df = read_any_table(score_path, sheet=score_sheet)
    col_msg = find_col(df, ["MsgID", "msgid", "id", "messageid"])
    col_body = find_col(df, ["Body", "body", "texto", "text", "message", "conteudo"])
    if not col_msg or not col_body:
        raise ValueError("O arquivo de score precisa ter colunas MsgID e Body.")
    df = df.rename(columns={col_msg: "MsgID", col_body: "Body"})
    df["Body"] = df["Body"].apply(clean_text)
    df = df.dropna(subset=["MsgID", "Body"]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("Nenhuma linha válida para pontuar.")

    X = embed_texts(model, df["Body"].tolist())
    probs = calibrator.predict_proba(X)[:, 1]

    out = df[["MsgID", "Body"]].copy()
    out["prob_risco"] = probs
    out = out.sort_values("prob_risco", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    ensure_dir(outdir)
    all_path = os.path.join(outdir, "scores_no_seu_excel.csv")
    out.to_csv(all_path, index=False)
    print(f">> Scores completos salvos em: {all_path}")

    topk_eff = min(topk, len(out))
    top_path = os.path.join(outdir, f"top_{topk_eff}_para_revisao.csv")
    out.head(topk_eff).to_csv(top_path, index=False)
    print(f">> Top-{topk_eff} salvo em: {top_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    args = get_args()
    random.seed(args.random_state); np.random.seed(args.random_state)
    ensure_dir(args.output_dir)

    # 1) treina (sempre com sintético; + extra se fornecido)
    model, calibrator = train_model(
        model_name=args.model_name,
        extra_path=args.train_extra_xlsx,
        extra_sheet=args.train_sheet,
        test_size=args.test_size,
        rs=args.random_state,
        outdir=args.output_dir
    )

    # 2) pontua seu Excel final
    score_excel(
        model, calibrator,
        score_path=args.score_xlsx,
        score_sheet=args.score_sheet,
        outdir=args.output_dir,
        topk=args.topk
    )

if __name__ == "__main__":
    main()

Como usar

A) Só sintético → treina e pontua seu Excel

python train_and_score.py \
  --score_xlsx MEU_EXCEL_FINAL.xlsx --score_sheet 0 \
  --output_dir saida_scores --topk 200

B) Com treino extra rotulado (reforço)

python train_and_score.py \
  --train_extra_xlsx MEUS_ROTULADOS.xlsx --train_sheet 0 \
  --score_xlsx MEU_EXCEL_FINAL.xlsx --score_sheet 0 \
  --output_dir saida_scores --topk 200

Formatos esperados

Treino extra (MEUS_ROTULADOS.xlsx): colunas Body (texto) e Label (0/1).

Excel final (MEU_EXCEL_FINAL.xlsx): colunas MsgID e Body.


Saídas

saida_scores/pr_curve_valid.png (curva PR do treino, informativa)

saida_scores/scores_no_seu_excel.csv (MsgID, Body, prob_risco, rank)

saida_scores/top_200_para_revisao.csv (Top-K)


Se quiser, eu adiciono um threshold e já gero um JSON no formato do teu verificador LLM (ex.: lista de MsgID acima do corte) pra plugar direto no LangChain/LangGraph.

