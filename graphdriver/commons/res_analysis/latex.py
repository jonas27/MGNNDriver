import pandas as pd
from graphdriver.commons.res_analysis import relative, scores, topk
from graphdriver.utils import paths


def ranking(path="./ranking.tex"):
    dfs = []
    rels = relative.relative_all()
    df = pd.DataFrame(rels).T
    df = ranking_standardize_df(df=df, column_name="Rank median")
    dfs.append(df)

    for k in [5, 10, 30]:
        k_per = k / 100
        tk = topk.topk_all(k=k_per)
        df = pd.DataFrame(tk).T
        df = ranking_standardize_df(df=df, column_name="TPR in top {}\%".format(k))
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df.to_latex(path, escape=False)
    return dfs


def ranking_standardize_df(df, column_name: str):
    df.loc["mean"] = df.mean()
    df = df.applymap("{0:.3f}".format)
    df[column_name] = df["mean"] + "$\pm$" + df["std"]
    df = df[[column_name]]
    df.index = "\textbf{" + df.index.str.upper() + "}"
    return df


def scors(path="./scores.tex"):
    dfs = []
    types = [[["genes", "ppi"], "MGNNdriver"], [["genes"], "MGNNdriver-exp"], [["ppi"], "MGNNdriver-ppi"]]
    for t in types:
        scores_dict = scores.scores_all(network_type=t[0])
        df = pd.DataFrame(scores_dict).T
        df = ranking_standardize_df(df, t[1])
        dfs.append(df)

    df = pd.concat(dfs, axis=1)

    # add deepdriver
    df_deep = paths.pd_load(path=paths.results_deepdriver() + "results_summary")
    df_deep.loc["mean"] = df_deep.mean()
    df_deep = df_deep.applymap("{0:.3f}".format)
    df_deep["column_name"] = df_deep["deepdriver_mean"] + "$\pm$" + df_deep["deepdriver_std"]
    df["DeepDriver"] = df_deep["column_name"].to_list()

    df["Emogi"] = "--"
    if not path is None:
        df.to_latex(path, escape=False)
    return df
