import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import polars as pl
    import numpy as np
    import math
    return math, np, pd, pl


@app.cell
def _(pl):
    ### Cargamos datos recodificados de UE de USA
    ue_cbp = pl.read_csv("output/recodificacion/ue/cbp.csv")
    ue_cbp = ue_cbp.with_columns(
        year = pl.lit(2024)
    )
    ue_cbp
    return (ue_cbp,)


@app.cell
def _(ue_cbp):
    ### Probamos con el paquete del GL
    from ecomplexity import ecomplexity

    # Calculate complexity
    trade_cols = {'time':'year', 'loc':'zona', 'prod':'actividad', 'val':'valor'}
    cdata = ecomplexity(ue_cbp.to_pandas(), trade_cols)
    cdata
    return (cdata,)


@app.cell
def _(pd):
    ### Cargamos recodificacion
    recodificacion = pd.read_csv("datos/recodificacion/recodificacion_hnd_usa_mex.csv")
    recodificacion = recodificacion.rename(columns={"codigo": "actividad"})
    return (recodificacion,)


@app.cell
def _(recodificacion):
    recodificacion.query("clasificador=='ciiu_rev_4' and codigo_nuevo==166")
    return


@app.cell
def _(pl, ue_cbp):

    datos_rca = ue_cbp.with_columns(
        rca = (
            pl.col("valor")/pl.col("valor").sum().over("zona")
        ) /
        (
            pl.col("valor").sum().over("actividad")/pl.col("valor").sum()
        )

    ).with_columns(
        pl.col("actividad").cast(pl.Int64)
    )
    datos_rca
    return (datos_rca,)


@app.cell
def _(datos_rca, pl):
    ## Umbral de RCA
    rca_umbral = 1.0

    datos_rca_m = datos_rca.with_columns(
        M = pl.when(
            pl.col("rca")>= rca_umbral   
        ).then(
            pl.lit(1)
        ).otherwise(
            pl.lit(0)
        )
    )

    datos_rca_m
    return (datos_rca_m,)


@app.cell
def _(datos_rca_m):
    M_df = datos_rca_m.pivot("actividad", 
                      index = "zona", 
                      values = "M"
                ).fill_null(0)
    M_df
    return (M_df,)


@app.cell
def _(M_df, pl):
    M = M_df.select(
        pl.exclude("zona")
    ).to_numpy()

    M
    return (M,)


@app.cell
def _(M, np):
    ## Calculamos diversidad
    diversidad = M.sum(axis = 1)
    D = np.diag(diversidad)

    ## Calculamos ubicuidad
    ubicuidad = M.sum(axis = 0)
    U = np.diag(ubicuidad)
    return D, U, diversidad


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde cc
    M_tilde_cc = np.linalg.pinv(D) @ M @ np.linalg.pinv(U) @ M.T
    M_tilde_cc 
    return (M_tilde_cc,)


@app.cell
def _(M_tilde_cc, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    eigenvalues_cc, eigenvectors_cc = np.linalg.eig(M_tilde_cc)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    Kc = eigenvectors_cc[:, 1].astype(float)

    Kc
    return (Kc,)


@app.cell
def _(D, Kc, M, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    #eigenvalues_pp, eigenvectors_pp = np.linalg.eig(M_tilde_pp)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    #Kp = eigenvectors_pp[:, 1].astype(float)
    Kp = np.linalg.pinv(M) @ D @ Kc

    Kp 
    return (Kp,)


@app.cell
def _(Kc, Kp, diversidad, math, np):
    ## Adjust sign of ECI and PCI so it makes sense, as per book
    corr_mat = np.corrcoef(diversidad, Kc)
    s1 = math.copysign(1.0, corr_mat[0,1])
    Kp_adj = s1*Kp
    Kc_adj = s1*Kc
    return Kc_adj, Kp_adj


@app.cell
def _(Kc_adj, Kp_adj, np):
    ## Ajustamos y normalizamos ECI y PCI
    eci = (Kc_adj - np.mean(Kc_adj))/np.std(Kc_adj)
    pci = (Kp_adj - np.mean(Kc_adj))/np.std(Kc_adj)
    return eci, pci


@app.cell
def _(M_df, eci, pl):
    ### Reunimos los datos calculados 
    df_eci_test = pl.DataFrame(
          {
            "zona" : M_df["zona"], 
            "eci" : eci  
        }  
    )

    df_eci_test
    return


@app.cell
def _(cdata, pl):
    pl.from_pandas(cdata[["zona", "eci"]]).unique()
    return


@app.cell
def _(M_df, pci, pl):
    ### Reunimos los datos calculados
    df_pci_test = pl.DataFrame(
          {
            "ciiu" : [int(i) for i in M_df.columns[1:]], 
            "pci" : pci  
        }  
    )

    df_pci_test 
    return


@app.cell
def _(cdata, pl):
    pl.from_pandas(cdata[["actividad", "pci"]]).unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bipartite Configuration Model

    The Bipartite Configuration Model (BiCM) is a statistical null model for binary bipartite networks [Squartini2011], [Saracco2015]. It offers an unbiased method for analyzing node similarities and obtaining statistically validated monopartite projections [Saracco2017].

    The BiCM belongs to a series of entropy-based null models for binary bipartite networks, see also
    """)
    return


@app.cell
def _():
    from src.bicm import BiCM
    return (BiCM,)


@app.cell
def _(np):
    mat = np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]])
    return (mat,)


@app.cell
def _(BiCM, mat):
    cm = BiCM(bin_mat=mat)
    return (cm,)


@app.cell
def _(cm):
    cm.make_bicm()
    return


@app.cell
def _(cm):
    cm.adj_matrix
    return


@app.cell
def _(cm):
    cm.lambda_motifs
    return


if __name__ == "__main__":
    app.run()
