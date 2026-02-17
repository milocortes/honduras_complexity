import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import math
    return math, np, pl


@app.cell
def _(pl):
    ## Cargamos N zonas metropolitanas de méxico
    conjunto_zm = pl.read_csv("datos/zm_comparables/70_ZM.csv")
    return (conjunto_zm,)


@app.cell
def _(pl):
    ## Cargamos datos de denue
    denue = pl.read_parquet(
                    "datos/denue/denue_filtrado_clase.parquet"
            ).with_columns(
                pl.concat_str([
                    pl.col("cve_ent").map_elements(lambda x : f"{x:02}"), 
                    pl.col("cve_mun").map_elements(lambda x : f"{x:03}"), 
                ]).alias("cvegeo")
            ).drop(
                    "cve_ent", "cve_mun"
            ).group_by("cvegeo", "codigo_act").len()

    return (denue,)


@app.cell
def _(denue):
    denue
    return


@app.cell
def _():
    return


@app.cell
def _(conjunto_zm, pl):
    ## Cargamos datos de Municipios en Zonas Metropolitanas
    zm_mun = pl.read_excel("datos/zm_cw_municipios/92 ZM_new.xlsx").select(
        "CVEGEO", "NOM_ZM"
    ).rename(
        {
            "CVEGEO" : "cvegeo", 
            "NOM_ZM" : "nom_zm"
        }
    )

    zm_mun = zm_mun.filter(
        pl.col("nom_zm").is_in(conjunto_zm["zm_nombre"])
    )
    zm_mun
    return (zm_mun,)


@app.cell
def _(denue, zm_mun):
    denue_zm = denue.join(
        zm_mun,
        on = "cvegeo", 
        how="left"
    ).drop_nulls().drop(
        "cvegeo"
    )
    denue_zm
    return (denue_zm,)


@app.cell
def _(denue_zm, pl):
    denue_zm_agg = denue_zm.group_by(
        "nom_zm", "codigo_act"
    ).agg(
        pl.col("len").sum()
    )
    denue_zm_agg
    return (denue_zm_agg,)


@app.cell
def _(pl):
    ## Cargamos recodificación de actividades 
    codificacion = pl.read_csv(
                    "datos/recodificacion/recodificacion.csv"
                    ).filter(
                        clasificador="scian"
                    ).select(
                        "codigo", "codigo_nuevo"
                    )
    codificacion
    return (codificacion,)


@app.cell
def _(codificacion, denue_zm_agg, pl):
    ## Reunimos con los conteos de denue a zm
    denue_zm_recod = denue_zm_agg.join(
        codificacion, 
        left_on="codigo_act", 
        right_on="codigo", 
        how="left"
    ).drop_nulls().select(
        "nom_zm", "codigo_nuevo", "len"
    ).group_by(
        "nom_zm", "codigo_nuevo"
    ).agg(
        pl.col("len").sum()
    )

    denue_zm_recod
    return (denue_zm_recod,)


@app.cell
def _(denue_zm_recod, pl):
    datos_rca = denue_zm_recod.with_columns(
        rca = (
            pl.col("len")/pl.col("len").sum().over("nom_zm")
        ) /
        (
            pl.col("len").sum().over("codigo_nuevo")/pl.col("len").sum()
        )

    ).with_columns(
        pl.col("codigo_nuevo").cast(pl.Int64)
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
    M_df = datos_rca_m.pivot("codigo_nuevo", 
                      index = "nom_zm", 
                      values = "M"
                ).fill_null(0)
    M_df
    return (M_df,)


@app.cell
def _(M_df, pl):
    M = M_df.select(
        pl.exclude("nom_zm")
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
    return D, U, diversidad, ubicuidad


@app.cell
def _(M, np, ubicuidad):
    proximity = M.T @ M / ubicuidad[np.newaxis, :]  
    proximity = np.minimum(proximity, proximity.T)
    proximity = np.nan_to_num(proximity)
    proximity
    return (proximity,)


@app.cell
def _(M, np, proximity):
    density = (np.dot(M,proximity)/np.sum(proximity, axis=1))
    density = np.nan_to_num(density)
    density
    return


@app.cell
def _(M, np, proximity):
    distancia = (np.dot((1 - M),proximity)/np.sum(proximity, axis=1))
    distancia = np.nan_to_num(distancia)
    distancia
    return


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde cc
    M_tilde_cc = np.linalg.pinv(D) @ M @ np.linalg.pinv(U) @ M.T
    M_tilde_cc 
    return (M_tilde_cc,)


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde pp
    M_tilde_pp = np.linalg.pinv(U) @ M.T @ np.linalg.pinv(D) @ M
    M_tilde_pp 
    return (M_tilde_pp,)


@app.cell
def _(M_tilde_cc, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    eigenvalues_cc, eigenvectors_cc = np.linalg.eig(M_tilde_cc)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    Kc = eigenvectors_cc[:, 1].astype(float)

    Kc
    return (Kc,)


@app.cell
def _(M_tilde_pp, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    eigenvalues_pp, eigenvectors_pp = np.linalg.eig(M_tilde_pp)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    Kp = eigenvectors_pp[:, 1].astype(float)

    Kp 
    return (Kp,)


@app.cell
def _(Kc, diversidad, math, np):
    ## Adjust sign of ECI and PCI so it makes sense, as per book
    corr_mat = np.corrcoef(diversidad, Kc)
    s1 = math.copysign(1.0, corr_mat[0,1])
    return (s1,)


@app.cell
def _(Kc, Kp, np, s1):
    ## Ajustamos y normalizamos ECI y PCI
    eci = s1*(Kc - np.mean(Kc))/np.std(Kc)
    pci = (Kp - np.mean(Kp))/np.std(Kp)
    return eci, pci


@app.cell
def _(M_df, eci, pl):
    ### Reunimos los datos calculados 
    df_eci_test = pl.DataFrame(
          {
            "zm" : M_df["nom_zm"], 
            "eci" : eci  
        }  
    )

    df_eci_test
    return


@app.cell
def _(M_df, pci, pl):
    ### Reunimos los datos calculados
    df_pci_test = pl.DataFrame(
          {
            "ciiu_recod" : [int(i) for i in M_df.columns[1:]], 
            "pci" : pci  
        }  
    )

    df_pci_test
    return


@app.cell
def _(pl):
    pl.read_csv(
                    "datos/recodificacion/recodificacion.csv"
                    ).filter(
                        clasificador="ciiu_rev_4"
                    ).select(
                        "codigo_nuevo", "nombre_actividad"
                    )
    return


if __name__ == "__main__":
    app.run()
