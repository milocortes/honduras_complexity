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
    return (pl,)


@app.cell
def _(pl):
    ## Cargamos correspondencia CIIU Rev 2 (3 Digitos) a CIIU Rev 4 (4 Dígitos)
    cw_ciiu_rev_2_ciiu_rev_4 = pl.read_csv("datos/recodificacion/ciiu-rev-2_to_ciiu-rev-4.csv")

    ## Calculamos el peso relativo de la actividad CIIU Rev 4 (4 Dígitos) en las correspondencias totales de actividades CIIU Rev 2 (3 Digitos) para posteriormente usarlas como pesos en el cálculo de la media ponderada de la actividad
    cw_ciiu_rev_2_ciiu_rev_4 = cw_ciiu_rev_2_ciiu_rev_4.with_columns(
        ( 
            pl.col("weight")/pl.col("weight").sum().over("ciiu4")
        ).alias("composicion")
    )
    cw_ciiu_rev_2_ciiu_rev_4
    return (cw_ciiu_rev_2_ciiu_rev_4,)


@app.cell
def _(pl):
    ## Cargamos Datos de Institutional Intensity en CIIU Rev 2 (3 Digitos)
    inst_intensity = pl.read_csv("datos/viabilidad_atractivo/institutional_intensity.csv")
    inst_intensity
    return (inst_intensity,)


@app.cell
def _(cw_ciiu_rev_2_ciiu_rev_4, inst_intensity, pl):
    ### Reunimos el valor de institutional intensity y el crosswalk CIIU-Rev-2-CIIU-Rev-4
    ### y calculamos la media ponderada por industria CIIU
    cw_institutional_intensity = cw_ciiu_rev_2_ciiu_rev_4.join(
        inst_intensity.select("ISIC", "Institutional Intensity"), 
        left_on="ciiu2", 
        right_on="ISIC", 
        how="left"
    ).group_by("ciiu4").agg(
            institutional_intensity = (pl.col("Institutional Intensity") * pl.col("composicion")).sum() / pl.col("composicion").sum()
        )
    cw_institutional_intensity
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
