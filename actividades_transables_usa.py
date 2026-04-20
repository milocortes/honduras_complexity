import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    import pandas as pd
    import polars as pl
    return pd, pl


@app.cell
def _(pl):
    ## Cargamos diccionario de recodificación de actividades
    recodificacion = pl.read_csv("datos/recodificacion/recodificacion_hnd_usa.csv")
    recodificacion = recodificacion.rename({"codigo": "actividad"})
    recodificacion
    return (recodificacion,)


@app.cell
def _(pl):
    cbp_actividades = pl.read_delta("delta_db/cbp")["actividad"].unique()
    cbp_actividades
    return (cbp_actividades,)


@app.cell
def _(pd, pl):
    ## Cargamos CBP 2023
    cbp = pd.read_csv("datos/usa/cbp/datos/cbp_msa/cbp23msa.txt")

    ## Nos quedamos sólo con los valores de las clases
    cbp = cbp[cbp["naics"].apply(lambda x : x[-1].isnumeric())].reset_index(drop=True)

    ## Nos quedamos con msa, naics y est
    cbp = cbp[["msa", "naics", "est", "emp"]]

    ## Sustituimos los nombres de las MSA
    ### Pegamos nombres de MSA
    msa_nombres = pd.read_csv("datos/usa/cbp/datos/Core_based_statistical_area_for_the_US_July_2023_-5413359380187677741.csv")
    #msa_nombres = msa_nombres[msa_nombres["CBSA Type"]=='Metropolitan Statistical Area']

    msa_nombres = msa_nombres[["CBSA Code", "CBSA Title"]]

    cbp = cbp.merge(
        msa_nombres, 
        left_on = "msa", 
        right_on= "CBSA Code", 
        how = "inner"
    )
    cbp = cbp[["CBSA Code", "CBSA Title","naics", "est", "emp"]]
    cbp.columns = ["zona_code", "zona", "actividad", "est", "emp"]
    cbp["actividad"] = cbp["actividad"].astype(int)
    cbp = pl.from_pandas(cbp)
    cbp
    return (cbp,)


@app.cell
def _(pd):
    ### Cargamos clases transables
    transables = pd.read_csv("datos/transables/clases_transables_naics.csv")
    transables
    return (transables,)


@app.cell
def _(cbp, transables):
    coincidencia_transables = set(transables["clase"]).intersection(cbp["actividad"])
    fuera_transables = set(transables["clase"]) - coincidencia_transables 
    fuera_transables
    return (coincidencia_transables,)


@app.cell
def _(cbp, cbp_actividades, coincidencia_transables, pl, recodificacion):
    acumula = []

    for actividad in cbp_actividades.to_numpy():
        actividades_naics = recodificacion.filter(
            (pl.col("codigo_nuevo") == actividad) &
            (pl.col("clasificador") == "naics_2022")
        )["actividad"].to_list()

        actividades_naics_transables = list(set(actividades_naics).intersection(coincidencia_transables))

        df_actividades = cbp.filter(
            pl.col("actividad").is_in(actividades_naics)
        )

        df_actividades_transables = cbp.filter(
            pl.col("actividad").is_in(actividades_naics_transables)
        )

        if df_actividades.is_empty():
            df_actividades_sum = pl.DataFrame(
                {
                    "actividad" : [actividad], 
                    "est" : [0], 
                    "emp" : [0]
                }
            )
        else: 
            df_actividades_sum = df_actividades.select("est", "emp").sum().with_columns(
                actividad = actividad
            ).select("actividad", "est", "emp")

        if df_actividades_transables.is_empty(): 
            df_actividades_transables_sum = pl.DataFrame(
                {
                    "actividad" : [actividad], 
                    "est_transable" : [0], 
                    "emp_transable" : [0]
                }
            )
        else: 
            df_actividades_transables_sum = df_actividades_transables.select("est", "emp").sum().with_columns(
                actividad = actividad
            ).select("actividad", "est", "emp").rename({"est" : "est_transable", "emp" : "emp_transable"})


        df_reune_actividades = df_actividades_sum.join(
            df_actividades_transables_sum, 
            on = "actividad"
        )

        acumula.append(
            df_reune_actividades
        )
    return (acumula,)


@app.cell
def _(acumula, pl):
    resumen_actividades = pl.concat(acumula)
    resumen_actividades = resumen_actividades.with_columns(
        razon_est_transables = pl.col("est_transable")/pl.col("est"), 
        razon_emp_transables = pl.col("emp_transable")/pl.col("emp"), 
    )
    resumen_actividades.filter(razon_est_transables=0)
    return (resumen_actividades,)


@app.cell
def _():
    289 - 59
    return


@app.cell
def _(pl, recodificacion, resumen_actividades):
    recodificacion.filter(
        (pl.col("codigo_nuevo").is_in(resumen_actividades.filter(razon_est_transables=0)["actividad"])) &
        (pl.col("clasificador") == 'ciiu_rev_4')
    )
    return


@app.cell
def _(recodificacion):
    recodificacion.filter(codigo_nuevo=298)
    return


@app.cell
def _(resumen_actividades):
    resumen_actividades.write_csv("output/actividades_transables/actividades_transables.csv")
    return


@app.cell
def _(resumen_actividades):
    resumen_actividades#.filter(razon_est_transables=1)
    return


if __name__ == "__main__":
    app.run()
