import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl 
    import pandas as pd
    from ecomplexity import ecomplexity
    return ecomplexity, pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cargamos industrias transables
    """)
    return


@app.cell
def _(pl):
    transables = pl.read_csv("output/actividades_transables/actividades_transables.csv")
    transables
    return (transables,)


@app.cell
def _(pl, transables):
    ue_actividades_transables = transables.filter(
        pl.col("razon_est_transables") > 0.0
    )["actividad"]

    empleo_actividades_transables = transables.filter(
        pl.col("razon_emp_transables") > 0.0
    )["actividad"]
    return empleo_actividades_transables, ue_actividades_transables


@app.cell
def _(ue_actividades_transables):
    ue_actividades_transables
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Carga datos de empleo y unidades económicas de HND y USA
    """)
    return


@app.cell
def _(pl):
    ## Guardamos UE de Municipios y Departamentos de honduras
    ue_honduras = pl.read_parquet("delta_db/ue_honduras.parquet")
    cbp = pl.read_parquet("delta_db/cbp.parquet")
    empleo_honduras = pl.read_parquet("delta_db/empleo_honduras.parquet")
    return cbp, empleo_honduras, ue_honduras


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unidades Económicas
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USA y Honduras País
    """)
    return


@app.cell
def _(cbp, pd, pl, ue_actividades_transables, ue_honduras):
    ## Construye datos de UE USA y HND nivel país
    industrias_cbp = list(cbp["actividad"].unique())

    ue_honduras_pais = ue_honduras.select(
                    "actividad", "ue"
                ).group_by(
                    "actividad"
                ).sum().with_columns(
                    zona = pl.lit("HND")
                ).select("zona", "actividad", "ue").filter(
                pl.col("actividad").is_in(ue_actividades_transables)
            )

    cbp_ue = cbp.select(
                "zona", "actividad", "ue"
            ).filter(
                pl.col("actividad").is_in(ue_actividades_transables)
            )

    usa_hnd_pais = pd.concat(
        [cbp_ue.to_pandas(), 
         ue_honduras_pais.to_pandas()], ignore_index=True
    )

    usa_hnd_pais["year"] = 2024
    usa_hnd_pais
    return cbp_ue, industrias_cbp, usa_hnd_pais


@app.cell
def _(ecomplexity, pl, usa_hnd_pais):
    # Calculate complexity for UE HND País
    trade_cols_ue = {'time':'year', 'loc':'zona', 'prod':'actividad', 'val':'ue'}
    ue_honduras_pais_cdata = pl.from_pandas(ecomplexity(usa_hnd_pais, trade_cols_ue))
    ue_honduras_pais_cdata
    return trade_cols_ue, ue_honduras_pais_cdata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USA y Departamentos de Honduras
    """)
    return


@app.cell
def _(
    cbp_ue,
    empleo_actividades_transables,
    industrias_cbp,
    pd,
    pl,
    ue_honduras,
):
    ## Construye datos de UE USA y HND nivel Departamentos
    ue_honduras_deptos = ue_honduras.select(
                    "NAME_1", "actividad", "ue"
                ).group_by(
                    "NAME_1","actividad"
                ).sum().select("NAME_1", "actividad", "ue").filter(
                pl.col("actividad").is_in(industrias_cbp)
            ).rename({"NAME_1" : "zona"}).filter(
                pl.col("actividad").is_in(empleo_actividades_transables)
            )

    usa_hnd_deptos = pd.concat(
        [cbp_ue.to_pandas(), 
         ue_honduras_deptos.to_pandas()], ignore_index=True
    )

    usa_hnd_deptos["year"] = 2024
    usa_hnd_deptos
    return (usa_hnd_deptos,)


@app.cell
def _(ecomplexity, pl, trade_cols_ue, usa_hnd_deptos):
    # Calculate complexity for UE HND Departamentos
    ue_honduras_deptos_cdata = pl.from_pandas(ecomplexity(usa_hnd_deptos, trade_cols_ue))
    ue_honduras_deptos_cdata
    return (ue_honduras_deptos_cdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Empleo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USA y Honduras País
    """)
    return


@app.cell
def _(
    cbp,
    empleo_actividades_transables,
    empleo_honduras,
    industrias_cbp,
    pd,
    pl,
):
    ## Construye datos de UE USA y HND nivel Departamento

    empleo_honduras_pais = empleo_honduras.select(
                    "actividad", "empleo"
                ).group_by(
                    "actividad"
                ).sum().with_columns(
                    zona = pl.lit("HND")
                ).select("zona", "actividad", "empleo").filter(
                pl.col("actividad").is_in(industrias_cbp)
            ).filter(
                pl.col("actividad").is_in(empleo_actividades_transables)
            )

    cbp_empleo = cbp.select(
                "zona", "actividad", "empleo"
            ).filter(
                pl.col("actividad").is_in(empleo_actividades_transables)
            )

    empleo_usa_hnd_pais = pd.concat(
        [cbp_empleo.to_pandas(), 
         empleo_honduras_pais.to_pandas()], ignore_index=True
    )


    empleo_usa_hnd_pais["year"] = 2024

    empleo_usa_hnd_pais
    return cbp_empleo, empleo_usa_hnd_pais


@app.cell
def _(ecomplexity, empleo_usa_hnd_pais, pl):
    # Calculate complexity for UE HND País
    trade_cols_empleo = {'time':'year', 'loc':'zona', 'prod':'actividad', 'val':'empleo'}
    empleo_honduras_pais_cdata = pl.from_pandas(ecomplexity(empleo_usa_hnd_pais, trade_cols_empleo))
    empleo_honduras_pais_cdata
    return empleo_honduras_pais_cdata, trade_cols_empleo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## USA y Departamentos de Honduras
    """)
    return


@app.cell
def _(
    cbp_empleo,
    empleo_actividades_transables,
    empleo_honduras,
    industrias_cbp,
    pd,
    pl,
):
    ## Construye datos de UE USA y HND nivel Departamentos
    empleo_honduras_deptos = empleo_honduras.select(
                    "NAME_1", "actividad", "empleo"
                ).group_by(
                    "NAME_1","actividad"
                ).sum().select("NAME_1", "actividad", "empleo").filter(
                pl.col("actividad").is_in(industrias_cbp)
            ).rename({"NAME_1" : "zona"}).filter(
                pl.col("actividad").is_in(empleo_actividades_transables)
            )

    empleo_usa_hnd_deptos = pd.concat(
        [cbp_empleo.to_pandas(), 
         empleo_honduras_deptos.to_pandas()], ignore_index=True
    )


    empleo_usa_hnd_deptos["year"] = 2024
    empleo_usa_hnd_deptos
    return (empleo_usa_hnd_deptos,)


@app.cell
def _(ecomplexity, empleo_usa_hnd_deptos, pl, trade_cols_empleo):
    empleo_honduras_deptos_cdata = pl.from_pandas(ecomplexity(empleo_usa_hnd_deptos, trade_cols_empleo))
    empleo_honduras_deptos_cdata
    return (empleo_honduras_deptos_cdata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guardamos resultados
    """)
    return


@app.cell
def _(
    empleo_honduras_deptos_cdata,
    empleo_honduras_pais_cdata,
    ue_honduras_deptos_cdata,
    ue_honduras_pais_cdata,
):
    ## Guardamos resultados
    ue_honduras_pais_cdata.write_parquet("delta_db/cdata/ue_honduras_pais_cdata.parquet")
    ue_honduras_deptos_cdata.write_parquet("delta_db/cdata/ue_honduras_deptos_cdata.parquet")
    empleo_honduras_pais_cdata.write_parquet("delta_db/cdata/empleo_honduras_pais_cdata.parquet")
    empleo_honduras_deptos_cdata.write_parquet("delta_db/cdata/empleo_honduras_deptos_cdata.parquet")
    return


@app.cell
def _(pl, ue_honduras_deptos_cdata):
    import altair as alt 

    ue_honduras_deptos_cdata.filter(
        (pl.col("zona")=="Cortes") & 
        (pl.col("mcp") == 0)
    ).with_columns(
        distance = 1 - pl.col("density")
    ).plot.point(
        x = alt.X("distance", scale=alt.Scale(zero=False)), 
        y = alt.Y("pci", scale=alt.Scale(zero=False))
    )
    return (alt,)


@app.cell
def _(alt, pl, ue_honduras_deptos_cdata):

    ue_honduras_deptos_cdata.filter(
        (pl.col("zona")=="Los Angeles-Long Beach-Anaheim, CA") & 
        (pl.col("mcp") == 0)
    ).with_columns(
        distance = 1 - pl.col("density")
    ).plot.point(
        x = alt.X("distance", scale=alt.Scale(zero=False)), 
        y = alt.Y("pci", scale=alt.Scale(zero=False))
    )
    return


@app.cell
def _(ue_honduras_deptos_cdata):
    ue_honduras_deptos_cdata
    return


if __name__ == "__main__":
    app.run()
