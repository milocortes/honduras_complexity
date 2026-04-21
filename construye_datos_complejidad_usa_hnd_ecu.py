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
    return (ue_actividades_transables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Carga recodificación
    """)
    return


@app.cell
def _(pl, ue_actividades_transables):
    ## Cargamos diccionario de recodificación de actividades
    recodificacion = pl.read_csv("datos/recodificacion/recodificacion_hnd_usa.csv")
    recodificacion = recodificacion.rename({"codigo": "actividad"})

    ## Mapeo de actividades transables con actividades CIIU Rev4
    recodificacion_map = recodificacion.filter(
        (pl.col("codigo_nuevo").is_in(ue_actividades_transables)) & 
        (pl.col("clasificador") == "ciiu_rev_4")
    ).select("codigo_nuevo", "actividad").with_columns(
        pl.col("actividad").cast(pl.String)
    )
    recodificacion_map
    return (recodificacion_map,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Carga datos de unidades económicas de HND, ECU y USA
    """)
    return


@app.cell
def _(pl):
    ## Guardamos UE de Municipios y Departamentos de honduras
    ue_honduras = pl.read_parquet("delta_db/ue_honduras.parquet")
    cbp = pl.read_parquet("delta_db/cbp.parquet")
    ue_ecuador = pl.read_parquet("datos/ecuador/SPSS_REEM_2024/ESTABLECIMIENTOS_2024.parquet")
    return cbp, ue_ecuador, ue_honduras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###USA, Ecuador y Honduras País
    """)
    return


@app.cell
def _(pl, recodificacion_map, ue_ecuador):
    # Recodificamos actividades CIIU Rev 4
    ue_ecuador_pais = ue_ecuador.select("codigo_clase").with_columns(
        pl.col("codigo_clase").map_elements(lambda x : x[1:])
    ).group_by("codigo_clase").count()

    # Reunimos datos con recodificación
    ue_ecuador_pais = ue_ecuador_pais.join(
        recodificacion_map, 
        left_on="codigo_clase", 
        right_on="actividad"
    )

    # Agrupamos datos con la nueva clasificación
    ue_ecuador_pais = ue_ecuador_pais.group_by("codigo_nuevo").agg(pl.col("count").sum())

    # Agregamos etiqueta de pais
    ue_ecuador_pais = ue_ecuador_pais.with_columns(
        zona = pl.lit("ECU"), 
        year = pl.lit("2024")
    ).rename(
        {"codigo_nuevo" : "actividad", "count" : "ue"}
    ).select("zona", "actividad", "ue", "year")
    ue_ecuador_pais
    return (ue_ecuador_pais,)


@app.cell
def _(cbp, pd, pl, ue_actividades_transables, ue_ecuador_pais, ue_honduras):
    ## Construye datos de UE USA, ECU y HND nivel país
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
        [
             cbp_ue.to_pandas(), 
             ue_honduras_pais.to_pandas(), 
             ue_ecuador_pais.to_pandas()
        ], ignore_index=True
    )

    usa_hnd_pais["year"] = 2024
    usa_hnd_pais
    return (usa_hnd_pais,)


@app.cell
def _(ecomplexity, pl, usa_hnd_pais):
    # Calculate complexity for UE HND País
    trade_cols_ue = {'time':'year', 'loc':'zona', 'prod':'actividad', 'val':'ue'}
    ue_honduras_pais_cdata = pl.from_pandas(ecomplexity(usa_hnd_pais, trade_cols_ue))
    ue_honduras_pais_cdata
    return (ue_honduras_pais_cdata,)


@app.cell
def _(pl, ue_honduras_pais_cdata):
    import altair as alt 

    ue_honduras_pais_cdata.filter(
        (pl.col("zona")=="HND") & 
        (pl.col("mcp") == 0)
    ).with_columns(
        distance = 1 - pl.col("density")
    ).plot.point(
        x = alt.X("distance", scale=alt.Scale(zero=False)), 
        y = alt.Y("pci", scale=alt.Scale(zero=False))
    )
    return (alt,)


@app.cell
def _(alt, pl, ue_honduras_pais_cdata):
    ue_honduras_pais_cdata.filter(
        (pl.col("zona")=="ECU") & 
        (pl.col("mcp") == 0)
    ).with_columns(
        distance = 1 - pl.col("density")
    ).plot.point(
        x = alt.X("distance", scale=alt.Scale(zero=False)), 
        y = alt.Y("pci", scale=alt.Scale(zero=False))
    )
    return


@app.cell
def _(ue_honduras_pais_cdata):
    ue_honduras_pais_cdata.select("zona", "eci").unique().sort("eci", descending=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
