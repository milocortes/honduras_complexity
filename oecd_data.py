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
    import pandas as pd
    from ecomplexity import ecomplexity
    import numpy as np

    import altair as alt
    from altair.datasets import data
    return alt, ecomplexity, np, pd, pl


@app.cell
def _(pd):
    # Cargamos recodificación
    recod = pd.read_csv("datos/recodificacion/recodificacion_hnd_usa.csv")

    # Cargamos transables 
    transables = pd.read_csv("datos/resumen_razones_transables.csv").query("razon_emp_transables > 0")

    recod = recod[recod["codigo_nuevo"].isin(transables["actividad"])]
    ciiu_transable = recod.query("clasificador =='ciiu_rev_4'")["codigo"].unique()
    ciiu_transable = [f"{i:04}" for i in ciiu_transable]
    return ciiu_transable, recod


@app.cell
def _(ciiu_transable):
    len(ciiu_transable)
    return


@app.cell
def _(pl):
    ## Cargamos datos
    datos = pl.read_parquet("datos/oecd/oecd_structural_bussiness.parquet")
    datos
    return (datos,)


@app.cell
def _(pd):
    ## Cargamos honduras empleo
    hnd = pd.read_csv("datos/empleo_honduras/empleo_honduras.csv")
    hnd["variable"] = hnd["variable"].apply(lambda x : f"{x:04}")
    hnd.columns = ["REF_AREA", "ciiu", "OBS_VALUE"]
    hnd["REF_AREA"] = "HND"
    hnd
    return (hnd,)


@app.cell
def _(datos, pl):
    empleo = datos.with_columns(
        ciiu = pl.col("ACTIVITY").map_elements(lambda x : x[1:])
    ).filter(
        (pl.col("Measure")=='Employees') & 
        (pl.col("TIME_PERIOD") == 2019)
    )
    empleo
    return (empleo,)


@app.cell
def _(ciiu_transable, empleo, hnd, pd):
    emp = empleo.select(
        "REF_AREA", "ciiu", "OBS_VALUE"
    ).to_pandas()

    emp = pd.concat([emp, hnd], ignore_index=True)

    emp["year"] = 2019
    emp = emp[emp["ciiu"].isin(ciiu_transable)].reset_index(drop = True)

    emp#["REF_AREA"].unique()
    return (emp,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(ecomplexity, emp):
    # Calculate complexity for UE
    trade_cols = {'time':'year', 'loc':'REF_AREA', 'prod':'ciiu', 'val':'OBS_VALUE'}
    cdata = ecomplexity(emp, trade_cols)
    return cdata, trade_cols


@app.cell
def _(cdata):
    cdata
    return


@app.cell
def _(cdata, pl):
    pl.from_pandas(cdata).select("REF_AREA", "eci").unique()
    return


@app.cell
def _(cdata, pl, recod):
    pl.from_pandas(cdata).select("ciiu", "pci").unique().join(
        pl.from_pandas(
            recod.query("clasificador == 'ciiu_rev_4'")
        ).select("codigo", "nombre_actividad").with_columns(
            pl.col("codigo").cast(pl.String)
        ), 
        left_on="ciiu", 
        right_on="codigo"
    ).select("ciiu", "nombre_actividad", "pci")
    return


@app.cell
def _():
    return


@app.cell
def _(alt, ciiu_transable, datos, hnd, np, pl):
    empleo_ocde = datos.with_columns(
        ciiu = pl.col("ACTIVITY").map_elements(lambda x : x[1:])
    ).filter(
        (pl.col("Measure")=='Employees') & 
        (pl.col("TIME_PERIOD") == 2019)
    ).select("REF_AREA", "ciiu", "OBS_VALUE").with_columns(
        pl.col("OBS_VALUE").map_elements(lambda x : np.log(x))
    )

    empleo_ocde = pl.concat(
        [empleo_ocde, pl.from_pandas(hnd).with_columns(
        pl.col("OBS_VALUE").map_elements(lambda x : np.log(x))
    )]
    ).filter(
        (pl.col("ciiu").is_in(ciiu_transable))
    )

    alt.Chart(empleo_ocde).mark_rect().encode(
        alt.X("REF_AREA"), 
        alt.Y("ciiu"), 
        alt.Color("OBS_VALUE")
    )
    return


@app.cell
def _():
    ### Que actividades no tenemos datos para los paises?
    ciiu_na = [
        #"6420", "6430", "6491", "6492", "6499", "6511", "6512", "6611", "6612", "6619", "6621", "6629", 
        #"0161", "0162", "0163", "0164", "0210", "0220", "0230", "0240", "0311", "0312", "0610", "0620", 
        "8510", "8522", "8530", "8541", "8549", "8550", "9000", "9102", "9103", "9200", "9311", "9312", "9319", "9321", "9329", "9412", 
        "6411" # Banca central
    ]
    return (ciiu_na,)


@app.cell
def _(ciiu_na, empleo, pl):
    empleo.select(
        "Economic activity", "ciiu"
    ).unique().filter(pl.col("ciiu").is_in(ciiu_na))
    return


@app.cell
def _(alt, ciiu_na, ciiu_transable, datos, hnd, np, pl):
    empleo_ocde_limpio = datos.with_columns(
        ciiu = pl.col("ACTIVITY").map_elements(lambda x : x[1:])
    ).filter(
        (pl.col("Measure")=='Employees') & 
        (pl.col("TIME_PERIOD") == 2019)
    ).select("REF_AREA", "ciiu", "OBS_VALUE").with_columns(
        pl.col("OBS_VALUE").map_elements(lambda x : np.log(x)).alias("OBS_VALUE_LOG")
    )

    empleo_ocde_limpio = pl.concat(
        [empleo_ocde_limpio, pl.from_pandas(hnd).with_columns(
        pl.col("OBS_VALUE").map_elements(lambda x : np.log(x)).alias("OBS_VALUE_LOG")
    )]
    ).filter(
        (pl.col("ciiu").is_in(ciiu_transable))
    ).filter(
        (~ pl.col("ciiu").is_in(ciiu_na))
    )
    alt.Chart(empleo_ocde_limpio).mark_rect().encode(
        alt.X("REF_AREA"), 
        alt.Y("ciiu"), 
        alt.Color("OBS_VALUE_LOG")
    )
    return (empleo_ocde_limpio,)


@app.cell
def _(ecomplexity, empleo_ocde_limpio, pl, trade_cols):
    cdata_limpio = ecomplexity(empleo_ocde_limpio.with_columns(
        year = pl.lit(2019)
    ).to_pandas(), trade_cols)
    return (cdata_limpio,)


@app.cell
def _(cdata_limpio, pl):
    pl.from_pandas(cdata_limpio).select("REF_AREA", "eci").unique().sort(by = "eci", descending = True)
    return


@app.cell
def _(cdata_limpio, pl, recod):
    pl.from_pandas(cdata_limpio).select("ciiu", "pci").unique().join(
        pl.from_pandas(
            recod.query("clasificador == 'ciiu_rev_4'")
        ).select("codigo", "nombre_actividad").with_columns(
            pl.col("codigo").cast(pl.String)
        ), 
        left_on="ciiu", 
        right_on="codigo"
    ).select("ciiu", "nombre_actividad", "pci")
    return


if __name__ == "__main__":
    app.run()
