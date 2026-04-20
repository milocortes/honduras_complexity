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
    import geopandas as gpd
    from unidecode import unidecode
    import math
    return gpd, pd, pl, unidecode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cargamos datos de Empleo y Establecimientos de USA (CBP)
    """)
    return


@app.cell
def _(pd):
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
    cbp
    return (cbp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cargamos datos de Unidades Económicas de Honduras, homogenizamos y recodificamos
    """)
    return


@app.cell
def _(gpd):
    ### Cargamos datos de GDP per capita
    hnd = gpd.read_file(
            "datos/gdp_per_capita_downscaled_gridded/polyg_adm2_gdp_perCapita_1990_2022.gpkg"
        ).query("iso3=='HND'")[["GID_2", "NAME_2"]]
    hnd
    return (hnd,)


@app.cell
def _(hnd, pl):
    ## Cargamos datos de UE de Honduras
    ue_honduras = pl.read_csv(
                        "datos/BASE-DEE-2024.csv"
                    ).select(["cod._departamento", "cod._municipio", "departamento", "codigo__ciiu_cuatro_digitos"])

    ## Diccionario de departamentos e identificador
    depto_dict = {i:j.lower().title() for i,j in ue_honduras.select("cod._departamento","departamento").unique().to_numpy() }
    depto_dict_min = { v.lower() : k for k,v in depto_dict.items()}


    ## Generamos llave para identificar municipios GID_2
    ue_honduras = ue_honduras.with_columns(
        GID_2 = pl.concat_str(
            pl.lit("HND"), 
            pl.col("cod._departamento"), 
            pl.concat_str([pl.col("cod._municipio"), pl.lit("1")], separator="_"), 
            separator="."
        ), 
        GID_1 = pl.concat_str(
            pl.lit("HND"), 
            pl.col("cod._departamento"), 
            separator="."
        )
    )

    ue_honduras = ue_honduras.to_pandas()
    ue_honduras["NAME_1"] = ue_honduras["cod._departamento"].replace(depto_dict)

    ue_honduras = ue_honduras.merge(
        hnd, 
        on = "GID_2"
    )
    ue_honduras = pl.from_pandas(
                        ue_honduras
                    ).select(
                        "GID_1", "GID_2", "NAME_1", "NAME_2", "codigo__ciiu_cuatro_digitos"
                    ).rename(
                        {
                            "codigo__ciiu_cuatro_digitos" : "ciiu"
                        }
                    )
    ue_honduras
    return depto_dict_min, ue_honduras


@app.cell
def _(ue_honduras):
    ## Obtenemos los conteos de UE
    ue_honduras_conteos = ue_honduras.group_by(
                "GID_1", "GID_2", "NAME_1", "NAME_2", "ciiu"
                ).len().rename({"len" : "ue"}).to_pandas()

    ue_honduras_conteos
    return (ue_honduras_conteos,)


@app.cell
def _(pd):
    ## Cargamos diccionario de recodificación de actividades
    recodificacion = pd.read_csv("datos/recodificacion/recodificacion_hnd_usa.csv")
    recodificacion = recodificacion.rename(columns={"codigo": "actividad"})
    recodificacion
    return (recodificacion,)


@app.cell
def _(recodificacion, ue_honduras_conteos):
    ### Identificamos las industrias para las cuales contamos con correspondencia
    industrias_ciiu_4_en_cw = set(ue_honduras_conteos["ciiu"]).intersection(recodificacion.query("clasificador == 'ciiu_rev_4'")["actividad"])
    industrias_ciiu_4_fuera_cw = industrias_ciiu_4_en_cw - set(ue_honduras_conteos["ciiu"]) 
    industrias_ciiu_4_fuera_cw
    return industrias_ciiu_4_en_cw, industrias_ciiu_4_fuera_cw


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ¿Qué industrias estamos dejando fuera?
    """)
    return


@app.cell
def _(industrias_ciiu_4_fuera_cw, pl, recodificacion):
    pl.from_pandas(recodificacion).filter(
        clasificador = 'ciiu_rev_4'
    ).drop("clasificador", "codigo_nuevo").filter(
        pl.col("actividad").is_in(industrias_ciiu_4_fuera_cw)
    )
    return


@app.cell
def _(industrias_ciiu_4_en_cw, recodificacion, ue_honduras_conteos):
    ### Nos quedamos con las industrias que hacen match
    ue_honduras_conteos_match = ue_honduras_conteos.query(f"ciiu in {list(industrias_ciiu_4_en_cw)}")
    ue_honduras_recod = ue_honduras_conteos_match.merge(
        recodificacion.query("clasificador =='ciiu_rev_4'"), 
        left_on = "ciiu",
        right_on="actividad",
        how = "inner"
    ).drop(
        columns=["clasificador", "nombre_actividad", "ciiu", "actividad"]
    ).groupby(
        ["GID_1", "GID_2", "NAME_1", "NAME_2", "codigo_nuevo"]
    ).agg(
        {"ue" : "sum"}
    ).reset_index().rename(columns = {"codigo_nuevo" : "actividad"})

    ue_honduras_recod
    return (ue_honduras_recod,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recodificamos datos de Empleo y Establecimientos de USA (CBP)
    """)
    return


@app.cell
def _(cbp, recodificacion):
    ### Reunimos el CW con los datos de USA
    cbp_recod = cbp.merge(
        recodificacion.query("clasificador =='naics_2022'"), 
        on = "actividad", 
        how = "inner"
    ).drop(
        columns = ["actividad", "clasificador", "nombre_actividad"]
    ).groupby(
        ["zona_code", "zona", "codigo_nuevo"]
    ).agg(
        {
            "est" : "sum", 
            "emp" : "sum"
        }
    ).reset_index().rename(columns={
        "codigo_nuevo" : "actividad", 
        "est" : "ue", 
        "emp" : "empleo"
    })
    cbp_recod
    return (cbp_recod,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cargamos datos de Empleo de Honduras, homogenizamos y recodificamos
    """)
    return


@app.cell
def _(pd):
    ## Datos empleo Honduras
    empleo_depto_hnd = pd.read_csv("datos/empleo_honduras/empleo_departamentos.csv")
    empleo_depto_hnd
    return (empleo_depto_hnd,)


@app.cell
def _(empleo_depto_hnd, industrias_ciiu_4_en_cw, recodificacion):
    ### Identificamos las industrias para las cuales contamos con correspondencia
    industrias_ciiu_4_en_cw_empleo = set(empleo_depto_hnd["actividad"]).intersection(recodificacion.query("clasificador == 'ciiu_rev_4'")["actividad"].to_numpy())
    industrias_ciiu_4_fuera_cw_empleo = industrias_ciiu_4_en_cw - set(empleo_depto_hnd["actividad"])  
    industrias_ciiu_4_fuera_cw_empleo
    return industrias_ciiu_4_en_cw_empleo, industrias_ciiu_4_fuera_cw_empleo


@app.cell
def _(industrias_ciiu_4_en_cw_empleo):
    2910 in list(industrias_ciiu_4_en_cw_empleo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### ¿Qué actividades estamos dejando fuera?
    """)
    return


@app.cell
def _(industrias_ciiu_4_fuera_cw_empleo, pl, recodificacion):
    pl.from_pandas(recodificacion).filter(
        clasificador = 'ciiu_rev_4'
    ).drop("clasificador", "codigo_nuevo").filter(
        pl.col("actividad").is_in(industrias_ciiu_4_fuera_cw_empleo)
    )
    return


@app.cell
def _(
    depto_dict_min,
    empleo_depto_hnd,
    industrias_ciiu_4_en_cw_empleo,
    pl,
    recodificacion,
    unidecode,
):
    ### Nos quedamos con las industrias que hacen match
    empleo_honduras_match = pl.from_pandas(empleo_depto_hnd).filter(pl.col("actividad").is_in(industrias_ciiu_4_en_cw_empleo)).to_pandas()
    empleo_honduras_recod = empleo_honduras_match.merge(
        recodificacion.query("clasificador =='ciiu_rev_4'"), 
        on = "actividad", 
        how = "inner"
    ).drop(columns=["clasificador", "actividad", "nombre_actividad"])

    ### Agregamos GID1 
    empleo_honduras_recod["GID_1"] = empleo_honduras_recod["zona"].apply(lambda x : f"HND.{depto_dict_min[unidecode(x).lower()]}")
    empleo_honduras_recod["NAME_1"] = empleo_honduras_recod["zona"].apply(lambda x : unidecode(x))

    empleo_honduras_recod = empleo_honduras_recod.rename(columns = {"valor" : "empleo", "codigo_nuevo" : "actividad"})
    empleo_honduras_recod = empleo_honduras_recod.drop(columns = "zona")[["GID_1", "NAME_1", "actividad", "empleo"]]
    empleo_honduras_recod
    return (empleo_honduras_recod,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Guardamos datos recodificados
    """)
    return


@app.cell
def _(cbp_recod, empleo_honduras_recod, pl, ue_honduras_recod):
    ## Guardamos UE de Municipios y Departamentos de honduras
    pl.from_pandas(ue_honduras_recod).write_parquet("delta_db/ue_honduras.parquet")
    pl.from_pandas(cbp_recod).write_parquet("delta_db/cbp.parquet")
    pl.from_pandas(empleo_honduras_recod).write_parquet("delta_db/empleo_honduras.parquet")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
