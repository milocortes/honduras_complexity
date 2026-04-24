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
    import matplotlib.pyplot as plt
    import numpy as np
    from ecomplexity import ecomplexity
    import altair as alt
    return alt, ecomplexity, np, pd, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Datos de OCDE SBS
    """)
    return


@app.cell
def _(pl):
    ### Define consulta tipo lazy para el acceso a los datos
    q = pl.scan_parquet('datos/oecd/ocde_sbs.parquet').filter(pl.col("Measure")=='Employees').select("REF_AREA", "ACTIVITY", "SIZE_CLASS", "OBS_VALUE", "TIME_PERIOD")

    ### Recolectamos la informacion
    df = q.collect()

    ### Lo convertimos a pandas
    df = df.to_pandas()

    ### Nos quedamos con las actividades a 4 digitos del CIIU
    df = df[df["ACTIVITY"].apply(lambda x : len(x)==5)]

    ### Define funcion que evalua si los últimos 4 caracteres son numéricos
    test_numericos = lambda cadena : all([i.isnumeric() for i in list(cadena)])

    df = df[df["ACTIVITY"].apply(lambda x : test_numericos(x[1:]))]

    ### Obten seccion
    df["seccion"] = df["ACTIVITY"].apply(lambda x : x[0])
    df["ACTIVITY"] = df["ACTIVITY"].apply(lambda x : x[1:])

    df
    return (df,)


@app.cell
def _(df, pd):
    ### Resume conteos de registros por sección
    acumula = []

    anio_min = df["TIME_PERIOD"].min()
    anio_max = df["TIME_PERIOD"].max()

    for i in range(anio_min, anio_max+1):
        consulta = df.query(f"TIME_PERIOD=={i}")["seccion"].value_counts().to_frame().T
        consulta["Total"] = consulta.sum(axis = 1)
        consulta["year"] = i
        acumula.append(consulta)

    resumen_seccion_conteos = pd.concat(acumula)
    resumen_seccion_conteos
    return


@app.cell
def _(df, np, plt):
    ## Resumen de empleo total por seccion

    resumen_empleo_anios = df.groupby(
        ["TIME_PERIOD", "seccion"]
    ).agg({"OBS_VALUE" : "sum"}).reset_index().pivot(index = "TIME_PERIOD", columns = "seccion", values = "OBS_VALUE")

    plt.imshow(np.log(resumen_empleo_anios.to_numpy()))

    return


@app.cell
def _(df):
    ### Consulta para verificar la cantidad de actividades en 2019 con valores mayores a 0
    df.query("TIME_PERIOD == 2019").query("OBS_VALUE>0").groupby(["REF_AREA", "ACTIVITY"]).agg({"OBS_VALUE" : "sum"}).reset_index().groupby("REF_AREA").agg({"ACTIVITY" : "count"})
    return


@app.cell
def _(df, pd, pl):
    ### Consulta para verificar la cantidad de actividades en 2019 con valores mayores a 0
    ### CONSIDERANDO ACTIVIDADES TRANSABLES
    # Cargamos recodificación
    recod = pd.read_csv("datos/recodificacion/recodificacion_hnd_usa.csv")

    ## Diccionario CIIU 4 a nombres
    mapp_ciiu = pl.from_pandas(recod.query("clasificador=='ciiu_rev_4'")[["codigo", "nombre_actividad"]].astype(str))

    # Cargamos transables 
    transables = pd.read_csv("datos/resumen_razones_transables.csv").query("razon_emp_transables > 0")

    recod = recod[recod["codigo_nuevo"].isin(transables["actividad"])]
    ciiu_transable = recod.query("clasificador =='ciiu_rev_4'")["codigo"].unique()
    ciiu_transable = [f"{i:04}" for i in ciiu_transable]

    df_actividades_transables = df.query("TIME_PERIOD == 2019").query(f"ACTIVITY in {ciiu_transable}").query("OBS_VALUE>0").groupby(["REF_AREA", "ACTIVITY"]).agg({"OBS_VALUE" : "sum"}).reset_index().groupby("REF_AREA").agg({"ACTIVITY" : "count"})

    df_actividades_transables["ACTIVITY"] = df_actividades_transables["ACTIVITY"]/len(ciiu_transable)

    df_actividades_transables
    return ciiu_transable, df_actividades_transables, mapp_ciiu


@app.cell
def _(df_actividades_transables):
    df_actividades_transables.sort_values(by="ACTIVITY", ascending=False).plot.bar()
    return


@app.cell
def _(ciiu_transable, df, df_actividades_transables, pd):
    ### Haremos el análisis de complejidad modificando la muestra de paises de acuerdo al umbral de la razón de actividades que reportan empleo vs total de actividades transables

    ### Que actividades no tenemos datos para los paises?
    ciiu_na = [
        #"6420", "6430", "6491", "6492", "6499", "6511", "6512", "6611", "6612", "6619", "6621", "6629", 
        #"0161", "0162", "0163", "0164", "0210", "0220", "0230", "0240", "0311", "0312", "0610", "0620", 
        "8510", "8522", "8530", "8541", "8549", "8550", "9000", "9102", "9103", "9200", "9311", "9312", "9319", "9321", "9329", "9412", 
        "6411" # Banca central
    ]

    umbral = 0.65
    anio_analisis = 2019

    paises_muestra = df_actividades_transables.reset_index().query(f"ACTIVITY>{umbral}")["REF_AREA"].to_list()

    insumos_complejidad = df.query(f"TIME_PERIOD == {anio_analisis}").query(f"ACTIVITY in {ciiu_transable}").query(f"REF_AREA in {paises_muestra}").query(f"ACTIVITY not in {ciiu_na}")


    insumos_complejidad = insumos_complejidad[["TIME_PERIOD", "REF_AREA", "ACTIVITY", "OBS_VALUE"]]

    muestra_actividades = list(insumos_complejidad["ACTIVITY"].unique())

    ### Cargamos Honduras
    hnd = pd.read_csv("datos/empleo_honduras/empleo_honduras.csv")
    hnd["variable"] = hnd["variable"].apply(lambda x : f"{x:04}")
    hnd.columns = ["REF_AREA", "ciiu", "OBS_VALUE"]
    hnd["REF_AREA"] = "HND"
    hnd["TIME_PERIOD"] = anio_analisis
    hnd = hnd[["TIME_PERIOD", "REF_AREA", "ciiu", "OBS_VALUE"]]
    hnd = hnd.rename(columns={"ciiu" : "ACTIVITY"})
    hnd = hnd.query(f"ACTIVITY in {muestra_actividades}")
    insumos_complejidad = pd.concat([insumos_complejidad, hnd])
    insumos_complejidad
    return anio_analisis, insumos_complejidad


@app.cell
def _(ecomplexity, insumos_complejidad, pl):
    # Calculate complexity
    trade_cols = {'time':"TIME_PERIOD", 'loc': "REF_AREA",  'prod': "ACTIVITY",  'val': "OBS_VALUE"}
    cdata = pl.from_pandas(ecomplexity(insumos_complejidad, trade_cols)).drop_nulls()
    cdata
    return (cdata,)


@app.cell
def _(cdata, pl):
    eci_paises = cdata.select("REF_AREA", "eci").unique().sort("eci", descending=True)
    eci_paises = eci_paises.with_columns(
        eci_rank = pl.col("eci").rank("ordinal", descending=True)
    )
    eci_paises
    return (eci_paises,)


@app.cell
def _(cdata, df, mapp_ciiu, pl):
    pci_industrias = cdata.select("ACTIVITY", "pci").join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).unique().drop_nulls().join(
        pl.from_pandas(df).select("ACTIVITY", "seccion").unique(),
        on = "ACTIVITY"
    ).sort(
        "pci", descending=True
    )
    pci_industrias
    return


@app.cell
def _(pl):
    ### Carga datos del atlas del ranking
    atlas = pl.read_csv("datos/atlas_datos/growth_proj_eci_rankings.csv").filter(
        pl.col("year") == 2019
    )
    atlas
    return (atlas,)


@app.cell
def _(anio_analisis, pd):
    ## Cargamos GDP
    gdp = pd.read_csv("https://raw.githubusercontent.com/milocortes/sisepuede_data/refs/heads/main/SocioEconomic/gdp_mmm_usd/input_to_sisepuede/projected/gdp_mmm_usd.csv").query(f"Year=={anio_analisis}")

    ## Cargamos poblacion rural
    pob_rural = pd.read_csv("https://raw.githubusercontent.com/milocortes/sisepuede_data/refs/heads/main/SocioEconomic/population_gnrl_rural/input_to_sisepuede/historical/population_gnrl_rural.csv").query(f"Year=={anio_analisis}")

    ## Cargamos poblacion urbana
    pob_urbana = pd.read_csv("https://raw.githubusercontent.com/milocortes/sisepuede_data/refs/heads/main/SocioEconomic/population_gnrl_urban/input_to_sisepuede/historical/population_gnrl_urban.csv").query(f"Year=={anio_analisis}")

    ## Reunimos población
    pob = pob_rural.merge(
        pob_urbana,
        on = ["Year","Nation","iso_code3"], 
    )
    pob["poblacion"] = pob["population_gnrl_rural"] + pob["population_gnrl_urban"]

    gdp = gdp.merge(
        pob, 
        on = ["Year","Nation","iso_code3"], 
    )
    gdp["gdp_percapita"] = (gdp["gdp_mmm_usd"]/gdp["poblacion"])*1_000_000
    gdp

    return (gdp,)


@app.cell
def _(alt, atlas, eci_paises):
    ### Gráficas de dispersion para Rankings
    eci_paises_atlas = eci_paises.join(
        atlas,
        left_on="REF_AREA", 
        right_on = "country_iso3_code"
    )


    alt.Chart(eci_paises_atlas).mark_point().encode(
        x='eci_rank',
        y='eci_rank_hs12',
        tooltip=["REF_AREA"]
    )

    return


@app.cell
def _(alt, eci_paises, gdp, pl):
    ### Gráficas de dispersion para GDP
    eci_paises_gdp = eci_paises.join(
        pl.from_pandas(gdp),
        left_on="REF_AREA", 
        right_on = "iso_code3"
    )

    alt.Chart(eci_paises_gdp).mark_point().encode(
        y=alt.Y('eci'),
        x=alt.X('gdp_percapita'),#.scale(type ="log"),
        tooltip=["REF_AREA"]
    )
    return


@app.cell
def _(np, pl):
    ## Calcula promedio ponderado de density, PCI y GO
    ### Define ponderadores
    product_selection_criteria = {
        "Low-hanging Fruit" : {"cog" : 0.25, "pci" : 0.15, "density" : 0.60},
        "Balanced Portfolio" : {"cog" : 0.35, "pci" : 0.15, "density" : 0.50},
        "Long Jumps" : {"cog" : 0.35, "pci" : 0.20, "density" : 0.45},
    }

    ### Mapeo portafolios - prefijos
    mapp_portafolios = {
        "lhf" : "Low-hanging Fruit", 
        "bp" : "Balanced Portfolio", 
        "lj" : "Long Jumps"
    }

    ### Creamos score como una media ponderada
    def calcula_score(portafolio : str, 
                      df_portafolios : pl.DataFrame
                     ) -> pl.DataFrame:

        df_portafolios_score = df_portafolios.with_columns(
            df_portafolios.select(
                pl.struct("density_norm", "pci", "cog").map_elements(
                    lambda s: np.average(
                        a = [s["density_norm"], s["pci"], s["cog"]],
                        weights = [
                            product_selection_criteria[mapp_portafolios[portafolio]]["density"],
                            product_selection_criteria[mapp_portafolios[portafolio]]["pci"], 
                            product_selection_criteria[mapp_portafolios[portafolio]]["cog"]
                        ]
                    ), 
                    return_dtype=pl.Float64
                ).alias(portafolio)
            )
        ).filter(
            (pl.col("mcp") == 0) & 
            (pl.col("REF_AREA") == "HND")
        ).select("REF_AREA", "ACTIVITY", "mcp", portafolio)

        return df_portafolios_score
    return (calcula_score,)


@app.cell
def _(pl):
    ## Calculamos densidad normalizada para cada conjunto de datos
    def calcula_density_z(df : pl.DataFrame
        ) -> pl.DataFrame:

        ### Normalizamos density
        df = df.with_columns(
            density_norm = (pl.col("density") - pl.col("density").mean())/pl.col("density").std()
        )

        return df 
    return (calcula_density_z,)


@app.cell
def _(calcula_density_z, cdata):
    cdata_norm = calcula_density_z(cdata)
    cdata_norm
    return (cdata_norm,)


@app.cell
def _(calcula_score, cdata_norm, df, mapp_ciiu, pl):
    calcula_score("lhf", cdata_norm).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).unique().drop_nulls().join(
        pl.from_pandas(df).select("ACTIVITY", "seccion").unique(),
        on = "ACTIVITY"
    ).sort(
        "lhf", descending=True
    )
    return


@app.cell
def _(alt, cdata):

    alt.Chart(cdata.filter(REF_AREA="HND")).mark_point().encode(
        x=alt.X('density'),
        y=alt.Y('pci'),#.scale(type ="log"),
        #tooltip=["REF_AREA"]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
