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
    from ecomplexity import proximity
    import altair as alt
    from great_tables import GT, html
    import polars.selectors as cs
    return GT, alt, cs, ecomplexity, html, np, pd, pl, plt, proximity


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
def _(pd):
    ### Cargamos selección de industrias de Pedro
    ciiu_pedro = pd.read_csv(
                    "datos/recodificacion/seleccion_pedro.csv"
                    ).query("incluye==1")[
                        ["clase_codigo", "clase_titulo"]
                    ]
    ### Formato de clave ciiu 04d
    ciiu_pedro["clase_codigo"] = ciiu_pedro["clase_codigo"].apply(lambda x : f"{x:04}")

    ### Lista de Actividades CIIU a considerar
    ciiu_seleccion_pedro = ciiu_pedro["clase_codigo"].to_list()

    ciiu_pedro
    return (ciiu_seleccion_pedro,)


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

    ## Nos quedamos con las actividades transables
    df_actividades_transables = (
                            df
                            .query("TIME_PERIOD == 2019")
                            .query(f"ACTIVITY in {ciiu_transable}")
                            .query("OBS_VALUE>0")
                            .groupby(["REF_AREA", "ACTIVITY"])
                            .agg({"OBS_VALUE" : "sum"})
                            .reset_index()
                            .groupby("REF_AREA")
                            .agg({"ACTIVITY" : "count"})    
    )



    df_actividades_transables["ACTIVITY"] = df_actividades_transables["ACTIVITY"]/len(ciiu_transable)

    df_actividades_transables
    return ciiu_transable, df_actividades_transables, mapp_ciiu


@app.cell
def _(df_actividades_transables):
    df_actividades_transables.sort_values(by="ACTIVITY", ascending=False).plot.bar()
    return


@app.cell
def _(ciiu_seleccion_pedro, ciiu_transable, df, df_actividades_transables, pd):
    ### Haremos el análisis de complejidad modificando la muestra de paises de acuerdo al umbral de la razón de actividades que reportan empleo vs total de actividades transables

    ### Que actividades no tenemos datos para los paises?
    ciiu_na = [
        #"6420", "6430", "6491", "6492", "6499", "6511", "6512", "6611", "6612", "6619", "6621", "6629", 
        #"0161", "0162", "0163", "0164", "0210", "0220", "0230", "0240", "0311", "0312", "0610", "0620", 
        #"8510", "8522", "8530", "8541", "8549", "8550", "9000", "9102", "9103", "9200", "9311", "9312", "9319", "9321", "9329", "9412", 
        "6411", # Banca central
        "1910", # Fabricación de productos de hornos de coque
        #"0810" , # Extracción de piedra, arena y arcilla
        #"0899", # Explotación de otras minas y canteras n.c.p
        #"0910", # Actividades de apoyo para la extracción de petróleo y de gas natural
        #"0990", # Actividades de apoyo para otras actividades de explotación de minas y canteras
    ]

    ### Define umbral para la selección de la muestra de países
    umbral = 0.69
    anio_analisis = 2019

    ### Obten países por encima del umbral
    paises_muestra = df_actividades_transables.reset_index().query(f"ACTIVITY>{umbral}")["REF_AREA"].to_list()

    ### Filtra los datos que cumplen con el criterio
    insumos_complejidad = (
                    df
                        # Filtra periodo de análisis
                        .query(f"TIME_PERIOD == {anio_analisis}")
                        # Filta selección de Pedro
                        .query(f"ACTIVITY in {ciiu_seleccion_pedro}")
                        # Filtra por actividades transables
                        .query(f"ACTIVITY in {ciiu_transable}")
                        # Filtra países muestra
                        .query(f"REF_AREA in {paises_muestra}")
                        # Excluye industrias seleccionadas manualmente
                        .query(f"ACTIVITY not in {ciiu_na}")    
                        # Filtramos por el total de la actividad
                        .query(f"SIZE_CLASS == '_T'")      
    )




    insumos_complejidad = insumos_complejidad[["TIME_PERIOD", "REF_AREA", "ACTIVITY", "OBS_VALUE"]]

    muestra_actividades = list(insumos_complejidad["ACTIVITY"].unique())

    ### Cargamos Honduras
    hnd = pd.read_csv("datos/empleo_honduras/empleo_honduras_2019.csv")
    hnd["ACTIVITY"] = hnd["ACTIVITY"].apply(lambda x : f"{x:04}")
    hnd = hnd.query(f"ACTIVITY in {muestra_actividades}")

    ### Cargamos a El Salvador
    slv = pd.read_csv("datos/SLV/slv_ciiu_2019.csv")
    slv["ACTIVITY"] = slv["ACTIVITY"].astype(str)
    slv = slv.query(f"ACTIVITY in {muestra_actividades}")

    ### Cargamos a Ecuador
    ecu = pd.read_csv("datos/empleo_ecuador/empleo_ecuador_2019.csv")
    ecu["ACTIVITY"] = ecu["ACTIVITY"].astype(str)
    ecu = ecu.query(f"ACTIVITY in {muestra_actividades}")

    """
    ### Cargamos USA
    ## Carga datos USA
    ## Cargamos CBP 2023
    cbp = pd.read_csv("datos/usa/cbp/datos/cbp_msa/cbp23msa.txt")

    ## Nos quedamos sólo con los valores de las clases
    cbp = cbp[cbp["naics"].apply(lambda x : x[-1].isnumeric())].reset_index(drop=True)

    ## Nos quedamos con msa, naics y est
    cbp = cbp[["naics", "emp"]].groupby("naics").sum().reset_index()
    cbp["naics"] = cbp["naics"].astype(int)

    ## Cargamos ponderadores NAICS-CIIU 
    ponderadores = pd.read_csv("datos/recodificacion/ponderadores_ciiu_isic_concordance.csv")
    ponderadores["naics"] = ponderadores["naics"].astype(int)

    ## Reunimos ponderadores y ciiu 
    cbp = cbp.merge(
        ponderadores,   
        on = "naics"
    )

    ## Calculamos empleo
    cbp["empleo_ciiu"] = cbp["emp"] * cbp["weight"]

    ## Nos quedamos con las columnas minimas
    cbp = cbp[["empleo_ciiu", "ciiu"]].groupby("ciiu").sum().reset_index()
    cbp["OBS_VALUE"] = cbp["empleo_ciiu"].astype(int)
    cbp["TIME_PERIOD"] = 2019
    cbp["REF_AREA"] = "USA"
    cbp["ACTIVITY"] = cbp["ciiu"].apply(lambda x : f"{x:04}")
    cbp = cbp[["TIME_PERIOD", "REF_AREA", "ACTIVITY", "OBS_VALUE"]]
    """
    ### Concatenamos datos de paises no considerados en los datos de OCDE 
    insumos_complejidad = pd.concat([insumos_complejidad, hnd, slv, ecu])

    ### Los datos repetidos los sumamos
    #insumos_complejidad = insumos_complejidad.groupby(["TIME_PERIOD", "REF_AREA", "ACTIVITY"]).sum().reset_index()
    insumos_complejidad
    return anio_analisis, insumos_complejidad, paises_muestra


@app.cell
def _():


    return


@app.cell
def _(ecomplexity, insumos_complejidad, pl):
    # Calculate complexity
    trade_cols = {'time':"TIME_PERIOD", 'loc': "REF_AREA",  'prod': "ACTIVITY",  'val': "OBS_VALUE"}
    cdata = pl.from_pandas(ecomplexity(insumos_complejidad, trade_cols)).drop_nulls()
    cdata = cdata.with_columns(
        distance = 1 -pl.col("density")
    )
    cdata
    return cdata, trade_cols


@app.cell
def _(insumos_complejidad, proximity, trade_cols):
    ## Calcula matriz de proximidad
    prox_df = proximity(insumos_complejidad, trade_cols)
    prox_df = prox_df[["ACTIVITY_1", "ACTIVITY_2", "proximity"]]
    prox_df = prox_df.pivot(index = "ACTIVITY_1", columns = "ACTIVITY_2", values = "proximity")
    ## Guarda matriz de proximidad
    prox_df.to_csv("output/proximidades/proximidades_ocde_data.csv")
    prox_df
    return


@app.cell
def _(cdata, pl):
    ### Ranking de Países de acuerdo a ECI
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


    alt.Chart(eci_paises_atlas).mark_circle(
        opacity=0.99,
        stroke='black',
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=180                
    ).encode(
        x='eci_rank',
        y='eci_rank_hs12',
        tooltip=["REF_AREA"]
    )
    return (eci_paises_atlas,)


@app.cell
def _(alt, eci_paises_atlas):
    alt.Chart(eci_paises_atlas).mark_circle(
        opacity=0.99,
        stroke='black',
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=180     
    ).encode(
        x='eci',
        y='eci_hs12',
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

    gdp_vs_eci = alt.Chart(eci_paises_gdp).mark_circle(
        opacity=0.99,
        stroke='black',
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=180,     
    ).encode(
        x=alt.X('gdp_percapita').title("GDP Per Cápita [Miles de Dólares por persona]"),
        y=alt.Y('eci').title("ECI"),
        color = alt.ColorValue("red"),
        tooltip=["REF_AREA"]
    ).properties(
        title=alt.TitleParams(
            "GDP percapita vs ECI",
            subtitle="Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    )

    gdp_vs_eci_reg_line = gdp_vs_eci.transform_regression(
            'gdp_percapita', 'eci'
        ).mark_line(size = 5).transform_calculate(
                Fit='"LinReg"'
            ).encode(
                stroke='Fit:N', 
            )


    labels_iso_code3 = gdp_vs_eci.mark_text(
        align='left',
        baseline='middle',
        dx=10  # Offset text to the right
    ).encode(
        text='REF_AREA', # Column to use for label
        color = alt.ColorValue("black"),

    )

    gdp_vs_eci_chart = gdp_vs_eci + gdp_vs_eci_reg_line + labels_iso_code3 

    gdp_vs_eci_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return (eci_paises_gdp,)


@app.cell
def _(alt, atlas, cs, eci_paises_gdp, gdp, pl):
    ### Dibujemos los puntos de acuerdo al GDP y el ECI de empleo y del atlas
    ## Pegamos ECI del atlas
    eci_paises_gdp_empleo_atlas = eci_paises_gdp.select(
        "REF_AREA", "eci"
    ).rename(
        {"REF_AREA"  : "country_iso3_code", "eci" : "ECI Empleo OECD SBS"}
    ).join(
        atlas.select("country_iso3_code", "eci_hs12").rename({"eci_hs12" : "ECI Atlas"}), 
        on = "country_iso3_code"
    ).unpivot(
        cs.numeric(), index="country_iso3_code").join(
        pl.from_pandas(gdp).select("iso_code3", "gdp_percapita").rename({"iso_code3" : "country_iso3_code"}),
        on = "country_iso3_code"
    )

    # 1. Define the base chart with encodings
    base = alt.Chart(eci_paises_gdp_empleo_atlas).encode(
        x=alt.X('gdp_percapita:Q').title("GDP Percapita [Miles de Dólares por persona]"),
        y=alt.Y('value:Q').title("ECI"),
        color=alt.Color('variable:N').title("Datos")  # This provides the grouping color
    ).properties(
        title=alt.TitleParams(
            "GDP percapita vs ECI",
            subtitle="Datos de Empleo de OECD SBS 2019 y Atlas de Complejidad",
            subtitleColor="gray"
        )
    )


    # 2. Layer points and regression lines
    chart = base.mark_circle(
        opacity=0.99,
        stroke='black',
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=180,  
    ) + base.transform_regression(
        'gdp_percapita', 'value', groupby=['variable']
    ).mark_line()


    labels_iso_code3_eci = base.mark_text(
        align='left',
        baseline='middle',
        dx=10  # Offset text to the right
    ).encode(
        text='country_iso3_code', # Column to use for label
        color = alt.ColorValue("black"),

    )

    all_chart = chart + labels_iso_code3_eci
    all_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell
def _(ecomplexity, paises_muestra, pd, pl):
    ### Cacularemos las medidas de complejidad usando los datos del Atlas
    ### PARA LA MUESTRA DE 21 PAISES
    atlas_export = pd.read_parquet("datos/atlas_datos/hs92_country_product_year_4.parquet")
    atlas_export = atlas_export.query(f"country_iso3_code in {paises_muestra + ['HND', 'ECU', 'SLV']}")
    # Calculate complexity
    trade_cols_export = {'time':"year", 'loc': "country_iso3_code",  'prod': "product_hs92_code",  'val': "export_value"}
    cdata_export = pl.from_pandas(ecomplexity(atlas_export, trade_cols_export)).drop_nulls()
    cdata_export = cdata_export.select("country_iso3_code", "eci").unique().rename({"eci" : "eci_atlas"})
    cdata_export
    return (cdata_export,)


@app.cell
def _(alt, cdata, cdata_export, pl):
    ### Reunimos los datos
    cdata_atlas_estimado = cdata.select(
        "REF_AREA", "eci"
    ).unique().rename(
        {"REF_AREA" : "country_iso3_code"}
    ).with_columns(
        eci_rank = pl.col("eci").rank("ordinal", descending=True)
    ).join(
        cdata_export.with_columns(
        eci_rank_atlas = pl.col("eci_atlas").rank("ordinal", descending=True)
    ), 
        on = "country_iso3_code"
    )

    ### Dibujamos la figura
    base_rankings = alt.Chart(cdata_atlas_estimado).mark_circle(
        opacity=0.99,
        stroke='black',
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=220     
    ).encode(
        x=alt.X('eci_rank').title("Ranking ECI OECD SBS"),
        y=alt.Y('eci_rank_atlas').title("Ranking ECI Atlas"),
        color = alt.ColorValue("red"),
        tooltip=["country_iso3_code"]
    ).properties(
        title=alt.TitleParams(
            "Comparación de Rankings de Paises",
            subtitle="Datos de Empleo de OECD SBS y Atlas de Complejidad 2019",
            subtitleColor="gray"
        )
    )

    base_rankings_reg_line = base_rankings.transform_regression(
            'eci_rank', 'eci_rank_atlas'
        ).mark_line(size = 5).transform_calculate(
                Fit='"LinReg"'
            ).encode(
                stroke='Fit:N', 
            )


    labels_iso_code3_rankings = base_rankings.mark_text(
        align='left',
        baseline='middle',
        dx=10  # Offset text to the right
    ).encode(
        text='country_iso3_code', # Column to use for label
        color = alt.ColorValue("black"),

    )

    line_red = alt.Chart().mark_rule(color='red', size = 5).transform_calculate(
                Fit='"Identidad f(x) = x"'
            ).encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.value('height'),
        y2=alt.value(0), 
        stroke='Fit:N', 
    )

    gdp_vs_eci_chart_rankings = base_rankings + line_red + labels_iso_code3_rankings 

    gdp_vs_eci_chart_rankings.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return (cdata_atlas_estimado,)


@app.cell
def _(cdata_atlas_estimado):
    cdata_atlas_estimado
    return


@app.cell
def _(np, pl):
    ## Calcula promedio ponderado de density, PCI y GO
    ### Define ponderadores
    product_selection_criteria = {
        "Low-hanging Fruit" : {"cog" : 0.05, "pci" : 0.05, "density" : 0.9},
        "Balanced Portfolio" : {"cog" : 0.1, "pci" : 0.1, "density" : 0.8},
        "Long Jumps" : {"cog" : 0.40, "pci" : 0.40, "density" : 0.2},
    }

    product_selection_criteria = {
        "Low-hanging Fruit" : {"cog" : 0.15, "pci" : 0.05, "density" : 0.8},
        "Balanced Portfolio" : {"cog" : 0.25, "pci" : 0.25, "density" : 0.5},
        "Long Jumps" : {"cog" : 0.45, "pci" : 0.35, "density" : 0.2},
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

        df_portafolios = df_portafolios.filter(
            (pl.col("mcp") == 0) & 
            (pl.col("REF_AREA") == "HND")
        )
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
        ).select("REF_AREA", "ACTIVITY", "mcp", portafolio)

        return df_portafolios_score
    return calcula_score, mapp_portafolios, product_selection_criteria


@app.cell
def _(pl):
    ## Calculamos densidad normalizada para cada conjunto de datos
    def calcula_density_z(df : pl.DataFrame
        ) -> pl.DataFrame:

        ### Normalizamos density
        df = df.with_columns(
            density_norm = (pl.col("density") - pl.col("density").mean())/pl.col("density").std(), 
            distance = 1 - pl.col("density")
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
    portafolio_cat = "bp" 
    calcula_score(portafolio_cat, cdata_norm).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).unique().drop_nulls().join(
        pl.from_pandas(df).select("ACTIVITY", "seccion").unique(),
        on = "ACTIVITY"
    ).sort(
        portafolio_cat, descending=True
    ).select(
            portafolio_cat,"nombre_actividad"
        ).with_columns(
            pl.col(portafolio_cat).rank("ordinal", descending=True).alias("rank")
        ).drop(portafolio_cat).head(10).select("rank", "nombre_actividad").rename({
            "nombre_actividad" : f"nombre_actividad_{portafolio_cat}", 
            "rank" : f"rank_{portafolio_cat}"
        })
    return


@app.cell
def _(calcula_score, cdata_norm, df, mapp_ciiu, pl):
    def obten_ranking(portafolio_cat : str, 
                      datos : pl.DataFrame, ) -> pl.DataFrame:

        return calcula_score(portafolio_cat, cdata_norm).join(
            mapp_ciiu,
            left_on="ACTIVITY", 
            right_on="codigo"
        ).unique().drop_nulls().join(
            pl.from_pandas(df).select("ACTIVITY", "seccion").unique(),
            on = "ACTIVITY"
        ).sort(
            portafolio_cat, descending=True
        ).select(
                portafolio_cat, "ACTIVITY", "nombre_actividad"
            ).with_columns(
                pl.col(portafolio_cat).rank("ordinal", descending=True).alias("rank")
            ).drop(portafolio_cat).head(20).select("rank", "ACTIVITY", "nombre_actividad").rename({
                "nombre_actividad" : f"nombre_actividad_{portafolio_cat}", 
                "rank" : f"rank_{portafolio_cat}", 
                "ACTIVITY" : f"ciiu_{portafolio_cat}"
        })
    return (obten_ranking,)


@app.cell
def _(cdata_norm, mapp_portafolios, obten_ranking, pl):
    ### Creamos portafolios 
    portafolios_categorias = [obten_ranking(portafolio,  cdata_norm) for portafolio in mapp_portafolios]
    portafolios_categorias = pl.concat(portafolios_categorias,  how = "horizontal")
    portafolios_categorias
    return (portafolios_categorias,)


@app.cell
def _(GT, html, portafolios_categorias):
    (
        GT(portafolios_categorias)
        .tab_header(
            title="Selección de Portafolios de Actividades",
            subtitle="Población Ocupada"
        )
        .tab_spanner(
            label="Low-hanging Fruit",
            columns=["rank_lhf", "nombre_actividad_lhf"]
        )
        .tab_spanner(
            label="Balanced Portfolio",
            columns=["rank_bp", "nombre_actividad_bp"]
        )
        .tab_spanner(
            label="Long Jumps",
            columns=["rank_lj", "nombre_actividad_lj"]
        )
        .cols_move_to_start(columns=["rank_lhf", "nombre_actividad_lhf", "rank_bp", "nombre_actividad_bp", "rank_lj", "nombre_actividad_lj"])
        .cols_label(
            rank_lhf = html("Ranking"),
            nombre_actividad_lhf = html("Actividad"),
            rank_bp = html("Ranking"),
            nombre_actividad_bp = html("Actividad"), 
            rank_lj = html("Ranking"),
            nombre_actividad_lj = html("Actividad")
        ).cols_hide(columns= ["ciiu_lhf", "ciiu_bp", "ciiu_lj"] )
    )
    return


@app.cell
def _(alt, cdata, mapp_ciiu, pl):

    alt.Chart(cdata.filter(
        (pl.col("REF_AREA")=="HND") & 
        (pl.col("rca")>0)

    ).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    )
             ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
             ).encode(
        x=alt.X('distance').scale(zero=False).title("Distancia"),
        y=alt.Y('pci').title("PCI"),#.scale(type ="log"),
        shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("rca").scale(type ="log", scheme='redblue', domainMid=1.0).title("RCA"),
        size = alt.Size("rca").scale(type ="log"),
        tooltip=["nombre_actividad","rca"]
    ).properties(
        title=alt.TitleParams(
            "Diagrama Distancia-PCI",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='top-left', 
        titleFontSize=18,
        labelFontSize=16,

    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagrama Relacionamiento-Complejidad (Intensivo)
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(alt, cdata, cdata_hnd, ciiu_pedro_2, mapp_ciiu, pl):
    cdata_intensivo = cdata.filter(
        (pl.col("REF_AREA")=="HND") & 
        (pl.col("rca")>0) & 
        (pl.col("mcp")==1)
    )

    # Create a horizontal line at y=50
    hline = alt.Chart().mark_rule(color='red').encode(
        y=alt.datum(-1.14) # ECI Honduras
    )

    ### Modificamos claves para la Sección corresponda las Industrias

    ciiu_textil = ciiu_pedro_2.select("clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo", "division_titulo").to_pandas()
    ciiu_textil.loc[ciiu_textil["division_titulo"]=='Fabricación de prendas de vestir', "seccion_titulo"]  = 'Industria Textil'
    ciiu_textil.loc[ciiu_textil["division_titulo"]=='Fabricación de productos textiles', "seccion_titulo"]  = 'Industria Textil'
    ciiu_textil = pl.from_pandas(ciiu_textil)

    plot_intensivo = alt.Chart(
        cdata_intensivo.join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).join(
        ciiu_textil,
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )
    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
             ).encode(
        x=alt.X('distance').scale(zero=False).title("Distancia"),
        y=alt.Y('pci').title("PCI").scale(domain=(cdata_hnd["pci"].min()-1.5,cdata_hnd["pci"].max()+0.4)),#.scale(type ="log"),
        color = alt.Color("seccion_titulo").title("Sección"),
        size = alt.Size("OBS_VALUE").scale(type ="log").title("Empleo"),
        tooltip=["nombre_actividad", "division_titulo", "OBS_VALUE"]
    )
    plot_intensivo = plot_intensivo + hline

    plot_intensivo.properties(
        title=alt.TitleParams(
            "Diagrama Distancia-PCI (Intensivo)",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    )

    return cdata_intensivo, ciiu_textil, hline, plot_intensivo


@app.cell
def _(mo, plot_intensivo):
    # Make it reactive ⚡
    plot_intensivo_mo = mo.ui.altair_chart(plot_intensivo )# + text)
    plot_intensivo_mo
    return


@app.cell
def _(alt, cdata_hnd, cdata_intensivo, ciiu_textil, hline, mapp_ciiu):
    #### Plot intensivo Logaritmo Empleo vs PCI

    plot_intensivo_empleo = alt.Chart(
        cdata_intensivo.join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).join(
        ciiu_textil,
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )
    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
             ).encode(
        x=alt.X('OBS_VALUE').scale(type ="log", zero=False).title("Empleo"),
        y=alt.Y('pci').title("PCI").scale(domain=(cdata_hnd["pci"].min()-1.5,cdata_hnd["pci"].max()+0.4)),#.scale(type ="log"),
        color = alt.Color("seccion_titulo").title("Sección"),
        size = alt.Size("rca").scale(type ="log").title("RCA"),
        tooltip=["nombre_actividad", "division_titulo", "OBS_VALUE"]
    )
    plot_intensivo_empleo = plot_intensivo_empleo + hline

    return (plot_intensivo_empleo,)


@app.cell
def _(alt, mo, plot_intensivo_empleo):
    # Make it reactive ⚡
    plot_intensivo_empleo_mo = mo.ui.altair_chart(plot_intensivo_empleo.properties(
        title=alt.TitleParams(
            "Diagrama Distancia-PCI (Intensivo)",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ))# + text)
    plot_intensivo_empleo_mo
    return


@app.cell
def _():
    """
    .configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='top-left', 
        titleFontSize=12,
        labelFontSize=10,

    ) 
    """    
    return


@app.cell
def _(alt, cdata, ciiu_pedro_2, mapp_ciiu, pl):
    cdata_intensivo_complejo = cdata.filter(
        (pl.col("REF_AREA")=="HND") & 
        (pl.col("rca")>0) & 
        (pl.col("mcp")==1) &
        (pl.col("pci") >= -1.41)
    )

    plot_intensivo_complejo = alt.Chart(
        cdata_intensivo_complejo.join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).join(
        ciiu_pedro_2.select("clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo"),
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )
    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
             ).encode(
        x=alt.X('distance').scale(zero=False).title("Distancia"),
        y=alt.Y('pci').title("PCI"),
        shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("seccion_titulo").title("Sección"),
        size = alt.Size("OBS_VALUE").scale(type ="log").title("Empleo"),
        tooltip=["nombre_actividad","OBS_VALUE"]
    ).properties(
        title=alt.TitleParams(
            "Diagrama Distancia-PCI (Intensivo)",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='top-left', 
        titleFontSize=12,
        labelFontSize=10,

    ) 
    plot_intensivo_complejo
    return


@app.cell
def _(alt, cdata, mapp_ciiu, pl):
    alt.Chart(cdata.filter(
        (pl.col("REF_AREA")=="DEU") & 
        (pl.col("rca")>0)

    ).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    )
             ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
             ).encode(
        x=alt.X('distance').scale(zero=False).title("Distancia"),
        y=alt.Y('pci').title("PCI"),#.scale(type ="log"),
        shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("rca").scale(type ="log", scheme='redblue', domainMid=1.0).title("RCA"),
        size = alt.Size("rca").scale(type ="log"),
        tooltip=["nombre_actividad","rca"]
    ).properties(
        title=alt.TitleParams(
            "Diagrama Distancia-PCI",
            subtitle="Alemania. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='bottom-left', 
        titleFontSize=18,
        labelFontSize=18,

    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grafica Oportunidades de Diversificación
    """)
    return


@app.cell
def _(pd, pl):
    ### Cargamos selección de industrias de Pedro
    ciiu_pedro_2 = pd.read_csv(
                    "datos/recodificacion/seleccion_pedro.csv"
                    ).query("incluye==1")#[["clase_codigo", "seccion_codigo", "seccion_titulo"]]
    ### Formato de clave ciiu 04d
    ciiu_pedro_2["clase_codigo"] = ciiu_pedro_2["clase_codigo"].apply(lambda x : f"{x:04}")
    ciiu_pedro_2 = pl.from_pandas(ciiu_pedro_2)#.select("seccion_codigo", "seccion_titulo").unique()
    #cdata.select("ACTIVITY").unique().join(
    #   pl.from_pandas(ciiu_pedro_2), 
    #    left_on = "ACTIVITY", 
    #    right_on="clase_codigo"
    #).group_by("seccion_titulo").len()
    ciiu_pedro_2
    return (ciiu_pedro_2,)


@app.cell
def _(cdata_norm, ciiu_pedro_2, mapp_portafolios, obten_ranking, pl):
    def agrega_col(df, columna) : 
        return df.with_columns(
            portafolio = pl.lit(columna)
        )

    portafolios = pl.concat(
        [
            agrega_col(
                obten_ranking(portafolio,  cdata_norm).rename(
                {
                    f"rank_{portafolio}" : "ranking", 
                    f"ciiu_{portafolio}" : "clase_codigo", 
                    f"nombre_actividad_{portafolio}" : "clase_titulo", 
                }
            ), 
                portafolio
            )
        for portafolio in mapp_portafolios]  
    )

    ## Agrega Sección Codigo
    portafolios = portafolios.join(
        ciiu_pedro_2.select("clase_codigo", "seccion_codigo", "seccion_titulo"),
        on = "clase_codigo"
    )
    portafolios
    return (portafolios,)


@app.cell
def _(portafolios):
    ## Guardamos datos de portafolios
    portafolios.write_csv("portafolios/portafolios.csv")
    return


@app.cell
def _(cdata_norm, ciiu_pedro_2, pl):
    ### Preparamos df para visualizacion
    cdata_hnd = cdata_norm.filter(
        (pl.col("REF_AREA")=='HND') &
        (pl.col("mcp")==0)
    ).join(
        ciiu_pedro_2.select("clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo"),
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )

    cdata_hnd
    return (cdata_hnd,)


@app.cell
def _(cdata_hnd):
    ### Guardamos datos de complejidad de visualización
    cdata_hnd.write_csv("portafolios/cdata_hnd.csv")
    return


@app.cell
def _(mo):
    ### Complexity metrics
    complexity_metric = {
        "Complexity" : "pci",
        "Opportunity Gain" : "cog"
    }

    ### Definimos Dropdowns

    #### Selection criteria dropdown
    drop_product_selection_criteria = mo.ui.dropdown(
        options=["Low-hanging Fruit", "Balanced Portfolio" , "Long Jumps"],
        value="Low-hanging Fruit",
        label="Choose Product Selection Criteria",
        searchable=True,
    )

    #### Complexity metric dropdown
    drop_complexity_metric =  mo.ui.dropdown(
        options=["Complexity", "Opportunity Gain"],
        value="Complexity",
        label="Choose Complexity Metric",
        searchable=True,
    )
    return (
        complexity_metric,
        drop_complexity_metric,
        drop_product_selection_criteria,
    )


@app.cell
def _(
    cdata_hnd,
    drop_product_selection_criteria,
    mapp_portafolios,
    pl,
    portafolios,
):
    ### Subset productos priorizados y no priorizados
    ### Mapeo portafolios - prefijos
    mapp_portafolios_inv = {v:k for k,v in mapp_portafolios.items()}

    ### Clases a priorizar
    clases_ciiu_priorizar = portafolios.filter(portafolio=mapp_portafolios_inv[drop_product_selection_criteria.value])["clase_codigo"].to_numpy()

    #### Priorizados
    points_prioriza = cdata_hnd.filter(
                (pl.col("ACTIVITY").is_in(clases_ciiu_priorizar)) 
    )

    #### No Priorizados
    points_resto = cdata_hnd.filter(
                ~pl.col("ACTIVITY").is_in(clases_ciiu_priorizar)   
    )
    return points_prioriza, points_resto


@app.cell
def _(
    alt,
    cdata_hnd,
    complexity_metric,
    drop_complexity_metric,
    drop_product_selection_criteria,
    points_prioriza,
    points_resto,
    product_selection_criteria,
):
    # Create an Altair chart
    selection_weigths = ", ".join([f"{i} = {j}" for i,j in product_selection_criteria[drop_product_selection_criteria.value].items()])
    selection_weigths = "Weights : " + selection_weigths



    ### Priorized product plots
    relateness_plot_prioriza = alt.Chart(points_prioriza).mark_point(filled=True, size=230, stroke = "black").encode(
        alt.X('distance', title="Distancia").scale(domain=(cdata_hnd["distance"].min()-0.02,cdata_hnd["distance"].max() + 0.02)), # Encoding along the x-axis
        alt.Y(complexity_metric[drop_complexity_metric.value], title=drop_complexity_metric.value).scale(domain=(-4,8)), # Encoding along the y-axis
        color='seccion_titulo', # Category encoding by color
        tooltip=['clase_titulo', 'seccion_titulo', 'distance', complexity_metric[drop_complexity_metric.value]]
    ).properties(
        title = [f"Relatedness-complexity diagram - HND - Year : 2019", 
                 f"{drop_product_selection_criteria.value}", 
                selection_weigths],

    )
    ### Priorized product plots (Para el texto en negro)
    relateness_plot_prioriza_negro = alt.Chart(points_prioriza).mark_point().encode(
        alt.X('distance', title="Distancia").scale(domain=(cdata_hnd["distance"].min()-0.02,cdata_hnd["distance"].max() + 0.02)), # Encoding along the x-axis
        alt.Y(complexity_metric[drop_complexity_metric.value], title=drop_complexity_metric.value), # Encoding along the y-axis
        #color='Sector', # Category encoding by color
        tooltip=['clase_titulo', 'seccion_titulo', 'distance', complexity_metric[drop_complexity_metric.value]]
    ).properties(
        title = [f"Relatedness-complexity diagram - HND - Year : 2019", 
                 f"{drop_product_selection_criteria.value}", 
                selection_weigths],

    )

    # 3. Create a separate text layer
    text = relateness_plot_prioriza_negro.mark_text(
        align='left',
        baseline='middle',
        fontSize = 7,
        fontStyle = "bold",
        #fontWeight = "bold",
        dx=7 # Offset the text slightly to the right of the point
    ).encode(
        text='clase_titulo:N' # Nominal data type for labels
    )


    ### Unpriorized product plots
    relateness_plot = alt.Chart(points_resto).mark_point(filled=True, size=230, opacity=0.3).encode(
        alt.X('distance', title="Distancia").scale(domain=(cdata_hnd["distance"].min(),cdata_hnd["distance"].max())), # Encoding along the x-axis
        alt.Y(complexity_metric[drop_complexity_metric.value], title=drop_complexity_metric.value), # Encoding along the y-axis
        color=alt.Color('seccion_titulo', title = "Seccion"), # Category encoding by color
        tooltip=['clase_titulo', 'seccion_titulo', 'distance', complexity_metric[drop_complexity_metric.value]]
    )
    return relateness_plot, relateness_plot_prioriza


@app.cell
def _(mo, relateness_plot, relateness_plot_prioriza):
    # Make it reactive ⚡
    relateness_plot_prioriza_mo = mo.ui.altair_chart(relateness_plot_prioriza )# + text)
    relateness_plot_mo = mo.ui.altair_chart(relateness_plot)
    return relateness_plot_mo, relateness_plot_prioriza_mo


@app.cell
def _(
    complexity_metric,
    drop_complexity_metric,
    drop_product_selection_criteria,
    mo,
    relateness_plot_mo,
    relateness_plot_prioriza_mo,
):
    # In a new cell, display the chart and its data filtered by the selection

    if complexity_metric[drop_complexity_metric.value] == "ICI_UE":
        stack_plots = [
                    drop_product_selection_criteria,drop_complexity_metric,
                    relateness_plot_prioriza_mo + relateness_plot_mo ,
                    #points_prioriza.select("Sector", "Subsector", "rama_id", "Industria", "score").sort(by="score", descending=False)
            ]
    else:
        stack_plots = [
                    drop_product_selection_criteria,drop_complexity_metric,
                    relateness_plot_prioriza_mo + relateness_plot_mo,
                    #points_prioriza.select("Sector", "Subsector", "rama_id", "Industria", "score").sort(by="score", descending=False)
            ]

    mo.vstack(stack_plots)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tablas Resumen Diversificación
    """)
    return


@app.cell
def _():
    #portafolios.select("portafolio", "clase_titulo", "seccion_titulo").to_pandas().pivot(index= ["seccion_titulo", "clase_titulo"], columns='portafolio') 
    return


@app.cell
def _(pl):
    ### Datos Great Tables

    pre_tax_col = "gini_market__age_total"
    post_tax_col = "gini_disposable__age_total"

    # Read the data
    df_gt = pl.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-08-05/income_inequality_raw.csv",
        schema={
            "Entity": pl.String,
            "Code": pl.String,
            "Year": pl.Int64,
            post_tax_col: pl.Float64,
            pre_tax_col: pl.Float64,
            "population_historical": pl.Int64,
            "owid_region": pl.String,
        },
        null_values=["NA", ""],
    )

    # Propogate the region field to all rows of that country
    df_gt = (
        df_gt.sort("Entity")
        .group_by("Entity", maintain_order=True)
        .agg(
            [
                pl.col("Code"),
                pl.col("Year"),
                pl.col(post_tax_col),
                pl.col(pre_tax_col),
                pl.col("population_historical"),
                # Most important action happens here
                pl.col("owid_region").fill_null(strategy="backward"),
            ]
        )
        .explode(
            [
                "Code",
                "Year",
                post_tax_col,
                pre_tax_col,
                "population_historical",
                "owid_region",
            ]
        )
    )

    # Drop rows where there is a null in either pre-tax or post-tax cols
    df_gt = df_gt.drop_nulls(
        subset=(
            pl.col(post_tax_col),
            pl.col(pre_tax_col),
        )
    )

    # Compute the percent reduction in gini coefficient.
    df_gt = df_gt.with_columns(
        ((pl.col(pre_tax_col) - pl.col(post_tax_col)) / pl.col(pre_tax_col) * 100)
        .round(2)
        .alias("gini_pct_change")
    )

    # Calculate 5-year benchmark (mean) of percent change for each country
    df_gt = df_gt.with_columns(
        pl.col("gini_pct_change")
        .rolling_mean(window_size=5)
        .over(pl.col("Entity"))
        .alias("gini_pct_benchmark_5yr")
    )

    # Select rows with large population in the year 2020, sorted by coefficient post-tax
    df_gt = (
        # Choose a smaller pop to include more countries
        df_gt.filter(pl.col("population_historical").gt(40000000))
        .filter(pl.col("Year").eq(2020))
        .sort(by=pl.col(post_tax_col))
    )


    # Scale population
    df_gt = df_gt.with_columns((pl.col("population_historical").log10()).alias("pop_log"))
    pop_min = df_gt["pop_log"].min() / 1
    pop_max = df_gt["pop_log"].max()

    # Set up gt-extras icons, scaling population to 1-10 range
    df_gt = df_gt.with_columns(
        ((pl.col("pop_log") - pop_min) / (pop_max - pop_min) * 10 + 1)
        .round(0)
        .cast(pl.Int64)
        .alias("pop_icons")
    )

    # Format original population value with commas
    df_gt = df_gt.with_columns(
        pl.col("population_historical").map_elements(
            lambda x: f"{int(x):,}" if x is not None else None, return_dtype=pl.String
        )
    )
    df_gt
    return df_gt, post_tax_col, pre_tax_col


@app.cell
def _(GT, df_gt, html, pl, post_tax_col, pre_tax_col):
    import gt_extras as gte

    # Apply gte.fa_icon_repeat to each entry in the pop_icons column
    df_with_icons = df_gt.with_columns(
        pl.col("pop_icons").map_elements(
            lambda x: gte.fa_icon_repeat(name="person", repeats=int(x)),
            return_dtype=pl.String,
        )
    )

    # Generate the table, before gt-extras add-ons
    gt = (
        GT(df_with_icons, rowname_col="Entity", groupname_col="owid_region")
        .tab_header(
            "Income Inequality Before and After Taxes in 2020",
            "As measured by the Gini coefficient, where 0 is best and 1 is worst",
        )
        .cols_move("pop_icons", after=pre_tax_col)
        .cols_align("left")
        .cols_hide(["Year", "pop_log", "population_historical"])
        .fmt_flag("Code")
        .cols_label(
            {
                "Code": "",
                "gini_pct_change": "Improvement Post Taxes",
                "pop_icons": "Population",
            }
        )
        .tab_source_note(
            html(
                """
                <div>
                <strong>Source:</strong> Data from <a href="https://github.com/rfordatascience/tidytuesday">#TidyTuesday</a> (2025-08-05).<br>
                    <div>
                    <strong>Dumbbell plot:</strong>
                    <span style="color:#106ea0;">Blue:</span> post-tax Gini coefficient
                    <span style="color:#e0b165;">Gold:</span> pre-tax Gini coefficient
                    <br>
                    </div>
                <strong>Bullet plot:</strong> Percent reduction in Gini after taxes for each country, compared to its 5-year average benchmark.
                </div>
                """
            )
        )
    )

    # Apply the gt-extras functions via pipe
    (
        gt.pipe(
            gte.gt_plt_dumbbell,
            col1=pre_tax_col,
            col2=post_tax_col,
            col1_color="#e0b165",
            col2_color="#106ea0",
            dot_border_color="transparent",
            num_decimals=2,
            width=240,
            label="Pre-tax to Post-tax Coefficient",
        )
        .pipe(
            gte.gt_plt_bullet,
            "gini_pct_change",
            "gini_pct_benchmark_5yr",
            fill="#963d4c",
            target_color="#3D3D3D",
            bar_height=15,
            width=200,
        )
        .pipe(
            gte.gt_merge_stack,
            col1="pop_icons",
            col2="population_historical",
        )
        .pipe(gte.gt_theme_guardian)
    )
    return (df_with_icons,)


@app.cell
def _(df_with_icons):
    df_with_icons
    return


if __name__ == "__main__":
    app.run()
