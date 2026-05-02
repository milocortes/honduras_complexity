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
    return alt, cs, ecomplexity, np, pd, pl, plt, proximity


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

    ### Concatenamos datos de paises no considerados en los datos de OCDE 
    insumos_complejidad = pd.concat([insumos_complejidad, hnd, slv, ecu])

    ### Los datos repetidos los sumamos
    #insumos_complejidad = insumos_complejidad.groupby(["TIME_PERIOD", "REF_AREA", "ACTIVITY"]).sum().reset_index()
    insumos_complejidad
    return anio_analisis, ciiu_na, insumos_complejidad, paises_muestra


@app.cell
def _(ecomplexity, insumos_complejidad, pl):
    # Calculate complexity
    trade_cols = {'time':"TIME_PERIOD", 'loc': "REF_AREA",  'prod': "ACTIVITY",  'val': "OBS_VALUE"}
    cdata = pl.from_pandas(ecomplexity(insumos_complejidad, trade_cols)).drop_nulls()
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
        x=alt.X('gdp_percapita').title("GDP Per Cápita"),
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
        x=alt.X('gdp_percapita:Q').title("GDP Percapita"),
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
                pl.struct("density", "pci", "cog").map_elements(
                    lambda s: np.average(
                        a = [s["density"], s["pci"], s["cog"]],
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
        x=alt.X('density').scale(zero=False).title("Densidad"),
        y=alt.Y('pci').title("PCI"),#.scale(type ="log"),
        shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("rca").scale(type ="log", scheme='redblue', domainMid=1.0).title("RCA"),
        size = alt.Size("rca").scale(type ="log"),
        tooltip=["nombre_actividad","rca"]
    ).properties(
        title=alt.TitleParams(
            "Diagrama Densidad-PCI",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='bottom-left', 
        titleFontSize=18,
        labelFontSize=16,

    )

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
        x=alt.X('density').scale(zero=False).title("Densidad"),
        y=alt.Y('pci').title("PCI"),#.scale(type ="log"),
        shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("rca").scale(type ="log", scheme='redblue', domainMid=1.0).title("RCA"),
        size = alt.Size("rca").scale(type ="log"),
        tooltip=["nombre_actividad","rca"]
    ).properties(
        title=alt.TitleParams(
            "Diagrama Densidad-PCI",
            subtitle="Alemania. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10,
        orient='bottom-right', 
        titleFontSize=18,
        labelFontSize=18,

    )
    return


@app.cell
def _(cdata, mapp_ciiu, pl):
    cdata.filter(
        (pl.col("REF_AREA")=="HND") , 
        (pl.col("mcp")==1)
    ).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).select("ACTIVITY", "nombre_actividad", "pci", "rca", "density").sort("density", descending=True)#.filter(ACTIVITY='2930')
    return


@app.cell
def _(cdata, mapp_ciiu, pl):
    cdata.filter(
        (pl.col("REF_AREA")=="HND") , 
        (pl.col("mcp")==0)
    ).join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).select("nombre_actividad", "pci", "rca", "density").plot.point(
        x = "density", 
        y = "rca"
    )
    return


@app.cell
def _(
    anio_analisis,
    ciiu_na,
    ciiu_seleccion_pedro,
    ciiu_transable,
    df,
    paises_muestra,
    pl,
):
    ## Test duplicados Datos OCDE
    insumos_complejidad_test_duplicados = pl.from_pandas(
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
                        #.query(f"SIZE_CLASS == '_T'")  
    )

    insumos_complejidad_test_duplicados.filter(
        pl.struct(["TIME_PERIOD", "REF_AREA", "ACTIVITY"]).is_duplicated()
    )#["SIZE_CLASS"].unique()
    return


@app.cell
def _(insumos_complejidad):
    insumos_complejidad.query("ACTIVITY=='2930'")
    return


@app.cell
def _(cdata):
    cdata
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
