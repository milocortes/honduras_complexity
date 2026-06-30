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
    return pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Métricas de Viabilidad y Atractivo

    **Attractiveness**:
    - Capacidad para movilizar FDI (world and region) :rocket:
    - ⁠Industry growth worldwide (past five years) :rocket:
    - ⁠Industry growth worldwide (past five years-Atlas export growth) :rocket:
    - ⁠Possibility to substitute US imports from Asia (China) :rocket:
    - ⁠Capacity to create employment among specific groups (women, youth, low-skill) :rocket:

    **Viability**:
    - Strength in countries like Honduras (RCA in peer group)
    - ⁠⁠Availability of inputs (doble razor, let us talk) :rocket:
    - Reliance on a constraint or potential constraint (energy, security) :rocket:
    - Reliance on a constraint or potential constraint (electricity-SCIAN México) :rocket:
    """)
    return


@app.cell
def _():
    ## Cargamos datos
    produccion = {
        "VAFC" : "Value added at factor costs",
        "INGS" : "Total Purchases of goods and services",
        "INEN" : "Purchases of energy products",
        "PROD" : "Production",
        "INGS" : "Total Purchases of goods and services",
    }

    empleo = {
        "EMPN" : "Total employment (persons employed)",
        "EMPF" : "Female employees",
        "INEN" : "Purchases of energy products",
        "VAPE" : "Labour productivity",
        "EMPE" : "Employees",
    }
    return


@app.cell
def _(pd, pl):
    ### Define consulta tipo lazy para el acceso a los datos
    def obten_datos(dataset : str) -> pd.DataFrame: 
        #q = pl.scan_csv('datos/viabilidad_atractivo/oecd_sbp_produccion_gasto_insumos.csv').select("ACTIVITY", "OBS_VALUE", "MEASURE", "TIME_PERIOD").group_by("ACTIVITY", "TIME_PERIOD", "MEASURE").sum()
        q = pl.scan_parquet(f'datos/viabilidad_atractivo/{dataset}.parquet').select("ACTIVITY", "OBS_VALUE", "MEASURE", "TIME_PERIOD").group_by("ACTIVITY", "TIME_PERIOD", "MEASURE").sum()
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
        df["ACTIVITY"] = df["ACTIVITY"].apply(lambda x : x[1:])

        df = df.pivot(index=['TIME_PERIOD', 'ACTIVITY'], columns='MEASURE', values='OBS_VALUE')

        return df.reset_index()
    return (obten_datos,)


@app.cell
def _(obten_datos):
    ### Cargamos datos de produccion
    df_produccion = obten_datos("oecd_sbp_produccion_gasto_insumos")
    df_produccion
    return (df_produccion,)


@app.cell
def _(obten_datos):
    ### Cargamos datos de empleo
    df_empleo = obten_datos("oecd_sbp_empleo_energia")
    df_empleo
    return (df_empleo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attractiveness
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Capacidad para movilizar FDI (world and region)
    """)
    return


@app.cell
def _(pd):
    ## Carga FDI
    fdi = pd.read_parquet("datos/viabilidad_atractivo/fDi_Subsectors/fdi_subsectores_iso_code3.parquet")

    ## Carga regiones 
    regiones = pd.read_csv("datos/viabilidad_atractivo/fDi_Subsectors/paises_iso_code.csv")

    ## Carga crosswalk de los subsectores fdi - CIIU
    fdi_ciiu = pd.read_csv("datos/viabilidad_atractivo/fDi_Subsectors/correspondencia_fdi_ciiu_rev4.csv")
    fdi_ciiu["CIIU"] = fdi_ciiu["CIIU"].apply(lambda x  : f"{x:04d}")

    ## Agregamos regiones del mundo a datos de fdi
    fdi  = fdi.merge(regiones[["iso_alpha_3", "un_sub_region"]], left_on="iso_code3", right_on="iso_alpha_3", how="left")
    fdi["un_sub_region"] = fdi["un_sub_region"].fillna("Western Asia")

    ## Cambiamos a entero el año de inicio del proyecto
    fdi["Project date"] = fdi["Project date"].apply(lambda x : x.split("/")[-1]).astype(int)

    ## Agregamos la correspondencia de actividad CIIU y subsector fdi
    fdi = fdi.merge(fdi_ciiu[["Nombre fDi (Subsector).1", "CIIU"]], left_on="Sub-sector", right_on="Nombre fDi (Subsector).1", how="left")

    ## Filtramos dataset a la región de LAC
    fdi_lac = fdi.query("un_sub_region == 'Latin America and the Caribbean'")
    return fdi, fdi_lac


@app.cell
def _(fdi):
    fdi
    return


@app.cell
def _():
    #pl.from_pandas(fdi[["CIIU", "Project date", "Capital investment"]]).sort(
    #    ["Project date", "CIIU"], maintain_order=True
    #).group_by("Project date", "CIIU",maintain_order=True).sum().select(
    #    pl.col("Project date", "CIIU", "Capital investment"),
    #    pl.col("Capital investment").cum_sum().over("CIIU").alias("investment_cum_sum"),
    #)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Agrupamos por actividad CIIU para tener el monto acumulado de inversión en capital y creacion de empleo entre 2019 y 2024
    """)
    return


@app.cell
def _(fdi, fdi_lac):
    ## Agrupamos por actividad CIIU para tener el monto acumulado de inversión en capital y creacion de empleo entre 2019 y 2024
    fdi_capital_investment = fdi[["CIIU", "Capital investment", "Jobs created"]].groupby("CIIU").sum().reset_index()
    fdi_lac_capital_investment = fdi_lac[["CIIU", "Capital investment", "Jobs created"]].groupby("CIIU").sum().reset_index()
    return fdi_capital_investment, fdi_lac_capital_investment


@app.cell
def _(fdi_capital_investment):
    fdi_capital_investment
    return


@app.cell
def _(fdi_lac_capital_investment):
    fdi_lac_capital_investment
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Calculamos la tasa de crecimiento compuesta de la inversión entre 2019 y 2024 para cada industria
    """)
    return


@app.cell
def _(fdi, pl):
    ## Tasa de crecimiento compuesta para inversión de industrias en todo el mundo
    fdi_cagr_investment = pl.from_pandas(fdi[["CIIU", "Project date", "Capital investment"]]).sort(
        ["Project date", "CIIU"], maintain_order=True
    ).group_by("Project date", "CIIU",maintain_order=True).sum().select(
        pl.col("Project date", "CIIU", "Capital investment"),
        pl.col("Capital investment").cum_sum().over("CIIU").alias("investment_cum_sum"),
    ).group_by("CIIU", maintain_order=True).agg(
            beginning_val = pl.col("investment_cum_sum").first(),
            ending_val = pl.col("investment_cum_sum").last(),
            n_years = pl.col("Project date").max() - pl.col("Project date").min(),
        ).with_columns(
        cagr_investment = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    fdi_cagr_investment
    return (fdi_cagr_investment,)


@app.cell
def _(fdi_lac, pl):
    ## Tasa de crecimiento compuesta para inversión de industrias en lac
    fdi_lac_cagr_investment = pl.from_pandas(fdi_lac[["CIIU", "Project date", "Capital investment"]]).sort(
        ["Project date", "CIIU"], maintain_order=True
    ).group_by("Project date", "CIIU",maintain_order=True).sum().select(
        pl.col("Project date", "CIIU", "Capital investment"),
        pl.col("Capital investment").cum_sum().over("CIIU").alias("investment_cum_sum"),
    ).group_by("CIIU", maintain_order=True).agg(
            beginning_val = pl.col("investment_cum_sum").first(),
            ending_val = pl.col("investment_cum_sum").last(),
            n_years = pl.col("Project date").max() - pl.col("Project date").min(),
        ).with_columns(
        cagr_investment = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    fdi_lac_cagr_investment
    return (fdi_lac_cagr_investment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Calculamos la tasa de crecimiento compuesta del empleo entre 2019 y 2024 para cada industria
    """)
    return


@app.cell
def _(fdi, pl):
    ## Tasa de crecimiento compuesta para inversión de industrias en todo el mundo
    fdi_cagr_empleo = pl.from_pandas(fdi[["CIIU", "Project date", "Jobs created"]]).sort(
        ["Project date", "CIIU"], maintain_order=True
    ).group_by("Project date", "CIIU",maintain_order=True).sum().select(
        pl.col("Project date", "CIIU", "Jobs created"),
        pl.col("Jobs created").cum_sum().over("CIIU").alias("empleo_cum_sum"),
    ).group_by("CIIU", maintain_order=True).agg(
            beginning_val = pl.col("empleo_cum_sum").first(),
            ending_val = pl.col("empleo_cum_sum").last(),
            n_years = pl.col("Project date").max() - pl.col("Project date").min(),
        ).with_columns(
        cagr_empleo = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    fdi_cagr_empleo
    return (fdi_cagr_empleo,)


@app.cell
def _(fdi_lac, pl):
    ## Tasa de crecimiento compuesta para inversión de industrias en lac
    fdi_lac_cagr_empleo = pl.from_pandas(fdi_lac[["CIIU", "Project date", "Jobs created"]]).sort(
        ["Project date", "CIIU"], maintain_order=True
    ).group_by("Project date", "CIIU",maintain_order=True).sum().select(
        pl.col("Project date", "CIIU", "Jobs created"),
        pl.col("Jobs created").cum_sum().over("CIIU").alias("empleo_cum_sum"),
    ).group_by("CIIU", maintain_order=True).agg(
            beginning_val = pl.col("empleo_cum_sum").first(),
            ending_val = pl.col("empleo_cum_sum").last(),
            n_years = pl.col("Project date").max() - pl.col("Project date").min(),
        ).with_columns(
        cagr_empleo = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    fdi_lac_cagr_empleo
    return (fdi_lac_cagr_empleo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Calculamos la elasticidad del crecimiento del empleo al crecimiento de la inversión

    Employment Elasticity of Growth

    This measures how employment responds to changes in FDI in a specific sector. It tells you how much the sector's employment grows for every 1% increase in sectoral growth.

    \begin{equation}
    \text { Elasticity }(\epsilon)=\frac{\% \text { Change in Employment }}{\% \text { Change in FDI }}
    \end{equation}

    * If ε > 0 and < 1, the sector creates jobs but FDI is also rising.
    * If ε > 1, the sector is highly labor-intensive and creates many jobs relative to FDI.

    https://infonomics-society.org/wp-content/uploads/ijcdse/published-papers/volume-6-2015/Economic-Growth-and-Sectoral-Capacity-for-Employment.pdf
    """)
    return


@app.cell
def _(fdi_cagr_empleo, fdi_cagr_investment, pl):
    elasticidad_empleo_fdi = fdi_cagr_investment.select("CIIU", "cagr_investment").join(
        fdi_cagr_empleo.select("CIIU", "cagr_empleo"), 
        on = "CIIU",
    ).with_columns(
        elasticidad = pl.col("cagr_empleo")/pl.col("cagr_investment")
    )
    elasticidad_empleo_fdi
    return (elasticidad_empleo_fdi,)


@app.cell
def _(fdi_lac_cagr_empleo, fdi_lac_cagr_investment, pl):
    elasticidad_lac_empleo_fdi = fdi_lac_cagr_investment.select("CIIU", "cagr_investment").join(
        fdi_lac_cagr_empleo.select("CIIU", "cagr_empleo"), 
        on = "CIIU",
    ).with_columns(
        elasticidad = pl.col("cagr_empleo")/pl.col("cagr_investment")
    )
    elasticidad_lac_empleo_fdi
    return (elasticidad_lac_empleo_fdi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Industry growth worldwide (past five years)

    Calcularemos el crecimiento de la industria CIIU como el crecimiento en la producción.
    """)
    return


@app.cell
def _(df_produccion, pl):
    industry_growth_rate = pl.from_pandas(
        df_produccion[
            ["TIME_PERIOD", "ACTIVITY", "PROD"]
        ].query(
            f"TIME_PERIOD in {[2018, 2019]}"
        )
    ).sort(
        ["ACTIVITY", "TIME_PERIOD"]
    ).group_by("ACTIVITY", maintain_order=True).agg(
            beginning_val = pl.col("PROD").first(),
            ending_val = pl.col("PROD").last(),
            n_years = pl.col("TIME_PERIOD").max() - pl.col("TIME_PERIOD").min(),
            #pl.col("PROD").pct_change().alias("Growth_Rate")
        ).with_columns(
        cagr_production = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    industry_growth_rate
    return (industry_growth_rate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⁠Industry growth worldwide (past five years-Atlas export growth)
    Calcularemos el crecimiento de la industria CIIU al calcular el crecimiento en exportaciones de los productos que componen a cada industria.

    De acuerdo a la metodología podemos descomponer la industria CIIU por los productos que la intengra, ponderado por el peso relativo de cada producto en la industria.

    Con tales ponderadores podemos crear con los datos del Atlas de Complejidad Económica un indicador del crecimiento exportador de la industria en el mundo.

    Aquí podemos usar la suma de exportaciones e importaciones para cuantificar una medida de comercio global.
    """)
    return


@app.cell
def _(pl):
    ## Cargamos datos del atlas
    atlas_hs12 = pl.read_parquet("datos/viabilidad_atractivo/hs12_country_product_year_4.parquet")
    atlas_hs12
    return (atlas_hs12,)


@app.cell
def _(atlas_hs12, pl):
    ## Valor de exportaciones de HS12 de 2012 a 2024
    exportaciones_hs = atlas_hs12.group_by("product_hs12_code", "year").agg(
        pl.col("export_value").sum()
    ).filter(
        pl.col("year").is_in([2019,2024])
    )
    exportaciones_hs
    return (exportaciones_hs,)


@app.cell
def _(pl):
    ## Cargamos crosswalk entre CIIU y HS12
    ciiu_hs12 = pl.read_csv("datos/recodificacion/ponderadores_ciiu_hs12_concordance.csv")
    ciiu_hs12 = ciiu_hs12.filter(pl.col("weight")!='NA').with_columns(
        pl.col("hs12").cast(pl.Int64), 
        pl.col("weight").cast(pl.Float64), 
    )
    ciiu_hs12
    return (ciiu_hs12,)


@app.cell
def _(ciiu_hs12, exportaciones_hs, pl):
    ## Reunimos datos de exportaciones por producto HS12 y el crosswalk CIIU-HS12
    industry_growth_rate_exports = exportaciones_hs.join(
        ciiu_hs12, 
        left_on="product_hs12_code", 
        right_on="hs12"
    ).with_columns(
        (pl.col("export_value")*pl.col("weight")).alias("export_value")
    ).group_by("ciiu", "year").agg(
        pl.col("export_value").sum()
    ).sort(
        ["ciiu", "year"]
    ).group_by("ciiu", maintain_order=True).agg(
            beginning_val = pl.col("export_value").first(),
            ending_val = pl.col("export_value").last(),
            n_years = pl.col("year").max() - pl.col("year").min(),
            #pl.col("PROD").pct_change().alias("Growth_Rate")
        ).with_columns(
        cagr_exports = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    industry_growth_rate_exports
    return (industry_growth_rate_exports,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⁠Possibility to substitute US imports from Asia (China)

    Usamos datos de atlas también. Pensemos un poco más como hacerlo.
    """)
    return


@app.cell
def _(pl):
    ## Carga datos
    china_imports = pl.read_csv("datos/viabilidad_atractivo/importaciones_usa_china_hs12.csv").select("product_hs12_code", "share_imports_china")
    china_imports
    return (china_imports,)


@app.cell
def _(china_imports, ciiu_hs12, pl):
    ### Reunimos datos de share import of china con la correspondencia CIIU y HS12
    ciiu_china_intensiveness = ciiu_hs12.join(
        china_imports, 
        left_on="hs12", 
        right_on="product_hs12_code"
    ).group_by("ciiu").agg(
        share_imports_china = (pl.col("share_imports_china") * pl.col("weight")).sum() / pl.col("weight").sum()
    )
    ciiu_china_intensiveness
    return (ciiu_china_intensiveness,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Capacity to create employment among specific groups (women, youth, low-skill)

    Employment Elasticity of Growth

    This measures how employment responds to changes in economic output (GDP) in a specific sector. It tells you how much the sector's employment grows for every 1% increase in sectoral growth.

    \begin{equation}
    \text { Elasticity }(\epsilon)=\frac{\% \text { Change in Employment }}{\% \text { Change in Output }}
    \end{equation}

    * If ε > 0 and < 1, the sector creates jobs but productivity is also rising.
    * If ε > 1, the sector is highly labor-intensive and creates many jobs relative to economic output.

    https://infonomics-society.org/wp-content/uploads/ijcdse/published-papers/volume-6-2015/Economic-Growth-and-Sectoral-Capacity-for-Employment.pdf
    """)
    return


@app.cell
def _(df_empleo, pl):
    employment_growth_rate = pl.from_pandas(
        df_empleo[
            ["TIME_PERIOD", "ACTIVITY", "EMPN"]
        ].query(
            f"TIME_PERIOD in {[2018, 2019]}"
        )
    ).sort(
        ["ACTIVITY", "TIME_PERIOD"]
    ).group_by("ACTIVITY", maintain_order=True).agg(
            beginning_val = pl.col("EMPN").first(),
            ending_val = pl.col("EMPN").last(),
            n_years = pl.col("TIME_PERIOD").max() - pl.col("TIME_PERIOD").min(),
            #pl.col("PROD").pct_change().alias("Growth_Rate")
        ).with_columns(
            cagr_employment = ((pl.col("ending_val") / pl.col("beginning_val")) ** (1 / pl.col("n_years")) - 1)*100
    )

    employment_growth_rate
    return (employment_growth_rate,)


@app.cell
def _(employment_growth_rate, industry_growth_rate, pl):
    employment_elasticity = industry_growth_rate.select(
                                  "ACTIVITY", "cagr_production"
                            ).join(
                                employment_growth_rate.select(
                                    "ACTIVITY", "cagr_employment"
                                ), 
                                on = "ACTIVITY"
                            ).with_columns(
                                elasticity = pl.col("cagr_employment")/pl.col("cagr_production")
                            )
    employment_elasticity
    return (employment_elasticity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viability
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Strength in countries like Honduras (RCA in peer group)
    """)
    return


@app.cell
def _(pl):
    ## Cargamos datos de complejidad y nos quedamos con los registros de honduras
    cdata = pl.read_csv("datos/viabilidad_atractivo/cdata.csv")

    ## Analizamos solo los pares
    ## Calculamos el rca promedio entre los pares
    rca_peers = cdata.filter(
        pl.col("REF_AREA").is_in(["SLV", "ECU"])
    ).group_by("ACTIVITY").agg(
        pl.col("rca").mean().alias("rca_peers")
    ).rename({"ACTIVITY" : "ciiu"})

    rca_peers
    return cdata, rca_peers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Availability of inputs (doble razor, let us talk)

    Además de los datos del Atlas, usaremos los datos de [AI-generated Production Network - AIPNET](https://aipnet.io/) para identificar la cadena de producción de los productos.
    """)
    return


@app.cell
def _(pl):
    ## Cargamos cadena de producción de los productos hs12 de aipnet
    aipnet = pl.read_csv("datos/viabilidad_atractivo/aipnet_hs12_4d.csv")
    aipnet
    return (aipnet,)


@app.cell
def _(aipnet, ciiu_hs12):
    ## Reunimos datos de AIPNET con el crosswalk de CIIU-HS12
    aipnet_ciiu = aipnet.join(
        ciiu_hs12, 
        left_on="hs2012_code_upstream",
        right_on="hs12",
    ).select(
        "ciiu", "weight", "hs2012_code_upstream", "hs2012_code_downstream"
    ).rename(
        {
            "hs2012_code_upstream" : "hs12"
        }
    )
    aipnet_ciiu
    return (aipnet_ciiu,)


@app.cell
def _(ciiu_hs12):
    ciiu_hs12
    return


@app.cell
def _(atlas_hs12, pl):
    ### Filtramos datos de HND
    atlas_hs12_hnd = atlas_hs12.filter(
        (pl.col("country_iso3_code")=="HND") &
        (pl.col("year")==2024)
    )
    atlas_hs12_hnd
    return (atlas_hs12_hnd,)


@app.cell
def _(aipnet_ciiu, atlas_hs12_hnd, pl):
    ## Creamos dataframe que contiene el porcentaje de insumos presentes para la producción del producto hs12
    threshold_intensidad_importacion = 0.2 

    aipnet_ciiu_razon_insumos = aipnet_ciiu.join(
        atlas_hs12_hnd.select("product_hs12_code", "export_rca", "import_value"), 
        left_on="hs2012_code_downstream", 
        right_on="product_hs12_code", 
        how = "left"
    ).fill_null(0).with_columns(
        ## Etiquetamos con 1 los productos que se exportan con ventaja comparativa
        M = pl.when(
            pl.col("export_rca")>=1
        ).then(
            pl.lit(1)
        ).otherwise(
            pl.lit(0)
        ),
        ## Calculamos el porcentaje de importación por producto que importa cada cada producto para el total de importación que implica su cadena de producción
        razon_importacion = pl.col("import_value")/pl.col("import_value").sum().over("ciiu","hs12")
    ).with_columns(
        ## Variable que indica si el producto se importa con intensidad (el insumo representa el 20% de las importaciones totales con las que se produce el producto)
        se_importa = pl.when(
            pl.col("razon_importacion") >= threshold_intensidad_importacion
        ).then(
            pl.lit(1)
        ).otherwise(
            0
        )
    ).with_columns(
        ## Un insumo está disponible por dos condiciones : 
        ## 1) Lo exporta con ventaja comparativa o 
        ## 2) lo importa con intensidad 
        disponible = pl.when(
            (pl.col("M")==1) | (pl.col("se_importa")==1)
        ).then(
            pl.lit(1)
        ).otherwise(
            pl.lit(0)
        )
    ).group_by("ciiu","hs12", "weight").agg(
        pl.col("disponible").sum().alias("inputs_presentes"),
        pl.col("disponible").count().alias("inputs_totales"),
    ).with_columns(
        razon_insumos_presentes = pl.col("inputs_presentes")/pl.col("inputs_totales")
    ).with_columns(
        weight__insumos_presentes = pl.col("weight")*pl.col("razon_insumos_presentes")
    )
    aipnet_ciiu_razon_insumos
    return (aipnet_ciiu_razon_insumos,)


@app.cell
def _():
    return


@app.cell
def _(aipnet_ciiu_razon_insumos, pl):
    ### Calculamos la razón de insumos presentes para cada industria CIIU
    ciiu_insumos_presentes = aipnet_ciiu_razon_insumos.group_by(
        "ciiu"
    ).agg(
        pl.col("weight__insumos_presentes").sum().alias("razon_insumos_presentes")
    )
    ciiu_insumos_presentes
    return (ciiu_insumos_presentes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reliance on a constraint or potential constraint (energy, security)
    """)
    return


@app.cell
def _(df_produccion, pl):
    share_energy = pl.from_pandas(
        df_produccion
    ).filter(
        TIME_PERIOD=2019
    ).with_columns(
        share_energy = pl.col("INEN")/pl.col("INGS")
    ).select(
        "ACTIVITY", "share_energy"
    )
    share_energy
    return (share_energy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reliance on a constraint or potential constraint (electricity-SCIAN México)
    """)
    return


@app.cell
def _(pl):
    ## Cargamos crosswalk entre CIIU y NAICS
    ciiu_naics = pl.read_csv("datos/recodificacion/ponderadores_ciiu_naics2017_concordance.csv")
    ciiu_naics
    return (ciiu_naics,)


@app.cell
def _(pd):
    ## Cargamos consumo de energia electrica
    electricidad = pd.read_csv("datos/viabilidad_atractivo/electricidad_saic_2003-2023.csv")
    electricidad["actividad"] = electricidad["actividad"].apply(lambda x : x.split()[1])

    electricidad_colname = "K412A Gasto por consumo de energía eléctrica (millones de pesos)"
    total_colname = "K000A Total de gastos por consumo de bienes y servicios (millones de pesos)"

    electricidad["razon_electricidad_gasto_total"] = electricidad[electricidad_colname]/electricidad[total_colname]
    electricidad = electricidad.drop(columns=[electricidad_colname, total_colname])
    electricidad = electricidad.pivot(index="actividad", columns="anio", values="razon_electricidad_gasto_total").reset_index()
    electricidad
    return (electricidad,)


@app.cell
def _(electricidad, pl):
    electricidad_share = pl.from_pandas(
        electricidad[["actividad", 2023]].rename(
            columns = {
                2023 : "razon_electricidad_gasto_total", 
                "actividad" : "naics"
            }
        )
    ).with_columns(
        pl.col("naics").cast(pl.Int32)
    )
    electricidad_share
    return (electricidad_share,)


@app.cell
def _(ciiu_naics, electricidad_share, pl):
    ### Reunimos razon de consumo de electricidad y el crosswalk CIIU-NAICS
    ### y calculamos la media ponderada por industria CIIU
    ciiu_razon_electricidad_gasto_total = ciiu_naics.join(
        electricidad_share, 
        on = "naics", 
        how="left"
    ).group_by("ciiu").agg(
        razon_electricidad_gasto_total = (pl.col("razon_electricidad_gasto_total") * pl.col("weight")).sum() / pl.col("weight").sum()
    )
    ciiu_razon_electricidad_gasto_total
    return (ciiu_razon_electricidad_gasto_total,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reunimos los datos
    """)
    return


@app.cell
def _(
    ciiu_china_intensiveness,
    ciiu_insumos_presentes,
    ciiu_razon_electricidad_gasto_total,
    elasticidad_empleo_fdi,
    elasticidad_lac_empleo_fdi,
    employment_elasticity,
    fdi_cagr_investment,
    fdi_capital_investment,
    fdi_lac_cagr_investment,
    fdi_lac_capital_investment,
    industry_growth_rate,
    industry_growth_rate_exports,
    pl,
    share_energy,
):
    # Attractiveness
    ## Capacidad para movilizar FDI (world and region)
    ###  Monto acumulado de inversión en capital y creacion de empleo entre 2019 y 2024
    fdi_capital_investment_final = pl.from_pandas(fdi_capital_investment).select("CIIU", "Capital investment").rename({"CIIU":"ciiu", "Capital investment" : "cumulative_investment_world"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )
    fdi_lac_capital_investment_final = pl.from_pandas(fdi_lac_capital_investment).select("CIIU", "Capital investment").rename({"CIIU":"ciiu", "Capital investment" : "cumulative_investment_lac"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ### Tasa de crecimiento compuesta de la inversión entre 2019 y 2024 para cada industria
    fdi_cagr_investment_final = fdi_cagr_investment.select("CIIU", "cagr_investment").rename({"CIIU" : "ciiu", "cagr_investment" : "cagr_investment_world"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )
    fdi_lac_cagr_investment_final = fdi_lac_cagr_investment.select("CIIU", "cagr_investment").rename({"CIIU" : "ciiu", "cagr_investment" : "cagr_investment_lac"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ### Elasticidad del crecimiento del empleo al crecimiento de FDI
    elasticidad_empleo_fdi_final = elasticidad_empleo_fdi.select("CIIU", "elasticidad").rename({"CIIU" : "ciiu", "elasticidad" : "elasticidad_empleo_fdi_world"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )
    elasticidad_lac_empleo_fdi_final = elasticidad_lac_empleo_fdi.select("CIIU", "elasticidad").rename({"CIIU" : "ciiu", "elasticidad" : "elasticidad_empleo_fdi_lac"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Industry growth worldwide (past five years)
    industry_growth_rate_final = industry_growth_rate.select("ACTIVITY","cagr_production").rename({"ACTIVITY" : "ciiu"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Industry growth worldwide (past five years-Atlas export growth)
    industry_growth_rate_exports_final = industry_growth_rate_exports.select("ciiu", "cagr_exports").with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Possibility to substitute US imports from Asia (China)
    ciiu_china_intensiveness_final = ciiu_china_intensiveness.clone().with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Capacity to create employment among specific groups (women, youth, low-skill)
    employment_elasticity_final = employment_elasticity.select("ACTIVITY", "elasticity").rename({"ACTIVITY" : "ciiu", "elasticity" : "elasticidad_empleo_producto"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    # Viability
    ## Strength in countries like Honduras (RCA in peer group)
    ## Availability of inputs (doble razor, let us talk)
    ciiu_insumos_presentes_final = ciiu_insumos_presentes.clone().with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Reliance on a constraint or potential constraint (energy, security)
    share_energy_final = share_energy.rename({"ACTIVITY" : "ciiu"}).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    ## Reliance on a constraint or potential constraint (electricity-SCIAN México)
    ciiu_razon_electricidad_gasto_total_final = ciiu_razon_electricidad_gasto_total.clone().with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )
    return (
        ciiu_china_intensiveness_final,
        ciiu_insumos_presentes_final,
        ciiu_razon_electricidad_gasto_total_final,
        elasticidad_empleo_fdi_final,
        elasticidad_lac_empleo_fdi_final,
        employment_elasticity_final,
        fdi_cagr_investment_final,
        fdi_capital_investment_final,
        fdi_lac_cagr_investment_final,
        fdi_lac_capital_investment_final,
        industry_growth_rate_exports_final,
        industry_growth_rate_final,
        share_energy_final,
    )


@app.cell
def _(cdata):
    ## Cargamos datos de complejidad y nos quedamos con los registros de honduras
    cdata_hnd = cdata.filter(REF_AREA="HND")
    cdata_hnd
    return (cdata_hnd,)


@app.cell
def _(
    cdata_hnd,
    ciiu_china_intensiveness_final,
    ciiu_insumos_presentes_final,
    ciiu_razon_electricidad_gasto_total_final,
    elasticidad_empleo_fdi_final,
    elasticidad_lac_empleo_fdi_final,
    employment_elasticity_final,
    fdi_cagr_investment_final,
    fdi_capital_investment_final,
    fdi_lac_cagr_investment_final,
    fdi_lac_capital_investment_final,
    industry_growth_rate_exports_final,
    industry_growth_rate_final,
    pl,
    rca_peers,
    share_energy_final,
):
    ## Obtenemos las actividades CIIU a analizar
    ciiu_analiza = cdata_hnd.select("ACTIVITY").rename({"ACTIVITY" : "ciiu"})

    ## Concatena con los indicadores calculados
    factores = pl.concat(
        [
            ciiu_analiza, 
            fdi_capital_investment_final,
            fdi_lac_capital_investment_final,
            fdi_cagr_investment_final,
            fdi_lac_cagr_investment_final,
            elasticidad_empleo_fdi_final,
            elasticidad_lac_empleo_fdi_final,
            industry_growth_rate_final,
            industry_growth_rate_exports_final,
            ciiu_china_intensiveness_final,
            employment_elasticity_final,
            rca_peers,
            ciiu_insumos_presentes_final,
            share_energy_final,
            ciiu_razon_electricidad_gasto_total_final
        ], how="align"
    ).filter(
        pl.col("ciiu").is_in(cdata_hnd["ACTIVITY"])
    )
    factores
    return (factores,)


@app.cell
def _(factores, pd, pl):
    ## Imputamos datos con Kmedias
    from sklearn.impute import KNNImputer

    # Initialize the imputer (setting K=2 neighbors)
    imputer = KNNImputer(n_neighbors=2, weights="uniform")

    # Fit and transform the data
    factores_imputados = pl.from_pandas(
        pd.DataFrame(imputer.fit_transform(factores.to_pandas()), columns=factores.columns)
    )
    factores_imputados
    return (factores_imputados,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TOPSIS
    """)
    return


@app.cell
def _():
    import numpy as np
    from pymcdm.methods import TOPSIS
    from pymcdm.helpers import rrankdata
    return TOPSIS, np, rrankdata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Viabilidad
    """)
    return


@app.cell
def _(TOPSIS, factores_imputados, np, rrankdata):
    # TOPSIS atractivo
    atractivo_factores = [
        "cumulative_investment_world", 
        "cumulative_investment_lac",
        "cagr_investment_world",
        "cagr_investment_lac",
        "elasticidad_empleo_fdi_world",
        "elasticidad_empleo_fdi_lac",
        "cagr_production",
        "cagr_exports",
        "share_imports_china", 
        "elasticidad_empleo_producto"
    ]

    alts_atractivo = factores_imputados.select(atractivo_factores).to_numpy()

    # Define criteria weights (should sum up to 1)
    weights_atractivo = np.array([1/len(atractivo_factores)]*len(atractivo_factores))

    # Define criteria types (1 for profit, -1 for cost)
    types_atractivo = np.array([1]*len(atractivo_factores))

    # Create object of the method
    # Note, that default normalization method for TOPSIS is minmax
    topsis_atractivo = TOPSIS()

    # Determine preferences and ranking for alternatives
    pref_atractivo = topsis_atractivo(alts_atractivo, weights_atractivo, types_atractivo)
    ranking_atractivo = rrankdata(pref_atractivo)
    return (pref_atractivo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Atractivo
    """)
    return


@app.cell
def _(TOPSIS, factores_imputados, np, rrankdata):
    # TOPSIS Viabilidad
    viabilidad_factores = [
        "rca_peers",
        "razon_insumos_presentes", 
        "share_energy",
        "razon_electricidad_gasto_total",
    ]

    alts_viabilidad = factores_imputados.select(viabilidad_factores).to_numpy()

    # Define criteria weights (should sum up to 1)
    weights_viabilidad = np.array([1/len(viabilidad_factores)]*len(viabilidad_factores))

    # Define criteria types (1 for profit, -1 for cost)
    types_viabilidad = np.array([1, 1, -1, -1])

    # Create object of the method
    # Note, that default normalization method for TOPSIS is minmax
    topsis_viabilidad = TOPSIS()

    # Determine preferences and ranking for alternatives
    pref_viabilidad = topsis_viabilidad(alts_viabilidad, weights_viabilidad, types_viabilidad)
    ranking_viabilidad = rrankdata(pref_viabilidad)
    return (pref_viabilidad,)


@app.cell
def _(factores_imputados, pl, pref_atractivo, pref_viabilidad):
    ### Creamos data frame con los scores de viabilidad y atractivo
    scores_viabilidad_atractivo = factores_imputados.select("ciiu").with_columns(
            topsis_atractivo = pref_atractivo, 
            topsis_viabilidad = pref_viabilidad, 
    ).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )
    scores_viabilidad_atractivo
    return (scores_viabilidad_atractivo,)


@app.cell
def _(pd, pl):
    # Cargamos recodificación
    recod = pd.read_csv("datos/viabilidad_atractivo/recodificacion_hnd_usa.csv")

    ## Diccionario CIIU 4 a nombres
    mapp_ciiu = pl.from_pandas(recod.query("clasificador=='ciiu_rev_4'")[["codigo", "nombre_actividad"]])

    ### Cargamos selección de industrias de Pedro
    ciiu_pedro_2 = pl.from_pandas(
        pd.read_csv("datos/viabilidad_atractivo/seleccion_pedro.csv").query("incluye==1")
    )

    ### Resultados finales Intensivo
    resultados_finales_intensivo = pd.read_excel("datos/viabilidad_atractivo/Resultados Complexity_final.xlsx", sheet_name="Intensivo")

    ### Resultados finales Extensivo
    resultados_finales_extensivo = pd.read_excel("datos/viabilidad_atractivo/Resultados Complexity_final.xlsx", sheet_name="Extensivo")
    return (
        ciiu_pedro_2,
        mapp_ciiu,
        resultados_finales_extensivo,
        resultados_finales_intensivo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Margen Intensivo
    """)
    return


@app.cell
def _(
    cdata_hnd,
    ciiu_pedro_2,
    mapp_ciiu,
    pd,
    pl,
    resultados_finales_intensivo,
    scores_viabilidad_atractivo,
):
    import altair as alt 

    cdata_intensivo = cdata_hnd.filter(
        (pl.col("REF_AREA")=="HND") & 
        (pl.col("rca")>0) & 
        (pl.col("mcp")==1)
    )
    cdata_intensivo = cdata_intensivo.join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).join(
        ciiu_pedro_2.select("clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo", "division_titulo"),
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )

    cdata_intensivo = cdata_intensivo.join(
        scores_viabilidad_atractivo, 
        left_on="ACTIVITY", 
        right_on="ciiu"
    ).filter(
        pl.col("ACTIVITY").is_in(resultados_finales_intensivo["ciiu4_cod"])
    )


    plot_intensivo = alt.Chart(
        cdata_intensivo    
    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
            ).encode(
        x=alt.X('topsis_viabilidad').scale(zero=False).title("Viabilidad"),
        y=alt.Y('topsis_atractivo').scale(zero=False).title("Atractivo"),#.scale(type ="log"),
        color = alt.Color("seccion_titulo").title("Sección"),
        #size = alt.Size("OBS_VALUE").scale(type ="log").title("Empleo"),
        tooltip=[

                alt.Tooltip('nombre_actividad', title='Actividad'), 
                alt.Tooltip('division_titulo', title='División CIIU Rev 4'),
                alt.Tooltip('OBS_VALUE', title='Empleo'),
        ] 

    )

    # Create a horizontal line at y = -1.14
    rule_atractivo = alt.Chart(pd.DataFrame({'y': [cdata_intensivo["topsis_atractivo"].mean()]})).mark_rule(color='red').encode(y='y:Q')
    rule_viabilidad = alt.Chart(pd.DataFrame({'x': [cdata_intensivo["topsis_viabilidad"].mean()]})).mark_rule(color='red').encode(x='x:Q')


    (plot_intensivo + rule_atractivo + rule_viabilidad).properties(
    #plot_intensivo.properties(
            title=alt.TitleParams(
                "Diagrama Viabilidad-Atractivo",
                subtitle="Margen Intensivo",
                subtitleColor="gray"
            )
    )
    return alt, cdata_intensivo


@app.cell
def _(cdata_intensivo):
    cdata_intensivo.select("clase_titulo", "topsis_atractivo", "topsis_viabilidad")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Margen Extensivo
    """)
    return


@app.cell
def _(
    alt,
    cdata_hnd,
    ciiu_pedro_2,
    mapp_ciiu,
    pd,
    pl,
    resultados_finales_extensivo,
    scores_viabilidad_atractivo,
):
    cdata_extensivo = cdata_hnd.filter(
        (pl.col("REF_AREA")=="HND") & 
        (pl.col("rca")>0) & 
        (pl.col("mcp")==0)
    )
    cdata_extensivo = cdata_extensivo.join(
        mapp_ciiu,
        left_on="ACTIVITY", 
        right_on="codigo"
    ).join(
        ciiu_pedro_2.select("clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo", "division_titulo"),
        left_on= "ACTIVITY", 
        right_on = "clase_codigo"
    )

    cdata_extensivo = cdata_extensivo.join(
        scores_viabilidad_atractivo, 
        left_on="ACTIVITY", 
        right_on="ciiu"
    ).filter(
        pl.col("ACTIVITY").is_in(resultados_finales_extensivo["ciiu4_cod"])
    )


    plot_extensivo = alt.Chart(
        cdata_extensivo    
    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
            ).encode(
        x=alt.X('topsis_viabilidad').scale(zero=False).title("Viabilidad"),
        y=alt.Y('topsis_atractivo').scale(zero=False).title("Atractivo"),#.scale(type ="log"),
        color = alt.Color("seccion_titulo").title("Sección"),
        #size = alt.Size("OBS_VALUE").scale(type ="log").title("Empleo"),
        tooltip=[

                alt.Tooltip('nombre_actividad', title='Actividad'), 
                alt.Tooltip('division_titulo', title='División CIIU Rev 4'),
                alt.Tooltip('OBS_VALUE', title='Empleo'),
        ] 

    )

    # Create a horizontal line at y = -1.14
    rule_extensivo_atractivo = alt.Chart(pd.DataFrame({'y': [cdata_extensivo["topsis_atractivo"].mean()]})).mark_rule(color='red').encode(y='y:Q')
    rule_extensivo_viabilidad = alt.Chart(pd.DataFrame({'x': [cdata_extensivo["topsis_viabilidad"].mean()]})).mark_rule(color='red').encode(x='x:Q')


    (plot_extensivo + rule_extensivo_atractivo + rule_extensivo_viabilidad).properties(
    #plot_intensivo.properties(
            title=alt.TitleParams(
                "Diagrama Viabilidad-Atractivo",
                subtitle="Margen Extensivo",
                subtitleColor="gray"
            )
    )
    return (cdata_extensivo,)


@app.cell
def _(cdata_extensivo):
    cdata_extensivo.select("clase_titulo", "topsis_atractivo", "topsis_viabilidad")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
