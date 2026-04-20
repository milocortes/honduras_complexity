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
    import altair as alt
    import pandas as pd 
    return alt, pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cargamos datos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Complejidad
    """)
    return


@app.cell
def _(pl):
    ## Cargamos datos
    ue_honduras_pais_cdata = pl.read_parquet("delta_db/cdata/ue_honduras_pais_cdata.parquet")
    ue_honduras_deptos_cdata = pl.read_parquet("delta_db/cdata/ue_honduras_deptos_cdata.parquet")
    empleo_honduras_pais_cdata = pl.read_parquet("delta_db/cdata/empleo_honduras_pais_cdata.parquet")
    empleo_honduras_deptos_cdata = pl.read_parquet("delta_db/cdata/empleo_honduras_deptos_cdata.parquet")
    return (
        empleo_honduras_deptos_cdata,
        empleo_honduras_pais_cdata,
        ue_honduras_deptos_cdata,
        ue_honduras_pais_cdata,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GDP
    """)
    return


@app.cell
def _(pl):
    gdp = pl.read_csv("datos/gdp_hnd_msa/gdp_hnd_msa.csv")
    return (gdp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Población
    """)
    return


@app.cell
def _(pd):
    ### Cargamos datos de población
    poblacion = pd.read_csv("datos/poblacion/pop_usa_mx_hnd.csv")
    poblacion["tipo"] = poblacion["tipo"].replace({"Metropolitan Statistical Area" : "MSA", "Micropolitan Statistical Area" : "MSA"})
    poblacion_empleo = poblacion.copy()
    poblacion_ue = poblacion.copy()
    poblacion
    return (poblacion_empleo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reunimos datos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unidades Económicas
    """)
    return


@app.cell
def _(gdp, pl, ue_honduras_pais_cdata):
    ## Reunimos datos
    ue_honduras_pais_eci = ue_honduras_pais_cdata.with_columns(
        pl.col("zona").replace({"HND" : "Honduras"})
    ).select(
                "zona", "eci"
            ).unique().join(
                gdp,
                left_on="zona", 
                right_on="name"
            )

    ## Agregamos etiqueta de Honduras USA
    ue_honduras_pais_eci = ue_honduras_pais_eci.with_columns(
        pl.when(
            pl.col("zona") == "Honduras"
        ).then(
            pl.lit("Honduras")
        ).otherwise(
            pl.lit("MSA")
        ).alias("Tipo")
    )

    ue_honduras_pais_eci
    return (ue_honduras_pais_eci,)


@app.cell
def _(alt, ue_honduras_pais_eci):
    ue_pais_gdp_scatter = alt.Chart(ue_honduras_pais_eci, 
                                    title = "Unidades Económicas"
            ).mark_circle(       
            opacity=0.99,
            stroke='black',
            strokeWidth=1.2,
            strokeOpacity=0.9, 
            size=180
            ).encode(
            x=alt.X('gdp:Q', scale=alt.Scale(type='log')).title("GDP Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("Tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    ue_pais_gdp_reg_line = ue_pais_gdp_scatter.transform_regression(
        'gdp', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N', 
            color =  alt.ColorValue('black')
        )

    ue_pais_gdp_chart = ue_pais_gdp_scatter + ue_pais_gdp_reg_line
    ue_pais_gdp_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell
def _():
    return


@app.cell
def _(alt, ue_honduras_pais_eci):
    ue_pais_pob_scatter = alt.Chart(ue_honduras_pais_eci, 
                                    title = "Unidades Económicas"
            ).mark_circle(
            opacity=0.99,
            stroke='black',
            strokeWidth=1.2,
            strokeOpacity=0.9, 
            size=180
            ).encode(
            x=alt.X('pob_2020:Q', scale=alt.Scale(type='log')).title("Población Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("Tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    ue_pais_pob_reg_line = ue_pais_pob_scatter.transform_regression(
        'pob_2020', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N'
        )

    ue_pais_pob_chart = ue_pais_pob_scatter + ue_pais_pob_reg_line
    ue_pais_pob_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Población Ocupada
    """)
    return


@app.cell
def _(empleo_honduras_pais_cdata, gdp, pl):
    ## Reunimos datos
    pob_honduras_pais_eci = empleo_honduras_pais_cdata.with_columns(
        pl.col("zona").replace({"HND" : "Honduras"})
    ).select(
                "zona", "eci"
            ).unique().join(
                gdp,
                left_on="zona", 
                right_on="name"
            )

    ## Agregamos etiqueta de Honduras USA
    pob_honduras_pais_eci = pob_honduras_pais_eci.with_columns(
        pl.when(
            pl.col("zona") == "Honduras"
        ).then(
            pl.lit("Honduras")
        ).otherwise(
            pl.lit("MSA")
        ).alias("Tipo")
    )

    pob_honduras_pais_eci
    return (pob_honduras_pais_eci,)


@app.cell
def _(alt, pob_honduras_pais_eci):
    pob_pais_gdp_scatter = alt.Chart(pob_honduras_pais_eci, 
                                     title = "Población Ocupada").mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180
                                     ).encode(
            x=alt.X('gdp:Q', scale=alt.Scale(type='log')).title("GDP Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("Tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    pob_pais_gdp_reg_line = pob_pais_gdp_scatter.transform_regression(
        'gdp', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N'
        )

    pob_pais_gdp_chart = pob_pais_gdp_scatter + pob_pais_gdp_reg_line
    pob_pais_gdp_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell
def _(alt, pob_honduras_pais_eci):
    pob_pais_pob_scatter = alt.Chart(pob_honduras_pais_eci, 
                                     title = "Población Ocupada"
            ).mark_circle(
            opacity=0.99,
            stroke='black',
            strokeWidth=1.2,
            strokeOpacity=0.9, 
            size=180
            ).encode(
            x=alt.X('pob_2020:Q', scale=alt.Scale(type='log')).title("Población Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("Tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    pob_pais_pob_reg_line = pob_pais_pob_scatter.transform_regression(
        'pob_2020', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N'
        )

    pob_pais_pob_chart = pob_pais_pob_scatter + pob_pais_pob_reg_line
    pob_pais_pob_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subnacional
    """)
    return


@app.cell
def _():
    cw_empleo = {
        "Cortés" : "Cortes", 
        "Francisco Morazán" : "Francisco Morazán", 
        "Yoro" : "Yoro", 
        "Olancho" : "Olancho", 
        "Comayagua" : "Comayagua", 
        "El Paraíso" : "El Paraíso", 
        "Atlántida" : "Atlántida", 
        "Choluteca" : "Choluteca", 
        "Santa Barbara" : "Santa Bárbara", 
        "Copán" : "Copan", 
        #"Lempira" : "", 
        "Colón" : "Colon", 
        "Intibucá" : "", 
        "La Paz" : "La Paz", 
        "Valle" : "Valle", 
        #"Ocotepeque" : "", 
        #"Gracias a Dios" : "", 
        "Islas de la Bahía" : "Islas de La Bahía", 
    }

    cw_ue = {
        "Cortés" : "CORTES", 
        "Francisco Morazán" : "FRANCISCO MORAZAN", 
        "Yoro" : "YORO", 
        "Olancho" : "OLANCHO", 
        "Comayagua" : "COMAYAGUA", 
        "El Paraíso" : "EL PARAISO", 
        "Atlántida" : "ATLANTIDA", 
        "Choluteca" : "CHOLUTECA", 
        "Santa Barbara" : "SANTA BARBARA", 
        "Copán" : "COPAN", 
        "Lempira" : "LEMPIRA", 
        "Colón" : "COLON", 
        "Intibucá" : "INTIBUCA", 
        "La Paz" : "La Paz", 
        "Valle" : "VALLE", 
        "Ocotepeque" : "OCOTEPEQUE", 
        "Gracias a Dios" : "GRACIAS A DIOS", 
        "Islas de la Bahía" : "ISLAS DE LA BAHIA", 
    }
    return (cw_empleo,)


@app.cell
def _(cw_empleo, poblacion_empleo):
    poblacion_empleo["nom_region"] = poblacion_empleo["nom_region"].replace(cw_empleo)
    #poblacion_ue["nom_region"] = poblacion_ue["nom_region"].replace(cw_ue)
    return


@app.cell
def _(
    empleo_honduras_deptos_cdata,
    pl,
    poblacion_empleo,
    ue_honduras_deptos_cdata,
):
    ue_honduras_deptos_pob = ue_honduras_deptos_cdata.select(
        "zona","eci"
    ).unique().join(
        pl.from_pandas(poblacion_empleo), 
        left_on="zona", 
        right_on="nom_region"
    )

    empleo_honduras_deptos_pob = empleo_honduras_deptos_cdata.select(
        "zona","eci"
    ).unique().join(
        pl.from_pandas(poblacion_empleo), 
        left_on="zona", 
        right_on="nom_region"
    )
    return empleo_honduras_deptos_pob, ue_honduras_deptos_pob


@app.cell
def _(alt, empleo_honduras_deptos_pob):
    empleo_depto_pob_scatter = alt.Chart(empleo_honduras_deptos_pob, 
                                         title = "Población Ocupada"
                                        ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180
                                        ).encode(
            x=alt.X('pob_2020:Q', scale=alt.Scale(type='log')).title("Población Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    empleo_depto_pob_reg_line = empleo_depto_pob_scatter.transform_regression(
        'pob_2020', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N'
        )

    empleo_depto_pob_chart = empleo_depto_pob_scatter + empleo_depto_pob_reg_line
    empleo_depto_pob_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell
def _(alt, ue_honduras_deptos_pob):
    ue_depto_pob_scatter = alt.Chart(ue_honduras_deptos_pob, 
                                     title = "Unidades Económicas"
                                    ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180
                                    ).encode(
            x=alt.X('pob_2020:Q', scale=alt.Scale(type='log')).title("Población Log"),
            y=alt.Y('eci').title("ECI"),
            color = alt.Color("tipo").title("Tipo"),#.scale(domain=domain, range=range_),
            tooltip=['pob_2020', 'eci', "zona"]
        )
    ue_depto_pob_reg_line = ue_depto_pob_scatter.transform_regression(
        'pob_2020', 'eci',
        method="log"
    ).mark_line(size = 5).transform_calculate(
            Fit='"LinReg"'
        ).encode(
            stroke='Fit:N'
        )

    ue_depto_pob_chart = ue_depto_pob_scatter + ue_depto_pob_reg_line
    ue_depto_pob_chart.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        orient='top-left')
    return


@app.cell
def _(ue_honduras_deptos_pob):
    ue_honduras_deptos_pob.filter(tipo="Departamento")
    return


@app.cell
def _(ue_honduras_deptos_cdata):
    ue_honduras_deptos_cdata
    return


if __name__ == "__main__":
    app.run()
