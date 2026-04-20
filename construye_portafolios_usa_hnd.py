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
    import numpy as np
    from great_tables import GT, html
    return GT, html, np, pl


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

    ## Cargamos nombres de recodificación
    recodificacion = pl.read_csv("datos/recodificacion/recodificacion_hnd_usa_nombres.csv")

    ## Calculamos densidad normalizada para cada conjunto de datos
    def calcula_density_z(df : pl.DataFrame
        ) -> pl.DataFrame:

        ### Normalizamos density
        df = df.with_columns(
            density_norm = (pl.col("density") - pl.col("density").mean())/pl.col("density").std()
        )

        return df 

    ## Normalizamos densidad
    datos = {
        "ue_nacional" : calcula_density_z(ue_honduras_pais_cdata), 
        "ue_subnacional" : calcula_density_z(ue_honduras_deptos_cdata), 
        "po_nacional" : calcula_density_z(empleo_honduras_pais_cdata), 
        "po_subnacional" : calcula_density_z(empleo_honduras_deptos_cdata), 

    }
    return datos, recodificacion


@app.cell
def _(np, pl, recodificacion):
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
            (pl.col("zona") == "HND")
        ).select("zona", "actividad", "mcp", portafolio)

        df_portafolios_score = df_portafolios_score.join(
            recodificacion, 
            on = "actividad"
        ).sort(by = portafolio, descending=True)


        return df_portafolios_score
    return calcula_score, mapp_portafolios


@app.cell
def _(calcula_score, pl):
    def obten_ranking(portafolio : str, 
                      magnitud : str,
                      datos : pl.DataFrame, ) -> pl.DataFrame:

        return calcula_score(
            portafolio, datos[magnitud]
        ).select(
            portafolio,"nombre_actividad"
        ).with_columns(
            pl.col(portafolio).rank(descending=True).alias("rank")
        ).drop(portafolio).head(10).select("rank", "nombre_actividad").rename({
            "nombre_actividad" : f"nombre_actividad_{portafolio}", 
            "rank" : f"rank_{portafolio}"
        })
    return (obten_ranking,)


@app.cell
def _(datos, mapp_portafolios, obten_ranking, pl):
    ### Creamos portafolios 
    portafolios_po = [obten_ranking(portafolio, "po_nacional", datos) for portafolio in mapp_portafolios]
    portafolios_ue = [obten_ranking(portafolio, "ue_nacional", datos) for portafolio in mapp_portafolios]

    portafolios_po = pl.concat(portafolios_po,  how = "horizontal")
    portafolios_ue = pl.concat(portafolios_ue,  how = "horizontal")
    return portafolios_po, portafolios_ue


@app.cell
def _(GT, html, portafolios_po):
    (
        GT(portafolios_po)
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
        )
    )
    return


@app.cell
def _(GT, html, portafolios_ue):
    (
        GT(portafolios_ue)
        .tab_header(
            title="Selección de Portafolios de Actividades",
            subtitle="Unidades Económicas"
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
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
