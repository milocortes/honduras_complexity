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
    from datetime import datetime
    import altair as alt
    import numpy as np 
    return ecomplexity, np, pl


@app.cell
def _(pl):
    ## Cargamos INDSTAT
    indstat = pl.read_parquet("datos/INDSTAT/data.parquet")

    ## Cargamos categorias de actividades
    ciiu = pl.read_csv("datos/INDSTAT/activity.csv").filter(
        pl.col("ActivityCode").map_elements(lambda x : len(x)==4)
    )["ActivityCode"].to_numpy()

    ciiu_nombres = pl.read_csv("datos/INDSTAT/activity.csv").filter(
        pl.col("ActivityCode").map_elements(lambda x : len(x)==4)
    )

    ciiu_nombres = ciiu_nombres.with_columns(pl.col("ActivityCode").cast(pl.Int64))

    ## Excluimos los ActivityCombination que agregan más de una categoría
    indstat = indstat.filter(
        pl.col("ActivityCombination").is_in(ciiu)
    )

    indstat_empl = indstat.filter(pl.col("Variable") == 'Employees').select("Year", "Country", "ActivityCode", "Activity", "Value")
    indstat_ue = indstat.filter(pl.col("Variable") == 'Establishments').select("Year", "Country", "ActivityCode", "Activity", "Value")

    ## Agregamos columna date
    indstat_empl = indstat_empl.with_columns(
        date = pl.datetime(pl.col("Year"), 1,1,1,1)
    )
    return ciiu_nombres, indstat, indstat_empl


@app.cell
def _():
    return


@app.cell
def _(indstat, pl):
    indstat.filter(
        (pl.col("Year") == 2021) & 
        (pl.col("Variable") == "Employees") &
        (pl.col("Country") == 'Brazil')
    )
    return


@app.cell
def _(indstat_empl):
    indstat_empl
    return


@app.cell
def _(indstat_empl, pl):
    ## Para cada pais, obtenemos el último valor disponible
    indstat_empl.group_by("Country").agg(
        pl.col("Year").max()
    ).group_by("Year").agg(
        # Joins the strings in "string_col" for each group, separated by a comma and space
        concatenated_strings=pl.col("Country").str.join(", ")
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(indstat_empl, pl):
    ### La siguiente expresión define dos planes : 
    ### 1) Calcula el promedio de los valores para una ventana de 5 años antes 
    ### 2) Filtra por el último año disponible para cada país

    indstat_empl_mean = indstat_empl.with_columns(
        ### 1) Calcula el promedio de los valores para una ventana de 5 años antes 
        pl.col(
            "Value"
        ).rolling_mean_by(
            by="date", 
            window_size="5y"
        ).over(
            "Country", "ActivityCode"
        ).cast(pl.Int64).alias("mean")
    ).filter(
        ### 2) Filtra por el último año disponible para cada país
        pl.col("Year") == (pl.max("Year").over("Country") - pl.lit(1))
    )
    indstat_empl_mean
    return (indstat_empl_mean,)


@app.cell
def _():
    return


@app.cell
def _(indstat_empl_mean, pl):
    ## Minimo dataset
    employees_cdata = indstat_empl_mean.select("Country", "ActivityCode", "Value")
    employees_cdata.columns = ["country", "activity", "value"]
    employees_cdata = employees_cdata.with_columns(
        pl.col("value").cast(pl.Int64)
    )
    employees_cdata
    return (employees_cdata,)


@app.cell
def _(employees_cdata):
    employees_cdata["activity"].unique()
    return


@app.cell
def _(employees_cdata, pl):
    ## Datos empleo Honduras
    hnd_empleo = pl.read_csv("datos/empleo_honduras/empleo_honduras.csv")

    ## Renombramos columnas
    hnd_empleo.columns = ["country", "activity", "value"]

    ## Filtramos para las industrias comparables
    hnd_empleo = hnd_empleo.filter(
        pl.col("activity").is_in(employees_cdata["activity"].unique())
    )

    hnd_empleo
    return (hnd_empleo,)


@app.cell
def _(employees_cdata, hnd_empleo, pl):
    ## Reunimos datos
    emp_total_cdata = pl.concat(
        [employees_cdata, hnd_empleo]
    ) 
    emp_total_cdata = emp_total_cdata.with_columns(
        year = pl.lit(2022)
    )

    emp_total_cdata
    return (emp_total_cdata,)


@app.cell
def _(ecomplexity, emp_total_cdata, pl):
    # Calculate complexity for UE HND País
    trade_cols_ue = {'time':'year', 'loc':'country', 'prod':'activity', 'val':'value'}
    emp_cdata = pl.from_pandas(ecomplexity(emp_total_cdata.with_columns(
        year = pl.lit(2022)
    ).to_pandas(), trade_cols_ue))

    ### Normalizamos density
    emp_cdata = emp_cdata.with_columns(
        density_norm = (pl.col("density") - pl.col("density").mean())/pl.col("density").std()
    )

    emp_cdata
    return (emp_cdata,)


@app.cell
def _(pl):
    ### Agregamos datos de población y gdp para verificar el ECI
    national_accounts = pl.read_excel("datos/national_accounts/0140D50E61EB55398C3F7494E5306D8F5C35.xlsx")
    national_accounts_gdp = national_accounts.filter(pl.col("Variable Code") == 'GdpCod').with_columns(pl.col("Year").cast(pl.Int64))
    national_accounts_pop = national_accounts.filter(pl.col("Variable Code") == 'Pop').with_columns(pl.col("Year").cast(pl.Int64))
    national_accounts_pop
    return national_accounts_gdp, national_accounts_pop


@app.cell
def _(emp_cdata, indstat_empl_mean, national_accounts_pop):
    emp_cdata_pop =  indstat_empl_mean.select("Year", "Country").unique().join(
        national_accounts_pop.select("Year", "Country", "Value").rename({"Value" : "pop"}), 
        on = ["Year", "Country"]
    ).join(
        emp_cdata.select("country", "eci").unique(), 
        left_on = "Country", 
        right_on = "country"
    )
    emp_cdata_pop
    return


@app.cell
def _(emp_cdata, indstat_empl_mean, national_accounts_gdp):
    emp_cdata_gdp =  indstat_empl_mean.select("Year", "Country").unique().join(
        national_accounts_gdp.select("Year", "Country", "Value").rename({"Value" : "gdp"}), 
        on = ["Year", "Country"]
    ).join(
        emp_cdata.select("country", "eci").unique(), 
        left_on = "Country", 
        right_on = "country"
    )
    emp_cdata_gdp
    return


@app.cell
def _():
    return


@app.cell
def _(emp_cdata, np, pl):
    ## Calcula promedio ponderado de density, PCI y GO
    ### Define ponderadores
    product_selection_criteria = {
        "Low-hanging Fruit" : {"cog" : 0.25, "pci" : 0.15, "density" : 0.60},
        "Balanced Portfolio" : {"cog" : 0.35, "pci" : 0.15, "density" : 0.50},
        "Long Jumps" : {"cog" : 0.35, "pci" : 0.20, "density" : 0.45},
    }

    ### Definimos portafolio
    portafolio = "Long Jumps"

    ### Creamos score como una media ponderada
    df_portafolios_score = emp_cdata.with_columns(
            emp_cdata.select(
                pl.struct("density_norm", "pci", "cog").map_elements(
                    lambda s: np.average(
                        a = [s["density_norm"], s["pci"], s["cog"]],
                        weights = [
                            product_selection_criteria[portafolio]["density"],
                            product_selection_criteria[portafolio]["pci"], 
                            product_selection_criteria[portafolio]["cog"]
                        ]
                    ), 
                    return_dtype=pl.Float64
                ).alias("score")
            )
        )

    df_portafolios_score
    return (df_portafolios_score,)


@app.cell
def _(ciiu_nombres, df_portafolios_score):
    df_portafolios_score.filter(country="Honduras").filter(
        mcp = 0
    ).select("activity", "score").join(
        ciiu_nombres, 
        left_on = "activity", 
        right_on = "ActivityCode"
    ).sort(
        "score", 
        descending = True
    )
    return


@app.cell
def _(df_portafolios_score):
    df_portafolios_score.select("country", "eci").unique()
    return


@app.cell
def _():
    92/195
    return


@app.cell
def _():
    71/100
    return


@app.cell
def _(ciiu_nombres, emp_cdata):
    emp_cdata.filter(country = "Honduras").filter(mcp=1).join(
        ciiu_nombres, 
        left_on = "activity", 
        right_on = "ActivityCode"
    ).select("rca", "Activity")
    return


if __name__ == "__main__":
    app.run()
