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
    import altair as alt
    return alt, pl


@app.cell
def _():
    ### Clasificación 4d 
    cafe = "0901"
    cafe_extractos = "2101"
    return (cafe,)


@app.cell
def _(pl):
    ### Cargamos datos
    aipnet_hs12_4d = pl.read_parquet('datos/atlas_datos/hs12/aipnet_hs12_4d.parquet')
    aipnet_hs12_6d = pl.read_parquet('datos/atlas_datos/hs12/aipnet_hs12_6d.parquet')
    hs12_country_product_year_4 = pl.scan_parquet('datos/atlas_datos/hs12/hs12_country_product_year_4.parquet').with_columns(
        pl.col("product_hs12_code").map_elements(
            lambda x : f"{x:04d}",
            return_dtype=pl.String
        )
    ).filter(
        (
            pl.col("year") == 2024
        )
    ).collect()

    hs12_country_product_year_6 = pl.scan_parquet('datos/atlas_datos/hs12/hs12_country_product_year_6.parquet').with_columns(
        pl.col("product_hs12_code").map_elements(
            lambda x : f"{x:06d}",
            return_dtype=pl.String
        )
    ).filter(
        (
            pl.col("year") == 2024
        )
    ).collect()

    product_hs12 = pl.read_csv('datos/atlas_datos/hs12/product_hs12.csv', ignore_errors=True)
    return (hs12_country_product_year_4,)


@app.cell
def _():


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Importancia del Cafe de Honduras en el mercado mundial
    """)
    return


@app.cell
def _(cafe, hs12_country_product_year_4, pl):
    cafe_bar_data = hs12_country_product_year_4.filter(
        (
             pl.col("product_hs12_code") == cafe   
        ) 
    ).select(
        "country_iso3_code", "export_value", "export_rca"
    ).sort(
        "export_value", descending=True
    ).with_columns(
        pl.col("export_value")/1_000_000_000
    ).head(15)

    cafe_bar_data

    return (cafe_bar_data,)


@app.cell
def _(alt, cafe_bar_data):
    base = alt.Chart(cafe_bar_data).encode(
        x=alt.X('sum(export_value):Q').stack('zero').title("Exportaciones [Miles de Millones US]"),
        y=alt.Y('country_iso3_code:O').sort('-x').title("País"),
        text=alt.Text('export_rca:Q', format='.01f')
    )
    bars = base.mark_bar(
        tooltip=alt.expr("luminance(scale('color', datum.export_rca))")
    ).encode(
        color=alt.Color('export_rca:Q').title("RCA")
    )

    text = base.mark_text(
        align='right',
        dx=-3,
        #color=alt.expr("luminance(scale('color', datum.export_rca)) > 0.5 ? 'black' : 'white'")
    )
    (bars + text).properties(
                    title=alt.TitleParams(
                        "Importancia de Honduras en el Mercado Mundial de Café",
                        subtitle="Atlas de Complejidad, 2024",
                        subtitleColor="gray"
                    )
                )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cadena de Producción
    """)
    return


@app.cell
def _(pl):
    insumos_tree = pl.read_csv("datos/hs12_insumos_tree/hs12_insumos_tree.csv")
    insumos_tree
    return (insumos_tree,)


@app.cell
def _(cafe, hs12_country_product_year_4, insumos_tree, pl):
    def test_boleano(valor):
        if valor==1:
            return "Sí"
        else:
            return "No"
        
    cafe_insumos_tree = insumos_tree.with_columns(
        pl.col("hs2012_code").map_elements(lambda x : f"{x:04d}"), 
        pl.col("hs2012_code_upstream").map_elements(lambda x : f"{x:04d}"), 
        pl.col("Se Exporta").map_elements(lambda x : test_boleano(x)), 
    ).filter(hs2012_code=cafe).join(
        hs12_country_product_year_4.filter(
            country_iso3_code="HND"
        ).select(
            "product_hs12_code", "distance", "pci", "export_rca"
        ), 
        left_on="hs2012_code_upstream", 
        right_on="product_hs12_code"
    )

    cafe_insumos_tree
    return (cafe_insumos_tree,)


@app.cell
def _(cafe_insumos_tree):
    cafe_insumos_tree
    return


@app.cell
def _(alt, cafe_insumos_tree):
    plot_industrias = alt.Chart(cafe_insumos_tree
            ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
            ).encode(
        x=alt.X('distance').scale(zero=False, padding=30).title("Distancia"),
        y=alt.Y('pci').title("PCI").scale(padding=30),#.scale(type ="log"),
        #shape = alt.Shape("mcp:N").title("M"),
        color = alt.Color("Se Exporta:N").title("Especializado"),
        #size = alt.Size("export_rca"),
        tooltip=[
                        alt.Tooltip('Producto Downstream (H12)', title='Producto Downstream (H12)')
        ] 
    )

    labels = alt.Chart(cafe_insumos_tree
            ).mark_circle(
                opacity=0.99,
                stroke='black',
                strokeWidth=1.2,
                strokeOpacity=0.9, 
                size=180,     
            ).encode(
        x=alt.X('distance').scale(zero=False, padding=30).title("Distancia"),
        y=alt.Y('pci').title("PCI").scale(padding=30),#.scale(type ="log"),
        #shape = alt.Shape("mcp:N").title("M"),
        #size = alt.Size("export_rca"),
        tooltip=[
                        alt.Tooltip('Producto Downstream (H12)', title='Producto Downstream (H12)')
        ] 
    ).mark_text(
        align='left',
        baseline='middle',
        dx=10
    ).encode(
        text='Producto Downstream (H12)'
    )

    (plot_industrias + labels).properties(
        title=alt.TitleParams(
            "Insumos del Café",
            subtitle="Distancia-PCI. Atlas de Complejidad Económica, 2024",
            subtitleColor="gray"
        ),
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


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
