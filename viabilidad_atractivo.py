import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Métricas de Viabilidad y Atractivo

    **Attractiveness**:
    - Capacidad para movilizar FDI (world and region)
    - ⁠Industry growth worldwide (past five years)
    - ⁠Possibility to substitute US imports from Asia (China)
    - ⁠Capacity to create employment among specific groups (women, youth, low-skill)

    **Viability**:
    - Strength in countries like Honduras (RCA in peer group)
    - ⁠⁠Availability of inputs (doble razor, let us talk)
    - Reliance on a constraint or potential constraint (energy, security)
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Industry growth worldwide (past five years)

    Calcularemos el crecimiento de la industria CIIU al calcular el crecimiento en exportaciones de los productos que componen a cada industria.

    De acuerdo a la metodología podemos descomponer la industria CIIU por los productos que la intengra, ponderado por el peso relativo de cada producto en la industria.

    Con tales ponderadores podemos crear con los datos del Atlas de Complejidad Económica un indicador del crecimiento exportador de la industria en el mundo.

    Aquí podemos usar la suma de exportaciones e importaciones para cuantificar una medida de comercio global.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⁠Possibility to substitute US imports from Asia (China)

    Usamos datos de atlas también. Pensemos un poco más como hacerlo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Capacity to create employment among specific groups (women, youth, low-skill)
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Availability of inputs (doble razor, let us talk)

    Además de los datos del Atlas, usaremos los datos de [AI-generated Production Network - AIPNET](https://aipnet.io/) para identificar la cadena de producción de los productos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reliance on a constraint or potential constraint (energy, security)}

    Aquí podemos usar datos de IEA. Los datos están muy agregados, pero es lo mejor que tenemos.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
