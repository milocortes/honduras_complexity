import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Preparación del entorno
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este bloque carga la librería `marimo`, que será utilizada para estructurar el análisis en un entorno de notebook reactivo. El alias `mo` facilita el uso posterior de componentes interactivos y elementos de visualización.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este bloque prepara el entorno de trabajo cargando las dependencias necesarias para procesar datos, calcular indicadores de complejidad económica y presentar resultados en formatos gráficos y tabla.

    En particular, `polars` se utiliza para la manipulación eficiente de datos; `pandas` para trabajar con estructuras tabulares ampliamente compatibles; `matplotlib.pyplot` para generar gráficos básicos; `numpy` para operaciones numéricas; `ecomplexity` y `proximity` para estimar medidas de complejidad económica y relaciones de cercanía; `altair` para construir visualizaciones declarativas; `great_tables` para producir tablas formateadas; y `polars.selectors` para facilitar la selección de columnas dentro del flujo de procesamiento.
    """)
    return


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
    ## 2. Carga y preparación de datos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Exploración de tabla datos de OCDE SBS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este bloque carga y depura la base Structural Business Statistics (SBS) de la OCDE a partir del archivo `datos/oecd/ocde_sbs.parquet`. La consulta inicial se define en modo *lazy* con `polars`, lo que permite filtrar la variable `Employees` y seleccionar únicamente las columnas relevantes antes de materializar los datos.

    Posteriormente, la base se convierte a `pandas` y se filtra para conservar únicamente actividades económicas codificadas a cuatro dígitos del CIIU. Para ello, se identifican los códigos `ACTIVITY` con una letra inicial de sección y cuatro caracteres numéricos posteriores; luego, la letra se almacena en la variable `seccion` y el código de actividad se estandariza dejando solo los cuatro dígitos.

    Como resultado, se obtiene el data frame `df`, que contiene observaciones de empleo por país, actividad económica CIIU a cuatro dígitos, clase de tamaño de empresa y año. Esta base depurada sirve como insumo para los pasos posteriores de agregación, análisis sectorial o cálculo de indicadores.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Análisis de cobertura de los datos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En el siguiente bloque se construye un resumen anual de la cobertura de la base `df`, contando el número de registros disponibles por sección de actividad económica. Para ello, identifica el rango de años en `TIME_PERIOD`, filtra la base año por año y calcula la frecuencia de observaciones asociadas a cada valor de `seccion`.

    Como resultado, se obtiene el DataFrame `resumen_seccion_conteos`, donde cada fila representa un año, las columnas de secciones indican el número de registros observados por sección económica y la columna `Total` resume el número total de registros disponibles en ese año. Esta tabla sirve como control de calidad para evaluar la cobertura temporal y sectorial de la información antes de continuar con el análisis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La tabla `resumen_seccion_conteos` permite evaluar la cobertura anual de la base por sección económica. Los valores `NaN` indican que en ese año no se encontraron registros para una sección específica, mientras que la columna `Total` resume el número total de observaciones disponibles para cada año.

    Aunque los años 2021 y 2022 presentan un mayor número total de registros, 2019 puede considerarse el año con mejor cobertura sectorial balanceada dentro del período reciente. Esto se debe a que combina un volumen alto de observaciones con una distribución más estable entre secciones económicas, evitando que la cobertura esté concentrada en pocas secciones específicas.

    Además, al ser un año reciente previo a las posibles alteraciones observadas después de 2020, ofrece una base más adecuada para realizar comparaciones sectoriales y continuar el análisis con mayor consistencia.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La siguiente visualización con presenta los valores de la tabla anterior en forma de mapa de calor, usando una transformación logarítmica para facilitar la comparación visual entre sectores con distintos niveles de empleo.

    El gráfico resultante permite revisar la distribución temporal y sectorial del empleo agregado, identificar años con mayor cobertura efectiva y detectar secciones económicas con valores relativamente altos o bajos dentro de la base.
    """)
    return


@app.cell
def _(df, np, plt):
    ## Resumen de empleo total por seccion

    resumen_empleo_anios = df.groupby(
        ["TIME_PERIOD", "seccion"]
    ).agg({"OBS_VALUE": "sum"}).reset_index().pivot(
        index="TIME_PERIOD",
        columns="seccion",
        values="OBS_VALUE"
    )

    plt.figure(figsize=(10, 8))

    plt.imshow(np.log(resumen_empleo_anios.to_numpy()))

    plt.xticks(
        ticks=np.arange(len(resumen_empleo_anios.columns)),
        labels=resumen_empleo_anios.columns
    )

    plt.yticks(
        ticks=np.arange(len(resumen_empleo_anios.index)),
        labels=resumen_empleo_anios.index
    )

    plt.xlabel("Sección económica")
    plt.ylabel("Año")
    plt.title("Empleo agregado por año y sección económica")
    plt.colorbar(label="Logaritmo del empleo agregado")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A continuación, se construye una medida de cobertura por país y año a partir del número de actividades económicas a cuatro dígitos con empleo positivo. Para ello, primero se filtra la base conservando únicamente observaciones con `OBS_VALUE > 0`, lo que permite identificar actividades con empleo efectivamente reportado.

    Luego, la información se agrupa por `TIME_PERIOD`, `REF_AREA` y `ACTIVITY` para consolidar los registros de cada actividad dentro de un mismo país y año. Posteriormente, se cuenta el número de actividades distintas por país y año, generando el DataFrame `cobertura_pais_anio`. En esta tabla, la variable `cobertura` indica cuántas actividades económicas presentan empleo positivo para cada combinación país-año.

    A partir de esta tabla, se construye una matriz donde las filas representan países, las columnas representan años y los valores corresponden al número de actividades económicas con empleo positivo. Esta matriz se visualiza mediante un mapa de calor, lo que permite revisar de manera directa la evolución de la cobertura de la base en el tiempo y entre países.
    """)
    return


@app.cell
def _(df):
    # Calcula cobertura por año, país y sección
    # Cobertura = número de actividades económicas con empleo positivo
    cobertura_pais_anio = (
        df.query("OBS_VALUE > 0")
        .groupby(["TIME_PERIOD", "REF_AREA", "ACTIVITY"])
        .agg({"OBS_VALUE": "sum"})
        .reset_index()
        .groupby(["TIME_PERIOD", "REF_AREA"])
        .agg({"ACTIVITY": "count"})
        .reset_index()
        .rename(columns={"ACTIVITY": "cobertura"})
    )

    cobertura_pais_anio
    return (cobertura_pais_anio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El mapa de calor muestra que la cobertura no es homogénea durante todo el período. Antes de 2008, la información aparece de forma limitada y concentrada en pocos países. A partir de 2008, la cobertura aumenta considerablemente y se observa una mayor cantidad de países con un número alto de actividades reportadas. Sin embargo, los años posteriores a 2020 muestran cambios importantes en la cobertura, con reducciones en el número de actividades económicas reportadas para la mayoría de países.

    Con base en estos resultados, 2019 se selecciona como año de referencia porque combina una cobertura alta con una estructura relativamente estable entre países.
    """)
    return


@app.cell
def _(cobertura_pais_anio, np, plt):
    # Matriz país-año para el mapa de calor

    matriz_cobertura_pais_anio = (
        cobertura_pais_anio
        .pivot(
            index="REF_AREA",
            columns="TIME_PERIOD",
            values="cobertura"
        )
        .sort_index()
    )

    plt.figure(figsize=(14, 10))

    plt.imshow(
        matriz_cobertura_pais_anio.to_numpy(),
        aspect="auto"
    )

    plt.xticks(
        ticks=np.arange(len(matriz_cobertura_pais_anio.columns)),
        labels=matriz_cobertura_pais_anio.columns,
        rotation=90
    )

    plt.yticks(
        ticks=np.arange(len(matriz_cobertura_pais_anio.index)),
        labels=matriz_cobertura_pais_anio.index
    )

    plt.xlabel("Año")
    plt.ylabel("País")
    plt.title("Cobertura de actividades económicas con empleo positivo por país y año")

    plt.colorbar(label="Número de actividades con empleo positivo")

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.3 Carga de tabla de actividades económicas depurada Honduras vs OCDE SBS
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este bloque carga una selección depurada de actividades económicas desde el archivo `datos/recodificacion/seleccion_pedro.csv`. La base se filtra para conservar únicamente las clases marcadas con `incluye == 1`, y se mantienen las variables `clase_codigo` y `clase_titulo`.

    Esta selección excluye actividades con baja comparabilidad entre países. En particular, se retiran actividades que aparecen listadas únicamente en la información de Honduras o actividades reportadas por un número muy reducido de países de la OCDE. Con ello, se define un universo de clases CIIU más consistente para el análisis comparativo. Debido a lo anterior, salen del análisis las secciones: A, B (parcialmente), K, O, P, Q, R, S (parcialmente), T y U.

    Posteriormente, los códigos de clase CIIU se estandarizan a un formato de cuatro dígitos mediante ceros a la izquierda cuando es necesario. Esta transformación asegura que los códigos de la selección sean compatibles con los códigos de actividad utilizados en la base principal.

    Como resultado, se obtiene una tabla con las actividades seleccionadas y sus títulos, así como una lista de códigos CIIU que será utilizada para filtrar o delimitar el universo de actividades económicas consideradas en el análisis.

    Como resultado, se tiene un dataframe con 305 actividades económicas a cuatro dígitos.
    """)
    return


@app.cell
def _(pd):
    ### Cargamos selección de industrias 
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Carga de tabla de actividades económicas transables y prueba de cobertura
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El siguiente bloque de código construye una medida de cobertura de actividades económicas transables para el año 2019. Para ello, se carga una tabla de recodificación que permite identificar los códigos CIIU Rev. 4 asociados al conjunto de actividades clasificadas como transables.

    La definición de actividades transables se toma del archivo `datos/resumen_razones_transables.csv`, conservando únicamente aquellas con `razon_emp_transables > 0`, dando como resultado 263 actividades transables. Luego, estos códigos se cruzan con la tabla de recodificación y se estandarizan a cuatro dígitos para asegurar su compatibilidad con la variable `ACTIVITY` de la base principal.

    Posteriormente, la base `df` se filtra al año 2019, a las actividades transables seleccionadas y a observaciones con `OBS_VALUE > 0`. La información se agrega por país y actividad económica para consolidar el empleo observado, y luego se cuenta el número de actividades transables reportadas por cada país.

    Como resultado, se obtiene el DataFrame `df_actividades_transables`, donde la columna `ACTIVITY` expresa la proporción de actividades transables con empleo positivo reportadas por cada país respecto al total de actividades transables consideradas. Esta medida permite evaluar qué países cuentan con suficiente cobertura dentro del universo de actividades transables antes de realizar el cálculo de complejidad económica.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se observa que varios países presentan una cobertura alta, como `ROU`, `DEU`, `HRV`, `LVA`, `ITA`, `NOR`, `ESP` y `KOR`, con valores cercanos o superiores al 80% del total de actividades transables. Esto indica que estos países reportan información para una parte amplia del universo transable, lo que favorece su uso en análisis comparativos.

    En contraste, países como `EST`, `JPN`, `CRI` y `LUX` presentan coberturas reducidas. Estos casos reportan una proporción baja de actividades transables con empleo positivo, por lo que su inclusión en comparaciones debe evaluarse con cuidado. Una cobertura limitada puede afectar la representatividad del análisis y generar resultados menos comparables frente a países con mayor disponibilidad de información.
    """)
    return


@app.cell
def _(df_actividades_transables):
    df_actividades_transables.sort_values(by="ACTIVITY", ascending=False).plot.bar()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2.5 Construcción de tabla de datos final
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Este bloque define los insumos para el análisis de complejidad económica a partir de una muestra de países con cobertura suficiente en actividades transables. La cobertura se evalúa con base en la proporción de actividades transables con empleo positivo, y se seleccionan únicamente los países que superan el umbral definido por `umbral = 0.69` para el año `2019`.

    La base principal se filtra para conservar observaciones del año de análisis, actividades incluidas en la selección depurada, actividades clasificadas como transables, países dentro de la muestra y registros correspondientes al total de la actividad económica (`SIZE_CLASS == '_T'`). Además, se excluyen manualmente cuya inclusión podría afectar la comparabilidad del ejercicio.

    Como resultado inicial, se obtiene una base de empleo por país y actividad económica con las variables `TIME_PERIOD`, `REF_AREA`, `ACTIVITY` y `OBS_VALUE`. A partir de esta base se identifica el conjunto de actividades efectivamente disponible en la muestra, que sirve como referencia para incorporar fuentes adicionales. Luego de los filtros quedan 213 actividades económicas.

    Posteriormente, se cargan las bases de Honduras, El Salvador y Ecuador desde archivos externos. En cada caso, los códigos de actividad se estandarizan al formato CIIU de cuatro dígitos y se filtran para conservar únicamente actividades presentes en la muestra principal. Estas bases se concatenan con la información previamente filtrada, permitiendo incorporar países adicionales al análisis bajo el mismo universo de actividades.

    El resultado final es el DataFrame `insumos_complejidad`, que contiene una base consolidada de empleo por país y actividad económica para 2019.
    """)
    return


@app.cell
def _(ciiu_seleccion_pedro, ciiu_transable, df, df_actividades_transables, pd):
    ### Haremos el análisis de complejidad modificando la muestra de paises de acuerdo al umbral de la razón de actividades que reportan empleo vs total de actividades transables

    ### Que actividades no tenemos datos para los paises?
    ciiu_na = [
        "1910", # Fabricación de productos de hornos de coque
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

    insumos_complejidad
    return anio_analisis, insumos_complejidad, paises_muestra


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Cálculos de complejidad económica
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    El cálculo de complejidad económica utiliza como insumo la base consolidada `insumos_complejidad`. Para adaptar los datos a la función `ecomplexity()`, se define el diccionario `trade_cols`, que identifica las columnas correspondientes al año, país, actividad económica y valor observado.

    La función `ecomplexity()` construye la matriz país-actividad a partir del empleo observado y estima indicadores de complejidad económica y cercanía productiva. El resultado se convierte a formato `polars` y se eliminan observaciones con valores nulos para conservar únicamente combinaciones con indicadores completos.

    Posteriormente, se crea la variable `distance` como `1 - density`. Esta transformación permite interpretar la cercanía productiva en sentido inverso: valores bajos de `distance` indican actividades más cercanas a las capacidades existentes de un país, mientras que valores altos indican actividades más alejadas.

    El resultado es el DataFrame `cdata`, que contiene indicadores de complejidad económica por país y actividad. Esta tabla sirve como base para identificar actividades con mayor o menor complejidad, así como oportunidades productivas más cercanas o más distantes para cada país.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    La matriz de proximidad se calcula a partir de la base consolidada `insumos_complejidad` y la definición de columnas contenida en `trade_cols`. La función `proximity()` estima la cercanía entre pares de actividades económicas según su patrón de coocurrencia en los países de la muestra.

    El resultado se conserva en formato de pares de actividades, con las variables `ACTIVITY_1` y `proximity`, y luego se transforma a una matriz cuadrada mediante `pivot()`. En esta matriz, cada fila y cada columna corresponden a una actividad económica, y cada celda representa la proximidad estimada entre ambas actividades.

    La matriz final se guarda en `output/proximidades/proximidades_ocde_data.csv` y queda disponible como `prox_df`. Este archivo sirve como insumo para analizar la estructura de relaciones entre actividades económicas, identificar sectores cercanos entre sí y construir métricas o visualizaciones basadas en la red productiva.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Análisis de resultados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 Rankings generales
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A partir de los indicadores calculados en `cdata`, se construye un ranking de actividades económicas según el índice de complejidad de producto o actividad (`pci`). La variable `pci` permite ordenar las actividades de acuerdo con su nivel relativo de complejidad dentro de la matriz país-actividad utilizada en el análisis.

    El resultado es el DataFrame `pci_industrias`, que contiene las actividades económicas ordenadas de mayor a menor complejidad. Esta tabla sirve para identificar las industrias más complejas del universo analizado y para relacionar los resultados de complejidad con la clasificación sectorial CIIU. La industria más compleja del listado es "fabricación de vehículos automotores", mientras que la menos compleja es "fabricación de otros productos de madera".
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A partir de los indicadores calculados en `cdata`, se construye un ranking de países según el índice de complejidad económica (`eci`). Para ello, se selecciona una observación única por país, se ordenan los valores de `eci` de mayor a menor y se asigna una posición ordinal mediante `rank()`.

    El resultado es el DataFrame `eci_paises`, que contiene el código de país (`REF_AREA`), el valor del índice de complejidad económica y su posición en el ranking. Esta tabla permite identificar qué países presentan estructuras productivas más complejas dentro de la muestra analizada. El país que ocupa el primer lugar en el ranking es Alemania, mientras que Honduras ocupa la posición 20 (el penúltimo de la lista).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Pruebas de consistencia
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Complejidad vs PIB per cápita
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Se cargan datos del PIB y población para el año 2019.

    La población total se calcula como la suma de la población rural y urbana. Posteriormente, esta información se integra con la base de PIB usando las variables `Year`, `Nation` e `iso_code3` como llaves de unión.

    El resultado es el DataFrame `gdp`, que contiene el PIB, la población total y el PIB per cápita para cada país disponible. El PIB per cápita se calcula dividiendo el PIB expresado en dólares entre la población total, por lo que se multiplica `gdp_mmm_usd` por `1_000_000` antes de realizar la división.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    El siguiente gráfico muestra la correlación positiva que existe entre el `eci` y el `PIB per cápita`. Honduras, entre las economías pares y aspiracionales comparadas, ocupa la última posición en ingreso por habitante y la penúltima en complejidad económica.
    """)
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
def _(mo):
    mo.md(r"""
    #### Complejidad vs PIB per cápita
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se carga la base `growth_proj_eci_rankings.csv` proveniente de los datos del Atlas de Complejidad Económica. La información se filtra para conservar únicamente el año 2019, que corresponde al año de referencia utilizado en el análisis.

    El resultado es el DataFrame `atlas`, que contiene rankings e indicadores de complejidad económica reportados por el Atlas para 2019. Esta base se utiliza como referencia externa para contrastar los resultados obtenidos con la metodología aplicada sobre los datos de empleo por actividad económica.
    """)
    return


@app.cell
def _(pl):
    ### Carga datos del atlas del ranking
    atlas = pl.read_csv("datos/atlas_datos/growth_proj_eci_rankings.csv").filter(
        pl.col("year") == 2019
    )
    atlas
    return (atlas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se construye una base comparativa que integra dos medidas de complejidad económica: el `ECI` calculado a partir de empleo en actividades económicas de `OECD SBS` y el `ECI` reportado por el Atlas de Complejidad Económica. Ambas medidas se integran usando el código ISO3 del país como llave.

    La información se transforma a formato largo mediante `unpivot()`, lo que permite representar ambas fuentes de `ECI` dentro de una misma estructura de datos. Posteriormente, se incorpora el PIB per cápita calculado previamente, generando una base con país, fuente del indicador, valor de `ECI` y nivel de ingreso por persona.

    La visualización compara el PIB per cápita con el `ECI` para ambas fuentes. Los resultados muestran que los cálculos de complejidad usando información de empleo guardan consistencia con los que se nutren de datos de comercio exterior. Aunque los primeros, de manera sistemática arrojan puntajes más bajos.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La complejidad económica se calcula también con datos de exportaciones del Atlas, usando la misma muestra de países considerada en el análisis principal. Para ello, se carga la base `hs92_country_product_year_4.parquet` y se filtra a los países seleccionados por cobertura, incorporando además a Honduras, Ecuador y El Salvador. Lo que deja la muestra final de 21 economías.

    La función `ecomplexity()` se aplica sobre una matriz país-producto construida con valores de exportación. El diccionario `trade_cols_export` define las columnas de año, país, producto y valor exportado requeridas por la función.

    El resultado se depura eliminando valores nulos y se reduce a una observación única por país. La variable `eci` se renombra como `eci_atlas`, generando el DataFrame `cdata_export`, que contiene el índice de complejidad económica estimado con exportaciones del Atlas para la muestra de países analizada.

    Esta estimación resultante permite comparar el indicador de complejidad basado en empleo sectorial con una medida alternativa basada en exportaciones. La comparación ayuda a evaluar si la estructura productiva capturada por el empleo guarda relación con la complejidad observada en los patrones comerciales internacionales.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    En esta etapa se reúnen dos estimaciones de complejidad económica para la misma muestra de países: el `ECI` calculado con datos de empleo sectorial y el `ECI` calculado con datos de exportaciones del Atlas. Para cada fuente se construye además un ranking ordinal, lo que permite comparar la posición relativa de cada país bajo ambos enfoques.

    La tabla resultante, `cdata_atlas_estimado`, contiene el código del país, el valor de `eci`, su ranking basado en empleo, el valor de `eci_atlas` y su ranking basado en exportaciones. Esta estructura permite evaluar de forma directa si ambos métodos ordenan a los países de manera similar.

    La figura compara ambos rankings mediante un diagrama de dispersión. El eje horizontal muestra el ranking de complejidad basado en `OECD SBS`, mientras que el eje vertical muestra el ranking derivado del Atlas. Cada punto representa un país y se incorpora una línea de identidad como referencia visual. Los países ubicados cerca de esa línea presentan posiciones similares en ambos rankings, mientras que los casos alejados reflejan discrepancias entre la complejidad estimada a partir del empleo y la complejidad estimada a partir de exportaciones.

    El resultado corrobora la existencia de una correlación positiva entre ambos rankings.
    """)
    return


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
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4.3 Portafolios de política
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se definen criterios de priorización para seleccionar actividades económicas bajo tres estrategias de portafolio: `Low-hanging Fruit`, `Balanced Portfolio` y `Long Jumps`. Cada estrategia asigna pesos distintos a las variables `density`, `pci` y `cog`, permitiendo adaptar la priorización según el balance deseado entre cercanía productiva, complejidad y oportunidad de crecimiento.

    La estrategia `Low-hanging Fruit` prioriza actividades cercanas a las capacidades existentes, dando mayor peso a `density`. `Balanced Portfolio` distribuye los pesos de forma más equilibrada entre cercanía, complejidad y oportunidad. `Long Jumps` favorece actividades con mayor complejidad y potencial de crecimiento, aun cuando puedan estar más alejadas de la estructura productiva actual.

    La función `calcula_score()` aplica estos pesos a un conjunto de actividades filtradas para Honduras (`REF_AREA == "HND"`) y restringidas a actividades donde `mcp == 0`. Esto permite enfocar la priorización en actividades que el país aún no desarrolla de manera revelada dentro de la matriz país-actividad.

    El puntaje final se calcula como una media ponderada de `density_norm`, `pci` y `cog`, usando los pesos definidos para el portafolio seleccionado. El resultado es una tabla con un score por actividad económica, que sirve para ordenar y comparar oportunidades productivas según la estrategia de selección utilizada.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La función `calcula_density_z()` normaliza la variable `density` y calcula una medida complementaria de distancia productiva. La normalización se realiza mediante una transformación tipo *z-score*, restando el promedio de `density` y dividiendo entre su desviación estándar.

    La variable resultante, `density_norm`, permite comparar actividades en una escala común, donde valores positivos representan densidades superiores al promedio y valores negativos representan densidades inferiores al promedio. Además, se calcula `distance` como `1 - density`, lo que permite expresar la cercanía productiva en sentido inverso.

    El resultado de esa función es un DataFrame con las columnas adicionales `density_norm` y `distance`. Esta transformación prepara los indicadores para construir scores de priorización y comparar actividades económicas bajo distintos criterios de selección.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La función `calcula_density_z()` se aplica al DataFrame `cdata` para generar una versión normalizada de los indicadores de complejidad. El resultado se almacena en `cdata_norm`.

    Esta transformación incorpora la variable `density_norm`, que expresa la densidad productiva en una escala estandarizada, y recalcula `distance` como `1 - density`. La normalización permite comparar la densidad con otros indicadores usados en la priorización de actividades económicas, especialmente dentro de los scores definidos para los distintos portafolios.

    El DataFrame `cdata_norm` conserva la información de complejidad económica original y añade variables preparadas para el cálculo de oportunidades productivas.
    """)
    return


@app.cell
def _(calcula_density_z, cdata):
    cdata_norm = calcula_density_z(cdata)
    cdata_norm
    return (cdata_norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se calcula el ranking de actividades económicas priorizadas para el portafolio `bp`, correspondiente a `Balanced Portfolio`. Este portafolio combina cercanía productiva, complejidad y oportunidad de crecimiento mediante los pesos definidos previamente en `product_selection_criteria`.

    La función `calcula_score()` se aplica sobre `cdata_norm`, restringiendo el cálculo a Honduras y a actividades donde `mcp == 0`. Esto enfoca la priorización en actividades que aún no aparecen como desarrolladas de manera revelada en la matriz país-actividad.

    Luego, el resultado se cruza con el diccionario CIIU para incorporar el nombre de cada actividad y con la base original para agregar la sección económica. Las actividades se ordenan de mayor a menor puntaje, se asigna un ranking ordinal y se conservan las 10 primeras posiciones.

    El resultado es una tabla con las principales oportunidades productivas para Honduras bajo la estrategia `Balanced Portfolio`. Esta salida permite identificar actividades que combinan una cercanía razonable con niveles relativamente atractivos de complejidad y oportunidad.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La función `obten_ranking()` automatiza la construcción de rankings de actividades económicas para una estrategia de portafolio específica. A partir del identificador del portafolio, calcula el score correspondiente con `calcula_score()`, incorpora el nombre de cada actividad mediante el cruce con `mapp_ciiu` y agrega la etiqueta desde la base original.

    Las actividades se ordenan de mayor a menor puntaje y se asigna un ranking ordinal. Posteriormente, se conservan las primeras 20 actividades mejor posicionadas y se renombran las columnas de salida de acuerdo con el portafolio seleccionado.

    El resultado es una tabla con el ranking, el código CIIU y el nombre de las actividades priorizadas. Esta función permite generar salidas comparables para distintos portafolios, como `lhf`, `bp` o `lj`, manteniendo una estructura uniforme para el análisis de oportunidades productivas.
    """)
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
def _(mo):
    mo.md(r"""
    A continuación, se muestra el top 20 de cada uno de los portafolios:
    """)
    return


@app.cell
def _(cdata_norm, mapp_portafolios, obten_ranking, pl):
    ### Creamos portafolios 
    portafolios_categorias = [obten_ranking(portafolio,  cdata_norm) for portafolio in mapp_portafolios]
    portafolios_categorias = pl.concat(portafolios_categorias,  how = "horizontal")
    portafolios_categorias
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    La figura presenta un diagrama Distancia-PCI para Honduras a partir de los indicadores de complejidad calculados con datos de empleo de `OECD SBS` para 2019. La visualización se restringe a actividades con `rca > 0`, lo que permite concentrar el análisis en actividades con presencia relativa positiva en el país.

    En el eje horizontal se muestra la variable `distance`, que mide la distancia productiva de cada actividad respecto a las capacidades existentes. Valores más bajos indican actividades más cercanas a la estructura productiva actual. En el eje vertical se presenta el `pci`, que representa la complejidad relativa de cada actividad económica.

    El color y el tamaño de los puntos reflejan el valor de `rca` en escala logarítmica, mientras que la forma identifica el estado de `mcp`. De esta manera, el gráfico permite analizar simultáneamente cercanía productiva, complejidad económica e intensidad relativa de las actividades observadas.

    Esta visualización sirve para identificar actividades existentes en Honduras que combinan mayor complejidad, cercanía a las capacidades actuales y una presencia relativa más fuerte. También funciona como diagnóstico inicial para comparar la estructura productiva observada con las oportunidades priorizadas en los portafolios.
    """)
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
    mo.md("""
    El siguiente gráfico sirve como referencia comparativa frente al caso de Honduras, ya que permite observar cómo se distribuyen las actividades de una economía con mayor complejidad en el espacio Distancia-PCI.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 Análisis en el margen intensivo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se construye un diagrama Distancia-PCI intensivo para Honduras, restringiendo la muestra a actividades con `rca > 0` y `mcp == 1`. Esta selección permite concentrar el análisis en actividades donde el país ya muestra una presencia productiva revelada dentro de la matriz país-actividad.

    La figura compara la distancia productiva de cada actividad con su nivel de complejidad (`PCI`). La distancia se representa en el eje horizontal y el `PCI` en el eje vertical. Además, se incorpora una línea horizontal en el valor del `ECI` de Honduras, que funciona como referencia para distinguir actividades con complejidad superior o inferior al nivel agregado del país.

    Antes de construir la figura, se ajusta la clasificación sectorial para agrupar las divisiones de fabricación de prendas de vestir y fabricación de productos textiles bajo la categoría `Industria Textil`. Esta reclasificación permite representar de forma más clara un conjunto de actividades relevantes dentro de la estructura productiva hondureña.

    En la visualización, el color identifica la sección económica ajustada y el tamaño del punto representa el empleo observado en escala logarítmica. El resultado permite identificar actividades existentes en Honduras que combinan mayor complejidad, cercanía productiva y peso laboral, aportando una lectura de las capacidades actuales del país y su posición relativa frente al nivel de complejidad nacional.
    """)
    return


@app.cell
def _(alt, cdata, mapp_ciiu, pd, pl):
    cdata_intensivo = cdata.filter(
        (pl.col("REF_AREA") == "HND") & 
        (pl.col("rca") > 0) & 
        (pl.col("mcp") == 1)
    )

    hline = alt.Chart().mark_rule(color="red").encode(
        y=alt.datum(-1.14)
    )

    ciiu_textil = pd.read_csv("datos/recodificacion/seleccion_pedro.csv").query("incluye == 1")

    ciiu_textil["clase_codigo"] = ciiu_textil["clase_codigo"].apply(lambda x: f"{x:04}")

    ciiu_textil = ciiu_textil[
        ["clase_codigo", "clase_titulo", "seccion_codigo", "seccion_titulo", "division_titulo"]
    ]

    ciiu_textil.loc[
        ciiu_textil["division_titulo"] == "Fabricación de prendas de vestir",
        "seccion_titulo"
    ] = "Industria Textil"

    ciiu_textil.loc[
        ciiu_textil["division_titulo"] == "Fabricación de productos textiles",
        "seccion_titulo"
    ] = "Industria Textil"

    ciiu_textil = pl.from_pandas(ciiu_textil)

    plot_intensivo = alt.Chart(
        cdata_intensivo
        .join(
            mapp_ciiu,
            left_on="ACTIVITY", 
            right_on="codigo"
        )
        .join(
            ciiu_textil,
            left_on="ACTIVITY", 
            right_on="clase_codigo"
        )
    ).mark_circle(
        opacity=0.99,
        stroke="black",
        strokeWidth=1.2,
        strokeOpacity=0.9,
        size=180
    ).encode(
        x=alt.X("distance").scale(zero=False).title("Distancia"),
        y=alt.Y("pci").title("PCI").scale(
            domain=(
                cdata_intensivo["pci"].min() - 1.5,
                cdata_intensivo["pci"].max() + 0.4
            )
        ),
        color=alt.Color("seccion_titulo").title("Sección"),
        size=alt.Size("OBS_VALUE").scale(type="log").title("Empleo"),
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
    return cdata_intensivo, ciiu_textil, hline


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    La figura relaciona el empleo observado con la complejidad de las actividades intensivas de Honduras. La muestra se restringe a actividades con `rca > 0` y `mcp == 1`, lo que permite concentrar el análisis en actividades donde el país ya presenta una capacidad productiva revelada.

    El eje horizontal muestra `OBS_VALUE`, interpretado como empleo observado, en escala logarítmica. Esta transformación facilita comparar actividades con tamaños laborales muy distintos. El eje vertical muestra el `PCI`, que representa la complejidad relativa de cada actividad económica.

    El color de los puntos identifica la sección económica ajustada y el tamaño representa el `RCA` en escala logarítmica. Además, se incorpora una línea horizontal asociada al `ECI` de Honduras, que permite comparar la complejidad de cada actividad con el nivel agregado de complejidad del país.

    El resultado permite identificar actividades que combinan una base laboral existente y relativamente amplia con mayor complejidad económica, así como actividades intensivas en empleo pero con menor complejidad relativa.
    """)
    return


@app.cell
def _(alt, cdata_intensivo, ciiu_textil, hline, mapp_ciiu):
    #### Plot intensivo Logaritmo Empleo vs PCI

    plot_intensivo_empleo = alt.Chart(
        cdata_intensivo.join(
            mapp_ciiu,
            left_on="ACTIVITY", 
            right_on="codigo"
        ).join(
            ciiu_textil,
            left_on="ACTIVITY", 
            right_on="clase_codigo"
        )
    ).mark_circle(
        opacity=0.99,
        stroke="black",
        strokeWidth=1.2,
        strokeOpacity=0.9, 
        size=180,     
    ).encode(
        x=alt.X("OBS_VALUE").scale(type="log", zero=False).title("Empleo"),
        y=alt.Y("pci").title("PCI").scale(
            domain=(
                cdata_intensivo["pci"].min() - 1.5,
                cdata_intensivo["pci"].max() + 0.4
            )
        ),
        color=alt.Color("seccion_titulo").title("Sección"),
        size=alt.Size("rca").scale(type="log").title("RCA"),
        tooltip=["nombre_actividad", "division_titulo", "OBS_VALUE"]
    )

    plot_intensivo_empleo = plot_intensivo_empleo + hline

    plot_intensivo_empleo.properties(
        title=alt.TitleParams(
            "Logaritmo de empleo vs PCI (Intensivo)",
            subtitle="Honduras. Datos de Empleo de OECD SBS 2019",
            subtitleColor="gray"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 Análisis en el margen extensivo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se consolida la información de los rankings de actividades económicas generados para cada estrategia de portafolio. La función `agrega_col()` incorpora una columna llamada `portafolio`, que permite identificar la estrategia asociada a cada recomendación.

    Para cada portafolio definido en `mapp_portafolios`, se ejecuta `obten_ranking()` usando como insumo `cdata_norm`. Los resultados se renombran a una estructura común con las columnas `ranking`, `clase_codigo` y `clase_titulo`. Esta estandarización permite concatenar los rankings de todas las estrategias en una única tabla.

    Posteriormente, se carga la selección depurada de actividades desde `seleccion_pedro.csv`, conservando únicamente las actividades incluidas en el análisis. Los códigos CIIU se formatean a cuatro dígitos para garantizar consistencia en los cruces. De esta tabla se extraen las variables `seccion_codigo` y `seccion_titulo`, que identifican la sección económica de cada actividad.

    Finalmente, la tabla `portafolios` se cruza con la información sectorial mediante `clase_codigo`. El resultado es una base consolidada que permite comparar las actividades priorizadas por cada estrategia de portafolio junto con su posición en el ranking y su clasificación económica.
    """)
    return


@app.cell
def _(cdata_norm, mapp_portafolios, obten_ranking, pd, pl):
    def agrega_col(df, columna): 
        return df.with_columns(
            portafolio=pl.lit(columna)
        )

    portafolios = pl.concat(
        [
            agrega_col(
                obten_ranking(portafolio, cdata_norm).rename(
                    {
                        f"rank_{portafolio}": "ranking", 
                        f"ciiu_{portafolio}": "clase_codigo", 
                        f"nombre_actividad_{portafolio}": "clase_titulo", 
                    }
                ), 
                portafolio
            )
            for portafolio in mapp_portafolios
        ]  
    )

    ciiu_secciones = pd.read_csv("datos/recodificacion/seleccion_pedro.csv").query("incluye == 1")

    ciiu_secciones["clase_codigo"] = ciiu_secciones["clase_codigo"].apply(lambda x: f"{x:04}")

    ciiu_secciones = pl.from_pandas(
        ciiu_secciones[["clase_codigo", "seccion_codigo", "seccion_titulo"]]
    )

    portafolios = portafolios.join(
        ciiu_secciones,
        on="clase_codigo"
    )

    portafolios
    return (portafolios,)


@app.cell
def _(cdata_norm, pd, pl):
    ### Preparamos df para visualizacion
    ciiu_actividades = pd.read_csv("datos/recodificacion/seleccion_pedro.csv").query("incluye == 1")

    ciiu_actividades["clase_codigo"] = ciiu_actividades["clase_codigo"].apply(lambda x: f"{x:04}")

    ciiu_actividades = pl.from_pandas(
        ciiu_actividades[
            [
                "clase_codigo",
                "clase_titulo",
                "seccion_codigo",
                "seccion_titulo"
            ]
        ]
    )

    cdata_hnd = (
        cdata_norm
        .with_columns(
            pl.col("ACTIVITY").cast(pl.Utf8).str.zfill(4)
        )
        .filter(
            (pl.col("REF_AREA") == "HND") &
            (pl.col("mcp") == 0)
        )
        .join(
            ciiu_actividades,
            left_on="ACTIVITY", 
            right_on="clase_codigo",
            how="left"
        )
    )

    cdata_hnd
    return cdata_hnd, ciiu_actividades


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Se define el diccionario `complexity_metric`, que vincula nombres descriptivos de métricas con las columnas utilizadas internamente en la base de datos. La opción `Complexity` corresponde a `pci`, mientras que `Opportunity Gain` corresponde a `cog`.

    Además, se crean dos menús desplegables interactivos mediante `marimo`. El primero, `drop_product_selection_criteria`, permite seleccionar el criterio de priorización de actividades entre `Low-hanging Fruit`, `Balanced Portfolio` y `Long Jumps`.

    El segundo, `drop_complexity_metric`, permite seleccionar la métrica de complejidad que se utilizará en análisis o visualizaciones posteriores. Las opciones disponibles son `Complexity` y `Opportunity Gain`.

    Estos controles permiten explorar los resultados de forma interactiva, cambiando los criterios de selección y las métricas de análisis sin modificar manualmente el código.
    """)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se construye el diccionario `mapp_portafolios_inv`, que invierte el mapeo entre códigos internos y nombres descriptivos de portafolios. Esto permite convertir la selección realizada en el menú desplegable `drop_product_selection_criteria` al código utilizado internamente en la tabla `portafolios`.

    A partir del portafolio seleccionado, se extraen las actividades CIIU que forman parte de esa estrategia de priorización. Esta lista se almacena en `clases_ciiu_priorizar` y permite identificar cuáles actividades deben resaltarse en los análisis posteriores.

    Luego, la base `cdata_hnd` se divide en dos subconjuntos. `points_prioriza` contiene las actividades incluidas en el portafolio seleccionado, mientras que `points_resto` contiene el resto de actividades no priorizadas.

    Esta separación permite comparar visualmente las actividades priorizadas frente al conjunto restante de oportunidades, facilitando la interpretación de los resultados según el criterio de selección elegido por el usuario.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se configura un gráfico interactivo que compara las actividades priorizadas y no priorizadas para Honduras. Primero se construye `selection_weigths`, una etiqueta de texto que resume los pesos asignados al criterio de priorización seleccionado en `drop_product_selection_criteria`.

    La capa `relateness_plot_prioriza` contiene las actividades incluidas en el portafolio seleccionado. Estas actividades se muestran con puntos de mayor tamaño y borde negro, lo que permite destacarlas frente al resto de oportunidades. El eje horizontal representa la variable `distance`, mientras que el eje vertical utiliza la métrica seleccionada en `drop_complexity_metric`, que puede corresponder a `pci` o `cog`.

    La capa `relateness_plot_prioriza_negro` se utiliza como base para generar etiquetas de texto sobre las actividades priorizadas. A partir de esta capa se crea `text`, que muestra el nombre de cada actividad mediante la variable `clase_titulo`.

    Por su parte, `relateness_plot` representa las actividades no priorizadas. Estas se grafican con menor opacidad para servir como referencia visual del conjunto de oportunidades disponibles. El color de los puntos se define por `seccion_titulo`, permitiendo diferenciar las actividades según su sección económica.

    En conjunto, estas capas permiten construir un diagrama de distancia y complejidad que resalta las actividades priorizadas según el portafolio seleccionado, manteniendo como contexto el resto de actividades no priorizadas.
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Se construye una vista interactiva que combina los controles de selección y el gráfico de resultados. Los menús desplegables `drop_product_selection_criteria` y `drop_complexity_metric` permiten modificar el criterio de priorización y la métrica utilizada en el eje vertical.

    La celda evalúa si la métrica seleccionada corresponde a `ICI_UE`. Si se cumple esa condición, se muestran las capas alternativas `relateness_plot_prioriza_mo` y `relateness_plot_mo`. Para el resto de métricas, se muestran las capas estándar `relateness_plot_prioriza` y `relateness_plot`.

    Los elementos seleccionados se almacenan en `stack_plots`, una lista que combina controles y visualizaciones. Finalmente, `mo.vstack(stack_plots)` apila estos componentes verticalmente dentro del notebook.

    Esta estructura permite explorar de forma interactiva las oportunidades productivas de Honduras, alternando entre distintos criterios de portafolio y métricas de complejidad sin modificar manualmente el código.
    """)
    return


@app.cell
def _(
    complexity_metric,
    drop_complexity_metric,
    drop_product_selection_criteria,
    mo,
    relateness_plot,
    relateness_plot_mo,
    relateness_plot_prioriza,
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
                    relateness_plot_prioriza + relateness_plot,
                    #points_prioriza.select("Sector", "Subsector", "rama_id", "Industria", "score").sort(by="score", descending=False)
            ]

    mo.vstack(stack_plots)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Exportar resultados
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Se exportan los resultados en un archivo con formato xlsx dividido en tres hojas:
    1. Total de actividades
    2. Margen intensivo
    3. Margen extensivo
    """)
    return


@app.cell
def _(cdata, cdata_norm, ciiu_actividades, pd, pl, portafolios):
    from pathlib import Path

    # Crear carpeta de salida
    output_dir = Path("output/reporte")
    output_dir.mkdir(parents=True, exist_ok=True)

    archivo_salida = output_dir / "resultados_complexity_final.xlsx"


    def to_pandas_safe(tabla):
        if isinstance(tabla, pl.DataFrame):
            return tabla.to_pandas()
        return tabla


    def asegura_ciiu_4d(tabla, columna):
        if isinstance(tabla, pl.DataFrame):
            return tabla.with_columns(
                pl.col(columna).cast(pl.Utf8).str.zfill(4)
            )
        else:
            tabla[columna] = tabla[columna].astype(str).str.zfill(4)
            return tabla


    # Usar ciiu_actividades ya existente
    # No se redefine para evitar error en marimo

    ciiu_actividades_reporte = (
        ciiu_actividades
        .with_columns(
            pl.col("clase_codigo").cast(pl.Utf8).str.zfill(4)
        )
        .unique()
    )


    # Preparar bases de complejidad con códigos CIIU en cuatro dígitos

    cdata_export_reporte = (
        cdata
        .with_columns(
            pl.col("ACTIVITY").cast(pl.Utf8).str.zfill(4).alias("ACTIVITY")
        )
    )

    cdata_norm_export_reporte = (
        cdata_norm
        .with_columns(
            pl.col("ACTIVITY").cast(pl.Utf8).str.zfill(4).alias("ACTIVITY")
        )
    )


    # Hoja 1: 213 act

    reporte_213_act = (
        cdata_export_reporte
        .filter(pl.col("REF_AREA") == "HND")
        .join(
            ciiu_actividades_reporte,
            left_on="ACTIVITY",
            right_on="clase_codigo",
            how="left"
        )
        .select(
            pl.col("REF_AREA").alias("ISO 3"),
            pl.lit("Honduras").alias("País"),
            pl.col("ACTIVITY").alias("ciiu4_cod"),
            pl.col("clase_titulo").alias("ciiu4_actividad"),
            pl.col("seccion_codigo"),
            pl.col("seccion_titulo"),
            pl.col("OBS_VALUE").alias("Empleo"),
            pl.col("TIME_PERIOD").alias("Año"),
            "rca",
            "mcp",
            "diversity",
            "ubiquity",
            "eci",
            "pci",
            "density"
        )
        .sort("ciiu4_cod")
    )


    # Hoja 2: Intensivo

    reporte_intensivo = (
        cdata_export_reporte
        .filter(
            (pl.col("REF_AREA") == "HND") &
            (pl.col("rca") > 0) &
            (pl.col("mcp") == 1)
        )
        .join(
            ciiu_actividades_reporte,
            left_on="ACTIVITY",
            right_on="clase_codigo",
            how="left"
        )
        .select(
            pl.col("REF_AREA").alias("ISO 3"),
            pl.lit("Honduras").alias("País"),
            pl.col("ACTIVITY").alias("ciiu4_cod"),
            pl.col("clase_titulo").alias("ciiu4_actividad"),
            pl.col("seccion_codigo"),
            pl.col("seccion_titulo"),
            pl.col("OBS_VALUE").alias("Empleo"),
            pl.col("TIME_PERIOD").alias("Año"),
            "rca",
            "mcp",
            "diversity",
            "ubiquity",
            "eci",
            "pci",
            "density"
        )
        .sort("pci", descending=True)
    )


    # Hoja 3: Extensivo

    mapp_nombre_portafolios = {
        "lhf": "Oportunidades cercanas",
        "bp": "Portafolio balanceado",
        "lj": "Saltos largos"
    }

    portafolios_export_reporte = (
        portafolios
        .with_columns(
            pl.col("clase_codigo").cast(pl.Utf8).str.zfill(4).alias("clase_codigo")
        )
    )

    reporte_extensivo = (
        portafolios_export_reporte
        .join(
            cdata_norm_export_reporte
            .filter(pl.col("REF_AREA") == "HND")
            .select(
                "ACTIVITY",
                "cog",
                "pci",
                "density"
            ),
            left_on="clase_codigo",
            right_on="ACTIVITY",
            how="left"
        )
        .with_columns(
            source=pl.lit("ladder"),
            Portafolio=pl.col("portafolio").replace(mapp_nombre_portafolios)
        )
        .select(
            "source",
            pl.col("portafolio").alias("portfolio"),
            "Portafolio",
            "ranking",
            pl.col("clase_codigo").alias("ciiu4_cod"),
            pl.col("clase_titulo").alias("Clase CIIU"),
            pl.col("seccion_codigo"),
            pl.col("seccion_titulo"),
            pl.col("cog").alias("COG"),
            pl.col("pci").alias("PCI"),
            pl.col("density").alias("Density")
        )
        .sort(["portfolio", "ranking"])
    )


    # Asegurar códigos CIIU de cuatro dígitos

    reporte_213_act = asegura_ciiu_4d(reporte_213_act, "ciiu4_cod")
    reporte_intensivo = asegura_ciiu_4d(reporte_intensivo, "ciiu4_cod")
    reporte_extensivo = asegura_ciiu_4d(reporte_extensivo, "ciiu4_cod")


    # Verificación de etiquetas faltantes

    verifica_etiquetas_faltantes_213 = (
        reporte_213_act
        .filter(pl.col("ciiu4_actividad").is_null())
        .select("ciiu4_cod")
        .unique()
        .sort("ciiu4_cod")
    )

    verifica_etiquetas_faltantes_intensivo = (
        reporte_intensivo
        .filter(pl.col("ciiu4_actividad").is_null())
        .select("ciiu4_cod")
        .unique()
        .sort("ciiu4_cod")
    )

    print("Etiquetas faltantes en 213 act:")
    print(verifica_etiquetas_faltantes_213)

    print("Etiquetas faltantes en Intensivo:")
    print(verifica_etiquetas_faltantes_intensivo)


    # Exportar reporte

    with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:

        to_pandas_safe(reporte_213_act).to_excel(
            writer,
            sheet_name="213 act",
            index=False
        )

        to_pandas_safe(reporte_intensivo).to_excel(
            writer,
            sheet_name="Intensivo",
            index=False
        )

        to_pandas_safe(reporte_extensivo).to_excel(
            writer,
            sheet_name="Extensivo",
            index=False
        )

        for sheet_name in ["213 act", "Intensivo", "Extensivo"]:
            ws = writer.book[sheet_name]

            for cell in ws[1]:
                if cell.value == "ciiu4_cod":
                    col_letter = cell.column_letter

                    for row in range(2, ws.max_row + 1):
                        ws[f"{col_letter}{row}"].number_format = "@"
                        ws[f"{col_letter}{row}"].value = str(ws[f"{col_letter}{row}"].value).zfill(4)

    archivo_salida
    return


if __name__ == "__main__":
    app.run()
