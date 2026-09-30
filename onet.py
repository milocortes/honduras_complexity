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
    import sqlalchemy
    import pandas as pd
    return pd, pl, sqlalchemy


@app.cell
def _(sqlalchemy):
    ## Iniciamos conexión con sqlite
    DATABASE_URL = "sqlite:///datos/onet/output/onet.db"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(engine, sqlalchemy):
    # Inspect the database
    inspector = sqlalchemy.inspect(engine)

    # Get all table names
    table_names = inspector.get_table_names()
    return (table_names,)


@app.cell
def _(table_names):
    table_names
    return


@app.cell
def _(engine, pd, pl):
    ## Cargamos datos de ONET
    def get_tabla(tabla : str) -> pd.DataFrame:
        return pl.read_database(
                    query=f"SELECT * FROM {tabla}", 
                    connection=engine.connect(), 
                    infer_schema_length=None
                ).to_pandas()

    onet = get_tabla("naics4d_onet_empleo")
    onet
    return get_tabla, onet


@app.cell
def _(onet, pl):
    pl.from_pandas(onet).select("NAICS", "NAICS_TITLE").unique()
    return


@app.cell
def _(get_tabla):
    ## Cargamos CW entre NAICS y CIIU
    cw = get_tabla("cw_naics_ciiu")
    cw
    return (cw,)


@app.cell
def _(cw, onet, pl):
    ## Reunimos ONET y CW
    onet_ciiu = onet[["NAICS", "OCC_CODE", "OCC_TITLE", "OCC_GROUP", "TOT_EMP"]].merge(
        cw, 
        left_on="NAICS", 
        right_on="naics"
    ).query("TOT_EMP !='**'")
    onet_ciiu = pl.from_pandas(onet_ciiu)

    ## Calculamos el empleo de la clase CIIU con los ponderadores ya agrupamos
    onet_ciiu = onet_ciiu.filter(
        pl.col("OCC_GROUP") == "detailed"
    ).with_columns(
        emp_ciiu = (
             pl.col("TOT_EMP").cast(pl.Int32)*pl.col("weight")   
        ).cast(pl.Int32)
    ).select(
        "ciiu", "emp_ciiu", "OCC_CODE", "OCC_TITLE"
    ).group_by(
        "ciiu", "OCC_CODE", "OCC_TITLE"
    ).sum()

    onet_ciiu
    return (onet_ciiu,)


@app.cell
def _():
    return


@app.cell
def _(onet_ciiu, pl):
    ## Carga clusters
    clusters = pl.read_csv(
                    "datos/clusters_complejidad/clusters_actividad.csv"
                ).with_columns(
                    pl.col("ACTIVITY").cast(pl.String)
                )

    ## Reune cluster y datos ONET
    clusters = clusters.join(
        onet_ciiu, 
        left_on="ACTIVITY", 
        right_on="ciiu"
    ).drop(
        "ACTIVITY", "clase_titulo"
    ).group_by(
        "Clusters", "OCC_CODE", "OCC_TITLE"
    ).sum()

    ir_empleo =     clusters.select(
                                        "OCC_CODE", "emp_ciiu"
                                    ).group_by(
                                        "OCC_CODE"
                                    ).sum().with_columns(
                                        importancia_relativa = pl.col("emp_ciiu")/pl.col("emp_ciiu").sum()
                                    ).drop("emp_ciiu")


    clusters = clusters.join(
        ir_empleo, 
        on = "OCC_CODE", 
        how = "left"
    )

    clusters
    return clusters, ir_empleo


@app.cell
def _(get_tabla, pl):
    ## Carga descripción de ocupaciones
    ocupaciones = get_tabla("occupation_data").assign(
        OCC_CODE = lambda df : df["O*NET-SOC Code"].apply(lambda x : x.split(".")[0]), 
        onet_code_suffix = lambda df : df["O*NET-SOC Code"].apply(lambda x : x.split(".")[1]), 
    ).loc[
        lambda df : df["onet_code_suffix"] == '00'
    ].drop(columns = ["onet_code_suffix", "O*NET-SOC Code", "Title"] )

    ocupaciones = pl.from_pandas(ocupaciones)
    ocupaciones
    return (ocupaciones,)


@app.cell
def _(clusters, ocupaciones):
    ## Guardamos datos
    clusters.join(
        ocupaciones, 
        on = "OCC_CODE", 
        how="left"
    ).write_csv("output/clusters_ocupacion.csv")
    return


@app.cell
def _(clusters, ocupaciones):
    clusters.join(
        ocupaciones, 
        on = "OCC_CODE", 
        how="left"
    )
    return


@app.cell
def _():
    return


@app.cell
def _(get_tabla, pd, pl):
    # Cargamos los datos de skills, knowledge y abilities
    skills = get_tabla("skills")
    knowledge = get_tabla("knowledge")
    abilities = get_tabla("abilities")

    # Nos quedamos los recursos de Incumbent y Analyst
    skills = skills[skills["Domain Source"] == "Analyst"].reset_index(drop = True)
    abilities = abilities[abilities["Domain Source"] == "Analyst"].reset_index(drop = True)
    knowledge = knowledge[knowledge["Domain Source"] == "Incumbent"].reset_index(drop = True)

    # Descargamos información de Scale ID
    escalas = pd.read_table("https://www.onetcenter.org/dl_files/database/db_28_2_text/Scales%20Reference.txt")

    # Agregamos información adicional
    skills = skills.merge(right=escalas, how = "left", on = "Scale ID")
    abilities = abilities.merge(right=escalas, how = "left", on = "Scale ID")
    knowledge = knowledge.merge(right=escalas, how = "left", on = "Scale ID")

    ### Agregamos información del nombre de los identificadores 
    onet_taxonomia = pd.read_html("https://www.onetcenter.org/taxonomy/2019/list.html")[0]
    onet_taxonomia = onet_taxonomia.rename(columns = {"O*NET-SOC 2019 Code" : "O*NET-SOC Code"})

    skills = skills.merge(right=onet_taxonomia, how="inner", on="O*NET-SOC Code")
    abilities = abilities.merge(right=onet_taxonomia, how="inner", on="O*NET-SOC Code")
    knowledge = knowledge.merge(right=onet_taxonomia, how="inner", on="O*NET-SOC Code")

    ### Agregamos las categorías de O*NET-SOC Code
    skills["O*NET-SOC Code"] = skills["O*NET-SOC Code"].apply(lambda x : str(x).split(".")[0])
    abilities["O*NET-SOC Code"] = abilities["O*NET-SOC Code"].apply(lambda x : str(x).split(".")[0])
    knowledge["O*NET-SOC Code"] = knowledge["O*NET-SOC Code"].apply(lambda x : str(x).split(".")[0])

    ### Convertimos a Polars
    skills = pl.from_pandas(skills).filter(
        pl.col("Scale ID")=='IM'
    )
    abilities = pl.from_pandas(abilities).filter(
        pl.col("Scale ID")=='IM' 
    )
    knowledge = pl.from_pandas(knowledge).filter(
        pl.col("Scale ID")=='IM'
    )
    return abilities, knowledge, skills


@app.cell
def _(skills):
    skills
    return


@app.cell
def _(clusters, ir_empleo, pl, skills):
    ## Skills
    skills_weighted_avg = skills.select(
        "O*NET-SOC Code", "Element Name", "Data Value"
    ).join(
        ir_empleo, 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="inner"
    ).unique(subset=["O*NET-SOC Code", "Element Name"]).join(
        skills.select(
            "O*NET-SOC Code", "Element Name", "Data Value"
        ).join(
            ir_empleo, 
            left_on="O*NET-SOC Code", 
            right_on="OCC_CODE", 
            how="inner"
        ).group_by("Element Name").agg(
            weighted_avg = (pl.col("Data Value") * pl.col("importancia_relativa")).sum() / pl.col("importancia_relativa").sum()
        ), 
        on = "Element Name", 
        how = "left"
    ).join(
        clusters.select("OCC_CODE", "OCC_TITLE").unique(), 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="left"
    )

    skills_weighted_avg
    return (skills_weighted_avg,)


@app.cell
def _(ir_empleo):
    ir_empleo
    return


@app.cell
def _(clusters, ir_empleo, knowledge, pl):
    ## Knowledge
    knowledge_weighted_avg = knowledge.select(
        "O*NET-SOC Code", "Element Name", "Data Value"
    ).join(
        ir_empleo, 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="inner"
    ).unique(subset=["O*NET-SOC Code", "Element Name"]).join(
        knowledge.select(
            "O*NET-SOC Code", "Element Name", "Data Value"
        ).join(
            ir_empleo, 
            left_on="O*NET-SOC Code", 
            right_on="OCC_CODE", 
            how="inner"
        ).group_by("Element Name").agg(
            weighted_avg = (pl.col("Data Value") * pl.col("importancia_relativa")).sum() / pl.col("importancia_relativa").sum()
        ), 
        on = "Element Name", 
        how = "left"
    ).join(
        clusters.select("OCC_CODE", "OCC_TITLE").unique(), 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="left"
    )
    knowledge_weighted_avg
    return (knowledge_weighted_avg,)


@app.cell
def _(abilities, clusters, ir_empleo, pl):
    ## Abilities
    abilities_weighted_avg = abilities.select(
        "O*NET-SOC Code", "Element Name", "Data Value"
    ).join(
        ir_empleo, 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="inner"
    ).unique(subset=["O*NET-SOC Code", "Element Name"]).join(
        abilities.select(
            "O*NET-SOC Code", "Element Name", "Data Value"
        ).join(
            ir_empleo, 
            left_on="O*NET-SOC Code", 
            right_on="OCC_CODE", 
            how="inner"
        ).group_by("Element Name").agg(
            weighted_avg = (pl.col("Data Value") * pl.col("importancia_relativa")).sum() / pl.col("importancia_relativa").sum()
        ), 
        on = "Element Name", 
        how = "left"
    ).join(
        clusters.select("OCC_CODE", "OCC_TITLE").unique(), 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="left"
    )

    abilities_weighted_avg
    return (abilities_weighted_avg,)


@app.cell
def _(abilities_weighted_avg, knowledge_weighted_avg, skills_weighted_avg):
    skills_weighted_avg.write_csv("output/skills_weighted_avg.csv")
    abilities_weighted_avg.write_csv("output/abilities_weighted_avg.csv")
    knowledge_weighted_avg.write_csv("output/knowledge_weighted_avg.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Job Zones
    """)
    return


@app.cell
def _(get_tabla, pl):
    job_zones = pl.from_pandas(get_tabla("job_zones"))
    job_zones = job_zones.with_columns(
        pl.col("O*NET-SOC Code").map_elements(lambda x : x.split(".")[0])
    ).group_by("O*NET-SOC Code").agg(
        pl.col("Job Zone").mean().cast(pl.Int8)
    )
    job_zones
    return (job_zones,)


@app.cell
def _(onet, pl):
    empleo_total_onet = pl.from_pandas(
        onet
    ).filter(
        (pl.col("TOT_EMP" )!='**') &
        (pl.col("OCC_GROUP") == "detailed") 
    ).with_columns(
        pl.col("TOT_EMP").cast(pl.Int32)   
    ).group_by("OCC_CODE").agg(
        pl.col("TOT_EMP").sum()
    )
    empleo_total_onet
    return (empleo_total_onet,)


@app.cell
def _(empleo_total_onet, get_tabla, job_zones, pl):
    job_zones_shares = job_zones.join(
        empleo_total_onet, 
        left_on="O*NET-SOC Code", 
        right_on="OCC_CODE", 
        how="inner"
    ).with_columns(
        peso = pl.col("TOT_EMP") / pl.col("TOT_EMP").sum().over("Job Zone")
    ).join(
        pl.from_pandas(get_tabla("occupation_data")).with_columns(
            pl.col("O*NET-SOC Code").map_elements(lambda x : x.split(".")[0])
        ), 
        on = "O*NET-SOC Code"
    ).with_columns(
        pl.col("Job Zone").cast(pl.String).str.replace_all('2','1-2',literal=True)
    )
    job_zones_shares
    return


@app.cell
def _():
    return


@app.cell
def _(get_tabla):
    get_tabla("job_zone_reference")
    return


@app.cell
def _(get_tabla):
    get_tabla("job_zone_reference")[["Job Zone", "SVP Range"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ejercicio para las industrias : Turismo, Logística, Agroindustria, Construicción
    """)
    return


@app.cell
def _(pl):
    ## Cargamos delta table en el repositorio de honduras
    from fsspec.implementations.github import GithubFileSystem
    from pathlib import Path 

    fs = GithubFileSystem(
        org = "milocortes",
        repo="complejidad_economica_honduras"
    )

    ## Cargamos data del filesystem 
    with fs.open("datos/catalogo_ciiu_rev4/part-00000-7050957a-8cb5-477e-8eaf-c50fe5772c5f-c000.snappy.parquet") as _f: 
        catalogo_ciiu = pl.read_parquet(_f)

    catalogo_ciiu_str = catalogo_ciiu.select(
        "clase_codigo", "clase_titulo", "division_codigo", "division_titulo"
    ).with_columns(
        pl.col("clase_codigo").map_elements(lambda x : f"{x:04d}", return_dtype=pl.String)
    )
    catalogo_ciiu_str
    catalogo_ciiu
    return (catalogo_ciiu_str,)


@app.cell
def _(catalogo_ciiu_str, onet_ciiu):
    ## Reunimos con datos de onet 
    onet_ciiu_nombres = onet_ciiu.join(
        catalogo_ciiu_str, 
        left_on="ciiu", 
        right_on="clase_codigo"
    )
    onet_ciiu_nombres
    return (onet_ciiu_nombres,)


@app.cell
def _(abilities, knowledge, pl, skills):
    def calcula_empleo_importancia(
        df : pl.DataFrame
        ) -> pl.DataFrame: 
        return df.group_by(
            "OCC_CODE", "OCC_TITLE"
        ).agg(
            pl.col("emp_ciiu").sum()
        ).with_columns(
            share = pl.col("emp_ciiu")/pl.col("emp_ciiu").sum()
        ).sort("share", descending=True)

    def calcula_habilidades_cluster(
        df_habilidad : pl.DataFrame, 
        df_cluster : pl.DataFrame
        ) -> pl.DataFrame: 

        return df_habilidad.select(
            "O*NET-SOC Code", "Element Name", "Data Value"
        ).join(
            calcula_empleo_importancia(df_cluster), 
            left_on="O*NET-SOC Code", 
            right_on="OCC_CODE", 
            how="inner"
        ).unique(subset=["O*NET-SOC Code", "Element Name"]).group_by("Element Name").agg(
                weighted_avg = (pl.col("Data Value") * pl.col("share")).sum() / pl.col("share").sum()
        ).sort("weighted_avg", descending=True)

    habilidades_map = {
        "skills" : skills, 
        "abilities" : abilities, 
        "knowledge" : knowledge
    }
    return (
        calcula_empleo_importancia,
        calcula_habilidades_cluster,
        habilidades_map,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Actividades en Industrias
    ### Agroindustrias

    - Sección C: Industrias manufactureras (División 10 a 12 - Procesamiento agroindustrial)
        - División 10: Elaboración de productos alimenticios (matanza, procesamiento de carne, pescado, frutas, aceites, lácteos, molinería, panadería y azúcares).
        - División 11: Elaboración de bebidas (vinos, cervezas, bebidas malteadas y gaseosas).
        - División 12: Elaboración de productos de tabaco.
    """)
    return


@app.cell
def _(onet_ciiu_nombres, pl):
    ## Filtramos diviones de agroindustria
    division_agroindustria = [10,11,12]
    agroindustrias = onet_ciiu_nombres.filter(pl.col("division_codigo").is_in(division_agroindustria))
    agroindustrias
    return (agroindustrias,)


@app.cell
def _(agroindustrias, calcula_empleo_importancia):
    ## Ocupaciones más importantes 
    ocupaciones_agroindustria = calcula_empleo_importancia(agroindustrias)
    ocupaciones_agroindustria
    return (ocupaciones_agroindustria,)


@app.cell
def _(agroindustrias, calcula_habilidades_cluster, habilidades_map):
    ## Skills más importantes 
    agroindustrias_skills = { f"agroindustria_{i}" : calcula_habilidades_cluster(j, agroindustrias) for i,j in habilidades_map.items()}
    agroindustrias_skills
    return (agroindustrias_skills,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Turismo

    - Alojamiento (División 55):
        - 5510: Actividades de alojamiento para estancias cortas (hoteles, hostales, centros vacacionales).
        - 5520: Campamentos, zonas para vehículos recreativos y parques de campamentos.
        - 5590: Otros tipos de alojamiento n.c.p. (no clasificados previamente).

    - Servicios de Comidas y Bebidas (División 56):
        - 5610: Actividades de restaurantes y de servicio móvil de comidas.
        - 5630: Bar y servicio de bebidas.

    - Agencias de Viajes y Operadores Turísticos (División 79):
        - 7911: Actividades de las agencias de viajes (venta de viajes).
        - 7912: Actividades de operadores turísticos (organización de paquetes).
        - 7990: Otros servicios de reserva y actividades conexas (guías turísticos, asistencia).

    - Transporte de Pasajeros (Sección H):
        - 4922: Transporte terrestre de pasajeros por vía urbana y suburbana / interurbano (cuando es turístico).
        - 5011 / 5021: Transporte de pasajeros por agua (cruceros, ferris turísticos).
        - 5110: Transporte de pasajeros por aire.
    - Actividades Recreativas y Culturales (Sección R):
        - 9321: Actividades de parques de atracciones y parques temáticos.
        - 9329: Otras actividades de esparcimiento y recreación n.c.p.
    """)
    return


@app.cell
def _(onet_ciiu_nombres, pl):
    ## Filtramos clases en Turismo
    clases_turismo = ["5510", "5520", "5590", "5610", "5630", "7911", "7912", "7990", "4922", "5011", "5021", "5110", "9321", "99329"]
    turismo = onet_ciiu_nombres.filter(pl.col("ciiu").is_in(clases_turismo))
    turismo
    return (turismo,)


@app.cell
def _(calcula_empleo_importancia, turismo):
    ## Ocupaciones más importantes 
    ocupaciones_turismo = calcula_empleo_importancia(turismo)
    ocupaciones_turismo

    return (ocupaciones_turismo,)


@app.cell
def _(calcula_habilidades_cluster, habilidades_map, turismo):
    ## Skills más importantes 
    turismo_skills = { f"turismo_{i}" : calcula_habilidades_cluster(j, turismo) for i,j in habilidades_map.items()}
    turismo_skills
    return (turismo_skills,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Logística
    - División 49: Transporte terrestre y transporte por tuberías
        - Transporte de carga por carretera.
        - Transporte por vía férrea y tuberías
    - División 50: Transporte acuático
        - Transporte de carga y pasajeros por vías marítimas y de navegación interior.
    - División 51: Transporte aéreo
        - Transporte de carga y pasajeros vía aérea.
    - División 52: Almacenamiento y actividades de apoyo al transporte
        - Depósito y almacenamiento de mercancías.
        - Manipulación de carga, gestión de puertos, aeropuertos y estaciones.
        - Actividades de agencia o intermediación de transporte
    - División 53: Actividades postales y de mensajería
        - Recogida, clasificación y entrega de paquetes y correo
    """)
    return


@app.cell
def _(onet_ciiu_nombres, pl):
    ## Filtramos diviones de logistica
    divisiones_logistica = [49, 50, 51, 52, 53]
    logistica = onet_ciiu_nombres.filter(pl.col("division_codigo").is_in(divisiones_logistica))
    logistica
    return (logistica,)


@app.cell
def _(calcula_empleo_importancia, logistica):
    ## Ocupaciones más importantes 
    ocupaciones_logistica = calcula_empleo_importancia(logistica)
    ocupaciones_logistica
    return (ocupaciones_logistica,)


@app.cell
def _(calcula_habilidades_cluster, habilidades_map, logistica):
    ## Skills más importantes 
    logistica_skills = { f"logistica_{i}" : calcula_habilidades_cluster(j, logistica) for i,j in habilidades_map.items()}
    logistica_skills
    return (logistica_skills,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Construcción
    - División 41: Construcción de edificios
        - Incluye la construcción de edificios residenciales y no residenciales.
    - División 42: Obras de ingeniería civil
        - Incluye la construcción de infraestructuras de transporte (carreteras, puentes, vías férreas), redes de servicios y obras pesadas.
    - División 43: Actividades especializadas de construcción
        - Incluye trabajos de demolición, preparación de terrenos, instalaciones eléctricas, fontanería y otros acabados u obras especializadas
    """)
    return


@app.cell
def _(onet_ciiu_nombres, pl):
    ## Filtramos diviones de construcción
    divisiones_construccion = [41, 42, 43]
    construccion = onet_ciiu_nombres.filter(pl.col("division_codigo").is_in(divisiones_construccion))
    construccion
    return (construccion,)


@app.cell
def _(calcula_empleo_importancia, construccion):
    ## Ocupaciones más importantes 
    ocupaciones_construccion = calcula_empleo_importancia(construccion)
    ocupaciones_construccion
    return (ocupaciones_construccion,)


@app.cell
def _(calcula_habilidades_cluster, construccion, habilidades_map):
    ## Skills más importantes 
    construccion_skills = { f"construccion_{i}" : calcula_habilidades_cluster(j, construccion) for i,j in habilidades_map.items()}
    construccion_skills
    return (construccion_skills,)


@app.cell
def _(
    agroindustrias_skills,
    construccion_skills,
    logistica_skills,
    ocupaciones_agroindustria,
    ocupaciones_construccion,
    ocupaciones_logistica,
    ocupaciones_turismo,
    pd,
    turismo_skills,
):
    # Exportamos a excel
    with pd.ExcelWriter("ocupaciones_skills_4_sectores.xlsx") as writer:
        ocupaciones_turismo.to_pandas().to_excel(writer, sheet_name="ocupaciones_turismo", index=False)
        for skill_ds,df_skill_ds in turismo_skills.items():
            df_skill_ds.to_pandas().to_excel(writer, sheet_name = skill_ds, index=False)
    
        ocupaciones_logistica.to_pandas().to_excel(writer, sheet_name="ocupaciones_logistica", index=False)
        for skill_ds,df_skill_ds in logistica_skills.items():
            df_skill_ds.to_pandas().to_excel(writer, sheet_name = skill_ds, index=False)
    
        ocupaciones_agroindustria.to_pandas().to_excel(writer, sheet_name="ocupaciones_agroindustria", index=False)
        for skill_ds,df_skill_ds in agroindustrias_skills.items():
            df_skill_ds.to_pandas().to_excel(writer, sheet_name = skill_ds, index=False)                

        ocupaciones_construccion.to_pandas().to_excel(writer, sheet_name="ocupaciones_construccion", index=False)
        for skill_ds,df_skill_ds in construccion_skills.items():
            df_skill_ds.to_pandas().to_excel(writer, sheet_name = skill_ds, index=False)

    
        
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
