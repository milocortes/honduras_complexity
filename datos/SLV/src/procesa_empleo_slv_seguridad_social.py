import pandas as pd 
import numpy as np 
import glob 
import os

import h5py

import yaml


# Definimos el periodo de cobertura
anio_inicio = 2013
anio_fin = 2022

#anio_inicio = 2019
#anio_fin = 2020

# Definimos rutas
PATH_FILE = os.getcwd()
DATA_PATH = os.path.abspath(os.path.join(PATH_FILE, "..", "datos","seguridad_social"))
OUTPUT_PATH = os.path.abspath(os.path.join(PATH_FILE, "..", "output"))
DOCS_PATH = os.path.abspath(os.path.join(PATH_FILE, "..", "docs"))
CW_PATH = os.path.abspath(os.path.join(PATH_FILE, *[".."]*3, "cws", "empleo"))

sheets_name = ["Regimen General", "Regimen Especial"]

regimenes = ["DOMESTICO", "INDEPENDIENTE COBERTURA FAMILIAR", "INDEPENDIENTE COBERTURA INDIVIDUAL", "MARINOS MERCANTES", "APRENDIZ", "ATLETA DE ALTO RENDIMIENTO", "INSCRITO"]

# Cargamos los datos
ss_files_path = glob.glob(DATA_PATH+"/*.xlsx")
ss_files_path.sort()

acumula_anios = []

for ss_file in ss_files_path:
    anio = int(ss_file.split("/")[-1].split(".")[0].split("-")[-1])
    
    if anio in range(anio_inicio, anio_fin):
        print(anio)
        general = pd.read_excel(ss_file, sheet_name = sheets_name[0])
        especial = pd.read_excel(ss_file, sheet_name = sheets_name[1])

        colnames_general = list(general.iloc[3,])
        colnames_especial = list(especial.iloc[3,])

        general = general.iloc[4:,].reset_index(drop = True)
        especial = especial.iloc[4:,].reset_index(drop = True)

        general.columns = colnames_general
        especial.columns = colnames_especial

        empleo = pd.concat([general, especial], ignore_index = True)
        empleo = empleo[empleo["REGIMEN"].isin(regimenes)]

        empleo = empleo[empleo["CODIGO CIIU"].apply(lambda x : len(str(x))!=3)]

        empleo["ciiu_4"] = empleo["CODIGO CIIU"].apply(lambda x : str(x)[:4])

        empleo_ciiu = empleo["ciiu_4"].value_counts().sort_index()
        empleo_ciiu.name = anio
        empleo_ciiu = pd.DataFrame(empleo_ciiu).transpose()
        acumula_anios.append(empleo_ciiu)

df_ciiu = pd.concat(acumula_anios)
df_ciiu = df_ciiu.replace(np.nan, 0.0)

"""
conteos = []
for anio,dato in enumerate(acumula_anios):
    print(anio+anio_inicio)
    parcial = [anio+anio_inicio]

    for i in range(3, 6):
        parcial.append(sum(1 for col in dato.columns if len(str(col))==i))

    conteos.append(parcial)
"""


df_ciiu = df_ciiu.reset_index().rename(columns = {"index" : "anio"})

## Cargamos el crosswalk entre el ciiu-4 dígitos y la recodificación ciiu-4 digitos
ciiu_recodificado = pd.read_csv(os.path.join(CW_PATH, "ciiu-recodificacion-dario-diodato", "output", "recodificacion_ciiu-rev-4.csv"))
ciiu_nueva_clasificacion = {i:[] for i in ciiu_recodificado["ciiu_nueva_cod"]}

for ciiu_nuevo, ciiu_old in ciiu_recodificado[["ciiu_nueva_cod", "actividades_ciiu_integra"]].to_records(index = False):
    for ciiu_individual in ciiu_old.split(","):
        ciiu_nueva_clasificacion[ciiu_nuevo].append(ciiu_individual)

df_ciiu_recodificado = df_ciiu[["anio"]]

for ciiu_nuevo, ciiu_old_grupo in ciiu_nueva_clasificacion.items():
    ciiu_old_coinciden = list(set(ciiu_old_grupo).intersection(df_ciiu.columns))

    if ciiu_old_coinciden:
        ciiu_old_coinciden.sort()
        df_ciiu_recodificado[ciiu_nuevo] = df_ciiu[ciiu_old_coinciden].sum(axis = 1)
    else: 
        df_ciiu_recodificado[ciiu_nuevo] = 0

## Creamos HDF5

with h5py.File(os.path.join(OUTPUT_PATH, 'matrices_laborales_slv.h5'), 'w') as f_h5:

    ## Generamos los grupos principales 
    # /SLV
    #      /SLV/datos
    #      /SLV/tags
    grupo_slv = f_h5.create_group("SLV")
    grupo_datos = grupo_slv.create_group("datos")
    grupo_tags = grupo_slv.create_group("tags")

    ## Generamos grupos por tipo de clasificador
    #      /SLV/datos
    #      /SLV/datos/ciiu_original
    #      /SLV/datos/ciiu_recodificado
    grupo_datos_ciiu_original = grupo_datos.create_group("ciiu-original")
    grupo_datos_ciiu_recodificado = grupo_datos.create_group("ciiu-recodificado")

    grupo_datos_ciiu_original.create_dataset("2013-2022", (df_ciiu.shape[0], df_ciiu.shape[1]), data = df_ciiu.to_numpy())
    grupo_datos_ciiu_recodificado.create_dataset("2013-2022", (df_ciiu_recodificado.shape[0], df_ciiu_recodificado.shape[1]), data = df_ciiu_recodificado.to_numpy())

    #grupo_tags = grupo_datos.create_group("tag")

    tags_actividades_ciiu = [str(n).encode("ascii", "ignore") for n in df_ciiu.columns]
    tags_actividades_ciiu_recodificacion = [str(n).encode("ascii", "ignore") for n in df_ciiu_recodificado.columns]

    grupo_tags.create_dataset("ciiu-original", (len(tags_actividades_ciiu),), 'S10', tags_actividades_ciiu)
    grupo_tags.create_dataset("ciiu-recodificado", (len(tags_actividades_ciiu_recodificacion),), 'S10', tags_actividades_ciiu_recodificacion)



## Definimos metadatos a partir del archivo yaml
empleo_slv = h5py.File(os.path.join(OUTPUT_PATH, 'matrices_laborales_slv.h5'), "r+")

with open(os.path.join(DOCS_PATH, "metadata.yml"), "r") as file:
    metadatos = yaml.load(file, Loader=yaml.Loader)

empleo_slv["SLV"].attrs['titulo'] = metadatos["variable"]["titulo"]
empleo_slv["SLV"].attrs['clasificacion_industrial'] = metadatos["variable"]["clasificacion"]
empleo_slv["SLV"].attrs['descripcion'] = metadatos["resources"]["descrip"]

#empleo_slv["SLV"]["datos"]['2013-2022'].dims[0].label = metadatos["variable"]["label-0"]
#empleo_slv["SLV"]["datos"]['2013-2022'].dims[1].label = metadatos["variable"]["label-1"]
empleo_slv.flush()
empleo_slv.close()