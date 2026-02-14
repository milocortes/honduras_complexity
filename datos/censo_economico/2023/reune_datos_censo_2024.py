import pandas as pd 
import glob
import numpy as np

files = glob.glob("csvs/*.csv")

acumula_df = []
 
for i, f in enumerate(files):
	print(i,f)
	df = pd.read_csv(f)
	las_columns = df.columns
	df = pd.DataFrame(df.reset_index().to_numpy()[:,:-1], columns=las_columns)
	df = df[df["CODIGO"].apply(lambda x: len(str(x))==6)]
	df = df[["ENTIDAD", "MUNICIPIO", "CODIGO", "ID_ESTRATO", "H001A"]]
	df = df[~df["ID_ESTRATO"].isna()]
	df["MUNICIPIO"] = df["MUNICIPIO"].apply(lambda x: str(x).lstrip())
	df = df[df["MUNICIPIO"]!='']
	acumula_df.append(df[['ENTIDAD', 'MUNICIPIO', 'CODIGO', 'H001A']])
	
entidades_clases = pd.concat(acumula_df, ignore_index = True)
entidades_clases = entidades_clases.sort_values(by=["ENTIDAD", "MUNICIPIO"])
entidades_clases = entidades_clases.rename(columns = {"H001A" : "empleo"})
entidades_clases["empleo"] = entidades_clases["empleo"].replace(np.nan,0).astype(int)
entidades_clases.to_csv("censo_2019_empleo_clases.csv", index = False)

