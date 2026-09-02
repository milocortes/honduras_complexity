import pypst
import pandas as pd 

## Carga tablas
intensivo = pd.read_csv("datos/anexos/intensivo_actividades.csv")
extensivo = pd.read_csv("datos/anexos/extensivo_actividades.csv").rename(columns = {"Clusters" : "Clúster"})
textiles = pd.read_csv("datos/od/textiles_hs12.csv")

## Define indices
intensivo = intensivo.set_index(["Clúster", "Clave CIIU4", "Actividad CIIU4"])
extensivo = extensivo.set_index(["Clúster", "Clave CIIU4", "Actividad CIIU4"])


table_intensivo = pypst.Table.from_dataframe(intensivo)
table_extensivo = pypst.Table.from_dataframe(extensivo)
table_textiles = pypst.Table.from_dataframe(textiles, include_index = False)

def estiliza_tabla(table, data):
    
    table.stroke = "none"
    table.align = "(x, _) => if calc.odd(x) {left} else {right}"
    table.add_hline(1, stroke="1.5pt")
    table.add_hline(len(data) + data.columns.nlevels, stroke="1.5pt")

estiliza_tabla(table_intensivo, intensivo)
estiliza_tabla(table_extensivo, extensivo)

print(table_intensivo.render())
print(table_extensivo.render())
print(table_textiles.render())

