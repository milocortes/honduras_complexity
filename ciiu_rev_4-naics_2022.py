import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    ## Cargamos paquetes 
    import networkx as nx
    import numpy as np
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from py2cytoscape import util as cy 
    from py2cytoscape import cytoscapejs as cyjs

    import requests
    import json

    from IPython.display import Image
    from IPython.display import Markdown as md
    import os 
    return cy, json, nx, pd, pl, requests


@app.cell
def _(pl):
    # Cargamos las correspondencias entre CIIU Rev 3-CIIU Rev 4-SCIAN 2023-CIIU Rev 4-NAICS 2022
    ciiu_rev_4_to_naics_2022 = pl.read_csv("datos/usa/isic_to_naics/ciiu_rev_4_to_naics_2022.csv")

    # Convertimos a pandas
    ciiu_rev_4_to_naics_2022_pd = ciiu_rev_4_to_naics_2022.to_pandas() 


    ciiu_rev_4_to_naics_2022_pd["ciiu_rev_4"] = ciiu_rev_4_to_naics_2022_pd["ciiu_rev_4"].apply(lambda x : f"ciiu_rev_4-{x}")
    ciiu_rev_4_to_naics_2022_pd["naics_2022"] = ciiu_rev_4_to_naics_2022_pd["naics_2022"].apply(lambda x : f"naics_2022-{x}")

    return ciiu_rev_4_to_naics_2022, ciiu_rev_4_to_naics_2022_pd


@app.cell
def _(ciiu_rev_4_to_naics_2022):
    ## Reunimos CIIU 3-CIIU 4 con CIIU 4-SCIAN 2023 y CIIU 4-NAICS 2022
    ciiu_rev_4_to_naics_2022
    return


@app.cell
def _(ciiu_rev_4_to_naics_2022_pd, nx):
    # empty graph/network
    G = nx.Graph()

    for k in ciiu_rev_4_to_naics_2022_pd.index:
        i = tuple(ciiu_rev_4_to_naics_2022_pd.iloc[[k]]["ciiu_rev_4"].values)[0] 
        j = tuple(ciiu_rev_4_to_naics_2022_pd.iloc[[k]]["naics_2022"].values)[0] 
        G.add_edge(str(i),str(j)) 
    return (G,)


@app.function
def basic_stats(G):
    print("nodes: %d" % G.number_of_nodes())
    print("edges: %d" % G.number_of_edges())


@app.cell
def _(G):
    basic_stats(G)
    return


@app.cell
def _(G, ciiu_rev_4_to_naics_2022_pd, nx):
    #gen sets
    set_k = ciiu_rev_4_to_naics_2022_pd[['naics_2022']]
    set_k = set_k.drop_duplicates(keep='first')
    set_k['set_type']=int(2)

    # node attributes
    node_type = set_k.set_index("naics_2022").set_type.to_dict()

    for k_ in node_type:
        node_type[k_]=int(node_type[k_])

    # assign attributes to networkx G
    nx.set_node_attributes(G,node_type,"n_type")
    return


@app.cell
def _(G, cy):
    # move network from networkx to cy
    G.node = G.nodes
    cytoscape_network = cy.from_networkx(G)
    return (cytoscape_network,)


@app.cell
def _(requests):
    # Basic Setup
    PORT_NUMBER = 1234
    IP = 'localhost'
    BASE = 'http://' + IP + ':' + str(PORT_NUMBER) + '/v1/'
    HEADERS = {'Content-Type': 'application/json'}
    requests.delete(BASE + 'session')
    return BASE, HEADERS


@app.cell
def _(BASE, HEADERS, cytoscape_network, json, requests):
    res1 = requests.post(BASE + 'networks', data=json.dumps(cytoscape_network), headers=HEADERS)
    res1_dict = res1.json()
    new_suid = res1_dict['networkSUID']
    requests.get(BASE + 'apply/layouts/force-directed/' + str(new_suid))
    return (new_suid,)


@app.cell
def _(BASE, HEADERS, json, new_suid, requests):
    style_name = 'Basic_Style'

    my_style = {
      "title" : style_name,
      "mappings" : [{
        "mappingType" : "passthrough",
        "mappingColumn" : "n_name",
        "mappingColumnType" : "String",
        "visualProperty" : "NODE_LABEL"
      },{
        "mappingType" : "discrete",
        "mappingColumn" : "n_type",
        "mappingColumnType" : "Double",
        "visualProperty" : "NODE_FILL_COLOR",
           "map" : [ {
          "key" : "1",
          "value" : "#CC0000"
        }, {
          "key" : "2",
          "value" : "#009999"
        }]  
      }], 
       'defaults': [{
           'visualProperty': 'EDGE_TRANSPARENCY', 
           'value': 200
      }, {
        "visualProperty" : "NODE_SIZE",
        'value': 50
      },{
        "visualProperty" : "EDGE_WIDTH",
        'value': 15
      },{
           'visualProperty': 'NODE_LABEL_TRANSPARENCY', 
           'value': 0
       }, {
           'visualProperty': 'NODE_TRANSPARENCY', 
           'value': 200
       }]
    }


    # Create new Visual Style
    res = requests.post(BASE + "styles", data=json.dumps(my_style), headers=HEADERS)
    new_style_name = res.json()['title']

    # Apply it to current netwrok
    requests.get(BASE + 'apply/styles/' + new_style_name + '/' + str(new_suid))

    return


@app.cell
def _(G):
    from networkx.algorithms import community
    C2 = community.label_propagation_communities(G)

    return (C2,)


@app.cell
def _(C2):
    list_comp2=sorted(C2, key = len, reverse=True)
    nc2 = len(list_comp2)
    nc2
    return list_comp2, nc2


@app.cell
def _(list_comp2, nc2, pd):
    # final concordance tables generation
    C123 = pd.DataFrame([(n,c, 0) for c in range(nc2) for n in list_comp2[c]] , columns= ["code","code_N","s"])

    # Eliminamos registros para los cuales hay correspondencia entre CIIU-Rev-3 : CIIU-Rev-4, pero no entre CIIU-Rev-4:SCIAN-2023
    C123 = C123[~ C123["code"].apply(lambda x : len(x) == 4)].reset_index(drop = True)
    C123
    return (C123,)


@app.cell
def _(C123):
    ## Generamos las correspondencias entre los sistemas de clasificación
    codificacion = C123.copy()
    codificacion.columns = ["codigo", "codigo_nuevo", "clasificador"]
    codificacion["clasificador"] = "ciiu_rev_4"
    codificacion.loc[codificacion["codigo"].apply(lambda x : "naics_2022" in x), "clasificador"] = "naics_2022"

    ## Ajustamos código
    #codificacion["codigo"] = codificacion["codigo"].apply(lambda x : x.split("-")[-1] if "rev" in x else x)
    codificacion["codigo"] = codificacion["codigo"].apply(lambda x : x.split("-")[-1])

    codificacion
    return (codificacion,)


@app.cell
def _(pd):
    ### Agregamos nombres de actividades
    ciiu_rev_4_nombres = pd.read_csv("datos/cw/ciiu_rev_4_nombres_actidades.csv")
    naics_2022_nombres = pd.read_csv("datos/usa/isic_to_naics/naics_2022_nombres_actividades.csv")

    ciiu_rev_4_nombres["ciiu_rev_4"] = ciiu_rev_4_nombres["ciiu_rev_4"].astype(str)
    return ciiu_rev_4_nombres, naics_2022_nombres


@app.cell
def _(ciiu_rev_4_nombres, codificacion, naics_2022_nombres):
    # Creamos columna de nombre de la actividad
    codificacion["nombre_actividad"] = ""

    codificacion.loc[codificacion["clasificador"] == 'naics_2022', "nombre_actividad"] = codificacion.loc[codificacion["clasificador"] == 'naics_2022', "codigo"].replace({ str(i):j  for i,j in naics_2022_nombres.to_records(index = False)})

    codificacion.loc[codificacion["clasificador"] == 'ciiu_rev_4', "nombre_actividad"] = codificacion.loc[codificacion["clasificador"] == 'ciiu_rev_4', "codigo"].replace({ i:j  for i,j in ciiu_rev_4_nombres.to_records(index = False)})

    return


@app.cell
def _(codificacion):
    codificacion
    return


@app.cell
def _(codificacion):
    codificacion.to_csv("datos/recodificacion/recodificacion_hnd_usa.csv", index = False)
    return


if __name__ == "__main__":
    app.run()
