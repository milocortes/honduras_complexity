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
    return Image, cy, json, nx, pd, pl, requests


@app.cell
def _(pl):
    # Cargamos las correspondencias entre CIIU Rev 3-CIIU Rev 4-SCIAN 2023
    ciiu_rev_3_to_ciiu_rev_4 = pl.read_csv("datos/cw/ciiu_rev_3_to_ciiu_rev_4.csv")
    ciiu_rev_4_to_scian_2023 = pl.read_csv("datos/cw/ciiu_rev_4_to_scian_2023.csv")

    # Convertimos a pandas
    ciiu_rev_3_to_ciiu_rev_4_pd = ciiu_rev_3_to_ciiu_rev_4.to_pandas()
    ciiu_rev_4_to_scian_2023_pd = ciiu_rev_4_to_scian_2023.to_pandas() 

    ciiu_rev_4_to_scian_2023_pd["ciiu_rev_4"] = ciiu_rev_4_to_scian_2023_pd["ciiu_rev_4"].apply(lambda x : f"ciiu_rev_4-{x}")
    ciiu_rev_3_to_ciiu_rev_4_pd["ciiu_rev_3"] = ciiu_rev_3_to_ciiu_rev_4_pd["ciiu_rev_3"].apply(lambda x : f"ciiu_rev_3-{x}")
    ciiu_rev_3_to_ciiu_rev_4_pd["ciiu_rev_4"] = ciiu_rev_3_to_ciiu_rev_4_pd["ciiu_rev_4"].apply(lambda x : f"ciiu_rev_4-{x}")
    return (
        ciiu_rev_3_to_ciiu_rev_4,
        ciiu_rev_3_to_ciiu_rev_4_pd,
        ciiu_rev_4_to_scian_2023,
        ciiu_rev_4_to_scian_2023_pd,
    )


@app.cell
def _():
    return


@app.cell
def _(ciiu_rev_3_to_ciiu_rev_4, ciiu_rev_4_to_scian_2023, pl):
    ciiu_rev_3_to_ciiu_rev_4_to_scian_2023 = ciiu_rev_4_to_scian_2023.join(
        ciiu_rev_3_to_ciiu_rev_4, 
        on = "ciiu_rev_4", 
        how="left"
    )
    ciiu_rev_3_to_ciiu_rev_4_to_scian_2023 = ciiu_rev_3_to_ciiu_rev_4_to_scian_2023.unpivot(["ciiu_rev_3", "ciiu_rev_4"], index="scian_2023")

    ciiu_rev_3_to_ciiu_rev_4_to_scian_2023 = ciiu_rev_3_to_ciiu_rev_4_to_scian_2023.with_columns(
        pl.concat_str(
            pl.col("variable"), 
            pl.col("value"), 
            separator="-"
        ).alias(
            "ciiu"
        )
    )

    ciiu_rev_3_to_ciiu_rev_4_to_scian_2023 = ciiu_rev_3_to_ciiu_rev_4_to_scian_2023.drop(
        "variable", "value"
    ).rename(
        {
            "scian_2023" : "scian"
        }
    )

    ciiu_rev_3_to_ciiu_rev_4_to_scian_2023
    return


@app.cell
def _(ciiu_rev_3_to_ciiu_rev_4_pd, ciiu_rev_4_to_scian_2023_pd, nx):
    # empty graph/network
    G = nx.Graph()


    # add edges    
    for k in ciiu_rev_3_to_ciiu_rev_4_pd.index:
        i = tuple(ciiu_rev_3_to_ciiu_rev_4_pd.iloc[[k]]["ciiu_rev_3"].values)[0] 
        j = tuple(ciiu_rev_3_to_ciiu_rev_4_pd.iloc[[k]]["ciiu_rev_4"].values)[0] 
        G.add_edge(str(i),str(j)) 

    for k in ciiu_rev_4_to_scian_2023_pd.index:
        i = tuple(ciiu_rev_4_to_scian_2023_pd.iloc[[k]]["ciiu_rev_4"].values)[0] 
        j = tuple(ciiu_rev_4_to_scian_2023_pd.iloc[[k]]["scian_2023"].values)[0] 
        G.add_edge(str(i),str(j)) 
    return (G,)


@app.cell
def _(ciiu_rev_3_to_ciiu_rev_4_pd):
    tuple(ciiu_rev_3_to_ciiu_rev_4_pd.iloc[[1]]["ciiu_rev_3"].values)[0] 
    return


@app.function
def basic_stats(G):
    print("nodes: %d" % G.number_of_nodes())
    print("edges: %d" % G.number_of_edges())


@app.cell
def _(G):
    basic_stats(G)
    return


@app.cell
def _(G, ciiu_rev_3_to_ciiu_rev_4_pd, ciiu_rev_4_to_scian_2023_pd, nx):
    #gen sets
    set_i = ciiu_rev_4_to_scian_2023_pd[['scian_2023']]
    set_i = set_i.drop_duplicates(keep='first')
    set_i['set_type']=int(1)

    set_j = ciiu_rev_4_to_scian_2023_pd[['ciiu_rev_4']]
    set_j = set_j.drop_duplicates(keep='first')
    set_j['set_type']=int(2)

    set_h = ciiu_rev_3_to_ciiu_rev_4_pd[['ciiu_rev_3']]
    set_h = set_h.drop_duplicates(keep='first')
    set_h['set_type']=int(3)

    #set_i.scian = set_i.scian.astype(int)
    #set_j.ciiu = set_j.ciiu.astype(int)
    #set_h.naics = set_h.naics.astype(int)

    # node attributes
    node_type = set_i.set_index("scian_2023").set_type.to_dict()
    node_type.update(set_j.set_index("ciiu_rev_4").set_type.to_dict() )
    node_type.update(set_h.set_index("ciiu_rev_3").set_type.to_dict() )

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
def _():
    return


@app.cell
def _(BASE, HEADERS, cytoscape_network, json, requests):
    res1 = requests.post(BASE + 'networks', data=json.dumps(cytoscape_network), headers=HEADERS)
    res1_dict = res1.json()
    new_suid = res1_dict['networkSUID']
    requests.get(BASE + 'apply/layouts/force-directed/' + str(new_suid))
    return (new_suid,)


@app.cell
def _(BASE, HEADERS, Image, json, new_suid, requests):
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
        }, {
          "key" : "3",
          "value" : "#e59710"
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

    # Display
    Image(BASE+'networks/' + str(new_suid) + '/views/first.png')
    return


@app.cell
def _():
    from networkx.algorithms import community
    return (community,)


@app.cell
def _(G, community):
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
    codificacion["clasificador"] = "scian"
    codificacion.loc[codificacion["codigo"].apply(lambda x : "ciiu_rev_3" in x), "clasificador"] = "ciiu_rev_3"
    codificacion.loc[codificacion["codigo"].apply(lambda x : "ciiu_rev_4" in x), "clasificador"] = "ciiu_rev_4"

    ## Ajustamos código
    codificacion["codigo"] = codificacion["codigo"].apply(lambda x : x.split("-")[-1] if "rev" in x else x)
    codificacion
    return (codificacion,)


@app.cell
def _(codificacion):
    codificacion.query("codigo_nuevo==5")
    return


@app.cell
def _(codificacion):
    codificacion.query("clasificador=='ciiu_rev_4'")
    return


@app.cell
def _(pd):
    ### Agregamos nombres de actividades
    scian_2023_nombres = pd.read_csv("datos/cw/scian_2023_nombres_actividades.csv")
    ciiu_rev_4_nombres = pd.read_csv("datos/cw/ciiu_rev_4_nombres_actidades.csv")

    ciiu_rev_4_nombres["ciiu_rev_4"] = ciiu_rev_4_nombres["ciiu_rev_4"].astype(str)
    return ciiu_rev_4_nombres, scian_2023_nombres


@app.cell
def _(ciiu_rev_4_nombres, codificacion, scian_2023_nombres):
    # Creamos columna de nombre de la actividad
    codificacion["nombre_actividad"] = ""

    codificacion.loc[codificacion["clasificador"] == 'scian', "nombre_actividad"] = codificacion.loc[codificacion["clasificador"] == 'scian', "codigo"].replace({ i:j  for i,j in scian_2023_nombres.to_records(index = False)})

    codificacion.loc[codificacion["clasificador"] == 'ciiu_rev_4', "nombre_actividad"] = codificacion.loc[codificacion["clasificador"] == 'ciiu_rev_4', "codigo"].replace({ i:j  for i,j in ciiu_rev_4_nombres.to_records(index = False)})
    return


@app.cell
def _(codificacion):
    ", ".join(codificacion.query("clasificador == 'scian' and codigo_nuevo == 80")["nombre_actividad"])
    return


@app.cell
def _(codificacion):
    from keybert import KeyBERT

    doc = ", ".join(codificacion.query("clasificador == 'scian' and codigo_nuevo == 80")["nombre_actividad"])
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(doc)
    return KeyBERT, doc, kw_model


@app.cell
def _(doc, kw_model):
    kw_model.extract_keywords(doc, keyphrase_ngram_range=(1,4), stop_words=None)
    return


@app.cell(disabled=True)
def _(KeyBERT, codificacion):
    for codigo_nuevo in codificacion["codigo_nuevo"].unique():
        doc_ = ", ".join(codificacion.query(f"clasificador == 'scian' and codigo_nuevo == {codigo_nuevo}")["nombre_actividad"])
        kw_model_ = KeyBERT()
        keywords_ = kw_model_.extract_keywords(doc_)

        print(f"--------------- {codigo_nuevo} -----------------")
        print(doc_)
        print(kw_model_.extract_keywords(doc_, keyphrase_ngram_range=(1,4), stop_words=None))
    return


@app.cell
def _(codificacion):
    codificacion.to_csv("datos/recodificacion/recodificacion.csv", index = False)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
