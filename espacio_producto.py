import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Espacio de Industrias
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    from ecomplexity import proximity
    import pandas as pd
    return math, np, nx, pd, pl, plt, proximity


@app.cell
def _(pd, pl):
    ## Cargamos datos de empleo
    empleo_hnd = pl.read_csv("output/recodificacion/empleo/honduras.csv")
    empleo_cbp = pd.read_csv("output/recodificacion/empleo/cbp.csv")

    empleo_industrias_cbp = list(empleo_cbp["actividad"].unique())

    empleo = pd.concat(
        [empleo_cbp, 
         empleo_hnd.filter(pl.col("actividad").is_in(empleo_industrias_cbp)).to_pandas()
        ], 
        ignore_index=True
    )
    empleo["year"] = 2024
    empleo
    return (empleo,)


@app.cell
def _(pd, pl):
    ## Cargamos datos de UE
    ue_hnd = pl.read_csv("output/recodificacion/ue/honduras.csv")
    ue_cbp = pd.read_csv("output/recodificacion/ue/cbp.csv")

    ue_industrias_cbp = list(ue_cbp["actividad"].unique())

    ue = pd.concat(
        [ue_cbp, 
         ue_hnd.filter(
             pl.col("actividad").is_in(ue_industrias_cbp)
         ).to_pandas()
        ], 
        ignore_index=True
    )
    ue["year"] = 2024

    ue
    return (ue,)


@app.cell
def _(empleo, proximity, ue):
    # Calculate complexity for UE
    trade_cols = {'time':'year', 'loc':'zona', 'prod':'actividad', 'val':'valor'}
    ue_prox = proximity(ue, trade_cols).query("proximity!=1 and proximity>0").drop(columns = "year").rename(columns = {"proximity" : "weight"}) 
    empleo_prox = proximity(empleo, trade_cols).query("proximity!=1 and proximity>0").drop(columns = "year").rename(columns = {"proximity" : "weight"}) 
    return empleo_prox, ue_prox


@app.cell
def _():


    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Product Space

    To visualize the product space we use some simple design criteria. First, we want the visualization of the product space to be a connected network. By this, we mean avoiding islands of isolated products. The second criterion is that we
    want the network visualization to be relatively sparse. Trying to visualize too many links can create unnecessary visual complexity where the most relevant connections will be occluded. This is achieved by creating a visualization in which the average number of links per node is not larger than 5 and results in a representation that can summarize the structure of the product space using
    the strongest 1% of the links

    To make sure the visualization of the product space is connected, we calculate the maximum spanning tree (MST) of the proximity matrix. MST is the set of links that connects all the nodes in the network using a minimum number of connections and the maximum possible sum of proximities. We calculated the MST using Kruskal’s algorithm. Basically the algorithm sorts the values of the
    proximity matrix in descending order and then includes links in the MST if and only if they connect an isolated product. By definition, the MST includes all products, but the number of links is the minimum possible.

    The second step is to add the strongest connections that were not selected for the MST. In this visualization we included the first 1,006 connections satisfying our criterion. By definition a spanning tree for 774 nodes contains 773
    edges. With the additional 1,006 connections we end up with 1,779 edges and an average degree of nearly 4.6.

    After selecting the links using the above mentioned criteria we build a visualization using a force-directed layout algorithm. In this algorithm nodes repel each other, just like electric charges, while edges act as spring trying to bring connected nodes together. This helps to create a visualization in which densely connected sets of nodes are put together while nodes that are not connected are pushed apart.

    Finally, we manually clean up the layout to minimize edge crossings and provide the most clearly representation possible.
    """)
    return


@app.cell
def _(nx, ue_prox):
    ## Crea gráficas con NetworkX
    G_ue = nx.from_pandas_edgelist(
        ue_prox,
        source='actividad_1', 
        target='actividad_2', 
        edge_attr=['weight']
    )

    ## Construye Maximum Spanning Tree
    mst_ue = nx.maximum_spanning_tree(G_ue)
    return (mst_ue,)


@app.cell
def _(empleo_prox, nx):
    ## Crea gráficas con NetworkX
    G_empleo = nx.from_pandas_edgelist(
        empleo_prox,
        source='actividad_1', 
        target='actividad_2', 
        edge_attr=['weight']
    )

    ## Construye Maximum Spanning Tree
    mst_empleo = nx.maximum_spanning_tree(G_empleo)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exponential Random Graph

    Para lograr crear una visualización en la que el grado promedio por nodo sea alrededor de 6 y resulte en una representación que pueda resumir la estructura del espacio producto usando el 1% de los links más fuertes, usamos el método de Monte Carlo via Cadenas de Markov en Gráficas Aleatorias Exponenciales para rediseñar la red para que tenga las promiedad de grado promedio que necesitamos, la propiedad observable que desearíamos tener.
    """)
    return


@app.cell
def _(ue_prox):
    # Multiplicador de Lagrange que define el número promedio de enlaces
    θ = 5.0

    # Edges candidatas a ser conectadas
    candidatas = [(i,j) for i,j in ue_prox.query("weight!=1 and (weight < 0.65 and weight>0.35 )")[["actividad_1", "actividad_2"]].to_records(index = False)]


    return candidatas, θ


@app.cell
def _(candidatas, math, np, nx, plt):
    def erg_mcmc(G : nx.Graph, 
                 θ : float, 
                 iteraciones : int, 
                 edges_candidatas : list) -> nx.Graph:

        G_erg = G.copy()
    
        # Lista que monitorea el cambio en los enlaces promedio por nodo
        promedio_enlaces = []

        # Comenzamos el algoritmo
        for it in range(iteraciones):
    
            # Compute RM
            if it % 10000 == 0:
                print(f"It: {it}")
            
            candidatas_ids = list(range(len(edges_candidatas)))
            candidatas_ids
    
            candidato_id = np.random.choice(candidatas_ids)
        
            edge = candidatas[candidato_id]
    
            i,j = edge
        
            if G_erg.has_edge(i,j):
                δ = -1
            else:
                δ = 1
        
            # Verificamos si el cambio es aceptado
            r = np.random.uniform(0,1) 
        
            if r <= math.exp(-θ * δ):
                if δ == -1:
                    pass
                else:
                    G_erg.add_edge(i,j)
                
            ## Monitoreamos cantidad promedio de edges
            # Calculate the sum of degrees
            sum_of_degrees = sum(dict(G_erg.degree()).values())
        
            # Get the number of nodes
            num_nodes = G_erg.number_of_nodes()
        
            # Calculate the global average degree
            global_average_degree = sum_of_degrees / num_nodes
        
            promedio_enlaces.append(global_average_degree)
        

        ## Grafiquemos el grado promedio por nodo de cada iteración
        plt.plot(promedio_enlaces)
        plt.xlabel("Iteración")
        plt.ylabel("Average number of links per node")
        plt.show()
        return G_erg
    return (erg_mcmc,)


@app.cell
def _(candidatas, erg_mcmc, mst_ue, θ):
    mst_erg_ue = erg_mcmc(mst_ue, θ, 100_000, candidatas)

    return (mst_erg_ue,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exportamos las redes rediseñadas en formato .graphml para obtener el layout ForceAtlas en Gephi
    """)
    return


@app.cell
def _(mst_erg_ue, nx):
    ### Export to GraphML
    nx.write_graphml(mst_erg_ue, "mst_erg_ue.graphml")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cargamos las redes con el layout ForceAtlas para rotarlo
    """)
    return


@app.cell
def _(nx):
    # Importing a gexf graph
    mst_erg_force_atlas = nx.read_gml('mst_erg_ue_force_atlas.gml')

    return (mst_erg_force_atlas,)


@app.cell
def _(np, nx, plt):
    ## Verificamos el layout de Gephi
    def grafica_force_atlas(
            G : nx.Graph,
            var_name : str
        ) -> None:

        new_pos = {node : np.array([G.nodes[node]["graphics"]["x"], G.nodes[node]["graphics"]["y"] ])for node in G.nodes  }
        # Draw the graph
        nx.draw(G, new_pos, with_labels=True, node_color='skyblue', node_size=100, font_size=5)
        plt.title(f"Force-Directed Layout {var_name}")
        plt.show()
    return (grafica_force_atlas,)


@app.cell
def _(grafica_force_atlas, mst_erg_force_atlas):
    grafica_force_atlas(mst_erg_force_atlas, "VA")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rotamos la red hasta tener la representación que necesitamos
    """)
    return


@app.cell
def _(math, np, nx, plt):
    def nuevo_vector(punto : np.array, 
                     theta : float
        ) -> np.array:
        vector_rotado = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)] 
            ]
        ) @  punto
    
        return vector_rotado

    def rotacion(
        G : nx.Graph,
        grados : float,
        var_name : str
        ) -> dict:

        new_pos = {node : np.array([G.nodes[node]["graphics"]["x"], G.nodes[node]["graphics"]["y"] ])for node in G.nodes  }

        x_min = min(np.array([i for i in new_pos.values()])[:,0])
        y_min = min(np.array([i for i in new_pos.values()])[:,1])
    
        pos_positivos = {rama : point - np.array([x_min, y_min]) for rama, point in new_pos.items()}

        pos_rotada = {rama : list(nuevo_vector(point, grados)) for rama, point in pos_positivos.items()}
    
        # Draw the graph
        nx.draw(G, pos_rotada, with_labels=True, node_color='skyblue', node_size=100, font_size=5)
        plt.title(f"Force-Directed Layout {var_name}")
        plt.show()

        return pos_rotada

    return (rotacion,)


@app.cell
def _(mst_erg_force_atlas, rotacion):
    pos_rotada = rotacion(mst_erg_force_atlas, 72.6, "VA")
    return (pos_rotada,)


@app.cell
def _(mst_erg_ue, nx, pos_rotada):
    ### Export to GraphML
    ## Ajusta la posición para que esten en atributos x,y no en un arreglo de numpy

    for node,(x,y) in pos_rotada.items():
        node = int(node)
        mst_erg_ue.nodes[node]['x'] = float(x)
        mst_erg_ue.nodes[node]['y'] = float(y)
    
    nx.write_graphml(mst_erg_ue, "mst_erg_force_atlas_rotado.graphml")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
