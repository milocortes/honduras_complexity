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
    import numpy as np 
    import math 

    ## Cargamos datos
    dee = pl.read_parquet("datos/BASE-DEE-2024.parquet")

    ## Creamos llave cod.municipio y municipio
    dee = dee.with_columns(
        municipio = pl.col("cod._municipio").cast(str) + "-" +pl.col("municipio").cast(str)
    )
    dee
    return dee, math, np, pl


@app.cell
def _(dee):
    ## Obtenemos conteos de UE

    dee_data = dee.select(
            "municipio", "codigo__ciiu_cuatro_digitos"
            ).group_by(
            "municipio", "codigo__ciiu_cuatro_digitos"
            ).len().rename(
                {
                    "municipio" : "municipio", 
                    "codigo__ciiu_cuatro_digitos" : "ciiu", 
                    "len" : "conteo"
                }
            )

    dee_data
    return (dee_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matriz de Especialización, $R$

    * Una Matriz de Especialización, $R$, se define al dividir cada entrada de la matriz $X_{c p}$ por la suma de sus respectivas filas o columnas.
    * Está medida se le conoce como **cociente de ubicación** o **Revealed Comparative Advantage, (RCA)**.


    \begin{equation}
        R_{c p}= \frac{x_{c p} / \Sigma_p x_{c p}}{\Sigma_c x_{c p} / \Sigma_c \Sigma_p x_{c p}}
    \end{equation}

    * Ubicaciones con $R_{c p} > 1$ se consideran **especializadas en la actividad $p$**.
    """)
    return


@app.cell
def _(dee_data, pl):
    datos_rca = dee_data.with_columns(
        rca = (
            pl.col("conteo")/pl.col("conteo").sum().over("municipio")
        ) /
        (
            pl.col("conteo").sum().over("ciiu")/pl.col("conteo").sum()
        )

    ).with_columns(
        pl.col("ciiu").cast(pl.Int64)
    )

    datos_rca
    return (datos_rca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Matriz de Especialización Binaria, $M$

     Definimos la Matriz de Especialización Binaria, $M$, como

    \begin{equation}
    M_{c p}=\left\{\begin{array}{lll}
    1 & \text { if } & R_{c p} \geq R^{\star} \\
    0 & \text { if } & R_{c p}<R^{\star}
    \end{array}\right.
    \end{equation}

    donde $R^{\star}=1$ cuando usamos $R$ y $R^{\star}=0.25$ cuando usamos $R^{\text{pop}}$
    """)
    return


@app.cell
def _(datos_rca, pl):
    ## Umbral de RCA
    rca_umbral = 1.0

    datos_rca_m = datos_rca.with_columns(
        M = pl.when(
            pl.col("rca")>= rca_umbral   
        ).then(
            pl.lit(1)
        ).otherwise(
            pl.lit(0)
        )
    )
    return (datos_rca_m,)


@app.cell
def _(datos_rca_m):
    M_df = datos_rca_m.pivot("ciiu", 
                      index = "municipio", 
                      values = "M"
                ).fill_null(0).sort("municipio")
    M_df
    return (M_df,)


@app.cell
def _(M_df, pl):
    M = M_df.select(
        pl.exclude("municipio")
    ).to_numpy()

    M
    return (M,)


@app.cell
def _(M, np):
    ## Calculamos diversidad
    diversidad = M.sum(axis = 1)
    D = np.diag(diversidad)

    ## Calculamos ubicuidad
    ubicuidad = M.sum(axis = 0)
    U = np.diag(ubicuidad)
    return D, U, diversidad, ubicuidad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Proximidad

    Hay múltiples formas de medir **Proximidad**. Algunas, como la **probabilidad condicional mínima**, miran a la colocalización o coaglomeración de actividades:

    \begin{equation}
      \phi_{p p^{\prime}}=\frac{\sum_c M_{c p} M_{c p^{\prime}}}{\max \left(M_p, M_{p,}\right)}
    \end{equation}
    """)
    return


@app.cell
def _(M, np, ubicuidad):
    proximity = M.T @ M / ubicuidad[np.newaxis, :]  
    proximity = np.minimum(proximity, proximity.T)
    proximity = np.nan_to_num(proximity)
    proximity
    return (proximity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Relatedness Density

    * Con la medida de proximidad, podemos calcular **Relatedness Density** como la fracción de actividades relacionadas presentes en una ubicación

    \begin{equation}
    \omega_{c p}=\frac{\sum_{p \prime} M_{c p \prime} \phi_{p p^{\prime}}}{\sum_{p,} \phi_{p p^{\prime}}} \quad \circ \quad \omega_{c p}=\frac{\sum_{c^{\prime}} M_{c^{\prime}} \phi_{c^{\prime} c}}{\sum_{c^{\prime}} \phi_{c^{\prime} c}}
    \end{equation}
    """)
    return


@app.cell
def _(M, np, proximity):
    density = (np.dot(M,proximity)/np.sum(proximity, axis=1))
    density = np.nan_to_num(density)
    density
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Distancia
      - La Proximidad mide la similaridad entre pares de actividades-productos.
      - Necesitamos una medida que cuantifique la **Distancia** entre las actividades especializadas en un país y las actividades donde no está especializada.


    \begin{equation}
    d_{c p}=\frac{\sum_{p'} (1 - M_{c p'}) \phi_{p p^{\prime}}}{\sum_{p,} \phi_{p p^{\prime}}}
    \end{equation}

    - La distancia nos da una idea de qué tan lejos está cada actividad dado el ecosistema productivo del país.
    """)
    return


@app.cell
def _(M, np, proximity):
    distancia = (np.dot((1 - M),proximity)/np.sum(proximity, axis=1))
    distancia = np.nan_to_num(distancia)
    distancia
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Economic Complexity

    - Las medidas de **Complejidad Económica** miden la capacidad económica con **métodos de reducción de dimensionalidad**.
    - Representan también funciones de producción generalizadas de dimensionalidad reducida.
    - Las medidas de complejidad económica pueden ser utilizadas para medir la presencia de múltiples factores económicos en una forma que es **agnóstica** sobre cuales podrían ser esos factores.
    - Formalmente, la complejidad $K_c$ de una ubicación $c$ y la complejidad $K_p$ de una actividad $p$ puede definirse como una función una de la otra:

    \begin{equation}
    K_c=f\left(M_{c p}, K_p\right)
    \end{equation}

    \begin{equation}
    K_p=g\left(M_{c p}, K_c\right)
    \end{equation}

    - Estas ecuaciones declaran que la complejidad de una ubicación es una función de la complejidad de las actividades que están presentes en esta, y viceversa.
    - Una economía es tan compleja como las actividades que puede realizar, y una actividad es tan compleja como los lugares que pueden realizarla.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Cómo medimos la complejidad?

    - La idea de medir la complejidad usando estas ecuaciones acopladas fue introducida por Cesar Hidalgo y Ricardo Haussman. Los autores utilizan promedios simples para $f$ y $g$.
    - Las medidas resultantes se conocen como **Índice de Complejidad Económica, (ECI; $K_c$)** y el **Índice de Complejidad de Producto, (PCI; $K_p$)**.
    - Estas medidas están definidas por el siguiente sistema de ecuaciones:

    \begin{equation}
    K_c=\frac{1}{M_c} \sum_p M_{c p} K_p
    \end{equation}

    \begin{equation}
    K_p=\frac{1}{M_p} \sum_c M_{c p} K_c
    \end{equation}

    Reemplazando la segunda ecuación en la primera:

    \begin{equation}
        K_c=\widetilde{M}_{c c}, K_{c t}
    \end{equation}

    donde

    \begin{equation}
    \widetilde{M}_{c c \prime}=\sum_p \frac{M_{c p} M_{c \prime p}}{M_c M_p}
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Originalmente el cálculo de ECI y PCI se definió mediante un método iterativo llamado  **algoritmo de reflexión** que primero calcula la diversidad y ubicuidad para posteriormente y luego utiliza recursivamente la información de uno para corregir el otro.
    - Se puede demostrar que el método de reflección es equivalente a encontrar los eigenvalores de la matriz $\widetilde{M}$

    \begin{equation}
    \tilde{M}=D^{-1} M U^{-1} M^{\prime}
    \end{equation}

    donde $D$ es la matriz diagonal formada a partir del vector de diversidad y $U$ es la matriz diagonal formada a partir del vector de ubicuidad

    En el contexto de datos de comercio entre países, podemos pensar a $\widetilde{M}$ como una matriz de diversidad-ponderada (o normalizada) que refleja qué tan similares son las canastas exportadoras de los dos países, es decir, que tan similares son sus patrones de especialización.

    De la ecuación anterior podemos ver que:

    \begin{equation}
    \tilde{M}=D^{-1} S
    \end{equation}

    donde $S = M U^{-1} M^\prime$ es una matriz de similaridad simétrica en que cada elemento $S_{c c'}$ representa los productos que el pais $c$ tiene en común con el país $c'$, ponderado o normalizado por la inversa de la ubicuidad de cada producto.
    """)
    return


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde cc
    M_tilde_cc = np.linalg.pinv(D) @ M @ np.linalg.pinv(U) @ M.T
    M_tilde_cc 
    return (M_tilde_cc,)


@app.cell
def _(D, M, U, np):
    # Calculamos M tilde pp
    M_tilde_pp = np.linalg.pinv(U) @ M.T @ np.linalg.pinv(D) @ M
    M_tilde_pp 
    return (M_tilde_pp,)


@app.cell
def _(M_tilde_cc, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    eigenvalues_cc, eigenvectors_cc = np.linalg.eig(M_tilde_cc)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    Kc = eigenvectors_cc[:, 1].astype(float)

    Kc
    return (Kc,)


@app.cell
def _(M_tilde_pp, np):
    ## Calculamos los eigenvectores y eigenvalores de la matriz M tilde cc
    eigenvalues_pp, eigenvectors_pp = np.linalg.eig(M_tilde_pp)

    ## Obtenemos el eigenvector asociado con el segundo eigenvalor más grande
    Kp = eigenvectors_pp[:, 1].astype(float)

    Kp 
    return (Kp,)


@app.cell
def _(Kc, diversidad, math, np):
    ## Adjust sign of ECI and PCI so it makes sense, as per book
    corr_mat = np.corrcoef(diversidad, Kc)
    s1 = math.copysign(1.0, corr_mat[0,1])
    return (s1,)


@app.cell
def _(Kc, Kp, np, s1):
    ## Ajustamos y normalizamos ECI y PCI
    eci = s1*(Kc - np.mean(Kc))/np.std(Kc)
    pci = (Kp - np.mean(Kp))/np.std(Kp)
    return eci, pci


@app.cell
def _(M_df, eci, pl):
    ### Reunimos los datos calculados 
    df_eci_test = pl.DataFrame(
          {
            "municipio" : M_df["municipio"], 
            "eci" : eci  
        }  
    )

    df_eci_test
    return


@app.cell
def _(M_df, dee, pci, pl):
    ### Reunimos los datos calculados
    df_pci_test = pl.DataFrame(
          {
            "ciiu" : [int(i) for i in M_df.columns[1:]], 
            "pci" : pci  
        }  
    )

    df_pci_test = df_pci_test.join(
        dee.select(
        "codigo__ciiu_cuatro_digitos", "descripcion_de_codigo_ciiu_cuatro_digitos"
    ).unique().rename(
        {
            "codigo__ciiu_cuatro_digitos" : "ciiu", 
            "descripcion_de_codigo_ciiu_cuatro_digitos" : "descripcion"
        }
        ),
        on = "ciiu" 
    )

    df_pci_test
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
