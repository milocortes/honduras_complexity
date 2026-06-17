# Métricas de Viabilidad y Atractiv

Se calcularon las siguientes medidas de Viabilidad y Atractivo:

**Attractiveness**:

- Capacidad para movilizar FDI (world and region)
- ⁠Industry growth worldwide (past five years)
- ⁠Industry growth worldwide (past five years-Atlas export growth)
- ⁠Possibility to substitute US imports from Asia (China)
- ⁠Capacity to create employment among specific groups (women, youth, low-skill)

**Viability**:
- Strength in countries like Honduras (RCA in peer group)
- ⁠⁠Availability of inputs (doble razor, let us talk) 
- Reliance on a constraint or potential constraint (energy, security) 
- Reliance on a constraint or potential constraint (electricity-SCIAN México) 

### **Attractiveness**
#### Capacidad para movilizar FDI (world and region)

- Monto acumulado del monto acumulado de inversión en capital y creacion de empleo entre 2019 y 2024 en el mundo.

- Monto acumulado de inversión en capital y creacion de empleo entre 2019 y 2024 en América Latina.

- Tasa de crecimiento compuesta de la inversión entre 2019 y 2024 en el mundo.

- Tasa de crecimiento compuesta de la inversión entre 2019 y 2024 en América Latina.

- Elasticidad de crecimiento del empleo al crecimiento de la inversión.
	
  - Este indicador mide cómo responde el empleo a los cambios en la inversión extranjera directa (IED) en un sector específico. Indica cuánto crece el empleo de la industria por cada 1 % de aumento en el crecimiento sectorial del FDI.
  
  $$
  \begin{equation}
  \text { Elasticity }(\epsilon)=\frac{\% \text { Change in Employment }}{\% \text { Change in FDI }}
  \end{equation}
  $$
  * If ε > 0 and < 1, the sector creates jobs but FDI is also rising.
  * If ε > 1, the sector is highly labor-intensive and creates many jobs relative to FDI.
  
  ```

Fuente de Datos : FDI Markets

#### ⁠Industry growth worldwide (past five years)

- Crecimiento del empleo de las industrias en el mundo. Fuente : OECD SBP.

#### ⁠Industry growth worldwide (past five years-Atlas export growth)

- Calculamos el crecimiento de la industria CIIU al calcular el crecimiento en exportaciones de los productos que componen a cada industria. De acuerdo a la metodología de Liao et al (2020) podemos descomponer la industria CIIU por los productos que la intengra, ponderado por el peso relativo de cada producto en la industria. Con tales ponderadores podemos crear con los datos del Atlas de Complejidad Económica un indicador del crecimiento exportador de la industria en el mundo. Fuente : Atlas de Complejidad 

#### ⁠Possibility to substitute US imports from Asia (China)

- Se calcula la posibilidad de sustituir importaciones de China en USA al calcular la razón promedio ponderada de la industria CIIU a ser importada por USA desde China. Usando la metodología de Liao et al (2020) se calcula la razón de importación por producto proveniente de China con respecto al total de importación para USA. Con el peso relativo de cada producto en la industria se calcula la razón promedio poderada de la industria. Fuente : Atlas de Complejidad 

#### ⁠Capacity to create employment 

- Elasticidad de crecimiento del empleo al crecimiento del producto de la industria.

  - Este indicador mide cómo responde el empleo a los cambios en el producto de la industria. Indica cuánto crece el empleo de la industria por cada 1 % de aumento en el producto.

  $$
  \begin{equation}
  \text { Elasticity }(\epsilon)=\frac{\% \text { Change in Employment }}{\% \text { Change in Output }}
  \end{equation}
  $$

  * If ε > 0 and < 1, the sector creates jobs but Output is also rising.
  * If ε > 1, the sector is highly labor-intensive and creates many jobs relative to Output.

  ```
  
  ```

### **Viability**
#### Strength in countries like Honduras (RCA in peer group)

* Elasticidad promedio de las industrias en Ecuador y El Salvador.

#### ⁠Availability of inputs (doble razor, let us talk)

* Se calculo una razón de productos disponibles o presentes por industria.

* Para la construcción de este indicador, se usó la metodología de Liao et al (2020) podemos descomponer la industria CIIU por los productos que la intengra, ponderado por el peso relativo de cada producto en la industria. Además, usamos los datos de [AI-generated Production Network - AIPNET](https://aipnet.io/) para identificar la cadena de producción de los productos.
* Para cada producto, calculamos la razón de productos disponibles en el país al contabilizar la cantidad de productos en el país que tienen RCA mayor o igual a 1 con respecto al total de productos que se necesita para la producción. 
* Con esta razón de productos disponibles **por producto**, usamos los ponderadores de Liao et al (2020) para calcular la razón de disponibilidad **por industria** al multiplicar y sumar la razón de productos disponibles por producto y los ponderadores del peso relativo del producto en la industria.
* Fuente : Atlas de Complejidad

#### Reliance on a constraint or potential constraint (energy, security)

* El indicador es la razón entre el valor de las compras de productos de energía y el valor total de las compras en bienes y servicios de la industria.
* Fuente : OECD Structural Business Statistics

#### Reliance on a constraint or potential constraint (electricity-SCIAN México)

* El indicador es la razón entre el Gasto por consumo de energía eléctrica y los Gastos Totales por consumo de bienes y servicios de la industria.
* Fuente : Censos Económicos 2023, INEGI.
