#import "@preview/orange-book:0.7.1": book, part, chapter, my-bibliography, appendices, make-index, index, theorem, definition, notation,remark,corollary,proposition,example,exercise, problem, vocabulary, scr, update-heading-image

#import "@preview/thmbox:0.3.0": *
#import "@preview/mitex:0.2.6": *

#let ukj-blue = rgb(0, 84, 163)

#chapter("Definición y cálculo de los Factores de Viabilidad y Atractivo")//, image: image("./honduras1.jpg"))


Para robustercer la estrategia de desarrollo productivo y las recomendaciones de política, las actividades priorizadas se analizan mediante diferentes métricas de viabilidad y atractivo.


- #text(fill: ukj-blue)[*Fortaleza en países como Honduras (RCA en el grupo de pares)*]: Elasticidad promedio de las industrias en Ecuador y El Salvador.​

- #text(fill: ukj-blue)[*Disponibilidad de Insumos*]: Razón de productos disponibles o presentes por industria. Para la construcción de este indicador se usó la metodología de Liao et al (2020) quienes descomponen la industria CIIU por los productos que la intengra, ponderado por el peso relativo de cada producto en la industria. Además, usamos los datos de AI-generated Production Network - AIPNET para identificar la cadena de producción de los productos. Para cada producto, calculamos la razón de productos disponibles en el país al contabilizar la cantidad de productos en el país que tienen RCA mayor o igual a 1 con respecto al total de productos que se necesita para la producción. Con esta razón de productos disponibles por producto, usamos los ponderadores de Liao et al (2020) para calcular la razón de disponibilidad por industria al multiplicar y sumar la razón de productos disponibles por producto y los ponderadores del peso relativo del producto en la industria.

- #text(fill: ukj-blue)[*Dependencia o restricción potencial (Electricidad)*]: El indicador es la razón entre el Gasto por consumo de energía eléctrica y los Gastos Totales por consumo de bienes y servicios de la industria (Fuente: Censos Económicos 2023, INEGI).

- #text(fill: ukj-blue)[*La intensidad institucional*]: Es la proporción de insumos intermedios que no pueden adquirirse en mercados organizados y que no tienen un precio de referencia (Fuente : Levchenko, A. A. (2013). International trade and institutional change. The Journal of Law, Economics, & Organization, 29(5), 1145-1181). Si esa proporción es cercana a 1, es negativo, los insumos y sus precios están regidos por mecanismos desorganizados (mercados negros, mercados criminales, mercados informales, etc.).

- #text(fill: ukj-blue)[*Monto acumulado de inversión en capital (LAC)*]: Monto acumulado de la inversión en capital entre 2019 y 2024 en América Latina (Fuente: FDI Markets).

- #text(fill: ukj-blue)[*Tasa de crecimiento de la inversión (LAC)*]: Tasa de crecimiento compuesta de la inversión entre 2019 y 2024 en América Latina (Fuente: FDI Markets).​

- #text(fill: ukj-blue)[*Elasticidad Empleo/Inversión (LAC)*]: Mide cómo responde el empleo a los cambios en la inversión extranjera directa (IED) en un sector específico. Indica cuánto crece el empleo de la industria por cada 1 % de aumento en el crecimiento sectorial del FDI (Fuente: con datos de FDI Mrkets).​

- #text(fill: ukj-blue)[*Crecimiento de la Producción mundial*]: Crecimiento de la Producción de las industrias en el mundo (Fuente: OECD Structural Business Statistics).

- #text(fill: ukj-blue)[*Crecimiento de las Exportaciones mundiales*]: Calculamos el crecimiento de la industria CIIU al calcular el crecimiento en exportaciones de los productos que componen a cada industria. Siguiendo la metodología de Liao et al (2020), se descompone la industria CIIU por los productos que la integran y se pondera por el peso relativo de cada producto en la industria. Con estos ponderadores se calcula el crecimiento exportador de la industria en el mundo usando los datos del Atlas de Complejidad Económica de Harvard.

- #text(fill: ukj-blue)[*Dependencia de EU de importaciones desde China*]: Se calcula a partir de la razón promedio ponderada de la industria CIIU a ser importada por EU desde China. Siguiendo la metodología de Liao et al (2020) y a partir de los datos de exportaciones del Atlas de Complejidad Económica de Harvard, se calcula la razón de importación por producto proveniente de China con respecto al total de importación para EU. Con el peso relativo de cada producto en la industria se calcula la razón promedio ponderada de la industria.

- #text(fill: ukj-blue)[*Capacidad para crear empleo*]: Elasticidad de crecimiento del empleo al crecimiento del producto de la industria. Este indicador mide cómo responde el empleo a los cambios en el producto de la industria. Indica cuánto crece el empleo de la industria por cada 1 % de aumento en el producto (Fuente: Con datos de OECD Structural Business Statistics).

