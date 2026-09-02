#set text(size : 11pt, font: "Lato")

En los últimos 30 años, Honduras ha registrado el crecimiento económico más bajo entre sus pares, manteniéndose rezagado de forma persistente con la excepción del periodo 2000–2010.​

#figure(
  image("../images/ce/ce_img_01.pdf", page: 1),
  caption: [PIB per cápita 1990-2024. Dólares Constantes de 2015],
)

El crecimiento ha estado impulsado por el consumo, con una alta dependencia de las remesas, que ascienden al 30% del PIB, y que, a la vez, ha generado un déficit comercial cada vez más grande​. El Consumo explica típicamente la mayor parte del crecimiento del PIB en Honduras​. Las exportaciones han aportado cada vez menos al crecimiento económico, aunque a inicios de los años 2000 su contribución fue importante.​ El déficit comercial se ha ampliado considerablemente, debido a que las importaciones han crecido a ritmos mucho más acelerados que las exportaciones​

#figure(
  image("../images/ce/ce_img_02.pdf", page: 1),
  caption: [Construcciones del PIB diferentes periodos (componentes del PIB en p.p)],
)

El desempeño exportador ha estado rezagado frente al de sus pares, con excepción de la década 2000-2010, donde fueron impulsadas por los sectores agrícola y textil​. Salvo en la década de 2000–2010, el desempeño exportador ha quedado rezagado frente al de los países pares.​


#figure(
  image("../images/ce/ce_img_03.pdf", page: 1),
  caption: [Exportaciones 1990-2023. Dólares constantes de 2015. Índice 1990=0],
)

Consistente con este patrón exportador y el impulso de las remesas sobre el consumo, la economía se ha orientado hacia el sector no transable, que alcanza el 66% del valor agregado.​ No solo que la productividad está estancada, la estructura productiva se está reorientando a una de bienes no transables que dependen de la demanda interna.​

#figure(
  image("../images/ce/ce_img_04.pdf", page: 1),
  caption: [Composición del VA y crecimiento de los sectores transables y no transables (2000-2024)],
)

= La Complejidad Económica para el diseño de las estrategias de política industrial y diversificación productiva
Para reorientar la economía hacia sectores transables y que las exportaciones retomen un papel principal en el crecimiento económico, se propone el análisis de complejidad económica​.

El análisis de complejidad económica permite identificar el stock de capacidades productivas de un país y oportunidades de diversificación a través de la reutilización de esas capacidades.

Hay dos principios para entender la importancia de la Complejidad Económica en el diseño estrategias de política industrial y de diversificación productiva:​

- El desarrollo económico sostenido se explica como un fenómeno iterativo de acumulación de capacidades productivas que permiten a las regiones producir una mayor diversidad de bienes y servicios con un mayor nivel de sofisticación o mayor complejidad económica​.
- Los países tienden a diversificarse hacia bienes y servicios que requieren capacidades productivas similares a las de industrias existentes​

#figure(
  image("../images/ce/ce_img_05.pdf", page: 1),
  caption: [GDP per Cápita PPP vs Indice de Complejidad Económica],
)

Las capacidades productivas no son observables, pero se pueden inferir con base en un análisis de los productos y servicios que un país es capaz de producir de forma competitiva.​

#figure(
  image("../images/ce/ce_img_06.pdf", page: 1),
  caption: [Composición de Exportaciones de Japón 2024],
)

#figure(
  image("../images/ce/ce_img_07.pdf", page: 1),
  caption: [Composición de Exportaciones de Chad 2024],
)


En 2024, Honduras exportaba con ventaja comparativa 129 productos (RCA>=1), por debajo de otros países comparables en la región como Guatemala (180) y El Salvador (157).​

#figure(
  image("../images/ce/ce_img_08.pdf", page: 1),
  caption: [Composición de Exportaciones de Honduras 2024],
)

En general, las exportaciones más importantes de Honduras – con excepciones de algunos productos de menores niveles de exportación - presentan un bajo nivel de sofisticación o complejidad económica.​

#figure(
  image("../images/ce/ce_img_09.pdf", page: 1),
  caption: [Complejidad Económica de las Exportaciones de Honduras 2024],
)

El marco de complejidad económica, aplicando análisis y métricas de redes complejas, permite medir la similitud del know-how y el nivel de las capacidades requeridas para producir bienes y servicios

En este ejemplo, cada nodo es un producto
La proximidad entre dos nodos es una medida de la similitud del know-how de las  capacidades productivas requeridas para desarrollar ambos productos
Los países y las regiones se diversifican “saltando” hacia productos adyacentes que requieran capacidades productivas similares a las ya existentes en el lugar

El marco de complejidad económica nos permite estimar estas y otras métricas relevantes para la priorización de industrias y el diseño de estrategias de desarrollo#footnote[El Anexo 1 incluye la descripción metodológica de las principales métricas de complejidad económica.​]: 

- Índice de Complejidad del Producto (PCI): Nivel de sofisticación del conocimiento y capacidades necesarias para producir una actividad o producto
- Ganancia de oportunidad (COG): Valor estratégico de una actividad para abrir nuevas oportunidades de diversificación hacia sectores más complejos
- Distancia: Grado de cercanía entre las capacidades actuales y las necesarias para desarrollar una nueva actividad

#figure(
  image("../images/ce/ce_img_10.pdf", page: 1),
  caption: [Espacio-Producto. Atlas de Complejidad Económica],
)

Mediante estos análisis es posible identificar que la cesta exportadora de Honduras de los últimos 30 años se ha mantenido concentrada en los sectores de agricultura y textiles#footnote[En este ejemplo, cada nodo es un producto, se encuentra encendido (coloreado) si Honduras lo exporta con VCR>1​]. 

#figure(
  image("../images/ce/ce_img_11.pdf", page: 1),
  caption: [Productos con RCA > 1 de Honduras 1995],
)

Para 2024, por un lado se habían perdido algunos productos del cluster de textiles, y por el otro, agregado nuevos productos químicos, de maquinaria y construcción en áreas centrales del espacio de productos#footnote[Adicionalmente, a partir de datos administrativos de fuentes nacionales, se incluyeron: Honduras, El Salvador y Ecuador. En el Anexo 2 se incluyen los pasos que se siguieron para definir la base final de este análisis.​].​

#figure(
  image("../images/ce/ce_img_12.pdf", page: 1),
  caption: [Productos con RCA > 1 de Honduras 2024],
)

= Datos usados para el Análisis de Complejidad
Se analizaron y combinaron distintas bases de datos para hacer la mejor radiografía posible de la estructura productiva de Honduras, incluyendo producción doméstica y mediante servicios.​

#figure(
  image("../images/ce/ce_img_13.pdf", page: 1),
  caption: [Bases de Datos Exploradas],
)

Para asegurar que la base SBS de la OECD fuera comparable se siguieron los siguientes pasos:

#figure(
  image("../images/anexos/bases_datos_02.pdf", page: 1),
  caption: [Proceso de Depuración y Limpieza de la base de datos SBS de la OECD],
)



