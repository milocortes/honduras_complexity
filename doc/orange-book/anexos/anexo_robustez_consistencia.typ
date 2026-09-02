#import "@preview/orange-book:0.7.1": book, part, chapter, my-bibliography, appendices, make-index, index, theorem, definition, notation,remark,corollary,proposition,example,exercise, problem, vocabulary, scr, update-heading-image

#import "@preview/thmbox:0.3.0": *
#import "@preview/mitex:0.2.6": *

#let ukj-blue = rgb(0, 84, 163)

#chapter("Pruebas de Consistencia y Robustez de los Resultados")//, image: image("./honduras1.jpg"))

Acorde con la literatura internacional de complejidad económica, los resultados de nuestro análisis encuentran una relación
positiva entre el Índice de Complejidad Económica (ECI) de los países analizados y el PIB per cápita de los mismos.

#figure(
  image("../images/anexos/robustez_01.svg"),
  caption: [GDP per capita vs ECI. Datos de Empleo OECD SBS 2019],
)

Nuestra estimación del Índice de Complejidad Económica (ECI) de los países analizados a partir de los datos de empleo, son
consistentes con las estimaciones del Atlas de Complejidad Económica de la Universidad de Harvard en donde utilizan datos
de exportaciones de productos.

#figure(
  image("../images/anexos/robustez_02.svg"),
  caption: [GDP per capita vs ECI. Datos de Empleo OECD SBS 2019 y Atlas de Complejidad​ Económica],
)

Con base en lo anterior, el ranking de los países analizados es consistente entre nuestros resultados y las estimaciones del
Atlas de Complejidad Económica de la Universidad de Harvard en donde utilizan datos de exportaciones de productos.

#figure(
  image("../images/anexos/robustez_03.svg"),
  caption: [Comparación de Rankings de Países. Datos de Empleo OECD SBS 2019],
)

Acorde con la literatura internacional de complejidad económica, los resultados de las métricas de densidad y el Índice de
Complejidad Económica de las Actividades (PCI), presentan una relación negativa. Esto es, la estructura productiva de
Honduras está más relacionada con actividades económicas de menor complejidad.

#figure(
  image("../images/anexos/robustez_04.svg"),
  caption: [Diagrama Distancia-PCI. Honduras. Datos de Empleo OECD SBS 2019],
)

Para hacer más claro el punto anterior, este slide presenta la relación entre las métricas de densidad y el Índice de
Complejidad Económica de las Actividades (PCI) de Alemania. Como se esperaría, en el caso de países productivamente más
desarrollados, la relación se vuelve positiva.

#figure(
  image("../images/anexos/robustez_05.svg"),
  caption: [Diagrama Distancia-PCI. Alemania. Datos de Empleo OECD SBS 2019],
)
