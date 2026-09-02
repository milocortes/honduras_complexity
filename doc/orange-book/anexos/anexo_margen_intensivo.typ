#import "@preview/orange-book:0.7.1": book, part, chapter, my-bibliography, appendices, make-index, index, theorem, definition, notation,remark,corollary,proposition,example,exercise, problem, vocabulary, scr, update-heading-image

#chapter("Resultados detallados de la priorización de sectores en el Margen Intensivo")//, image: image("./honduras1.jpg"))

#figure(
  table(
  columns: 6,
  stroke: 0.5pt + black,

    fill: (x, y) => {
      if y == 0 { rgb("d1e7dd") }      // Header color
      else if x == 0 { rgb("f8f9fa") } // Nested/Rowspan column color
    },
  align: center + horizon, // Centers text horizontally and vertically
  table.hline(y: 1, stroke: 1.5pt),
  table.hline(y: 27, stroke: 1.5pt),
  table.header(
    [Clúster],[Clave CIIU4],[Actividad CIIU4],[Empleo],[RCA],[PCI]
  ),
  [C1 Manufactura avanzada y metalmecánica], [2651], [  Fabricación de equipo de medición, prueba, navegación y control], [1629], [1.02], [3.78],
  table.cell(rowspan: 5)[C2 Química, materiales y farmacéutica], [2013], [  Fabricación de plásticos y caucho sintético en formas primarias], [868], [1.31], [2.93],
  [2100], [  Fabricación de productos farmacéuticos, sustancias químicas medicinales y productos botánicos de uso farmacéutico], [2869], [1.28], [0.08],
  [2391], [  Fabricación de productos refractarios], [112], [1.15], [-0.59],
  [2394], [  Fabricación de cemento, cal y yeso], [676], [2.74], [-1.19],
  [2023], [  Fabricación de jabones y detergentes, preparados para limpiar y pulir, perfumes y preparados de tocador], [2090], [1.53], [-1.25],
  table.cell(rowspan: 5)[C3 Agroindustria y alimentos procesados], [1062], [  Elaboración de almidones y productos derivados del almidón], [330], [5.05], [1.32],
  [1103], [  Elaboración de bebidas malteadas y de malta], [925], [2.42], [-0.31],
  [1050], [  Elaboración de productos lácteos], [4783], [2.74], [-0.63],
  [1080], [  Elaboración de piensos preparados para animales], [2036], [3.47], [-0.88],
  [1020], [  Elaboración y conservación de pescado, crustáceos y moluscos], [2923], [3.46], [-1.11],
  [#table.cell(rowspan: 2)[C4 Servicios empresariales intensivos en conocimiento (KIBS)]], [7320], [  Estudios de mercado y encuestas de opinión pública], [1237], [2.53], [-0.37],
  [8220], [  Actividades de centros de llamadas], [3662], [1.55], [-2.74],
  [#table.cell(rowspan: 6)[C5 Turismo]], [7990], [  Otros servicios de reservas y actividades conexas], [962], [3.38], [1.45],
  [7721], [  Alquiler y arrendamiento de equipo recreativo y deportivo], [178], [1.63], [0.6],
  [5110], [  Transporte de pasajeros por vía aérea], [1009], [1.47], [0.3],
  [5011], [  Transporte de pasajeros marítimo y de cabotaje], [576], [1.33], [-0.2],
  [5222], [  Actividades de servicios vinculadas al transporte acuático], [1560], [3.06], [-0.54],
  [7710], [  Alquiler y arrendamiento de vehículos automotores], [795], [1.08], [-0.77],
  [#table.cell(rowspan: 7)[C6 Industria textil y de confección]], [2030], [  Fabricación de fibras artificiales], [857], [7.39], [1.1],
  [1313], [  Acabado de productos textiles], [71320], [100.29], [-1.28],
  [1391], [  Fabricación de tejidos de punto y ganchillo], [2839], [26.87], [-1.57],
  [1311], [  Preparación e hilatura de fibras textiles], [9914], [31.93], [-2.14],
  [1399], [  Fabricación de otros productos textiles n.c.p.], [3752], [6.3], [-2.64],
  [1392], [  Fabricación de artículos confeccionados de materiales textiles, excepto prendas de vestir], [9677], [13.38], [-3.06],
  [1410], [  Fabricación de prendas de vestir, excepto prendas de piel], [16142], [4.69], [-3.35]
),  caption: [Priorización de actividades en el Margen Intensivo (actividades en las que se especializa Honduras)],
) 
