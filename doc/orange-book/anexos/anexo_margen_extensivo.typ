#import "@preview/orange-book:0.7.1": book, part, chapter, my-bibliography, appendices, make-index, index, theorem, definition, notation,remark,corollary,proposition,example,exercise, problem, vocabulary, scr, update-heading-image

#chapter("Resultados detallados de la priorización de sectores en el Margen Extensivo")//, image: image("./honduras1.jpg"))

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
    [Clúster],[Clave CIIU4],[Actividad CIIU4],[COG],[PCI],[Density]
  ),
  [#table.cell(rowspan: 11)[C1 Manufactura avanzada y metalmecánica]], [2910], [  Fabricación de vehículos automotores], [1.49], [6.96], [0.12],
  [2816], [  Fabricación de equipo de elevación y manipulación], [1.92], [6.85], [0.16],
  [2829], [  Fabricación de otros tipos de maquinaria de uso especial], [1.92], [6.85], [0.16],
  [3040], [  Fabricación de vehículos militares de combate], [1.59], [6.7], [0.15],
  [2815], [  Fabricación de hornos, hogares y quemadores], [1.63], [5.7], [0.13],
  [2811], [  Fabricación de motores y turbinas, excepto motores para aeronaves, vehículos automotores y motocicletas], [1.49], [5.32], [0.13],
  [2819], [  Fabricación de otros tipos de maquinaria de uso general], [1.44], [5.0], [0.12],
  [2822], [  Fabricación de maquinaria para la conformación de metales y de máquinas herramienta], [1.44], [5.0], [0.12],
  [2591], [  Forja, prensado, estampado y laminado de metales; pulvimetalurgia], [1.79], [4.62], [0.17],
  [2660], [  Fabricación de equipo de irradiación y equipo electrónico de uso médico y terapéutico], [1.75], [4.57], [0.17],
  [2420], [  Fabricación de metales preciosos básicos y de otros metales no ferrosos], [0.9], [0.84], [0.27],
  [#table.cell(rowspan: 5)[C2 Química, materiales y farmacéutica]], [2011], [  Fabricación de sustancias químicas básicas], [1.74], [5.56], [0.18],
  [2395], [  Fabricación de artículos de hormigón, cemento y yeso], [0.74], [0.58], [0.28],
  [2012], [  Fabricación de abonos y compuestos de nitrógeno], [0.6], [-0.02], [0.29],
  [2021], [  Fabricación de plaguicidas y otros productos químicos de uso agropecuario], [0.34], [-1.01], [0.3],
  [1920], [  Fabricación de productos de la refinación del petróleo], [0.25], [-1.68], [0.3],
  [#table.cell(rowspan: 4)[C3 Agroindustria y alimentos procesados]], [1075], [  Elaboración de comidas y platos preparados], [1.33], [4.15], [0.22],
  [1061], [  Elaboración de productos de molienda], [0.48], [-0.45], [0.29],
  [1101], [  Destilación, rectificación y mezcla de bebidas alcohólicas], [0.27], [-1.45], [0.3],
  [1073], [  Elaboración de cacao y chocolate y de productos de confitería], [0.0], [-2.68], [0.32],
  [#table.cell(rowspan: 4)[C4 Servicios empresariales intensivos en conocimiento (KIBS)]], [7110], [  Actividades de arquitectura e ingeniería y actividades conexas de consultoría técnica], [1.46], [3.69], [0.23],
  [8291], [  Actividades de agencias de cobro y agencias de calificación crediticia], [1.42], [3.45], [0.22],
  [6910], [  Actividades jurídicas], [1.31], [2.76], [0.25],
  [7020], [  Actividades de consultoría de gestión], [0.55], [0.18], [0.29],
  [#table.cell(rowspan: 3)[C5 Turismo]], [5021], [  Transporte de pasajeros por vías de navegación interiores], [1.62], [4.17], [0.16],
  [5224], [  Manipulación de la carga], [0.61], [0.45], [0.28],
  [7911], [  Actividades de agencias de viajes], [0.22], [-1.39], [0.31],
  [C6 Industria textil y de confección], [1394], [  Fabricación de cuerdas, cordeles, bramantes y redes], [0.45], [-0.25], [0.3]
)
  ,  caption: [Priorización de actividades en el Margen Extensivo (oportunidades de diversificación para Honduras)],
) 

#figure(
table(
  stroke: 0.5pt + black,

    fill: (x, y) => {
      if y == 0 { rgb("d1e7dd") }      // Header color
      else if x == 0 { rgb("f8f9fa") } // Nested/Rowspan column color
    },
  align: center + horizon, // Centers text horizontally and vertically
  table.hline(y: 1, stroke: 1.5pt),
  table.hline(y: 27, stroke: 1.5pt),
  columns: 5,
  table.header[HS12][Actividad][Exportaciones Mundiales (USD M, 2024)][Distance][PCI],
  [6307], [Otros artículos confeccionados], [17166], [0.81], [-0.28],
  [6403], [Calzado de cuero], [53260], [0.82], [-0.19],
  [6406], [Partes de calzado], [9073], [0.82], [-0.22],
  [5806], [Tejidos estrechos], [4320], [0.83], [0.11],
  [9403], [Otros muebles y sus partes], [100539], [0.83], [0.21],
  [5808], [Trenzas en pieza], [444], [0.83], [-0.13],
  [9406], [Construcciones prefabricadas], [10964], [0.84], [0.0],
  [5515], [Otros tejidos de fibras sintéticas discontinuas], [4034], [0.84], [0.05],
  [9401], [Asientos], [85941], [0.84], [0.33],
  [5601], [Guata de materias textiles], [2881], [0.84], [-0.07],
  [6001], [Tejidos de pelo, punto], [6556], [0.84], [0.11],
  [5702], [Alfombras y tapetes tejidos], [5356], [0.84], [-0.23],
  [5511], [Hilo de fibras sintéticas discontinuas, destinado a la venta al por menor], [474], [0.84], [-0.03],
  [5514], [Tejidos con < 85 % de fibras sintéticas discontinuas, con un peso > 170 g/m²], [1987], [0.85], [-0.17],
  [6308], [Juegos de costura de tela e hilo], [82], [0.85], [0.43],
  [5811], [Productos textiles acolchados], [234], [0.85], [0.28],
  [5510], [Hilo de fibras artificiales discontinuas, no destinado a la venta al por menor], [1156], [0.85], [0.4],
  [6002], [Tejidos de punto, >5% de hilo elastomérico o hilo de caucho], [438], [0.85], [0.2],
  [5407], [Tejidos de filamento sintético], [29919], [0.85], [-0.18],
  [5506], [Fibras sintéticas discontinuas, procesadas], [259], [0.85], [-0.26],
  [5703], [Alfombras de mechón insertado], [6761], [0.85], [0.22],
  [6404], [Calzado textil], [40349], [0.86], [-0.13],
  [6303], [Cortinas], [5454], [0.86], [0.17],
  [5705], [Otras alfombras], [2633], [0.86], [-0.01],
  [5602], [Fieltro], [1407], [0.86], [0.64],
  [5501], [Estopa de filamento sintético], [838], [0.86], [0.05],
  [5512], [Tejidos con > 85 % de fibras sintéticas discontinuas], [2916], [0.86], [0.12],
  [5207], [Hilo de algodón para la venta al por menor], [439], [0.86], [0.1],
  [5701], [Alfombras de nudo], [869], [0.86], [-0.21],
  [6005], [Tejidos de punto por urdimbre], [3502], [0.86], [0.63],
  [5306], [Hilo de lino], [484], [0.86], [0.18],
  [6213], [Pañuelos], [151], [0.87], [-0.27],
  [6507], [Diademas], [453], [0.87], [0.28],
  [5902], [Tejido para neumáticos], [2638], [0.87], [1.02],
  [5704], [Alfombras de fieltro], [581], [0.87], [0.42],
  [5408], [Tejidos de filamento artificial], [1097], [0.87], [0.48],
  [5516], [Tejidos de fibras artificiales discontinuas], [4740], [0.87], [0.22],
  [5311], [Tejidos de fibras textiles vegetales], [122], [0.87], [-0.05],
  [5301], [Lino, crudo o procesado], [1816], [0.88], [-0.14],
  [5603], [Textiles no tejidos], [16532], [0.88], [0.54],
  [5109], [Hilo de lana o pelo animal, destinado a la venta al por menor], [461], [0.88], [0.1],
  [5801], [Tejidos de felpa], [2169], [0.88], [0.86],
  [5309], [Tejidos de lino], [2399], [0.88], [0.69],
  [5504], [Fibras artificiales discontinuas, sin procesar para hilado], [3055], [0.88], [0.82],
  [5911], [Artículos textiles para uso técnico], [5471], [0.88], [0.97],
  [6215], [Corbatas, pajaritas y fulares], [404], [0.88], [-0.06],
  [6603], [Partes de paraguas, sombrillas o bastones], [390], [0.88], [0.73],
  [5111], [Tejidos de lana cardada], [815], [0.88], [0.71],
  [5112], [Tejidos de lana peinada], [1644], [0.88], [0.71],
  [6506], [Otros artículos de sombrerería], [4620], [0.88], [0.33],
  [5604], [Textiles de caucho], [842], [0.88], [0.67],
  [5606], [Hilo trenzado], [598], [0.89], [0.97],
  [5406], [Hilo de filamento sintético para la venta al por menor], [145], [0.89], [0.28],
  [5403], [Hilo de filamento artificial], [1526], [0.89], [0.75],
  [6601], [Paraguas], [3294], [0.89], [0.2],
  [6702], [Flores artificiales], [5057], [0.89], [0.52],
  [9402], [Mobiliario médico, dental o veterinario], [4859], [0.89], [0.9],
  [9405], [Lámparas], [61767], [0.89], [0.62],
  [5903], [Tejidos impregnados con plástico], [14028], [0.89], [0.78],
  [5910], [Correas de transmisión o Cinturones de materia textil], [665], [0.89], [1.19],
  [5007], [Tejidos de seda], [786], [0.9], [0.18],
  [5906], [Tejidos cauchutados], [1650], [0.9], [0.95],
  [6602], [Bastones], [290], [0.9], [0.82],
  [5909], [Mangueras y tubos similares de materia textil], [414], [0.9], [0.87],
  [5502], [Estopa de filamento artificial], [2144], [0.9], [0.45],
  [5905], [Revestimientos textiles para paredes], [141], [0.9], [1.02],
  [5904], [Linóleo], [250], [0.9], [0.78],
  [5907], [Otros tejidos impregnados, recubiertos o revestidos], [811], [0.91], [1.05],
  [9702], [Grabados originales], [649], [0.93], [0.7]
)
  , caption: [Priorización de Productos Textiles en el Margen Extensivo (oportunidades de diversificación para Honduras). HS12, Atlas de Complejidad Económica de Harvard (otros productos textiles)])