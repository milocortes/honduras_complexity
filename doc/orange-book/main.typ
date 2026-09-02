#import "@preview/orange-book:0.7.1": book, part, chapter, my-bibliography, appendices, make-index, index, theorem, definition, notation,remark,corollary,proposition,example,exercise, problem, vocabulary, scr, update-heading-image

//#set text(font: "Linux Libertine")
//#set text(font: "TeX Gyre Pagella")
//#set text(font: "Lato")
//#show math.equation: set text(font: "Fira Math")
//#show math.equation: set text(font: "Lato Math")
//#show raw: set text(font: "Fira Code")
#set text(size : 11pt, font: "Lato")
#show figure.where(kind: table): set block(breakable: true)

#show: book.with(
  title: "Análisis de Complejidad Económica en Honduras para la Diversificación Productiva",
  subtitle: "A Practical Guide",
  date: datetime.today,
  author: "Goro Akechi",
  main-color: rgb("#1982f3"),
  lang: "es",
  cover: image("./background.svg"),
  image-index: image("./honduras1.jpg"),
  list-of-figure-title: "Lista de Figuras",
  list-of-table-title: "Lista de Tablas",
  supplement-chapter: "Capítulo",
  supplement-part: "Parte",
  part-style: 0,
  copyright: [
    //Copyright © 2023 Flavio Barisi

    PUBLISHED BY PUBLISHER

    //#link("https://github.com/flavio20002/typst-orange-template", "TEMPLATE-WEBSITE")

    Licensed under the Apache 2.0 License (the “License”).
    You may not use this file except in compliance with the License. You may obtain a copy of
    the License at https://www.apache.org/licenses/LICENSE-2.0. Unless required by
    applicable law or agreed to in writing, software distributed under the License is distributed on an
    “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.

    _First printing, September 2026_
  ],
  lowercase-references: false
)


// Custom thmbox
#let solution(name: none, body) = {
  context{
    thmbox("solution","Solution",
    stroke: (left: 4pt + green),
    radius: 0em,
    inset: 0.65em,
    namefmt: x => [*--- #x.*],
    separator: h(0.2em),
    titlefmt: x => text(fill: green, weight: "bold", x),
    fill: green.lighten(90%), 
    base_level: 1)(name:name, body)
  }
}

//#set par(leading: 0.6em)
#set par(spacing: 1.2em)
#import "@preview/mitex:0.2.7": *
#import "@preview/tablem:0.3.0": tablem, three-line-table
#import "@preview/booktabs:0.0.4": *

#show: booktabs-default-table-style

#let three-line-table = tablem.with(
  render: (columns: auto, align: auto, ..args) => {
    table(
      columns: columns,
      stroke:  (x: none),
      //align: center + horizon,
      table.hline(y: 0),
      table.hline(y: 1, stroke: .5pt),
      ..args,
      table.hline(),
    )
  }
)

//#include "original.typ"

// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
// **********  Complejidad Económica de Honduras
// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
#part("Complejidad Económica de Honduras") 
#chapter("Hechos estilizados de la Economía de Honduras", image: image("./honduras1.jpg"), l: "chap1")
//#index("intro_modelo")
#include "secciones/complejidad_economica.typ"


// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
// **********  Identificación de Oportunidades de Diversificación
// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
#part("Identificación de Oportunidades de Diversificación") 
#chapter("Identificación de Oportunidades de Diversificación", image: image("./honduras1.jpg"), l: "chap2")
//#index("intro_modelo")
#include "secciones/oportunidades_diversificacion.typ"


// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
// **********  Factores de Viabilidad y Atractivo
// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
#part("Factores de Viabilidad y Atractivo") 

#chapter("Factores de Viabilidad y Atractivo", image: image("./honduras1.jpg"), l: "chap3")
//#index("intro_modelo")
#include "secciones/factores_viabilidad_atrativo.typ"



#my-bibliography( bibliography("sample.bib"))

//#make-index(title: "Index")


// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
// **********  Apéndices
// +++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++
// 
// 
#show: appendices.with("Appendices", hide-parent: false)

#include "anexos/anexo_metodologia_complejidad_economica.typ"
#include "anexos/anexo_bases_datos.typ"
#include "anexos/anexo_robustez_consistencia.typ"
#include "anexos/anexo_margen_intensivo.typ"
#include "anexos/anexo_margen_extensivo.typ"


#include "anexos/anexo_onudi.typ"

#include "anexos/anexo_factores_viabilidad_atrativo.typ"

