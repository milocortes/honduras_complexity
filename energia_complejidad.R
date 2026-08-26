# Libraries
library(ggplot2)
library(dplyr)
#library(hrbrthemes)
library(viridis)
library(ggrepel)

setwd("/home/milo/Documents/egtp/iniciativas/honduras")

industrias = read.csv("energia_complejidad_industrias.csv")

industrias_plot = ggplot(industrias, 
       aes(x = share_energy, y = pci, size = PROD, fill = "blue", label = division_titulo)) +
  geom_point(alpha = .5, 
             shape = 21) + geom_text(nudge_y = 0.5, size = 2.5) + 
  scale_size_continuous(range = c(1, 16)) + 
  labs(title = "Índice de Complejidad Económica y Dependencia Energética",
       subtitle = "CIIU Rev 4 División",
       x = "Dependencia Energética [%]",
       y = "Indice de Complejidad de la Industria") +
  theme(legend.position = "none") 

industrias_plot 