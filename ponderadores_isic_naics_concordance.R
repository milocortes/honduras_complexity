## Limpiamos area de trabajo 
rm(list =ls())

library(concordance)

setwd("/home/milo/Documents/egtp/iniciativas/honduras/datos/recodificacion")

ciiu = read.csv("ciiu_213_actividades.csv")

completo = data.frame()


for (clase in ciiu$ciiu){
    print(clase)
    clase = sprintf("%04d", clase)
    matching = concord_naics_isic(sourcevar = clase,
                origin = "ISIC4", destination = "NAICS2017",
                dest.digit = 4, all = TRUE)

    matching = data.frame(matching)
    colnames(matching) = c("naics", "weight")
    matching$ciiu = clase

    completo = rbind.data.frame(completo, matching)
}

write.csv(completo, "ponderadores_ciiu_naics2017_concordance.csv", row.names = F)