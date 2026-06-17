## Limpiamos area de trabajo 
rm(list =ls())

library(concordance)

setwd("/home/milo/Documents/egtp/iniciativas/honduras/datos/recodificacion")

ciiu = read.csv("ciiu_213_actividades.csv")

completo = data.frame()


for (clase in ciiu$ciiu){
    print(clase)
    clase = sprintf("%04d", clase)
    matching = concord_hs_isic(sourcevar = clase,
                   origin = "ISIC4", destination = "HS4",
                   dest.digit = 4, all = TRUE)

    matching = data.frame(matching)
    colnames(matching) = c("hs12", "weight")
    matching$ciiu = clase

    completo = rbind.data.frame(completo, matching)
}

write.csv(completo, "ponderadores_ciiu_hs12_concordance.csv", row.names = F)