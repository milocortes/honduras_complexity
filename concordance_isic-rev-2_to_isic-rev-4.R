library(concordance)

setwd("/home/milo/Documents/egtp/iniciativas/honduras_complexity")

institutional_intensity = read.csv("datos/viabilidad_atractivo/institutional_intensity.csv")

clases_isic2 = formatC(institutional_intensity$ISIC, width = 3, format = "d", flag = "0")

isic2_to_isic4 = concord(sourcevar = clases_isic2,
                origin = "ISIC2", destination = "ISIC4",
                dest.digit = 4, all = TRUE)

df_isic2_to_isic4 = data.frame()

for (clase in clases_isic2) {
    df = data.frame(isic2_to_isic4[[clase]])
    df$ciiu2 = clase
    df_isic2_to_isic4 = rbind.data.frame(df_isic2_to_isic4, df)
}

df_isic2_to_isic4 = df_isic2_to_isic4[, c("ciiu2", "match", "weight")]

colnames(df_isic2_to_isic4) = c("ciiu2", "ciiu4", "weight")

write.csv(df_isic2_to_isic4, "datos/recodificacion/ciiu-rev-2_to_ciiu-rev-4.csv", row.names = F)