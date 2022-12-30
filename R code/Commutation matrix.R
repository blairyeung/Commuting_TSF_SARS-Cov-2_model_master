#Commuting Heat map
library(reshape2)
library(viridis)
library(tidyverse)
library(tidyquant)
library(ggplot2)
library(tidyverse)
library(viridis)
library(patchwork)
library(networkD3)
library(ggplot2)
library(ggalluvial)


df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Model Dependencies/Ontario_commute.csv')

colnames(df) <- c('GEOCODE_POR', 'GEONAME_POR', 'GEOCODE_POW', 'GEONAME_POW', 'Total_worker', 'Male_worker', 'Female_worker')

sets <- c(1140,2030,4320,6375,9155,12495,41730,88930,146200,207385,311250)


ggplot(data = df, aes(x = GEONAME_POR, y = GEONAME_POW)) +
  geom_raster(aes(fill = Total_worker)) +
  scale_fill_gradient(low = '#ffff80', high = '#6b0000', trans = 'log10')

df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Model Dependencies/Ontario_commute_district.csv')

colnames(df) <- c('line', 'GEOCODE_POR', 'GEOCODE_POW', 'Total_worker')

sets <- c(1140,2030,4320,6375,9155,12495,41730,88930,146200,207385,311250)


ggplot(data = df, aes(x = as.character(GEOCODE_POR), y = as.character(GEOCODE_POW))) +
  geom_raster(aes(fill = Total_worker)) +
  scale_fill_gradient(low = 'white', high = '#6b0000', trans = 'log10')


p <- ggplot(df, aes(y = Total_worker, axis1 = GEOCODE_POR, axis2 = GEOCODE_POW)) +
  geom_alluvium(width = 1/12) +
  geom_stratum(width = 1/12, fill = "white", color = "grey") +
  geom_text(stat = "stratum", aes(label = after_stat(stratum))) +
  scale_x_discrete(limits = c("Resident", "Work"), expand = c(.05, .05))

ggsave('plot.jpg', width = 10, height = 8)

p  
