#Commuting Heat map
library(reshape2)
library(viridis)
library(tidyverse)
library(tidyquant)
library(ggplot2)

df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Model Dependencies/Ontario_commute.csv')

colnames(df) <- c('GEOCODE_POR', 'GEONAME_POR', 'GEOCODE_POW', 'GEONAME_POW', 'Total_worker', 'Male_worker', 'Female_worker')

sets <- c(1140,2030,4320,6375,9155,12495,41730,88930,146200,207385,311250)


ggplot(data = df, aes(x = GEONAME_POR, y = GEONAME_POW)) +
  geom_raster(aes(fill = Total_worker)) +
  scale_fill_gradient(low = '#ffff80', high = '#6b0000', trans = 'log10')
