library('ggplot2')



df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Analysis/Probabilistic_analysis/Raw_for_analysis/vaccine_paper_1.csv')

p <- ggplot(df) + ylab('Attack rate') + xlab('Months upon immumnity gain') +
  geom_point(aes(x = Month, y = Recovered_unvaccinated, color='blue')) + 
  geom_line(aes(x = Month, y = Recovered_unvaccinated, color='blue')) +
  geom_point(aes(x = Month, y = Two_dose, color='red')) +
  geom_line(aes(x = Month, y = Two_dose, color='red')) +
  scale_colour_discrete(name = "Cohort", labels = c("Natural immunity", "Vaccine immunity (2 dose)")) + 
  facet_grid(. ~ Age_band) 

p

ggsave("vaccine_unvaccinated.jpg", width = 10, height = 3)


p <- ggplot(df) + ylab('Frequency') + xlab('Months upon immumnity gain') +
  geom_point(aes(x = Month, y = Recovered_vaccinated, color='blue')) + 
  geom_line(aes(x = Month, y = Recovered_vaccinated, color='blue')) +
  geom_point(aes(x = Month, y = Vaccined_recovered, color='green')) + 
  geom_line(aes(x = Month, y = Vaccined_recovered, color='green')) +
  geom_point(aes(x = Month, y = Two_dose, color='red')) +
  geom_line(aes(x = Month, y = Two_dose, color='red')) +
  scale_colour_discrete(name = "Cohort", labels = c("Hybird immunity 1", "Hybird immunity 2", "Vaccine immunity (2 dose)")) + 
  facet_grid(. ~ Age_band) 

p

ggsave("vaccine_vaccined.jpg", width = 10, height = 3)
