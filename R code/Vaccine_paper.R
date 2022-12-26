library('ggplot2')


df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Analysis/Probabilistic_analysis/Raw_for_analysis/vaccine_paper_2.csv')

p <- ggplot(df) + ylab('Effectiveness') + xlab('Months since vaccination') +
  geom_point(aes(x = Month, y = Effectiveness, color=Source)) + 
  geom_line(aes(x = Month, y = Effectiveness, color=Source)) +
  geom_ribbon(aes(x = Month, ymin = Effectiveness_min, ymax = Effectiveness_max, fill=Source), alpha=0.5) +
  facet_grid(Type ~ Dose) 

p

write.csv(df, 'back_up.csv')

ggsave("two_papers.jpg", width = 15, height = 6)

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
