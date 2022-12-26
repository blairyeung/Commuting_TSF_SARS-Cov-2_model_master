library('ggplot2')
library("ggsci")

df <- read.csv('D:/Github/Commuting_TSF_SARS-Cov-2_model_master/Analysis/Probabilistic_analysis/Infereced_data/Vaccine_effectiveness.csv')

# colnames(df) <- c('Dose', 'Days', 'mean', 'mean_min', 'mean_max', 'predicted', 'Andrews_et_al', 'Andrews_et_al_max', 'Andrews_et_al_min', 'Ferdinands_et_al', 'Ferdinands_et_al_max', 'Ferdinands_et_al_min', 'CDC', 'CDC_max', 'CDC_min')

colnames(df) <- c('Dose','Source', 'Days', 'mean', 'mean_min', 'mean_max', 'predicted')

p <- ggplot(df) + ylab('Effectiveness') + xlab('Days since vaccination') +
  geom_ribbon(aes(x = Days, ymin=mean_max, ymax=mean_min, fill=Source), alpha=0.3) +
  geom_line(aes(x = Days, y = mean, color=Source), size=0.75) +
  geom_line(aes(x = Days, y = predicted), size=0.75, color='black', linetype=2) +
  scale_color_nejm() + scale_fill_nejm() +
  facet_grid(. ~ Dose) 

p

