library(ggplot2)

avg = 15

by = 0.002

b1 <- seq(0, avg, by = by)    

by1 <- dgamma(b1, avg, rate =  2)

b2 <- seq(0, avg, by = by)   

by3 <- dnorm(b2,  mean = 0, sd = 0.1)

by1 <- by3 * by1

plot(by1)

by2 <- dgamma(b3, 2.5, rate =  4)

bound <- avg * 3/2

cuml <- c(1:bound)

print(1:bound)

for (i in c(1:bound)){
  ratio <- length(b1) / bound
  cuml[i] <- 0
  for (j in c(1:ratio)){
    cuml[i] <- cuml[i] + by1[ratio * i + j - ratio]
  }
}

# tot <- sum(cuml)

x <- c(1:bound)

df <- data.frame(x, cuml)

plot(cuml)

ggplot(df, aes(x = x, y = cuml/tot)) +
  geom_bar(stat = 'identity') + ylab('Frequency') + xlab('Days upon infection') +
  geom_line()
