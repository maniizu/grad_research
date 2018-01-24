#データを読み込む
data_au <- read.csv("data_au.csv")
#data_au_2011 <- read.csv("cate_data_au_11.csv")
#data_au_2013 <- read.csv("cate_data_au_13.csv")
#data_au_2015 <- read.csv("cate_data_au_15.csv")

data_au_2011 <- subset(data_au, year==2011)
data_au_2013 <- subset(data_au, year==2013)
data_au_2015 <- subset(data_au, year==2015)

#data_au <- data_au[, !(colnames(data_au_2011) %in% c("capacity", "year"))]
data_au_2011 <- data_au_2011[, !(colnames(data_au_2011) %in% c("capacity", "year"))]
data_au_2013 <- data_au_2013[, !(colnames(data_au_2013) %in% c("capacity", "year"))]
data_au_2015 <- data_au_2015[, !(colnames(data_au_2015) %in% c("capacity", "year"))]

res_au_2011.lm <- lm(aud~., data=rbind(data_au_2013, data_au_2015))
res_au_2013.lm <- lm(aud~., data=rbind(data_au_2011, data_au_2015))
res_au_2015.lm <- lm(aud~., data=rbind(data_au_2011, data_au_2013))

library(MASS)

res_au_2011.lm.step <- stepAIC(res_au_2011.lm)
summary(res_au_2011.lm.step)

res_au_2013.lm.step <- stepAIC(res_au_2013.lm)
summary(res_au_2013.lm.step)

res_au_2015.lm.step <- stepAIC(res_au_2015.lm)
summary(res_au_2015.lm.step)

#congestion_rate
data_co <- read.csv("data_co.csv")

data_co_2011 <- subset(data_co, year==2011)
data_co_2013 <- subset(data_co, year==2013)
data_co_2015 <- subset(data_co, year==2015)

data_co_2011 <- data_co_2011[, !(colnames(data_co_2011) %in% c("capacity", "year"))]
data_co_2013 <- data_co_2013[, !(colnames(data_co_2013) %in% c("capacity", "year"))]
data_co_2015 <- data_co_2015[, !(colnames(data_co_2015) %in% c("capacity", "year"))]

res_co_2011.lm <- lm(congestion_rate~., data=rbind(data_co_2013, data_co_2015))
res_co_2013.lm <- lm(congestion_rate~., data=rbind(data_co_2011, data_co_2015))
res_co_2015.lm <- lm(congestion_rate~., data=rbind(data_co_2011, data_co_2013))

library(MASS)

res_co_2011.lm.step <- stepAIC(res_co_2011.lm)
summary(res_co_2011.lm.step)

res_co_2013.lm.step <- stepAIC(res_co_2013.lm)
summary(res_co_2013.lm.step)

res_co_2015.lm.step <- stepAIC(res_co_2015.lm)
summary(res_co_2015.lm.step)
