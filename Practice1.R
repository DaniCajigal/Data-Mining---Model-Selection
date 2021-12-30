library(class)
library(ggplot2)
library(GGally)
library(ggcorrplot)
library(MASS)
library(caret)

data(iris)
summary(iris)

#histogram and density plots showing the distribution of the quantitative variables 
iris[,1:4] <- scale(iris[,1:4])
par(mfrow=c(2,2))
plot(density(iris$Sepal.Length), col=iris$Species)
plot(density(iris$Sepal.Width))
plot(density(iris$Petal.Length))
plot(density(iris$Petal.Width))

par(mfrow=c(2,2))
hist(iris$Sepal.Length, col="blue", breaks=20)
hist(iris$Sepal.Width, col="blue", breaks=20)
hist(iris$Petal.Length, col="blue", breaks=20)
hist(iris$Petal.Width, col="blue", breaks=20)

#If we observe the distribution of Petal Length and Petal width of the Iris dataset as a whole, we see that they do not follow a normal distribution.

#We first divide the Iris dataset into training and test dataset to apply KNN classification. 
#60% of the data is used for training while the KNN classification is tested on the remaining 40% of the data.

set.seed(12366894)
setosa<- rbind(iris[iris$Species=="setosa",])
versicolor<- rbind(iris[iris$Species=="versicolor",])
virginica<- rbind(iris[iris$Species=="virginica",])

ind <- sample(1:nrow(setosa), nrow(setosa)*0.6)
iris.train<- rbind(setosa[ind,], versicolor[ind,], virginica[ind,])
iris.test<- rbind(setosa[-ind,], versicolor[-ind,], virginica[-ind,])

#_____________________________________________________________________________________________________________________________________________________
#KNN:

#We run the KNN classification algorithm for different values of K and see the value of K from K=1 to K=15 which gives the lowest error.
error <- c()
for (i in 1:15)
{
  knn.fit <- knn(train = iris.train[,1:4], test = iris.test[,1:4], cl = iris.train$Species, k = i)
  error[i] = 1- mean(knn.fit == iris.test$Species)
}

ggplot(data = data.frame(error), aes(x = 1:15, y = error)) + geom_line(color = "Blue")

#We can see that K=5 gives the lowest test error.
#We run KNN classification on the data set using the value of K as K=5
set.seed(12366894)
iris_pred <- knn(train = iris.train[,1:4], test = iris.test[,1:4], cl = iris.train$Species, k=5)
confusion_knn <- confusionMatrix(iris.test$Species,iris_pred)

#_____________________________________________________________________________________________________________________________________________________
#LDA:

#LDAs are useful for classification problems where there are more than 2 values for the predictor variable which is the case here,
#otherwise we could use logistic regression. Using a LDA is predicated on the normality assumption, and from the bell-curve shape of the 
#density plots, its reasonable to proceed.

model_LDA <- lda(Species ~ ., iris.train)
predict_LDA <- predict(model_LDA, iris.test)
confusion_LDA <- confusionMatrix(predict_LDA$class, iris.test$Species)

#_____________________________________________________________________________________________________________________________________________________
#MCNEMAR


LDA_vector <- data.frame(predict_LDA$class == iris.test$Species)
knn_vector <- data.frame(iris_pred == iris.test$Species)

#From the previous vectors we can obtain the contingency table, in this case this calculation has been made in Excel

data <- matrix(c(57, 2, 0, 1),
       nrow = 2,
       dimnames = list("Model 1" = c("correct", "wrong"),
                       "Model 2" = c("correct", "wrong")))


test <- mcnemar.test(data, y = NULL, correct = TRUE)
test
