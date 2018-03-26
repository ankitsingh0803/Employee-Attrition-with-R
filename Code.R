#Import all libraries - ggplot2 for visualization
library("ggplot2")
library("klaR")
library("caret")
library("e1071")

# Read csv file and display first 5 rows.
Attrition = read.csv(file.choose())
head(Attrition)

#Convert the categorical 'Attrition' variable to numerical
Attrition$AttritionInt = ifelse(Attrition$Attrition == "Yes", 1, 0)

#factorize 'Business Travel' attribute as 'Yes' if 'Travel_Frequently' and 'No' otherwise.
Attrition$TravelFrequently = as.factor(ifelse(Attrition$BusinessTravel == "Travel_Frequently", "Yes", "No"))
table(Attrition$Attrition, Attrition$EnvironmentSatisfaction)
table(Attrition$Attrition, Attrition$TravelFrequently)
tapply(Attrition$AttritionInt, Attrition$EnvironmentSatisfaction, mean)
tapply(Attrition$AttritionInt, Attrition$OverTime, mean)

# Data Visualization using ggplot2 library (histograms, bar-plot, box-plot etc.)

ggplot(Attrition, aes(x = JobSatisfaction)) +
  geom_histogram(aes(fill = Attrition), position = "dodge", binwidth = 0.5)

ggplot(Attrition, aes(JobInvolvement, ..count..)) +
  geom_bar(aes(fill = Attrition), position = "dodge")

ggplot(Attrition, aes(y = YearsInCurrentRole, x = Age)) +
  geom_point(aes(colour = Attrition))

ggplot(Attrition, aes(y = YearsInCurrentRole, x = Attrition)) +
  geom_boxplot()

ggplot(Attrition, aes(y = Age, x = Attrition)) +
  geom_violin(scale = "count", color = "Blue")

ggplot(Attrition, aes(EnvironmentSatisfaction, ..count..)) +
  geom_bar(aes(fill = Attrition), position = "dodge")

mosaicplot(OverTime ~ TravelFrequently, data = Attrition,
           subset = (Attrition == "Yes"), color = TRUE)
mosaicplot(OverTime ~ TravelFrequently, data = Attrition,
           subset = (Attrition == "No"), color = TRUE)
mosaicplot(~ OverTime + Attrition, data = Attrition, color = TRUE)

library(vcd)
mosaic(~ Attrition + OverTime + TravelFrequently, data = Attrition,
       shade = TRUE, legend = TRUE)

# set seed to get cosntant random output each time the code is run.
set.seed(1234)
# shuffle indices.
ShuffledAttrition = Attrition[sample(nrow(Attrition)),]
head(ShuffledAttrition)
# split data into train and test sets in the ratio 70:30
train = ShuffledAttrition[1:(0.7*nrow(ShuffledAttrition)),]
head(train)
test = ShuffledAttrition[(0.7*nrow(ShuffledAttrition)+ 1):nrow(ShuffledAttrition),]
#test = subset(test, select = -Attrition)
# logistic regression model to predict 'Attrition'
log_reg = glm(Attrition ~ EnvironmentSatisfaction + YearsInCurrentRole + TravelFrequently +
                JobInvolvement + OverTime + Age, data = train, family = binomial(link="logit"))
summary(log_reg)
# modle predictions for 'Attrition' attribute.
predictions1 <- as.factor(ifelse(predict(log_reg, test, type = "response") > 0.5,
                                 "Yes", "No"))
head(predictions1)

# confusion matrix for model evaluation
confusionMatrix(test$Attrition, predictions1, positive = "Yes")

# Naive Bayes classifier is good with text
nb = NaiveBayes(Attrition ~ EnvironmentSatisfaction + YearsInCurrentRole + TravelFrequently +
                  JobInvolvement + OverTime + Age, data = train)
summary(nb)
predictions2 = predict(nb, test)
confusionMatrix(test$Attrition, predictions2$class, positive = "Yes")
cor(Attrition$JobInvolvement, Attrition$EnvironmentSatisfaction)

# Support Vector Machines
mod_svm = svm(Attrition ~ EnvironmentSatisfaction + YearsInCurrentRole + TravelFrequently +
                JobInvolvement + OverTime + Age, data = train)
summary(mod_svm)
predictions3 = predict(mod_svm, test)
head(predictions3)
confusionMatrix(test$Attrition, predictions3, positive = "Yes")

# Linear Discriminant Analysis
library(MASS)
mod_lda = lda(Attrition ~ EnvironmentSatisfaction + YearsInCurrentRole + TravelFrequently + 
                JobInvolvement + OverTime + Age, data = train)
summary(mod_lda)
predictions4 = predict(mod_lda, test)
head(predictions4)
confusionMatrix(predictions4$class, test$Attrition)

library(randomForest)
rf = randomForest(Attrition ~ EnvironmentSatisfaction + YearsInCurrentRole + TravelFrequently + 
                    JobInvolvement + OverTime + Age, data = train)
summary(rf)
predictions5 = predict(rf, test)
head(predictions5)
confusionMatrix(predictions5, test$Attrition)


