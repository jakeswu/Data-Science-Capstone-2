#### HX CAPSTONE 2 - CHENGLIANG (JAKE) WU #####
#### HR ANALYTICS ####
if(!require(fastDummies)) install.packages("fastDummies",repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot",repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr",repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("ggplot2",repos = "http://cran.us.r-project.org")

library(dplyr)
library(tibble)
library(caret)
library(fastDummies)
library(corrplot)
library(stringr)

############################
### 1. DOWNLOAD THE DATA ###
############################
#Download IBM's HR Analytics Dataset#
#Source: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
temp <- tempfile(fileext = ".csv")
download.file("https://drive.google.com/u/0/uc?id=1MYrMS546iSBcps-1lkO0aHV-nVJn8htD&export=download",temp)
readfile <- read.csv(temp,quote="")

#make sense of the data
as_tibble(readfile)
str(readfile)
which(is.na(readfile))

############################
### 2. EXPLORATORY DATA ANALYSIS ###
############################
#### 2.1. PRE-PROCESSING / WRANGLING ####
#To be able to work on the data, we have to wrangle certain columns, which are clearly not suited for mathematical processing
#e.g., the Gender Column comprises of "Male" and "Female", character strings which R won't be able to interpret
#Wrangle multi-categorical data into dummy variables. 
readfiletest1 <- dummy_cols(readfile, select_columns = c('JobRole','MaritalStatus','Department','EducationField'),remove_selected_columns=TRUE)

#convert ordinal variables or bi-categorical variables into numerical equivalents
#Convert Travel Volume into numerical
readfiletest1["BusinessTravel"][readfiletest1["BusinessTravel"]=="Non-Travel"]<-0
readfiletest1["BusinessTravel"][readfiletest1["BusinessTravel"]=="Travel_Rarely"]<-1
readfiletest1["BusinessTravel"][readfiletest1["BusinessTravel"]=="Travel_Frequently"]<-2

#Convert Gender into numerical (1=Male, 0=Female)
readfiletest1["Gender"][readfiletest1["Gender"]=="Female"]<-0
readfiletest1["Gender"][readfiletest1["Gender"]=="Male"]<-1

#Convert Attrition into numerical (1=Yes, 0=No)
readfiletest1["Attrition"][readfiletest1["Attrition"]=="No"]<-0
readfiletest1["Attrition"][readfiletest1["Attrition"]=="Yes"]<-1

#Convert OverTime into numerical (1=Yes, 0=No)
readfiletest1["OverTime"][readfiletest1["OverTime"]=="No"]<-0
readfiletest1["OverTime"][readfiletest1["OverTime"]=="Yes"]<-1

readfiletest1$BusinessTravel<-as.numeric(as.character(readfiletest1$BusinessTravel))
readfiletest1$Gender<-as.numeric(as.character(readfiletest1$Gender))
readfiletest1$Attrition<-as.numeric(as.character(readfiletest1$Attrition))
readfiletest1$OverTime<-as.numeric(as.character(readfiletest1$OverTime))

#close the spaces in the column headers (e.g. Sales Representative -> Sales.Representative) for easy referencing
names(readfiletest1) <- make.names(names(readfiletest1), unique=TRUE)

#we have many variables, identify the near zero variance predictors using caret package's nearzerovar function
#<https://riptutorial.com/r/example/24920/removing-features-with-zero-or-near-zero-variance>
nearZeroVar(readfiletest1)

#remove those columns from the dataset
readfiletest2 <- subset(readfiletest1,select=-c(nearZeroVar(readfiletest1)))

#### 2.2. SPLIT DATA INTO TRAINING AND TEST ####
# generate training and test sets. I use a 80:20 split (https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11583)
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, remove the sample.kind argument
test_index <- createDataPartition(as.factor(readfiletest2$Attrition), times = 1, p = 0.2, list = FALSE)
test_set <- readfiletest2[test_index, ]
train_set <- readfiletest2[-test_index, ]

#### 2.3. EXPLORATORY ANALYSIS ON TRAIN SET ####
#We have >40 predictors, which we can further trim first.

##### 2.3.1. Remove highly correlated variables (they don't add value)
cor_mat<-cor(readfiletest2)
findCorrelation(cor_mat, .99)

#find linear combinations
findLinearCombos(readfiletest2)

#no highly correlated variables, we can continue

##### 2.3.1. Remove non-significantly correlated variables (between attrition + predictor) #####

#First, generate correlation table with p-values below a certain value
testRes <- cor.mtest(train_set, conf.level = 0.95)

#Then, generate correlation plot
corrplot(cor(train_set), p.mat = testRes$p, method = 'circle', type = 'lower', insig='blank',
         addCoef.col ='black', number.cex = 0.2, order = 'AOE', diag=FALSE, tl.cex=0.3, tl.srt=45)

#Identify column indices that have no relationship with attrition
#i.e. in the correlation table containing p-values, identify the 2nd row, which is attrition
#then, identify which column indices exceed the p-value of 0.05
remove_index <- which(((as.data.frame(testRes$p))[2,]) > 0.05)

#manually remove those columns that have no (statistically-significant) correlation with attrition
train_set_cleaned <- subset(train_set,select=-remove_index)

#generate correlation table once again with p-values below a certain value
testRes_cleaned <- cor.mtest(train_set_cleaned, conf.level = 0.95)

#generate correlation plot again, we should see a much cleaner table
corrplot(cor(train_set_cleaned), p.mat = testRes_cleaned$p, method = 'circle', type = 'lower', insig='blank',
         addCoef.col ='black', number.cex = 0.25, order = 'AOE', diag=FALSE, tl.cex=0.3, tl.srt=45)

###### Some discussion ######
#Overtime, being single, a Sales Reps/being in Sales Dept, and a high degree of business travel is associated with attrition
#To a lesser extent, long distance from home and being educated in marketing is also associated with attrition
#On the flip side, high job level, higher monthly income, being older, being married, and being in the role for a long time = less attrition

#### 3. Trying some models ####
###start with logistic regression##

#Fit data
lm_fit <- lm(Attrition~.,train_set_cleaned)

#Generate a list of outcomes
p_hat <- predict(lm_fit,test_set)

#Set outcomes to be 1 (i.e., attrition) if probability exceeds 0.5 
y_hat <- ifelse(p_hat > 0.5, "1", "0") |> factor()
test_set$Attrition<-as.factor(test_set$Attrition)

#Run a confusion matrix to evaluate the performance.
cm_1<-confusionMatrix(y_hat, test_set$Attrition)

#calculate accuracy and recall
Accuracy_1<-cm_1$overall[["Accuracy"]]
Recall_1<-cm_1$byClass[["Recall"]]

Accuracy_1
Recall_1

#Add a results table to comparatively show how Accuracy and Recall Fare. 
Results_table <- tibble(Method = "Logistic", Accuracy = Accuracy_1, Recall=Recall_1)
Results_table

###3.2. K-NEAREST NEIGHBORS (KNN) ###
#<https://rpubs.com/yevonnael/ibm-hr-analytics>

#Fit data
knn_fit <- knn3(Attrition~.,data=train_set_cleaned, k=sqrt(nrow(train_set_cleaned)))

#Set outcomes to be 1 (i.e., attrition) if probability exceeds 0.5 
y_hat2raw <- as.data.frame(predict(knn_fit, test_set)) %>% mutate('class'=names(.)[apply(., 1, which.max)]) %>% select(class)
y_hat2 <- as.factor(y_hat2raw$class)

#Run a confusion matrix to evaluate the performance.
cm_2<-confusionMatrix(y_hat2, test_set$Attrition)

#calculate performance metrics
Accuracy_2<-cm_2$overall[["Accuracy"]]
Recall_2<-cm_2$byClass[["Recall"]]

Accuracy_2
Recall_2

#Add a results table to comparatively show how Accuracy and Recall Fare. 
Results_table <- Results_table %>% add_row(Method="KNN",Accuracy=Accuracy_2,Recall=Recall_2)
as.data.frame(Results_table)


###3.3. CLASSIFICATION TREE###
set.seed(1, sample.kind = "Rounding") # if using R 3.5 or earlier, remove the sample.kind argument
train_rpart<- train(Attrition~., method="rpart", data=train_set_cleaned, tuneGrid=data.frame(cp=seq(0,0.05,0.002)))
train_rpart$bestTune
#Accuracy based on best value of k
y_hat6raw <- predict(train_rpart,test_set)
y_hat6 <- ifelse(y_hat6raw>0.5,"1","0") %>% factor(levels=c("0","1"))
cm_6 <- confusionMatrix(y_hat6, test_set$Attrition)

#calculate performance metrics
Accuracy_6<-cm_6$overall[["Accuracy"]]
Recall_6<-cm_6$byClass[["Recall"]]

Accuracy_6
Recall_6

#Add a results table to comparatively show how Accuracy and Recall Fare. 
Results_table <- Results_table %>% add_row(Method="Classification Tree",Accuracy=Accuracy_6,Recall=Recall_6)
as.data.frame(Results_table)

