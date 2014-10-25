#--Coursera / machine Learning / Assignment

#--The goal of your project is to predict the manner in which they did the exercise.
#--This is the "classe" variable in the training set.
#--You may use any of the other variables to predict with.
#--You should create a report describing how you built your model,
#--how you used cross validation,
#--what you think the expected out of sample error is,
#--and why you made the choices you did.
#--You will also use your prediction model to predict 20 different test cases.

library (caret)
library (cvTools)    #--response must be numeric
library (plyr)
library (dplyr)
library (reshape2)
library (ggplot2)
library (knitr)
library (rpart)
library (randomForest)
library (deepnet)
library (randomForestSRC)
library (ipred)
library (party)

get.data = function () {
   #--19622 rows, 160 columns
   #--three timestamp columns
   setwd ("d://coursera//machine//project")
   dir()
   train = read.csv ("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
   train = tbl_df(train)
   colnames(train)[1] = "seq"
   train
   dim(train)
   summary(train)

   z = grep ("timestamp", colnames(train))
   colnames(train)[z]
   train = train[,-z]

   z = grep("user_name", colnames(train))
   train = train[,-z]

   train$seq         = NULL
   train$new_window  = NULL
   train$num_window  = NULL

   dim(train)
   head(colnames(train))
   train
}


explore = function (data) {
   #--count nas in each colname
   print (dim(data))
   tab = data.frame(var = colnames(data))
   for (var in colnames(data)) tab$nas[tab$var == var] = sum(is.na(data[,var]))
   tab = tab[order(tab$nas),]
   tab$seq = 1:dim(tab)[1]
   tab = tab[,c("seq", "var", "nas")]
   print (tab)
}

univariate = function (data, var) {
   #--plot sorted values of var, colored by class
   df                = data.frame(var = data[,var], classe = data$classe)
   df$classe         = as.character(df$classe)
   rownames(df)      = NULL
   colnames(df)[1]   = "var"
   df                = df[order(df$var, df$classe),]
   df$seq             = 1:dim(df)[1]
   runlengths        = rle(df$classe)
   lengths           = runlengths$lengths
   m                 = max(lengths)
   index.a           = which(lengths == m)
   index.b           = sum(lengths[1:(index.a-1)]) + 1
   index.c           = index.b + m - 1
   index.a
   index.b
   index.c
   x.ref       = c(index.b, index.c)
   msg.1       = paste("longest run =", max(runlengths$lengths))
   msg.2       = paste("number of runs =", length(runlengths$lengths))
   q           = qplot (x=seq, y = var, data = df, colour = df$classe, main = var )
   x.txt       =  .1 * dim(df)[1]
   y.txt       = min(df$var) + .8 * (max(df$var) - min(df$var))
   q           = q + geom_text(data = NULL, x = x.txt, y = y.txt,
                     label = paste(msg.1, msg.2, sep = "\n"))
   q           = q + geom_vline(xintercept = x.ref, colour = "red")
   #print (q)
   list (var = as.character(var), n = length(runlengths$lengths), max = max(runlengths$lengths))
}

drop.na.columns = function (df) {
   #--drop columns if they have any NAs
   counts = data.frame(var = colnames(df))
   for (var in colnames(df)) {
      counts$na[counts$var == var] = sum (is.na(df[,var]))
   }
   head (counts)
  na.counts = colSums(is.na(df)) == 0
  table (na.counts)
  df2 = df[, na.counts ]
  df2
}


explore.univariate = function (df, preds) {
   stats = data.frame (var = preds)
   for (var in preds) {
      print                            (var)
      stat                             = univariate (df, var)
      print                            (stat)
      stats$runs[stats$var == var]    = stat$n
      stats$longest[stats$var == var] = stat$max
      stats$plot[stats$var == var]    = stat$plot
   }
   stats$var = as.character(stats$var)
   stats
}


#The out-of-bag (oob) error estimate
#In random forests, there is no need for cross-validation or a separate test set
#to get an unbiased estimate of the test set error.
#It is estimated internally, during the run, as follows:

#   Each tree is constructed using a different bootstrap sample from the original data.
#About one-third of the cases are left out of the bootstrap sample and not used
#in the construction of the kth tree.

#Put each case left out in the construction of the kth tree down the kth tree to get
#a classification. In this way, a test set classification is obtained for each case
#in about one-third of the trees. At the end of the run, take j to be the class that
#got most of the votes every time case n was oob. The proportion of times that j
#is not equal to the true class of n averaged over all cases is the oob error estimate.
#This has proven to be unbiased in many tests.
#--https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

train = get.data()

explore (train) #--summarize columns by NA count

train2 = drop.na.columns(train)
dim(train2)



preds = head(colnames(train2), -1)


run.stats = explore.univariate(train2, preds)
run.stats = run.stats[order(run.stats$runs),]
run.stats$var = as.character(run.stats$var)
head (run.stats)
univariate (train2, run.stats$var[1])
univariate (train2, run.stats$var[2])

run.stats = run.stats[order( - run.stats$longest),]
head (run.stats)
univariate (train2, run.stats$var[1])
univariate (train2, run.stats$var[2])


extract.majority = function (m) {
   #--m is a matrix
   #--return a vector of vars from colnames
   m = as.data.frame(m)
   m$max = NULL
   classes = colnames(m)
   m$max = apply (m, 1, max)
   head (m)
   predicted =rep(NA, dim(m)[1])

   for (class in classes) {
      index = m[,class] == m$max
      predicted[index] = class
   }
   predicted
}

#-----------------------------------------------------------
#--rpart
#

k.fold.rpart.cross = function (df, fit, K = 10) {
   n  = dim (df)[1]
   u  = runif (n)
   df = df[order(u),]
   length(unique(u))   #--shuffle the rows

   row = 1:dim(df)[1]
   folds = row %% K
   head (folds, 20)
   accuracy = NULL
   table (folds)
   for (fold in unique(folds)) {
      local = df[-which(folds == fold),]
      test  = df[ which(folds == fold,)]
      local.fit = rpart (classe ~ ., data = local, type = "class")
      predicted = predict (local.fit, test, type = "class")
      mc = table (test$classe, predicted)
      accuracy = c(accuracy, sum(diag(mc)) / sum (mc))
   }
   accuracy
}


set.seed (766)
fit.1 = rpart (classe ~ ., data = train2, method = "class")
plot (fit.1, main = "rpart fit.1")
plotcp (fit.1, main = "rpart, fit.1")
predicted = predict (fit.1, train2, type = "class")

confusionMatrix (train2$classe, predicted)

mc = table (train2$class, predicted)
resub.err = 1 - sum(diag(mc)) / sum(mc)

#--accuracy is 0.7556

fit.2 = prune(fit.1, cp = 0.015)
plotcp (fit.2, main = "rpart, fit.2")
predicted.2 = predict(fit.2, train2, type = "class")

confusionMatrix (train2$classe, predicted.2)
#--accuracy is 0.6857

accuracy = k.fold.cross (train2, fit.2, K=10)
summary (accuracy)


#=================================================================
#--random forest


k.fold.rf.cross = function (df, ntree, K = 10, R = 1) {
   #--build K models, holding out 1/K of the data each time
   #--and then scoring against the held out data
   #-- do this R times

   accuracy = NULL

   for (index in 1:R) {
      n  = dim (df)[1]
      u  = runif (n)
      df = df[order(u),]
      length(unique(u))   #--shuffle the rows

      row = 1:dim(df)[1]
      folds = row %% K
      head (folds, 20)

      table (folds)
      for (fold in unique(folds)) {
         local = df[-which(folds == fold),]
         test  = df[ which(folds == fold),]
         local.fit = randomForest (classe ~ ., data = local, ntree=ntree, type = "class")
         predicted = predict (local.fit, test, type = "class")
         mc = table (test$classe, predicted)
         accuracy = c(accuracy, sum(diag(mc)) / sum (mc))
         print (paste(fold, tail(accuracy,1)))
         }
   }
   accuracy
}

set.seed (1188)
ntree = 50
fit.rf = randomForest (classe ~ ., data = train2, importance = T, ntree = ntree)
predicted = predict (fit.rf, train, type = "class")
mc = table (train2$classe, predicted)
mc
summary(fit.rf)
confusionMatrix (train2$classe, predicted)
plot (fit.rf, main = "RF")

set.seed (271828)
rm (accuracy)
accuracy = k.fold.rf.cross (train2, ntree=ntree, K = 10, R = 5)
summary(accuracy)

#--http://www.statistik.uni-dortmund.de/useR-2008/slides/Strobl+Zeileis.pdf


imp.rf = as.data.frame(randomForest::importance(fit.rf, type = 2))
imp.rf$var = row.names(imp.rf)
imp.rf = imp.rf[order(-imp.rf$MeanDecreaseGini),]
row.names(imp.rf) = NULL
head (imp.rf)
tail (imp.rf)

#=============================================

pml_write_files = function(x){
   n = length(x)
   for(i in 1:n){
      filename = paste0("problem_id_",i,".txt")
      write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
   }
}

test = read.csv("data//pml-testing.csv")
answers = predict (fit.rf, test, type = "class")
answers

pml_write_files(answers)

knit2html ("machine_project.Rmd")
knit2pdf  ("machine_project.Rmd")
