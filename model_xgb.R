## setting working directory (edit your path before running script)
path <- ""
setwd(path)


## loading libraries
library(data.table)
library(plyr)
library(xgboost)


## loading data
data <- fread("Train_seers_accuracy.csv")


## cleaning data

# converting DOB to Date format
data$DOB[nchar(data$DOB) == 8] <- paste0("0", data$DOB[nchar(data$DOB) == 8])
data$DOB <- paste0(substr(data$DOB,1,7), "19", substr(data$DOB,8,9))
data$DOB <- as.Date(data$DOB, "%d-%b-%Y")

# date features
data$Transaction_Date[nchar(data$Transaction_Date) == 8] <- paste0("0", data$Transaction_Date[nchar(data$Transaction_Date) == 8])
data$Transaction_Date <- as.Date(data$Transaction_Date, "%d-%b-%y")
data$Transaction_Year <- format(data$Transaction_Date, "%Y")
data$Transaction_Month <- format(data$Transaction_Date, "%Y-%m")

# cleaning features
data$Age <- as.numeric(as.Date("2016-01-01") - data$DOB) / 365
data$Gender <- ifelse(data$Gender == "F", 1, 0)
data$Purchased_in_Sale <- ifelse(data$Purchased_in_Sale == "Y", 1, 0)
data$Referred_Friend[data$Referred_Friend == ""] <- "NO"
data$Sales_Executive_ID <- as.numeric(substr(data$Sales_Executive_ID,6,10))


## creating build and val datasets
X_build_1 <- data[data$Transaction_Year < 2006,]
X_build_2 <- data[data$Transaction_Year < 2005,]

X_val_1 <- data[data$Transaction_Year == 2006,]
X_val_2 <- data[data$Transaction_Year == 2005,]

X_all <- data


## feature engineering for build-val-1
X_month_1 <- X_build_1[, .(Count = .N), .(Client_ID, Transaction_Month)]
X_count_1 <- X_month_1[, .(Count = .N), .(Client_ID)]

# mean values per Client ID
X_age_1 <- X_build_1[, .(Age = mean(Age)), .(Client_ID)]
X_gender_1 <- X_build_1[, .(Gender = mean(Gender)), .(Client_ID)]
X_sale_1 <- X_build_1[, .(Sale = mean(Purchased_in_Sale)), .(Client_ID)]
X_executive_id_1 <- X_build_1[, .(Executive = median(Sales_Executive_ID)), .(Client_ID)]

# encoding categorical variables
X_source_1 <- dcast(X_build_1, Client_ID ~ Lead_Source_Category, length, value.var="Client_ID", fill=0)
X_var1_1 <- dcast(X_build_1, Client_ID ~ Var1, length, value.var="Client_ID", fill=0)
X_var2_1 <- dcast(X_build_1, Client_ID ~ Var2, length, value.var="Client_ID", fill=0)
X_var3_1 <- dcast(X_build_1, Client_ID ~ Var3, length, value.var="Client_ID", fill=0)
X_friend_1 <- dcast(X_build_1, Client_ID ~ Referred_Friend, length, value.var="Client_ID", fill=0)
X_executive_1 <- dcast(X_build_1, Client_ID ~ Sales_Executive_Category, length, value.var="Client_ID", fill=0)
X_payment_1 <- dcast(X_build_1, Client_ID ~ Payment_Mode, length, value.var="Client_ID", fill=0)
X_store_1 <- dcast(X_build_1, Client_ID ~ Store_ID, length, value.var="Client_ID", fill=0)
X_product_1 <- dcast(X_build_1, Client_ID ~ Product_Category, length, value.var="Client_ID", fill=0)

# date features
X_date_1 <- X_build_1[, .(Max_Date = max(Transaction_Date)), .(Client_ID)]
X_date_1 <- X_date_1[, ":="(Last_Date = as.numeric(as.Date("2006-01-01") - Max_Date),
                            Max_Date = NULL)]

# merging all features
X_train_1 <- Reduce(function(x, y) merge(x, y, all=T, by="Client_ID"),
                    list(X_count_1, X_age_1, X_gender_1, X_sale_1, X_executive_id_1, X_source_1, X_var1_1, X_var2_1, X_var3_1, X_friend_1, X_executive_1, X_payment_1, X_store_1, X_product_1, X_date_1))

# target variable
target_1 <- ifelse(X_train_1$Client_ID %in% unique(X_val_1$Client_ID), 1, 0)


## feature engineering for build-val-2
X_month_2 <- X_build_2[, .(Count = .N), .(Client_ID, Transaction_Month)]
X_count_2 <- X_month_2[, .(Count = .N), .(Client_ID)]

# mean values per Client ID
X_age_2 <- X_build_2[, .(Age = mean(Age)), .(Client_ID)]
X_gender_2 <- X_build_2[, .(Gender = mean(Gender)), .(Client_ID)]
X_sale_2 <- X_build_2[, .(Sale = mean(Purchased_in_Sale)), .(Client_ID)]
X_executive_id_2 <- X_build_2[, .(Executive = median(Sales_Executive_ID)), .(Client_ID)]

# encoding categorical variables
X_source_2 <- dcast(X_build_2, Client_ID ~ Lead_Source_Category, length, value.var="Client_ID", fill=0)
X_var1_2 <- dcast(X_build_2, Client_ID ~ Var1, length, value.var="Client_ID", fill=0)
X_var2_2 <- dcast(X_build_2, Client_ID ~ Var2, length, value.var="Client_ID", fill=0)
X_var3_2 <- dcast(X_build_2, Client_ID ~ Var3, length, value.var="Client_ID", fill=0)
X_friend_2 <- dcast(X_build_2, Client_ID ~ Referred_Friend, length, value.var="Client_ID", fill=0)
X_executive_2 <- dcast(X_build_2, Client_ID ~ Sales_Executive_Category, length, value.var="Client_ID", fill=0)
X_payment_2 <- dcast(X_build_2, Client_ID ~ Payment_Mode, length, value.var="Client_ID", fill=0)
X_store_2 <- dcast(X_build_2, Client_ID ~ Store_ID, length, value.var="Client_ID", fill=0)
X_product_2 <- dcast(X_build_2, Client_ID ~ Product_Category, length, value.var="Client_ID", fill=0)

# date features
X_date_2 <- X_build_2[, .(Max_Date = max(Transaction_Date)), .(Client_ID)]
X_date_2 <- X_date_2[, ":="(Last_Date = as.numeric(as.Date("2005-01-01") - Max_Date),
                            Max_Date = NULL)]

# merging all features
X_train_2 <- Reduce(function(x, y) merge(x, y, all=T, by="Client_ID"),
                    list(X_count_2, X_age_2, X_gender_2, X_sale_2, X_executive_id_2, X_source_2, X_var1_2, X_var2_2, X_var3_2, X_friend_2, X_executive_2, X_payment_2, X_store_2, X_product_2, X_date_2))

# target variable
target_2 <- ifelse(X_train_2$Client_ID %in% unique(X_val_2$Client_ID), 1, 0)


## feature engineering for test data
X_month <- X_all[, .(Count = .N), .(Client_ID, Transaction_Month)]
X_count <- X_all[, .(Count = .N), .(Client_ID)]

# mean values per Client ID
X_age <- X_all[, .(Age = mean(Age)), .(Client_ID)]
X_gender <- X_all[, .(Gender = mean(Gender)), .(Client_ID)]
X_sale <- X_all[, .(Sale = mean(Purchased_in_Sale)), .(Client_ID)]
X_executive_id <- X_all[, .(Executive = median(Sales_Executive_ID)), .(Client_ID)]

# encoding categorical variables
X_source <- dcast(X_all, Client_ID ~ Lead_Source_Category, length, value.var="Client_ID", fill=0)
X_var1 <- dcast(X_all, Client_ID ~ Var1, length, value.var="Client_ID", fill=0)
X_var2 <- dcast(X_all, Client_ID ~ Var2, length, value.var="Client_ID", fill=0)
X_var3 <- dcast(X_all, Client_ID ~ Var3, length, value.var="Client_ID", fill=0)
X_friend <- dcast(X_all, Client_ID ~ Referred_Friend, length, value.var="Client_ID", fill=0)
X_executive <- dcast(X_all, Client_ID ~ Sales_Executive_Category, length, value.var="Client_ID", fill=0)
X_payment <- dcast(X_all, Client_ID ~ Payment_Mode, length, value.var="Client_ID", fill=0)
X_store <- dcast(X_all, Client_ID ~ Store_ID, length, value.var="Client_ID", fill=0)

# date features
X_date <- X_all[, .(Max_Date = max(Transaction_Date)), .(Client_ID)]
X_date <- X_date[, ":="(Last_Date = as.numeric(as.Date("2007-01-01") - Max_Date),
                        Max_Date = NULL)]

# merging all features
X_test <- Reduce(function(x, y) merge(x, y, all=T, by="Client_ID"),
                 list(X_count, X_age, X_gender, X_sale, X_executive_id, X_source, X_var1, X_var2, X_var3, X_friend, X_executive, X_payment, X_store, X_date))


## creating train and test datasets
X_train <- rbind.fill(X_train_1, X_train_2)
target <- c(target_1, target_2)

n <- nrow(X_test)

# reording features as per train data (pretty lame way of doing it)
X_test <- rbind.fill(X_test, X_train)
X_test <- X_test[1:n,names(X_train)]

X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0


## xgboost
seed <- 235
set.seed(seed)

# cross-validation
model_xgb_cv <- xgb.cv(data=as.matrix(X_train), label=as.matrix(target), objective="binary:logistic", nfold=5, nrounds=1200, eta=0.02, max_depth=5, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="auc")
# CV: 0.8825

# model building
model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(target), objective="binary:logistic", nrounds=10, eta=0.02, max_depth=5, subsample=0.6, colsample_bytree=0.85, min_child_weight=1, eval_metric="auc")

# model scoring
pred <- predict(model_xgb, as.matrix(X_test))

# submission
submit <- data.frame("Client_ID" = X_test$Client_ID, "Cross_Sell" = pred)
write.csv(submit, "submit.csv", row.names=F)

# LB: 0.8856

