library(dplyr)
library(lubridate)
library(zoo)
library(caret)
library(xgboost)
library(RcppRoll)
setwd("/Users/efeme/Desktop/pred")
file_list <- list.files(pattern = "\\.csv")
data_frames <- list()
for (file in file_list) {
  data <- read.csv(file)
  data_frames[[file]] <- data
}
hourly_data <- bind_rows(data_frames, .id = "file_source")
hourly_data <- hourly_data[,c("timestamp","price","short_name")]
hourly_data$timestamp <- ymd_hms(hourly_data$timestamp, tz = "UTC")
hourly_data$date <- as.Date(hourly_data$timestamp)

data2 <- read.csv("Daily_Series - 20231210_20240112.csv")
data2 <- data2[data2$timestamp>tail(hourly_data$timestamp,1),]
data2$timestamp <- ymd_hms(data2$timestamp)
data2$date <- as.Date(data2$timestamp)

hourly_data <- rbind(hourly_data,data2)



# Define custom WMAPE function for model evaluation
wmape <- function(actual, predicted) {
  sum(abs(actual - predicted))*100/sum(actual)
}

unique_companies <- unique(hourly_data$short_name)
company_data <- list()
for (company in unique_companies) {
  company_series <- hourly_data[hourly_data$short_name == company,]
  company_data[[company]] <- company_series
}

predictions <- list()
test_date <- "2024-01-11"
for (company in unique_companies) {
  data <- company_data[[company]]
  data <- data %>%
    mutate(timestamp = ymd_hms(timestamp),
           hour = hour(timestamp),
           weekday = as.numeric(factor(wday(timestamp, label = TRUE))),
           price_lag1 = lag(price, 1),
           price_lag10 = lag(price, 10),
           price_lag2 = lag(price, 2),
           price_lag20 = lag(price, 20),
           price_lag3 = lag(price, 3),
           price_diff1 = price - price_lag1,
           price_diff10 = price - price_lag10,
           rolling_mean_3 = c(NA,NA,roll_mean(price, 3, na.rm =FALSE)),
           rolling_sd_3 = c(NA,NA,roll_sd(price, 3, na.rm = FALSE))) %>%
    na.omit()

  train_data <- data[data$date <= test_date, ]
  test_data <- data[data$date == test_date, ]
  test_data$weekday <- 5


# Build the model using XGBoost
  model <- xgboost(data = as.matrix(train_data %>% select(-timestamp, -price, -short_name,-date)),
                 label = train_data$price,
                 objective = "reg:squarederror",
                 eval_metric = "rmse",
                 nrounds = 100)

# Make predictions on the test set
predictions[[company]] <- predict(model, as.matrix(test_data %>% select(-timestamp, -price, -short_name,-date)))
}
actual_values <- hourly_data[hourly_data$date==test_date,c("price","short_name","timestamp")]
predictions_df <- matrix(0,nrow=10,ncol=30)
colnames(predictions_df) <- unique_companies
wmape_value <- 0 
for(name in unique_companies) {
  predictions_df[,name] <- predictions[[name]]
  wmape_value <- wmape_value + wmape(actual_values[actual_values$short_name==name,"price"], predictions_df[,name])
}


predictions_df <- matrix(0,nrow=10,ncol=30)
colnames(predictions_df) <- unique_companies
for(name in unique_companies) {
  predictions_df[,name] <- predictions[[name]]
}
predictions_df <- as.data.frame(predictions_df)
library(writexl)
write_xlsx(predictions_df,path="predictions.xlsx")

start_date <- "2023-12-27"
actual_data <- hourly_data[hourly_data$date >= start_date,]

library(writexl)
write_xlsx(actual_data,path="actuals.xlsx")
