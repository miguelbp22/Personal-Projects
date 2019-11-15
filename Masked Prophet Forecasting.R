library(prophet)
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(anytime)
library(RODBC)
library(scales)


driver.name <- "MASKED"
serv.name <- "MASKED"
db.name <-  "POS"
host.name <-  "MASKED"
port <- "5480"
user.name <- "MASKED"
pwd <- "MASKED"


#Use a full connection string to connect to a SAMPLE database
con.text <- paste("DRIVER=",driver.name,
                  ";Server=",serv.name,
                  ";Database=",db.name,
                  ";Hostname=",host.name,
                  ";Port=",port,
                  ";PROTOCOL=TCPIP",
                  ";UID=", user.name,
                  ";PWD=",pwd,sep="")

con.text

con1 <- odbcDriverConnect(con.text)

res <- sqlQuery(con1, "(select week_end_date as date, lead(round(sum(pos_total_retail_value), 0)) over (order by grs2, date) as POS, grs2 as brand, 
 to_char(cast(WEEK_END_DATE as date),'YYYY') as year, to_char(cast(WEEK_END_DATE as date),'MM') as month, 
to_char(cast(week_end_date as date), 'YYYY-MM') as time
from ADMIN.POS_weekly_CHAIN_VIEW
where
year_pos in (2016, 2017, 2018, 2019)
and extrapolation = 'Y'
and country = 'UNITED STATES'
and grs2 in ('BRAND1','BRAND2', 'BRAND3')
group by 3,1
order by 3,1)
union
(select week_end_date as date, lead(round(sum(pos_total_retail_value), 0)) over (order by grs2, date) as POS, grs2 as brand, 
 to_char(cast(WEEK_END_DATE as date),'YYYY') as year, to_char(cast(WEEK_END_DATE as date),'MM') as month, 
to_char(cast(week_end_date as date), 'YYYY-MM') as time
from ADMIN.POS_weekly_CHAIN_VIEW
where
year_pos in (2016, 2017, 2018, 2019)
and extrapolation = 'Y'
and country = 'UNITED STATES'
and retailer != 'MASKED'
and grs2 in ('BRAND4')
group by 3,1
order by 3,1)
order by 3,1", believeNRows=FALSE)

percent <- function(x, digits = 2, format = "f", ...) {
  paste0(formatC(100 * x, format = format, digits = digits, ...), "%")
}


data <- read.csv("C:/Users/BARRER11/Documents/Forecast/forecast_data_0906 (002).csv")
drivers <- read.csv("C:/Users/BARRER11/Documents/Forecast/drivers_updated.csv")

prophet_forecasting <- function(data, drivers){
  
  #getting rid of original date format
  data <- data[ ,c(1:ncol(data)-1)]
  drivers <- drivers[ ,c(1:ncol(drivers)-1)]
  
  #assigning names to cols of data
  colnames(data) <- c("ds","y","brand","year","month", "time")
  colnames(drivers) <- c("ds", "brand", "dist", "year")
  
  #assigning values to vars of interest
  brands = c(unique(as.character(data$brand)))
  num_brands <- length(brands)
  years <- unique(data$year)
  weeks_to_date <- plyr::count(data, "year")[4,2]/num_brands
  weeks_to_predict <- 52 - weeks_to_date
  
  christmas <- data_frame(
    holiday = 'christmas',
    ds = as.Date(c('2016-12-24', '2017-12-23','2018-12-22', '2019-12-21')),
    lower_window = -42,
    upper_window = 0)
  
  easter <- data_frame(
    holiday = 'easter',
    ds = as.Date(c('2016-03-26', '2017-04-15','2018-03-31', '2019-04-13')),
    lower_window = -20,
    upper_window = 0)
  
  holidays <- bind_rows(christmas, easter)
  
  breakdown <- data.frame(matrix(NA, nrow = num_brands, ncol = 9))
  colnames(breakdown) <- c("Brand","Predicted POS YOY Change", "Training MAPE", "Predicted YTD 2019 POS", "Predicted YTD 2019  POS -", "Predicted YTD 2019  POS +", "Predicted 2019 POS", "Predicted 2019 POS -", "Predicted 2019 POS +")
  
  #initializing loop through brands
  for(i in 1:num_brands){
    
    #creating data subset at each brand level, cleaning it, and taking log of POS
    data_sub = data[which(data$brand == brands[i]), ]
    data_sub$ds <- factor(data_sub$ds, levels = as.character(data_sub$ds))
    data_sub$y <- log(data_sub$y)

    #making new dataframe of POS data by year, giving it column names, taking the weekly average across years and making it its own column
    yearly_data <- data.frame(data_sub[which(data_sub$year == years[1]), 2], 
                              data_sub[which(data_sub$year == years[2]), 2], 
                              data_sub[which(data_sub$year == years[3]), 2])
    colnames(yearly_data) <- c(paste0('y',years[1]), 
                               paste0('y',years[2]), 
                               paste0('y',years[3]))
    yearly_data$avg <- rowMeans(yearly_data)
    
    #changing the POS variable in whole dataset to be POS for that week - weekly average + 1 (+1 to account for negative differences for when we take the exp later)
    data_sub[which(data_sub$year == years[1]), 2] <- yearly_data[ ,1] - yearly_data[ ,4] + 1
    data_sub[which(data_sub$year == years[2]), 2] <- yearly_data[ ,2] - yearly_data[ ,4] + 1
    data_sub[which(data_sub$year == years[3]), 2] <- yearly_data[ ,3] - yearly_data[ ,4] + 1
    data_sub[which(data_sub$year %in% years[1:(length(years)-1)] == FALSE), 2] <- data_sub[which(data_sub$year %in% years[1:(length(years)-1)] == FALSE), 2] - yearly_data[1:weeks_to_date, 4] + 1
    
    #subsetting the drivers to be the rows of the brand being looped over
    drivers_sub <- drivers[which(drivers$brand == brands[i]), ]

    train <-merge(data_sub, drivers_sub[, c(1,3)], by = "ds")
    
    #####IF WE ACTUALLY SPLIT THE DATA TO TRAIN/TEST####
    #train <- merge(data_sub[which(data_sub$year %in% ys[1:3]), c(1,2)], drivers_sub[which(drivers_sub$year %in% ys[1:3]), c(1,3)], by = 'ds')
    #test <- merge(data_sub[which(data_sub$year %in% ys[4]), c(1,2)], drivers_sub[which(drivers_sub$year %in% ys[4]), c(1,3)], by = 'ds')
    
    #initiating a prophet item, with holiday seasonality taken into account, adding the distribution ('dist') regressor, and fitting the prophet item to the data
    m <- prophet(holidays = holidays, yearly.seasonality = TRUE, changepoint.range=0.8, interval.width = 0.5)
    m <- add_regressor(m,"dist")
    
    m <- fit.prophet(m, train)
    
    #making a new dataframe from the prophet item and adding the weeks to predict, and adding the dist regressor to that dataframe
    future <- make_future_dataframe(m, periods=weeks_to_predict, freq='week')
    
    #####IF WE ACTUALLY SPLIT THE DATA TO TRAIN/TEST####
    #future <- make_future_dataframe(m, periods = nrow(test) + weeks_to_predict, freq='week')
    future$dist<-drivers_sub$dist[which(drivers_sub$brand==brands[i])]
    
    #predicting the fitted prophet item on the future dataset, just keeping the data, rediction and intervals
    forecast <- predict(m, future)
    predicted <- forecast[c('ds', 'yhat','yhat_lower','yhat_upper')]
    
    #turning all the log'd data back to original numbers by taking inverses
    train[,2]<-exp(train[,2]-1+yearly_data[,4])
    #####IF WE ACTUALLY SPLIT THE DATA TO TRAIN/TEST####
    #test[,2] <- exp(test[,2]-1+yearly_data[1:nrow(test),4])
    predicted[,2]<-exp(predicted[,2]-1+yearly_data[,4])
    predicted[,3]<-exp(predicted[,3]-1+yearly_data[,4])
    predicted[,4]<-exp(predicted[,4]-1+yearly_data[,4])
    
    #predicting the YOY change from 2018-2019
    change <- percent((sum(c(tail(predicted[,2],weeks_to_predict),tail(train[,2],weeks_to_date)))-sum(train[105:156,2]))/sum(train[105:156,2]))
    print(paste("Predicted YOY Change: ", change))
    
    #setting up the data to have data, actual, pred, and the year-month for each week, then summing all pred and actuals by year month
    mape_data <-data.frame("date"=as.factor(train[,1]),"actual"=train[,2],"pred"=predicted[1:dim(train)[1],2],"time"=train[,6])
    mape_data$date <- factor(mape_data$date, levels=as.character(mape_data$date))
    month_data_train<- ddply(mape_data,~time,summarise,actual=sum(actual),pred=sum(pred))
    colnames(month_data_train)[1]<-c("date")
    
    #stacks the data by date
    plot_data_train<-melt(month_data_train[, c("date", "actual", "pred")], id="date")
    plot_data_train$date<-anydate(plot_data_train$date)
    
    #calculating the mape
    mape<-percent(mean(abs(month_data_train$actual-month_data_train$pred)/month_data_train$actual))
    
    #creating plot
    mape_plot <- ggplot(plot_data_train) + geom_line(aes(x=date, y=value, color=variable))  + labs(title=paste("Brand", i, "Training Actual vs Pred, MAPE: ",mape))+scale_x_date(date_breaks = "3 month", date_labels =  "%b %Y") + scale_y_continuous(labels = comma)
    print(mape_plot)
    
    #####IF WE ACTUALLY SPLIT THR DATA TO TRAIN/TEST#####
    #hw_mape_data_train <-data.frame("date"=as.factor(hw_train[,1]),"actual"=hw_train[,2],"pred"=hw_predicted[1:dim(hw_train)[1],2],"time"=hw_data[1:dim(hw_train)[1],6])
    #hw_mape_data_train$date <- factor(hw_mape_data_train$date, levels=as.character(hw_mape_data_train$date))
    #hw_month_data_train<- ddply(hw_mape_data_train,~time,summarise,actual=sum(actual),pred=sum(pred))
    #colnames(hw_month_data_train)[1]<-c("date")
    
                  #TESTING DATA
    #hw_mape_data_test <-data.frame("date"=as.factor(hw_test[,1]),"actual"=hw_test[,2],"pred"=hw_predicted[(dim(hw_train)[1]+1):(dim(hw_train)[1]+nrow(hw_test)),2],"time"=hw_data[(dim(hw_train)[1]+1):(dim(hw_train)[1]+nrow(hw_test)),6])
    #hw_mape_data_test$date <- factor(hw_mape_data_test$date, levels=as.character(hw_mape_data_test$date))
    #hw_month_data_test<- ddply(hw_mape_data_test,~time,summarise,actual=sum(actual),pred=sum(pred))
    #colnames(hw_month_data_test)[1]<-c("date")
    
                  #stacks the data by date
    #hw_plot_data_train<-melt(hw_month_data_train[, c("date", "actual", "pred")], id="date")
    #hw_plot_data_train$date<-anydate(hw_plot_data_train$date)
    
    #hw_plot_data_test<-melt(hw_month_data_test[, c("date", "actual", "pred")], id="date")
    #hw_plot_data_test$date<-anydate(hw_plot_data_test$date)
    
    #hw_plot_data_full <- data.frame(rbind(hw_plot_data_train, hw_plot_data_test))
    
    #ggplot(hw_plot_data_full) + geom_line(aes(x=date, y=value, color=variable))
                  #calculating the mape
    #hw_mape_train<-percent(mean(abs(hw_month_data_train$actual-hw_month_data_train$pred)/hw_month_data_train$actual))
    #hw_mape_test <- percent(mean(abs(hw_month_data_test$actual-hw_month_data_test$pred)/hw_month_data_test$actual))
    
    breakdown[i,] <- c(brands[i],change,mape,colSums(tail(predicted,52)[1:weeks_to_date,-1]),colSums(tail(predicted,52)[,-1]))
  }
  #write.csv(breakdown, "Forecast_Model_0923.csv", row.names = FALSE)
  return(breakdown)
}

prophet_forecasting(data, drivers)
