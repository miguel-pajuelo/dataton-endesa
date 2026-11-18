################################################################################
##############        Forecasting:     FT+SARIMA         #######################
#######################  INTER - LUCÍA LOZANO     ##############################
################################################################################

library(MLTools)
library(fpp2)
library(ggplot2)
library(TSA)
library(lmtest)  #contains coeftest function
library(tseries) #contains adf.test function
library(Hmisc) # for computing lagged variables

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset -------------------------------------------------------------------------------------------------------
fdata <- read_excel( "2026_Midterm_44.xlsx")


# Convert to time series object ->

fdata_ts <- ts(fdata, frequency = 12) #Cambiar la frecuencia
autoplot(fdata_ts, facets = TRUE)

# Scales --> Escalo para poner en unidades si hace falta
scale_y <- 10  #Los escalo para tenerlo más simple, porque tenemos más de una serie.

# Create time series and scale values 
y <- fdata_ts[,c("y")]/scale_y
x <- fdata_ts[,c("x")]
autoplot(cbind(y,x), facets = TRUE)

y_orig <- fdata_ts[,"y"]


# Another way -> por filas
ind <- seq(1:nrow(fdata))
ind_TV <- ind <= 36
ind_TR <- !ind_TV

# Training subset
y.tr <- ts(y[ind_TR], frequency = 12) #Modificamos la frecuencia
x.tr <- ts(x[ind_TR], frequency = 12)
autoplot(cbind(y.tr,x.tr), facets = TRUE)


ggtsdisplay(y.tr, lag=50)

## FIRST --------------------------------------------
#### Fit initial FT model with large s
#This arima function belongs to the TSA package
TF.fit <- arima(y.tr,
                xtransf = x.tr,
                order=c(1,0,0),
                seasonal = list(order=c(1,0,0),period=7),
                transfer = list(c(0,9)),
                method="ML")

TF.RegressionError.plot(y.tr,x.tr,TF.fit,lag.max = 50)

## SECOND --------------------------------------------
#### Fit initial FT model with large s including regular differentiation
TF.fit <- arima(y.tr,
                xtransf = x.tr,
                order=c(1,0,0),
                seasonal = list(order=c(1,1,0),period=12),
                transfer = list(c(0,9)),
                method="ML")
summary(TF.fit) #summary of training errors and estimated coefficients
coeftest(TF.fit) #statistical significance of estimated coefficients
 #Check regression error to see the need of differentiation
TF.RegressionError.plot(y.tr,x.tr,TF.fit,lag.max = 50)
#Regression error is stationary, then, we can continue
#NOTE: If this regression error was not stationary in variance,boxcox should be applied to input and output series.

#Check numerator coefficients of explanatory variable
TF.Identification.plot(x.tr,TF.fit)


## THIRD --------------------------------------------
xlag <- x.tr
xlag <- Lag(x.tr,2) # b1
xlag[is.na(xlag)]=0
arima.fit <- arima(y.tr,
                   xtransf = xlag,
                   order=c(2,0,1),
                   seasonal = list(order=c(1,1,0),period=12),
                   transfer = list(c(2,1)),
                   method="ML")
summary(arima.fit) #summary of training errors and estimated coefficients
coeftest(arima.fit) #statistical significance of estimated coefficients
#Check residuals
CheckResiduals.ICAI(arima.fit, lag=50)
#All coefficients significant --> Good.
#All correlations are small   --> Good.


#Check Cross correlation residuals - expl. variables
res <- residuals(arima.fit)
res[is.na(res)] <- 0
ccf(y = res, x = x.tr)

########
#All expl.variables seem not correlated with residuals -> Good 


## Finished model identification

#Check fitted
autoplot(y.tr, series = "Real")+
  forecast::autolayer(fitted(arima.fit), series = "Fitted")

#Check training errors of the model
accuracy(fitted(arima.fit),y.tr)


################################################################################
## Forecast for new data with h = 1 for 3 months
#################################################################################

y.TV.est <- rep(NA, length(y))
for (i in seq(length(y.tr)+1, length(fdata$y), 1)){# loop for validation period
  y.TV.est[i] <- TF.forecast(y.old = subset(y,end=i-3), #past values of the series
                             x.old = subset(x,end=i-3), #Past values of the explanatory variables
                             x.new = subset(x,start = i,end=i), #New values of the explanatory variables
                             model = arima.fit, #fitted transfer function model
                             h=3) #Forecast horizon
}
#Todo el conjunto de validación con horizonte


# Accuracy in TV ----
# scaled values
accuracy(y.TV.est,y)
# original values
accuracy(y.TV.est * scale_y, y_orig)

# Plot forecast
autoplot(y, series = "Real")+
  forecast::autolayer( ts(y.TV.est, frequency = 12), series = "Fitted (TV)")
