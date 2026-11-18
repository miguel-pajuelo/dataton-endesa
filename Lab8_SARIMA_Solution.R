#################################################################################
##############        Forecasting:     ARIMA         ############################
##############     ----------- solution ---------    ############################
#################################################################################

library(MLTools)
library(fpp2)
library(ggplot2)
library(readxl)
library(lmtest)  #contains coeftest function
library(tseries) #contains adf.test function
library(lubridate)

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load dataset -------------------------------------------------------------------------------------------------------
fdata <- read.table(file = "DailyPrice_Load_Wind_Spain_2019_2020_TR.csv",
                    header = T,
                    sep = ";")
fdata$Date <- lubridate::dmy(fdata$Date) #Lo pasamos así a tipo date
#Tipo DATE cuando no tengo horas.
#Con datos diarios hay dos estacionales; anual y semanal. Es más importante la semanal.
y <- ts(fdata$Price, frequency = 365, start=c(2019,1,1))
autoplot(y)

# TR - TS index
#Fijamos idx hasta que llego a 2020-09-30
ind_TR <- fdata$Date <= lubridate::ymd("2020-09-30")
ind_TV <- !ind_TR #Cojo la negación del idx de train.

# Convert to time series object
y <- ts(fdata$Price[ind_TR], frequency = 7)
#for daily data
autoplot(y)
ggtsdisplay(y,lag.max = 50)

## Identification and fitting process -------------------------------------------------------------------------------------------------------

# Box-Cox transformation
Lambda <- BoxCox.lambda.plot(y,7)
#Si vemos nube de puntos que no tiene tendencia lineal no aplicamos Box-Cox
#Añadimos el 7 porque el ancho de la ventana tiene que ser multiplo del periodo
# Lambda <- BoxCox.lambda(y) #other option
#z <- BoxCox(y,Lambda)
z <- y  #Como no aplicamos z = y
autoplot(z)

# Differentiation: if the ACF decreases very slowly -> needs differenciation
ggtsdisplay(z,lag.max = 100)

#Regular Differentiation
# If differencing is needed
Bz <- diff(z,differences = 1)
ggtsdisplay(Bz,lag.max = 100) #differences contains the order of differentiation

# Seasonal Differentiation
# If differencing is needed
B7Bz <- diff(Bz, lag = 7, differences = 1)

# ACF and PACF of the time series -> identify significant lags and order
ggtsdisplay(B7Bz,lag.max = 100)

# Fit seasonal model with estimated order
arima.fit <- Arima(y,
                   order=c(2,0,1),
                   seasonal = list(order=c(0,1,1), period=7),
                   lambda = NULL,
                   include.constant = FALSE)
summary(arima.fit) # summary of training errors and estimated coefficients
coeftest(arima.fit) # statistical significance of estimated coefficients
autoplot(arima.fit) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit, bins = 100, lag = 100)


 # Check fitted forecast
autoplot(y, series = "Real")+
  forecast::autolayer(arima.fit$fitted, series = "Fitted")

# Accuracy in TR
accuracy(fitted(arima.fit),y)


## Validation error for h = 1 -------------------------------------------------------------------------------------------------------
# Obtain the forecast in validation for horizon = 1 using the trained parameters of the model
yTV <- fdata$Price[ind_TV]
y.TV.est <- yTV*NA
for (i in seq(length(y)+1, length(fdata$Price), 1)){# loop for validation period
  y.TV.est[i] <- forecast::forecast(subset(ts(fdata$Price),end=i-1), # y series up to sample i 
                          model = arima.fit,       # Model trained (Also valid for exponential smoothing models)
                          h=1)$mean              # h is the forecast horizon
  
} #Predigo todos los días con un horizonte determinado
# Cambiamos el 1 si quiero cambiar horizonte (h=1 y end = i-1)

#Para la segunda incluyo la primera y predigo la siguiente

# Remove na values
y.TV.est <- na.omit(y.TV.est)

# Accuracy in TV ----
accuracy(y.TV.est,yTV) #Error de validación

# Plot forecast; para las siguientes predicciones y la última con horizonte h
autoplot(ts(yTV), series = "Real (TV)")+
  forecast::autolayer(ts(y.TV.est), series = "Fitted (TV)")

##########################################################################################
# Once the final model has been chosen, we refit the model with the complete training set
##########################################################################################
y <- ts(fdata$Price, frequency = 7)

# Fit seasonal model with estimated order
arima.fit <- Arima(y,
                   order=c(2,1,1),
                   seasonal = list(order=c(0,1,1), period=7),
                   lambda = NULL,
                   include.constant = FALSE)
summary(arima.fit) # summary of training errors and estimated coefficients
coeftest(arima.fit) # statistical significance of estimated coefficients
autoplot(arima.fit) # root plot

# Check residuals
CheckResiduals.ICAI(arima.fit, bins = 100, lag = 100)

## Future forecast
pred <- forecast(arima.fit, h=7) #Siguientes 7 valores, reajustamos con todos los datos.
y_est <- pred$mean

#El primero lo haría con horizonte 1, el segundo con horizonte 2.... asi hasta horizonte 7.

write.table(y_est, file = "PredArima.dat", col.names = FALSE, row.names
            = FALSE)





