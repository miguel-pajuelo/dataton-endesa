################################################################################
#DATATON ENDESA - PREDICCION DE GENERACION ELECTRICA
################################################################################

library(MLTools)
library(fpp2)
library(lmtest)
library(tseries) #contains adf.test function
library(TSA)
library(NeuralSens)
library(caret)
library(kernlab)
library(nnet)
library(NeuralNetTools)
library(Hmisc)
library(imputeTS)


## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Lectura datos -------------------------------------------------------------------------------------------------
fdata <- readxl::read_excel(file.path(dirname(rstudioapi::getActiveDocumentContext()$path), "Datathon_2026_TR.xlsx"))
fdata <- as.data.frame(fdata)


# Convert to time series object (datos cada 15 min -> 96 observaciones por dia)
fdata_ts <- ts(fdata[, -1], frequency = 96)   # quitamos la columna Date
autoplot(fdata_ts, facets = TRUE)
summary(fdata_ts)


# Imputation of missing values ----------------------------------------------------------
# Grafico de missing values de cada variable
ggplot_na_distribution(fdata_ts[,"Net_Power"],            title = "Net_Power")
ggplot_na_distribution(fdata_ts[,"Amb_Temp"],             title = "Amb_Temp")
ggplot_na_distribution(fdata_ts[,"H2O_Cond_Inlet_Temp"],  title = "H2O_Cond_Inlet_Temp")
ggplot_na_distribution(fdata_ts[,"H2O_Cond_Outlet_Temp"], title = "H2O_Cond_Outlet_Temp")
ggplot_na_distribution(fdata_ts[,"Cool_Water_Flow"],      title = "Cool_Water_Flow")
ggplot_na_distribution(fdata_ts[,"Cond_Press"],           title = "Cond_Press")
ggplot_na_distribution(fdata_ts[,"IGV_Pos"],              title = "IGV_Pos")
ggplot_na_distribution(fdata_ts[,"Sea_Level"],            title = "Sea_Level")

# Imputacion con filtro de Kalman
imp_kalman <- na_kalman(fdata_ts, model = "auto.arima")
ggplot_na_imputations(fdata_ts[,"IGV_Pos"], imp_kalman[,"IGV_Pos"])

# Nos quedamos con Kalman
fdata_ts <- imp_kalman

# Volvemos a data.frame con la fecha, para seguir con el resto del script
fdata <- data.frame(Date = fdata$Date, as.data.frame(fdata_ts))


# Initialize output and input variables ---------------------------------------------------
y_var  <- "Net_Power"
x_vars <- c("Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
            "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level")
fdata.Reg <- fdata[, c(y_var, x_vars)]


### Include lagged variables ------------------------------------------------------------------
# Datos cada 15 min: lag1 = 15 min, lag2 = 30 min, lag4 = 1 hora

fdata.Reg$H2O_Cond_Inlet_Temp_lag1 <- Lag(fdata$H2O_Cond_Inlet_Temp, 1)
fdata.Reg$H2O_Cond_Inlet_Temp_lag4 <- Lag(fdata$H2O_Cond_Inlet_Temp, 4)

fdata.Reg$H2O_Cond_Outlet_Temp_lag1 <- Lag(fdata$H2O_Cond_Outlet_Temp, 1)
fdata.Reg$H2O_Cond_Outlet_Temp_lag4 <- Lag(fdata$H2O_Cond_Outlet_Temp, 4)

fdata.Reg$Cool_Water_Flow_lag1 <- Lag(fdata$Cool_Water_Flow, 1)
fdata.Reg$Cool_Water_Flow_lag4 <- Lag(fdata$Cool_Water_Flow, 4)

fdata.Reg$IGV_Pos_lag1 <- Lag(fdata$IGV_Pos, 1)
fdata.Reg$IGV_Pos_lag4 <- Lag(fdata$IGV_Pos, 4)

fdata.Reg$Net_Power_lag1 <- Lag(fdata$Net_Power, 1)
fdata.Reg$Net_Power_lag2 <- Lag(fdata$Net_Power, 2)


fdata.Reg.tr <- fdata.Reg
#Remove missing values (las primeras 4 filas, por los lags)
fdata.Reg.tr <- na.omit(fdata.Reg.tr)


## Initialize trainControl -----------------------------------------------------------------------
#Use resampling for measuring generalization error
#K-fold with 10 folds
ctrl_tune <- trainControl(method = "cv",                     
                          number = 10,
                          summaryFunction = defaultSummary,    #Performance summary for comparing models in hold-out samples.
                          savePredictions = TRUE)              #save predictions


#------------ Neural network --------------------------------
set.seed(150) #For replication
mlp.fit = train(form = y ~ x1 + x2 + x1_lag1 + y_lag1 + y_lag7 + x2_lag1, #Use formula method to account for categorical variables
                data = fdata.Reg.tr, 
                method = "nnet",
                linout = TRUE,
                # tuneGrid = data.frame(size =5, decay = 0),
                tuneGrid = expand.grid(size = seq(10,20,length.out =3), decay =  10^(c(-3:0))),
                maxit = 200,
                preProcess = c("center","scale"),
                trControl = ctrl_tune, 
                metric = "RMSE")

mlp.fit #information about the resampling settings
ggplot(mlp.fit)+scale_x_log10()
plotnet(mlp.fit$finalModel) #Plot the network
SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis


#################################################################################
## Forecast for new data with h = 14
#################################################################################

#------------------READ TEST FILE --------------------------
fdata.Reg.tv <- readxl::read_excel("test.xlsx")
fdata.Reg.tv <- as.data.frame(fdata.Reg.tv)

#----------------------------------------------------

fdata.Reg.new <- fdata.Reg.tv[, c("x1", "x2")]

#create auxiliary output variable
fdata.Reg.new$y <- rep(NA, nrow(fdata.Reg.new))

#join the datasets
fdata.join <- rbind(fdata[, c("x1", "x2", "y")], fdata.Reg.new)

#create lagged variables
fdata.join$x1_lag1 <- Lag(fdata.join$x1, 1)
fdata.join$x1_lag7 <- Lag(fdata.join$x1, 7)

fdata.join$x2_lag1 <- Lag(fdata.join$x2, 1)
fdata.join$x2_lag7 <- Lag(fdata.join$x2, 7)

fdata.join$y_lag1 <- Lag(fdata.join$y, 1)
fdata.join$y_lag7 <- Lag(fdata.join$y, 7)


#loop for forecasting
tstart <- nrow(fdata)
h <- nrow(fdata.Reg.tv)

for (i in (tstart+1):(tstart+h)) {
  
  #Predict and substitute in output variable
  fdata.join$y[i] <- predict(mlp.fit, newdata = fdata.join[i, ])
  
  #Recalculate lagged variables using the estimated value
  fdata.join$y_lag1 <- Lag(fdata.join$y, 1)
  fdata.join$y_lag7 <- Lag(fdata.join$y, 7)
}

#forecasts
predicciones <- fdata.join$y[(tstart+1):(tstart+h)]

#Show predictions in console
cat("\nPredicciones de y para las", h, "observaciones:\n")
print(predicciones)

#Show predictions with input variables
resultados <- data.frame(
  x1 = fdata.Reg.tv$x1,
  x2 = fdata.Reg.tv$x2,
  y_pred = predicciones
)

print(resultados)