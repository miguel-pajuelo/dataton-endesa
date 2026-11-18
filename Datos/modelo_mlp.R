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
fdata <- readxl::read_excel("Datathon_2026_TR.xlsx")
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

# Graficos de las columnas imputadas (original con NA vs imputada)
# OJO: tiene que ir ANTES de sobrescribir fdata_ts, porque necesita la serie con NAs
ggplot_na_imputations(fdata_ts[,"H2O_Cond_Inlet_Temp"],  imp_kalman[,"H2O_Cond_Inlet_Temp"],  title = "H2O_Cond_Inlet_Temp")
ggplot_na_imputations(fdata_ts[,"H2O_Cond_Outlet_Temp"], imp_kalman[,"H2O_Cond_Outlet_Temp"], title = "H2O_Cond_Outlet_Temp")
ggplot_na_imputations(fdata_ts[,"Cool_Water_Flow"],      imp_kalman[,"Cool_Water_Flow"],      title = "Cool_Water_Flow")
ggplot_na_imputations(fdata_ts[,"Cond_Press"],           imp_kalman[,"Cond_Press"],           title = "Cond_Press")
ggplot_na_imputations(fdata_ts[,"IGV_Pos"],              imp_kalman[,"IGV_Pos"],              title = "IGV_Pos")

# Todas las series ya imputadas
autoplot(imp_kalman, facets = TRUE)

# Nos quedamos con Kalman
fdata_ts <- imp_kalman

# Volvemos a data.frame con la fecha, para seguir con el resto del script
fdata <- data.frame(Date = fdata$Date, as.data.frame(fdata_ts))


# Initialize output and input variables ---------------------------------------------------
fdata.Reg <- fdata[, c("Net_Power", "Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
                       "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level")]


### Include lagged variables ------------------------------------------------------------------
# Datos cada 15 min: lag1 = 15 min, lag2 = 30 min, lag4 = 1 hora, lag50 = 12h30 (marea)

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

# Lags de marea (marea semidiurna ~12h25min = ~50 pasos de 15 min)
fdata.Reg$Sea_Level_lag50 <- Lag(fdata$Sea_Level, 50)
fdata.Reg$Net_Power_lag50 <- Lag(fdata$Net_Power, 50)


### Split train / validation ------------------------------------------------------------------
# Reservamos los ultimos 2 dias (192 datos de 15 min) para validacion,
# la misma longitud que el test. El MLP se entrena SOLO con el resto.
n_val   <- 192
n_total <- nrow(fdata.Reg)
idx_tr  <- 1:(n_total - n_val)
idx_val <- (n_total - n_val + 1):n_total

fdata.Reg.tr <- fdata.Reg[idx_tr, ]
#Remove missing values (las primeras 50 filas, por los lags)
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
mlp.fit = train(form = Net_Power ~ Amb_Temp + H2O_Cond_Inlet_Temp + H2O_Cond_Outlet_Temp +
                  Cool_Water_Flow + Cond_Press + IGV_Pos + Sea_Level +
                  H2O_Cond_Inlet_Temp_lag1 + H2O_Cond_Inlet_Temp_lag4 +
                  H2O_Cond_Outlet_Temp_lag1 + H2O_Cond_Outlet_Temp_lag4 +
                  Cool_Water_Flow_lag1 + Cool_Water_Flow_lag4 +
                  IGV_Pos_lag1 + IGV_Pos_lag4 +
                  Net_Power_lag1 + Net_Power_lag2 +
                  Sea_Level_lag50 + Net_Power_lag50,
                data = fdata.Reg.tr,
                method = "nnet",
                linout = TRUE,
                # tuneGrid = data.frame(size =5, decay = 0),
                tuneGrid = expand.grid(size = seq(10,20,length.out =3), decay =  10^(c(-3:0))),
                maxit = 200,
                preProcess = c("center","scale"),
                trControl = ctrl_tune,
                metric = "MAE")

mlp.fit #information about the resampling settings
ggplot(mlp.fit)+scale_x_log10()
plotnet(mlp.fit$finalModel) #Plot the network
SensAnalysisMLP(mlp.fit) #Statistical sensitivity analysis


#################################################################################
## Validacion: prediccion recursiva de los 2 dias reservados (192 pasos)
#################################################################################
fdata.val <- fdata.Reg

# Guardamos los valores reales y los borramos (como si no los conocieramos)
y_real <- fdata.val$Net_Power[idx_val]
fdata.val$Net_Power[idx_val] <- NA

# Recalculamos los lags de Net_Power sin los valores reales
fdata.val$Net_Power_lag1  <- Lag(fdata.val$Net_Power, 1)
fdata.val$Net_Power_lag2  <- Lag(fdata.val$Net_Power, 2)
fdata.val$Net_Power_lag50 <- Lag(fdata.val$Net_Power, 50)

for (i in idx_val) {
  
  #Predict and substitute in output variable
  fdata.val$Net_Power[i] <- predict(mlp.fit, newdata = fdata.val[i, ])
  
  #Recalculate lagged variables using the estimated value
  fdata.val$Net_Power_lag1  <- Lag(fdata.val$Net_Power, 1)
  fdata.val$Net_Power_lag2  <- Lag(fdata.val$Net_Power, 2)
  fdata.val$Net_Power_lag50 <- Lag(fdata.val$Net_Power, 50)
}

y_pred_val <- fdata.val$Net_Power[idx_val]

#------------ Errores ------------
MAE_val  <- mean(abs(y_real - y_pred_val))
RMSE_val <- sqrt(mean((y_real - y_pred_val)^2))

# Referencia: modelo naive (repetir el ultimo valor conocido)
y_naive   <- rep(fdata$Net_Power[n_total - n_val], n_val)
MAE_naive <- mean(abs(y_real - y_naive))

cat("\nMAE  validacion (192 pasos):", round(MAE_val, 4), "\n")
cat("RMSE validacion (192 pasos):", round(RMSE_val, 4), "\n")
cat("MAE  naive (referencia)    :", round(MAE_naive, 4), "\n")

#------------ Grafico real vs predicho ------------
val_plot <- data.frame(Date = fdata$Date[idx_val], Real = y_real, Prediccion = y_pred_val)

ggplot(val_plot, aes(x = Date)) +
  geom_line(aes(y = Real,       colour = "Real")) +
  geom_line(aes(y = Prediccion, colour = "Prediccion")) +
  labs(title = paste("Validacion - MAE =", round(MAE_val, 4)), y = "Net_Power", colour = "")


#################################################################################
## Forecast for new data with h = 192 (2 dias cada 15 min)
## (mismo MLP; los lags usan los datos reales de TR, incluidos los 2 dias de validacion)
#################################################################################

#------------------READ TEST FILE --------------------------
fdata.Reg.tv <- readxl::read_excel("Datathon_2026_TS.xlsx")
fdata.Reg.tv <- as.data.frame(fdata.Reg.tv)

#----------------------------------------------------

fdata.Reg.new <- fdata.Reg.tv[, c("Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
                                  "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level")]

#create auxiliary output variable
fdata.Reg.new$Net_Power <- rep(NA, nrow(fdata.Reg.new))

#join the datasets
fdata.join <- rbind(fdata[, c("Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
                              "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level", "Net_Power")],
                    fdata.Reg.new)

#create lagged variables
fdata.join$H2O_Cond_Inlet_Temp_lag1 <- Lag(fdata.join$H2O_Cond_Inlet_Temp, 1)
fdata.join$H2O_Cond_Inlet_Temp_lag4 <- Lag(fdata.join$H2O_Cond_Inlet_Temp, 4)

fdata.join$H2O_Cond_Outlet_Temp_lag1 <- Lag(fdata.join$H2O_Cond_Outlet_Temp, 1)
fdata.join$H2O_Cond_Outlet_Temp_lag4 <- Lag(fdata.join$H2O_Cond_Outlet_Temp, 4)

fdata.join$Cool_Water_Flow_lag1 <- Lag(fdata.join$Cool_Water_Flow, 1)
fdata.join$Cool_Water_Flow_lag4 <- Lag(fdata.join$Cool_Water_Flow, 4)

fdata.join$IGV_Pos_lag1 <- Lag(fdata.join$IGV_Pos, 1)
fdata.join$IGV_Pos_lag4 <- Lag(fdata.join$IGV_Pos, 4)

fdata.join$Net_Power_lag1 <- Lag(fdata.join$Net_Power, 1)
fdata.join$Net_Power_lag2 <- Lag(fdata.join$Net_Power, 2)

fdata.join$Sea_Level_lag50 <- Lag(fdata.join$Sea_Level, 50)
fdata.join$Net_Power_lag50 <- Lag(fdata.join$Net_Power, 50)


#loop for forecasting
tstart <- nrow(fdata)
h <- nrow(fdata.Reg.tv)

for (i in (tstart+1):(tstart+h)) {
  
  #Predict and substitute in output variable
  fdata.join$Net_Power[i] <- predict(mlp.fit, newdata = fdata.join[i, ])
  
  #Recalculate lagged variables using the estimated value
  fdata.join$Net_Power_lag1  <- Lag(fdata.join$Net_Power, 1)
  fdata.join$Net_Power_lag2  <- Lag(fdata.join$Net_Power, 2)
  fdata.join$Net_Power_lag50 <- Lag(fdata.join$Net_Power, 50)
}

#forecasts
predicciones <- fdata.join$Net_Power[(tstart+1):(tstart+h)]

#Show predictions in console
cat("\nPredicciones de Net_Power para las", h, "observaciones:\n")
print(predicciones)

#Show predictions with input variables
resultados <- data.frame(
  Date           = fdata.Reg.tv$Date,
  fdata.Reg.new[, c("Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
                    "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level")],
  Net_Power_pred = predicciones
)

#################################################################################
## MODELO DEFINITIVO: mismo MLP entrenado con TODOS los datos de TR
#################################################################################
# Usamos los hiperparametros (size, decay) que eligio la validacion cruzada en mlp.fit,
# asi no hay que volver a hacer la busqueda del grid.

fdata.Reg.final <- na.omit(fdata.Reg)   # todos los datos (solo quitamos las 50 primeras filas por los lags)

mlp.fit$bestTune   # hiperparametros elegidos

set.seed(150) #For replication
mlp.final = train(form = Net_Power ~ Amb_Temp + H2O_Cond_Inlet_Temp + H2O_Cond_Outlet_Temp +
                    Cool_Water_Flow + Cond_Press + IGV_Pos + Sea_Level +
                    H2O_Cond_Inlet_Temp_lag1 + H2O_Cond_Inlet_Temp_lag4 +
                    H2O_Cond_Outlet_Temp_lag1 + H2O_Cond_Outlet_Temp_lag4 +
                    Cool_Water_Flow_lag1 + Cool_Water_Flow_lag4 +
                    IGV_Pos_lag1 + IGV_Pos_lag4 +
                    Net_Power_lag1 + Net_Power_lag2 +
                    Sea_Level_lag50 + Net_Power_lag50,
                  data = fdata.Reg.final,
                  method = "nnet",
                  linout = TRUE,
                  tuneGrid = mlp.fit$bestTune,
                  maxit = 200,
                  preProcess = c("center","scale"),
                  trControl = trainControl(method = "none"))

mlp.final


#------------ Prediccion recursiva del test (30/07/2026 00:00 - 31/07/2026 23:45) ------------
# Partimos de fdata.join (TR + TS) y borramos las predicciones del modelo anterior
fdata.join.final <- fdata.join
fdata.join.final$Net_Power[(tstart+1):(tstart+h)] <- NA

fdata.join.final$Net_Power_lag1  <- Lag(fdata.join.final$Net_Power, 1)
fdata.join.final$Net_Power_lag2  <- Lag(fdata.join.final$Net_Power, 2)
fdata.join.final$Net_Power_lag50 <- Lag(fdata.join.final$Net_Power, 50)

for (i in (tstart+1):(tstart+h)) {
  
  #Predict and substitute in output variable
  fdata.join.final$Net_Power[i] <- predict(mlp.final, newdata = fdata.join.final[i, ])
  
  #Recalculate lagged variables using the estimated value
  fdata.join.final$Net_Power_lag1  <- Lag(fdata.join.final$Net_Power, 1)
  fdata.join.final$Net_Power_lag2  <- Lag(fdata.join.final$Net_Power, 2)
  fdata.join.final$Net_Power_lag50 <- Lag(fdata.join.final$Net_Power, 50)
}

predicciones_final <- fdata.join.final$Net_Power[(tstart+1):(tstart+h)]


#------------ Comprobaciones antes de subir ------------
length(predicciones_final)        # tiene que ser 192
sum(is.na(predicciones_final))    # tiene que ser 0
range(fdata.Reg.tv$Date)          # 2026-07-30 00:00:00 -> 2026-07-31 23:45:00

# Grafico: ultimos 2 dias reales de TR + prediccion del test
plot_final <- data.frame(Date      = c(fdata$Date[idx_val], fdata.Reg.tv$Date),
                         Net_Power = c(fdata$Net_Power[idx_val], predicciones_final),
                         Tipo      = c(rep("Real (TR)", n_val), rep("Prediccion (TS)", h)))

ggplot(plot_final, aes(x = Date, y = Net_Power, colour = Tipo)) +
  geom_line() +
  labs(title = "Prediccion final de Net_Power", colour = "")


#------------ Fichero para subir a la Plataforma ------------
# Texto plano, una unica columna, 192 filas, sin encabezado
write.table(predicciones_final,
            file      = "predicciones_Net_Power.txt",
            row.names = FALSE,
            col.names = FALSE,
            quote     = FALSE)

cat("\nFichero guardado en:", file.path(getwd(), "predicciones_Net_Power.txt"), "\n")