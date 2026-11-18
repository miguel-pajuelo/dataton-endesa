################################################################################
# Lab 8: Nonlinear forecasting - Gradient Boosting Machines (GBM)
################################################################################

library(caret)
library(gbm)
library(readxl)
library(Hmisc)
library(MLTools)
library(forecast)
library(ggplot2)
#La librería imputeTs me convierte en serie
#1º hago plot con las columnas: si hay outliers sale línea roja
#Lo más sencillo es hacer interpolación
#Podemos hacer descomposición estacional y llamando na_seasplit(fdata_ts)
#representa la nueva serie con los valores ya rellenados, funciona bien
#con componente estacional marcada
#Podemos hacer descomposición na_kalman(fdata_ts,model="auto.arima"), 
#que sirve también para rellenar los NA
#Nos quedamos con la que nos sirva en este caso

## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Read data -------------------------------------------------------------------------------------------------
fdata <- readxl::read_excel("datos.xlsx")
fdata <- as.data.frame(fdata)
summary(fdata) #Esto nos dice NAs: nº y esos son los NA que hay en esa col

# Initialize output and input variables
fdata.Reg <- fdata[, c("y", "x1", "x2")] 

# Include lagged variables
fdata.Reg$x1_lag1 <- Lag(fdata$x1, 1)
fdata.Reg$x1_lag7 <- Lag(fdata$x1, 7)
fdata.Reg$x2_lag1 <- Lag(fdata$x2, 1)
fdata.Reg$x2_lag7 <- Lag(fdata$x2, 7)
fdata.Reg$y_lag1 <- Lag(fdata$y, 1)
fdata.Reg$y_lag7 <- Lag(fdata$y, 7)

# Remove missing values
fdata.Reg.tr <- na.omit(fdata.Reg)


# 10-fold cross-validation; same as the original XGBoost practice.
ctrl_tune <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = defaultSummary,
  savePredictions = "final"
)
#Utilizamos 10 folds y definimos el grid de hiperparámetros

set.seed(150)

# GBM tuning: 3 x 2 x 3 = 18 combinations.
gbmGrid <- expand.grid(
  n.trees = c(50, 100, 150, 200),
  interaction.depth = c(2, 3, 5),      # Probamos profundidades un poco menores
  shrinkage = c(0.01, 0.05, 0.1),      # Mantenemos tasas de aprendizaje variadas
  n.minobsinnode = 10 #Mínimo número de observaciones en 10
)

# Train GBM through caret (Gaussian regression, no feature scaling needed).
gbm.fit <- train(
  y ~ ., #Explicamos la demanda en función de todas las vars
  #Si solo quisiéramos unas pocas haremos nombre var + nombre var...
  data = fdata.Reg.tr,
  method = "gbm", #Solo cambiamos el método
  tuneGrid = gbmGrid,
  trControl = ctrl_tune,
  metric = "RMSE",
  verbose = FALSE
)
#Sale en validación que si uso más árboles quizás el error es menor, al
#igual que si bajamos el shrinkage.

#Para ver si es no lineal ploteo las gráficas de relación entradas y retardos con salida


# Cross-validation results and optimal hyperparameters.
print(gbm.fit)
print(gbm.fit$bestTune)
print(ggplot(gbm.fit))

# Variable importance. (como el análisis de sensibilidades)
var_imp <- varImp(gbm.fit)
print(var_imp)
ggplot(var_imp)

# In-sample predictions and diagnostics
gbm_pred <- predict(gbm.fit, newdata = fdata.Reg.tr)

#Uso todas las variables de entrada (DEM es salida), y el segundo argumento es la variable de salida
PlotModelDiagnosis(fdata.Reg.tr[, names(fdata.Reg.tr) != "y"], fdata.Reg.tr$y, gbm_pred, together = TRUE)
#Valor residuos de modelos en función de var.

#Error measurements
accuracy(fdata.Reg.tr$y,gbm_pred)

# Model diagnosis-> no saldrá ruido blanco. Calculamos error de train
CheckResiduals.ICAI(gbm.fit, lag=50)

#Check Cross correlation residuals - expl. variables
res <- fdata.Reg.tr$y - gbm_pred
ccf(y = res, x = fdata.Reg.tr$x1)
ccf(y = res, x = fdata.Reg.tr$x2)

#Plot predicted values
plot(fdata.Reg.tr$y,type="l")
lines(gbm_pred,col = "red")
title("Actual (black) vs predicted (red)")


################################################################################
# Recursive forecasting
################################################################################
# Future values of the exogenous variables -> Ahí solo están vars explicativas
fdata.Reg.tv <- as.data.frame(read_excel("test.xlsx"))
#Leemos conj. validación, presentamos retardos

# Historical data followed by future exogenous values (DEM unknown)
fdata.Reg.new <- fdata.Reg.tv[, c("x1", "x2")]
fdata.Reg.new$y <- NA

fdata.join <- rbind(fdata[, c("y", "x1", "x2")], fdata.Reg.new)

# Lagged variables
fdata.join$x1_lag1 <- Lag(fdata.join$x1, 1)
fdata.join$x1_lag7 <- Lag(fdata.join$x1, 7)
fdata.join$x2_lag1 <- Lag(fdata.join$x2, 1)
fdata.join$x2_lag7 <- Lag(fdata.join$x2, 7)
fdata.join$y_lag1 <- Lag(fdata.join$y, 1)
fdata.join$y_lag7 <- Lag(fdata.join$y, 7)

# Each prediction is used as lagged DEM in the following days
tstart <- nrow(fdata)
h <- nrow(fdata.Reg.tv)  # forecast horizon = number of rows in the test file
#Predecimos horizontes de pred pedido en cada paso predigo muestra sig, y 
#genero retardos de esa variable con esa predicción
#Como no tengo el valor real lo hago con el predicho, asi que la retardo
for (i in (tstart + 1):(tstart + h)) {
  #Predict and substitute in output variable so it can be used in following forecasts
  fdata.join$y[i] <- predict(gbm.fit, newdata = fdata.join[i, ])
  #Recalculate lagged variables using the estimated value
  fdata.join$y_lag1 <- Lag(fdata.join$y, 1) #retardo regular y estacionalmente
  fdata.join$y_lag7 <- Lag(fdata.join$y, 7)
}
#Aplicamos estrategia recursiva
y_pred <- fdata.join$y[(tstart + 1):(tstart + h)]
y_pred


# Save predictions
writeLines(c("qZe5kn", as.character(y_pred)), "T1_sol.csv")

