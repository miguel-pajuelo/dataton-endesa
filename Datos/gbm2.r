################################################################################
# DATATHON ENDESA - PREDICCION DE GENERACION ELECTRICA  (version GBM)
################################################################################
# Cambios respecto al script del MLP:
#  1) Prediccion DIRECTA, sin lags de Net_Power ni bucle recursivo.
#     Las variables de la planta del test son conocidas, asi que no hace falta
#     arrastrar predicciones propias (el error se acumulaba paso a paso).
#  2) Variables fisicas nuevas:
#       dT = salto termico del agua de circulacion (salida - entrada)
#       Q  = dT * caudal  -> proporcional al calor que se evacua en el
#            condensador, es decir, a la potencia de la turbina de vapor
#  3) Lag y "lead" (valor futuro) de 15 min de IGV_Pos y Q, mas una media movil
#     centrada de 5 pasos. Los leads son legitimos porque las exogenas del test
#     se conocen enteras. 15 variables en total.
#  4) Hiperparametros elegidos con CV por dias completos (no con la validacion)
#     y una rejilla de modelos sencillos, para limitar el overfitting.
#  5) Media de 5 GBM con semillas distintas: reduce la varianza del modelo.
################################################################################

library(fpp2)
library(caret)
library(gbm)
library(imputeTS)
library(zoo)


## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Lectura datos -------------------------------------------------------------------------------------------------
fdata <- readxl::read_excel("Datathon_2026_TR.xlsx")
fdata <- as.data.frame(fdata)

fdata.tv <- readxl::read_excel("Datathon_2026_TS.xlsx")
fdata.tv <- as.data.frame(fdata.tv)


# Imputation of missing values (Kalman, igual que en el MLP) ---------------------------------------
fdata_ts   <- ts(fdata[, -1], frequency = 96)
imp_kalman <- na_kalman(fdata_ts, model = "auto.arima")
fdata      <- data.frame(Date = fdata$Date, as.data.frame(imp_kalman))


# Variables de entrada -----------------------------------------------------------------------------
vars_x <- c("Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp",
            "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level")

# Desplaza un vector k posiciones: k > 0 -> lag (pasado), k < 0 -> lead (futuro)
shift_v <- function(x, k) {
  if (k > 0) c(rep(NA, k), head(x, -k)) else c(tail(x, k), rep(NA, -k))
}

# Construye todas las variables del modelo
# flow_min se fija con el TR para que Q se calcule igual en train y en test
make_feats <- function(df, flow_min) {
  X <- df[, vars_x]
  X$dT <- df$H2O_Cond_Outlet_Temp - df$H2O_Cond_Inlet_Temp
  X$Q  <- X$dT * (df$Cool_Water_Flow - flow_min)
  base <- X

  # Solo las dos variables que mas explican (IGV -> turbina de gas, Q -> turbina de vapor),
  # con un unico lag y lead de 15 min y una media movil centrada de 5 pasos (1h15)
  for (v in c("IGV_Pos", "Q")) {
    X[[paste0(v, "_l1")]]  <- shift_v(base[[v]],  1)          # pasado
    X[[paste0(v, "_f1")]]  <- shift_v(base[[v]], -1)          # futuro
    X[[paste0(v, "_rm5")]] <- rollapply(base[[v]], width = 5, FUN = mean,
                                        partial = TRUE, align = "center")
  }

  # Los lags/leads dejan NA en los extremos: se rellenan con el valor mas cercano
  X[] <- lapply(X, function(z) na.locf(na.locf(z, na.rm = FALSE), fromLast = TRUE, na.rm = FALSE))
  X
}

# Unimos TR y TS ANTES de crear lags/leads, para que en la frontera
# (29/07 23:45 -> 30/07 00:00) se usen los valores reales del otro fichero
fdata.join <- rbind(fdata[, c("Date", vars_x)], fdata.tv[, c("Date", vars_x)])
flow_min   <- min(fdata$Cool_Water_Flow)
X.join     <- make_feats(fdata.join, flow_min)

n_tr  <- nrow(fdata)
n_ts  <- nrow(fdata.tv)
X.all <- X.join[1:n_tr, ]                       # filas del TR
X.ts  <- X.join[(n_tr + 1):(n_tr + n_ts), ]     # filas del TS
y.all <- fdata$Net_Power


### Split train / validation ------------------------------------------------------------------
# Ultimos 2 dias (192 datos) para validacion, misma longitud que el test
n_val   <- 192
idx_tr  <- 1:(n_tr - n_val)
idx_val <- (n_tr - n_val + 1):n_tr


#------------ Seleccion de hiperparametros con CV por DIAS --------------------
# Una CV k-fold aleatoria pone en train y en test cuartos de hora consecutivos,
# casi identicos, y el error sale demasiado optimista. Aqui cada fold deja fuera
# dias completos. La validacion (ultimos 2 dias) NO se usa para elegir nada:
# solo para comprobar al final.
dias_tr <- as.Date(fdata$Date[idx_tr])
set.seed(150)
folds_dia <- groupKFold(dias_tr, k = 7)

ctrl_dias <- trainControl(method = "cv",
                          index = folds_dia,
                          summaryFunction = defaultSummary,
                          savePredictions = TRUE)

# Rejilla pequena y orientada a modelos sencillos:
#  - arboles de profundidad 2-3
#  - minimo 20-30 observaciones por hoja (evita hojas que memorizan picos)
#  - pocos arboles con learning rate bajo
gbm_grid <- expand.grid(n.trees           = c(300, 400, 600),
                        interaction.depth = c(2, 3),
                        shrinkage         = 0.03,
                        n.minobsinnode    = c(20, 30))

set.seed(150)
gbm.cv <- train(x = X.all[idx_tr, ], y = y.all[idx_tr],
                method = "gbm",
                distribution = "gaussian",
                bag.fraction = 0.7,
                tuneGrid = gbm_grid,
                trControl = ctrl_dias,
                metric = "MAE",
                maximize = FALSE,
                verbose = FALSE)

gbm.cv
gbm.cv$bestTune
ggplot(gbm.cv)


#------------ Ensemble de 5 semillas con los hiperparametros elegidos --------
seeds <- 1:5

fit_gbm_ens <- function(X, y) {
  lapply(seeds, function(s) {
    set.seed(s)
    train(x = X, y = y,
          method = "gbm",
          distribution = "gaussian",
          bag.fraction = 0.7,
          tuneGrid = gbm.cv$bestTune,
          trControl = trainControl(method = "none"),
          verbose = FALSE)
  })
}
predict_ens <- function(models, X) rowMeans(sapply(models, predict, newdata = X))


#################################################################################
## Validacion
#################################################################################
gbm.val <- fit_gbm_ens(X.all[idx_tr, ], y.all[idx_tr])

y_real     <- y.all[idx_val]
y_pred_val <- predict_ens(gbm.val, X.all[idx_val, ])

MAE_val  <- mean(abs(y_real - y_pred_val))
RMSE_val <- sqrt(mean((y_real - y_pred_val)^2))
MAE_naive <- mean(abs(y_real - y.all[n_tr - n_val]))

# Diagnostico de overfitting: error en train vs CV por dias vs validacion.
# Si train es mucho menor que los otros dos, el modelo memoriza.
MAE_train <- mean(abs(y.all[idx_tr] - predict_ens(gbm.val, X.all[idx_tr, ])))
MAE_cv    <- min(gbm.cv$results$MAE)

cat("\nMAE  train                 :", round(MAE_train, 4), "\n")
cat("MAE  CV por dias           :", round(MAE_cv, 4), "\n")
cat("MAE  validacion (192 pasos):", round(MAE_val, 4), "\n")
cat("RMSE validacion (192 pasos):", round(RMSE_val, 4), "\n")
cat("MAE  naive (referencia)    :", round(MAE_naive, 4), "\n")

# Importancia de variables (primer modelo del ensemble)
summary(gbm.val[[1]]$finalModel, plotit = FALSE)[1:15, ]

#------------ Grafico real vs predicho ------------
val_plot <- data.frame(Date = fdata$Date[idx_val], Real = y_real, Prediccion = y_pred_val)

ggplot(val_plot, aes(x = Date)) +
  geom_line(aes(y = Real,       colour = "Real")) +
  geom_line(aes(y = Prediccion, colour = "Prediccion")) +
  labs(title = paste("Validacion GBM - MAE =", round(MAE_val, 4)), y = "Net_Power", colour = "")


#################################################################################
## MODELO DEFINITIVO: mismo GBM entrenado con TODOS los datos de TR
#################################################################################
gbm.final <- fit_gbm_ens(X.all, y.all)

predicciones_final <- predict_ens(gbm.final, X.ts)


#------------ Comprobaciones antes de subir ------------
length(predicciones_final)        # tiene que ser 192
sum(is.na(predicciones_final))    # tiene que ser 0
range(fdata.tv$Date)              # 2026-07-30 00:00:00 -> 2026-07-31 23:45:00

plot_final <- data.frame(Date      = c(fdata$Date[idx_val], fdata.tv$Date),
                         Net_Power = c(y.all[idx_val], predicciones_final),
                         Tipo      = c(rep("Real (TR)", n_val), rep("Prediccion (TS)", n_ts)))

ggplot(plot_final, aes(x = Date, y = Net_Power, colour = Tipo)) +
  geom_line() +
  labs(title = "Prediccion final de Net_Power (GBM)", colour = "")


#------------ Fichero de predicciones ------------
write.table(predicciones_final,
            file      = "predicciones_Net_Power_gbm.txt",
            row.names = FALSE,
            col.names = FALSE,
            quote     = FALSE)

cat("\nFichero guardado en:", file.path(getwd(), "predicciones_Net_Power_gbm.txt"), "\n")
