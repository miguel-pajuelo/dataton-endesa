################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 05_Modelo_GBM.R
# Objetivo:
#   Ajustar un modelo Gradient Boosting Machine (GBM) estándar sobre datos enriquecidos.
#   Proporciona una alternativa robusta basada en árboles que complementa a XGBoost y MLP.
################################################################################

# 1. Carga de paquetes ---------------------------------------------------------
paquetes <- c("readxl", "writexl", "caret", "gbm", "ggplot2")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) install.packages(nuevos)

library(readxl)
library(writexl)
library(caret)
library(gbm)
library(ggplot2)

# 2. Configuración y carga de datos --------------------------------------------
tryCatch({
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}, error = function(e) {})

buscar_fichero <- function(nombre) {
  rutas <- c(
    nombre,
    file.path("Datos", nombre),
    file.path("..", "Datos", nombre),
    file.path("..", nombre)
  )
  encontrada <- rutas[file.exists(rutas)][1]
  if (is.na(encontrada)) stop(paste("No se encuentra el fichero:", nombre))
  return(encontrada)
}

ruta_tr <- buscar_fichero("Datathon_2026_TR_features.xlsx")
ruta_ts <- buscar_fichero("Datathon_2026_TS_features.xlsx")
dir_salida <- dirname(ruta_tr)

df_tr <- as.data.frame(read_excel(ruta_tr))
df_ts <- as.data.frame(read_excel(ruta_ts))

target <- "Net_Power"
columnas_ignorar <- c("Date", "Net_Power", "Net_Power_lag1", "Net_Power_lag2", "Net_Power_lag4")
features <- setdiff(names(df_tr), columnas_ignorar)

df_model_tr <- na.omit(df_tr[, c(target, features)])

# Partición de validación fija (192 pasos)
h <- 192
n_total <- nrow(df_model_tr)
idx_train <- 1:(n_total - h)
idx_val   <- (n_total - h + 1):n_total

train_split <- df_model_tr[idx_train, ]
val_split   <- df_model_tr[idx_val, ]

# 3. Entrenamiento GBM con Caret -----------------------------------------------
ctrl_cv <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = defaultSummary
)

gbm_grid <- expand.grid(
  n.trees           = c(100, 200, 350),
  interaction.depth = c(3, 5),
  shrinkage         = c(0.02, 0.05, 0.1),
  n.minobsinnode    = c(10)
)

cat("\n--- Entrenando GBM ---\n")
set.seed(150)
gbm_cv <- train(
  as.formula(paste(target, "~ .")),
  data = train_split,
  method = "gbm",
  distribution = "gaussian",
  tuneGrid = gbm_grid,
  trControl = ctrl_cv,
  metric = "MAE",
  verbose = FALSE
)

# 4. Evaluación en Validación Local --------------------------------------------
val_pred_gbm <- predict(gbm_cv, newdata = val_split)
mae_gbm <- mean(abs(val_split[[target]] - val_pred_gbm))

cat("\n=======================================================\n")
cat(sprintf(">>> MAE VALIDACIÓN LOCAL GBM: %.5f <<<\n", mae_gbm))
cat("=======================================================\n")

# 5. Reentrenamiento y Predicción TEST -----------------------------------------
set.seed(150)
gbm_final <- train(
  as.formula(paste(target, "~ .")),
  data = df_model_tr,
  method = "gbm",
  distribution = "gaussian",
  tuneGrid = gbm_cv$bestTune,
  trControl = trainControl(method = "none"),
  verbose = FALSE
)

pred_test_gbm <- predict(gbm_final, newdata = df_ts[, features])

# 6. Exportación ---------------------------------------------------------------
ruta_entrega_gbm <- file.path(dir_salida, "predicciones_GBM.txt")
write.table(
  pred_test_gbm,
  file = ruta_entrega_gbm,
  row.names = FALSE,
  col.names = FALSE
)

cat("Fichero guardado en:", ruta_entrega_gbm, "\n")
