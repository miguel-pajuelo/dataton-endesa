################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 04_Modelo_XGBoost.R
# Objetivo:
#   Ajustar un modelo XGBoost (Extreme Gradient Boosting) sobre los datos
#   enriquecidos. XGBoost gestiona de forma nativa la no linealidad, interacciones
#   físicas y selecciona automáticamente las variables más relevantes.
################################################################################

# 1. Carga de paquetes ---------------------------------------------------------
paquetes <- c("readxl", "writexl", "caret", "xgboost", "ggplot2")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) install.packages(nuevos)

library(readxl)
library(writexl)
library(caret)
library(xgboost)
library(ggplot2)

# 2. Configuración y carga de datos enriquecidos --------------------------------
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

cat("Cargando datasets enriquecidos:\n")
cat(" - Train:", ruta_tr, "\n")
cat(" - Test: ", ruta_ts, "\n")

df_tr <- as.data.frame(read_excel(ruta_tr))
df_ts <- as.data.frame(read_excel(ruta_ts))

target <- "Net_Power"

# Variables predictoras completas
columnas_ignorar <- c("Date", "Net_Power", "Net_Power_lag1", "Net_Power_lag2", "Net_Power_lag4")
features <- setdiff(names(df_tr), columnas_ignorar)

# Limpiar NAs iniciales debidos a retardos
df_model_tr <- na.omit(df_tr[, c(target, features)])
cat("Variables de entrada:", length(features), "\n")
cat("Filas de entrenamiento disponibles:", nrow(df_model_tr), "\n")

# 3. Partición de Validación Local (Misma que MLP: últimos 2 días = 192 pasos) --
h <- 192
n_total <- nrow(df_model_tr)
idx_train <- 1:(n_total - h)
idx_val   <- (n_total - h + 1):n_total

train_split <- df_model_tr[idx_train, ]
val_split   <- df_model_tr[idx_val, ]

# 4. Configuración y Grid de Hiperparámetros XGBoost ---------------------------
ctrl_cv <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = defaultSummary,
  savePredictions = "final"
)

# Rejilla optimizada para rapidez y prevención de sobreajuste
xgb_grid <- expand.grid(
  nrounds          = c(150, 300, 500),
  max_depth        = c(3, 4, 6),           # Profundidades moderadas para evitar overfitting
  eta              = c(0.02, 0.05, 0.1),   # Tasas de aprendizaje
  gamma            = c(0, 0.1),            # Regularización por división
  colsample_bytree = c(0.7, 0.85),         # Muestreo aleatorio de columnas
  min_child_weight = c(1, 3),              # Peso mínimo en nodo hoja
  subsample        = c(0.8)                # Muestreo de filas
)

cat("\n=======================================================\n")
cat(" ENTRENANDO XGBOOST (Optimizando MAE con Validación Cruzada)\n")
cat("=======================================================\n")

set.seed(150)
xgb_cv <- train(
  as.formula(paste(target, "~ .")),
  data = train_split,
  method = "xgbTree",
  tuneGrid = xgb_grid,
  trControl = ctrl_cv,
  metric = "MAE",
  verbosity = 0
)

cat("\nMejores hiperparámetros de XGBoost:\n")
print(xgb_cv$bestTune)

# 5. Evaluación en Validación Local --------------------------------------------
val_pred_xgb <- predict(xgb_cv, newdata = val_split)
mae_xgb <- mean(abs(val_split[[target]] - val_pred_xgb))
rmse_xgb <- sqrt(mean((val_split[[target]] - val_pred_xgb)^2))

cat("\n=======================================================\n")
cat(sprintf(">>> MAE VALIDACIÓN LOCAL XGBOOST: %.5f <<<\n", mae_xgb))
cat(sprintf(">>> RMSE VALIDACIÓN LOCAL:        %.5f <<<\n", rmse_xgb))
cat("=======================================================\n")

# Gráfico Real vs Predicho
p_xgb <- ggplot() +
  geom_line(aes(x = 1:h, y = val_split[[target]], color = "Real"), linewidth = 0.8) +
  geom_line(aes(x = 1:h, y = val_pred_xgb, color = paste0("XGBoost (MAE ", round(mae_xgb, 4), ")")), 
            linetype = "dashed", linewidth = 0.8) +
  labs(
    title = paste("Validación Local XGBoost - MAE:", round(mae_xgb, 5)),
    x = "Paso temporal (15 min)",
    y = "Net_Power",
    color = "Serie"
  ) +
  theme_minimal()
print(p_xgb)

# Importancia de variables en XGBoost
var_imp_xgb <- varImp(xgb_cv)
cat("\nTop 15 variables más importantes en XGBoost:\n")
print(head(var_imp_xgb$importance, 15))

# 6. Reentrenamiento con 100% de datos y Predicción en TEST --------------------
cat("\n--- Reentrenando XGBoost con todo el conjunto TR ---\n")
set.seed(150)
xgb_final <- train(
  as.formula(paste(target, "~ .")),
  data = df_model_tr,
  method = "xgbTree",
  tuneGrid = xgb_cv$bestTune,
  trControl = trainControl(method = "none"),
  metric = "MAE",
  verbosity = 0
)

# Predicción en el conjunto de test
df_model_ts <- df_ts[, features]
pred_test_xgb <- predict(xgb_final, newdata = df_model_ts)

stopifnot(length(pred_test_xgb) == 192)
stopifnot(sum(is.na(pred_test_xgb)) == 0)

cat("\nResumen de predicciones XGBoost:\n")
print(summary(pred_test_xgb))

# 7. Exportación oficial para entrega ------------------------------------------
ruta_entrega_xgb <- file.path(dir_salida, "predicciones_XGBoost.txt")
write.table(
  pred_test_xgb,
  file = ruta_entrega_xgb,
  row.names = FALSE,
  col.names = FALSE
)

write.csv(
  data.frame(Date = df_ts$Date, Net_Power_Pred = pred_test_xgb),
  file = file.path(dir_salida, "predicciones_XGBoost_detalle.csv"),
  row.names = FALSE
)

cat("\n=======================================================\n")
cat("¡FICHERO DE ENTREGA XGBOOST GENERADO CON ÉXITO!\n")
cat("Ruta:", ruta_entrega_xgb, "\n")
cat("Formato: 192 filas, 1 sola columna, sin encabezado.\n")
cat("=======================================================\n")
