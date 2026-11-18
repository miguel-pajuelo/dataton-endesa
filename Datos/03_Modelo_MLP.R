################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 03_Modelo_MLP.R
# Objetivo:
#   Ajustar un Perceptrón Multicapa (MLP / Red Neuronal nnet) sobre los datos
#   enriquecidos para predecir Net_Power en el conjunto de test (192 observaciones).
################################################################################

# 1. Carga de paquetes ---------------------------------------------------------
paquetes <- c("readxl", "writexl", "caret", "nnet", "ggplot2", "Hmisc")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) install.packages(nuevos)

library(readxl)
library(writexl)
library(caret)
library(nnet)
library(ggplot2)
library(Hmisc)

# Librerías opcionales de diagnóstico del ICAI
for (pkg in c("NeuralNetTools", "NeuralSens", "MLTools")) {
  if (pkg %in% installed.packages()[, "Package"]) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

# 2. Configurar directorio y cargar datos enriquecidos --------------------------
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

# 3. Selección de variables para el Perceptrón ----------------------------------
# Para una red neuronal, es crucial seleccionar variables informativas pero no
# redundantes para evitar sobreajuste y problemas de convergencia.

target <- "Net_Power"

features <- c(
  # Variables físicas originales
  "Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp", 
  "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level",
  
  # Variables termodinámicas derivadas
  "Delta_T_Cond", "Heat_Rejection", "Air_Density_Proxy", 
  "GasTurbine_Charge", "Cond_Pressure_Flow_Ratio", "Log_Cond_Press",
  
  # Variables temporales cíclicas (muy importantes para MLP)
  "Sin_Daily", "Cos_Daily", "Sin_SemiDaily", "Cos_SemiDaily", "Is_Weekend",
  
  # Dinámicas / tendencias (diferencias de 15 min)
  "Diff_Amb_Temp", "Diff_Cond_Press", "Diff_IGV_Pos", "Diff_H2O_Inlet",
  
  # Retardos de exógenas (inercia térmica)
  "Amb_Temp_lag1", "Amb_Temp_lag4",
  "H2O_Cond_Inlet_Temp_lag1", "H2O_Cond_Inlet_Temp_lag4",
  "Cond_Press_lag1", "Cond_Press_lag4",
  "IGV_Pos_lag1", "Delta_T_Cond_lag1"
)

# Filtramos columnas que existen realmente en el dataframe
features <- intersect(features, names(df_tr))
cat("\nNúmero de variables predictoras seleccionadas:", length(features), "\n")

# Preparar conjunto de datos para modelado
df_model_tr <- df_tr[, c(target, features)]

# Eliminamos las filas iniciales con NA por cálculo de lags (solo las primeras 4 filas)
df_model_tr <- na.omit(df_model_tr)
cat("Filas de entrenamiento disponibles tras na.omit:", nrow(df_model_tr), "\n")

# 4. Validación local: Hold-out de 192 pasos (2 últimos días de TR) -------------
# Evaluamos el modelo en las últimas 192 observaciones (equivalente al horizonte de test)
h <- 192
n_total <- nrow(df_model_tr)

idx_train <- 1:(n_total - h)
idx_val   <- (n_total - h + 1):n_total

train_split <- df_model_tr[idx_train, ]
val_split   <- df_model_tr[idx_val, ]

# 5. Configuración de hiperparámetros y entrenamiento con caret -----------------
ctrl_cv <- trainControl(
  method = "cv",
  number = 5,                         # 5-fold CV para rapidez (quedan < 1.5 horas)
  summaryFunction = defaultSummary,
  savePredictions = "final"
)

# Rejilla de hiperparámetros para nnet:
#  - size: número de neuronas ocultas
#  - decay: penalización L2 (weight decay) para regularizar
mlp_grid <- expand.grid(
  size  = c(5, 8, 12),
  decay = c(0.001, 0.01, 0.1)
)

set.seed(150)
cat("\n--- Entrenando Perceptrón Multicapa (MLP) sobre partición de validación ---\n")

mlp_cv <- train(
  as.formula(paste(target, "~ .")),
  data = train_split,
  method = "nnet",
  linout = TRUE,                       # Salida lineal continua para regresión
  maxit = 300,                         # Iteraciones máximas
  preProcess = c("center", "scale"),   # ¡CRUCIAL para redes neuronales!
  tuneGrid = mlp_grid,
  trControl = ctrl_cv,
  metric = "MAE",                      # Métrica oficial del Datathon
  trace = FALSE
)

cat("\nMejores hiperparámetros encontrados en CV:\n")
print(mlp_cv$bestTune)

# 6. Evaluación en validación local (MAE) ---------------------------------------
val_pred <- predict(mlp_cv, newdata = val_split)
mae_local <- mean(abs(val_split[[target]] - val_pred))
rmse_local <- sqrt(mean((val_split[[target]] - val_pred)^2))

cat("\n===================================================\n")
cat(sprintf(">>> MAE LOCAL EN VALIDACIÓN (192 pasos): %.4f <<<\n", mae_local))
cat(sprintf(">>> RMSE LOCAL EN VALIDACIÓN:            %.4f <<<\n", rmse_local))
cat("===================================================\n")

# Gráfico de validación: Real vs Predicho
p_val <- ggplot() +
  geom_line(aes(x = 1:h, y = val_split[[target]], color = "Real")) +
  geom_line(aes(x = 1:h, y = val_pred, color = "Predicción MLP"), linetype = "dashed") +
  labs(
    title = paste("Validación Local MLP (Últimos 2 días de TR) - MAE:", round(mae_local, 4)),
    x = "Paso temporal (15 min)",
    y = "Net_Power",
    color = "Serie"
  ) +
  theme_minimal()
print(p_val)

# 7. Reentrenamiento con todo el conjunto TR (100% de datos) ------------------
cat("\n--- Reentrenando MLP con todo el dataset de entrenamiento ---\n")
set.seed(150)

mlp_final <- train(
  as.formula(paste(target, "~ .")),
  data = df_model_tr,
  method = "nnet",
  linout = TRUE,
  maxit = 350,
  preProcess = c("center", "scale"),
  tuneGrid = mlp_cv$bestTune,          # Usamos los mejores parámetros ya validados
  trControl = trainControl(method = "none"), # Entrenamiento directo rápido
  trace = FALSE
)

# 8. Generación de predicciones para TEST (TS) ---------------------------------
cat("\n--- Generando predicciones sobre Datathon_2026_TS (192 filas) ---\n")

# Nos aseguramos de que el conjunto de test tenga las variables necesarias
df_model_ts <- df_ts[, features]

# Predicción directa con el MLP final
predicciones_ts <- predict(mlp_final, newdata = df_model_ts)

cat("Total de predicciones generadas:", length(predicciones_ts), "\n")
cat("Resumen de predicciones de Net_Power:\n")
print(summary(predicciones_ts))

# Comprobaciones de seguridad obligatorias del reto:
stopifnot(length(predicciones_ts) == 192)
stopifnot(sum(is.na(predicciones_ts)) == 0)

# 9. Exportación en el formato oficial de la competición -----------------------
# Requisito del PDF: Fichero de texto plano, 1 única columna, 192 filas, sin cabecera.

ruta_entrega_txt <- file.path(dir_salida, "predicciones_MLP_v1.txt")
write.table(
  predicciones_ts,
  file = ruta_entrega_txt,
  row.names = FALSE,
  col.names = FALSE
)

# También guardamos una copia con tabla comparativa en CSV
df_resultado <- data.frame(
  Date = df_ts$Date,
  Net_Power_Pred = predicciones_ts
)
write.csv(df_resultado, file.path(dir_salida, "predicciones_MLP_v1_detalle.csv"), row.names = FALSE)

cat("\n===================================================\n")
cat("¡FICHERO DE ENTREGA GENERADO CON ÉXITO!\n")
cat("Ruta:", ruta_entrega_txt, "\n")
cat("Formato verificado: 1 columna, exactamente 192 filas, sin encabezado.\n")
cat("Listo para subir a la carpeta de OneDrive / Moodle.\n")
cat("===================================================\n")
