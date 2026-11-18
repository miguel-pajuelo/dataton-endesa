################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 03_Modelo_MLP.R (Versión Mejorada con Diagnóstico y Poda de Variables)
#
# Novedades:
#   1. Gráfico visual de la arquitectura del Perceptrón (plotnet).
#   2. Análisis riguroso de importancia de variables (varImp y sensibilidad).
#   3. Poda automática de variables redundantes/ruidosas para eliminar el overfitting.
#   4. Ajuste fino de regularización (decay) y tamaño de capa oculta (size).
#   5. Comparativa de MAE local: Modelo Completo vs. Modelo Podado.
#   6. Generación del fichero de entrega optimizado: predicciones_MLP_podado.txt.
################################################################################

# 1. Carga de paquetes ---------------------------------------------------------
paquetes <- c("readxl", "writexl", "caret", "nnet", "ggplot2", "Hmisc", "NeuralNetTools")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) install.packages(nuevos)

library(readxl)
library(writexl)
library(caret)
library(nnet)
library(ggplot2)
library(Hmisc)

# Intentar cargar paquetes de visualización de redes de ICAI
if ("NeuralNetTools" %in% installed.packages()[, "Package"]) {
  library(NeuralNetTools)
}
if ("NeuralSens" %in% installed.packages()[, "Package"]) {
  library(NeuralSens)
}

# 2. Localización y lectura de datos enriquecidos ------------------------------
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

# 3. Conjunto inicial de variables candidatas ----------------------------------
todas_features <- c(
  "Amb_Temp", "H2O_Cond_Inlet_Temp", "H2O_Cond_Outlet_Temp", 
  "Cool_Water_Flow", "Cond_Press", "IGV_Pos", "Sea_Level",
  "Delta_T_Cond", "Heat_Rejection", "Air_Density_Proxy", 
  "GasTurbine_Charge", "Cond_Pressure_Flow_Ratio", "Log_Cond_Press",
  "Sin_Daily", "Cos_Daily", "Sin_SemiDaily", "Cos_SemiDaily", "Is_Weekend",
  "Diff_Amb_Temp", "Diff_Cond_Press", "Diff_IGV_Pos", "Diff_H2O_Inlet",
  "Amb_Temp_lag1", "Amb_Temp_lag4",
  "H2O_Cond_Inlet_Temp_lag1", "H2O_Cond_Inlet_Temp_lag4",
  "Cond_Press_lag1", "Cond_Press_lag4",
  "IGV_Pos_lag1", "Delta_T_Cond_lag1"
)

features_disponibles <- intersect(todas_features, names(df_tr))
df_model_tr <- na.omit(df_tr[, c(target, features_disponibles)])

# Partición de validación fija (los últimos 2 días de TR = 192 cuartos de hora)
h <- 192
n_total <- nrow(df_model_tr)
idx_train <- 1:(n_total - h)
idx_val   <- (n_total - h + 1):n_total

train_split <- df_model_tr[idx_train, ]
val_split   <- df_model_tr[idx_val, ]

# 4. FASE 1: Modelo Completo para Diagnóstico de Importancia -------------------
cat("\n=======================================================\n")
cat(" FASE 1: ENTRENAMIENTO MODELO BASE PARA ANÁLISIS DE IMPORTANCIA\n")
cat("=======================================================\n")

ctrl_cv <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = defaultSummary,
  savePredictions = "final"
)

# Rejilla con mayor regularización para mitigar overfitting
mlp_grid_base <- expand.grid(
  size  = c(5, 8),
  decay = c(0.01, 0.05, 0.1)
)

set.seed(150)
mlp_base <- train(
  as.formula(paste(target, "~ .")),
  data = train_split,
  method = "nnet",
  linout = TRUE,
  maxit = 250,
  preProcess = c("center", "scale"),
  tuneGrid = mlp_grid_base,
  trControl = ctrl_cv,
  metric = "MAE",
  trace = FALSE
)

# Predicción base y MAE en validación
val_pred_base <- predict(mlp_base, newdata = val_split)
mae_base <- mean(abs(val_split[[target]] - val_pred_base))
cat(sprintf(">>> MAE Validación (Modelo Base con %d variables): %.5f <<<\n", 
            length(features_disponibles), mae_base))

# Gráfico de la red neuronal (plotnet)
if ("NeuralNetTools" %in% installed.packages()[, "Package"]) {
  cat("\nGenerando gráfico de la arquitectura de la red (plotnet)...\n")
  try({
    plotnet(mlp_base$finalModel, 
            y_names = "Net_Power", 
            pad_x = 0.6, 
            circle_cex = 3, 
            cex_val = 0.6)
  })
}

# 5. Análisis de Importancia de Variables y Selección (Poda) -------------------
cat("\n=======================================================\n")
cat(" FASE 2: ANÁLISIS DE IMPORTANCIA Y PODA DE VARIABLES\n")
cat("=======================================================\n")

# Calcular importancia relativa según el algoritmo de Garson / Olden
var_imp <- varImp(mlp_base)
df_imp <- data.frame(
  Variable = rownames(var_imp$importance),
  Importancia = var_imp$importance$Overall
)
df_imp <- df_imp[order(df_imp$Importancia, decreasing = TRUE), ]

cat("\nRanking de importancia de variables:\n")
print(head(df_imp, 15))

# Gráfico de barras de importancia
p_imp <- ggplot(df_imp, aes(x = reorder(Variable, Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Importancia de Variables en el Perceptrón Multicapa (MLP)",
    x = "Variable",
    y = "Importancia Relativa (%)"
  ) +
  theme_minimal()
print(p_imp)

# CRITERIO DE PODA:
# Las redes neuronales sufren si hay demasiadas variables correlacionadas.
# Seleccionamos las TOP variables más determinantes (entre 8 y 12 variables clave).
# Nos aseguramos de mantener las variables clave del ciclo combinado:
# (IGV_Pos, GasTurbine_Charge, Cond_Press, Delta_T_Cond, componentes cíclicas y lags críticos)

top_k <- 10  # Número óptimo de variables para evitar overfitting
variables_podadas <- df_imp$Variable[1:top_k]

# Aseguramos que las variables de física esenciales estén si están en el top 18
vars_esenciales <- c("IGV_Pos", "Cond_Press", "GasTurbine_Charge", "Delta_T_Cond", "Sin_Daily", "Cos_Daily")
variables_podadas <- unique(c(variables_podadas, intersect(vars_esenciales, df_imp$Variable[1:18])))

cat("\nVariables seleccionadas tras la poda (", length(variables_podadas), "):\n")
cat(paste(variables_podadas, collapse = " + "), "\n")

# 6. FASE 3: Entrenamiento del Perceptrón Podado (Pruned MLP) -------------------
cat("\n=======================================================\n")
cat(" FASE 3: ENTRENAMIENTO DEL PERCEPTRÓN PODADO\n")
cat("=======================================================\n")

formula_podada <- as.formula(paste(target, "~", paste(variables_podadas, collapse = " + ")))

# Grid de ajuste para el modelo podado (probamos regularizaciones más fuertes)
mlp_grid_pruned <- expand.grid(
  size  = c(3, 5, 8),
  decay = c(0.01, 0.05, 0.1, 0.2)
)

set.seed(150)
mlp_pruned <- train(
  formula_podada,
  data = train_split,
  method = "nnet",
  linout = TRUE,
  maxit = 350,
  preProcess = c("center", "scale"),
  tuneGrid = mlp_grid_pruned,
  trControl = ctrl_cv,
  metric = "MAE",
  trace = FALSE
)

cat("\nMejores hiperparámetros para el modelo podado:\n")
print(mlp_pruned$bestTune)

# Evaluación en validación local
val_pred_pruned <- predict(mlp_pruned, newdata = val_split)
mae_pruned <- mean(abs(val_split[[target]] - val_pred_pruned))

cat("\n=======================================================\n")
cat(sprintf(">>> MAE MODELO BASE (Sin poda):   %.5f <<<\n", mae_base))
cat(sprintf(">>> MAE MODELO PODADO (Con poda): %.5f <<<\n", mae_pruned))
mejora_pct <- ((mae_base - mae_pruned) / mae_base) * 100
cat(sprintf(">>> Variación de error:           %+.2f%% <<<\n", mejora_pct))
cat("=======================================================\n")

# Gráfico comparativo Real vs. Base vs. Podado
p_comp <- ggplot() +
  geom_line(aes(x = 1:h, y = val_split[[target]], color = "Real"), linewidth = 0.8) +
  geom_line(aes(x = 1:h, y = val_pred_base, color = paste0("Base (MAE ", round(mae_base, 4), ")")), linetype = "dotted") +
  geom_line(aes(x = 1:h, y = val_pred_pruned, color = paste0("Podado (MAE ", round(mae_pruned, 4), ")")), linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "Comparación en Validación Local (192 pasos - 28 y 29 de Julio)",
    x = "Paso temporal (cada 15 min)",
    y = "Net_Power",
    color = "Modelo"
  ) +
  theme_minimal()
print(p_comp)

# Gráfico de arquitectura del modelo podado
if ("NeuralNetTools" %in% installed.packages()[, "Package"]) {
  cat("\nArquitectura del modelo podado:\n")
  try({
    plotnet(mlp_pruned$finalModel, 
            y_names = "Net_Power", 
            pad_x = 0.6, 
            circle_cex = 4, 
            cex_val = 0.7)
  })
}

# 7. Reentrenamiento final con 100% de datos y Predicción en TEST ---------------
cat("\n=======================================================\n")
cat(" FASE 4: REENTRENAMIENTO DEFINITIVO Y PREDICCIÓN TEST\n")
cat("=======================================================\n")

# Usamos el mejor modelo entre base y podado
modelo_ganador <- if (mae_pruned <= mae_base) mlp_pruned else mlp_base
formula_ganadora <- if (mae_pruned <= mae_base) formula_podada else as.formula(paste(target, "~ ."))
cat("Modelo seleccionado para la entrega:", if (mae_pruned <= mae_base) "MODELO PODADO" else "MODELO BASE", "\n")

set.seed(150)
mlp_definitivo <- train(
  formula_ganadora,
  data = df_model_tr,
  method = "nnet",
  linout = TRUE,
  maxit = 400,
  preProcess = c("center", "scale"),
  tuneGrid = modelo_ganador$bestTune,
  trControl = trainControl(method = "none"),
  trace = FALSE
)

# Generar predicciones sobre TS
pred_test <- predict(mlp_definitivo, newdata = df_ts)

# Verificaciones obligatorias
stopifnot(length(pred_test) == 192)
stopifnot(sum(is.na(pred_test)) == 0)

cat("\nResumen estadístico de las 192 predicciones de test:\n")
print(summary(pred_test))

# 8. Exportación oficial para subir a la plataforma ----------------------------
ruta_entrega_podada <- file.path(dir_salida, "predicciones_MLP_podado.txt")
write.table(
  pred_test,
  file = ruta_entrega_podada,
  row.names = FALSE,
  col.names = FALSE
)

write.csv(
  data.frame(Date = df_ts$Date, Net_Power_Pred = pred_test),
  file = file.path(dir_salida, "predicciones_MLP_podado_detalle.csv"),
  row.names = FALSE
)

cat("\n=======================================================\n")
cat("¡FICHERO DE ENTREGA GENERADO CON ÉXITO!\n")
cat("Ruta:", ruta_entrega_podada, "\n")
cat("Formato: 192 filas, 1 sola columna, sin cabecera.\n")
cat("=======================================================\n")
