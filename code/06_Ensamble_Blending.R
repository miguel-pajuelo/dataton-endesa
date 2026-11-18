################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 06_Ensamble_Blending.R
# Objetivo:
#   Combinar (hacer blending / ensamble) de las predicciones de los distintos modelos
#   (XGBoost, MLP Podado y GBM) para reducir la varianza y minimizar el MAE final.
################################################################################

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
  if (is.na(encontrada)) return(NA)
  return(encontrada)
}

dir_salida <- ifelse(file.exists("Datos"), "Datos", ".")

# 1. Cargar predicciones individuales de Test ----------------------------------
ruta_mlp <- buscar_fichero("predicciones_MLP_podado.txt")
ruta_xgb <- buscar_fichero("predicciones_XGBoost.txt")
ruta_gbm <- buscar_fichero("predicciones_GBM.txt")

modelos_cargados <- list()

if (!is.na(ruta_mlp)) {
  vals_mlp <- read.table(ruta_mlp, header = FALSE)[[1]]
  if (length(vals_mlp) == 192) {
    modelos_cargados[["MLP"]] <- vals_mlp
    cat("Cargado MLP:    ", ruta_mlp, "(192 filas)\n")
  } else {
    warning(paste("¡ALERTA! El archivo MLP tiene", length(vals_mlp), "filas (debe tener 192). Se ignora."))
  }
}

if (!is.na(ruta_xgb)) {
  vals_xgb <- read.table(ruta_xgb, header = FALSE)[[1]]
  if (length(vals_xgb) == 192) {
    modelos_cargados[["XGBoost"]] <- vals_xgb
    cat("Cargado XGBoost:", ruta_xgb, "(192 filas)\n")
  } else {
    warning(paste("¡ALERTA! El archivo XGBoost tiene", length(vals_xgb), "filas (debe tener 192). Se ignora."))
  }
}

if (!is.na(ruta_gbm)) {
  vals_gbm <- read.table(ruta_gbm, header = FALSE)[[1]]
  if (length(vals_gbm) == 192) {
    modelos_cargados[["GBM"]] <- vals_gbm
    cat("Cargado GBM:    ", ruta_gbm, "(192 filas)\n")
  } else {
    warning(paste("¡ALERTA! El archivo GBM tiene", length(vals_gbm), "filas (debe tener 192). Se ignora."))
  }
}

if (length(modelos_cargados) < 2) {
  stop("Se necesitan al menos 2 modelos válidos de EXACTAMENTE 192 filas cada uno.")
}

# 2. Ponderaciones del Ensamble (Blending) --------------------------------------
# En regresión tabular, XGBoost suele ser el modelo con menor error, complementado
# por el MLP (que aporta una visión no lineal continua suave).

n <- length(modelos_cargados[["MLP"]])
df_preds <- as.data.frame(modelos_cargados)

cat("\nCorrelación entre predicciones de los modelos:\n")
print(cor(df_preds))

# Esquema de pesos sugerido:
# Si están XGBoost y MLP: 60% XGBoost + 40% MLP
# Si están los 3: 50% XGBoost + 30% MLP + 20% GBM
if ("XGBoost" %in% names(modelos_cargados) && "MLP" %in% names(modelos_cargados) && "GBM" %in% names(modelos_cargados)) {
  pred_ensamble <- 0.50 * df_preds$XGBoost + 0.30 * df_preds$MLP + 0.20 * df_preds$GBM
  cat("\nPonderación aplicada: 50% XGBoost + 30% MLP + 20% GBM\n")
} else if ("XGBoost" %in% names(modelos_cargados) && "MLP" %in% names(modelos_cargados)) {
  pred_ensamble <- 0.60 * df_preds$XGBoost + 0.40 * df_preds$MLP
  cat("\nPonderación aplicada: 60% XGBoost + 40% MLP\n")
} else {
  # Media simple
  pred_ensamble <- rowMeans(df_preds)
  cat("\nPonderación aplicada: Media simple de modelos disponibles\n")
}

# Verificaciones obligatorias
stopifnot(length(pred_ensamble) == 192)
stopifnot(sum(is.na(pred_ensamble)) == 0)

# 3. Exportación oficial para entrega ------------------------------------------
ruta_entrega_ensamble <- file.path(dir_salida, "predicciones_ENSAMBLE.txt")
write.table(
  pred_ensamble,
  file = ruta_entrega_ensamble,
  row.names = FALSE,
  col.names = FALSE
)

cat("\n=======================================================\n")
cat("¡FICHERO DE ENSAMBLE GENERADO CON ÉXITO!\n")
cat("Ruta:", ruta_entrega_ensamble, "\n")
cat("Formato: 192 filas, 1 sola columna, sin encabezado.\n")
cat("=======================================================\n")
