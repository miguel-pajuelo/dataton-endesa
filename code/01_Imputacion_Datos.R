################################################################################
# Datathon Endesa 2026 - Cátedra IA Comillas ICAI
# Script: 01_Imputacion_Datos.R
# Objetivo: Análisis e imputación de valores perdidos (NAs) en Datathon_2026_TR
################################################################################

# 1. Comprobación e instalación de librerías necesarias -----------------------
paquetes <- c("readxl", "writexl", "imputeTS", "ggplot2")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) {
  install.packages(nuevos)
}

library(readxl)
library(writexl)
library(imputeTS)
library(ggplot2)

# 2. Localización y lectura de datos ------------------------------------------
# Configurar directorio de trabajo si se ejecuta desde RStudio
tryCatch({
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}, error = function(e) {})

# Buscar la ruta del fichero de entrenamiento (soporta ejecutar desde raíz o desde /code)
posibles_rutas <- c(
  "Datos/Datathon_2026_TR.xlsx",
  "../Datos/Datathon_2026_TR.xlsx",
  "Datathon_2026_TR.xlsx"
)

ruta_tr <- posibles_rutas[file.exists(posibles_rutas)][1]

if (is.na(ruta_tr)) {
  stop("No se ha encontrado el archivo 'Datathon_2026_TR.xlsx'. Comprueba la ruta.")
}

cat("Cargando datos desde:", ruta_tr, "\n")
df_tr <- as.data.frame(read_excel(ruta_tr))

# 3. Diagnóstico de valores perdidos (NAs) ------------------------------------
cat("\n===================================================\n")
cat("               DIAGNÓSTICO DE NAs\n")
cat("===================================================\n")
cat("Dimensiones iniciales:", nrow(df_tr), "filas y", ncol(df_tr), "columnas\n\n")

na_por_columna <- colSums(is.na(df_tr))
print(na_por_columna)

total_na <- sum(na_por_columna)
cat("\nTotal de valores perdidos detectados:", total_na, "\n")

# Identificar columnas afectadas
cols_con_na <- names(na_por_columna[na_por_columna > 0])
cat("Variables con NAs a imputar:", paste(cols_con_na, collapse = ", "), "\n")

# Frecuencia cuarto-horaria (96 observaciones por día)
freq_cuarto_horaria <- 96

# Detalle de cada variable con NAs usando imputeTS
cat("\n--- Resumen estadístico de NAs (imputeTS) ---\n")
for (col in cols_con_na) {
  cat("\n>>> Variable:", col, "\n")
  ts_temp <- ts(df_tr[[col]], frequency = freq_cuarto_horaria)
  print(statsNA(ts_temp))
  
  # Gráfico opcional de distribución de NAs
  # print(ggplot_na_distribution(ts_temp, title = paste("NAs en", col)))
}

# 4. Imputación de valores perdidos -------------------------------------------
cat("\n===================================================\n")
cat("               IMPUTACIÓN DE NAs\n")
cat("===================================================\n")

# Creamos una copia para trabajar
df_tr_imputed <- df_tr

# Método principal recomendado: Interpolación lineal (na_interpolation)
# Al ser datos termodinámicos cada 15 min y huecos aislados de 1-2 pasos,
# la interpolación lineal preserva perfectamente la inercia física sin sesgos.
#
# Métodos alternativos vistos en clase (disponibles en el código):
#  - na_interpolation(ts_var, option = "spline")
#  - na_seasplit(ts_var, algorithm = "interpolation")
#  - na_kalman(ts_var, model = "auto.arima")

metodo_seleccionado <- "interpolation" # Opciones: "interpolation", "spline", "kalman", "seasplit"

for (col in cols_con_na) {
  ts_original <- ts(df_tr[[col]], frequency = freq_cuarto_horaria)
  
  if (metodo_seleccionado == "interpolation") {
    ts_imputada <- na_interpolation(ts_original, option = "linear")
  } else if (metodo_seleccionado == "spline") {
    ts_imputada <- na_interpolation(ts_original, option = "spline")
  } else if (metodo_seleccionado == "kalman") {
    ts_imputada <- na_kalman(ts_original, model = "auto.arima")
  } else if (metodo_seleccionado == "seasplit") {
    ts_imputada <- na_seasplit(ts_original, algorithm = "interpolation")
  }
  
  # Guardamos la columna imputada
  df_tr_imputed[[col]] <- as.numeric(ts_imputada)
  
  # Visualización de la imputación realizada
  # ggplot_na_imputations(ts_original, ts_imputada, title = paste("Imputación:", col))
}

# 5. Verificación final -------------------------------------------------------
cat("\n===================================================\n")
cat("               VERIFICACIÓN FINAL\n")
cat("===================================================\n")
na_restantes <- sum(is.na(df_tr_imputed))
cat("NAs restantes tras imputación:", na_restantes, "\n")

if (na_restantes == 0) {
  cat("¡ÉXITO! Todas las variables han sido imputadas correctamente.\n")
} else {
  warning("Aún quedan valores perdidos. Revisa las columnas.")
}

# 6. Exportar datos limpios ---------------------------------------------------
# Definir directorio de salida (misma carpeta donde está el fichero original)
dir_salida <- dirname(ruta_tr)
ruta_salida_xlsx <- file.path(dir_salida, "Datathon_2026_TR_imputed.xlsx")
ruta_salida_csv  <- file.path(dir_salida, "Datathon_2026_TR_imputed.csv")

# Guardar en formato Excel y CSV
write_xlsx(df_tr_imputed, ruta_salida_xlsx)
write.csv(df_tr_imputed, ruta_salida_csv, row.names = FALSE)

cat("\nFicheros guardados con éxito:")
cat("\n - Excel:", ruta_salida_xlsx)
cat("\n - CSV:  ", ruta_salida_csv, "\n")
