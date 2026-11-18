################################################################################
# DATATHON ENDESA 2026 - PREDICCIÓN DE GENERACIÓN ELÉCTRICA
# Script: 02_Feature_Engineering.R
# Objetivo:
#   1. Imputar valores perdidos (utilizando Kalman de script_dataton.R).
#   2. Generar nuevas variables (físicas/termodinámicas, temporales, dinámicas y lags).
#   3. Generar TR_features y TS_features listos para XGBoost, MLP, GBM o ARIMAX.
################################################################################

# 1. Comprobación de paquetes --------------------------------------------------
paquetes <- c("readxl", "writexl", "imputeTS", "Hmisc", "ggplot2", "dplyr", "lubridate")
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) {
  install.packages(nuevos)
}

library(readxl)
library(writexl)
library(imputeTS)
library(Hmisc)

# 2. Localización de ficheros --------------------------------------------------
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

ruta_tr <- buscar_fichero("Datathon_2026_TR.xlsx")
ruta_ts <- buscar_fichero("Datathon_2026_TS.xlsx")
dir_datos <- dirname(ruta_tr)

cat("Cargando datos:\n - Train:", ruta_tr, "\n - Test: ", ruta_ts, "\n")
df_tr_raw <- as.data.frame(read_excel(ruta_tr))
df_ts_raw <- as.data.frame(read_excel(ruta_ts))

# 3. Imputación de valores perdidos en TR (método Kalman de script_dataton.R) --
cat("\n--- Imputando NAs en TR con Filtro de Kalman ---\n")
cols_numericas_tr <- setdiff(names(df_tr_raw), "Date")
ts_tr <- ts(df_tr_raw[, cols_numericas_tr], frequency = 96) # 96 cuartos de hora/día
ts_tr_imp <- na_kalman(ts_tr, model = "auto.arima")

df_tr_clean <- data.frame(Date = df_tr_raw$Date, as.data.frame(ts_tr_imp))
cat("NAs restantes en TR tras imputación:", sum(is.na(df_tr_clean)), "\n")

# Comprobar NAs en TS
cat("NAs en TS:", sum(is.na(df_ts_raw)), "\n")
df_ts_clean <- df_ts_raw

# 4. Función de Feature Engineering (Física + Calendario + Lags) --------------
# Concatenamos temporalmente TR y TS para que los retardos (lags) y diferencias
# al inicio de TS utilicen los datos reales de los últimos días de TR.
df_ts_clean$Net_Power <- NA # En TS no conocemos el target

columnas_comunes <- c("Date", "Net_Power", "Amb_Temp", "H2O_Cond_Inlet_Temp", 
                      "H2O_Cond_Outlet_Temp", "Cool_Water_Flow", "Cond_Press", 
                      "IGV_Pos", "Sea_Level")

n_tr <- nrow(df_tr_clean)
n_ts <- nrow(df_ts_clean)

df_all <- rbind(df_tr_clean[, columnas_comunes], df_ts_clean[, columnas_comunes])

cat("\n--- Generando nuevas variables (Feature Engineering) ---\n")

# A. VARIABLES FÍSICAS Y TERMODINÁMICAS (Ciclo Combinado) ---------------------
# 1. Salto térmico del agua en el condensador (Delta T)
df_all$Delta_T_Cond <- df_all$H2O_Cond_Outlet_Temp - df_all$H2O_Cond_Inlet_Temp

# 2. Calor evacuado en el condensador: Q ~ Caudal * Delta T
df_all$Heat_Rejection <- df_all$Cool_Water_Flow * df_all$Delta_T_Cond

# 3. Efecto de la densidad del aire en la turbina de gas
#    La densidad del aire es inversamente proporcional a la temperatura absoluta (K)
df_all$Air_Density_Proxy <- 1 / (df_all$Amb_Temp + 273.15)

# 4. Potencia estimada de admisión de la turbina de gas: Apertura IGV * Densidad de aire
df_all$GasTurbine_Charge <- df_all$IGV_Pos * df_all$Air_Density_Proxy

# 5. Contrapresión del condensador normalizada por el caudal de refrigeración
df_all$Cond_Pressure_Flow_Ratio <- df_all$Cond_Press / (df_all$Cool_Water_Flow + 0.1)

# 6. Comportamiento logarítmico de la presión de vacío
df_all$Log_Cond_Press <- log(pmax(df_all$Cond_Press, 1e-4))

# 7. Eficacia aproximada del enfriamiento (temperatura de salida vs temperatura ambiente)
df_all$Approach_Temp <- df_all$H2O_Cond_Outlet_Temp - df_all$Amb_Temp


# B. VARIABLES TEMPORALES Y ESTACIONALES (Ciclo diario y semanal) -------------
# Convertir Date a POSIXct si no lo está
fechas <- as.POSIXct(df_all$Date)

df_all$Hour <- as.numeric(format(fechas, "%H"))
df_all$Minute <- as.numeric(format(fechas, "%M"))

# Periodo cuarto-horario dentro del día: de 1 a 96
df_all$Period_96 <- (df_all$Hour * 4) + (df_all$Minute / 15) + 1

# Componente armónica cíclica diaria (24 horas)
df_all$Sin_Daily <- sin(2 * pi * df_all$Period_96 / 96)
df_all$Cos_Daily <- cos(2 * pi * df_all$Period_96 / 96)

# Segundo armónico cíclico (12 horas - picos de demanda mediodía y noche)
df_all$Sin_SemiDaily <- sin(4 * pi * df_all$Period_96 / 96)
df_all$Cos_SemiDaily <- cos(4 * pi * df_all$Period_96 / 96)

# Día de la semana (1 = Domingo, 2 = Lunes, ..., 7 = Sábado en R)
df_all$DayOfWeek <- as.numeric(format(fechas, "%u")) # 1 = Lunes a 7 = Domingo
df_all$Is_Weekend <- ifelse(df_all$DayOfWeek %in% c(6, 7), 1, 0)


# C. DINÁMICAS TEMPORALES / VELOCIDAD DE CAMBIO (Diferencias 15 min) ----------
# ¿La variable está subiendo o bajando respecto al cuarto de hora anterior?
df_all$Diff_Amb_Temp     <- df_all$Amb_Temp - Lag(df_all$Amb_Temp, 1)
df_all$Diff_Cond_Press   <- df_all$Cond_Press - Lag(df_all$Cond_Press, 1)
df_all$Diff_IGV_Pos      <- df_all$IGV_Pos - Lag(df_all$IGV_Pos, 1)
df_all$Diff_H2O_Inlet    <- df_all$H2O_Cond_Inlet_Temp - Lag(df_all$H2O_Cond_Inlet_Temp, 1)


# D. RETARDOS (LAGS) DE VARIABLES EXÓGENAS ------------------------------------
# 1 paso = 15 min; 4 pasos = 1 hora; 96 pasos = 24 horas (mismo instante de ayer)
df_all$Amb_Temp_lag1 <- Lag(df_all$Amb_Temp, 1)
df_all$Amb_Temp_lag4 <- Lag(df_all$Amb_Temp, 4)
df_all$Amb_Temp_lag96 <- Lag(df_all$Amb_Temp, 96)

df_all$H2O_Cond_Inlet_Temp_lag1 <- Lag(df_all$H2O_Cond_Inlet_Temp, 1)
df_all$H2O_Cond_Inlet_Temp_lag4 <- Lag(df_all$H2O_Cond_Inlet_Temp, 4)
df_all$H2O_Cond_Inlet_Temp_lag96 <- Lag(df_all$H2O_Cond_Inlet_Temp, 96)

df_all$Cond_Press_lag1 <- Lag(df_all$Cond_Press, 1)
df_all$Cond_Press_lag4 <- Lag(df_all$Cond_Press, 4)
df_all$Cond_Press_lag96 <- Lag(df_all$Cond_Press, 96)

df_all$IGV_Pos_lag1 <- Lag(df_all$IGV_Pos, 1)
df_all$IGV_Pos_lag4 <- Lag(df_all$IGV_Pos, 4)

df_all$Delta_T_Cond_lag1 <- Lag(df_all$Delta_T_Cond, 1)
df_all$Delta_T_Cond_lag4 <- Lag(df_all$Delta_T_Cond, 4)


# E. RETARDOS DE LA VARIABLE OBJETIVO (Net_Power) ------------------------------
# Lags de corto plazo (15 min, 30 min, 1 hora):
#  * En TR están completamente disponibles.
#  * En TS para predicción recursiva autorregresiva (sustituyendo valor predicho).
df_all$Net_Power_lag1 <- Lag(df_all$Net_Power, 1)
df_all$Net_Power_lag2 <- Lag(df_all$Net_Power, 2)
df_all$Net_Power_lag4 <- Lag(df_all$Net_Power, 4)

# Lags de largo plazo (24 horas = 96 periodos; 48 horas = 192 periodos):
#  * Net_Power_lag96 en TS (día 30) se alimenta del día 29 de TR.
#  * Net_Power_lag192 en TS (días 30 y 31) se alimenta de los días 28 y 29 de TR.
df_all$Net_Power_lag96  <- Lag(df_all$Net_Power, 96)
df_all$Net_Power_lag192 <- Lag(df_all$Net_Power, 192)


# 5. Volver a separar TR y TS --------------------------------------------------
df_tr_features <- df_all[1:n_tr, ]
df_ts_features <- df_all[(n_tr + 1):(n_tr + n_ts), ]

cat("Total de columnas tras Feature Engineering:", ncol(df_tr_features), "\n")
cat("Nuevas variables creadas:\n")
nuevas_vars <- setdiff(names(df_tr_features), columnas_comunes)
print(nuevas_vars)

# 6. Exportar conjuntos enriquecidos -------------------------------------------
ruta_salida_tr_xlsx <- file.path(dir_datos, "Datathon_2026_TR_features.xlsx")
ruta_salida_tr_csv  <- file.path(dir_datos, "Datathon_2026_TR_features.csv")
ruta_salida_ts_xlsx <- file.path(dir_datos, "Datathon_2026_TS_features.xlsx")
ruta_salida_ts_csv  <- file.path(dir_datos, "Datathon_2026_TS_features.csv")

write_xlsx(df_tr_features, ruta_salida_tr_xlsx)
write.csv(df_tr_features, ruta_salida_tr_csv, row.names = FALSE)

write_xlsx(df_ts_features, ruta_salida_ts_xlsx)
write.csv(df_ts_features, ruta_salida_ts_csv, row.names = FALSE)

cat("\n¡Completado con éxito!")
cat("\nFicheros de salida listos para los modelos:")
cat("\n - Train enriquecido: ", ruta_salida_tr_xlsx)
cat("\n - Test enriquecido:  ", ruta_salida_ts_xlsx, "\n")
