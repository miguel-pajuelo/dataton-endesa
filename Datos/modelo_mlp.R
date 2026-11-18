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