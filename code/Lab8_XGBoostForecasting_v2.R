#################################################################################
##############      Lab 8: Nonlinear forecasting    ############################
##############      XGBoost for nonlinear forecasting    ########################
##############     ----------- solution ---------    ############################
#################################################################################

# PARA WINDOWS
#install.packages(
#  "xgboost",
#  type = "win.binary",
#  contriburl = "https://packagemanager.posit.co/cran/2025-10-01/bin/windows/contrib/4.5"
#)

# PARA MAC
# if ("package:xgboost" %in% search()) detach("package:xgboost", unload = TRUE)
# remove.packages("xgboost")
# remotes::install_version("xgboost", version = "1.7.8.1", repos = "https://cloud.r-project.org")


library(MLTools)
library(fpp2)
library(lmtest)
library(tseries) #contains adf.test function
library(TSA)
library(caret)
library(kernlab)
library(Hmisc)
library(xgboost)
library(ggplot2)



## Set working directory ---------------------------------------------------------------------------------------------
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read data -------------------------------------------------------------------------------------------------
fdata <- readxl::read_excel("DAILY_DEMAND_TR.xlsx")
fdata <- as.data.frame(fdata)

# Initialize output and input variables
fdata.Reg <- fdata[, c("DEM", "WD", "TEMP")] 

# Include lagged variables
fdata.Reg$WD_lag1 <- Lag(fdata$WD, 1)
fdata.Reg$WD_lag7 <- Lag(fdata$WD, 7)
fdata.Reg$TEMP_lag1 <- Lag(fdata$TEMP, 1)
fdata.Reg$TEMP_lag7 <- Lag(fdata$TEMP, 7)
fdata.Reg$DEM_lag1 <- Lag(fdata$DEM, 1)
fdata.Reg$DEM_lag7 <- Lag(fdata$DEM, 7)

# Remove missing values
fdata.Reg.tr <- na.omit(fdata.Reg)

## Initialize trainControl -----------------------------------------------------------------------
ctrl_tune <- trainControl(
    method = "cv",
    number = 10,
    summaryFunction = defaultSummary,
    savePredictions = TRUE
)

#------------ XGBoost model --------------------------------
set.seed(150) # For replication

# Define tuning grid
xgbGrid <- expand.grid(
    nrounds = c(100, 200, 600),          # Number of boosting iterations (trees)
    max_depth = c(3, 6),            # Maximum depth of a tree (controls model complexity)
    #eta = c(0.01, 0.1),            # Learning rate (step size shrinkage)
    eta = c(0.05, 0.1, 0.3),         # Learning rate (step size shrinkage)
    gamma = c(0),                   # Minimum loss reduction required to make a split (regularization)
    colsample_bytree = c(0.8),      # Subsample ratio of columns when constructing each tree
    min_child_weight = c(1),        # Minimum sum of instance weight (hessian) needed in a child (controls overfitting)
    subsample = c(0.8)              # Subsample ratio of the training instances (row sampling)
)

# Train the XGBoost model
xgb.fit <- train(
    DEM ~ ., 
    data = fdata.Reg.tr, 
    method = "xgbTree",
    tuneGrid = xgbGrid,
    preProcess = c("center", "scale"),
    trControl = ctrl_tune, 
    verbosity = 0,
    metric = "RMSE"
)


xgb.fit #information about the resampling settings

# Display model information
print(xgb.fit)
ggplot(xgb.fit) + scale_x_log10()

# Variable importance
var_imp <- varImp(xgb.fit)
print(var_imp)
plot(var_imp)

# Predict training data
xgb_pred <- predict(xgb.fit, newdata = fdata.Reg.tr)  

# Model diagnostics
PlotModelDiagnosis(fdata.Reg.tr[, names(fdata.Reg.tr) != "DEM"], fdata.Reg.tr$DEM, xgb_pred, together = TRUE)

# Error measurements
accuracy(fdata.Reg.tr$DEM, xgb_pred)

# Model diagnosis
CheckResiduals.ICAI(xgb.fit, lag=50)

#Check Cross correlation residuals - expl. variables
res <- fdata.Reg.tr$DEM - xgb_pred
ccf(y = res, x = fdata.Reg.tr$WD)
ccf(y = res, x = fdata.Reg.tr$TEMP)


# Plot actual vs predicted
plot(fdata.Reg.tr$DEM, type = "l", main = "Actual vs Predicted", ylab = "Demand")
lines(xgb_pred, col = "red")
legend("topright", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

#################################################################################
## Forecast for new data with h = 7
#################################################################################
# Read new data
fdata.Reg.tv <- readxl::read_excel("DAILY_DEMAND_TV.xlsx")
fdata.Reg.tv <- as.data.frame(fdata.Reg.tv)

# Historical data followed by future exogenous values (DEM unknown)
fdata.Reg.new <- fdata.Reg.tv[, c("WD", "TEMP")]
fdata.Reg.new$DEM <- NA

fdata.join <- rbind(fdata[, c("DEM", "WD", "TEMP")], fdata.Reg.new)

# Create lagged variables
fdata.join$WD_lag1 <- Lag(fdata.join$WD, 1)
fdata.join$WD_lag7 <- Lag(fdata.join$WD, 7)
fdata.join$TEMP_lag1 <- Lag(fdata.join$TEMP, 1)
fdata.join$TEMP_lag7 <- Lag(fdata.join$TEMP, 7)
fdata.join$DEM_lag1 <- Lag(fdata.join$DEM, 1)
fdata.join$DEM_lag7 <- Lag(fdata.join$DEM, 7)

# Each prediction is used as lagged DEM in the following days
tstart <- nrow(fdata)
h <- nrow(fdata.Reg.tv)  # forecast horizon = number of rows in the test file
for (i in (tstart + 1):(tstart + h)) {
  #Predict and substitute in output variable so it can be used in following forecasts
  fdata.join$DEM[i] <- predict(xgb.fit, newdata = fdata.join[i, ])
  #Recalculate lagged variables using the estimated value
  fdata.join$DEM_lag1 <- Lag(fdata.join$DEM, 1)
  fdata.join$DEM_lag7 <- Lag(fdata.join$DEM, 7)
}
y_pred <- fdata.join$DEM[(tstart + 1):(tstart + h)]
y_pred

