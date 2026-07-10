# Clear existing data and graphics
rm(list=ls())
#graphics.off()

# Load libraries ----
pacman::p_load(
  tidyverse, 
  medicaldata,  # opt data set
  rvinecopulib,  # for r-vine copula implementation
  e1071,  # for r-vine copula implementation
  caret,  # for r-vine copula implementation and to train ML models using CV
  EnvStats,  # for r-vine copula implementation - NOTE: predict OVERWRITES predict from base R stats package!!!
  truncnorm,  # for r-vine copula implementation
  dgof,  # for Kolmogorov-Smirnov tests
  caTools,  # to easily split data into train/test sets
  xgboost,  # for XGBoost model
  VIM, # to impute missing values using KNN
  class,  # to train KNN model
  patchwork,  # to plot multiple ggplots at once with same legend
  mice,  # to perform multiple imputation by chained equations
  purrr, 
  foreach, 
  doParallel, 
  furrr,
  ggfortify  # for pca plots
)

# Prepare opt data ----

# Variables to include:
data.opt <- opt %>% dplyr::select(c(PID, Group, Age, Black, White, Nat.Am, Asian, 
                                    Education, Public.Asstce, 
                                    # Hypertension, Diabetes, # remove b/c too few "Yes"
                                    Prev.preg, N.qualifying.teeth, BL.GE, BL..BOP,
                                    BL.PD.avg, BL..PD.4, BL..PD.5, BL.CAL.avg, BL..CAL.2,
                                    BL..CAL.3, BL.Calc.I, BL.Pl.I, BL.Anti.inf, 
                                    # BL.Cortico, # remove b/c too few of stratum "1"
                                    BL.Antibio, BL.Bac.vag, Birthweight,
                                    # has NA:
                                    # BMI, Use.Tob, Drug.Add,  # removing these adds 55 rows back
                                    V3.PD.avg, V5.PD.avg)) %>%
  mutate(Race_fine = case_when(Black == "Yes" & White == "No " & Nat.Am == "No " & Asian == "No " ~ "Black",
                               Black == "No " & White == "Yes" & Nat.Am == "No " & Asian == "No " ~ "White",
                               Black == "No " & White == "No " & Nat.Am == "Yes" & Asian == "No " ~ "Indigenous",
                               Black == "No " & White == "No " & Nat.Am == "No " & Asian == "Yes" ~ "Asian",
                               Black == "Yes" & (White == "Yes" | Nat.Am == "Yes" | Asian == "Yes") ~ "Mixed",
                               White == "Yes" & (Black == "Yes" | Nat.Am == "Yes" | Asian == "Yes") ~ "Mixed",
                               Nat.Am == "Yes" & (White == "Yes" | Black == "Yes" | Asian == "Yes") ~ "Mixed",
                               Asian == "Yes" & (White == "Yes" | Nat.Am == "Yes" | Black == "Yes") ~ "Mixed",
                               Black == "No " & White == "No " & Nat.Am == "No " & Asian == "No " ~ "Other")
  ) %>%
  # Group small categories together to avoid issue of entire strata dropping out when imposing missingness
  mutate(Race = case_when(Race_fine == "Asian" ~ "Other",
                          Race_fine == "Black" ~ "Black",
                          Race_fine == "Indigenous" ~ "Indigenous",
                          Race_fine == "Mixed" ~ "Other",
                          Race_fine == "Other" ~ "Other",
                          Race_fine == "White" ~ "White")) %>%
  dplyr::select(PID, Age, Race, Education, Public.Asstce, Prev.preg, N.qualifying.teeth, 
                BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, BL..PD.5, BL.CAL.avg, BL..CAL.2, 
                BL..CAL.3, BL.Calc.I, BL.Pl.I, BL.Anti.inf, BL.Antibio, BL.Bac.vag, 
                Group, V3.PD.avg, V5.PD.avg, Birthweight
  )

data.opt.cc <- data.opt[complete.cases(data.opt), ]  # 621 rows

# Generate missingness ----

# Standardize continuous variables so they are all on the same scale
# i.e., subtract the mean, divide by the standard deviation
real_cc <- data.opt.cc %>%
  mutate(Age_std = (Age - mean(data.opt.cc$Age))/sd(data.opt.cc$Age),
         N.qualifying.teeth_std = (N.qualifying.teeth - mean(data.opt.cc$N.qualifying.teeth))/sd(data.opt.cc$N.qualifying.teeth),
         BL.GE_std = (BL.GE - mean(data.opt.cc$BL.GE))/sd(data.opt.cc$BL.GE),
         BL..BOP_std = (BL..BOP - mean(data.opt.cc$BL..BOP))/sd(data.opt.cc$BL..BOP),
         BL.PD.avg_std = (BL.PD.avg - mean(data.opt.cc$BL.PD.avg))/sd(data.opt.cc$BL.PD.avg),
         BL..PD.4_std = (BL..PD.4 - mean(data.opt.cc$BL..PD.4))/sd(data.opt.cc$BL..PD.4),
         BL..PD.5_std = (BL..PD.5 - mean(data.opt.cc$BL..PD.5))/sd(data.opt.cc$BL..PD.5),
         BL.CAL.avg_std = (BL.CAL.avg - mean(data.opt.cc$BL.CAL.avg))/sd(data.opt.cc$BL.CAL.avg),
         BL..CAL.2_std = (BL..CAL.2 - mean(data.opt.cc$BL..CAL.2))/sd(data.opt.cc$BL..CAL.2),
         BL..CAL.3_std = (BL..CAL.3 - mean(data.opt.cc$BL..CAL.3))/sd(data.opt.cc$BL..CAL.3),
         BL.Calc.I_std = (BL.Calc.I - mean(data.opt.cc$BL.Calc.I))/sd(data.opt.cc$BL.Calc.I),
         BL.Pl.I_std = (BL.Pl.I - mean(data.opt.cc$BL.Pl.I))/sd(data.opt.cc$BL.Pl.I),
         V3.PD.avg_std = (V3.PD.avg - mean(data.opt.cc$V3.PD.avg))/sd(data.opt.cc$V3.PD.avg),
         V5.PD.avg_std = (V5.PD.avg - mean(data.opt.cc$V5.PD.avg))/sd(data.opt.cc$V5.PD.avg),
         Birthweight_std = (Birthweight - mean(data.opt.cc$Birthweight))/sd(data.opt.cc$Birthweight)
  )

## For MAR, determine vars associated w/ variable at given time point in trial ----

# V3.PD.avg
# First, fit a linear regression model to determine which variables are most associated 
# with the variable at the given time point in the trial (V3.PD.avg)
V3.PD.avg_mod <- lm(formula = V3.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                      as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                      BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + 
                      BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + BL.Antibio + BL.Bac.vag + as.factor(Group),
                    data = real_cc)

summary(V3.PD.avg_mod)
# most predictive: BL.PD.avg, BL..PD.5, BL..CAL.3, Group

# V5.PD.avg
# First, fit a linear regression model to determine which variables are most associated 
# with the variable at the given time point in the trial (V5.PD.avg)
V5.PD.avg_mod <- lm(formula = V5.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                      as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                      BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                      BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf +
                      BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg,
                    data = real_cc)

summary(V5.PD.avg_mod)
# most predictive: Prev.preg, BL.PD.avg, BL..PD.5, BL.Calc.I, Group, V3.PD.avg

# Birthweight
# First, fit a linear regression model to determine which variables are most associated 
# with the variable at the given time point in the trial (Birthweight)
Birthweight_mod <- lm(formula = Birthweight ~ Age + as.factor(Race) + as.factor(Education) + 
                        as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                        BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                        BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                        BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg + V5.PD.avg,
                      data = real_cc)

summary(Birthweight_mod)
# most predictive: V3.PD.avg, V5.PD.avg

# Set random seed ----
set.seed(20250224)

# Scenario 4A: Non-monotone x MAR  x 50% missing x strong mechanism ----
# Variables with missingness: Z_1 (V3.PD.avg), Y (Birthweight)

## Z_1 ----

# Define fixed parameter values
alpha_1 <- 10   # Coefficient for X_1 (BL.PD.avg - STANDARDIZED, so coef needs to be much larger)
alpha_2 <- 10  # Coefficient for X_2 (BL..PD.5 - STANDARDIZED, so coef needs to be much larger)
alpha_3 <- 10  # Coefficient for X_3 (BL..CAL.3 - STANDARDIZED, so coef needs to be much larger)
alpha_4 <- 3.5  # Coefficient for A (tx T v. C)

# Function to find alpha_0 - use standardized continuous variables
find_alpha_0 <- function(alpha_0) {
  logit_p <- alpha_0 + 
    (alpha_1 * real_cc$BL.PD.avg_std) + 
    (alpha_2 * real_cc$BL..PD.5_std) + 
    (alpha_3 * real_cc$BL..CAL.3_std) +     
    ifelse(real_cc$Group == "T", alpha_4, 0) 
  p <- 1 / (1 + exp(-logit_p))  # convert log-odds to probability
  mean(p) - 0.5  # want mean(p) ~ 0.50
}

# Solve for alpha_0
alpha_0 <- uniroot(find_alpha_0, c(-5, 5), extendInt = "yes")$root
print(alpha_0)

# Compute logit(p) for each observation
logit_p <- alpha_0 + 
  (alpha_1 * real_cc$BL.PD.avg_std) +                 # Effect of BL.PD.avg (continuous)
  (alpha_2 * real_cc$BL..PD.5_std) +  # Effect of BL..PD.5 (continuous)
  (alpha_3 * real_cc$BL..CAL.3_std) +  # Effect of BL..CAL.3 (continuous)
  ifelse(real_cc$Group == "T", alpha_4, 0)    # Effect of Group T v. C (binary treatment)

# Calculate p (i.e., convert to probability)
p <- exp(logit_p) / (1 + exp(logit_p))  

# Generate the binary missingness indicator for visit 3
R_3 <- rbinom(n = nrow(real_cc), 1, p)

# Check the proportion of 1s
mean(R_3)  # Should be around 0.50 (since R = 1 if observed)

## Y ----

# Define fixed parameter values
beta_1 <- 20   # Coefficient for Z_1 (V3.PD.avg - STANDARDIZED, so coef needs to be much larger)
beta_2 <- 28  # Coefficient for Z_2(V5.PD.avg - STANDARDIZED, so coef needs to be much larger)

# Function to find gamma_0 - use standardized continuous variables
find_beta_0 <- function(beta_0) {
  logit_p_Y <- beta_0 + 
    (beta_1 * real_cc$V3.PD.avg_std) +
    (beta_2 * real_cc$V5.PD.avg_std)
  p <- 1 / (1 + exp(-logit_p_Y))  # convert log-odds to probability
  mean(p) - 0.5  # want mean(p) ~ 0.5
}

# Solve for beta_0
beta_0 <- uniroot(find_beta_0, c(-5, 5), extendInt = "yes")$root
print(beta_0)

# Compute logit(p) for each observation
logit_p_Y <- beta_0 + 
  # (beta_1 * real_cc$V3.PD.avg_std) +  # Effect of V3.PD.avg (continuous)
  (beta_2 * real_cc$V5.PD.avg_std)  # Effect of V5.PD.avg (continuous)

# Calculate p (i.e., convert to probability)
p_Y <- exp(logit_p_Y) / (1 + exp(logit_p_Y))  

# Generate the binary missingness indicator for final visit
R_Y <- rbinom(n = nrow(real_cc), 1, p_Y)

# Check the proportion of 1s
mean(R_Y)  # Should be around 0.50 (since R = 1 if observed)

# Add R_3, R_Y as columns to df with all var
data_simmiss_scen4A <- real_cc %>%
  mutate(R_3 = R_3,
         R_Y = R_Y)

## Impose missingness ----
data_simmiss_scen4A <- data_simmiss_scen4A %>%
  rename(V3.PD.avg_complete = V3.PD.avg,
         Birthweight_complete = Birthweight) %>%
  mutate(V3.PD.avg = ifelse(R_3 == 1, V3.PD.avg_complete, NA),
         Birthweight = ifelse(R_Y == 1, Birthweight_complete, NA))

# Make sure variables are all of correct type 
data_simmiss_scen4A <- data_simmiss_scen4A %>%
  mutate(Age = as.numeric(Age),
         Race = as.factor(Race),
         Education = as.factor(Education),
         Public.Asstce = as.factor(Public.Asstce),
         Prev.preg = as.factor(Prev.preg),
         N.qualifying.teeth = as.numeric(N.qualifying.teeth),
         BL.GE = as.numeric(BL.GE),
         BL..BOP = as.numeric(BL..BOP),
         BL.PD.avg = as.numeric(BL.PD.avg),
         BL..PD.4 = as.numeric(BL..PD.4),
         BL..PD.5 = as.numeric(BL..PD.5),
         BL.CAL.avg = as.numeric(BL.CAL.avg),
         BL..CAL.2 = as.numeric(BL..CAL.2),
         BL..CAL.3 = as.numeric(BL..CAL.3),
         BL.Calc.I = as.numeric(BL.Calc.I),
         BL.Pl.I = as.numeric(BL.Pl.I),
         BL.Anti.inf = as.factor(BL.Anti.inf),
         BL.Antibio = as.factor(BL.Antibio),
         BL.Bac.vag = as.factor(BL.Bac.vag),
         Group = as.factor(Group),
         V3.PD.avg = as.numeric(V3.PD.avg),
         V5.PD.avg = as.numeric(V5.PD.avg),
         Birthweight = as.numeric(Birthweight)) 

# Scenario 4B: Monotone     x MAR  x 50% missing x strong mechanism ----
# Variables with missingness: Z_1 (V3.PD.avg), Z_2 (V5.PD.avg), Y (Birthweight)

## Z_1 ----

# Define fixed parameter values
alpha_1 <- 10   # Coefficient for X_1 (BL.PD.avg - STANDARDIZED, so coef needs to be much larger)
alpha_2 <- 10  # Coefficient for X_2 (BL..PD.5 - STANDARDIZED, so coef needs to be much larger)
alpha_3 <- 10  # Coefficient for X_3 (BL..CAL.3 - STANDARDIZED, so coef needs to be much larger)
alpha_4 <- 3.5  # Coefficient for A (tx T v. C)

# Function to find alpha_0 - use standardized continuous variables
find_alpha_0 <- function(alpha_0) {
  logit_p <- alpha_0 + 
    (alpha_1 * real_cc$BL.PD.avg_std) + 
    (alpha_2 * real_cc$BL..PD.5_std) + 
    (alpha_3 * real_cc$BL..CAL.3_std) +     
    ifelse(real_cc$Group == "T", alpha_4, 0) 
  p <- 1 / (1 + exp(-logit_p))  # convert log-odds to probability
  mean(p) - 0.5  # want mean(p) ~ 0.50
}

# Solve for alpha_0
alpha_0 <- uniroot(find_alpha_0, c(-5, 5), extendInt = "yes")$root
print(alpha_0)

# Compute logit(p) for each observation
logit_p_1 <- alpha_0 + 
  (alpha_1 * real_cc$BL.PD.avg_std) +                 # Effect of BL.PD.avg (continuous)
  (alpha_2 * real_cc$BL..PD.5_std) +  # Effect of BL..PD.5 (continuous)
  (alpha_3 * real_cc$BL..CAL.3_std) +  # Effect of BL..CAL.3 (continuous)
  ifelse(real_cc$Group == "T", alpha_4, 0)    # Effect of Group T v. C (binary treatment)

# Calculate p (i.e., convert to probability)
p_1 <- exp(logit_p_1) / (1 + exp(logit_p_1))  

# Generate the binary missingness indicator for visit 3
R_3 <- rbinom(n = nrow(real_cc), 1, p_1)

# Check the proportion of 1s
mean(R_3)  # Should be around 0.5 (since R = 1 if observed)

## Z_2 ----

# Define fixed parameter values
beta_1 <- 3 # Coefficient for X_2 (Prev.preg Yes v. No )
beta_2 <- 15   # Coefficient for X_2 (BL.PD.avg - STANDARDIZED, so coef needs to be much larger)
beta_3 <- 15   # Coefficient for X_3 (BL..PD.5 - STANDARDIZED, so coef needs to be much larger)
beta_4 <- 15   # Coefficient for X_4 (BL.Calc.I - STANDARDIZED, so coef needs to be much larger)
beta_5 <- 5   # Coefficient for A (Group T v. C)
beta_6 <- 17   # Coefficient for Z_1 (V3.PD.avg - STANDARDIZED, so coef needs to be much larger)

# Function to find beta_0 - use standardized continuous variables
find_beta_0 <- function(beta_0) {
  logit_p <- beta_0 + 
    ifelse(real_cc$Prev.preg == "Yes", beta_1, 0) + 
    (beta_2 * real_cc$BL.PD.avg_std) + 
    (beta_3 * real_cc$BL..PD.5_std) + 
    (beta_4 * real_cc$BL.Calc.I_std) + 
    ifelse(real_cc$Group == "T", beta_5, 0) + 
    (beta_6 * real_cc$V3.PD.avg_std) 
  p <- 1 / (1 + exp(-logit_p))  # convert log-odds to probability
  mean(p) - 0.5  # want mean(p) ~ 0.5 
}

# Solve for beta_0
beta_0 <- uniroot(find_beta_0, c(-5, 5), extendInt = "yes")$root
print(beta_0)

# Compute logit(p) for each observation
logit_p_2 <- beta_0 + 
  ifelse(real_cc$Prev.preg == "Yes", beta_1, 0) +  # Effect of Prev.preg (binary)
  (beta_2 * real_cc$BL.PD.avg_std) +  # Effect of BL.PD.avg (continuous)
  (beta_3 * real_cc$BL..PD.5_std) +            # Effect of BL..PD.5 (continuous)
  (beta_4 * real_cc$BL.Calc.I_std) +  # Effect of BL.Calc.I (continuous)
  ifelse(real_cc$Group == "T", beta_5, 0) +    # Effect of Group T v. C (binary treatment)
  (beta_6 * real_cc$V3.PD.avg_std)  # Effect of V3.PD.avg (continuous)

# Calculate p (i.e., convert to probability)
p_2 <- exp(logit_p_2) / (1 + exp(logit_p_2))  

# Generate the binary missingness indicator for visit 5
# Note: under monotone missingness, we manually set R_5 = 0 when R_3 = 0
# (Recall that R = 0 means missing)
R_5_raw <- rbinom(n = nrow(real_cc), 1, p_2)
R_5 <- ifelse(R_3 == 0, 0, R_5_raw)

# Check the proportion of 1s
mean(R_5)  # Should be less than mean(R_3) (since missingness should be greater than at Z_1)

## Y ----

# Define fixed parameter values
gamma_1 <- 20   # Coefficient for Z_1 (V3.PD.avg - STANDARDIZED, so coef needs to be much larger)
gamma_2 <- 28  # Coefficient for Z_2(V5.PD.avg - STANDARDIZED, so coef needs to be much larger)

# Function to find gamma_0 - use standardized continuous variables
find_gamma_0 <- function(gamma_0) {
  logit_p <- gamma_0 + 
    (gamma_1 * real_cc$V3.PD.avg_std) +
    (gamma_2 * real_cc$V5.PD.avg_std)
  p <- 1 / (1 + exp(-logit_p))  # convert log-odds to probability
  mean(p) - 0.5  # want mean(p) ~ 0.5
}

# Solve for beta_0
gamma_0 <- uniroot(find_gamma_0, c(-5, 5), extendInt = "yes")$root
print(gamma_0)

# Compute logit(p) for each observation
logit_p_Y <- gamma_0 + 
  (gamma_1 * real_cc$V3.PD.avg_std) +  # Effect of V3.PD.avg (continuous)
  (gamma_2 * real_cc$V5.PD.avg_std)  # Effect of V5.PD.avg (continuous)

# Calculate p (i.e., convert to probability)
p_Y <- exp(logit_p_Y) / (1 + exp(logit_p_Y))  

# Generate the binary missingness indicator for final visit
# Note: under monotone missingness, we manually set R_Y = 0 when R_5 = 0
# (Recall that R = 0 means missing)
R_Y_raw <- rbinom(n = nrow(real_cc), 1, p_Y)
R_Y <- ifelse(R_5 == 0, 0, R_Y_raw)

# Check the proportion of 1s
mean(R_Y)  # should be less than mean(R_5)

# Add R_3, R_5, R_Y as columns to df with all var
data_simmiss_scen4B <- real_cc %>%
  mutate(R_3 = R_3,
         R_5 = R_5,
         R_Y = R_Y)

## Impose missingness ----
data_simmiss_scen4B <- data_simmiss_scen4B %>%
  rename(V3.PD.avg_complete = V3.PD.avg,
         V5.PD.avg_complete = V5.PD.avg,
         Birthweight_complete = Birthweight) %>%
  mutate(V3.PD.avg = ifelse(R_3 == 1, V3.PD.avg_complete, NA),
         V5.PD.avg = ifelse(R_5 == 1, V5.PD.avg_complete, NA),
         Birthweight = ifelse(R_Y == 1, Birthweight_complete, NA))

# Make sure variables are all of correct type 
data_simmiss_scen4B <- data_simmiss_scen4B %>%
  mutate(Age = as.numeric(Age),
         Race = as.factor(Race),
         Education = as.factor(Education),
         Public.Asstce = as.factor(Public.Asstce),
         Prev.preg = as.factor(Prev.preg),
         N.qualifying.teeth = as.numeric(N.qualifying.teeth),
         BL.GE = as.numeric(BL.GE),
         BL..BOP = as.numeric(BL..BOP),
         BL.PD.avg = as.numeric(BL.PD.avg),
         BL..PD.4 = as.numeric(BL..PD.4),
         BL..PD.5 = as.numeric(BL..PD.5),
         BL.CAL.avg = as.numeric(BL.CAL.avg),
         BL..CAL.2 = as.numeric(BL..CAL.2),
         BL..CAL.3 = as.numeric(BL..CAL.3),
         BL.Calc.I = as.numeric(BL.Calc.I),
         BL.Pl.I = as.numeric(BL.Pl.I),
         BL.Anti.inf = as.factor(BL.Anti.inf),
         BL.Antibio = as.factor(BL.Antibio),
         BL.Bac.vag = as.factor(BL.Bac.vag),
         Group = as.factor(Group),
         V3.PD.avg = as.numeric(V3.PD.avg),
         V5.PD.avg = as.numeric(V5.PD.avg),
         Birthweight = as.numeric(Birthweight)) 

# FUNCTIONS TO LOAD ----
DataBaseline <- data_simmiss_scen4A %>% 
  dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                  BL.Bac.vag, Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, 
                  BL..PD.4, BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, 
                  BL.Pl.I))
# R-vine copula ----

# Function to transform uniform distribution to original scale according to empirical distribution
PseudoObsInverse <- function(DataBaseline, UniformData) {
  # DataBaseline is the matrix of covariates (original data)
  # UniformData is the matrix of uniform data to be transformed to original scale
  PsInverse <- list()
  for (j in 1:ncol(DataBaseline)){ 
    ecdfj <- ecdf(DataBaseline[, j])  # empirical cdf
    ECDFvar <- get("x", environment(ecdfj))
    ECDFjump <- get("y", environment(ecdfj))
    PsInverse[[j]] <- stepfun(ECDFjump[-length(ECDFjump)], ECDFvar)  # define step function
  } 
  ScaledData <- matrix(0, nrow(UniformData), ncol(UniformData))
  for (j in 1:ncol(UniformData)){ ScaledData[, j] <- PsInverse[[j]](UniformData[, j]) }
  ScaledData <- as.data.frame(ScaledData)
  # output
  return(ScaledData)
}

# Function to estimate the 'R-vine copula' model of the baseline data (covariates)
Estimation_Copula <- function(DataBaseline)   {
  # DataBaseline is the matrix of covariates (original data)
  
  ## Data  preparation
  # Transformation of continuous variables (in original scale) into uniform distribution variables  
  # Pseudo-observations compute using rvinecopulib package   
  U_cont <- pseudo_obs(DataBaseline[, 8:19])  # columns 8:19 are the continuous variables
  
  # Distribution of the discrete variables
  disc_1 <- as.integer(DataBaseline[, 1])  # binary variable should have levels 0, 1
  disc_2 <- as.integer(DataBaseline[, 2])  # categorical variable should have levels 0, 1, 2, etc.
  disc_3 <- as.integer(DataBaseline[, 3])
  disc_4 <- as.integer(DataBaseline[, 4])
  disc_5 <- as.integer(DataBaseline[, 5])
  disc_6 <- as.integer(DataBaseline[, 6])
  disc_7 <- as.integer(DataBaseline[, 7])
  freq_disc1 <- prop.table(table(DataBaseline[, 1]))
  freq_disc2 <- prop.table(table(DataBaseline[, 2]))
  freq_disc3 <- prop.table(table(DataBaseline[, 3]))
  freq_disc4 <- prop.table(table(DataBaseline[, 4]))
  freq_disc5 <- prop.table(table(DataBaseline[, 5]))
  freq_disc6 <- prop.table(table(DataBaseline[, 6]))
  freq_disc7 <- prop.table(table(DataBaseline[, 7]))
  
  # Preparation of the discrete variables needed to use 'vinecop' function for mixed data (package rvinecopulib)
  Freq_disc_t1 <- cbind(pdiscrete(disc_1 + 1, freq_disc1), pdiscrete(disc_2 + 1, freq_disc2),
                        pdiscrete(disc_3 + 1, freq_disc3), pdiscrete(disc_4 + 1, freq_disc4),
                        pdiscrete(disc_5 + 1, freq_disc5), pdiscrete(disc_6 + 1, freq_disc6),
                        pdiscrete(disc_7 + 1, freq_disc7))
  Freq_disc_t0 <- cbind(pdiscrete(disc_1, freq_disc1), pdiscrete(disc_2, freq_disc2),
                        pdiscrete(disc_3, freq_disc3), pdiscrete(disc_4, freq_disc4),
                        pdiscrete(disc_5, freq_disc5), pdiscrete(disc_6, freq_disc6),
                        pdiscrete(disc_7, freq_disc7))
  U_mixte <- cbind(Freq_disc_t1, U_cont, Freq_disc_t0) # need Freq_disc_t0 to handle discrete obs (check details of rdocumentation)
  
  # Estimation of the R-vine model for mixed data using rvinecopulib package
  fit_DataDriven <- vinecop(U_mixte, var_types = c(rep("d", 7), rep("c", 12)))
  
  # Definition of the R-vine distribution 
  Fit_dist <- vinecop_dist(fit_DataDriven$pair_copulas, fit_DataDriven$structure, fit_DataDriven$var_types)
  
  ## Output
  return(Fit_dist)
} 

# Function to generate a sample according to the estimated R-vine model and baseline data (covariates)
Simulation_Copula <- function(N, Fit_dist, DataBaseline)   {
  # N is number of observations to be generated (sample size)
  # Fit_dist is the R-vine model estimated on original data  
  # DataBaseline is the matrix of covariates (original data)
  
  # Generation of a uniform sample using the estimated R-vine copula distribution
  U_Simu <- rvinecop(N, Fit_dist)
  # Transform uniform distribution to original scale according to empirical distribution  
  # (reverse function for 'pseudo_obs' one)
  # This function is defined above
  VGenCop <- PseudoObsInverse(DataBaseline, U_Simu)
  # Data preparation
  for (i in 1:7){ VGenCop[,i] = as.factor(VGenCop[, i]) }  # discrete vars
  for (i in 8:19){ VGenCop[,i] = as.numeric(as.character(VGenCop[, i])) }  # continuous vars
  colnames(VGenCop) <- colnames(DataBaseline)
  levels(VGenCop[, 1]) <- c('Black', 'Indigenous', 'Other', 'White')  # levels in original data 
  levels(VGenCop[, 2]) <- c('8-12 yrs ', 'LT 8 yrs ', 'MT 12 yrs') # Education
  levels(VGenCop[, 3]) <- c('No ', 'Yes')  # public assistance
  levels(VGenCop[, 4]) <- c('No ', 'Yes')  # previous pregnancy 
  levels(VGenCop[, 5]) <- c('0', '1')   # BL.Anti.inf
  levels(VGenCop[, 6]) <- c('0', '1')   # BL.Antibio
  levels(VGenCop[, 7]) <- c('0', '1')  # BL.Bac.vag
  
  # Output
  return(VGenCop)
}   



# Execution models ----

# Function to generate random treatment assignment
Simulation_Treatment <- function(N, Probabilities) {
  # N = number of trials = number of patients = number of rows in virtual baseline cohort
  # Treatment_Arms = all possible treatment arms = vector of possible treatments e.g., c(T, C)
  # Probabilities = probabilities for each outcome = equal chance between all treatment options
  
  tx_assign_random_num = rbinom(n = N, size = 1, prob = Probabilities)
  tx_assign_random <- ifelse(tx_assign_random_num == 1, "T", "C")
  
  return(tx_assign_random)
}

# Function to generate cd420 based on previous learned model and new data (simulated covariates) 
Simulation_PostRandom_V3 <- function(Model, Covariates_synth, Tx_synth) {
  # Model is the prediction model used to predict the post-randomization variable at visit 3 (V3.PD.avg)
  # Covariates_synth are the synthetic covariates used to generate synthetic V3.PD.avg
  # Tx_synth are the synthetic treatment assignments used to generate synthetic V3.PD.avg
  
  # Prediction
  V3.PD.avg_predict <- stats::predict(Model, newdata = cbind(Covariates_synth, Group = Tx_synth))
  
  # Residuals
  V3.PD.avg_resid <- residuals(Model)
  
  # Initialize vector to store synthetic V3.PD.avg values
  V3.PD.avg_synthetic = rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(V3.PD.avg_predict)) {
    # Get all possible values of pred + resid for the i'th observation
    pred_resid_sums <- V3.PD.avg_predict[i] + V3.PD.avg_resid
    
    # All pred + resid >= 0
    pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
    
    # Randomly sample from non-negative sum values
    sample_val = sample(pred_resid_sums_pos, size = 1)
    
    # Save sample
    V3.PD.avg_synthetic[i] <- sample_val
  }
  return(V3.PD.avg_synthetic)
}

# Function to generate cd496 based on previous learned model and new data (simulated covariates)
Simulation_PostRandom_V5 <- function(Model, Covariates_synth, Tx_synth, PostRandom_V3) {
  # Model is the prediction model used to predict the post-randomization variable at visit 5 (V5.PD.avg)
  # Covariates_synth are the synthetic covariates used to generate synthetic V5.PD.avg
  # Tx_synth are the synthetic treatment assignments used to generate synthetic V5.PD.avg
  # PostRandom_V3 are the values for synthetic post randomization variable at visit 3 (V3.PD.avg)
  
  # Prediction
  V5.PD.avg_predict <- stats::predict(Model, 
                                      newdata = cbind(Covariates_synth, 
                                                      Group = Tx_synth, 
                                                      V3.PD.avg = PostRandom_V3))
  
  # Residuals
  V5.PD.avg_resid <- residuals(Model)
  
  # Initialize vector to store synthetic V5.PD.avg values
  V5.PD.avg_synthetic = rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(V5.PD.avg_predict)) {
    # Get all possible values of pred + resid for the i'th observation
    pred_resid_sums <- V5.PD.avg_predict[i] + V5.PD.avg_resid
    
    # All pred + resid >= 0
    pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
    
    # Randomly sample from non-negative sum values
    sample_val = sample(pred_resid_sums_pos, size = 1)
    
    # Save sample
    V5.PD.avg_synthetic[i] <- sample_val
  }
  return(V5.PD.avg_synthetic)
}

# Function to predict the outcome variable based on previous learned model and new data (simulated covariates)
# X + A + Z_1 + Z_2
Simulation_DataOutcome <- function(Model, Covariates_synth, Tx_synth, PostRandom_V3, PostRandom_V5) {
  # Model is the prediction model used to predict the outcome
  # Covariates_synth are the synthetically-generated covariates used to predict the outcome
  # Prediction
  outcome_predict <- stats::predict(Model, 
                                    newdata = cbind(Covariates_synth, 
                                                    Group = Tx_synth, 
                                                    V3.PD.avg = PostRandom_V3,
                                                    V5.PD.avg = PostRandom_V5))
  
  # Residuals
  outcome_resid <- residuals(Model)
  
  # Initialize vector to store synthetic outcome values
  outcome_synthetic = rep(NA, nrow(Covariates_synth))
  
  for (i in 1:length(outcome_predict)) {
    # Get all possible values of pred + resid for the i'th observation
    pred_resid_sums <- outcome_predict[i] + outcome_resid
    
    # All pred + resid >= 0
    pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
    
    # Randomly sample from non-negative sum values
    sample_val = sample(pred_resid_sums_pos, size = 1)
    
    # Save sample
    outcome_synthetic[i] <- sample_val
  }
  return(outcome_synthetic)
}


# Generate 1 synthetic data set functions ----

## CC (Non-monotone and monotone) ----

# CC Data Pre-Processing step (same for non-monotone and monotone missingness)
# Function to generate entire data set, CC
generate1dataset_cc_preproc_retrycopula <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Ouput: dataframe of synthetic data
  
  # First, remove rows with missing values as a data pre-processing step
  real_data <- real_data[complete.cases(real_data), ]  
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(Birthweight)
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                    BL.Bac.vag)) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, 
                    BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, BL.Pl.I))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(Group) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(V3.PD.avg, V5.PD.avg))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Marginal distribution of the covariates are estimated using empirical estimator 
  seed <- i
  fittedcop <- NULL
  fitted_ok <- FALSE
  max_retries <- 50
  attempt <- 1
  
  # Estimation of the R-vine model based on original data
  # Rvine_dist <- Estimation_Copula(Cov)
  # DataBaseline is the matrix of covariates (original data)
  DataBaseline <- Cov
  
  ## Data  preparation
  # Transformation of continuous variables (in original scale) into uniform distribution variables  
  # Pseudo-observations compute using rvinecopulib package   
  U_cont <- pseudo_obs(DataBaseline[, 8:19])  # columns 8:19 are the continuous variables
  
  # Distribution of the discrete variables
  disc_1 <- as.integer(DataBaseline[, 1])  # binary variable should have levels 0, 1
  disc_2 <- as.integer(DataBaseline[, 2])  # categorical variable should have levels 0, 1, 2, etc.
  disc_3 <- as.integer(DataBaseline[, 3])
  disc_4 <- as.integer(DataBaseline[, 4])
  disc_5 <- as.integer(DataBaseline[, 5])
  disc_6 <- as.integer(DataBaseline[, 6])
  disc_7 <- as.integer(DataBaseline[, 7])
  freq_disc1 <- prop.table(table(DataBaseline[, 1]))
  freq_disc2 <- prop.table(table(DataBaseline[, 2]))
  freq_disc3 <- prop.table(table(DataBaseline[, 3]))
  freq_disc4 <- prop.table(table(DataBaseline[, 4]))
  freq_disc5 <- prop.table(table(DataBaseline[, 5]))
  freq_disc6 <- prop.table(table(DataBaseline[, 6]))
  freq_disc7 <- prop.table(table(DataBaseline[, 7]))
  
  # Preparation of the discrete variables needed to use 'vinecop' function for mixed data (package rvinecopulib)
  Freq_disc_t1 <- cbind(pdiscrete(disc_1 + 1, freq_disc1), pdiscrete(disc_2 + 1, freq_disc2),
                        pdiscrete(disc_3 + 1, freq_disc3), pdiscrete(disc_4 + 1, freq_disc4),
                        pdiscrete(disc_5 + 1, freq_disc5), pdiscrete(disc_6 + 1, freq_disc6),
                        pdiscrete(disc_7 + 1, freq_disc7))
  Freq_disc_t0 <- cbind(pdiscrete(disc_1, freq_disc1), pdiscrete(disc_2, freq_disc2),
                        pdiscrete(disc_3, freq_disc3), pdiscrete(disc_4, freq_disc4),
                        pdiscrete(disc_5, freq_disc5), pdiscrete(disc_6, freq_disc6),
                        pdiscrete(disc_7, freq_disc7))
  U_mixte <- cbind(Freq_disc_t1, U_cont, Freq_disc_t0) # need Freq_disc_t0 to handle discrete obs (check details of rdocumentation)
  
  # Estimation of the R-vine model for mixed data using rvinecopulib package
  while (!fitted_ok && attempt <= max_retries) {
    set.seed(seed)
    
    try_fit <- try(
      vinecop(U_mixte, var_types = c(rep("d", 7), rep("c", 12))),
      silent = TRUE
    )
    
    if (inherits(try_fit, "try-error") || length(try_fit$pair_copulas[[1]]) + 1 == 0) {
      # fitting failed -> increase seed and retry
      # second condition is checking if copula dimension is zero
      seed <- seed + 1
      attempt <- attempt + 1
    } else {
      fittedcop <- try_fit
      fitted_ok <- TRUE
    }
  }
  
  if (!fitted_ok) {
    return(list(seed_used = seed, sim = NA))  # bail out gracefully
  } 
  
  # Definition of the R-vine distribution 
  Rvine_dist <- vinecop_dist(fittedcop$pair_copulas, fittedcop$structure, fittedcop$var_types)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copula(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Probabilities = c(0.5, 0.5))
  
  # Simulation of post-randomization variable at visit 3 (V3.PD.avg)
  # Model to predict V3.PD.avg
  # The prediction model is learned based on original data using a linear regression model
  V3.PD.avg_data_learn <- cbind(db, Group = Tx, V3.PD.avg = Post_Random$V3.PD.avg)
  
  V3.PD.avg_model <- lm(V3.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + as.factor(Group),
                        data = V3.PD.avg_data_learn)
  
  V3.PD.avgSimu <- Simulation_PostRandom_V3(Model = V3.PD.avg_model, Covariates_synth = DataSimu, 
                                            Tx_synth = TxSimu)
  
  # Simulation of post-randomization variable at visit 5 (V5.PD.avg_model)
  # Model to predict V5.PD.avg_model
  # The prediction model is learned based on original data using a linear regression model
  V5.PD.avg_data_learn <- cbind(db, Group = Tx, V3.PD.avg = Post_Random$V3.PD.avg, V5.PD.avg = Post_Random$V5.PD.avg)
  
  V5.PD.avg_model <- lm(V5.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg,
                        data = V5.PD.avg_data_learn)
  
  
  V5.PD.avgSimu <- Simulation_PostRandom_V5(Model = V5.PD.avg_model, Covariates_synth = DataSimu,
                                            Tx_synth = TxSimu, PostRandom_V3 = V3.PD.avgSimu)
  
  # Simulation of the outcome for each virtual patients
  # Model to predict the outcome variable
  # The prediction model is learned based on original data using a logistic regression model 
  outcome_data_learn <- cbind(db,
                              # db[, !(names(db) %in% c("Outcome"))], 
                              Group = Tx, 
                              V3.PD.avg = Post_Random$V3.PD.avg, 
                              V5.PD.avg = Post_Random$V5.PD.avg
                              # ,
                              #                           Birthweight = Outcome
  )
  
  outcome_model <- lm(Birthweight ~ Age + as.factor(Race) + as.factor(Education) + 
                        as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                        BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                        BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                        BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg + V5.PD.avg, 
                      data = outcome_data_learn)
  
  OutcomeSimu <- Simulation_DataOutcome(Model = outcome_model, Covariates_synth = DataSimu,
                                        Tx_synth = TxSimu, PostRandom_V3 = V3.PD.avgSimu,
                                        PostRandom_V5 = V5.PD.avgSimu)
  
  # Final table of synthetic data generated via R-vine copula + execution models
  data_synthetic <- cbind(PID = c(1:nrow(DataSimu)),
                          DataSimu, 
                          Group = as.factor(TxSimu),
                          V3.PD.avg = V3.PD.avgSimu,
                          V5.PD.avg = V5.PD.avgSimu,
                          Birthweight = OutcomeSimu)
  
  return(data_synthetic)
}

## Non-monotone missingness ----

# IPW
# Missingness Indicator method, Pr(Y = obs | X, A, R_Z1, Z2, R_Z1:Z1)
generate1dataset_aipw_nonmono_missind <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Output: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(Birthweight)
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                    BL.Bac.vag)) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, 
                    BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, BL.Pl.I))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(Group) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(V3.PD.avg, V5.PD.avg))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Marginal distribution of the covariates are estimated using empirical estimator 
  set.seed(random_seed)
  
  # Estimation of the R-vine model based on original data
  Rvine_dist <- Estimation_Copula(Cov)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copula(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Probabilities = c(0.5, 0.5))
  
  # Simulation of (complete) post-randomization variable at visit 3 (V3.PD.avg)
  
  # Data subset - include real patient ID (V3.PD.avg has missingness)
  V3.PD.avg_data_learn <- cbind(PID = real_data$PID, db, Group = Tx, 
                                V3.PD.avg = Post_Random$V3.PD.avg, R_3 = real_data$R_3)
  
  # Model to estimate probability of being observed at visit 3
  # Note: this model is also used to generate synthetic missingness (R_3)
  V3.PD.avg_probobs_model <- glm(as.factor(R_3) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group),
                                 data = V3.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at visit 3 for REAL data
  V3.PD.avg_data_learn$V3.PD.avg_probobs <-  predict.glm(V3.PD.avg_probobs_model,
                                                         type = "response")
  
  # Weighted model to predict V3.PD.avg
  # Note: weights = 1/p_i, where p_i = prob of being observed (estimated by V3.PD.avg_probobs)
  V3.PD.avg_model <- lm(V3.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + as.factor(Group),
                        data = V3.PD.avg_data_learn,
                        weights = (1/V3.PD.avg_probobs))  # prob of being observed for REAL data
  
  # Generate synthetic V3.PD.avg (fully observed)
  V3.PD.avgSimu <- Simulation_PostRandom_V3(Model = V3.PD.avg_model, Covariates_synth = DataSimu, 
                                            Tx_synth = TxSimu)
  
  # Simulation of post-randomization variable at visit 5 (V5.PD.avg) - this variable is fully obs. in real data
  # Model to predict V5.PD.avg
  # The prediction model is learned based on original data using a linear regression model
  V5.PD.avg_data_learn <- cbind(V3.PD.avg_data_learn, V5.PD.avg = Post_Random$V5.PD.avg)
  
  # Fit a weighted execution model to account for missingness in Z1 (V3.PD.avg)
  V5.PD.avg_model <- lm(V5.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + V3.PD.avg + as.factor(Group),
                        data = V5.PD.avg_data_learn,
                        weights = (1/V3.PD.avg_probobs))  # prob of Z1 being observed for REAL data
  
  V5.PD.avgSimu <- Simulation_PostRandom_V5(Model = V5.PD.avg_model, Covariates_synth = DataSimu,
                                            Tx_synth = TxSimu, PostRandom_V3 = V3.PD.avgSimu)
  
  # Simulation of the outcome
  
  # Data subset (not really a subset, here we use all variables)
  # V3.PD.avg, Birthweight (the outcome) have missingness
  outcome_data_learn <- cbind(V5.PD.avg_data_learn,
                              Birthweight = Outcome,
                              R_Y = real_data$R_Y,
                              R_3xZ1 = ifelse(V5.PD.avg_data_learn$R_3 == 1, 
                                              V5.PD.avg_data_learn$V3.PD.avg,
                                              0))
  
  # Estimated probabilities of being observed at final visit for REAL data
  # based on the following model: 
  # Pr(Y = obs | X, A, R_Z1, Z2, R_Z1*Z1)
  
  # Model to estimate probability of being observed at final visit
  Y_probobs_model <- glm(as.factor(R_Y) ~ Age + as.factor(Race) + as.factor(Education) + 
                           as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                           BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                           BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                           BL.Antibio + BL.Bac.vag + as.factor(Group) + 
                           R_3 + V5.PD.avg + R_3xZ1,
                         data = outcome_data_learn,
                         family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at outcome for REAL data
  outcome_data_learn$Birthweight_probobs <-  predict.glm(Y_probobs_model,
                                                         type = "response")  
  
  # Weighted model to predict Birthweight (outcome)
  # Note: weights = 1/p_i, where p_i = prob of being observed (estimated by Birthweight_probobs)
  outcome_model <- lm(Birthweight ~ Age + as.factor(Race) + as.factor(Education) + 
                        as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                        BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                        BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                        BL.Antibio + BL.Bac.vag + as.factor(Group) +
                        R_3 + V5.PD.avg + R_3xZ1,
                      data = outcome_data_learn,
                      weights = (1/(Birthweight_probobs)))  # prob of being observed for REAL data
  
  # Generate synthetic outcome (all observed)
  # (Note: can't use general R function for execution model because need to add R_3 variables in model)
  OutcomeSimu <- predict.lm(outcome_model,
                            newdata = cbind(DataSimu,
                                            Group = TxSimu,
                                            V3.PD.avg = V3.PD.avgSimu,
                                            V5.PD.avg = V5.PD.avgSimu,
                                            # R_3 = ifelse(!is.na(V3.PD.avgSimu), 1, 0), # should be all 1
                                            # R_3xZ1 = ifelse(R_3 == 1, 
                                            #                 V3.PD.avgSimu,
                                            #                  0)  # should be all V3.PD.avg values
                                            R_3 = rep(1, nrow(DataSimu)), # should be all 1
                                            R_3xZ1 = V3.PD.avgSimu  # should be all V3.PD.avg values
                            ))
  
  # Generate missingness (V3.PD.avg, Y)
  
  # V3.PD.avg (Z1):
  
  # Estimated probabilities of being observed at visit 3 for SIMULATED data
  V3.PD.avgSimu_probobs <- predict.glm(V3.PD.avg_probobs_model,
                                       newdata = cbind(DataSimu,
                                                       Group = TxSimu),
                                       type = "response")
  
  # Generate synthetic R_3
  R_3Simu <- rbinom(n = n_obs, size = 1, prob = V3.PD.avgSimu_probobs)
  
  # Generate V3.PD.avg (Z1) with missingness
  V3.PD.avgSimu_withmiss <- cbind(R_3Simu, V3.PD.avgSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_3Simu == 1, V3.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  
  
  # Outcome (Birthweight, Y):
  
  # Estimated probabilities of being observed at outcome for SIMULATED data
  OutcomeSimu_probobs <- predict.glm(Y_probobs_model,
                                     newdata = cbind(DataSimu,
                                                     Group = TxSimu,
                                                     V3.PD.avg = V3.PD.avgSimu,
                                                     V5.PD.avg = V5.PD.avgSimu,
                                                     # R_3 = ifelse(!is.na(V3.PD.avgSimu), 1, 0), # should be all 1
                                                     # R_3xZ1 = ifelse(R_3 == 1, 
                                                     #                 V3.PD.avgSimu,
                                                     #                  0)  # should be all V3.PD.avg values
                                                     R_3 = 1, # should be all 1
                                                     R_3xZ1 = V3.PD.avgSimu  # should be all V3.PD.avg values
                                     ),
                                     type = "response")
  
  # Generate synthetic R_Y
  R_YSimu = rbinom(n = n_obs, size = 1, prob = OutcomeSimu_probobs)
  
  # Generate Birthweight (Y) with missingness
  OutcomeSimu_withmiss <- cbind(R_YSimu, OutcomeSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_YSimu == 1, OutcomeSimu, NA)) %>%
    pull(withmiss)
  
  
  # Final table of synthetic data
  data_synthetic <- cbind(PID = c(1:nrow(DataSimu)),
                          DataSimu, 
                          Group = as.factor(TxSimu),
                          V3.PD.avg_complete = V3.PD.avgSimu, 
                          V3.PD.avg = V3.PD.avgSimu_withmiss,
                          V5.PD.avg = V5.PD.avgSimu,  # fully obs. in real data
                          Birthweight_complete = OutcomeSimu,
                          Birthweight = OutcomeSimu_withmiss)
  
  return(data_synthetic)
}

# MI
# (Note: to generate synthetic missingness, we implement missingness indicator method)
generate1dataset_mi_nonmono <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Output: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(Birthweight)
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                    BL.Bac.vag)) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, 
                    BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, BL.Pl.I))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(Group) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(V3.PD.avg, V5.PD.avg))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Marginal distribution of the covariates are estimated using empirical estimator 
  set.seed(random_seed)
  
  # Estimation of the R-vine model based on original data
  Rvine_dist <- Estimation_Copula(Cov)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copula(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Probabilities = c(0.5, 0.5))
  
  # Post-randomization variables have missingness, perform MI to get m complete data sets
  Imp_data <- real_data %>% dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, 
                                            BL.Anti.inf, BL.Antibio, BL.Bac.vag, Age, 
                                            N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, 
                                            BL..PD.4, BL..PD.5, BL.CAL.avg, BL..CAL.2, 
                                            BL..CAL.3, BL.Calc.I, BL.Pl.I, Group, 
                                            V3.PD.avg, V5.PD.avg, Birthweight))
  
  # Calculate m, where m ~ max(prop. of missingness)*100
  m <- round(colMeans(is.na(Imp_data)) %>% max()*100)  # m ~ prop. of missingness*100
  
  # Perform multiple imputation by chained equations (mice)
  imp <- mice(data = Imp_data, m = m, maxit = 10, seed = random_seed, print = FALSE)
  
  
  # Simulation of post-randomization variable at week 20 (cd420)
  
  # First, fit execution model to each of the m imputed data sets
  V3.PD.avgmodel_imp <- with(imp,
                             lm(V3.PD.avg ~ as.factor(Race) + as.factor(Education) + 
                                  as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                  as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                  Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                  BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                  BL.Calc.I + BL.Pl.I + as.factor(Group))
  )
  
  # Predict V3.PD.avg using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  V3.PD.avg_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- V3.PD.avgmodel_imp$analyses[[j]]$coefficients %>% as.matrix()
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu)))
    
    # Predicted synthetic Z1, V3.PD.avg
    pred <- t(param) %*% t(data) %>% t()
    
    V3.PD.avg_preds[, j] <- pred
    
  }
  
  # Generate synthetic V3.PD.avg for each of the m model fits
  # Object to store m sets of generated V3.PD.avg
  V3.PD.avg_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    V3.PD.avg_predict <- V3.PD.avg_preds[, j]
    
    # Residuals from j'th model
    V3.PD.avg_resid <- V3.PD.avgmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(V3.PD.avg_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- V3.PD.avg_predict[i] + V3.PD.avg_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      V3.PD.avg_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic V3.PD.avg, Z^s_1 (all observed)
  V3.PD.avgSimu <- apply(V3.PD.avg_mgens, 1, sample, size = 1)
  
  
  # Simulation of post-randomization variable at week 96 (V5.PD.avg)
  
  # First, fit execution model to each of the m imputed data sets
  # Note: even though V5.PD.avg is fully observed in the real data, V3.PD.avg has missingness
  #       so we still proceed with using the imputed data sets
  V5.PD.avgmodel_imp <- with(imp,
                             lm(V5.PD.avg ~ as.factor(Race) + as.factor(Education) + 
                                  as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                  as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                  Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                  BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                  BL.Calc.I + BL.Pl.I + as.factor(Group) + V3.PD.avg)
  )
  
  # Predict V5.PD.avg using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  V5.PD.avg_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- V5.PD.avgmodel_imp$analyses[[j]]$coefficients %>% as.matrix()
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu), V3.PD.avgSimu))
    
    # Predicted synthetic Z2, V5.PD.avg
    pred <- t(param) %*% t(data) %>% t()
    
    V5.PD.avg_preds[, j] <- pred
    
  }
  
  # Generate synthetic V5.PD.avg for each of the m model fits
  # Object to store m sets of generated V5.PD.avg
  V5.PD.avg_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    V5.PD.avg_predict <- V5.PD.avg_preds[, j]
    
    # Residuals from j'th model
    V5.PD.avg_resid <- V5.PD.avgmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(V5.PD.avg_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- V5.PD.avg_predict[i] + V5.PD.avg_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      V5.PD.avg_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic V5.PD.avg, Z^s_2 (all observed)
  V5.PD.avgSimu <- apply(V5.PD.avg_mgens, 1, sample, size = 1)
  
  
  # Simulation of the outcome for each virtual patients
  
  # First, fit execution model to each of the m imputed data sets
  Birthweightmodel_imp <- with(imp,
                               lm(Birthweight ~ as.factor(Race) + as.factor(Education) + 
                                    as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                    as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                    Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                    BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                    BL.Calc.I + BL.Pl.I + as.factor(Group) + V3.PD.avg + V5.PD.avg)
  )
  
  # Predict Birthweight using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  Birthweight_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- Birthweightmodel_imp$analyses[[j]]$coefficients %>% as.matrix() 
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu), V3.PD.avgSimu, V5.PD.avgSimu))
    
    # Predicted synthetic Y, Birthweight
    pred <- t(param) %*% t(data) %>% t()
    
    Birthweight_preds[, j] <- pred
    
  }
  
  # Generate synthetic Birthweight for each of the m model fits
  # Object to store m sets of generated Birthweight
  Birthweight_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    Birthweight_predict <- Birthweight_preds[, j]
    
    # Residuals from j'th model
    Birthweight_resid <- Birthweightmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(Birthweight_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- Birthweight_predict[i] + Birthweight_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      # It is possible that there are no sums > 0 --> if this is the case, set sample_val = 300 (min of CC real data)
      # sample_val <- ifelse(length(pred_resid_sums_pos) != 0, sample(pred_resid_sums_pos, size = 1), 300) 
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      Birthweight_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic Birthweight, Y (all observed)
  OutcomeSimu <- apply(Birthweight_mgens, 1, sample, size = 1)
  
  
  # Finally, generate synthetic missingness (V3.PD.avg, Birthweight)
  
  # V3.PD.avg (Z1)
  
  # Data subset - include real patient ID
  V3.PD.avg_data_learn <- cbind(PID = real_data$PID, db, Group = Tx, 
                                V3.PD.avg = Post_Random$V3.PD.avg, R_3 = real_data$R_3)
  
  # Model to estimate probability of being observed at visit 3
  # Note: this model is also used to generate synthetic missingness (R_3)
  V3.PD.avg_probobs_model <- glm(as.factor(R_3) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group),
                                 data = V3.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at visit 3 for SIMULATED data
  V3.PD.avgSimu_probobs <- predict.glm(V3.PD.avg_probobs_model,
                                       newdata = cbind(DataSimu,
                                                       Group = TxSimu),
                                       type = "response")
  
  # Generate synthetic R_3
  R_3Simu <- rbinom(n = n_obs, size = 1, prob = V3.PD.avgSimu_probobs)
  
  # Generate V3.PD.avg (Z1) with missingness
  V3.PD.avgSimu_withmiss <- cbind(R_3Simu, V3.PD.avgSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_3Simu == 1, V3.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  # Birthweight (Y)
  
  # Data subset (not really a subset, here we use all variables)
  # V3.PD.avg, Birthweight (the outcome) have missingness
  outcome_data_learn <- cbind(V3.PD.avg_data_learn, 
                              V5.PD.avg = Post_Random$V5.PD.avg,
                              Birthweight = Outcome,
                              R_Y = real_data$R_Y,
                              R_3xZ1 = ifelse(V3.PD.avg_data_learn$R_3 == 1, 
                                              V3.PD.avg_data_learn$V3.PD.avg,
                                              0))
  
  # Estimated probabilities of being observed at final visit for REAL data
  # based on the following model: 
  # Pr(Y = obs | X, A, R_Z1, Z2, R_Z1*Z1)
  
  # Model to estimate probability of being observed at final visit
  Y_probobs_model <- glm(as.factor(R_Y) ~ Age + as.factor(Race) + as.factor(Education) + 
                           as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                           BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                           BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                           BL.Antibio + BL.Bac.vag + as.factor(Group) + 
                           R_3 + V5.PD.avg + R_3xZ1,
                         data = outcome_data_learn,
                         family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at outcome for SIMULATED data
  OutcomeSimu_probobs <- predict.glm(Y_probobs_model,
                                     newdata = cbind(DataSimu,
                                                     Group = TxSimu,
                                                     V3.PD.avg = V3.PD.avgSimu,
                                                     V5.PD.avg = V5.PD.avgSimu,
                                                     # R_3 = ifelse(!is.na(V3.PD.avgSimu), 1, 0), # should be all 1
                                                     # R_3xZ1 = ifelse(R_3 == 1, 
                                                     #                 V3.PD.avgSimu,
                                                     #                  0)  # should be all V3.PD.avg values
                                                     R_3 = 1, # should be all 1
                                                     R_3xZ1 = V3.PD.avgSimu  # should be all V3.PD.avg values
                                     ),
                                     type = "response")
  
  # Generate synthetic R_Y
  R_YSimu = rbinom(n = n_obs, size = 1, prob = OutcomeSimu_probobs)
  
  # Generate Birthweight (Y) with missingness
  OutcomeSimu_withmiss <- cbind(R_YSimu, OutcomeSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_YSimu == 1, OutcomeSimu, NA)) %>%
    pull(withmiss)
  
  
  # Final table of synthetic data
  data_synthetic <- cbind(PID = c(1:nrow(DataSimu)),
                          DataSimu, 
                          Group = as.factor(TxSimu),
                          V3.PD.avg_complete = V3.PD.avgSimu, 
                          V3.PD.avg = V3.PD.avgSimu_withmiss,
                          V5.PD.avg = V5.PD.avgSimu,  # fully obs. in real data
                          Birthweight_complete = OutcomeSimu,
                          Birthweight = OutcomeSimu_withmiss)
  
  return(data_synthetic)
}


## Monotone missingness ----

# IPW
generate1dataset_aipw_mono <- function(real_data, random_seed, n_obs) {
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Output: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(Birthweight)
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                    BL.Bac.vag)) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, 
                    BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, BL.Pl.I))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(Group) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(V3.PD.avg, V5.PD.avg))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Marginal distribution of the covariates are estimated using empirical estimator 
  set.seed(random_seed)
  
  # Estimation of the R-vine model based on original data
  Rvine_dist <- Estimation_Copula(Cov)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copula(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Probabilities = c(0.5, 0.5))
  
  # Simulation of (complete) post-randomization variable at visit 3 (V3.PD.avg)
  
  # Data subset - include real patient ID (V3.PD.avg has missingness)
  V3.PD.avg_data_learn <- cbind(PID = real_data$PID, db, Group = Tx, 
                                V3.PD.avg = Post_Random$V3.PD.avg, R_3 = real_data$R_3)
  
  # Model to estimate probability of being observed at visit 3
  # Note: this model is also used to generate synthetic missingness (R_3)
  V3.PD.avg_probobs_model <- glm(as.factor(R_3) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group),
                                 data = V3.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at visit 3 for REAL data
  V3.PD.avg_data_learn$V3.PD.avg_probobs <-  predict.glm(V3.PD.avg_probobs_model,
                                                         type = "response")
  
  # Weighted model to predict V3.PD.avg
  # Note: weights = 1/p_i, where p_i = prob of being observed (estimated by V3.PD.avg_probobs)
  V3.PD.avg_model <- lm(V3.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + as.factor(Group),
                        data = V3.PD.avg_data_learn,
                        weights = (1/V3.PD.avg_probobs))  # prob of being observed for REAL data
  
  # Generate synthetic V3.PD.avg (fully observed)
  V3.PD.avgSimu <- Simulation_PostRandom_V3(Model = V3.PD.avg_model, Covariates_synth = DataSimu, 
                                            Tx_synth = TxSimu)
  
  # Simulation of post-randomization variable at visit 5 (V5.PD.avg)
  
  # Data subset (V3.PD.avg, V5.PD.avg have missingness)
  V5.PD.avg_data_learn <- cbind(V3.PD.avg_data_learn,
                                V5.PD.avg = Post_Random$V5.PD.avg, R_5 = real_data$R_5)
  
  # First term: Pr(Z2 = obs | X, A, Z1 = obs)
  # Model to estimate probability of being observed at visit 3, given Z1 observed (V3.PD.avg)
  V5.PD.avg_probobs_model <- glm(as.factor(R_5) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg,
                                 data = V5.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Pr(Z2 = obs | X, A, Z1 = obs)
  V5.PD.avg_probobs_z1obs <- predict.glm(V5.PD.avg_probobs_model, type = "response")
  
  # Add Pr(Z2 = obs | X, A, Z1 = obs) to V5.PD.avg_data_learn
  V5.PD.avg_data_learn_z1obs <- V5.PD.avg_data_learn[rownames(V5.PD.avg_data_learn) %in% names(V5.PD.avg_probobs_z1obs), ]
  V5.PD.avg_data_learn_z1obs$V5.PD.avg_probobs_z1obs <- V5.PD.avg_probobs_z1obs
  V5.PD.avg_data_learn <- V5.PD.avg_data_learn %>%
    left_join(V5.PD.avg_data_learn_z1obs, by = colnames(V5.PD.avg_data_learn))
  
  # Calculate probability of being observed at visit 5 for REAL data
  V5.PD.avg_data_learn <- V5.PD.avg_data_learn %>%
    mutate(V5.PD.avg_probobs = case_when(!is.na(V3.PD.avg) ~ V5.PD.avg_probobs_z1obs*V3.PD.avg_probobs,  # Z1 = obs
                                         is.na(V3.PD.avg) ~ 0))  # Z1 = miss --> Z2 must be missing
  
  # Weighted model to predict V5.PD.avg
  # Note: weights = 1/p_i, where p_i = prob of being observed
  V5.PD.avg_model <- lm(V5.PD.avg ~ Age + as.factor(Race) + as.factor(Education) + 
                          as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                          BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                          BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                          BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg,
                        data = V5.PD.avg_data_learn,
                        weights = (1/V5.PD.avg_probobs))  # prob of being observed for REAL data
  
  # Generate synthetic V5.PD.avg
  V5.PD.avgSimu <- Simulation_PostRandom_V5(Model = V5.PD.avg_model, Covariates_synth = DataSimu,
                                            Tx_synth = TxSimu, PostRandom_V3 = V3.PD.avgSimu)
  
  # Simulation of the outcome
  
  # Data subset (not really a subset, here we use all variables)
  # V3.PD.avg, V5.PD.avg, Birthweight (the outcome) have missingness
  outcome_data_learn <- cbind(V5.PD.avg_data_learn,
                              # Birthweight = Outcome,
                              R_Y = real_data$R_Y)
  
  # Estimated probabilities of being observed at final visit for REAL data
  # based on decomp: 
  # Pr(Y = obs | X, A, Z1, Z2) = Pr(Y = obs | X, A, Z1 = obs, Z2 = obs)*Pr(Z2 = obs | X, A, Z1)
  
  # Model to estimate probability of being observed at final visit, given Z1 and Z2 observed (V3.PD.avg, V5.PD.avg)
  Y_probobs_model <- glm(as.factor(R_Y) ~ Age + as.factor(Race) + as.factor(Education) + 
                           as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                           BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                           BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                           BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg + V5.PD.avg,
                         data = outcome_data_learn,
                         family = binomial(link = "logit"))
  
  # Pr(Y = obs | X, A, Z1 = obs, Z2 = obs)
  Y_probobs_z1z2obs <- predict.glm(Y_probobs_model, type = "response")
  
  # Add Pr(Y = obs | X, A, Z1 = obs, Z2 = obs) to outcome_data_learn
  outcome_data_learn_z1z2obs <- outcome_data_learn[rownames(outcome_data_learn) %in% names(Y_probobs_z1z2obs), ]
  outcome_data_learn_z1z2obs$Birthweight_probobs_z1z2obs <- Y_probobs_z1z2obs
  outcome_data_learn <- outcome_data_learn %>%
    left_join(outcome_data_learn_z1z2obs, by = colnames(outcome_data_learn))
  
  # Calculate probability of being observed at final visit for REAL data
  outcome_data_learn <- outcome_data_learn %>%
    mutate(Birthweight_probobs = case_when(!is.na(V5.PD.avg) ~ Birthweight_probobs_z1z2obs*V5.PD.avg_probobs,  # Z1, Z2 = obs
                                           is.na(V5.PD.avg) ~ 0))  # Z1 = miss, Z2 = miss --> Y must be missing  
  
  # Weighted model to predict Birthweight (outcome)
  # Note: weights = 1/p_i, where p_i = prob of being observed (estimated by Y_probobs)
  outcome_model <- lm(Birthweight ~ Age + as.factor(Race) + as.factor(Education) + 
                        as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                        BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                        BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                        BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg + V5.PD.avg,
                      data = outcome_data_learn,
                      weights = (1/(Birthweight_probobs)))  # prob of being observed for REAL data
  
  # Generate synthetic outcome (all observed)
  OutcomeSimu <- Simulation_DataOutcome(Model = outcome_model, Covariates_synth = DataSimu,
                                        Tx_synth = TxSimu, PostRandom_V3 = V3.PD.avgSimu,
                                        PostRandom_V5 = V5.PD.avgSimu)
  
  # Generate missingness (V3.PD.avg, V5.PD.avg, Y)
  
  # V3.PD.avg (Z1):
  
  # Estimated probabilities of being observed at visit 3 for SIMULATED data
  V3.PD.avgSimu_probobs <- predict.glm(V3.PD.avg_probobs_model,
                                       newdata = cbind(DataSimu,
                                                       Group = TxSimu),
                                       type = "response")
  
  # Generate synthetic R_3
  R_3Simu <- rbinom(n = n_obs, size = 1, prob = V3.PD.avgSimu_probobs)
  
  # Generate V3.PD.avg (Z1) with missingness
  V3.PD.avgSimu_withmiss <- cbind(R_3Simu, V3.PD.avgSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_3Simu == 1, V3.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  # V5.PD.avg (Z2):
  
  # Estimated probabilities of being observed at visit 5 for SIMULATED data
  # If Z1 = obs, then prob is predicted using fitted model for R_Z2
  # If Z1 = miss, then prob = 0 (i.e., prob of Z2 being observed = 0 when Z1 is missing)
  datasynth_v5 <- cbind(PID = 1:nrow(DataSimu), DataSimu, Group = TxSimu, 
                        V3.PD.avg = V3.PD.avgSimu, V5.PD.avg = V5.PD.avgSimu,
                        R_3 = R_3Simu, V3.PD.avg_withmiss = V3.PD.avgSimu_withmiss)
  
  # Pr(Z2_synth = obs | X_synth, A_synth, Z1_synth = obs)
  # Restrict to rows with OBSERVED synthetic Z1
  V5.PD.avgsynth_probobs_z1synthobs <- predict.glm(V5.PD.avg_probobs_model, 
                                                   newdata = datasynth_v5[!is.na(datasynth_v5$V3.PD.avg_withmiss), ],  
                                                   type = "response")
  
  # Add Pr(Z2_synth = obs | X_synth, A_synth, Z1_synth = obs) to datasynth_v5
  datasynth_v5_z1obs <- datasynth_v5[rownames(datasynth_v5) %in% names(V5.PD.avgsynth_probobs_z1synthobs), ]
  datasynth_v5_z1obs$V5.PD.avg_probobs_z1obs <- V5.PD.avgsynth_probobs_z1synthobs
  datasynth_v5 <- datasynth_v5 %>%
    left_join(datasynth_v5_z1obs, by = colnames(datasynth_v5))
  
  # Calculate prob. observed for Z2
  datasynth_v5 <- datasynth_v5 %>%
    mutate(V5.PD.avg_probobs = case_when(!is.na(V3.PD.avg_withmiss) ~ V5.PD.avg_probobs_z1obs,
                                         is.na(V3.PD.avg_withmiss) ~ 0))  # Z1 = miss --> Z2 must be missing
  
  # Generate synthetic R_5
  R_5Simu = rbinom(n = n_obs, size = 1, prob = datasynth_v5$V5.PD.avg_probobs)
  
  # Generate V5.PD.avg (Z2) with missingness
  V5.PD.avgSimu_withmiss <- cbind(R_5Simu, V5.PD.avgSimu = datasynth_v5$V5.PD.avg) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_5Simu == 1, V5.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  # Outcome (Birthweight, Y):
  
  # Estimated probabilities of being observed at final visit for SIMULATED data
  # If Z2 = obs (meaning Z1 = obs), then prob is predicted using fitted model for R_Y
  # If Z2 = miss, then prob = 0 (i.e., prob of Y being observed = 0 when Z2 is missing)
  datasynth_outcome <- cbind(datasynth_v5, 
                             R_5 = R_5Simu, V5.PD.avg_withmiss = V5.PD.avgSimu_withmiss, Birthweight = OutcomeSimu)
  
  # Pr(Y_synth = obs | X_synth, A_synth, Z1_synth = obs, Z2_synth = obs)
  # Restrict to rows with OBSERVED synthetic Z1 and OBSERVED synthetic Z2 (but if Z2 obs, then so is Z1)
  Ysynth_probobs_z1z2synthobs <- predict.glm(Y_probobs_model,
                                             newdata = datasynth_outcome[(!is.na(datasynth_outcome$V5.PD.avg_withmiss)), ],  
                                             type = "response")
  
  # Pr(Y_synth = obs | X_synth, A_synth, Z1_synth = obs, Z2_synth = obs) to datasynth_outcome
  datasynth_outcome_z1z2obs <- datasynth_outcome[rownames(datasynth_outcome) %in% names(Ysynth_probobs_z1z2synthobs), ]
  datasynth_outcome_z1z2obs$Birthweight_probobs_z1z2obs <- Ysynth_probobs_z1z2synthobs
  datasynth_outcome <- datasynth_outcome %>%
    left_join(datasynth_outcome_z1z2obs, by = colnames(datasynth_outcome))
  
  # Calculate prob. observed for Y
  datasynth_outcome <- datasynth_outcome %>%
    mutate(Birthweight_probobs = case_when(!is.na(V5.PD.avg_withmiss) ~ Birthweight_probobs_z1z2obs,
                                           is.na(V5.PD.avg_withmiss) ~ 0))  # Z2 = miss --> Y must be missing
  
  # Generate synthetic R_Y
  R_YSimu = rbinom(n = n_obs, size = 1, prob = datasynth_outcome$Birthweight_probobs)
  
  # Generate Birthweight (Y) with missingness
  OutcomeSimu_withmiss <- cbind(R_YSimu, OutcomeSimu = datasynth_outcome$Birthweight) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_YSimu == 1, OutcomeSimu, NA)) %>%
    pull(withmiss)
  
  # Final table of synthetic data generated via R-vine copula + execution models
  data_synthetic <- cbind(PID = c(1:nrow(DataSimu)),
                          DataSimu, 
                          Group = as.factor(TxSimu),
                          V3.PD.avg_complete = V3.PD.avgSimu, 
                          V3.PD.avg = V3.PD.avgSimu_withmiss,
                          V5.PD.avg_complete = V5.PD.avgSimu,
                          V5.PD.avg = V5.PD.avgSimu_withmiss,
                          Birthweight_complete = OutcomeSimu,
                          Birthweight = OutcomeSimu_withmiss)
  
  return(data_synthetic)
}

# MI
generate1dataset_mi_mono <- function(real_data, random_seed, n_obs){
  # Input: real_data is the real data dataframe, 
  #        random_seed is a number for the random seed,
  #        n_obs is the number of observations (i.e., rows) to generate
  # Output: dataframe of synthetic data
  
  # Definition of the outcome vector 
  Outcome <- real_data %>% dplyr::select(Birthweight)
  
  # Definition of the matrix of discrete covariates (at baseline)
  Cov_Discrete <- real_data %>% 
    dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, BL.Anti.inf, BL.Antibio, 
                    BL.Bac.vag)) %>%
    lapply(., as.factor) %>%
    as.data.frame()
  
  # Definition of the matrix of continuous covariates (at baseline)
  Cov_Cont <- real_data %>%
    dplyr::select(c(Age, N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, BL..PD.4, 
                    BL..PD.5, BL.CAL.avg, BL..CAL.2, BL..CAL.3, BL.Calc.I, BL.Pl.I))
  
  # Definition of the matrix of covariates (at baseline)
  Cov <- c(Cov_Discrete, Cov_Cont) %>% as.data.frame()
  
  # Definition of treatment assignment vector 
  Tx <- real_data %>% dplyr::select(Group) %>% unlist() %>% as.factor()
  
  # Definition of the matrix of post-randomization variables
  Post_Random <- real_data %>%
    dplyr::select(c(V3.PD.avg, V5.PD.avg))
  
  # Definition of the considered data
  data_allvar <- cbind(Cov, Tx, Post_Random, Outcome)
  
  # Definition of baseline data and outcome
  db <- cbind(Cov, Outcome)
  
  # Marginal distribution of the covariates are estimated using empirical estimator 
  set.seed(random_seed)
  
  # Estimation of the R-vine model based on original data
  Rvine_dist <- Estimation_Copula(Cov)
  
  # Simulation of virtual patients based on the R-vine model and empirical distribution (of the original data)
  DataSimu <- Simulation_Copula(n_obs, Rvine_dist, Cov)  # n_obs = number of rows in original data
  
  # Simulation of synthetic treatment allocation
  TxSimu <- Simulation_Treatment(N = n_obs, Probabilities = c(0.5, 0.5))
  
  # Post-randomization variables have missingness, perform MI to get m complete data sets
  Imp_data <- real_data %>% dplyr::select(c(Race, Education, Public.Asstce, Prev.preg, 
                                            BL.Anti.inf, BL.Antibio, BL.Bac.vag, Age, 
                                            N.qualifying.teeth, BL.GE, BL..BOP, BL.PD.avg, 
                                            BL..PD.4, BL..PD.5, BL.CAL.avg, BL..CAL.2, 
                                            BL..CAL.3, BL.Calc.I, BL.Pl.I, Group, 
                                            V3.PD.avg, V5.PD.avg, Birthweight))
  
  # Calculate m, where m ~ max(prop. of missingness)*100
  m <- round(colMeans(is.na(Imp_data)) %>% max()*100)  # m ~ prop. of missingness*100
  
  # Perform multiple imputation by chained equations (mice)
  imp <- mice(data = Imp_data, m = m, maxit = 10, seed = random_seed, print = FALSE)
  
  
  # Simulation of post-randomization variable at visit 3 (V3.PD.avg)
  
  # First, fit execution model to each of the m imputed data sets
  V3.PD.avgmodel_imp <- with(imp,
                             lm(V3.PD.avg ~ as.factor(Race) + as.factor(Education) + 
                                  as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                  as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                  Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                  BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                  BL.Calc.I + BL.Pl.I + as.factor(Group))
  )
  
  # Predict V3.PD.avg using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  V3.PD.avg_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- V3.PD.avgmodel_imp$analyses[[j]]$coefficients %>% as.matrix()
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu)))
    
    # Predicted synthetic Z1, V3.PD.avg
    pred <- t(param) %*% t(data) %>% t()
    
    V3.PD.avg_preds[, j] <- pred
    
  }
  
  # Generate synthetic V3.PD.avg for each of the m model fits
  # Object to store m sets of generated V3.PD.avg
  V3.PD.avg_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    V3.PD.avg_predict <- V3.PD.avg_preds[, j]
    
    # Residuals from j'th model
    V3.PD.avg_resid <- V3.PD.avgmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(V3.PD.avg_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- V3.PD.avg_predict[i] + V3.PD.avg_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      V3.PD.avg_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic V3.PD.avg, Z^s_1 (all observed)
  V3.PD.avgSimu <- apply(V3.PD.avg_mgens, 1, sample, size = 1)
  
  
  # Simulation of post-randomization variable at visit 5 (V5.PD.avg)
  
  # First, fit execution model to each of the m imputed data sets
  V5.PD.avgmodel_imp <- with(imp,
                             lm(V5.PD.avg ~ as.factor(Race) + as.factor(Education) + 
                                  as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                  as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                  Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                  BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                  BL.Calc.I + BL.Pl.I + as.factor(Group) + V3.PD.avg)
  )
  
  # Predict V5.PD.avg using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  V5.PD.avg_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- V5.PD.avgmodel_imp$analyses[[j]]$coefficients %>% as.matrix()
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu), V3.PD.avgSimu))
    
    # Predicted synthetic Z2, V5.PD.avg
    pred <- t(param) %*% t(data) %>% t()
    
    V5.PD.avg_preds[, j] <- pred
    
  }
  
  # Generate synthetic V5.PD.avg for each of the m model fits
  # Object to store m sets of generated V5.PD.avg
  V5.PD.avg_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    V5.PD.avg_predict <- V5.PD.avg_preds[, j]
    
    # Residuals from j'th model
    V5.PD.avg_resid <- V5.PD.avgmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(V5.PD.avg_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- V5.PD.avg_predict[i] + V5.PD.avg_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      V5.PD.avg_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic V5.PD.avg, Z^s_2 (all observed)
  V5.PD.avgSimu <- apply(V5.PD.avg_mgens, 1, sample, size = 1)
  
  
  # Simulation of the outcome for each virtual patients
  
  # First, fit execution model to each of the m imputed data sets
  Birthweightmodel_imp <- with(imp,
                               lm(Birthweight ~ as.factor(Race) + as.factor(Education) + 
                                    as.factor(Public.Asstce) + as.factor(Prev.preg) + 
                                    as.factor(BL.Anti.inf) + as.factor(BL.Antibio) + as.factor(BL.Bac.vag) +
                                    Age + N.qualifying.teeth + BL.GE + BL..BOP + BL.PD.avg + 
                                    BL..PD.4 + BL..PD.5 + BL.CAL.avg + BL..CAL.2 + BL..CAL.3 +
                                    BL.Calc.I + BL.Pl.I + as.factor(Group) + V3.PD.avg + V5.PD.avg)
  )
  
  # Predict Birthweight using each of the m model fits
  # Object to store predictions 
  # (rows represent synthetic patients/obs, columns represent predictions for the jth imputed data set)
  Birthweight_preds <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m fitted models
    
    # Model parameters
    param <- Birthweightmodel_imp$analyses[[j]]$coefficients %>% as.matrix() 
    
    # synthetic X,A with column of 1's for intercept
    data <- model.matrix(~ ., data = cbind(DataSimu, as.factor(TxSimu), V3.PD.avgSimu, V5.PD.avgSimu))
    
    # Predicted synthetic Y, Birthweight
    pred <- t(param) %*% t(data) %>% t()
    
    Birthweight_preds[, j] <- pred
    
  }
  
  # Generate synthetic Birthweight for each of the m model fits
  # Object to store m sets of generated Birthweight
  Birthweight_mgens <- matrix(data = NA, nrow = n_obs, ncol = m) %>% as.data.frame()
  
  for(j in 1:m) {  # iterate through all columns i.e., all m model fits
    
    # Predicted values using j'th model
    Birthweight_predict <- Birthweight_preds[, j]
    
    # Residuals from j'th model
    Birthweight_resid <- Birthweightmodel_imp$analyses[[j]]$residuals
    
    for (i in 1:nrow(Birthweight_mgens)) {
      # Get all possible values of pred + resid for the i'th observation
      pred_resid_sums <- Birthweight_predict[i] + Birthweight_resid
      
      # All pred + resid >= 0
      pred_resid_sums_pos <- pred_resid_sums[pred_resid_sums >= 0]
      
      # Randomly sample from non-negative sum values
      # It is possible that there are no sums > 0 --> if this is the case, set sample_val = 300 (min of CC real data)
      # sample_val <- ifelse(length(pred_resid_sums_pos) != 0, sample(pred_resid_sums_pos, size = 1), 300) 
      sample_val = sample(pred_resid_sums_pos, size = 1)
      
      # Save sample
      Birthweight_mgens[i, j] <- sample_val
    }
    
  }
  
  # Generate final synthetic Birthweight, Y (all observed)
  OutcomeSimu <- apply(Birthweight_mgens, 1, sample, size = 1)
  
  
  # Finally, generate synthetic missingness (V3.PD.avg, V5.PD.avg, Birthweight)
  
  # V3.PD.avg (Z1)
  
  # Data subset - include real patient ID
  V3.PD.avg_data_learn <- cbind(PID = real_data$PID, db, Group = Tx, 
                                V3.PD.avg = Post_Random$V3.PD.avg, R_3 = real_data$R_3)
  
  # Model to estimate probability of being observed at visit 3
  # Note: this model is also used to generate synthetic missingness (R_3)
  V3.PD.avg_probobs_model <- glm(as.factor(R_3) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group),
                                 data = V3.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at visit 3 for SIMULATED data
  V3.PD.avgSimu_probobs <- predict.glm(V3.PD.avg_probobs_model,
                                       newdata = cbind(DataSimu,
                                                       Group = TxSimu),
                                       type = "response")
  
  # Generate synthetic R_3
  R_3Simu <- rbinom(n = n_obs, size = 1, prob = V3.PD.avgSimu_probobs)
  
  # Generate V3.PD.avg (Z1) with missingness
  V3.PD.avgSimu_withmiss <- cbind(R_3Simu, V3.PD.avgSimu) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_3Simu == 1, V3.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  # V5.PD.avg (Z2)
  
  # Data subset (V3.PD.avg, V5.PD.avg have missingness)
  V5.PD.avg_data_learn <- cbind(V3.PD.avg_data_learn,
                                V5.PD.avg = Post_Random$V5.PD.avg, R_5 = real_data$R_5)
  
  # Estimated probabilities of being observed at visit 5 for REAL data
  # based on decomp: Pr(Z2 = obs | X, A, Z1) = Pr(Z2 = obs | X, A, Z1 = obs)*Pr(Z1 = obs | X, A)
  #                                                + Pr(Z2 = obs | X, A, Z1 = miss)*Pr(Z1 = miss | X, A)
  #                                          = Pr(Z2 = obs | X, A, Z1 = obs)*Pr(Z1 = obs | X, A)
  
  # First term: Pr(Z2 = obs | X, A, Z1 = obs)
  # Model to estimate probability of being observed at visit 5, given Z1 observed (V3.PD.avg)
  V5.PD.avg_probobs_model <- glm(as.factor(R_5) ~ Age + as.factor(Race) + as.factor(Education) + 
                                   as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                                   BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                                   BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                                   BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg,
                                 data = V5.PD.avg_data_learn,
                                 family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at visit 5 for SIMULATED data
  # If Z1 = obs, then prob is predicted using fitted model for R_Z2
  # If Z1 = miss, then prob = 0 (i.e., prob of Z2 being observed = 0 when Z1 is missing)
  datasynth_v5 <- cbind(PID = 1:nrow(DataSimu), DataSimu, Group = TxSimu, 
                        V3.PD.avg = V3.PD.avgSimu, V5.PD.avg = V5.PD.avgSimu,
                        R_3 = R_3Simu, V3.PD.avg_withmiss = V3.PD.avgSimu_withmiss)
  
  # Pr(Z2_synth = obs | X_synth, A_synth, Z1_synth = obs)
  # Restrict to rows with OBSERVED synthetic Z1
  V5.PD.avgsynth_probobs_z1synthobs <- predict.glm(V5.PD.avg_probobs_model, 
                                                   newdata = datasynth_v5[!is.na(datasynth_v5$V3.PD.avg_withmiss), ],  
                                                   type = "response")
  
  # Add Pr(Z2_synth = obs | X_synth, A_synth, Z1_synth = obs) to datasynth_wk96
  datasynth_v5_z1obs <- datasynth_v5[rownames(datasynth_v5) %in% names(V5.PD.avgsynth_probobs_z1synthobs), ]
  datasynth_v5_z1obs$V5.PD.avg_probobs_z1obs <- V5.PD.avgsynth_probobs_z1synthobs
  datasynth_v5 <- datasynth_v5 %>%
    left_join(datasynth_v5_z1obs, by = colnames(datasynth_v5))
  
  # Calculate prob. observed for Z2
  datasynth_v5 <- datasynth_v5 %>%
    mutate(V5.PD.avg_probobs = case_when(!is.na(V3.PD.avg_withmiss) ~ V5.PD.avg_probobs_z1obs,
                                         is.na(V3.PD.avg_withmiss) ~ 0))  # Z1 = miss --> Z2 must be missing
  
  # Generate synthetic R_5
  R_5Simu = rbinom(n = n_obs, size = 1, prob = datasynth_v5$V5.PD.avg_probobs)
  
  # Generate V5.PD.avg (Z2) with missingness
  V5.PD.avgSimu_withmiss <- cbind(R_5Simu, V5.PD.avgSimu = datasynth_v5$V5.PD.avg) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_5Simu == 1, V5.PD.avgSimu, NA)) %>%
    pull(withmiss)
  
  # Birthweight (Y)
  
  # Data subset (not really a subset, here we use all variables)
  # V3.PD.avg, V5.PD.avg, Birthweight (the outcome) have missingness
  outcome_data_learn <- cbind(V5.PD.avg_data_learn,
                              # Birthweight = Outcome,
                              R_Y = real_data$R_Y)
  
  # Estimated probabilities of being observed at final visit for REAL data
  # based on decomp: 
  # Pr(Y = obs | X, A, Z1, Z2) = Pr(Y = obs | X, A, Z1 = obs, Z2 = obs)*Pr(Z2 = obs | X, A, Z1)
  
  # Model to estimate probability of being observed at final visit, given Z1 and Z2 observed (V3.PD.avg, V5.PD.avg)
  Y_probobs_model <- glm(as.factor(R_Y) ~ Age + as.factor(Race) + as.factor(Education) + 
                           as.factor(Public.Asstce) + as.factor(Prev.preg) + N.qualifying.teeth + 
                           BL.GE + BL..BOP + BL.PD.avg + BL..PD.4 + BL..PD.5 + BL.CAL.avg + 
                           BL..CAL.2 + BL..CAL.3 + BL.Calc.I + BL.Pl.I + BL.Anti.inf + 
                           BL.Antibio + BL.Bac.vag + as.factor(Group) + V3.PD.avg + V5.PD.avg,
                         data = outcome_data_learn,
                         family = binomial(link = "logit"))
  
  # Estimated probabilities of being observed at final visit for SIMULATED data
  # If Z2 = obs (meaning Z1 = obs), then prob is predicted using fitted model for R_Y
  # If Z2 = miss, then prob = 0 (i.e., prob of Y being observed = 0 when Z2 is missing)
  datasynth_outcome <- cbind(datasynth_v5, 
                             R_5 = R_5Simu, V5.PD.avg_withmiss = V5.PD.avgSimu_withmiss, Birthweight = OutcomeSimu)
  
  # Pr(Y_synth = obs | X_synth, A_synth, Z1_synth = obs, Z2_synth = obs)
  # Restrict to rows with OBSERVED synthetic Z1 and OBSERVED synthetic Z2 (but if Z2 obs, then so is Z1)
  Ysynth_probobs_z1z2synthobs <- predict.glm(Y_probobs_model,
                                             newdata = datasynth_outcome[(!is.na(datasynth_outcome$V5.PD.avg_withmiss)), ],  
                                             type = "response")
  
  # Pr(Y_synth = obs | X_synth, A_synth, Z1_synth = obs, Z2_synth = obs) to datasynth_outcome
  datasynth_outcome_z1z2obs <- datasynth_outcome[rownames(datasynth_outcome) %in% names(Ysynth_probobs_z1z2synthobs), ]
  datasynth_outcome_z1z2obs$Birthweight_probobs_z1z2obs <- Ysynth_probobs_z1z2synthobs
  datasynth_outcome <- datasynth_outcome %>%
    left_join(datasynth_outcome_z1z2obs, by = colnames(datasynth_outcome))
  
  # Calculate prob. observed for Y
  datasynth_outcome <- datasynth_outcome %>%
    mutate(Birthweight_probobs = case_when(!is.na(V5.PD.avg_withmiss) ~ Birthweight_probobs_z1z2obs,
                                           is.na(V5.PD.avg_withmiss) ~ 0))  # Z2 = miss --> Y must be missing
  
  # Generate synthetic R_Y
  R_YSimu = rbinom(n = n_obs, size = 1, prob = datasynth_outcome$Birthweight_probobs)
  
  # Generate Birthweight (Y) with missingness
  OutcomeSimu_withmiss <- cbind(R_YSimu, OutcomeSimu = datasynth_outcome$Birthweight) %>% 
    as.data.frame() %>%
    mutate(withmiss = if_else(R_YSimu == 1, OutcomeSimu, NA)) %>%
    pull(withmiss)
  
  # Final table of synthetic data generated via R-vine copula + execution models
  data_synthetic <- cbind(PID = c(1:nrow(DataSimu)),
                          DataSimu,
                          Group = as.factor(TxSimu),
                          V3.PD.avg_complete = V3.PD.avgSimu,
                          V3.PD.avg = V3.PD.avgSimu_withmiss,
                          V5.PD.avg_complete = V5.PD.avgSimu,
                          V5.PD.avg = V5.PD.avgSimu_withmiss,
                          Birthweight_complete = OutcomeSimu,
                          Birthweight = OutcomeSimu_withmiss)
  
  return(data_synthetic)
}

# Metrics ----
# Function for calculating 1-KS for all variables, for a given dataset (i.e., method)
ks.stat <- function(Synthetic_Data, Real_Data) {
  # This function takes as input a synthetic data set and the real data set
  # and returns a dataframe with the 1-KS value and the variable name.
  # This function uses the ks.test() function from the dgof package
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store 1-KS values
  vals <- matrix(data = NA, nrow = ncol(Synthetic_Data), ncol = 2) %>% as.data.frame()
  colnames(vals) <- c("Variable", "KS Statistic")
  
  # Perform KS test, then store 1 - KS statistic for each variable
  for(i in 1:ncol(Synthetic_Data)) {
    # i'th variable
    col_name <- names(Synthetic_Data)[i]
    
    # Save variable name
    vals[i, 1] <- col_name
    
    # Perform KS test for i'th variable
    ks_test <- ks.test(Synthetic_Data[, col_name], Real_Data[, col_name], alternative = "two.sided")
    
    # KS statistic
    ks_stat <- ks_test$statistic %>% unname()
    
    # Save 1-KS statistic
    vals[i, 2] <- 1 - ks_stat
  }
  
  # Return df of 1 - KS statistics
  return(vals)
  
}

# Calculate this by hand:
# calculate difference between synthetic and real proportions of each category of a given variable,
# sum the absolute value of all differences,
# then divide the sum by 2.
# Note: this function works for both binary variables and categorical variables with more than 2 levels.
tvd.calc <- function(Synthetic_Data, Real_Data, Var) {
  # Synthetic_Data is a dataframe containing synthetic data
  # Real_Data is a dataframe containing real data
  # Var is a character string of the variable name
  
  synthetic_prop <- table(Synthetic_Data[!is.na(Synthetic_Data[, Var]), Var])/nrow(Synthetic_Data[!is.na(Synthetic_Data[, Var]), ])
  real_prop <- table(Real_Data[!is.na(Real_Data[, Var]), Var])/nrow(Real_Data[!is.na(Real_Data[, Var]), ])
  tvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  
  return(tvd)
}

# Function to return all tvd stats for all discrete variables in a given data set
tvd.stat <- function(Synthetic_Data, Real_Data) {
  # This function takes as input a synthetic data set and the real data set
  # and returns a dataframe with the 1 - TVD value and the variable name.
  # This function uses the previously-declared function, tvd.calc
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store 1 - TVD values
  vals <- matrix(data = NA, nrow = ncol(Synthetic_Data), ncol = 2) %>% as.data.frame()
  colnames(vals) <- c("Variable", "TVD Statistic")
  
  # Calculate TVD, then store 1 - TVD statistic for each variable
  for(i in 1:ncol(Synthetic_Data)) {
    # i'th variable
    col_name <- names(Synthetic_Data)[i]
    
    # Save variable name
    vals[i, 1] <- col_name
    
    # Save 1 - TVD statistic
    vals[i, 2] <- 1 - tvd.calc(Synthetic_Data, Real_Data, col_name)
  }
  
  # Return df of 1 - KS statistics
  return(vals)
  
}

# Normalized difference between correlation of two given continuous variables in 
# the real v. synthetic data
# Score = 1 - (0.5 * |Corr_synth - Corr_real|) --> "similarity score" due to  1 - (value)
# We will use Spearman correlation, not Pearson
Corr.Sim.Score.Spearman <- function(Synthetic_Data, Real_Data) {
  # Calculate correlation for all combinations of pairs of continuous variables
  # Returns dataframe of pairs of variables, correlation in synthetic and real data, and similarity score
  
  # Throw error if Synthetic_Data and Real_Data have different number of columns
  if(ncol(Synthetic_Data) != ncol(Real_Data)) {
    stop("Number of columns in synthetic and real data frames should be equal.")
  }
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Calculate correlations in synthetic data
  Corr_synth <- Synthetic_Data %>%
    as.matrix() %>%
    cor(use = "everything", method = "spearman") %>%
    as.data.frame %>%
    rownames_to_column(var = 'var1') %>%
    gather(var2, value, -var1) %>%
    rename(corr_synth = value)
  
  # Deal with missing values by using only complete data
  for(i in 1:nrow(Corr_synth)) {
    if(is.na(Corr_synth[i, "corr_synth"])) {
      Corr_synth[i, "corr_synth"] <- cor(Synthetic_Data[, Corr_synth[i, "var1"]],
                                         Synthetic_Data[, Corr_synth[i, "var2"]],
                                         use = "complete.obs",
                                         method = "spearman")
    }
  }
  
  # Calculate correlations in real data
  Corr_real <- Real_Data %>%
    as.matrix() %>%
    cor(use = "everything", method = "spearman") %>%
    as.data.frame %>%
    rownames_to_column(var = 'var1') %>%
    gather(var2, value, -var1) %>%
    rename(corr_real = value)
  
  # Deal with missing values by using only complete data
  for(i in 1:nrow(Corr_real)) {
    if(is.na(Corr_real[i, "corr_real"])) {
      Corr_real[i, "corr_real"] <- cor(Real_Data[, Corr_real[i, "var1"]],
                                       Real_Data[, Corr_real[i, "var2"]],
                                       use = "complete.obs",
                                       method = "spearman")
    }
  }
  
  # Join tables so all correlation values are in same table
  Corr_all <- left_join(Corr_synth, Corr_real, by = c("var1", "var2")) %>%
    # Remove variances (i.e., corr between a variable and itself)
    filter(corr_real != 1 & corr_synth != 1) %>%
    # Remove duplicate pairs
    mutate(var_order = paste(var1, var2) %>%
             strsplit(split = ' ') %>%
             map_chr( ~ sort(.x) %>% 
                        paste(collapse = ' '))) %>%
    mutate(cnt = 1) %>%
    group_by(var_order) %>%
    mutate(cumsum = cumsum(cnt)) %>%
    filter(cumsum != 2) %>%
    ungroup %>%
    select(-var_order, -cnt, -cumsum) %>%
    # Calculate normalized similarity score
    mutate(score = 1 - 0.5*(abs(corr_synth - corr_real)))
  
  # Return final table of results
  return(Corr_all)
}

# Function to return all bivariate tvd stats (i.e. 2-way contingency table diffs) THIS DOESNT WORK YET!!!!!
# for all discrete variables in a given data set
Contingency.Sim.stat <- function(Synthetic_Data, Real_Data) {
  # This function takes as input a synthetic data set and the real data set
  # and returns a dataframe with the contingency similarity score (1 - bivariate TVD) and the variable name.
  # This function uses the previously-declared function, bivariatetvd.calc
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store contingency similarity score values
  # Note: number of rows (i.e., values) = n(n-1)/2
  vals <- matrix(data = NA, 
                 nrow = ncol(Synthetic_Data)*(ncol(Synthetic_Data) - 1)/2, 
                 ncol = 3) %>% as.data.frame()
  colnames(vals) <- c("Var1", "Var2", "Stat")
  
  # Fill in values for Var1 and Var2 columns
  varlist <- names(Synthetic_Data)
  df <- expand.grid(varlist, varlist) %>% 
    mutate(combo = paste(Var1, Var2)) %>% 
    group_by(combo) %>% 
    unique()
  # Unique combos of var 1 and var 2
  unique_combos <- df %>% 
    group_by(combo) %>%
    dplyr::select(combo) %>% 
    apply(1, function(x) paste(sort(unlist(strsplit(x, " "))), collapse = " ")) %>% 
    unique() %>%
    as.data.frame()
  colnames(unique_combos) <- c("combo")
  
  # Var 1 and Var 2 as separate columns, and as combo in 1 column
  df <- df %>% 
    right_join(unique_combos, by = "combo") %>% 
    filter(Var1 != Var2)
  
  vals[, 1] <- df[, 1]
  vals[, 2] <- df[, 2]
  
  # Calculate bivariate TVD, then store 1 - bivariate TVD statistic for each variable
  for(i in 1:nrow(vals)) {
    var1 <- vals[i, 1]
    var2 <- vals[i, 2]
    
    vals[i, 3] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data,
                                        Real_Data = Real_Data,
                                        Var1 = var1, Var2 = var2)
  }
  
  # Return df of 1 - bivariate TVD statistics
  return(vals)
  
}

# Function to capture all bivariate comparisons
# Takes as input Synthetic_Data, Real_Data
# If var 1 is continuous and var 2 is continuous, calculate correlations (Spearman and Pearson)
# If var 1 is continuous and var 2 is discrete, or vice versa, 
# bin the continous variable, then calculate contingency score
# If var 1 is discrete and var 2 is discrete, calculate contingency score
# This function calls the previously-defined function: bivariatetvd.calc

# Calculate this by hand:
# calculate difference between synthetic and real proportions of each combination of categories
# of two given variables,
# sum the absolute value of all differences,
# then divide the sum by 2.
bivariatetvd.calc <- function(Synthetic_Data, Real_Data, Var1, Var2) {
  # Synthetic_Data is a dataframe containing synthetic data
  # Real_Data is a dataframe containing real data
  # Var1 and Var2 are character strings of variable names
  
  # Need to handle the case that level 70 of karnof was dropped/missing
  if(Var1 == "karnof" & nlevels(Synthetic_Data[, Var1]) == 3) {
    synthetic_prop <- table(Synthetic_Data[, Var1] %>% unlist(), 
                            Synthetic_Data[, Var2] %>% unlist())/nrow(Synthetic_Data)
    # Add prop = 0 for karnof = 70
    synthetic_prop <- rbind(c(0, 0), synthetic_prop)
    real_prop <- table(Real_Data[, Var1] %>% unlist(), 
                       Real_Data[, Var2] %>% unlist())/nrow(Real_Data)
    bivariatetvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  }
  
  else if(Var2 == "karnof" & nlevels(Synthetic_Data[, Var2]) == 3) {
    synthetic_prop <- table(Synthetic_Data[, Var1] %>% unlist(), 
                            Synthetic_Data[, Var2] %>% unlist())/nrow(Synthetic_Data)
    # Add prop = 0 for karnof = 70
    synthetic_prop <- cbind(c(0, 0), synthetic_prop)
    real_prop <- table(Real_Data[, Var1] %>% unlist(), 
                       Real_Data[, Var2] %>% unlist())/nrow(Real_Data)
    bivariatetvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  }
  
  else {
    synthetic_prop <- table(Synthetic_Data[, Var1] %>% unlist(), 
                            Synthetic_Data[, Var2] %>% unlist())/nrow(Synthetic_Data)
    real_prop <- table(Real_Data[, Var1] %>% unlist(), 
                       Real_Data[, Var2] %>% unlist())/nrow(Real_Data)
    bivariatetvd <- 0.5 * sum(abs(synthetic_prop - real_prop))
  }
  
  return(bivariatetvd)
}

bivar.metrics <- function(Synthetic_Data, Real_Data) {
  
  # Throw error if Synthetic_Data and Real_Data have different number of columns
  if(ncol(Synthetic_Data) != ncol(Real_Data)) {
    stop("Number of columns in synthetic and real data frames should be equal.")
  }
  
  # Throw error if column names of Synthetic_Data and Real_Data are not the same
  if(length(intersect(colnames(Synthetic_Data), colnames(Real_Data))) != ncol(Synthetic_Data)) {
    stop("Column names should be the same in synthetic and real data frames.")
  }
  
  # Initialize dataframe to store similarity score values
  # Note: number of rows (i.e., values) = n(n-1)/2
  vals <- matrix(data = NA, 
                 nrow = ncol(Synthetic_Data)*(ncol(Synthetic_Data) - 1)/2, 
                 ncol = 5) %>% as.data.frame()
  colnames(vals) <- c("Var1", "Var2", "Corr_Score_Spearman", "Corr_Score_Pearson", "Contin_Score")
  
  # Fill in values for Var1 and Var2 columns
  varlist <- names(Synthetic_Data)
  df <- expand.grid(varlist, varlist) %>% 
    mutate(combo = paste(Var1, Var2)) %>% 
    group_by(combo) %>% 
    unique()
  # Unique combos of var 1 and var 2
  unique_combos <- df %>% 
    group_by(combo) %>%
    dplyr::select(combo) %>% 
    apply(1, function(x) paste(sort(unlist(strsplit(x, " "))), collapse = " ")) %>% 
    unique() %>%
    as.data.frame()
  colnames(unique_combos) <- c("combo")
  
  # Var 1 and Var 2 as separate columns, and as combo in 1 column
  df <- df %>% 
    right_join(unique_combos, by = "combo") %>% 
    filter(Var1 != Var2)
  
  vals[, 1] <- df[, 1]
  vals[, 2] <- df[, 2]
  
  # Calculate scores
  for(i in 1:nrow(vals)) {
    var1 <- vals[i, 1] %>% as.character()
    var2 <- vals[i, 2] %>% as.character()
    
    # If var1 and var2 are both continuous, calculate Spearman and Pearson correlation
    if((is.numeric(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Synthetic_Data[, var2])) & 
        is.numeric(unlist(Real_Data[, var1]))& is.numeric(unlist(Real_Data[, var2])))) {
      
      # Spearman
      corr_spearman_synth <- cor(x = Synthetic_Data[, var1], 
                                 y = Synthetic_Data[, var2],
                                 use = "complete.obs",
                                 method = "spearman")
      corr_spearman_real <- cor(x = Real_Data[, var1], 
                                y = Real_Data[, var2],
                                use = "complete.obs",
                                method = "spearman")
      vals[i, 3] = 1 - 0.5*(abs(corr_spearman_synth - corr_spearman_real))
      
      # Pearson
      corr_pearson_synth <- cor(x = Synthetic_Data[, var1], 
                                y = Synthetic_Data[, var2],
                                use = "complete.obs",
                                method = "pearson")
      corr_pearson_real <- cor(x = Real_Data[, var1], 
                               y = Real_Data[, var2],
                               use = "complete.obs",
                               method = "pearson")
      vals[i, 4] = 1 - 0.5*(abs(corr_pearson_synth - corr_pearson_real))
    }
    
    # If var1 and var2 are both discrete, calculate contingency score
    else if((is.factor(unlist(Synthetic_Data[, var1])) & is.factor(unlist(Synthetic_Data[, var2])) & 
             is.factor(unlist(Real_Data[, var1])) & is.factor(unlist(Real_Data[, var2])))) {
      
      # Calculate and store contingency score
      vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data,
                                          Real_Data = Real_Data,
                                          Var1 = var1, 
                                          Var2 = var2)
    }
    
    # If one of var1 and var2 is continuous and the other is discrete,
    # bin the continuous variable and treat as categorical,
    # then calculate contingency score.
    else if((is.numeric(unlist(Synthetic_Data[, var1])) & is.factor(unlist(Synthetic_Data[, var2])) & 
             is.numeric(unlist(Real_Data[, var1])) & is.factor(unlist(Real_Data[, var2]))) |
            (is.factor(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Synthetic_Data[, var2])) & 
             is.factor(unlist(Real_Data[, var1])) & is.numeric(unlist(Real_Data[, var2])))) {
      
      # Identify the continuous variable
      # var1 is continuous
      if(is.numeric(unlist(Synthetic_Data[, var1])) & is.numeric(unlist(Real_Data[, var1]))) {
        
        # Bin this variable (using quantiles)
        Synthetic_Data_bin <- Synthetic_Data %>%
          mutate(cont_binned = cut(Synthetic_Data[, var1] %>% unlist(), 4))
        
        Real_Data_bin <- Real_Data %>%
          mutate(cont_binned = cut(Real_Data[, var1] %>% unlist(), 4))
        
        vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data_bin,
                                            Real_Data = Real_Data_bin,
                                            Var1 = "cont_binned", 
                                            Var2 = var2)
      }
      # var2 is continuous
      else {
        # Bin this variable (using quantiles)
        Synthetic_Data_bin <- Synthetic_Data %>%
          mutate(cont_binned = cut(Synthetic_Data[, var2] %>% unlist(), 4))
        
        Real_Data_bin <- Real_Data %>%
          mutate(cont_binned = cut(Real_Data[, var2] %>% unlist(), 4))
        
        vals[i, 5] <- 1 - bivariatetvd.calc(Synthetic_Data = Synthetic_Data_bin,
                                            Real_Data = Real_Data_bin,
                                            Var1 = var1, 
                                            Var2 = "cont_binned")
      }
    }
  }
  
  return(vals)
}




# Run 1 iteration of generating synthetic data ----
## Scenario 4 ----
# Scenario 4A (non-monotone x MAR x 25% missing x strong mechanism)

# CC: All Stage
simdata_scen4A_cc_preproc <- generate1dataset_cc_preproc_retrycopula(real_data = data_simmiss_scen4A,
                                                                     random_seed = 20250512,
                                                                     n_obs = nrow(data_simmiss_scen4A))

# IPW

# Missingness indicator
simdata_scen4A_ipw_missind <- generate1dataset_aipw_nonmono_missind(real_data = data_simmiss_scen4A,
                                                                    random_seed = 20250512,
                                                                    n_obs = nrow(data_simmiss_scen4A))

# MI 
# (when generating synthetic missingness, use missingness indicator method)
simdata_scen4A_mi <- generate1dataset_mi_nonmono(real_data = data_simmiss_scen4A,
                                                 random_seed = 20250512,
                                                 n_obs = nrow(data_simmiss_scen4A))


# Scenario 4B (monotone x MAR x 25% missing x strong mechanism)

# CC: All Stage
simdata_scen4B_cc_preproc <- generate1dataset_cc_preproc_retrycopula(real_data = data_simmiss_scen4B,
                                                                     random_seed = 20250512,
                                                                     n_obs = nrow(data_simmiss_scen4B))

# IPW
simdata_scen4B_ipw <- generate1dataset_aipw_mono(real_data = data_simmiss_scen4B,
                                                 random_seed = 20250512,
                                                 n_obs = nrow(data_simmiss_scen4B))

# MI
simdata_scen4B_mi <- generate1dataset_mi_mono(real_data = data_simmiss_scen4B,
                                              random_seed = 20250512,
                                              n_obs = nrow(data_simmiss_scen4B))

### PCA plots ----

var_cont = c("Age", "N.qualifying.teeth", "BL.GE", "BL..BOP", "BL.PD.avg", "BL..PD.4", 
             "BL..PD.5", "BL.CAL.avg", "BL..CAL.2", "BL..CAL.3", "BL.Calc.I", "BL.Pl.I")

#### Scenario 4 ----

# Scenario 4A

# Original data

# Perform PCA
scen4A_real_pca <- prcomp(data_simmiss_scen4A[complete.cases(data_simmiss_scen4A), var_cont], 
                          center = TRUE, scale. = TRUE)

pca_2components_real_scen4A <- autoplot(scen4A_real_pca, 
                                        data = data_simmiss_scen4A[complete.cases(data_simmiss_scen4A), var_cont]) +
  ggtitle("Real") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# CC: All Stage

# Perform PCA
scen4A_CC_preproc_pca <- prcomp(simdata_scen4A_cc_preproc[, var_cont], center = TRUE, scale. = TRUE)

pca_2components_CC_preproc_scen4A <- autoplot(scen4A_CC_preproc_pca, data = simdata_scen4A_cc_preproc[, var_cont]) +
  ggtitle("CC: AllStage") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# IPW - Miss. Ind.

# Perform PCA
scen4A_IPW_missind_pca <- prcomp(simdata_scen4A_ipw_missind[complete.cases(simdata_scen4A_ipw_missind), var_cont], 
                                 center = TRUE, scale. = TRUE)

pca_2components_IPW_missind_scen4A <- autoplot(scen4A_IPW_missind_pca,
                                               data = simdata_scen4A_ipw_missind[complete.cases(simdata_scen4A_ipw_missind), var_cont]) +
  ggtitle("IPW: Ind") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# MI

# Perform PCA
scen4A_MI_pca <- prcomp(simdata_scen4A_mi[complete.cases(simdata_scen4A_mi), var_cont],
                        center = TRUE, scale. = TRUE)

pca_2components_MI_scen4A <- autoplot(scen4A_MI_pca,
                                      data = simdata_scen4A_mi[complete.cases(simdata_scen4A_mi), var_cont]) +
  ggtitle("MI") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))


# Scenario 4B

# Original data

# Perform PCA
scen4B_real_pca <- prcomp(data_simmiss_scen4B[complete.cases(data_simmiss_scen4B), var_cont], 
                          center = TRUE, scale. = TRUE)

pca_2components_real_scen4B <- autoplot(scen4B_real_pca, 
                                        data = data_simmiss_scen4B[complete.cases(data_simmiss_scen4B), var_cont]) +
  ggtitle("Real") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# CC: All Stage

# Perform PCA
scen4B_CC_preproc_pca <- prcomp(simdata_scen4B_cc_preproc[, var_cont], center = TRUE, scale. = TRUE)

pca_2components_CC_preproc_scen4B <- autoplot(scen4B_CC_preproc_pca, data = simdata_scen4B_cc_preproc[, var_cont]) +
  ggtitle("CC: AllStage") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# IPW

# Perform PCA
scen4B_IPW_pca <- prcomp(simdata_scen4B_ipw[complete.cases(simdata_scen4B_ipw), var_cont], 
                         center = TRUE, scale. = TRUE)

pca_2components_IPW_scen4B <- autoplot(scen4B_IPW_pca,
                                       data = simdata_scen4B_ipw[complete.cases(simdata_scen4B_ipw), var_cont]) +
  ggtitle("IPW") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))

# MI

# Perform PCA
scen4B_MI_pca <- prcomp(simdata_scen4B_mi[complete.cases(simdata_scen4B_mi), var_cont],
                        center = TRUE, scale. = TRUE)

pca_2components_MI_scen4B <- autoplot(scen4B_MI_pca,
                                      data = simdata_scen4B_mi[complete.cases(simdata_scen4B_mi), var_cont]) +
  ggtitle("MI") +
  xlim(-0.35, 0.35) +
  ylim(-0.35, 0.35) +
  theme_bw() +
  theme(plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(hjust = 0.5))


# Group plots together
p1 <- pca_2components_real_scen4A
p2 <- pca_2components_CC_preproc_scen4A
p4 <- pca_2components_IPW_missind_scen4A
p6 <- pca_2components_MI_scen4A
p7 <- pca_2components_real_scen4B
p8 <- pca_2components_CC_preproc_scen4B
p10 <- pca_2components_IPW_scen4B
p11 <- pca_2components_MI_scen4B

# Wrap plots into two rows
row1_plots <- wrap_plots(p1, p2, p4, p6, ncol = 4)
row2_plots <- wrap_plots(p7, p8, p10, p11, ncol = 4)

# Create row headers using text grobs
row1_label <- wrap_elements(
  grid::textGrob("Non-Monotone Missing", gp = gpar(fontsize = 24, fontface = "bold"), 
                 x = unit(0, "npc"), just = "left")
)
row2_label <- wrap_elements(
  grid::textGrob("Monotone Missing", gp = gpar(fontsize = 24, fontface = "bold"), 
                 x = unit(0, "npc"), just = "left")
)

# Combine with labels and rows
final_plot <- row1_label / row1_plots / row2_label / row2_plots +
  plot_layout(heights = c(0.5, 1, 0.5, 1))  # Adjust header heights

# Display the result
final_plot  # 16 x 8



# Parallelized simulations ----

# Scenario 4A ----

# Variables
n_obs = nrow(data_simmiss_scen4A)
real_data = data_simmiss_scen4A
col_order = c("PID", "Age", "N.qualifying.teeth", "BL.GE", "BL..BOP", "BL.PD.avg", 
              "BL..PD.4", "BL..PD.5", "BL.CAL.avg", "BL..CAL.2", "BL..CAL.3", "BL.Calc.I", 
              "BL.Pl.I", "Race", "Education", "Public.Asstce", "Prev.preg", "BL.Anti.inf", 
              "BL.Antibio", "BL.Bac.vag", "Group", "V3.PD.avg", "V5.PD.avg", "Birthweight")
var_cont = c("Age", "N.qualifying.teeth", "BL.GE", "BL..BOP", "BL.PD.avg", "BL..PD.4", 
             "BL..PD.5", "BL.CAL.avg", "BL..CAL.2", "BL..CAL.3", "BL.Calc.I", "BL.Pl.I")
var_disc = c("Race", "Education", "Public.Asstce", "Prev.preg", "BL.Anti.inf", "BL.Antibio", 
             "BL.Bac.vag")

# CC - Data Pre-Processing Step
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.cc.preproc.parallel <- foreach(
  i = 1:1000, 
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm", 
                "dgof", "caTools", "xgboost", "VIM", "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Generate data set 
                  data_synth <- generate1dataset_cc_preproc_retrycopula(real_data = data_real,
                                                                        random_seed = random_seed + i,
                                                                        n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1, 
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1, 
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real, data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real, data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Variables with missingness only - NON-MONOTONE SETTING Z1, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare ALL synthetic to OBSERVED real (should be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should NOT be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_allsynth_obsreal, V3.PD.avg_allsynth_allreal), 
                                                  nrow = 1, ncol = 3) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "AllSynthObsReal", "AllSynthAllReal")
                  
                  # Y (Birthweight)
                  
                  # Compare ALL synthetic to OBSERVED real
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_allsynth_obsreal, Birthweight_allsynth_allreal),
                                                    nrow = 1, ncol = 3) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "AllSynthObsReal", "AllSynthAllReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Non-Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "CC_PreProc") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(univar_cont_metrics = list(univar_cont %>% as_tibble()),
                                     univar_disc_metrics = list(univar_disc %>% as_tibble()),
                                     bivar_metrics = list(bivar %>% as_tibble()),
                                     MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                                     MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                                     trialinf = list(trialinf_betaCI %>% as_tibble()),
                                     missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenccpreproc <- end_time - start_time)

final_res <- sim.eval.cc.preproc.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4A_CC_preproc_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

# IPW - Missingness Indicator
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.aipwmissind.parallel <- foreach(
  i = 1:1000, 
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm", 
                "dgof", "caTools", "xgboost", "VIM", "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Add R variables (indicators of missingness)
                  data_real <- data_real %>%
                    mutate(R_3 = case_when(!is.na(V3.PD.avg) ~ 1,  # R = 1 means observed
                                           TRUE ~ 0),
                           R_5 = case_when(!is.na(V5.PD.avg) ~ 1,
                                           TRUE ~ 0),
                           R_Y = case_when(!is.na(Birthweight) ~ 1,
                                           TRUE ~ 0))
                  
                  # Generate data set 
                  data_synth_raw <- generate1dataset_aipw_nonmono_missind(real_data = data_real,
                                                                          random_seed = random_seed + i,
                                                                          n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth_raw[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID", "R_3", "R_5", "R_Y"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data AND SYNTHETIC DATA)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Synthetic data
                  knn_imp_synthetic <- kNN(data_synth, k = 5)[, 1:ncol(data_synth)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          knn_imp_synthetic) %>%
                    mutate(label = c(rep(0, nrow(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))])),
                                     rep(1, nrow(knn_imp_synthetic))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>%
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  data_synth_raw <- data_synth_raw %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  mod_comp <- lm(formula = Birthweight_complete ~ as.factor(tx_bin), data = data_synth_raw)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  beta_est_comp <- mod_comp$coefficients[2] %>% as.numeric()
                  CI_est_comp <- confint(mod_comp)[2,] %>% as.numeric()
                  trialinf_betaCI_comp <- matrix(data = c(beta_est_comp, CI_est_comp), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI_comp) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Missingness prop
                  missprop <- matrix(
                    data = c(sum(is.na(data_synth$V3.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$Birthweight))/nrow(data_synth)),
                    nrow = 1, ncol = 2)
                  colnames(missprop) <- c("V3.PD.avg", "Birthweight")
                  
                  # Variables with missingness only - NON-MONOTONE SETTING Z1, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V3.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V3.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_obssynth_obsreal, V3.PD.avg_allsynth_allreal,
                                                           V3.PD.avg_obssynth_allreal, V3.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Y (Birthweight)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  Birthweight_obssynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  Birthweight_obssynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_obssynth_obsreal, Birthweight_allsynth_allreal,
                                                             Birthweight_obssynth_allreal, Birthweight_allsynth_obsreal),
                                                    nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                         "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Non-Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "IPW Miss Ind") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(
                      univar_cont_metrics = list(univar_cont %>% as_tibble()),
                      univar_disc_metrics = list(univar_disc %>% as_tibble()),
                      bivar_metrics = list(bivar %>% as_tibble()),
                      MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                      MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                      trialinf = list(trialinf_betaCI %>% as_tibble()),
                      trialinf_comp = list(trialinf_betaCI_comp %>% as_tibble()),
                      missingprop = list(missprop %>% as_tibble()),
                      missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenipwmissind <- end_time - start_time)

final_res <- sim.eval.aipwmissind.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4A_IPW_missind_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

# MI
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.mi.parallel <- foreach(
  i = 1:1000,
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm",
                "dgof", "caTools", "xgboost", "VIM", "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Add R variables (indicators of missingness)
                  data_real <- data_real %>%
                    mutate(R_3 = case_when(!is.na(V3.PD.avg) ~ 1,  # R = 1 means observed
                                           TRUE ~ 0),
                           R_5 = case_when(!is.na(V5.PD.avg) ~ 1,
                                           TRUE ~ 0),
                           R_Y = case_when(!is.na(Birthweight) ~ 1,
                                           TRUE ~ 0))
                  
                  # Generate data set 
                  data_synth_raw <- generate1dataset_mi_nonmono(real_data = data_real,
                                                                random_seed = random_seed + i,
                                                                n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth_raw[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID", "R_3", "R_5", "R_Y"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data AND SYNTHETIC DATA)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Synthetic data
                  knn_imp_synthetic <- kNN(data_synth, k = 5)[, 1:ncol(data_synth)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          knn_imp_synthetic) %>%
                    mutate(label = c(rep(0, nrow(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))])),
                                     rep(1, nrow(knn_imp_synthetic))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>%
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  data_synth_raw <- data_synth_raw %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  mod_comp <- lm(formula = Birthweight_complete ~ as.factor(tx_bin), data = data_synth_raw)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  beta_est_comp <- mod_comp$coefficients[2] %>% as.numeric()
                  CI_est_comp <- confint(mod_comp)[2,] %>% as.numeric()
                  trialinf_betaCI_comp <- matrix(data = c(beta_est_comp, CI_est_comp), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI_comp) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Missingness prop
                  missprop <- matrix(
                    data = c(sum(is.na(data_synth$V3.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$Birthweight))/nrow(data_synth)),
                    nrow = 1, ncol = 2)
                  colnames(missprop) <- c("V3.PD.avg", "Birthweight")
                  
                  # Variables with missingness only - NON-MONOTONE SETTING Z1, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V3.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V3.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_obssynth_obsreal, V3.PD.avg_allsynth_allreal,
                                                           V3.PD.avg_obssynth_allreal, V3.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Y (Birthweight)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  Birthweight_obssynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  Birthweight_obssynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_obssynth_obsreal, Birthweight_allsynth_allreal,
                                                             Birthweight_obssynth_allreal, Birthweight_allsynth_obsreal),
                                                    nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                         "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Non-Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "MI") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(
                      univar_cont_metrics = list(univar_cont %>% as_tibble()),
                      univar_disc_metrics = list(univar_disc %>% as_tibble()),
                      bivar_metrics = list(bivar %>% as_tibble()),
                      MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                      MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                      trialinf = list(trialinf_betaCI %>% as_tibble()),
                      trialinf_comp = list(trialinf_betaCI_comp %>% as_tibble()),
                      missingprop = list(missprop %>% as_tibble()),
                      missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenmi <- end_time - start_time)

final_res <- sim.eval.mi.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4A_MI_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

# Scenario 4B ----

# Variables
n_obs = nrow(data_simmiss_scen4B)
real_data = data_simmiss_scen4B
col_order = c("PID", "Age", "N.qualifying.teeth", "BL.GE", "BL..BOP", "BL.PD.avg", 
              "BL..PD.4", "BL..PD.5", "BL.CAL.avg", "BL..CAL.2", "BL..CAL.3", "BL.Calc.I", 
              "BL.Pl.I", "Race", "Education", "Public.Asstce", "Prev.preg", "BL.Anti.inf", 
              "BL.Antibio", "BL.Bac.vag", "Group", "V3.PD.avg", "V5.PD.avg", "Birthweight")
var_cont = c("Age", "N.qualifying.teeth", "BL.GE", "BL..BOP", "BL.PD.avg", "BL..PD.4", 
             "BL..PD.5", "BL.CAL.avg", "BL..CAL.2", "BL..CAL.3", "BL.Calc.I", "BL.Pl.I")
var_disc = c("Race", "Education", "Public.Asstce", "Prev.preg", "BL.Anti.inf", "BL.Antibio", 
             "BL.Bac.vag")

# CC - Data Pre-processing Step
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.cc.parallel <- foreach(
  i = 1:1000, 
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm", 
                "dgof", "caTools", "xgboost", "VIM", "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Generate data set 
                  data_synth <- generate1dataset_cc_preproc_retrycopula(real_data = data_real,
                                                                        random_seed = random_seed + i,
                                                                        n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1, 
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1, 
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real, data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real, data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Variables with missingness only - MONOTONE SETTING Z1, Z2, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare ALL synthetic to OBSERVED real (should be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should NOT be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_allsynth_obsreal, V3.PD.avg_allsynth_allreal), 
                                                  nrow = 1, ncol = 3) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "AllSynthObsReal", "AllSynthAllReal")
                  
                  # Z2 (V5.PD.avg)
                  
                  # Compare ALL synthetic to OBSERVED real (should be close)
                  V5.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should NOT be close)
                  V5.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  V5.PD.avg_missmetrics <- matrix(data = c("V5.PD.avg", V5.PD.avg_allsynth_obsreal, V5.PD.avg_allsynth_allreal), 
                                                  nrow = 1, ncol = 3) %>% as.data.frame()
                  colnames(V5.PD.avg_missmetrics) <- c("Var", "AllSynthObsReal", "AllSynthAllReal")
                  
                  # Y (Birthweight)
                  
                  # Compare ALL synthetic to OBSERVED real
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_allsynth_obsreal, Birthweight_allsynth_allreal),
                                                    nrow = 1, ncol = 3) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "AllSynthObsReal", "AllSynthAllReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, V5.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "CC Preproc") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(univar_cont_metrics = list(univar_cont %>% as_tibble()), 
                                     univar_disc_metrics = list(univar_disc %>% as_tibble()),
                                     bivar_metrics = list(bivar %>% as_tibble()),
                                     MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                                     MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                                     trialinf = list(trialinf_betaCI %>% as_tibble()),
                                     missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenccpreproc <- end_time - start_time)

final_res <- sim.eval.cc.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4B_CC_preproc_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

# IPW
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.aipw.parallel <- foreach(
  i = 1:1000,
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm",
                "dgof", "caTools", "xgboost", "VIM", "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Add R variables (indicators of missingness)
                  data_real <- data_real %>%
                    mutate(R_3 = case_when(!is.na(V3.PD.avg) ~ 1,  # R = 1 means observed
                                           TRUE ~ 0),
                           R_5 = case_when(!is.na(V5.PD.avg) ~ 1,
                                           TRUE ~ 0),
                           R_Y = case_when(!is.na(Birthweight) ~ 1,
                                           TRUE ~ 0))
                  
                  # Generate data set 
                  data_synth_raw <- generate1dataset_aipw_mono(real_data = data_real,
                                                               random_seed = random_seed + i,
                                                               n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth_raw[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID", "R_3", "R_5", "R_Y"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data AND SYNTHETIC DATA)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Synthetic data
                  knn_imp_synthetic <- kNN(data_synth, k = 5)[, 1:ncol(data_synth)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          knn_imp_synthetic) %>%
                    mutate(label = c(rep(0, nrow(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))])),
                                     rep(1, nrow(knn_imp_synthetic))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>%
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  data_synth_raw <- data_synth_raw %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  mod_comp <- lm(formula = Birthweight_complete ~ as.factor(tx_bin), data = data_synth_raw)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  beta_est_comp <- mod_comp$coefficients[2] %>% as.numeric()
                  CI_est_comp <- confint(mod_comp)[2,] %>% as.numeric()
                  trialinf_betaCI_comp <- matrix(data = c(beta_est_comp, CI_est_comp), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI_comp) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Missingness prop
                  missprop <- matrix(
                    data = c(sum(is.na(data_synth$V3.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$V5.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$Birthweight))/nrow(data_synth)),
                    nrow = 1, ncol = 3)
                  colnames(missprop) <- c("V3.PD.avg", "V5.PD.avg", "Birthweight")
                  
                  # Variables with missingness only - MONOTONE SETTING Z1, Z2, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V3.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V3.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_obssynth_obsreal, V3.PD.avg_allsynth_allreal,
                                                           V3.PD.avg_obssynth_allreal, V3.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Z2 (V5.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V5.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V5.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V5.PD.avg_complete"],
                                                            real_data[, "V5.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V5.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V5.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V5.PD.avg_complete"],
                                                            real_data[, "V5.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V5.PD.avg_missmetrics <- matrix(data = c("V5.PD.avg", V5.PD.avg_obssynth_obsreal, V5.PD.avg_allsynth_allreal,
                                                           V5.PD.avg_obssynth_allreal, V5.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V5.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Y (Birthweight)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  Birthweight_obssynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  Birthweight_obssynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_obssynth_obsreal, Birthweight_allsynth_allreal,
                                                             Birthweight_obssynth_allreal, Birthweight_allsynth_obsreal),
                                                    nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                         "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, V5.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "IPW") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(
                      univar_cont_metrics = list(univar_cont %>% as_tibble()),
                      univar_disc_metrics = list(univar_disc %>% as_tibble()),
                      bivar_metrics = list(bivar %>% as_tibble()),
                      MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                      MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                      trialinf = list(trialinf_betaCI %>% as_tibble()),
                      trialinf_comp = list(trialinf_betaCI_comp %>% as_tibble()),
                      missingprop = list(missprop %>% as_tibble()),
                      missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenipw <- end_time - start_time)

final_res <- sim.eval.aipw.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4B_IPW_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

# MI
random_seed = 20250529

start_time <- Sys.time()
doParallel::registerDoParallel(cores = 10)
sim.eval.mi.parallel <- foreach(
  i = 1:1000,
  .packages = c("tidyverse", "rvinecopulib", "e1071", "caret", "EnvStats", "truncnorm",
                "dgof", "caTools", "xgboost", "VIM",
                "class", "patchwork", "mice")) %dopar% {
                  
                  set.seed(20250529)
                  
                  # First, make sure columns are in the correct order
                  data_real <- real_data[, col_order]
                  
                  # Add R variables (indicators of missingness)
                  data_real <- data_real %>%
                    mutate(R_3 = case_when(!is.na(V3.PD.avg) ~ 1,  # R = 1 means observed
                                           TRUE ~ 0),
                           R_5 = case_when(!is.na(V5.PD.avg) ~ 1,
                                           TRUE ~ 0),
                           R_Y = case_when(!is.na(Birthweight) ~ 1,
                                           TRUE ~ 0))
                  
                  # Generate data set 
                  data_synth_raw <- generate1dataset_mi_mono(real_data = data_real,
                                                             random_seed = random_seed + i,
                                                             n_obs = n_obs)
                  
                  # Make sure columns are in the same order as real data set
                  # This is not actually necessary for R, but to make things consistent with python
                  data_synth <- data_synth_raw[, col_order]
                  
                  # Compute univariate metrics
                  
                  # Calculate univariate continuous metrics for current data set
                  univar_cont_cols <- ks.stat(Synthetic_Data = data_synth[, var_cont],
                                              Real_Data = data_real[, var_cont])
                  univar_cont <- matrix(data = univar_cont_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_cont_cols)) %>% as.data.frame()
                  colnames(univar_cont) <- univar_cont_cols[, 1]
                  
                  # Calculate univariate discrete metrics for current data set
                  univar_disc_cols <- tvd.stat(Synthetic_Data = data_synth[, var_disc],
                                               Real_Data = data_real[, var_disc])
                  univar_disc <- matrix(data = univar_disc_cols[, 2], nrow = 1,
                                        ncol = nrow(univar_disc_cols)) %>% as.data.frame()
                  colnames(univar_disc) <- univar_disc_cols[, 1]
                  
                  # Calculate all bivariate metrics for current data set
                  
                  # First, remove id column
                  data_synth_noid <- data_synth[, !(names(data_synth) %in% c("PID"))]
                  data_real_noid <- data_real[, !(names(data_real) %in% c("PID", "R_3", "R_5", "R_Y"))]
                  
                  bivar <- bivar.metrics(Synthetic_Data = data_synth_noid, Real_Data = data_real_noid)
                  
                  # Calculate ML efficacy metrics - detecting real data from synthetic data
                  
                  # XGBoost
                  
                  # Combine real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(data_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          data_synth) %>%
                    mutate(label = c(rep(0, nrow(data_real)), rep(1, nrow(data_synth))))
                  
                  # Split real data into training set and test set, and separate labels from covariates
                  # 70:30 split
                  train_split <- 0.7
                  
                  # Set random seed
                  set.seed(random_seed + i)
                  
                  # Sample rows for training set (70%)
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  # Make sure data is all type numeric and stored in a matrix
                  data_matrix <- apply(data_realsynth, 2, as.numeric) %>% as.matrix()
                  
                  data_train <- data_matrix[sample_indices, ]
                  data_test <- data_matrix[!sample_indices, ]
                  
                  # Train the prediction model on training set (real data)
                  xgb_mod <- xgboost(data = data_train[, !(names(data_realsynth) %in% c("label", "PID"))],
                                     label = data_train[, (names(data_realsynth) %in% c("label"))],
                                     max.depth = 6, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
                  
                  # Predict the outcome on the real data test set based on trained model
                  y_prob <- predict(xgb_mod, data_test[, !(names(data_realsynth) %in% c("PID", "label"))])
                  y_pred <- as.numeric(y_prob > 0.5)
                  
                  # Confusion matrix
                  cm <- confusionMatrix(as.factor(data_test[, "label"]), as.factor(y_pred))
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_xgb <- cm$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_xgb <- matrix(data = 1 - MLmetrics_xgb, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_xgb) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # KNN
                  
                  set.seed(random_seed + i)
                  
                  # Need to impute missing values first (for real data AND SYNTHETIC DATA)
                  
                  # Real data
                  knn_imp_real <- kNN(data_real, k = 5)[, 1:ncol(data_real)]
                  
                  # Synthetic data
                  knn_imp_synthetic <- kNN(data_synth, k = 5)[, 1:ncol(data_synth)]
                  
                  # Combine (imputed) real and synthetic data into 1 data set, and add label for real/synthetic
                  # 0 = real, 1 = synthetic
                  data_realsynth <- rbind(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))],
                                          knn_imp_synthetic) %>%
                    mutate(label = c(rep(0, nrow(knn_imp_real[, !(names(data_real) %in% c("R_3", "R_5", "R_Y"))])),
                                     rep(1, nrow(knn_imp_synthetic))))
                  
                  # Make sure all values are numeric
                  data_realsynth <- data_realsynth %>%
                    rename(Race_old = Race, Education_old = Education, Public.Asstce_old = Public.Asstce,
                           Prev.preg_old = Prev.preg, BL.Anti.inf_old = BL.Anti.inf, 
                           BL.Antibio_old = BL.Antibio, BL.Bac.vag_old = BL.Bac.vag,
                           Group_old = Group) %>%
                    mutate(Race = case_when(Race_old == "Black" ~ 0,
                                            Race_old == "Indigenous" ~ 1,
                                            Race_old == "Other" ~ 2,
                                            Race_old == "White" ~ 3),
                           Education = case_when(Education_old == "8-12 yrs " ~ 0,
                                                 Education_old == "LT 8 yrs " ~ 1,
                                                 Education_old == "MT 12 yrs" ~ 2),
                           Public.Asstce = case_when(Public.Asstce_old == "No " ~ 0,
                                                     Public.Asstce_old == "Yes" ~ 1),
                           Prev.preg = case_when(Prev.preg_old == "No " ~ 0,
                                                 Prev.preg_old == "Yes" ~ 1),
                           BL.Anti.inf = case_when(BL.Anti.inf_old == "0" ~ 0,
                                                   BL.Anti.inf_old == "1" ~ 1),
                           BL.Antibio = case_when(BL.Antibio_old == "0" ~ 0,
                                                  BL.Antibio_old == "1" ~ 1),
                           BL.Bac.vag = case_when(BL.Bac.vag_old == "0" ~ 0,
                                                  BL.Bac.vag_old == "1" ~ 1),
                           Group = case_when(Group_old == "C" ~ 0,
                                             Group_old == "T" ~ 1)) %>%
                    dplyr::select(-c("Race_old", "Education_old", "Public.Asstce_old", "Prev.preg_old", 
                                     "BL.Anti.inf_old", "BL.Antibio_old", "BL.Bac.vag_old", "Group_old"))
                  
                  # Split real data into training set and test set, and separate labels (outcome) from covariates
                  # 70:30 split
                  train_split <- 0.7
                  sample_indices <- sample.split(data_realsynth[, 1], SplitRatio = 0.70)
                  
                  data_train <- data_realsynth[sample_indices, ]
                  data_test <- data_realsynth[!sample_indices, ]
                  
                  # Train the prediction model on training set and predict label (real/synthetic)
                  # KNN algorithm for prediction of the outcome based on real data
                  knn_pred <- knn(train = data_train[, !(names(data_train) %in% c("PID", "label"))],
                                  test = data_test[, !(names(data_test) %in% c("PID", "label"))],
                                  cl = data_train[, names(data_train) %in% "label"],
                                  k = 5)
                  
                  # Confusion matrix
                  cm_knn <- confusionMatrix(as.factor(data_test[, "label"]), knn_pred)
                  
                  # Metrics (Accuracy, Precision, Recall, F1-Score)
                  MLmetrics_knn <- cm_knn$byClass[c("Balanced Accuracy", "Precision", "Recall", "F1")]
                  MLmetrics_complement_knn <- matrix(data = 1 - MLmetrics_knn, nrow = 1, ncol = 4) %>% as.data.frame()
                  colnames(MLmetrics_complement_knn) <- c("Balanced Accuracy", "Precision", "Recall", "F1")
                  
                  # Trial inference metrics
                  
                  # Create dichotomous tx variable for simplicity
                  data_synth <- data_synth %>%
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  data_synth_raw <- data_synth_raw %>% 
                    mutate(tx_bin = case_when(Group == "C" ~ 0,
                                              Group == "T" ~ 1))
                  
                  # Fit regression model (no confounders)
                  mod <- lm(formula = Birthweight ~ as.factor(tx_bin), data = data_synth)
                  
                  mod_comp <- lm(formula = Birthweight_complete ~ as.factor(tx_bin), data = data_synth_raw)
                  
                  # beta, CI for treatment
                  beta_est <- mod$coefficients[2] %>% as.numeric()
                  CI_est <- confint(mod)[2,] %>% as.numeric()
                  trialinf_betaCI <- matrix(data = c(beta_est, CI_est), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI) <- c("beta", "Lower CI", "Upper CI")
                  
                  beta_est_comp <- mod_comp$coefficients[2] %>% as.numeric()
                  CI_est_comp <- confint(mod_comp)[2,] %>% as.numeric()
                  trialinf_betaCI_comp <- matrix(data = c(beta_est_comp, CI_est_comp), nrow = 1, ncol = 3)
                  colnames(trialinf_betaCI_comp) <- c("beta", "Lower CI", "Upper CI")
                  
                  # Missingness prop
                  missprop <- matrix(
                    data = c(sum(is.na(data_synth$V3.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$V5.PD.avg))/nrow(data_synth),
                             sum(is.na(data_synth$Birthweight))/nrow(data_synth)),
                    nrow = 1, ncol = 3)
                  colnames(missprop) <- c("V3.PD.avg", "V5.PD.avg", "Birthweight")
                  
                  # Variables with missingness only - MONOTONE SETTING Z1, Z2, Y
                  
                  # Z1 (V3.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V3.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V3.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V3.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V3.PD.avg"],
                                                            real_data[, "V3.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V3.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V3.PD.avg_complete"],
                                                            real_data[, "V3.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V3.PD.avg_missmetrics <- matrix(data = c("V3.PD.avg", V3.PD.avg_obssynth_obsreal, V3.PD.avg_allsynth_allreal,
                                                           V3.PD.avg_obssynth_allreal, V3.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V3.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Z2 (V5.PD.avg)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  V5.PD.avg_obssynth_obsreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  V5.PD.avg_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "V5.PD.avg_complete"],
                                                            real_data[, "V5.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  V5.PD.avg_obssynth_allreal <- 1 - ks.test(data_synth[, "V5.PD.avg"],
                                                            real_data[, "V5.PD.avg_complete"],
                                                            alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  V5.PD.avg_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "V5.PD.avg_complete"],
                                                            real_data[, "V5.PD.avg"],
                                                            alternative = "two.sided")$statistic
                  
                  V5.PD.avg_missmetrics <- matrix(data = c("V5.PD.avg", V5.PD.avg_obssynth_obsreal, V5.PD.avg_allsynth_allreal,
                                                           V5.PD.avg_obssynth_allreal, V5.PD.avg_allsynth_obsreal), 
                                                  nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(V5.PD.avg_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                       "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Y (Birthweight)
                  
                  # Compare OBSERVED synthetic to OBSERVED real (should be close)
                  Birthweight_obssynth_obsreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to ALL real (should be close)
                  Birthweight_allsynth_allreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare OBSERVED synthetic to ALL real (should NOT be close)
                  Birthweight_obssynth_allreal <- 1 - ks.test(data_synth[, "Birthweight"],
                                                              real_data[, "Birthweight_complete"],
                                                              alternative = "two.sided")$statistic
                  
                  # Compare ALL synthetic to OBSERVED real (should NOT be close)
                  Birthweight_allsynth_obsreal <- 1 - ks.test(data_synth_raw[, "Birthweight_complete"],
                                                              real_data[, "Birthweight"],
                                                              alternative = "two.sided")$statistic
                  
                  Birthweight_missmetrics <- matrix(data = c("Birthweight", Birthweight_obssynth_obsreal, Birthweight_allsynth_allreal,
                                                             Birthweight_obssynth_allreal, Birthweight_allsynth_obsreal),
                                                    nrow = 1, ncol = 5) %>% as.data.frame()
                  colnames(Birthweight_missmetrics) <- c("Var", "ObsSynthObsReal", "AllSynthAllReal",
                                                         "ObsSynthAllReal", "AllSynthObsReal")
                  
                  # Combine all missingness metrics into one object
                  missmetrics <- rbind(V3.PD.avg_missmetrics, V5.PD.avg_missmetrics, Birthweight_missmetrics)
                  
                  # Tibble of metric results
                  tibble(
                    # First 5 columns are scenario settings
                    par1 = "Monotone", par2 = "MAR", par3 = "50", par4 = "Strong", method = "MI") %>% 
                    # Tibble contains metric results
                    bind_cols(tibble(
                      univar_cont_metrics = list(univar_cont %>% as_tibble()),
                      univar_disc_metrics = list(univar_disc %>% as_tibble()),
                      bivar_metrics = list(bivar %>% as_tibble()),
                      MLeff_xgb = list(MLmetrics_complement_xgb %>% as_tibble()),
                      MLeff_knn = list(MLmetrics_complement_knn %>% as_tibble()),
                      trialinf = list(trialinf_betaCI %>% as_tibble()),
                      trialinf_comp = list(trialinf_betaCI_comp %>% as_tibble()),
                      missingprop = list(missprop %>% as_tibble()),
                      missmetrics = list(missmetrics %>% as_tibble())
                    ))
                }
doParallel::stopImplicitCluster()
end_time <- Sys.time()
(time_takenmi <- end_time - start_time)

final_res <- sim.eval.mi.parallel %>%
  list_rbind()

# Name the columns you want to combine
list_cols <- names(final_res)[6:ncol(final_res)]

# Create a named list of combined tibbles, one for each list-column
Scen4B_MI_results <- purrr::map(set_names(list_cols), ~ bind_rows(final_res[[.x]]))

