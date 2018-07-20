## preamble, load in data etc

library(tidyverse)
library(TMB)
library(microbenchmark)

df <- read.csv('ping pong scores.csv', stringsAsFactors = F)

df <-
    filter(df, !is.na(Me))

# Empirical win rate
sum(df$Me) / (sum(df$Me) + sum(df$Opponent))
# [1] 0.592246


get_prob_of_scoreline <- function(me, you, prob_win_point){
    
    # warning, not vectorised!
    stopifnot(all(length(me) == 1, length(you) == 1, length(prob_win_point) == 1))
    
    min_me_you <- min(me, you)
    
    
    if (you >= 20) {
        out_prob <- choose(40, 20) * 2^(min_me_you - 20) * prob_win_point^me * (1 - prob_win_point)^you
        
    } else {
        out_prob <- choose(20 + min_me_you, min_me_you) * prob_win_point^me * (1 - prob_win_point)^you
        
    }
    
    return(out_prob);
}

get_prob_of_scoreline_vectorised <- function(me, you, prob_win_point){
    
    min_me_you <- pmin(me, you)
    
    
    
    out_prob <- 
        as.numeric(you >= 20) * 
        choose(40, 20) * 2^(min_me_you - 20) * prob_win_point^me * (1 - prob_win_point)^you +
        (1 - as.numeric(you >= 20)) *
        choose(20 + min_me_you, min_me_you) * prob_win_point^me * (1 - prob_win_point)^you
    
    
    
    return(out_prob);
}


# add probability of observing scoreline, assuming point win prob
# is independent and identically distributed (success prob ~= 0.59)
df <-
    df %>%
    rowwise() %>% # get_prob_of_scoreline is not vectorised
    mutate(score_diff = Me - Opponent) %>%
    mutate(prob_scoreline = get_prob_of_scoreline(Me, Opponent, 0.592246)) %>%
    mutate(log_lik = log(prob_scoreline)) %>%
    ungroup


# data exploration

df %>%
    mutate(prob_scoreline = 100 * prob_scoreline) %>% # express as percentage so in same scale as winning margin
    gather(key, value, score_diff, prob_scoreline) %>%
    mutate(key = factor(if_else(key == 'score_diff', 'Winning margin', 'Probability of scoreline (%)'))) %>%
    # mutate(value = if_else(key == 'Winning margin', value, NA_real_)) %>%
    ggplot(aes(Game, value, colour = key, linetype = key)) +
    geom_point() +
    geom_line() +
    scale_y_continuous(breaks = seq(0, 14, 2)) +
    scale_color_manual(values = c(NA, 'red')) +
    theme_bw() +
    xlab('Game number') +
    ylab('Winning margin')


# with probability of observing scoreline
df %>%
    mutate(prob_scoreline = 100 * prob_scoreline) %>% # express as percentage so in same scale as winning margin
    gather(key, value, score_diff, prob_scoreline) %>%
    mutate(key = if_else(key == 'score_diff', 'Winning margin', 'Probability of scoreline (%)')) %>%
    ggplot(aes(Game, value, colour = key, linetype = key)) +
    geom_point() +
    geom_line() +
    scale_y_continuous(breaks = seq(0, 14, 2)) +
    scale_color_manual(values = c('blue', 'red')) +
    theme_bw() +
    xlab('Game number') +
    ylab('Winning margin')


## load TMB DLL
dyn.load('cpp_files/reg_20180312_init_obj_fun.dll')


## Get TMB objective function which includes .$gr (gradient) and .$he (hessian) elements.
optim_in <- MakeADFun(data = list(me = df$Me, you = df$Opponent, score_diff = df$Me - df$Opponent),
                      parameters = list(p_0 = 0.592246, alpha_0 = 0, alpha_1 = 0, beta = 0),
                      DLL = 'reg_20180312_init_obj_fun',
                      silent = T)

# see that optim_in has the right elements to be passed into R's optimisation function optim. 
str(optim_in)
# List of 10
#     $ par     : Named num [1:4] 0.592246 0 0 0
#     ..- attr(*, "names")= chr [1:4] "p_0" "alpha_0" "alpha_1" "beta"
#     $ fn      :function (x = last.par, ...)  
#     $ gr      :function (x = last.par, ...)  
#     $ he      :function (x = last.par, atomic = usingAtomics())  
#     $ hessian : logi FALSE
#     $ method  : chr "BFGS"
#     $ retape  :function (set.defaults = TRUE)  
#     $ env     :<environment: 0x000000001bd5b158> 
#     $ report  :function (par = last.par)  
#     $ simulate:function (par = last.par, complete = FALSE)  


# optimise using optim
(optim_out <- do.call(optim, optim_in))


## Analyse results

# Get fitted point win probabilities
optim_in$fn(optim_out$par)
df <-
    df %>%
    mutate(fitted_prob = optim_in$report()$p_vec_out) %>%
    mutate(fitted_prob_scoreline = get_prob_of_scoreline_vectorised(Me, Opponent, fitted_prob)) %>%
    mutate(fitted_log_lik = log(fitted_prob_scoreline))

df %>%
    mutate(emp_prob = 0.592246) %>%
    gather(key, value, score_diff, emp_prob, fitted_prob) %>%
    mutate(key = case_when(key == 'score_diff' ~ 'Winning margin', 
                           key == 'emp_prob'~ 'Mean win prob',
                           key == 'fitted_prob' ~ 'Model win prob')) %>%
    mutate(facet = factor(if_else(key == "Winning margin", key, "Probability"), 
                          levels = c("Winning margin", "Probability"))) %>%
    ggplot(aes(Game, value, colour = key)) +
    geom_line() +
    geom_point() +
    xlab('Game number') +
    ylab('') +
    facet_grid(facet ~ ., scales = "free")

# "residuals"
plot(optim_in$report()$p_vec_out, df$Me - df$Opponent)


## TMB supplied objective function, numerical derivatives


optim_in_no_jac <- optim_in
optim_in_no_jac$gr <- NULL
optim_in_no_jac$he <- NULL

(optim_out_no_jac <- do.call(optim, optim_in_no_jac))


## R version

optim_in_R <- list()

R_obj_fun_template <- function(par, me, you, score_diff) {
    
    p_0 <- par[1]
    alpha_0 <- par[2]
    alpha_1 <- par[3]
    beta <- par[4]
    
    n_obs <- length(score_diff)
    
    p <- p_0 # initialise initial model fitted probability
    nll <- 0 # initialise model negloglik
    
    for (i in seq_along(you)) {
        
        if (i > 1) {
            # mixing coefficient
            alpha <- plogis(alpha_0 + alpha_1 * score_diff[i - 1]);
            
            # update p with model
            p <-  alpha * p_0 + (1 - alpha) *  plogis(qlogis(p) + beta);
            
            
        }
        
        # increment nll
        nll <- nll - log(get_prob_of_scoreline(me[i], you[i], p));
        
    }
    
    
    
    return(nll)
}

optim_in_R$fn <- function(par) R_obj_fun_template(par, df$Me, df$Opponent, df$score_diff)
optim_in_R$par <- c(0.592246, 0, 0, 0)
# quasi-Newton optimisation with approximations to the Hessian
# i.e. second derivatives are never required
optim_in_R$method <- 'BFGS'

(optim_out_R <- do.call(optim, optim_in_R))

## Rcpp version

library(Rcpp)

R_obj_fun_template_Rcpp <- cppFunction(
    "double R_obj_fun_templatep_Rcpp(NumericVector par, IntegerVector me, IntegerVector you, IntegerVector score_diff) {

    // parameters:
    double p_0 = par[0];
    double alpha_0 = par[1];
    double alpha_1 = par[2];
    double beta = par[3];
    
    int n_obs = me.size();
    
    double p = p_0; // initialise initial model fitted probability
    double nll = 0; // initialise model negloglik
    
    
    for (int i = 0; i < n_obs; i++) {
    
    if (i > 0) {
    // mixing coefficient
    double alpha = Rf_plogis(alpha_0 + alpha_1 * score_diff[i - 1], 0.0, 1.0, 1, 0);
    
    // update p with model
    
    p =  alpha * p_0 + (1 - alpha) *  Rf_plogis(Rf_qlogis(p, 0.0, 1.0, 1, 0) + beta, 0.0, 1.0, 1, 0);
    
    
    }
    nll += -log(get_prob_of_scoreline_Rcpp(me[i], you[i], p));
    
    
    }
    
    
    return nll;
    
    }",
    includes = "double get_prob_of_scoreline_Rcpp(int me, int you, double prob_win_point) {
    double out_prob;
    
    
    int min_me_you = std::min(me, you);
    
    
    if (you >= 20) {
    out_prob = Rf_choose(40, 20) * pow(2, min_me_you - 20) * pow(prob_win_point, me) * pow(1 - prob_win_point, you);
    
    } else {
    out_prob = Rf_choose(20 + min_me_you, min_me_you) * pow(prob_win_point, me) * pow(1 - prob_win_point, you);
    
    }
    
    return(out_prob);
    
    
    }")


optim_in_Rcpp <- list()
optim_in_Rcpp$fn <- function(par) R_obj_fun_template_Rcpp(par, df$Me, df$Opponent, df$Me - df$Opponent)
optim_in_Rcpp$par <- c(0.592246, 0, 0, 0)
optim_in_Rcpp$method <- 'BFGS'


(optim_out_Rcpp <- do.call(optim, optim_in_Rcpp))

microbenchmark(TMB = do.call(optim, optim_in), 
               TMB_numerical = do.call(optim, optim_in_no_jac), 
               Rcpp = do.call(optim, optim_in_Rcpp),
               base_R = do.call(optim, optim_in_R))
