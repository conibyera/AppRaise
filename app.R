################################################################################
#The R Shiny code used in this paper is released under the #GNU General Public 
#License (GPL).Users are free to use, modify, and distribute the code under the
#terms of the GPL.
#
#AppRaise free software: you can redistribute it and/or modify it under the terms
#of the GNU General Public License as published by the Free Software Foundation, 
#either version 3 of the License, or (at your option) any later version.
################################################################################

library(shiny)
library(shinyjs)
library(rstan)
library(bayesplot)
library(sn)
library(VGAM)
library(ggplot2)
library(shinycssloaders)
library(cmdstanr)
library(posterior)

# Set the number of cores to use for parallel processing
options(mc.cores = parallel::detectCores())

# Define the Stan model code
stan_code <- '

data {
  int<lower=0> NN;
  real yhat;
  real yhat2;
  real stdev;
  vector[NN] b;
  vector[NN] s;
  vector[NN] d;
  vector[NN] e;
  vector[NN] en;
  vector[2 * NN] ab_values; 
  vector[3 * NN] skn_values; 
  vector[2 * NN] de_values;
  vector[NN] ex_values; 
  vector[NN] exneg_values;
  real threshold_value;
}

transformed data {
  vector [NN] xi;
  vector[2 * NN] ab;
  vector[3 * NN] skn;
  vector[2 * NN] de;
  vector[NN] ex;
  vector[NN] exneg;
  
  ab = ab_values;
  skn = skn_values;
  de = de_values;
  ex = ex_values;
  exneg = exneg_values;

  for (i in 1:NN) {
    if (b[i] > 0 && b[i] != 999) {
      xi[i] = beta_rng(ab[2*i-1], ab[2*i]);
    } else if (s[i] > 0 && s[i] != 999) {
      xi[i] = skew_normal_rng(skn[3*i-2], skn[3*i-1], skn[3*i]);
    } else if (d[i] > 0 && d[i] != 999) {
      xi[i] = double_exponential_rng(de[2*i-1], de[2*i]);
    } else if (e[i] > 0 && e[i] != 999) {
      xi[i] = exponential_rng(ex[i]);
    } else if (en[i] > 0 && en[i] != 999) {
      xi[i] = -exponential_rng(exneg[i]);
    }
  }
}

parameters {
  real thetastar;
}

transformed parameters {
  real theta;
  real mid;
  theta = thetastar - sum(xi);
  if (threshold_value * theta > 0){
     mid = step(abs(theta) - abs(threshold_value));
  } else if (threshold_value < 0){
     mid = step(threshold_value);}
     else if (theta < 0){
     mid = step(theta);
     } else
     {
      mid = step(abs(theta - (threshold_value)));
     }
}

model {
  thetastar ~ normal(yhat2, pow(10,5));
  yhat ~ normal(thetastar, stdev);
}
'
# Save the Stan code to a file
writeLines(stan_code, "model.stan")

# Define the UI
ui <- fluidPage(
  useShinyjs(),
  tags$style(HTML("
    .title-container {
      background-color: #E0E0E0;
      color: #003366;
      padding: 10px;
      text-align: center;
      border-radius: 5px;
      margin-bottom: 20px;
      margin-top: 10px;
      box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title-container h1 {
      margin: 0;
    }
    .sidebar {
      background-color: #E0E0E0;
      padding: 20px;
      border-radius: 5px;
      box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar label, .sidebar .control-label {
      color: #003366;
    }
    .sidebar .shiny-input-container {
      color: white;
    }
    
   ")),
  div(class = "title-container", 
      h1("AppRaise: A Tool for Quantifying Uncertainty in Systematic Reviews Using a Posterior Mixture Model")
  ),
  sidebarLayout(
    sidebarPanel(class = "sidebar",
                 selectizeInput("study_name", "Select or Add Study Name:", choices = NULL, multiple = FALSE, options = list(create = TRUE)),           
                 numericInput("study_weight", "Enter Study Weight:", value = 1, min = 0),           
                 numericInput("obs_estimate", "Reported Point Estimate:", value = -0.6),
                 numericInput("obs_std", "Reported Standard Error:", value = 0.12, min = 0.0001),
                 numericInput("num_biases", "Number of Bias Types Identified:", value = 2, min = 1, max = 5, step = 1),
                 selectInput("scale_measure", "Select Measure",
                             choices = c("Log Risk Ratio", "Risk Difference", "Log Hazard Ratio","Incidence Rate Difference" ,"Log Incidence Rate Ratio", "Log Odds Ratio" ,"Mean Difference", "Median Difference", "Proportion","Mean","Median")),
                 numericInput("threshold_value", "Threshold for Significance:", value = -0.4),
                 selectizeInput("b_types", "Biases with Beta Prior:", choices = c("", "Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias"), multiple = TRUE),
                 uiOutput("ab_inputs"),
                 selectizeInput("s_types", "Biases with Skew Normal Prior:", choices = c("", "Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias"), multiple = TRUE),
                 uiOutput("skn_inputs"),
                 selectizeInput("d_types", "Biases with Laplace Prior:", choices = c("", "Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias"), multiple = TRUE),
                 uiOutput("de_inputs"),
                 selectizeInput("e_types", "Biases with Exponential Prior:", choices = c("", "Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias"), multiple = TRUE),
                 uiOutput("ex_inputs"),
                 selectizeInput("en_types", "Biases with Negative Exponential Prior:", choices = c("", "Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias"), multiple = TRUE),
                 uiOutput("exneg_inputs"),
                 actionButton("run_model", "Run Model"),
                 uiOutput("error_message")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Instructions", htmlOutput("instructions")),
        tabPanel("Prior Distributions of Biases",
                 fluidRow(
                   plotOutput("prior_distributions"),
                   uiOutput("prior_text")
                 )
        ),
        tabPanel("Posterior Distribution of Target Parameter",
                 fluidRow(
                   plotOutput("posterior_distribution"),
                   uiOutput("posterior_text")
                 )
        ),
        tabPanel("Trace Plot",
                 fluidRow(
                   plotOutput("trace_plot"),
                   uiOutput("trace_text")
                 )
        ),
        tabPanel("Probability of Significance",
                 fluidRow(
                   uiOutput("weighted_average_value")
                 )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Show instructions on range restriction
  output$instructions <- renderUI({
    tagList(tags$h5("Input Range:", style = "color:#003366; font-weight:bold;"), tags$p("Please make sure to enter values for the following fields within the specified range: 'Study Weight' [0-1], 
                   'Standard Error' [> 0], 'Alpha' [> 0], 'Beta' [> 0], 'Scale' [> 0], and 'Rate' [> 0].", 
                                                                                        style = "color:black;"),
            tags$h5("Study Name:", style = "color:#003366; font-weight:bold;"), tags$p("When a user add or select a study name, input its field values,
                                                                                       and run the model, the system will internally store the generated probability of 
                                                                                       significance for that study. When a user enters or selects a new study and fills 
                                                                                       in its field values, the system will generate a new probability specifically for that 
                                                                                       study. Ideally, the study weights should sum to 1; if they do not, the system will 
                                                                                       normalize them to ensure they sum to 1.", style = "color:black;"),
            tags$h5("Prior Probability Distributions:", style = "color:#003366; font-weight:bold;"), tags$p("This ", a("link",href="https://kabali.shinyapps.io/Distributions/", target = "_blank") ," will direct users to a page where they can adjust input parameters until the plots accurately reflect their beliefs about 
                                                                                                            uncertainty in bias sizes. Once satisfied, they can transfer those input parameters back to this page and run the model.", 
                                                                                                            style = "color:black;"), 
            tags$h5("When the Threshold for Significance is 0:", style = "color:#003366; font-weight:bold;"), tags$p("If you set the threshold to 0, the system will 
            interpret this as an interest in any non-zero values, regardless of whether they are positive or negative. To indicate an interest in values greater than 0, 
            use a small positive number (e.g., 0.001) as the threshold. Conversely, to focus on values less than 0, use a small negative number (e.g., -0.001) as the threshold."), 
            
            tags$h5("Citing the App:", 
                    style = "color:#003366; font-weight:bold;"),
            paste("Kabali C. AppRaise: A Tool for Quantifying Uncertainty in Systematic Reviews Using a Posterior Mixture Model [Internet]. [cited", format(Sys.Date(), "%b %d %Y"), "]. Available through: https://conibyera.github.io/appraise-start/", 
                  sep = " "),
            tags$p(" "),
            p("For any questions please contact Conrad Kabali at conrad.kabali@utoronto.ca.")
    )
  })
  
  bias_labels <- c("Confounding", "Selection Bias", "Measurement Errors", "Model Misspecification", "Other Bias")
  bias_map <- c("Confounding" = 1, "Selection Bias" = 2, "Measurement Errors" = 3, "Model Misspecification" = 4, "Other Bias" = 5)
  
  # Reactive values to store study data
  study_data <- reactiveValues(names = character(), weights = numeric(), mid_values = numeric(), theta_samples = numeric())
  
  
  generate_numeric_input <- function(id_prefix, bias_number, label_prefix, default_value = 999, min_value = 0.001) {
    numericInput(paste0(id_prefix, bias_number), paste0(label_prefix), value = default_value, min = min_value, step = 0.001)
  }
  
  output$ab_inputs <- renderUI({
    lapply(input$b_types, function(bias) {
      bias_number <- bias_map[bias]
      tagList(fluidRow(
        column(6,generate_numeric_input("ab_alpha", bias_number, "Enter Alpha Parameter", default_value = 999)),
        column(6,generate_numeric_input("ab_beta", bias_number, "Enter Beta Parameter", default_value = 999))
      ))
    })
  })
  
  output$skn_inputs <- renderUI({
    lapply(input$s_types, function(bias) {
      bias_number <- bias_map[bias]
      tagList(fluidRow(
        column(4,generate_numeric_input("skn_location", bias_number, "Enter Location Parameter", default_value = 999, min_value = -100000)),
        column(4,generate_numeric_input("skn_scale", bias_number, "Enter Scale Parameter", default_value = 999)),
        column(4,generate_numeric_input("skn_shape", bias_number, "Enter Shape Parameter", default_value = 999, min_value = -100000))
      ))
    })
  })
  
  output$de_inputs <- renderUI({
    lapply(input$d_types, function(bias) {
      bias_number <- bias_map[bias]
      tagList(fluidRow(
        column(6,generate_numeric_input("de_mean", bias_number, "Enter Mean Parameter", default_value = 999, min_value = -100000)),
        column(6,generate_numeric_input("de_scale", bias_number, "Enter Scale Parameter", default_value = 999))
      ))
    })
  })
  
  output$ex_inputs <- renderUI({
    lapply(input$e_types, function(bias) {
      bias_number <- bias_map[bias]
      fluidRow(column(12,generate_numeric_input("ex_lambda", bias_number, "Enter Rate Parameter", default_value = 999)))
    })
  })
  
  output$exneg_inputs <- renderUI({
    lapply(input$en_types, function(bias) {
      bias_number <- bias_map[bias]
      fluidRow(column(12, generate_numeric_input("exneg_lambda", bias_number, "Enter Rate Parameter", default_value = 999)))
    })
  })
  
  check_duplicates <- reactive({
    selected_values <- c(input$b_types, input$s_types, input$d_types, input$e_types, input$en_types)
    duplicated_values <- selected_values[duplicated(selected_values)]
    return(length(duplicated_values) > 0)
  })
  
  check_num_biases <- reactive({
    selected_values <- c(input$b_types, input$s_types, input$d_types, input$e_types, input$en_types)
    return(length(selected_values) == input$num_biases)
  })
  
  validate_positive <- function(values, exceptions = NULL) {
    values <- ifelse(is.na(values), 999, values)  # Replace NA with 999
    print(paste("Original values:", toString(values)))
    if (!is.null(exceptions)) {
      valid_indices <- setdiff(seq_along(values), exceptions)
      values <- values[valid_indices]
      print(paste("Values after applying exceptions:", toString(values)))
    }
    result <- all(values > 0)
    print(paste("Validation result:", result))
    return(result)
  }
  
  observe({
    if (!check_duplicates() && check_num_biases()) {
      shinyjs::enable("run_model")
      output$error_message <- renderUI({ HTML('') })
    } else {
      shinyjs::disable("run_model")
    }
  })
  
  # Create a reactive value to store the threshold value
  threshold_value <- reactiveVal(NULL)
  
  observeEvent(input$run_model, {
    if (check_duplicates()) {
      output$error_message <- renderUI({ HTML('<span style="color:red;">Error! A bias cannot be modeled with more than one distribution.</span>') })
    } else if (!check_num_biases()) {
      output$error_message <- renderUI({ HTML('<span style="color:red;">Error! Number of biases selected does not match the specified number of biases.</span>') })
    } else {
      output$error_message <- renderUI({ HTML('') })
      
      # Initialize bias vectors
      b <- rep(999, input$num_biases)
      s <- rep(999, input$num_biases)
      d <- rep(999, input$num_biases)
      e <- rep(999, input$num_biases)
      en <- rep(999, input$num_biases)
      
      ab_values <- rep(999, 2 * input$num_biases)
      skn_values <- rep(999, 3 * input$num_biases)
      de_values <- rep(999, 2 * input$num_biases)
      ex_values <- rep(999, input$num_biases)
      exneg_values <- rep(999, input$num_biases)
      
      # Assign selected biases to their respective arrays
      j <- 1
      for (bias in input$b_types) {
        bias_index <- bias_map[bias]
        b[j] <- bias_index
        ab_values[2 * (j - 1) + 1] <- input[[paste0("ab_alpha", bias_index)]]
        ab_values[2 * j] <- input[[paste0("ab_beta", bias_index)]]
        j = j+1
      }
      
      
      for (bias in input$s_types) {
        bias_index <- bias_map[bias]
        s[j] <- bias_index
        skn_values[3 * (j - 1) + 1] <- input[[paste0("skn_location", bias_index)]]
        skn_values[3 * (j - 1) + 2] <- input[[paste0("skn_scale", bias_index)]]
        skn_values[3 * j] <- input[[paste0("skn_shape", bias_index)]]
        j = j+1
      }
      
      
      for (bias in input$d_types) {
        bias_index <- bias_map[bias]
        d[j] <- bias_index
        de_values[2 * (j - 1) + 1] <- input[[paste0("de_mean", bias_index)]]
        de_values[2 * j] <- input[[paste0("de_scale", bias_index)]]
        j = j +1
      }
      
      
      for (bias in input$e_types) {
        bias_index <- bias_map[bias]
        e[j] <- bias_index
        ex_values[j] <- input[[paste0("ex_lambda", bias_index)]]
        j = j + 1
      }
      
      
      for (bias in input$en_types) {
        bias_index <- bias_map[bias]
        en[j] <- bias_index
        exneg_values[j] <- input[[paste0("exneg_lambda", bias_index)]]
        j = j + 1
      }
      
      skn_exceptions <- c(1, 3, 4, 6, 7, 9, 10, 12, 13, 15)
      de_exceptions <- c(1, 3, 5, 7, 9)
      
      # Print values before validation
      print(paste("ab_values:", toString(ab_values)))
      print(paste("skn_values:", toString(skn_values)))
      print(paste("de_values:", toString(de_values)))
      print(paste("ex_values:", toString(ex_values)))
      print(paste("exneg_values:", toString(exneg_values)))
      
      if (!validate_positive(ab_values) || 
          !validate_positive(skn_values, skn_exceptions) || 
          !validate_positive(de_values, de_exceptions) || 
          !validate_positive(ex_values) || 
          !validate_positive(exneg_values)) {
        output$error_message <- renderUI({ HTML('<span style="color:red;">Error! Make sure not to enter non-positive values in the fields where only positive values are allowed.</span>') })
        return()
      }
      
      model_data <- list(
        NN = input$num_biases,
        yhat = input$obs_estimate,
        yhat2 = input$obs_estimate,
        stdev = input$obs_std,
        b = as.array(b),
        s = as.array(s),
        d = as.array(d),
        e = as.array(e),
        en = as.array(en),
        ab_values = ab_values,
        skn_values = skn_values,
        de_values = de_values,
        ex_values = as.array(ex_values),
        exneg_values = as.array(exneg_values),
        threshold_value = input$threshold_value
      )
      
      # Print the input data for debugging
      print(model_data)
      
      
      withProgress(message = 'Running model', detail= 'please wait...', value = 0.25, {
        tryCatch({
          mod <- cmdstan_model("model.stan")
          fit <- mod$sample(data = model_data, iter_sampling = 5000, iter_warmup = 1000, seed = 12345, parallel_chains = 4)
          
          
          incProgress(1)
          Sys.sleep(0.1)
          

          # Extracting metadata
          total_iterations <- fit$metadata()$iter_sampling
          warmup_iterations <- fit$metadata()$iter_warmup
          num_chains <- fit$num_chains()
          
          # Extract draws
          theta_samples <- fit$draws(variables = "theta", format = "draws_array")
          mid_samples <- fit$draws(variables = "mid", format = "draws_array")
          
          # Post-warmup draws
          theta_samples <- theta_samples[(warmup_iterations + 1):total_iterations, , ]
          mid_samples <- mid_samples[(warmup_iterations + 1):total_iterations, , ]
          
          # Convert to matrix form
          theta_samples <- as.matrix(theta_samples)
          mid_samples <- as.matrix(mid_samples)
         
          
          #theta_samples <- as.vector(fit$draws(variables = "theta"))
          #mid_samples <- as.vector(fit$draws(variables = "mid"))
          
          # Exclude warmup draws
          #theta_samples <- theta_samples[-(1:num_warmup), , drop = FALSE]
          #mid_samples <- mid_samples[-(1:num_warmup), , drop = FALSE]
          
          study_name <- input$study_name
          study_weight <- input$study_weight
          study_mid_value <- mean(mid_samples)
          
          isolate({
            study_data$names <- c(study_data$names, study_name)
            study_data$weights <- c(study_data$weights, study_weight)
            study_data$mid_values <- c(study_data$mid_values, study_mid_value)
            
            if (is.null(study_data$theta_samples)) {
              study_data$theta_samples <- list()
            }
            study_data$theta_samples <- append(study_data$theta_samples, list(as.vector(theta_samples)))
          })
          
          print(fit$summary())
         
          # Extract draws
          all_draws <- fit$draws(variables = "thetastar", format = "draws_array")
          
          # Post-warmup draws
          post_warmup_draws <- all_draws[(warmup_iterations + 1):total_iterations, , ]
          
          # Convert to matrix form
          samples <- as.matrix(post_warmup_draws)
          
          # Initialize xi_samples
          xi_samples <- matrix(NA, nrow = nrow(samples), ncol = model_data$NN)
          
          # Check results
          print(dim(theta_samples))
          print(dim(samples))
          print(dim(xi_samples))
          print(min(theta_samples))
          
          #samples <- as.vector(fit$draws(variables = "thetastar"))
          # Exclude warmup draws
          #samples <- samples[-(1:num_warmup), , drop = FALSE]
          
          #print(length(samples))
          #xi_samples <- matrix(NA, nrow = length(samples), ncol = model_data$NN)
          for (i in 1:model_data$NN) {
            if (model_data$b[i] != 999) {
              xi_samples[, i] <- rbeta(nrow(samples), model_data$ab_values[2*i-1], model_data$ab_values[2*i])
            } else if (model_data$s[i] != 999) {
              xi_samples[, i] <- rsn(n=nrow(samples), xi=model_data$skn_values[3*i-2], omega=model_data$skn_values[3*i-1], alpha=model_data$skn_values[3*i])
            }
            else if (model_data$d[i] != 999) {
              xi_samples[, i] <- rlaplace(n=nrow(samples), model_data$de_values[2*i-1], model_data$de_values[2*i])
            }
            else if (model_data$e[i] != 999) {
              xi_samples[, i] <- rexp(n=nrow(samples), model_data$ex_values[i])
            }
            else if (model_data$en[i] != 999) {
              xi_samples[, i] <- -rexp(n=nrow(samples), model_data$exneg_values[i])
            }
          }
          
          
          #print(paste("theta_samples:", toString(theta_samples)))
          #print(paste("xi_samples:", toString(xi_samples)))
          
          if (is.null(theta_samples) || is.null(xi_samples) || !is.numeric(theta_samples) || !is.numeric(xi_samples)) {
            output$error_message <- renderUI({ HTML('<span style="color:red;">Error! Extracted samples are not numeric.</span>') })
            return()
          }
          
          output$prior_distributions <- renderPlot({
            par(mfrow = c(1, input$num_biases))
            for (i in 1:input$num_biases) {
              if (model_data$b[i] != 999) {
                b_graph = hist(xi_samples[, i], plot = F)
                b_graph$density = b_graph$counts/sum(b_graph$counts)*100
                plot(b_graph,freq=FALSE, ylab = "Probability (%)", main = paste0(bias_labels[b[i]]," (Beta)", sep = " "), xlab = paste0(input$scale_measure, " ", "Due to ",bias_labels[b[i]] ), col = "lightblue")
                
              } else if (model_data$s[i] != 999) {
                s_graph <- hist(xi_samples[, i], plot = F)
                s_graph$density = s_graph$counts/sum(s_graph$counts)*100
                plot(s_graph, freq=F, ylab = "Probability (%)", main = paste0(bias_labels[s[i]]," (Skew Normal)"), xlab = paste0(input$scale_measure," ", "Due to ",bias_labels[s[i]] ), col = "lightblue")
                
              } else if (model_data$d[i] != 999) {
                d_graph <- hist(xi_samples[, i], plot = F)
                d_graph$density = d_graph$counts/sum(d_graph$counts)*100
                plot(d_graph, freq=F, ylab = "Probability (%)", main = paste0(bias_labels[d[i]], " (Laplace)"), xlab = paste0(input$scale_measure," ", "Due to ",bias_labels[d[i]] ), col = "lightblue")
                
                
              } else if (model_data$e[i] != 999) {
                e_graph <- hist(xi_samples[, i], plot = F)
                e_graph$density = e_graph$counts/sum(e_graph$counts)*100
                plot(e_graph, freq = F, ylab = "Probability (%)", main = paste0(bias_labels[e[i]]," (Exponential)"), xlab = paste0(input$scale_measure," ", "Due to ",bias_labels[e[i]] ), col = "lightblue")
                
              } else if (model_data$en[i] != 999) {
                en_graph <- hist(xi_samples[, i], plot = F)
                en_graph$density = en_graph$counts/sum(en_graph$counts)*100
                plot(en_graph, freq = F, ylab = "Probability (%)", main = paste0(bias_labels[en[i]]," (Exponential)"), xlab = paste0(input$scale_measure," ", "Due to ",bias_labels[en[i]] ), col = "lightblue")
              }
            }
            
          })
          
          output$prior_text <- renderUI({
            tagList(tags$p("The prior distribution(s) for bias parameters represent the belief (probability) that a portion of the observed (reported) 
                       ", input$scale_measure, " is influenced by a specific type of bias, which can either increase or decrease the true ",input$scale_measure, ", depending on the context.
                        This",a("link",href="https://kabali.shinyapps.io/Distributions/",target = "_blank"), " opens the page that displays  various probability distributions as histograms and allows users to adjust their structures in real-time to reflect beliefs about bias sizes. 
                        Once users achieve the desired distribution(s), they can transfer the corresponding parameter values to the respective fields here to reproduce the histograms. This 
                        uncertainty in bias size is then combined with random errors (entered as 'Reported Standard Error') to produce the posterior distribution of the predicted true ",input$scale_measure, 
                           "which is displayed in the 'Posterior Distribution of Target Parameter' tab."))
          })
          
          output$posterior_distribution <- renderPlot({
            post_graph <- hist(as.vector(theta_samples), plot = F)
            post_graph$density = post_graph$counts/sum(post_graph$counts)*100
            plot(post_graph,freq=FALSE, main = " ", xlab = input$scale_measure, ylab = "Probability (%)", col = "lightblue")
            abline(v = input$threshold_value, col = "red", lwd = 2)
            })
          
          # Update the reactive value for threshold
          threshold_value(input$threshold_value)
          
          output$posterior_text <- renderUI({
            HTML(paste("The probability that ", input$scale_measure, " exceeds ", threshold_value(), " is ",mean((fit$draws(variables = "mid"))*100),"%. It is based on evidence
                        reviewed in only a singe study by ",input$study_name, ". This probability can change after  
                        combining evidence across multple studies using a posterior mixture model, as reported in the 'Probability of Significance' tab."))
          })
          
          
          #output$trace_plot <- renderPlot({
          #traceplot(fit, pars = c("theta"))
          #})
          mcmc_array <- fit$draws(variables = "theta", inc_warmup = FALSE)
          nuts_params <- nuts_params(fit)
          trace_plot <- mcmc_trace(as.array(mcmc_array), pars = "theta", np = nuts_params)
          y_axis_label <- input$scale_measure
          x_axis_label <- "Iteration"
          modified_trace_plot <- trace_plot + labs(y = y_axis_label, x = x_axis_label)
          output$trace_plot <- renderPlot({
            print(modified_trace_plot)
          })
          
          output$trace_text <- renderUI({
            HTML(paste("The traceplot is used to visualize the convergence of the chains for the posterior distribution of the
                       ", input$scale_measure, " and diagnose potential issues with the simulation. A well-mixed chain with a 
                       stable distribution indicates convergence. The chain should stabilize around a fixed distribution, indicating
                       stationarity. The plot can also reveal autocorrelation, which may suggest issues with convergence or mixing. 
                       Comparing multiple chains helps diagnose convergence issues and ensures consistency across chains."))
          })
          
          
          output$weighted_average_value <- renderUI({
            weights <- study_data$weights
            mid_values <- study_data$mid_values
            theta_samples_list <- study_data$theta_samples  # now a list of numeric vectors
            
            if (length(weights) > 0 && length(mid_values) > 0) {
              # Weighted average of midpoints
              weighted_avg <- sum(weights * mid_values) / sum(weights)
              
              # Weighted posterior mean (mean of each vector times its weight)
              posterior_mean <- sum(weights * sapply(theta_samples_list, mean)) / sum(weights)
              
              # Weighted posterior distribution
              rep_counts <- pmax(round(weights * 10), 1)
              weighted_samples <- unlist(
                mapply(function(samples, count) rep(samples, count),
                       theta_samples_list, rep_counts, SIMPLIFY = FALSE)
              )
              
              posterior_int <- quantile(weighted_samples, probs = c(0.025, 0.975))
              
              tagList(
                tags$h1(""),
                tags$p("After reviewing the body of evidence and accounting for the studies' limitations, the probability of the predicted ",
                       input$scale_measure, " exceeding ", input$threshold_value,
                       " (the threshold for significance) is ", round(weighted_avg, 3) * 100, "%."),
                tags$p("The posterior ", input$scale_measure, " is ", round(posterior_mean, 3),
                       " with 95% credible limits [", round(posterior_int[1], 2), ", ", round(posterior_int[2], 2), "]")
              )
            } else {
              tagList(tags$h1(""), tags$p("No studies to calculate the probability"))
            }
          })
          
        }, error = function(e) {
          output$error_message <- renderUI({ HTML(paste('<span style="color:red;">Error:', e$message, '</span>')) })
        })
      })
    }
  })
}

# Run the application
shinyApp(ui = ui, server = server)

