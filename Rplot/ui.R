#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(ggplot2)
library(plotly)
library(tidyverse)
library(htmlwidgets)
library(pander)
library(latex2exp)
library(knitr)
library(dplyr) #data frame manipulation
require(scales) # to access break formatting functions
library(shiny)
library(shinyWidgets)
library(ggrepel)
require(Rose)
# Define UI for application that draws a histogram
shinyUI(fluidPage(

    # Application title
    titlePanel("\\(  Z_2-\\phi^4\\)"),

    # Sidebar with a slider input for number of bins
    sidebarPanel(
        selectInput("L", label = "L",
                    choices =as.integer( c(10,16, 20,24,26,32,40)), selected = 20),
        selectInput("T", label = "T",
                    choices =as.integer( c(24,32, 48,64,96,128)), selected = 128),
        selectInput("msq0", label = "msq0",
                    choices = c(0.1,-4.9,-4.95,-4.925,-4.98,-4.99,-5.0), selected = -4.925),
        selectInput("msq1", label = "msq1",
                    choices =c(0.1,-4.9,-4.89,-4.85), selected = -4.85),
        selectInput("l0", label = "$\\lambda_0$",
                    choices =c(0.05,2.5), selected = 2.5),
        selectInput("l1", label = "lambda_1",
                    choices =c(0.05,2.5), selected = 2.5),
        selectInput("mu", label = "mu",
                    choices =c(0.1, 5), selected = 5),
        selectInput("g", label = "g",
                    choices =c(0.0), selected = 0),
        selectInput("rep", label = "rep",
                    choices =c(0,1,2), selected = 0),
        #selectInput("Obs", label = "Obs",
        #            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
        #                       "C4_BH","E2_01", "meff(E2_01)"),
        #            selected = "C4_BH"),
        selectInput("logscale", label = "logscale",
                    choices =c("no","yes"), selected = "no")
        
    ),
    
    inputPanel(
        #   pickerInput(
        #     inputId = "manyObs_old",
        #     label = "Obs_old",
        #     choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
        #                "C4_BH_0","C4_BH_1","C4_BH","C4_BH+c","E2_01","GEVP_01"
        #                ,"C4_BH_0_s","C4_BH_1_s","C4_BH_s","C4_BH_s+c"),
        #     options = list(
        #         `actions-box` = TRUE,
        #         size = 10
        #         ,`selected-text-format` = "count > 3"
        #         
        #     ),
        #     multiple = TRUE,
        #     selected = c("meff0","C4_BH")
        # ),
        uiOutput("obs_list"),
        pickerInput(
            inputId = "log_meff_corr",
            label = "log_meff_corr",
            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                       "C4_BH_0","C4_BH_1","C4_BH","E2_01","two0_to_two1","four0_to_two1",
                       "four0_to_two0"),
            options = list(
                `actions-box` = TRUE,
                size = 10
                ,`selected-text-format` = "count > 3"
                
            ),
            multiple = TRUE,
            #selected = c("meff0","C4_BH")
        ),
        pickerInput(
            inputId = "raw_corr",
            label = "raw_corr",
            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                       "C4_BH_0","C4_BH_1","C4_BH","E2_01","two0_to_two1","four0_to_two1",
                       "four0_to_two0"
                       ,"C4_BH_0_s","C4_BH_1_s","C4_BH_s"),
            options = list(
                `actions-box` = TRUE,
                size = 10
                ,`selected-text-format` = "count > 3"
                
            ),
            multiple = TRUE,
            #selected = c("meff0","C4_BH")
        ),
        pickerInput(
            inputId = "shifted_corr",
            label = "shifted_corr",
            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                       "C4_BH_0","C4_BH_1","C4_BH","E2_01","two0_to_two1","four0_to_two1",
                       "four0_to_two0"),
            options = list(
                `actions-box` = TRUE,
                size = 10
                ,`selected-text-format` = "count > 3"
                
            ),
            multiple = TRUE,
            #selected = c("meff0","C4_BH")
        ),
        pickerInput(
            inputId = "log_meff_shifted_corr",
            label = "log_meff_shifted_corr",
            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                       "C4_BH_0","C4_BH_1","C4_BH","E2_01","two0_to_two1","four0_to_two1",
                       "four0_to_two0"),
            options = list(
                `actions-box` = TRUE,
                size = 10
                ,`selected-text-format` = "count > 3"
                
            ),
            multiple = TRUE,
            #selected = c("meff0","C4_BH")
        )
        
    )
   
    ,mainPanel( 
      withMathJax(),
                plotlyOutput(outputId = "plot_many", height = "600px")
                ,uiOutput("fit_P")
                #####################################
                ,tableOutput("mass_table_0")
                ,tableOutput("mass_table_1")
                ,tableOutput("mass_table_01")
                
                #####################################
                ,
                h2("effective mass \\(  \\log c(t)/c(t+1)\\)"),
                inputPanel(
                    pickerInput(
                        inputId = "manyObs_meff",
                        label = "Obs_meff",
                        choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                                   "C4_BH","E2_01","two0_to_two1","four0_to_two1",
                                   "four0_to_two0"),
                        options = list(
                            `actions-box` = TRUE,
                            size = 10
                            ,`selected-text-format` = "count > 3"
                            
                        ),
                        multiple = TRUE,
                        selected = c("meff0","C4_BH")
                    )
                ),
   
    
                plotlyOutput(outputId = "plot_many_meff", height ="600px"),
                h2("Raw correlator"),
                plotlyOutput(outputId = "plot_many_raw", height ="600px"),
                h2("Shifeted correlator"),
                plotlyOutput(outputId = "plot_many_shift", height ="600px"),
                h2("log_meff Shifeted correlator"),
                plotlyOutput(outputId = "plot_many_log_meff_shifted", height ="600px"),
                
    ),
                  #####################################
                h2("Summary Table"),
                dataTableOutput("summary_table")
    
    
    
    
    
    
))
