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

# Define UI for application that draws a histogram
shinyUI(fluidPage(

    # Application title
    titlePanel("\\(  Z_2-\\phi^4\\)"),

    # Sidebar with a slider input for number of bins
    sidebarPanel(
        selectInput("L", label = "L",
                    choices =as.integer( c(10, 20,40)), selected = 20),
        selectInput("T", label = "T",
                    choices =as.integer( c(24, 48,96,128)), selected = 128),
        selectInput("msq0", label = "msq0",
                    choices = c(-4.9,-4.95,-4.925,-4.98,-4.99,-5.0), selected = -4.925),
        selectInput("msq1", label = "msq1",
                    choices =c(-4.9,-4.89,-4.85), selected = -4.85),
        selectInput("l0", label = "$\\lambda_0$",
                    choices =c(2.5), selected = 2.5),
        selectInput("l1", label = "lambda_1",
                    choices =c(2.5), selected = 2.5),
        selectInput("mu", label = "mu",
                    choices =c(5), selected = 5),
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
          pickerInput(
            inputId = "manyObs",
            label = "Obs",
            choices =c("meff0","meff1","E2_0","E2_1","E2","E3_0","E3_1","E3",
                       "C4_BH","E2_01","GEVP_01"),
            options = list(
                `actions-box` = TRUE,
                size = 10
                ,`selected-text-format` = "count > 3"
                
            ),
            multiple = TRUE,
            selected = c("meff0","C4_BH")
        )
    )
    ,mainPanel( 
                plotlyOutput(outputId = "plot_many", height = "600px"),
                #####################################
                withMathJax(),
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
    ),
                #####################################
                tableOutput("mass_table_0"),
                tableOutput("mass_table_1"),
                tableOutput("mass_table_01"),
                #####################################
                h2("Summary Table"),
                dataTableOutput("summary_table")
    
    
    
    
    
    
))
