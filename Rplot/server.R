#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
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
library(Rose)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    #if(!exists("Rread_block.R", mode="function")) source("Rread_block.R")
    #if(!exists("print_error.R", mode="function")) source("print_error.R")
    #if(!exists("print_error.R", mode="function")) source("plot_routines.R")
    #######################################################################################
    #files
    ####################################################################################### 
    file<-reactive({
        T<-as.integer(input$T)
        L<-as.integer(input$L)
        msq0<-as.double(input$msq0)
        msq1<-as.double(input$msq1)
        l0<-as.double(input$l0)
        l1<-as.double(input$l1)
        mu<-as.double(input$mu)
        g <- as.double(input$g)
        rep<-as.integer(input$rep)
        #dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
        dir <- "Data" 
        
        file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_output",                         dir, T, L,  msq0,  msq1,  l0,  l1,  mu,  g,  rep)
        return(file)
    })
    file_meff<-reactive({
        T<-as.integer(input$T)
        L<-as.integer(input$L)
        msq0<-as.double(input$msq0)
        msq1<-as.double(input$msq1)
        l0<-as.double(input$l0)
        l1<-as.double(input$l1)
        mu<-as.double(input$mu)
        g <- as.double(input$g)
        rep<-as.integer(input$rep)
        #dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
        dir <- "Data" 
        
        file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_meff_correlators",                         dir, T, L,  msq0,  msq1,  l0,  l1,  mu,  g,  rep)
        return(file)
    })
    file_raw<-reactive({
        T<-as.integer(input$T)
        L<-as.integer(input$L)
        msq0<-as.double(input$msq0)
        msq1<-as.double(input$msq1)
        l0<-as.double(input$l0)
        l1<-as.double(input$l1)
        mu<-as.double(input$mu)
        g <- as.double(input$g)
        rep<-as.integer(input$rep)
        #dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
        dir <- "Data" 
        
        file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_raw_correlators"
                     ,  dir, T, L,  msq0,  msq1,  l0,  l1,  mu,  g,  rep)
        return(file)
    })
    file_shift<-reactive({
      T<-as.integer(input$T)
      L<-as.integer(input$L)
      msq0<-as.double(input$msq0)
      msq1<-as.double(input$msq1)
      l0<-as.double(input$l0)
      l1<-as.double(input$l1)
      mu<-as.double(input$mu)
      g <- as.double(input$g)
      rep<-as.integer(input$rep)
      #dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
      dir <- "Data" 
      
      file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_shifted_correlators"
                   ,  dir, T, L,  msq0,  msq1,  l0,  l1,  mu,  g,  rep)
      return(file)
    })
    file_log_meff_shifted<-reactive({
      T<-as.integer(input$T)
      L<-as.integer(input$L)
      msq0<-as.double(input$msq0)
      msq1<-as.double(input$msq1)
      l0<-as.double(input$l0)
      l1<-as.double(input$l1)
      mu<-as.double(input$mu)
      g <- as.double(input$g)
      rep<-as.integer(input$rep)
      #dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
      dir <- "Data" 
      
      file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_log_meff_shifted"
                   ,  dir, T, L,  msq0,  msq1,  l0,  l1,  mu,  g,  rep)
      return(file)
    })
    
    
    n_and_plot<-reactive({
        if (input$Obs=="meff0")   { n<- 2 ;fun<-  myggplot    ; nmeff<-1         }
        if (input$Obs=="meff1")    {n<- 3 ;fun<-  myggplot  ; nmeff<-2   }
        if (input$Obs=="E2_0")  {  n<- 5;fun<-  myggplot    ; nmeff<-3 }
        if (input$Obs=="E2_1")  {  n<- 6; fun<-  myggplot   ; nmeff<-4  }
        if (input$Obs=="E2")    {  n<- 7; fun<-  myggplot   ; nmeff<-5  }
        if (input$Obs=="E3_0")  {  n<- 8; fun<-  my_fit_ggplot  ; nmeff<-6   }
        if (input$Obs=="E3_1")   { n<- 9; fun<-  my_fit_ggplot  ; nmeff<-7   }
        if (input$Obs=="E3")    {n<- 10 ; fun<-  my_fit_ggplot  ; nmeff<-8   }
        if (input$Obs=="C4_BH")    {n<- 13; fun<-  my_fit_ggplot  ; nmeff<-11     }
        if (input$Obs=="E2_01")    {n<- 20; fun<-  my_fit_ggplot    ; nmeff<-12   }
        if (input$Obs=="meff(E2_01)")    {n<- 21; fun<-  my_fit_ggplot  ; nmeff<-12     }
        
        return (c(n,fun, nmeff))
    })
    
    #######################################################################################
    #plots
    ####################################################################################### 
    gg_new<-reactive({  
        file<-file()
        n1<-n_and_plot()
        n<-n1[[1]]
        T<-as.integer(input$T)
        mt<-read_df(file)
        
        d<- get_block_n(mt,n)
        fit<- get_fit_n(mt,n)
        fit_range<- get_plateaux_range(mt,n)
        
        #plot data
        tmp<-n_and_plot()[[2]]
        gg<-tmp(d,fit,fit_range,T/2,input$logscale)
        
        
        #plot(gg)
        return(gg)
    })
    
    output$plot<-renderPlotly({
        ggplotly(gg_new())
    })
    
    n_and_plot_many<-function(input)({
        if (input=="meff0")   { n<- 2 ;fun<-  myggplot    ; nmeff<-1         }
        if (input=="meff1")    {n<- 3 ;fun<-  myggplot  ; nmeff<-2   }
        if (input=="E2_0")  {  n<- 5;fun<-  myggplot    ; nmeff<-3 }
        if (input=="E2_1")  {  n<- 6; fun<-  myggplot   ; nmeff<-4  }
        if (input=="E2")    {  n<- 7; fun<-  myggplot   ; nmeff<-5  }
        if (input=="E3_0")  {  n<- 8; fun<-  my_fit_ggplot  ; nmeff<-6   }
        if (input=="E3_1")   { n<- 9; fun<-  my_fit_ggplot  ; nmeff<-7   }
        if (input=="E3")    {n<- 10 ; fun<-  my_fit_ggplot  ; nmeff<-8   }
        if (input=="C4_BH")    {n<- 13; fun<-  my_fit_ggplot  ; nmeff<-11     }
        if (input=="C4_BH+c")    {n<- 18; fun<-  my_fit_ggplot  ; nmeff<-11     }
        if (input=="E2_01")    {n<- 20; fun<-  my_fit_ggplot    ; nmeff<-12   }
        
        if (input=="two0_to_two1")    {n<- NA; fun<-  my_fit_ggplot    ; nmeff<-13   }
        if (input=="four0_to_two1")    {n<- NA; fun<-  my_fit_ggplot    ; nmeff<-14   }
        if (input=="four0_to_two0")    {n<- NA; fun<-  my_fit_ggplot    ; nmeff<-15   }
        if (input=="GEVP_01")    {n<- 21; fun<-  my_fit_ggplot    ; nmeff<-NA   }
      
        
        return (c(n,fun, nmeff))
    })
    gg_many<-reactive({  
        gg<- ggplot()
        
        for (myobs in input$manyObs){
            file<-file()
            n1<-n_and_plot_many(myobs)
            n<-n1[[1]]
            T<-as.integer(input$T)
            mt<-read_df(file)
            
            d<- get_block_n(mt,n)
            fit<- get_fit_n(mt,n)
            fit_range<- get_plateaux_range(mt,n)
            
            #plot data
            #tmp<-tostring(myobs)
            
            gg<-  many_fit_ggplot(d,fit,fit_range,T/2,input$logscale,gg,  myobs  )
      
        
        }
        
        return(gg)
        
    })
    output$plot_many<-renderPlotly({
        ggplotly(gg_many(),dynamicTicks = TRUE)%>%
        layout(   yaxis = list( showexponent = "all", exponentformat = "e")    )
        #add the config if you want to zoom with the wheel of mouse
        # ggplotly(gg_many(),dynamicTicks = TRUE) %>% config(scrollZoom = TRUE)
    })
    gg_many_meff<-reactive({  
        gg<- ggplot()
        for (myobs in input$manyObs_meff){
            file<-file_meff()
            n1<-n_and_plot_many(myobs)
            
            n<-n1[[3]]
            T<-as.integer(input$T)
            mt<-read_df(file)
            
            d<- get_block_n(mt,n)
            fit<- get_fit_n(mt,n)
            fit_range<- get_plateaux_range(mt,n)
            
            gg<-  many_fit_ggplot(d,fit,fit_range,T/2,input$logscale,gg,  myobs  )
   
        }
        
        return(gg)
    })
    output$plot_many_meff<-renderPlotly({
        ggplotly(gg_many_meff(),dynamicTicks = TRUE)%>%
        layout(   yaxis = list( showexponent = "all", exponentformat = "e")    )
    })
    gg_many_raw<-reactive({  
        gg<- ggplot()
        for (myobs in input$manyObs_meff){
            file<-file_raw()
            n1<-n_and_plot_many(myobs)
            
            n<-n1[[3]]
            T<-as.integer(input$T)
            mt<-read_df(file)
            
            d<- get_block_n(mt,n)
            fit<- get_fit_n(mt,n)
            fit_range<- get_plateaux_range(mt,n)
            gg<-  many_fit_ggplot(d,fit,fit_range,T/2,input$logscale,gg,  myobs  )
            
        }
        
        return(gg)
    })
    output$plot_many_raw<-renderPlotly({
        ggplotly(gg_many_raw(),dynamicTicks = TRUE)%>%
        layout(   yaxis = list( showexponent = "all", exponentformat = "e")    )
    })
    
    gg_many_shift<-reactive({
        
      gg<- ggplot()
      for (myobs in input$manyObs_meff){
        n1<-n_and_plot_many(myobs)
        file<-file_shift()
        n<-n1[[3]]
        T<-as.integer(input$T)
        mt<-read_df(file)
        
        d<- get_block_n(mt,n)
        fit<- get_fit_n(mt,n)
        fit_range<- get_plateaux_range(mt,n)
        gg<-  many_fit_ggplot(d,fit,fit_range,T/2,input$logscale,gg,  myobs  )

      }
      
      return(gg)
    })
    
  
    output$plot_many_shift<-renderPlotly({
        ggplotly(gg_many_shift(),dynamicTicks = TRUE)%>%
        layout(   yaxis = list( showexponent = "all", exponentformat = "e")    )
    })
    
    
    gg_many_log_meff_shifted<-reactive({
      
      gg<- ggplot()
      for (myobs in input$manyObs_meff){
        n1<-n_and_plot_many(myobs)
        file<-file_log_meff_shifted()
        n<-n1[[3]]
        T<-as.integer(input$T)
        mt<-read_df(file)
        
        d<- get_block_n(mt,n)
        fit<- get_fit_n(mt,n)
        fit_range<- get_plateaux_range(mt,n)
        gg<-  many_fit_ggplot(d,fit,fit_range,T/2,input$logscale,gg,  myobs  )
        
      }
      
      return(gg)
    })
    
    
    output$plot_many_log_meff_shifted<-renderPlotly({
      ggplotly(gg_many_log_meff_shifted(),dynamicTicks = TRUE)%>%
        layout(   yaxis = list( showexponent = "all", exponentformat = "e")    )
    })
    
    #######################################################################################
    #mass tables
    #######################################################################################    
    mass_tables_01<-reactive({  
        
        df<- data.frame(  "meff0"=c(0),  "meff1"=c(0) ,      "E2_01"=c(0),    
                          "DE2_01"=c(0)   , "a_Luscher"=c(0)  , "a_BH_01"=c(0),
                          "m_0+m_1_a_01"=c(0))
        
        file<-file()
        T<-as.integer(input$T)
        
        mt<-read_df(file)
        fit<- get_fit_n(mt,2)
        fit1<- get_fit_n(mt,3)
        fit01<- get_fit_n(mt,20)
        fitBH<- get_fit_n(mt,13)
        df[ 1,]<- list( mean_print(fit[1,1], fit[1,2]), 
                        mean_print(fit1[1,1], fit1[1,2]),
                        mean_print(fit01[1,1], fit01[1,2]),
                        mean_print(fit01[2,3], fit01[2,4]) ,
                        mean_print(fit01[2,1], fit01[2,2]),
                        mean_print(fitBH[1,1], fitBH[1,2])   ,
                        mean_print(fit01[2,9], fit01[1,10]))
        return(df)
        
    })
    
    mass_tables_0<-reactive({  
        
        df<- data.frame(  "meff0"=c(0),  "meff1"=c(0) ,      "E2_0"=c(0),    
                          "DE2_0"=c(0)   , "a_Luscher"=c(0)  , "a_BH_0"=c(0),
                          "m_0a_00"=c(0))
        
        file<-file()
        T<-as.integer(input$T)
        
        mt<-read_df(file)
        fit<- get_fit_n(mt,2)
        fit1<- get_fit_n(mt,3)
        fit01<- get_fit_n(mt,5)
        fitBH<- get_fit_n(mt,11)
        df[ 1,]<- list( mean_print(fit[1,1], fit[1,2]), 
                        mean_print(fit1[1,1], fit1[1,2]),
                        mean_print(fit01[1,1], fit01[1,2]),
                        mean_print(fit01[2,3], fit01[2,4]) ,
                        mean_print(fit01[2,1], fit01[2,2]),
                        mean_print(fitBH[1,1], fitBH[1,2]) ,
                        mean_print(fit01[2,9], fit01[2,10]) )
        return(df)
        
    })
    mass_tables_1<-reactive({  
        
        df<- data.frame(  "meff0"=c(0),  "meff1"=c(0) ,      "E2_1"=c(0),    
                          "DE2_0"=c(0)   , "a_Luscher"=c(0)  , "a_BH_1"=c(0)    )
        
        file<-file()
        T<-as.integer(input$T)
        
        mt<-read_df(file)
        fit<- get_fit_n(mt,2)
        fit1<- get_fit_n(mt,3)
        fit01<- get_fit_n(mt,6)
        fitBH<- get_fit_n(mt,12)
        df[ 1,]<- list( mean_print(fit[1,1], fit[1,2]), 
                        mean_print(fit1[1,1], fit1[1,2]),
                        mean_print(fit01[1,1], fit01[1,2]),
                        mean_print(fit01[2,3], fit01[2,4]) ,
                        mean_print(fit01[2,1], fit01[2,2]),
                        mean_print(fitBH[1,1], fitBH[1,2])        )
        return(df)
        
    })
    
    output$mass_table_01<-renderTable( {mass_tables_01()})
    output$mass_table_0<-renderTable( {mass_tables_0()})
    output$mass_table_1<-renderTable( {mass_tables_1()})
    

    
    #######################################################################################
    #symmary table
    ####################################################################################### 
    df_tot_compact<- data.frame(  "L"=c(0),"T"=c(0),
                                  "msq0"=c(0), "msq1"=c(0),
                                  "meff0"=c(0),  "meff1"=c(0) , 
                                  #   "E2_0"=c(0),  "E2_1"=c(0),
                                  #   "E3_0"=c(0),  "E3_1"=c(0),
                                  "a_01_BH"=c(0),
                                  "a_01_luscher"=c(0) ,"E2_01"=c(0) ,"DeltaE2_01"=c(0) ,
                                  "lambda0"=c(0) , "lambda1"=c(0),
                                  "mu"=c(0), "g"=c(0), "rep"=c(0)  
                                  ,"a_0_luscher"=c(0) ,"E2_0"=c(0) ,"DeltaE2_0"=c(0) 
                                  ,"a_1_luscher"=c(0) ,"E2_1"=c(0) ,"DeltaE2_1"=c(0) )
    count<-1
    for (dir in c( "Data" )){
        #for (dir in c( "Data" )){  
        for (msq1 in c(-4.9,-4.89,-4.85)){
            for (msq0 in c(-4.9,-4.95,-4.925,-4.98,-4.99,-5.0)){
                for (l0 in c(2.5)){  
                    for (l1 in c(2.5)){    
                        for (mu in c(5.0)){    
                            for (g in c(0)){
                                for (L in c(10,16,20,24,32,40)){
                                    for (T in c(24,32,48,128)){
                                        for (rep in c(0,1,2)){
                                            file1=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_output",
                                                          dir,T,L,msq0,msq1,l0,l1,mu,g,rep)
                                            if (file.exists(file1)){
                                                
                                                
                                                
                                                columns_file<-c(1:20)  
                                                mylist<- list(  L,T)
                                                mylist  <-append(mylist, list(msq0, msq1 ))
                                                
                                                
                                                mt<-read_df(file1)
                                                
                                                fit<- get_fit_n(mt,2)
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                fit<- get_fit_n(mt,3)
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2]))
                                                #E2
                                                fit<- get_fit_n(mt,3)
                                                #mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                fit<- get_fit_n(mt,5)
                                                #mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                
                                                #E3
                                                fit<- get_fit_n(mt,7)
                                                #mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                fit<- get_fit_n(mt,8)
                                                #mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                
                                                #a_BH
                                                fit<- get_fit_n(mt,13)
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                #a_lusher, E2,deltaE2   
                                                fit<- get_fit_n(mt,20)
                                                mylist  <- append(mylist, mean_print(fit[2,1], fit[2,2]) )
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                mylist  <- append(mylist, mean_print(fit[2,3], fit[2,4]))
                                                
                                                mylist  <-append(mylist, list( l0,l1,mu,g,rep ))
                                                
                                                #a_0_lusher, E2_0,deltaE2_0   
                                                fit<- get_fit_n(mt,5)
                                                mylist  <- append(mylist, mean_print(fit[2,1], fit[2,2]) )
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                mylist  <- append(mylist, mean_print(fit[2,3], fit[2,4]))
                                                
                                                
                                                #a_1_lusher, E2_1,deltaE2_1   
                                                fit<- get_fit_n(mt,6)
                                                mylist  <- append(mylist, mean_print(fit[2,1], fit[2,2]) )
                                                mylist  <- append(mylist, mean_print(fit[1,1], fit[1,2])) 
                                                mylist  <- append(mylist, mean_print(fit[2,3], fit[2,4]))
                                                
                                                
                                                
                                                
                                                df_tot_compact[count,] <-mylist
                                                count<- count+1
                                                
                                            }
                                            
                                        }}}}}}}}}}
    
    output$summary_table<-renderDataTable( df_tot_compact )
    
})
