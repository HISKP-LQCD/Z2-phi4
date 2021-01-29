library(knitr)

knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(fig.align='center',       comment='')
knitr::opts_chunk$set(fig.width=6, fig.height=4) 
require(ggplot2)
require(scales) # to access break formatting functions
library(latex2exp)
library(pander)

panderOptions('knitr.auto.asis', FALSE)

set_xmargin<- function(fit_range, T){
  xmin <-fit_range[1]-7
  xmax <-fit_range[2]+7
  if (xmin<0 ){ xmin<-0}
  if (xmax>T ){ xmax<-T}
  c(xmin,xmax)
}
#####################################################################
#####################################################################

myggplot<-function(d,fit, fit_range,n,T){
  
  gg <- ggplot(d, aes(x=x, y=y)) + geom_point() 
  gg  <- gg + xlim(set_xmargin(fit_range,T) ) + ylim(fit[1,1]-15*fit[1,2], fit[1,1]+15*fit[1,2]) 
  gg <- gg +geom_errorbar(aes(ymin=y-err, ymax=y+err),  width = 1)  
  gg <- gg+ labs(x = TeX('x_0/a'), y= TeX('$m_{eff}$'))
  # plot orizontal line with fit 
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1], xend = fit_range[2], yend = fit[1,1]) , linetype="dashed", color = "red")
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1]-fit[1,2], xend = fit_range[2], yend = fit[1,1]-fit[1,2]) , linetype="solid", color = "red")
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1]+fit[1,2], xend = fit_range[2], yend = fit[1,1]+fit[1,2]) , linetype="solid", color = "red")
  gg <- gg+theme_bw()
  s<- sprintf("%.6f",fit[1,1])
  err<- sprintf("%.6f",fit[1,2])
  
  
  pander(paste0("  fit: $m_{eff}=",s,"\\pm",err,"$")) 
  plot(gg)
}
#####################################################################
#####################################################################

my_fit_ggplot<-function(d,fit_par, fit_range,n,T){
  
  l<- length(d[1,])
  fit_precision<- (l -2)/3  # the number of x of the fits
  mydf <-data.frame('x'=c(0), 'y'=c(0), 'err'=c(0)
                    ,'xfit'=c(0), 'fit'=c(0), 'errfit'=c(0) )
  mydf<- mydf[-1,]
  
  colx <- c(1,c(1:fit_precision*3))[-2] # 1, 6, 9, 12,..#columns of the x
  colf <- c(4,c(1:fit_precision*3+1))[-2]# 4, 7, 10, 13,..#columns of the fit
  colferr <- c(5,c(1:fit_precision*3+2))[-2]# 5, 8, 11, 14,..#columns of the fit
  count<-1
  for(i in c(1:fit_precision)) {
    for (t in c(1: length(d[,1])) ){
      mylist  <-  list(d[t,1],d[t,2], d[t,3]  ) 
      mylist  <- append(mylist, list( d[t,colx[i]],d[t,colf[i]], d[t,colferr[i]]  ) )
      mydf[count,]<- mylist
      count<-count+1
    }
    
  }
  #get the extra column in a single dataframe 
  #mydf <-data.frame('x'=c(d$x,d$x), 'y'=c(d$y,d$y), 'err'=c(d$err,d$err),  'xfit'=c(d$x,d$xp), 'fit'=c(d$fit,d$fitp), 'errfit'=c(d$errfit,d$errfitp) )  
  gg <- ggplot(mydf, aes(x=x, y=y)) + geom_point() 
  #gg <- gg+ scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #              labels = trans_format("log10", math_format(10^.x)))
  gg <- gg +geom_errorbar(aes(ymin=y-err, ymax=y+err),  width = 0.3)  
  
  gg <- gg +geom_ribbon( aes(x=xfit, ymin=fit-errfit,ymax=fit+errfit ), color="red",alpha=0.3) 
  gg <- gg+ geom_line( aes(x=fit_range[1]), color="gray", linetype="dashed") 
  gg <- gg+ geom_line( aes(x=fit_range[2]), color="gray", linetype="dashed") 
  
  #gg <- gg +geom_line( aes(x=x, ymin=fit), color="red") 
  
  gg <- gg+ labs(x = TeX('x_0/a'), y= TeX('$c(x_0/a)$'))
  
  
  gg <- gg+theme_bw()
  len<-length(fit_par[1,])  /2-1
  for(i in c(1:len )  ){
     if(! is.na(fit_par[1,i*2])) {
       s<- sprintf("P[%d]=%.6f ", i,fit_par[1,i*2-1])
       err<- sprintf("%.6f",fit_par[1,i*2])
       pander(paste0("$",s,"\\pm ",err,"$ ")) 
     }
  }
    
  #s<- sprintf("%.6f",fit_par[1,1])
  #err<- sprintf("%.6f",fit_par[1,2])
  
  
  #pander(paste0("  fit: $m_{eff}=",s,"\\pm",err,"$")) 
  plot(gg)
}
#####################################################################
#####################################################################
plot_two_comp<- function(file,df,T,L,msq0,msq1,l0,l1,mu,g,rep ){
  columns_file<-c(1:20)  
  block <- read.table(file,header=FALSE,fill = TRUE , blank.lines.skip=TRUE,skip=1,
                      col.names =columns_file)
  block_full <- read.table(file,header=FALSE,fill = TRUE , blank.lines.skip=TRUE,skip=1, comment.char = "",   col.names = columns_file)
  l<-grep("fit",block_full[,2])
  
  meff <- data.frame('ave'=c(0), 'err'=c(0))
  len<- length(block[,1])
  count<-1
  mylist<- list(  L,T)
  
  cat('\n\n#### Mass  \n\n')
  for (n in c(2,3)){
    if (len >= (n*(T/2)-1)){
      cat('index n=',n%%2,'\n\n')
      a1<-gsub("\\[","c\\(", block_full[l,4][n])
      a2<-gsub("\\]","\\)", a1)
      fit_range <- eval(parse(text=a2))
      
      # store in fit the fit value
      fit <- block[(n*(T/2)),]
      
      data <- na.omit(block[((n-1)*(T/2)+1):(n*(T/2)-1),1:3])
      #plot data
      d <- data.frame("x"=data[[1]],"y"=data[[2]],"err"=data[[3]])
      myggplot(d,fit,fit_range,n,T/2) 
      #meff[count,]<- list(fit[1,1],fit[1,2])
      mylist  <- append(mylist, list(fit[1,1],fit[1,2]) )
      count <- count+1
    }
    else mylist <-append(mylist, list("NaN","NaN") )
    
  }
  mylist  <-append(mylist, list(msq0, msq1, l0,l1,mu,g,rep ))
  
  
  cat('\n\n#### Two particle energy  \n\n')
  for (n in c(5,6,7)){
    if (len >= (n*(T/2)-1)){
      cat('index n=',n-5,'\n\n')
      a1<-gsub("\\[","c\\(", block_full[l,4][n])
      a2<-gsub("\\]","\\)", a1)
      fit_range <- eval(parse(text=a2))
      
      # store in fit the fit value
      fit <- block[(n*(T/2)),]
      
      data <- na.omit(block[((n-1)*(T/2)+1):(n*(T/2)-1),1:3])
      #plot data
      d <- data.frame("x"=data[[1]],"y"=data[[2]],"err"=data[[3]])
      myggplot(d,fit,fit_range,n,T/2) 
      
      mylist  <- append(mylist, list(fit[1,1],fit[1,2]) )
      count <- count+1
    }
    else mylist  <-append(mylist, list("NaN","NaN")) 
  }
  
  cat('\n\n#### Three particle energy  \n\n')
  for (n in c(8,9,10) ){    #
    if (len >= (n*(T/2)-1)){
      cat('index n=',n-8,'\n\n')
      a1<-gsub("\\[","c\\(", block_full[l,4][n])
      a2<-gsub("\\]","\\)", a1)
      fit_range <- eval(parse(text=a2))
      
      # store in fit the fit value
      fit <- block[(n*(T/2)),]
      
      data <- (block[((n-1)*(T/2)+1):(n*(T/2)-1),])
      
      d <- data.frame("x"=data[[1]],"y"=abs(data[[2]]),"err"=abs(data[[3]]),
                      "fit"=abs(data[[4]]),"errfit"=abs(data[[5]]), 
                      "xp"=abs(data[[6]]), "fitp"=abs(data[[7]]), "errfitp"= abs(data[[8]]) )
      
      #my_fit_ggplot(d,fit,fit_range,n,T/2)
      
      mylist  <-append(mylist, list(fit[1,1],fit[1,2]) )
    }
    else mylist  <-append(mylist, list("NaN","NaN") )
    count<-count+1
  }
  
  
  cat('\n\n#### C4_BH  \n\n')
  for (n in c(11,12,13) ){   # c(11,12,13,14,15,16,17,18)
    if (len >= (n*(T/2)-1)){
      cat('index n=',n-11,block_full[l,1][n] ,'\n\n')
      
      a1<-gsub("\\[","c\\(", block_full[l,4][n])
      a2<-gsub("\\]","\\)", a1)
      fit_range <- eval(parse(text=a2))
      
      # store in fit the fit value
      fit <- block[(n*(T/2)),]
      
      data <- (block[((n-1)*(T/2)+1):(n*(T/2)-1),1:5])
      
      d <- data.frame("x"=data[[1]],"y"=(data[[2]]),"err"=(data[[3]]),
                      "fit"=(data[[4]]),"errfit"=(data[[5]]))
      if(n==13)
      my_fit_ggplot(d,fit,fit_range,n,T/2)
      
      mylist  <-append(mylist, list(fit[1,1],fit[1,2]) )
    }
    else mylist  <-append(mylist, list("NaN","NaN") )
    count<-count+1
  }
  
  
  #append to df
  count=length(df[,1])             
  df[ count+1,]<- mylist
  return(df)
  
}
#####################################################################
# main
#####################################################################



df<- data.frame(  "L"=c(0),"T"=c(0),
                  "meff0"=c(0), "Emeff0"=c(0), "meff1"=c(0) , "Emeff1"=c(0) ,
                  "msq0"=c(0), "msq1"=c(0),
                  "lambda0"=c(0) , "lambda1"=c(0),
                  "mu"=c(0), "g"=c(0), "rep"=c(0),
                  "E2_0"=c(0),"E2_0err"=c(0),
                  "E2_1"=c(0),"E2_1err"=c(0),
                  "E2"=c(0),"E2err"=c(0),
                  "E3_0"=c(0),"E3_0err"=c(0),
                  "E3_1"=c(0),"E3_1err"=c(0),
                  "E3"=c(0),"E3err"=c(0),
                  "a_0"=c(0),"a_0err"=c(0),
                  "a_1"=c(0),"a_1err"=c(0),
                  "a_01"=c(0),"a_01err"=c(0)
)
df<-df[-1,]
count=1

for (dir in c( "/home/marco/analysis/phi4/tuning_masses/out" )){
  for (msq1 in c(-4.9,-4.89,-4.85)){
    for (msq0 in c(-4.9,-4.95,-4.925,-4.98,-4.99,-5.0)){
      for (l0 in c(2.5)){  
        for (l1 in c(2.5)){    
          for (mu in c(5.0)){    
            for (g in c(0)){
              for (L in c(10,20,40)){
                for (T in c(24,48)){
                  for (rep in c(0)){
                    file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_output",
                                 dir,T,L,msq0,msq1,l0,l1,mu,g,rep)
                    if (file.exists(file)){
                      #print(file)
                      #cat(" m0^2=", msq0,'\t m1^2=', msq1, '\n lam0=', l0, '\t lam1=',l1,'\n mu=',mu,'\t g=',g )
                      pander(paste('\n\n## T', T, 'L',L, ' '))   
                      pander(paste("$\\quad m_0^2 = ", msq0,"\\quad$"))
                      pander(paste0("$m_1^2 = ", msq1,"\\quad$"))
                      pander(paste0("$\\lambda_0^2 = ", l0,"\\quad$"))
                      pander(paste0("$\\lambda_1^2 = ", l1,"\\quad$"))
                      pander(paste0("$\\mu^2 = ", mu,"\\quad$"))
                      pander(paste0("$g^2 = ", g,"\\quad$"))
                      pander(paste0("replica = ", rep,' '))
                      cat("\n\n")
                      df<-plot_two_comp(file,df, T, L ,msq0,msq1,l0,l1,mu,g,rep )
                      
                      
                      
                    }#if file exist
                  }}}}}}}}}}