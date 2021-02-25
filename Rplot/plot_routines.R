
set_xmargin<- function(fit_range, T){
  xmin <-fit_range[1]-7
  xmax <-fit_range[2]+7
  if (xmin<0 ){ xmin<-0}
  if (xmax>T ){ xmax<-T}
  c(xmin,xmax)
}

#####################################################################
#####################################################################

myggplot<-function(d,fit, fit_range,T,logscale="no"){
  
  gg <- ggplot(d, aes(x=d[,1], y=d[,2])) + geom_point() 
  #gg  <- gg + xlim(set_xmargin(fit_range,T) ) + ylim(fit[1,1]-15*fit[1,2], fit[1,1]+15*fit[1,2]) 
  gg <- gg +geom_errorbar(aes(ymin=d[,2]-d[,3], ymax=d[,2]+d[,3]),  width = 1)  
  gg <- gg+ labs(x = 't', y= 'y')
  # plot orizontal line with fit 
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1], xend = fit_range[2], yend = fit[1,1]) , linetype="dashed", color = "red")
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1]-fit[1,2], xend = fit_range[2], yend = fit[1,1]-fit[1,2]) , linetype="solid", color = "red")
  gg <- gg+ geom_segment(aes(x = fit_range[1], y = fit[1,1]+fit[1,2], xend = fit_range[2], yend = fit[1,1]+fit[1,2]) , linetype="solid", color = "red")
  gg <- gg+theme_bw()
  s<- sprintf("%.6f",fit[1,1])
  err<- sprintf("%.6f",fit[1,2])
  
  
  pander(paste0("  fit: $m_{eff}=",s,"\\pm",err,"$")) 
  #plot(gg)
  return(gg)
}
#####################################################################
#####################################################################

my_fit_ggplot<-function(d,fit_par, fit_range,T, logscale="no"){
  
  l<- length(d[1,])
  fit_precision<- 2 #(l -2)/3  # the number of x of the fits
  mydf <-data.frame('x'=c(0), 'y'=c(0), 'err'=c(0)
                    ,'xfit'=c(0), 'fit'=c(0), 'errfit'=c(0) )
  mydf<- mydf[-1,]
  # 
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
  if (logscale=="yes"){
    mydf[,3]<- mydf[,3]/mydf[,2]
    mydf[,6]<- mydf[,6]/mydf[,5]
    mydf<-mutate_at(mydf,c(2,5) ,function(x) log10(x))
  }
  #gg <- gg+ scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #            labels = trans_format("log10", math_format(10^.x)))
  
  gg <- ggplot(mydf, aes(x=x, y=y)) + geom_point() 
   
  gg <- gg +geom_errorbar(data=mydf, mapping=aes(x=x, ymin=y-err, ymax=y+err),
                          width = 0.3,inherit.aes = FALSE)  
  #
  
  gg <- gg +geom_ribbon( mapping=aes(x=xfit, ymin=fit-errfit,ymax=fit+errfit ),
                         color="darkgreen",alpha=0.3,fill="darkgreen",
                         inherit.aes = FALSE) 
  gg <- gg+ geom_line( aes(x=fit_range[1]), color="red", linetype="dashed") 
  gg <- gg+ geom_line( aes(x=fit_range[2]), color="red", linetype="dashed") 
  
  
  # 
  #gg <- gg+ labs(x = TeX('x_0/a'), y= TeX('$c(x_0/a)$'))
  # 
  # 
  # gg <- gg+theme_bw()
  # len<-length(fit_par[1,])  /2-1
  # for(i in c(1:len )  ){
  #   if(! is.na(fit_par[1,i*2])) {
  #     s<- sprintf("P[%d]=%.6f ", i,fit_par[1,i*2-1])
  #     err<- sprintf("%.6f",fit_par[1,i*2])
  #     pander(paste0("$",s,"\\pm ",err,"$ ")) 
  #   }
  # }
  
  
  return(gg)
} 
#####################################################################
#####################################################################
many_fit_ggplot<-function(d,fit_par, fit_range,T, logscale="no", g, mylabel){
  
  l<- length(d[1,])
  fit_precision<- 2 #(l -2)/3  # the number of x of the fits
  mydf <-data.frame('x'=c(0), 'y'=c(0), 'err'=c(0)
                    ,'xfit'=c(0), 'fit'=c(0), 'errfit'=c(0) )
  mydf<- mydf[-1,]
  # 
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
  if (logscale=="yes"){
    mydf[,3]<- mydf[,3]/mydf[,2]
    mydf[,6]<- mydf[,6]/mydf[,5]
    mydf<-mutate_at(mydf,c(2,5) ,function(x) log10(x))
  }
  #gg <- gg+ scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
  #            labels = trans_format("log10", math_format(10^.x)))
  
  gg <- g + geom_point(data=mydf,mapping=aes(x=x, y=y,color=mylabel),inherit.aes = FALSE)

  gg <- gg +geom_errorbar(data=mydf, mapping=aes(x=x, ymin=y-err, ymax=y+err,color=mylabel),
                          width = 0.3,inherit.aes = FALSE)  
  #
  
  gg <- gg +geom_ribbon( data=mydf, mapping=aes(x=xfit, ymin=fit-errfit,ymax=fit+errfit
                                                ,color=mylabel,fill=mylabel)
                         #, color="darkgreen"  
                         ,alpha=0.3
                         #,fill="darkgreen"    
                         ,inherit.aes = FALSE) 
  gg <- gg+ geom_line(data=mydf, aes(x=fit_range[1],y=y,  color=mylabel), linetype="dashed") 
  gg <- gg+ geom_line( data=mydf ,aes(x=fit_range[2],y=y, color=mylabel), linetype="dashed") 
  #gg  <- gg + xlim(set_xmargin(fit_range,128/2) ) + ylim(-2e+4, 1e+4) 
  
  #gg<- gg +geom_text(data=mydf, aes(x=x,y=y), label=mylabel) 
  #gg <- gg+ labs(x = TeX('x_0/a'), y= TeX('$c(x_0/a)$'))
  # 
  # 
   gg <- gg+theme_bw()
  # len<-length(fit_par[1,])  /2-1
  # for(i in c(1:len )  ){
  #   if(! is.na(fit_par[1,i*2])) {
  #     s<- sprintf("P[%d]=%.6f ", i,fit_par[1,i*2-1])
  #     err<- sprintf("%.6f",fit_par[1,i*2])
  #     pander(paste0("$",s,"\\pm ",err,"$ ")) 
  #   }
  # }
  # 
  
  return(gg)
} 
#####################################################################
#####################################################################

residual<-function(d,fit_par, fit_range,T, logscale="no"){
  
  
  
  
  gg <- ggplot(d, aes(x=d[,1], y=(d[,2] -d[,4] )   ) ) + geom_point() 
  gg <- gg +geom_errorbar(aes(ymin=d[,2]-d[,4]-d[,3], ymax=d[,2]-d[,4]+d[,3]),  width = 0.3)  
  #
  
  #gg <- gg +geom_ribbon( aes(x=xfit, ymin=fit-errfit,ymax=fit+errfit ), color="red",alpha=0.3) 
  gg <- gg+ geom_line( aes(x=fit_range[1]), color="gray", linetype="dashed") 
  gg <- gg+ geom_line( aes(x=fit_range[2]), color="gray", linetype="dashed") 

  
  # 
  #gg <- gg+ labs(x = TeX('x_0/a'), y= TeX('$c(x_0/a)$'))
  # 
  # u
  gg <- gg+theme_bw()
  
  return(gg)
} 

print_fit<- function(fit_par){
len<-length(fit_par[1,])  /2-1
 for(i in c(1:len )  ){
   if(! is.na(fit_par[1,i*2])) {
     s<- sprintf("P[%d]=%.6f ", i,fit_par[1,i*2-1])
     err<- sprintf("%.6f",fit_par[1,i*2])
     pander(paste0("$",s,"\\pm ",err,"$ ")) 
   }
 }
 
}