
mean_print<-function(ave,err){
  if( !is.numeric(ave) | is.na(ave)  ){
    s=sprintf("NA")
    return(s)
  }
  if(!is.numeric(err) | is.na(err) ){
    s=sprintf("%g",ave)
    return(s)
  }  
  
  if(err<0){
    s=sprintf("%g(%g)",ave,err)
    return(s)
  }
  if(ave<0)
    ave1<- -ave
  else
    ave1<-ave
  
  a<-(log10(ave1))
  e<-(log10(err+err/1000))
  if ( a<4  & a>-4){
    we<-(err/10^(as.integer(e-2)))
    if(e<0){
      e<-as.integer(e)
      format=sprintf("%%.%df(%%.0f)",-e+2)
      s=sprintf(format,ave1,we);
      
    }
    else if(e>1){
      s=sprintf("%.0f(%.0f)",ave1,err);
      
    }
    else{
      s=sprintf("%.1f(%.1f)",ave1,err);
      
    }
    
  }
  else{
    a<-as.integer(a)
    e<-as.integer(e)
    
    wm<-( ave1/10^(e))
    we<-(err/10^(e))
    s=sprintf("%.1f(%.1f)e%+-d",wm,we,e);
  }
  if(ave<0)
    s=sprintf("-%s",s);
  
  return(s)
  
}
ave=1.4
err=0.1*1000/1001
mean_print(ave,err)
