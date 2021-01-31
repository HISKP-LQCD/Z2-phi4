library(dplyr)



read_df<- function(file){
  columns_file<-c(1:20)  
  lc<- length(columns_file)+1
  
  all_file <- read.table(file,header=FALSE,fill = TRUE ,
                         blank.lines.skip=FALSE,skip=1, comment.char = "", 
                         col.names = columns_file)
  # get the blank lines
  ix <- c(0,which(all_file[,1]==""),nrow(all_file)  )
  # get the blank line after one blank line
  ixp <- c(0,which(all_file[ix+1,1]==""),nrow(all_file)  )
  ix<- ix[-ixp]
  # enumerate  with blocks 
  m <- cbind(all_file,rep(1:length(diff(ix)),diff(ix)))
  #remove blank lines
  m <- m[!(m[,1]==""),]
  
  mt<-cbind("index"=rep(1:length(diff(ix)),diff(ix)), all_file)
  mt <- mt[!(mt[,2]==""),]
  return(mt)
}

get_block_n<- function(df,n){
  bo<-which(df[,1]==(n*2-1))
  data <- df[bo,-1]
  data<-data[-grep("^#", data[,1])  ,  ]
  data<-mutate_all(data, function(x) as.numeric(as.character(x)))
  return(data)
}
get_fit_n<- function(df,n){
  be<-which(df[,1]==(n*2))
  data <- df[be,-1]
  data<-data[-grep("^#", data[,1])  ,  ]
  data<-mutate_all(data, function(x) as.numeric(as.character(x)))
  return(data)
}
get_plateaux_range<-function(df,n){
  l<-grep("fit",df[,3])
  a1<-gsub("\\[","c\\(", df[l,5][n])
  a2<-gsub("\\]","\\)", a1)
  fit_range <- eval(parse(text=a2))
  
  return(fit_range)
}

