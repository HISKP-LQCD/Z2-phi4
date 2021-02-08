library(dplyr)



read_df<- function(file){
  columns_file<-c(1:20)  
  lc<- length(columns_file)+1
  
  all_file <- read.table(file,header=FALSE,fill = TRUE ,
                         blank.lines.skip=FALSE,skip=0, comment.char = "", 
                         col.names = columns_file)
  # get the blank lines
  #add to the list an extra fictitious blank line
  ix <- c(which(all_file[,1]==""),nrow(all_file) )
  # get the blank line after one blank line
  ixp <- c(which(all_file[ix+1,1]=="")  )
  #ixp <- c(0,which(all_file[ix+1,1]=="") )
  ixm<- c(0,ix[-ixp])
  # enumerate  with blocks 
  iblock<-rep(1:length(diff(ixm)),diff(ixm))
  #m <- cbind(all_file,rep(1:length(diff(ix)),diff(ix)))
  #remove blank lines
  #m <- m[!(m[,1]==""),]

  mt<-cbind("index"=iblock, all_file)
  mt <- mt[!(mt[,2]==""),]
  if (mt[1,1]!=1)
    mt[,1]<-mt[,1]-mt[1,1]+1
  return(mt)
}

get_block_n<- function(df,n){
  bo<-which(df[,1]==(n*2-1))
  data <- df[bo,-1]
  if (!is_empty(grep("^#", data[,1]) ) ) {
    data<-data[-grep("^#", data[,1])  ,  ]
  }
  data<-mutate_all(data, function(x) as.numeric(as.character(x)))
  return(data)
}
get_fit_n<- function(df,n){
  be<-which(df[,1]==(n*2))
  data <- df[be,-1]
  if (!is_empty(grep("^#", data[,1]) ) ){
    data<-data[-grep("^#", data[,1])  ,  ]
  }
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

#  dir <- "/home/marco/analysis/phi4/tuning_masses/out" 
# # #dir <- "Data" 
# # 
#  file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_meff_correlators",
#               dir, 128, 20,  -4.925,  -4.85,  2.5,  2.5,  5,  0,  0)
#  mt<-read_df(file)
#  file=sprintf("%s/G2t_T%d_L%d_msq0%.6f_msq1%.6f_l0%.6f_l1%.6f_mu%.6f_g%.6f_rep%d_output",
#               dir, 48, 20,  -4.925,  -4.85,  2.5,  2.5,  5,  0,  0)
#  
#  mt<-read_df(file)
