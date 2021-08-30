#library(hadron)
if(!exists("foo", mode="function")) source("read_header.R")
args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)!=2) {
	          stop("usage: Rscript diff_gauge conf1 conf1 ", call.=FALSE)
}

f<-paste("./G2t_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0_reference")
#f<-paste("../build/main/data/G2t_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0_ref_mom1")
f1<-"./G2t_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0"

files<-c("data/G2t_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0",
         "data/checks_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0")
files<-c("data/G2t_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0",
         "data/checks_T20_L10_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0"
#	 "data/G2t_T64_L20_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0",
#	 "data/checks_T64_L20_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0"
	 )
#files<-c(args[1])
#refs<-c(args[2])
#for (file in files){
#  cat("considering file ",file,"\n")
#  f=sprintf("%s_reference",file)
#  f1=sprintf("%s",file)
  
cat("comparing:\n",args[1],"\n",args[2],"\n")
to.read =   file(args[2], "rb")
to.read1 =   file(args[1], "rb")

suppressWarnings(header<-read_header(to.read))
suppressWarnings(header1<-read_header(to.read1))

###############################################################
#read the data and store in d[ correlator, time,  conf_number  ]
configurations <- list()
configurations1 <- list()
d<-array(dim = c(header$ncorr, header$L[1], header$confs) )
d1<-array(dim = c(header1$ncorr, header1$L[1], header1$confs) )

if (header$ncorr != header1$ncorr){
  print("error in the number of correlators")
  print(args[2]) 
  print(header$ncorr)
  print(args[1])
  print(header1$ncorr)
}
  

if (header$L[1] != header1$L[1])
  print("error L[1]")


if (header$confs != header1$confs)
  print("error confs")

for (iconf in  c(1:header$confs)){
  configurations<-append(configurations, readBin(to.read, integer(),n = 1, endian = "little"))
  for(t in c(1:header$L[1] ) ){
    for (corr in  c(1:header$ncorr)){
      d[corr, t, iconf]<-readBin(to.read, double(),n = 1, endian = "little")
      
    }
  }
}

for (iconf in  c(1:header1$confs)){
  configurations1<-append(configurations1, readBin(to.read1, integer(),n = 1, endian = "little"))
  for(t in c(1:header1$L[1] ) ){
    for (corr in  c(1:header1$ncorr)){
      d1[corr, t, iconf]<-readBin(to.read1, double(),n = 1, endian = "little")
      
      
    }
  }
}




y<-"ok"
for (iconf in  c(1:header$confs)){
	   if(y=="error") break;
   for(t in c(1:header$L[1] ) ){
	   if(y=="error") break;
    for (corr in  c(1:header$ncorr)){
       if ( abs(d[corr, t, iconf] - d1[corr, t, iconf])> 1e-8 ){
        y<- "error"
        cat("error in cor=",corr, "  time=",t, "  conf=",iconf," diff=", abs(d[corr, t, iconf] - d1[corr, t, iconf]) ," ratio=",  d[corr, t, iconf] /d1[corr, t, iconf],"\n")
       }
      
    }
  }
}
print(y)
if (y=="error"){
  stop("Something erroneous has occurred!")
}
#}

