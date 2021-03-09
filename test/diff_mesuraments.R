if(!exists("foo", mode="function")) source("../Ranalysis/read_header.R")

f<-paste("data/G2t_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0_reference")
f1<-"data/G2t_T8_L4_msq0-4.900000_msq1-4.900000_l02.500000_l12.500000_mu5.000000_g0.000000_rep0"
to.read = 
  file(f, "rb")
to.read1 = 
  file(f1, "rb")

header<-read_header(to.read)
header1<-read_header(to.read1)

###############################################################
#read the data and store in d[ correlator, time,  conf_number  ]
configurations <- list()
configurations1 <- list()
d<-array(dim = c(header$ncorr, header$L[1], header$confs) )
d1<-array(dim = c(header1$ncorr, header1$L[1], header1$confs) )

if (header$ncorr != header1$ncorr){
  print("error in the number of correlators")
  print(f) 
  print(header$ncorr)
  print(f1)
  print(header1$ncorr)
}
  

if (header$L[1] != header1$L[1])
  print("error L[1]")


if (header$confs != header1$confs)
  cat("error confs",header$confs, "  ",header1$confs ,"\n")

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
   for(t in c(1:header$L[1] ) ){
    for (corr in  c(1:header$ncorr)){
       if ( abs(d[corr, t, iconf] - d1[corr, t, iconf])> 1e-10 ){
        y<- "error"
        cat("error in cor=",corr, "  time=",t, "  conf=",iconf,"\n")
        cat(d[corr, t, iconf] , "   ", d1[corr, t, iconf],"\n")
       }
      
    }
  }
}
print(y)
if (y=="error"){
  stop("Something erroneous has occurred!")
}

