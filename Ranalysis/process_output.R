library(hadron)
if(!exists("foo", mode="function")) source("read_header.R")

to.read = file("/home/marco/analysis/phi4/tuning_masses/G2t_T48_L20_msq0-4.925000_msq1-4.850000_l02.500000_l12.500000_mu5.000000_g0.000000_rep123_bin10000", "rb")

#read the header in to the structure header
header<-read_header(to.read)

###############################################################
#read the data and store in d[ correlator, time,  conf_number  ]
configurations <- list()
d<-array(dim = c(header$ncorr, header$L[1], header$confs) )

for (iconf in  c(1:header$confs)){
  configurations<-append(configurations, readBin(to.read, integer(),n = 1, endian = "little"))
  for(t in c(1:header$L[1] -1) ){
    for (corr in  c(1:header$ncorr)){
          d[corr, t, iconf]<-readBin(to.read, double(),n = 1, endian = "little")
      }
  }
}
###############################################################


#We can put the correlator 1 and 2 which are the twopt
#<phi0(t) phi0(0)>
#<phi1(t) phi1(0)>
#in the cf hadron container and compute the effective mass
mycf<-cf()
for(i in c(12)) {
  mycf_tmp <- cf_meta(nrObs =1, Time = header$L[1], nrStypes = 1)
  mycf_tmp <- cf_orig(mycf_tmp, cf = t(d[i, ,]))
  mycf_tmp <- symmetrise.cf(mycf_tmp, sym.vec = c(1))
  mycf <-c(mycf, mycf_tmp)


  boot.R <- 150
  boot.l <- 1
  seed <- 1433567
  cfb <- bootstrap.cf(cf=mycf_tmp, boot.R=boot.R, boot.l=boot.l, seed=seed)


  fit_sample <- new_matrixfit(cfb, 2, 10, model = 'single')
  plot(fit_sample, xlab="t" , ylab="$m_{eff}$" )
  residual_plot(fit_sample, ylim = c(1/1.05, 1.05))
  
}

