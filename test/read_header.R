



################################################################################
################################################################################

read_header<- function(to.read){
L<-readBin(to.read, integer(),n = 4, endian = "little")
L
#formulation<-readBin(con=to.read, what=raw(),size=100 , n = 1, endian = "little")
formulation<- readChar(to.read, nchars=100, useBytes = 100)
formulation

msq0<-readBin(to.read, double(),n = 1, endian = "little")
msq1<-readBin(to.read, double(),n = 1, endian = "little")

l0<-readBin(to.read, double(),n = 1, endian = "little")
l1<-readBin(to.read, double(),n = 1, endian = "little")

mu<-readBin(to.read, double(),n = 1, endian = "little")
g<-readBin(to.read, double(),n = 1, endian = "little")


metropolis_local_hit<-readBin(to.read, integer(),n = 1, endian = "little")
metropolis_global_hit<-readBin(to.read, integer(),n = 1, endian = "little")
metropolis_delta<-readBin(to.read, double(),n = 1, endian = "little")

cluster_hit<-readBin(to.read, integer(),n = 1, endian = "little")
cluster_min_size<-readBin(to.read, double(),n = 1, endian = "little")

seed<-readBin(to.read, integer(),n = 1, endian = "little")
replica<-readBin(to.read, integer(),n = 1, endian = "little")

ncorr<-readBin(to.read, integer(),n = 1, endian = "little")
size<-readBin(to.read, integer(),n = 1, size = 8, endian = "little")

head_size<-seek(to.read, where = 0, origin = "current", rw = "read")
end_file<- seek(to.read, where = 0, origin = "end", rw = "read")
end_file<-seek(to.read, where = NA, origin = "current", rw = "read")

confs<-(end_file-head_size)/(size*8+4)
header<- list("L"=L,"formulation"=formulation, "msq0"=msq0, "msq1"=msq1, "l0"=l0,
              "l1"=l1, "mu"=mu, "g"=g, "metropolis_local_hit"=metropolis_local_hit, 
              "metropolis_global_hit"=metropolis_global_hit,
              "cluster_hit"=cluster_hit, "cluster_min_size"=cluster_min_size,
              "seed"=seed, "replica"=replica, "ncorr"=ncorr, "size"=size,
              "head_size"=head_size, "end_file"=end_file, "confs"=confs)
seek(to.read, where = head_size, origin = "start", rw = "read")
return(header)
}
