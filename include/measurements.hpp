#ifndef mesuraments_H
#define mesuraments_H
 
 
void write_header_measuraments(FILE *f_conf, cluster::IO_params params );
 
 
double *compute_magnetisations_serial( Viewphi::HostMirror phi,  cluster::IO_params params);
double *compute_magnetisations( Viewphi phi,  cluster::IO_params params);
double  *compute_G2( Viewphi::HostMirror phi, cluster::IO_params params );
void  compute_G2t_serial_host(Viewphi::HostMirror phi, cluster::IO_params params , FILE *f_G2t );
void  compute_G2t(const Viewphi &phi, cluster::IO_params params , FILE *f_G2t ,int iconf);

#endif