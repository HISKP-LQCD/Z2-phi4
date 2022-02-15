#ifndef mesuraments_H
#define mesuraments_H


void write_header_measuraments(FILE* f_conf, cluster::IO_params params, int ncorr = Ncorr);


double* compute_magnetisations_serial(Viewphi::HostMirror phi, cluster::IO_params params);
double* compute_magnetisations(Viewphi phi, cluster::IO_params params);
void compute_energy(Viewphi phi, cluster::IO_params params, FILE *f);
double* compute_G2(Viewphi::HostMirror phi, cluster::IO_params params);
void  compute_G2t_serial_host(Viewphi::HostMirror phi, cluster::IO_params params, FILE* f_G2t);
void  compute_G2t(Viewphi::HostMirror h_phip, cluster::IO_params params, FILE* f_G2t, int iconf);

//void  compute_FT(const Viewphi phi, cluster::IO_params params ,  int iconf, Viewphi &phip);
//void  compute_FT_complex(const Viewphi phi, cluster::IO_params params ,  int iconf, complexphi &phip);

void  parallel_measurement(Viewphi phip, Viewphi::HostMirror h_phip, cluster::IO_params params, FILE* f_G2t, FILE* f_checks, int iconf);
//void  parallel_measurement_complex(complexphi phip, complexphi::HostMirror h_phip, complexphi s_phip,  cluster::IO_params params , FILE *f_G2t, FILE *f_checks  , int iconf);    
// void  parallel_measurement_complex(complexphi phip, complexphi::HostMirror h_phip, complexphi s_phip, complexphi phi2p,  cluster::IO_params params , FILE *f_G2t, FILE *f_checks  , int iconf); 
void  parallel_measurement_complex(manyphi mphip, manyphi::HostMirror h_mphip, cluster::IO_params params, FILE* f_G2t, FILE* f_checks, int iconf);

void  compute_checks(Viewphi::HostMirror h_phip, cluster::IO_params params, FILE* f, int iconf);
void  compute_checks_complex(manyphi::HostMirror h_phip, cluster::IO_params params, FILE* f, int iconf);

void check_spin(Viewphi phi, cluster::IO_params params);

void smearing_field(Viewphip& sphi, Viewphi& phi, cluster::IO_params params);

#endif
