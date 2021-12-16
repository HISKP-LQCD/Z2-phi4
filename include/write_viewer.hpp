#ifndef write_viewer_H
#define write_viewer_H



int check_layout();
void write_header(FILE *f_conf, cluster::IO_params params );
template <typename T>
void error_header(FILE *f_conf, T expected, const char *message);
void check_header(FILE *f_conf, cluster::IO_params &params );

void write_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int ii , const Viewphi phi  );
void read_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  ,  Viewphi &phi  );

void write_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi::HostMirror h_phip );
void read_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi &phip );

void write_single_conf_FT_complex(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf,  manyphi::HostMirror &h_phip );
void read_single_conf_FT_complex(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , manyphi::HostMirror &h_phip );

void write_conf_FT_complex(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf, manyphi::HostMirror h_phip);
void read_conf_FT_complex(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf, manyphi::HostMirror &phip);

#endif
