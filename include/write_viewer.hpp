#ifndef write_viewer_H
#define write_viewer_H



int check_layout();
void write_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int ii , const Viewphi phi  );

void read_viewer(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  ,  Viewphi &phi  );

void write_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi::HostMirror h_phip );
void read_conf_FT(FILE *f_conf,int layout_value, cluster::IO_params params, int iconf  , Viewphi::HostMirror &h_phip );

#endif
