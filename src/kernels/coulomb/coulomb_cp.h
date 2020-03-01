/* Interaction Kernels */
#ifndef H_K_COULOMB_CP_H
#define H_K_COULOMB_CP_H
 
#include "../../struct_kernel.h"


void K_Coulomb_CP_Lagrange(int number_of_sources_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_sources, int starting_index_of_cluster,
        double *source_x, double *source_y, double *source_z, double *source_q, double *source_w,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct kernel *kernel, int gpu_async_stream_id);

void K_Coulomb_CP_Hermite(int number_of_sources_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_sources, int starting_index_of_cluster,
        double *source_x, double *source_y, double *source_z, double *source_q, double *source_w,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct kernel *kernel, int gpu_async_stream_id);


#endif /* H_K_COULOMB_CP_H */
