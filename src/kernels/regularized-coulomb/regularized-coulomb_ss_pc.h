/* Interaction Kernels */
#ifndef H_K_REGULARIZED_COULOMB_SS_PC_LAGRANGE_H
#define H_K_REGULARIZED_COULOMB_SS_PC_LAGRANGE_H
 
#include "../../struct_kernel.h"


void K_RegularizedCoulomb_SS_PC_Lagrange(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster,
        double *target_x, double *target_y, double *target_z, double *target_charge, double *cluster_weight,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct kernel *kernel, double *potential, int gpu_async_stream_id);

void K_RegularizedCoulomb_SS_PC_Hermite(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster, int total_number_interpolation_points,
        double *target_x, double *target_y, double *target_z, double *target_charge,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge, double *cluster_weight,
        struct kernel *kernel, double *potential, int gpu_async_stream_id);


#endif /* H_K_REGULARIZED_COULOMB_SS_PC_LAGRANGE_H */
