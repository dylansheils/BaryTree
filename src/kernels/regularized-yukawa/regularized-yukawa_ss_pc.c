#include <math.h>
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "regularized-yukawa_ss_pc.h"


void K_RegularizedYukawa_SS_PC_Lagrange(int number_of_targets_in_batch,
        int number_of_interpolation_points_in_cluster, int starting_index_of_target, int starting_index_of_cluster,
        double *target_x, double *target_y, double *target_z, double *target_charge,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge, double *cluster_weight,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{
    double kappa=run_params->kernel_params[0];
    double epsilon=run_params->kernel_params[1];

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, target_charge, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, cluster_weight, potential)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        int ii = starting_index_of_target + i;
        double temporary_potential = 0.0;

        double tx = target_x[ii];
        double ty = target_y[ii];
        double tz = target_z[ii];
        double tq = target_charge[ii];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            int jj = starting_index_of_cluster + j;
            double dx = tx - cluster_x[jj];
            double dy = ty - cluster_y[jj];
            double dz = tz - cluster_z[jj];
            double r  = sqrt(dx*dx + dy*dy + dz*dz);

            temporary_potential += (cluster_charge[jj] - tq * cluster_weight[jj]) * exp(-kappa*r) /sqrt(r*r+epsilon*epsilon);
        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[ii] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}


