/* file minunit_example.c */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <zoltan.h>
#include <time.h>
#include <float.h>

#include "minunit.h"
#include "../src/treedriver.h"
#include "../src/treedriverWrapper.h"
#include "../src/directdriver.h"
#include "../src/struct_particles.h"


int tests_run = 0;
double sizeCheckFactor=1.0;


static char *test_direct_sum_on_10_particles_per_rank()
{

    int verbosity=0;

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int N=10;

    struct particles *sources = NULL;
    struct particles *targets = NULL;
    int *particleOrder = NULL;
    double *potential = NULL;
    double potential_engy = 0;

    sources = malloc(sizeof(struct particles));
    targets = malloc(sizeof(struct particles));
    potential = malloc(sizeof(double) * N);
    particleOrder = malloc(sizeof(int) * N);

    targets->num = N;
    targets->x = malloc(targets->num*sizeof(double));
    targets->y = malloc(targets->num*sizeof(double));
    targets->z = malloc(targets->num*sizeof(double));
    targets->q = malloc(targets->num*sizeof(double));
    targets->order = malloc(targets->num*sizeof(int));

    sources->num = N;
    sources->x = malloc(sources->num*sizeof(double));
    sources->y = malloc(sources->num*sizeof(double));
    sources->z = malloc(sources->num*sizeof(double));
    sources->q = malloc(sources->num*sizeof(double));
    sources->w = malloc(sources->num*sizeof(double));
    sources->order = malloc(sources->num*sizeof(int));


    for (int i=0; i<targets->num; i++){

        if (rank==0){
            // rank 0 gets the positive x=y=z line
            targets->x[i]=1.0*i;
            targets->y[i]=1.0*i;
            targets->z[i]=1.0*i;
            targets->q[i]=1.0*i;
            targets->order[i] = i;

            sources->x[i]=1.0*i;
            sources->y[i]=1.0*i;
            sources->z[i]=1.0*i;
            sources->q[i]=1.0*i;
            sources->w[i]=1.0;
            sources->order[i] = i;
        } else if (rank==1){
            // rank 1 gets the negative x=y=z line
            targets->x[i]=1.0*(i+N);
            targets->y[i]=1.0*(i+N);
            targets->z[i]=1.0*(i+N);
            targets->q[i]=1.0*(i+N);
            targets->order[i] = i;

            sources->x[i]=1.0*(i+N);
            sources->y[i]=1.0*(i+N);
            sources->z[i]=1.0*(i+N);
            sources->q[i]=1.0*(i+N);
            sources->w[i]=1.0;
            sources->order[i] = i;
        }

        potential[i]=0.0;
        }

    int max_per_leaf=100;
    int max_per_batch=100;
    double time_tree[9];

    char *kernelName, *singularityHandling, *approximationName;

    kernelName="coulomb";
    singularityHandling="skipping";
    approximationName="lagrange";
    int tree_type=1;
    double kappa=0.5;

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential, time_tree);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<targets->num; i++){
        int iloc = i+rank*targets->num;
        double trueValue=0.0;
        for (int j=0; j<iloc; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += j/(r);
        }
        for (int j=iloc+1; j<2*targets->num; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += j/(r);
        }

        if (verbosity>0) printf("Rank = %i\n", rank);
        if (verbosity>0) printf("True value = %1.7e\n", trueValue);
        if (verbosity>0) printf("Computed value = %1.7e\n\n", potential[i]);
        mu_assert("TEST test_direct_sum_on_10_particles_per_rank FAILED: Direct sum potential not correct for coulomb kernel with skipping",
                fabs(potential[i] - trueValue) < 1e-09);
    }




    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }
    singularityHandling="subtraction";
    kappa=5.5;

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential, time_tree);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<targets->num; i++){
        int iloc = (i+rank*targets->num);
        double trueValue=2.0 * M_PI * kappa * kappa * iloc;
        for (int j=0; j<iloc; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += (j - 1.0*iloc*exp(-r*r/kappa/kappa) )/(r);
        }
        for (int j=iloc+1; j<2*targets->num; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += (j - 1.0*iloc*exp(-r*r/kappa/kappa) )/(r);
        }

        if (verbosity>0) printf("Rank = %i\n", rank);
        if (verbosity>0) printf("True value = %1.7e\n", trueValue);
        if (verbosity>0) printf("Computed value = %1.7e\n\n", potential[i]);
        mu_assert("TEST test_direct_sum_on_10_particles_per_rank FAILED: Direct sum potential not correct for coulomb kernel with subtraction",
                fabs(potential[i] - trueValue) < 1e-10);
    }


    kernelName="yukawa";
    singularityHandling="skipping";

    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential, time_tree);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<targets->num; i++){
        int iloc = i+rank*targets->num;
        double trueValue=0.0;
        for (int j=0; j<iloc; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }
        for (int j=iloc+1; j<2*targets->num; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }

        if (verbosity>0) printf("Rank = %i\n", rank);
        if (verbosity>0) printf("True value = %1.7e\n", trueValue);
        if (verbosity>0) printf("Computed value = %1.7e\n\n", potential[i]);
        mu_assert("TEST test_direct_sum_on_10_particles_per_rank FAILED: Direct sum potential not correct for yukawa kernel with skipping",
                fabs(potential[i] - trueValue) < 1e-10);
    }


    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }
    singularityHandling="subtraction";

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential, time_tree);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<targets->num; i++){
        int iloc = i+rank*targets->num;
        double trueValue=4.0 * M_PI / kappa / kappa * iloc;
        for (int j=0; j<iloc; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += (j - iloc)*exp(-kappa*r)/(r);
        }
        for (int j=iloc+1; j<2*targets->num; j++){
            double r = abs(j-iloc)*sqrt(3);
            trueValue += (j - iloc)*exp(-kappa*r)/(r);
        }

        if (verbosity>0) printf("Rank = %i\n", rank);
        if (verbosity>0) printf("True value = %1.7e\n", trueValue);
        if (verbosity>0) printf("Computed value = %1.7e\n\n", potential[i]);
        mu_assert("TEST test_direct_sum_on_10_particles_per_rank FAILED: Direct sum potential not correct for yukawa kernel with subtraction",
                fabs(potential[i] - trueValue) < 1e-10);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    free(sources->x);
    free(sources->y);
    free(sources->z);
    free(sources->q);
    free(sources->w);
    free(sources->order);
    free(sources);

    free(targets->x);
    free(targets->y);
    free(targets->z);
    free(targets->q);
    free(targets->order);
    free(targets);

    free(potential);
    return 0;
}

static char * test_treecode_on_100_particles() {

    int N=100;

    int verbosity=0;

    struct particles *sources = NULL;
    struct particles *targets = NULL;
    int *particleOrder = NULL;
    double *potential = NULL;
    double potential_engy = 0;

    sources = malloc(sizeof(struct particles));
    targets = malloc(sizeof(struct particles));
    potential = malloc(sizeof(double) * N);
    particleOrder = malloc(sizeof(int) * N);

    targets->num = N;
    targets->x = malloc(targets->num*sizeof(double));
    targets->y = malloc(targets->num*sizeof(double));
    targets->z = malloc(targets->num*sizeof(double));
    targets->q = malloc(targets->num*sizeof(double));
    targets->order = malloc(targets->num*sizeof(int));

    sources->num = N;
    sources->x = malloc(sources->num*sizeof(double));
    sources->y = malloc(sources->num*sizeof(double));
    sources->z = malloc(sources->num*sizeof(double));
    sources->q = malloc(sources->num*sizeof(double));
    sources->w = malloc(sources->num*sizeof(double));
    sources->order = malloc(sources->num*sizeof(int));


    for (int i=0; i<targets->num; i++){

        targets->x[i]=1.0*i;
        targets->y[i]=1.0*i;
        targets->z[i]=1.0*i;
        targets->q[i]=1.0*i;
        targets->order[i] = i;

        sources->x[i]=1.0*i;
        sources->y[i]=1.0*i;
        sources->z[i]=1.0*i;
        sources->q[i]=1.0*i;
        sources->w[i]=1.0;
        sources->order[i] = i;

        potential[i]=0.0;
        }

    int max_per_leaf=3;
    int max_per_batch=3;
    double time_tree[9];

    char *kernelName, *singularityHandling, *approximationName;

    kernelName="coulomb";
    singularityHandling="skipping";
    int tree_type=1;
    double kappa=0.5;

    int order=2;
    double theta=0.7;

    /***********************************************/
    /******************* Test 1 ********************/
    /***********************************************/
    /********** lagrange-coulomb-skipping **********/
    /***********************************************/
    approximationName="lagrange";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=0.0;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j/(r);
        }
//        if (verbosity>-1){
        printf( "lagrange-coulomb-skipping error: %1.3e\n", fabs(potential[i] - trueValue)/fabs(trueValue) );
//        }
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-coulomb-skipping", fabs(potential[i] - trueValue)/fabs(trueValue) < 3e-3);

    }


    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    /***********************************************/
    /******************* Test 2 ********************/
    /***********************************************/
    /********* lagrange-coulomb-subtraction ********/
    /***********************************************/
    singularityHandling="subtraction";

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=2.0 * M_PI * kappa * kappa * i;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i*exp(-r*r/kappa*kappa) )/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i*exp(-r*r/kappa*kappa) )/(r);
        }

        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-coulomb-subtraction", fabs(potential[i] - trueValue)/fabs(trueValue) < 2e-2);
    }

    /***********************************************/
    /******************* Test 3 ********************/
    /***********************************************/
    /*********** lagrange-yukawa-skipping **********/
    /***********************************************/
    kernelName="yukawa";
    singularityHandling="skipping";

    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=0.0;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }

        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-yukawa-skipping", fabs(potential[i] - trueValue)/fabs(trueValue) < 8e-3);
    }

    /***********************************************/
    /******************* Test 4 ********************/
    /***********************************************/
    /********* lagrange-yukawa-subtraction *********/
    /***********************************************/
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }
    singularityHandling="subtraction";

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=4.0 * M_PI / kappa / kappa * i;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i)*exp(-kappa*r)/(r);
        }
        for (int j=i+1; j<sources->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i)*exp(-kappa*r)/(r);
        }
        // measure absolute error for this example, since true values are very close to zero.
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-yukawa-subtraction", fabs(potential[i] - trueValue) < 2e-2);

    }

    /***********************************************/
    /******************* Test 5 ********************/
    /***********************************************/
    /********* hermite-coulomb-skipping ************/
    /***********************************************/
    approximationName="hermite";
    kernelName="coulomb";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=0.0;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j/(r);
        }

        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-coulomb-skipping", fabs(potential[i] - trueValue)/fabs(trueValue) < 3e-4);
    }


    /***********************************************/
    /******************* Test 6 ********************/
    /***********************************************/
    /******** hermite-coulomb-subtraction **********/
    /***********************************************/
    approximationName="hermite";
    kernelName="coulomb";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=2.0 * M_PI * kappa * kappa * i;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i*exp(-r*r/kappa*kappa) )/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i*exp(-r*r/kappa*kappa) )/(r);
        }

        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-coulomb-subtraction", fabs(potential[i] - trueValue)/fabs(trueValue) < 2e-2);
    }

    /***********************************************/
    /******************* Test 7 ********************/
    /***********************************************/
    /********** hermite-yukawa-skipping ************/
    /***********************************************/
    approximationName="hermite";
    kernelName="yukawa";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=0.0;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }
        for (int j=i+1; j<targets->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += j*exp(-kappa*r)/(r);
        }

        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-yukawa-skipping", fabs(potential[i] - trueValue)/fabs(trueValue) < 5e-4);
    }

    /***********************************************/
    /******************* Test 8 ********************/
    /***********************************************/
    /********* hermite-yukawa-subtraction **********/
    /***********************************************/
    approximationName="hermite";
    kernelName="yukawa";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
    }

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        double trueValue=4.0 * M_PI / kappa / kappa * i;
        for (int j=0; j<i; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i)*exp(-kappa*r)/(r);
        }
        for (int j=i+1; j<sources->num; j++){
            double r = abs(j-i)*sqrt(3);
            trueValue += (j - i)*exp(-kappa*r)/(r);
        }
        // measure absolute error for this example, since true values are very close to zero.
        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-yukawa-subtraction", fabs(potential[i] - trueValue) < 2e-3);

    }

    free(sources->x);
    free(sources->y);
    free(sources->z);
    free(sources->q);
    free(sources->w);
    free(sources->order);
    free(sources);

    free(targets->x);
    free(targets->y);
    free(targets->z);
    free(targets->q);
    free(targets->order);
    free(targets);

    free(potential);
    return 0;
}


static char * test_treecode_on_1_target_10000_sources() {

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int N=10000;
    int verbosity=0;

    struct particles *sources = NULL;
    struct particles *targets = NULL;
    int *particleOrder = NULL;
    double *potential = NULL, *potential_direct = NULL;
    double potential_engy = 0;
    double potential_engy_direct = 0;

    sources = malloc(sizeof(struct particles));
    targets = malloc(sizeof(struct particles));
    potential = malloc(sizeof(double) * N);
    potential_direct = malloc(sizeof(double) * N);
    particleOrder = malloc(sizeof(int) * N);

    targets->num = 1; //single target at origin
    targets->x = malloc(targets->num*sizeof(double));
    targets->y = malloc(targets->num*sizeof(double));
    targets->z = malloc(targets->num*sizeof(double));
    targets->q = malloc(targets->num*sizeof(double));
    targets->order = malloc(targets->num*sizeof(int));

    sources->num = N; // 10,000 sources per rank
    sources->x = malloc(sources->num*sizeof(double));
    sources->y = malloc(sources->num*sizeof(double));
    sources->z = malloc(sources->num*sizeof(double));
    sources->q = malloc(sources->num*sizeof(double));
    sources->w = malloc(sources->num*sizeof(double));
    sources->order = malloc(sources->num*sizeof(int));




    for (int i=0; i<targets->num; i++){
        // single target at origin
        targets->x[i]=0.0;
        targets->y[i]=0.0;
        targets->z[i]=0.0;
        targets->q[i]=1.0;
        targets->order[i] = i;
    }

    srand(rank);
    for (int i=0; i<sources->num; i++){
        // 10,000 randomly distributed sources in the [-1,1] box, seeded by rank
        sources->x[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->y[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->z[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->q[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->w[i]=((double)rand()/(double)(RAND_MAX));
        sources->order[i] = i;
    }


    int max_per_leaf=100;
    int max_per_batch=100;
    double time_tree[9];

    char *kernelName, *singularityHandling, *approximationName;


    int tree_type=1; // particle-cluster
    double kappa=0.5;

    int order=4;
    double theta=0.8;

    /***********************************************/
    /******************* Test 1 ********************/
    /***********************************************/
    /********** lagrange-coulomb-skipping **********/
    /***********************************************/
    approximationName="lagrange";
    kernelName="coulomb";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,                              
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nlagrange-coulomb-skipping\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-coulomb-skipping", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 2e-4);
    }



    /***********************************************/
    /******************* Test 2 ********************/
    /***********************************************/
    /********* lagrange-coulomb-subtraction ********/
    /***********************************************/
    approximationName="lagrange";
    kernelName="coulomb";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,                              
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nlagrange-coulomb-subtraction\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-coulomb-subtraction", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 2e-5);
    }

    /***********************************************/
    /******************* Test 3 ********************/
    /***********************************************/
    /*********** lagrange-yukawa-skipping **********/
    /***********************************************/
    approximationName="lagrange";
    kernelName="yukawa";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nlagrange-yukawa-skipping\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-yukawa-skipping", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 2e-4);
    }

    /***********************************************/
    /******************* Test 4 ********************/
    /***********************************************/
    /********* lagrange-yukawa-subtraction *********/
    /***********************************************/
    approximationName="lagrange";
    kernelName="yukawa";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nlagrange-yukawa-subtraction\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: lagrange-yukawa-subtraction", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 6e-6);
    }

    /***********************************************/
    /******************* Test 5 ********************/
    /***********************************************/
    /********* hermite-coulomb-skipping ************/
    /***********************************************/
    approximationName="hermite";
    kernelName="coulomb";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nhermite-coulomb-skipping\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-coulomb-skipping", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 6e-8);
    }


    /***********************************************/
    /******************* Test 6 ********************/
    /***********************************************/
    /******** hermite-coulomb-subtraction **********/
    /***********************************************/
    approximationName="hermite";
    kernelName="coulomb";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nhermite-coulomb-subtraction\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-coulomb-subtraction", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 2e-7);
    }

    /***********************************************/
    /******************* Test 7 ********************/
    /***********************************************/
    /********** hermite-yukawa-skipping ************/
    /***********************************************/
    approximationName="hermite";
    kernelName="yukawa";
    singularityHandling="skipping";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nhermite-yukawa-skipping\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-yukawa-skipping", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 9e-8);
    }

    /***********************************************/
    /******************* Test 8 ********************/
    /***********************************************/
    /********* hermite-yukawa-subtraction **********/
    /***********************************************/
    approximationName="hermite";
    kernelName="yukawa";
    singularityHandling="subtraction";
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);

    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nhermite-yukawa-subtraction\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not correct for: hermite-yukawa-subtraction", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 3e-8);
    }

    free(sources->x);
    free(sources->y);
    free(sources->z);
    free(sources->q);
    free(sources->w);
    free(sources->order);
    free(sources);

    free(targets->x);
    free(targets->y);
    free(targets->z);
    free(targets->q);
    free(targets->order);
    free(targets);

    free(potential);
    free(potential_direct);
    return 0;
}


static char * test_treecodewrapper_on_1_target_10000_sources() {

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int N=50000;
    int verbosity=0;

    struct particles *sources = NULL;
    struct particles *targets = NULL;
    int *particleOrder = NULL;
    double *potential = NULL, *potentialWrapper = NULL, *potential_direct = NULL;
    double potential_engy = 0;
    double potential_engy_direct = 0;

    sources = malloc(sizeof(struct particles));
    targets = malloc(sizeof(struct particles));
    potential = malloc(sizeof(double) * N);
    potentialWrapper = malloc(sizeof(double) * N);
    potential_direct = malloc(sizeof(double) * N);
    particleOrder = malloc(sizeof(int) * N);

    targets->num = 10; //single target at origin
    targets->x = malloc(targets->num*sizeof(double));
    targets->y = malloc(targets->num*sizeof(double));
    targets->z = malloc(targets->num*sizeof(double));
    targets->q = malloc(targets->num*sizeof(double));
    targets->order = malloc(targets->num*sizeof(int));

    sources->num = N; // 10,000 sources per rank
    sources->x = malloc(sources->num*sizeof(double));
    sources->y = malloc(sources->num*sizeof(double));
    sources->z = malloc(sources->num*sizeof(double));
    sources->q = malloc(sources->num*sizeof(double));
    sources->w = malloc(sources->num*sizeof(double));
    sources->order = malloc(sources->num*sizeof(int));


    for (int i=0; i<targets->num; i++){
        // single target at origin
        targets->x[i]=i;
        targets->y[i]=i;
        targets->z[i]=i;
        targets->q[i]=1.0;
        targets->order[i] = i;
    }

    srand(rank);
    for (int i=0; i<sources->num; i++){
        // 10,000 randomly distributed sources in the [-1,1] box, seeded by rank
        sources->x[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->y[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->z[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->q[i]=((double)rand()/(double)(RAND_MAX)) * 2. - 1.;
        sources->w[i]=((double)rand()/(double)(RAND_MAX));
        sources->order[i] = i;
    }


    int max_per_leaf=20;
    int max_per_batch=2;
    double time_tree[9];

    char *kernelName, *singularityHandling, *approximationName;


    int tree_type=1; // particle-cluster
    double kappa=0.5;

    int order=2;
    double theta=0.6;

    /***********************************************/
    /******************* Test 1 ********************/
    /***********************************************/
    /********** lagrange-coulomb-skipping **********/
    /***********************************************/

    verbosity=0;

    approximationName="hermite";
    kernelName="coulomb";
    singularityHandling="skipping";
    printf("singularityHandling = %s\n", singularityHandling);
    for (int i=0; i<targets->num; i++){
        potential[i]=0.0;
        potentialWrapper[i]=0.0;
        potential_direct[i]=0.0;
    }

    directdriver(sources, targets, kernelName, kappa, singularityHandling, approximationName,
                 potential_direct, time_tree);

    treedriver(sources, targets, order, theta, max_per_leaf, max_per_batch,
               kernelName, kappa, singularityHandling, approximationName, tree_type,
               potential, time_tree, 1.0, verbosity);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Does sources exist before wrapper call?  %1.3e\n", sources->x[3]); fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    approximationName="lagrange";
    kernelName="coulomb";
    singularityHandling="skipping";
    printf("before wrapper: singularityHandling = %s\n", singularityHandling);
    treedriverWrapper(targets->num, N,
                      targets->x,targets->y,targets->z,targets->q,
                      sources->x, sources->y, sources->z, sources->q, sources->w,
                      potentialWrapper, kernelName, kappa, singularityHandling, approximationName,
                      order, theta, max_per_leaf, max_per_batch, verbosity);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Does targets still exist after wrapper call?  %1.3e\n", targets->x[3]);
//    printf("Does sources still exist after wrapper call?  %1.3e\n", sources->x[3]);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i=0; i<targets->num; i++){
        if (verbosity>0) printf("\nlagrange-coulomb-skipping\n");
        if (verbosity>0) printf("direct: %1.8e\n", potential_direct[i]);
        if (verbosity>0) printf("approx: %1.8e\n", potential[i]);
        if (verbosity>0) printf("absolute error: %1.2e\n", fabs(potential[i] - potential_direct[i]));
        if (verbosity>0) printf("relative error: %1.2e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        printf("treecode error: %1.3e\n", fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]));
        mu_assert("TEST FAILED: Treecode potential not close to direct:", \
                fabs(potential[i] - potential_direct[i])/fabs(potential_direct[i]) < 2e-4);
        printf("wrapper error: %1.3e\n", fabs(potential[i] - potentialWrapper[i])/fabs(potential[i]));
        mu_assert("TEST FAILED: TreecodeWrapper potential not exact as Treecode:", \
                fabs(potential[i] - potentialWrapper[i])/fabs(potential[i]) < 2e-14);
    }



    printf("about to free vectors...\n"); fflush(stdout);
    free(sources->x);
    printf("Freed sources->x.\n"); fflush(stdout);
    free(sources->y);
    free(sources->z);
    free(sources->q);
    free(sources->w);
    free(sources->order);
    free(sources);

    printf("Does sources still exist?  %1.3e\n", sources->x[0]);

    printf("Freed sources, now on to targets.\n"); fflush(stdout);
    free(targets->x);
    free(targets->y);
    free(targets->z);
    free(targets->q);
    free(targets->order);
    free(targets);
    printf("Freed targets, now on to potentials.\n"); fflush(stdout);
    free(potential);
    free(potentialWrapper);
    free(potential_direct);
    printf("Exiting treecodeWrapper test.\n"); fflush(stdout);
    return 0;
}


// Run all the tests
static char * all_tests() {
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    MPI_Barrier(MPI_COMM_WORLD);
    mu_run_test(test_treecodewrapper_on_1_target_10000_sources);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) printf("Completed treecodeWrapper test.\n");

//    mu_run_test(test_direct_sum_on_10_particles_per_rank);
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0) printf("Completed test_direct_sum_on_10_particles().\n");
////    mu_run_test(test_treecode_on_100_particles);
////    if (rank==0) printf("Completed test_treecode_on_100_particles().\n");  # this test isn't set up for mpi testing.
//    mu_run_test(test_treecode_on_1_target_10000_sources);
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0) printf("Completed test_treecode_on_1_target_10000_sources().\n");

return 0;
}

int main(int argc, char **argv) {
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);



    char *result = all_tests();
    if (rank==0){
        printf("Tests run: %d\n", tests_run);
        if (result != 0) {
            printf("=============================\n" \
                   "| SOME PARALLEL TESTS FAILED |\n" \
                   "=============================\n");
            printf("%s\n", result);
        }
        else {
            printf("============================\n" \
                   "| ALL PARALLEL TESTS PASSED |\n" \
                   "============================\n");
        }
    }



    MPI_Finalize();
    return result != 0;
}
