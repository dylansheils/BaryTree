#ifndef H_BARYTREE_TYPES_H
#define H_BARYTREE_TYPES_H


typedef enum KERNEL
{
    NO_KERNEL,
    COULOMB,
    YUKAWA,
    REGULARIZED_COULOMB,
    REGULARIZED_YUKAWA,
    ATAN,
    TCF,
    DCF,
    SIN_OVER_R,
    MQ,
    USER
} KERNEL;


typedef enum SINGULARITY
{
    NO_SINGULARITY,
    SKIPPING,
    SUBTRACTION
} SINGULARITY;


typedef enum APPROXIMATION
{
    NO_APPROX,
    LAGRANGE,
    HERMITE
} APPROXIMATION;


typedef enum COMPUTE_TYPE
{
    NO_COMPUTE_TYPE,
    PARTICLE_CLUSTER,
    CLUSTER_PARTICLE,
    CLUSTER_CLUSTER,
} COMPUTE_TYPE;


#endif /* H_BARYTREE_TYPES_H */
