#ifndef VEVP_HYPER_H
#define VEVP_HYPER_H

#include "petsc.h"

PetscScalar ResF(PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar[]);
PetscScalar ResT(PetscScalar, PetscScalar, PetscScalar[], PetscScalar[]);
PetscScalar ResP(PetscScalar, PetscScalar, PetscScalar[], PetscScalar[]);

PetscErrorCode VEVP_hyper_sol(PetscInt, PetscScalar,
                              PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar[],
                              PetscScalar[], PetscScalar[]);


#endif // VEVP-HYPER_H_
