#ifndef PAPER_VEVP_H
#define PAPER_VEVP_H


// define convenient names for DMStagStencilLocation
#define DOWN_LEFT  DMSTAG_DOWN_LEFT
#define DOWN       DMSTAG_DOWN
#define DOWN_RIGHT DMSTAG_DOWN_RIGHT
#define LEFT       DMSTAG_LEFT
#define ELEMENT    DMSTAG_ELEMENT
#define RIGHT      DMSTAG_RIGHT
#define UP_LEFT    DMSTAG_UP_LEFT
#define UP         DMSTAG_UP
#define UP_RIGHT   DMSTAG_UP_RIGHT

#include "petsc.h"
#include "../../src/fdpde_stokesdarcy2field.h"
#include "../../src/fdpde_advdiff.h"
#include "../../src/consteq.h"
#include "../../src/dmstagoutput.h"
#include "./vevp-funs/vevp-hyper.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscInt       tstep, tout, maxckpt, Nmax;
  PetscInt       vfopt;
  PetscInt       ts_scheme, adv_scheme;
  PetscScalar    gamma, eps;  //phasefield
  PetscScalar    L, H;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    eta_vk, vi;
  PetscScalar    eta1, eta2, eta3, eta4, zeta1, zeta2, zeta3, zeta4, C1, C2, C3, C4, F1, F2, F3,F4, G1, G2, G3,G4, R2_1, R2_2, R2_3, R2_4, FS;
  PetscScalar    lam_v, lambda,Z,q, phi_0,n;
  PetscScalar    phi_m, phi_bg, xsig, zsig, xpeak, zpeak; //initial porosity distribution
  PetscBool      plasticity, poroevo, gaussian;
  PetscScalar    t, dt, dtck, tmax;
  PetscScalar    angle;
  PetscScalar    dp;
  PetscScalar    R_pe, coeff_iso;  //sigma_T = C/R_pe, coeff_iso coefficients of the isotropic components 1/3 or 1/2
  PetscScalar    pfs;   // static pressure at the free surface
  PetscScalar    iota, psi; // iota = (1-alpha) - coefficient for fluid pressure, psi - dilation angle
  PetscScalar    tf_tol, phicut, alphaC, etamin; // tolerance for calculation on yield surface, cutoff porosity, minimum intrinsic shear viscosity
  PetscScalar    strain_max, hca, hcc; // transition length of plastic strain, relative change of softening.
  PetscScalar    Hair, erange, krange, ck, strainK, noise_max;
  char           fname_out[FNAME_LENGTH];
  char           fname_in[FNAME_LENGTH];
} Params;

// user defined and model-dependent variables
typedef struct {
  Params        *par;
  PetscBag       bag;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dmeps, dmf, dmPV, dmphi; //dmf - phasefield DM, dmPV - velocity
  Vec            xPV, xphiprev, f, fprev, dfx, dfz; //PV, porosity, phasefield
  Vec            volf; //volume fraction
  Vec            fp, fn, nfp, nfn; // fluid types for positive and negative normal direction of an interface
  Vec            xeps, xtau, xyield, xDP, xtau_old, xDP_old, VMag;
  Vec            x1, x2, x3; // store past solutions for prediction
  Vec            Pst, YS;  // static pressure, yield strength(stresses components)
  Vec            noise, strain; // random noise field, integrated plastic strain rates.
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy_Numerical(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient_PV(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_PV(DM, Vec, DMStagBCList, void*);
PetscErrorCode UpdateStrainRates(DM,Vec,void*);
PetscErrorCode UpdateStressOld(DM,void*);

PetscErrorCode UpdateUMag(DM, Vec, void*);
PetscErrorCode UpdatePStatic(DM, void*);
PetscErrorCode UpdateYS(DM, Vec, DM, Vec, DM, Vec, Vec, void*);

//PetscErrorCode Phase_Numerical(void*);
PetscErrorCode SetInitialField(DM,Vec,void*);
PetscErrorCode UpdateDF(DM,Vec,void*);
PetscErrorCode ExplicitStep(DM,Vec,Vec,PetscScalar,void*);
PetscErrorCode UpdateCornerF(DM, Vec, void*);
PetscErrorCode UpdateVolFrac(DM, Vec, void*);
PetscErrorCode CleanUpFPFN(DM, void*);
PetscErrorCode CollectFPFN(DM, void*);

PetscErrorCode FormCoefficient_phi(FDPDE, DM, Vec, DM, Vec, void*);
PetscErrorCode FormBCList_phi(DM, Vec, DMStagBCList, void*);
PetscErrorCode SetInitialPorosityProfile(DM,Vec,void*);
PetscErrorCode SetInitialPorosityCoefficient(DM,Vec,void*);

PetscErrorCode PorosityFilter(DM, Vec, void*);



PetscErrorCode IntegratePlasticStrain(DM, Vec, Vec, void*);

#endif
