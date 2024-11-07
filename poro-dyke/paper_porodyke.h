#ifndef PAPER_PORODYKE_H
#define PAPER_PORODYKE_H


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
#include "../../fdpde_stokesdarcy2field.h"
#include "../../fdpde_advdiff.h"
#include "../../consteq.h"
#include "../../dmstagoutput.h"
#include "../vevp-funs/vevp-hyper.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
#define FNAME_LENGTH 200

// parameters (bag)
typedef struct {
  PetscInt       nx, nz;
  PetscInt       tstep, tout, maxckpt, Nmax;
  PetscInt       vfopt;
  PetscInt       ts_scheme, adv_scheme, bcidx, air;
  PetscInt       permeability, poromean, kenhance; // TYPE of peambility, type of calculating porosity means, interpolation type of permeability enhancement
  PetscScalar    gamma, eps;  //phasefield
  PetscScalar    L, H, rd;
  PetscScalar    xmin, zmin, xmax, zmax;
  PetscScalar    eta_vk, vi,viz;
  PetscScalar    eta1, eta2, eta3, eta4, zeta1, zeta2, zeta3, zeta4, C1, C2, C3, C4, F1, F2, F3,F4, G1, G2, G3,G4, R2_1, R2_2, R2_3, R2_4, FS, gth, sth, delta;
  PetscScalar    lam_v, lambda,Z,q, phi_0,n;
  PetscScalar    phi_m, phi_bg, xsig, zsig, xpeak, zpeak; //initial porosity distribution
  PetscBool      plasticity, poroevo, gaussian;
  PetscScalar    t, dt, dtck, tmax, tscale;
  PetscScalar    sdel; //for BCs, dpdz = (1-sdel)*DeltaRhoG 
  PetscScalar    Kini, dini,kscale, qin, K0, drhog; //for BCs: initial scaled K and initial sdel calcualted based on initial flux rate qin
  PetscScalar    angle;
  PetscScalar    dp;
  PetscScalar    R_pe, coeff_iso;  //sigma_T = C/R_pe, coeff_iso coefficients of the isotropic components 1/3 or 1/2
  PetscScalar    pfs, dpfs, p0, dl, hpeak;   // static pressure at the free surface, (value on the left corner and on the right corner), dynamic pressure at the bottom
  PetscScalar    iota, psi; // iota = (1-alpha) - coefficient for fluid pressure, psi - dilation angle
  PetscScalar    tf_tol, phicut,nap, ncdl, alphaC, etamin; // tolerance for calculation on yield surface, cutoff porosity, minimum intrinsic shear viscosity
  PetscScalar    strain_max, hca, hcc; // transition length of plastic strain, relative change of softening.
  PetscScalar    Hair, erange, krange, ck, strainK, noise_max;
  PetscScalar    km, kd;// parameter required for the customised permeability model
  PetscScalar    km_anis, keps_anis; //Anisotrpic permeability enhancement
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
  Vec            xPV, xphiprev,xphi0, f, fprev, dfx, dfz; //PV, porosity, phasefield
  Vec            volf; //volume fraction
  Vec            fp, fn, nfp, nfn; // fluid types for positive and negative normal direction of an interface
  Vec            xeps, xtau, xyield, xDP, xtau_old, xDP_old, VMag;
  Vec            xepso, xkee, epsvp, evp, evpo;
  Vec            x1, x2, x3; // store past solutions for prediction
  Vec            Pst, YS;  // static pressure, yield strength(stresses components)
  Vec            noise, strain, Kdv; // random noise field, integrated plastic strain rates.
} UsrData;

// ---------------------------------------
// Function definitions
// ---------------------------------------
PetscErrorCode StokesDarcy_Numerical(void*);
PetscErrorCode InputParameters(UsrData**);
PetscErrorCode InputPrintData(UsrData*);
PetscErrorCode FormCoefficient_PV(FDPDE, DM, Vec, DM, Vec, void*);
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

PetscErrorCode UpdateFacePermeability(void*);
#endif
