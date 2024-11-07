// ---------------------------------------
// Simulation: buoyancy-driven dyke, matching with poro-frac model 
// ---------------------------------------
static char help[] = "Single column Buoyancy-driven dyke \n\n";

#include "paper_porodyke.h"

const char coeff_description_sd[] = 
"  << Stokes-Darcy Coefficients >> \n"
"  A = eta_vep \n"
"  B = phi*F*ez - div(chi_s * tau_old) + grad(chi_p * diff_pold) \n"
"      (F = ((rho^s-rho^f)*U*L/eta_ref)*(g*L/U^2)) (0, if g=0) \n"
"      (tau_old: stress rotation tensor; diff_pold: pressure difference  ) \n"
"  C = 0 \n"
"  D1 = zeta_vep - 2/3*eta_vep \n"
"  D2 = -Kphi*R^2 (R: ratio of the compaction length to the global length scale) \n"
"  D3 = Kphi*R^2*F*ez \n";

const char bc_description_sd[] =
"  << Stokes-Darcy BCs >> \n"
"  LEFT: Vx=-vi, dVz/dx=0 \n"
"  UP: zero tangential stress, Vz = -2*Hair/L * vi \n"
"  RIGHT: Vx=vi, dVz/dx=0\n"
"  BOTTOM: zero normal and tangential stress \n";


const char coeff_description_phi[] =
"  << Porosity Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 0 \n"
"  C = 0 \n"
"  u = [ux, uz] - StokesDarcy solid velocity \n";

const char bc_description_phi[] =
"  << Porosity BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: Zero flux \n";


PetscErrorCode FormBCList_PV_openside(DM, Vec, DMStagBCList, void*);

PetscErrorCode DamageEnhancedPermeability(DM, Vec, Vec, Vec, void*);


// ----
// Static functions
// ----
 
static PetscScalar Permeability_porosity(PetscScalar phi)

{ PetscScalar result;

  // result = PetscPowScalar(phi, 3)/PetscPowScalar(1-phi, 2);
   result = PetscPowScalar(phi, 3);

  return(result); 
}



// ---------------------------------------
// Application functions
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "StokesDarcy_Numerical"
PetscErrorCode StokesDarcy_Numerical(void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  FDPDE          fdPV, fdphi;
  DM             dmcoeff, dmphicoeff;
  Vec            xPV, xcoeff, xguess, xphi, xphiprev, phicoeff, phicoeffprev, phibkup;
  PetscInt       nx, nz, istep = 0, tstep, ickpt = 0, maxckpt;
  PetscScalar    xmin, zmin, xmax, zmax, dtck, tckpt, tmax, noise_max;
  PetscBool      iwrt, converged; //if write into files or not; if converged or not
  char           fout[FNAME_LENGTH];
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Element count
  nx = usr->par->nx;
  nz = usr->par->nz;

  // Domain coords
  xmin = usr->par->xmin;
  xmax = usr->par->xmin + usr->par->L;
  zmin = usr->par->zmin;
  zmax = usr->par->zmin + usr->par->H;

  tstep = usr->par->tstep;
  dtck  = usr->par->dtck;
  maxckpt = usr->par->maxckpt;
  tckpt = dtck; //first check point in time
  iwrt = PETSC_TRUE;
  tmax = usr->par->tmax;

  noise_max = usr->par->noise_max;

  PetscPrintf(usr->comm, "check point at the beginning\n");

  // 1. Stokes-Darcy: Create the FDPDE object, set the function and boundary conditions
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_sd,usr); CHKERRQ(ierr);
//  if (usr->par->bcidx==4) {ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV_openside,bc_description_sd,usr); CHKERRQ(ierr);}
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV_openside,bc_description_sd,usr); CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(fdPV->snes); CHKERRQ(ierr);
  ierr = FDPDEView(fdPV); CHKERRQ(ierr);

  // 2. Porosity: Create the FDPDE object, set the function and boundary conditions
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_ADVDIFF,&fdphi);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdphi);CHKERRQ(ierr);

  if (usr->par->adv_scheme==0) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_UPWIND);CHKERRQ(ierr); }
  if (usr->par->adv_scheme==1) { ierr = FDPDEAdvDiffSetAdvectSchemeType(fdphi,ADV_FROMM);CHKERRQ(ierr); }
  
  if (usr->par->ts_scheme ==  0) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_FORWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  1) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_BACKWARD_EULER);CHKERRQ(ierr); }
  if (usr->par->ts_scheme ==  2) { ierr = FDPDEAdvDiffSetTimeStepSchemeType(fdphi,TS_CRANK_NICHOLSON );CHKERRQ(ierr);}

  ierr = FDPDESetFunctionBCList(fdphi,FormBCList_phi,bc_description_phi,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdphi,FormCoefficient_phi,coeff_description_phi,usr); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(fdphi->snes); CHKERRQ(ierr);

  ierr = FDPDEView(fdphi); CHKERRQ(ierr);

  // 3. Get DMs for user
  ierr = FDPDEGetDM(fdPV,&usr->dmPV); CHKERRQ(ierr);
  ierr = FDPDEGetDM(fdphi,&usr->dmphi); CHKERRQ(ierr);
  
  // -- Create DM/vec for strain rates, yield, deviatoric/volumetric stress, and stress_old
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,4,0,4,0,&usr->dmeps); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmeps); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmeps,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xeps); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xyield); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xtau_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xDP_old); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->YS); CHKERRQ(ierr);

  // vector for strain rates at the previous step
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->xepso); CHKERRQ(ierr);
  
  // vector for plastic strain rates and plastic strain, note only values in element and vertex
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->epsvp); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->evp); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmeps,&usr->evpo); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->evp); CHKERRQ(ierr); //initialise the plastic strain as zeros.
  ierr = VecZeroEntries(usr->evpo); CHKERRQ(ierr); //initialise the plastic strain as zeros.
  
  // vector for face-wise permeability modification
  ierr = DMCreateGlobalVector(usr->dmPV,&usr->xkee); CHKERRQ(ierr);

// -- create vectors to store past solutions
  ierr = DMCreateGlobalVector(usr->dmPV,&usr->x1); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPV,&usr->x2); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmPV,&usr->x3); CHKERRQ(ierr);


  // -- Create DM/vec for the phase field
  ierr = DMStagCreateCompatibleDMStag(usr->dmPV,1,1,1,0,&usr->dmf); CHKERRQ(ierr);
  ierr = DMSetUp(usr->dmf); CHKERRQ(ierr);
  ierr = DMStagSetUniformCoordinatesProduct(usr->dmf,xmin,xmax,zmin,zmax,0.0,0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->f); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->fprev); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->dfx); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->dfz); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->volf); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->fp); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->fn); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->nfp); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(usr->dmf, &usr->nfn); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(usr->dmf, &usr->Pst); CHKERRQ(ierr);

  // -- Initialise the random noise field
  PetscRandom  rctx;
  ierr = DMCreateGlobalVector(usr->dmeps, &usr->noise); CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rctx); CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rctx, -noise_max, noise_max);CHKERRQ(ierr);
  ierr = VecSetRandom(usr->noise, rctx); CHKERRQ(ierr);
  //ierr = VecZeroEntries(usr->noise); CHKERRQ(ierr);
  PetscRandomDestroy(&rctx);

  // -- Create vec for the plastic strains
  ierr = DMCreateGlobalVector(usr->dmeps, &usr->strain); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->strain); CHKERRQ(ierr);

  // -- Create vec for the damage-enhanced permeability
  ierr = DMCreateGlobalVector(usr->dmeps, &usr->Kdv); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->Kdv); CHKERRQ(ierr);
  
  // -- Create a vec to store the magnitude of velocity at every cell center
  //    use the same DM with fp and fn for the conveience to know VMag in different fluids.
  ierr = DMCreateGlobalVector(usr->dmPV, &usr->VMag); CHKERRQ(ierr);
  
  // -- Create a vector to store PV and phi in userdata
  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV, &usr->xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xphi, &usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDuplicate(xphi, &usr->xphi0);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xphi);CHKERRQ(ierr);

  // -- duplicate phibkup for the porosity before filter
  ierr = VecDuplicate(usr->xphiprev, &phibkup); CHKERRQ(ierr);

  // -- Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  ierr = VecZeroEntries(usr->xtau_old); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->xDP_old); CHKERRQ(ierr);

  // Initialise the phase field
  ierr = SetInitialField(usr->dmf,usr->f,usr);CHKERRQ(ierr);

  // Set initial porosity profile (t=0)
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityProfile(usr->dmphi,xphiprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,usr->xphi0);CHKERRQ(ierr);  //save a copy of initial porosity
  ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

  ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,NULL);CHKERRQ(ierr);
  ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityCoefficient(dmphicoeff,phicoeffprev,usr);CHKERRQ(ierr);
  ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);

  //interpolate phase values on the face and edges before FDPDE solver
  ierr = UpdateCornerF(usr->dmf, usr->f, usr); CHKERRQ(ierr);
  ierr = UpdateVolFrac(usr->dmf, usr->f, usr); CHKERRQ(ierr);
  ierr = CleanUpFPFN(usr->dmf,usr); CHKERRQ(ierr);
  ierr = CollectFPFN(usr->dmf,usr); CHKERRQ(ierr);
  ierr = VecCopy(usr->nfp, usr->fp); CHKERRQ(ierr);
  ierr = VecCopy(usr->nfn, usr->fn); CHKERRQ(ierr);
  ierr = VecCopy(usr->f, usr->fprev);CHKERRQ(ierr);
  //update lithostatic pressure before fdpdesolve
  ierr = UpdatePStatic(usr->dmf, usr); CHKERRQ(ierr);

  // Create initial guess with a linear viscous, no evolution of porosity yet
  usr->par->plasticity = PETSC_FALSE; 
//  usr->par->poromean = 2; // If no plasticity, taking harmonic mean of porosity for interpolation
  PetscPrintf(usr->comm,"\n# INITIAL GUESS #\n");

  ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);

  ierr = VecCopy(xPV, usr->xPV); CHKERRQ(ierr);

  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout);CHKERRQ(ierr);

  ierr = FDPDEGetSolutionGuess(fdPV,&xguess); CHKERRQ(ierr); 
  ierr = VecCopy(usr->xPV,xguess);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xguess);CHKERRQ(ierr);

  usr->par->plasticity = PETSC_TRUE; //switch off/on plasticity
//  usr->par->poromean = 0; // Once plasticity is turned on, switching to arithmetic averaging

  // Open a file

  // output - initial state of the phase field
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phase_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmf,usr->f,fout);CHKERRQ(ierr);
  
  // FD SNES Solver
  PetscPrintf(usr->comm,"\n# SNES SOLVE #\n");


  // Open binary file to save timestep size at the checkpoint
  PetscInt       dtfd;
  PetscViewer    dtview;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dtck.dat",  &dtview); 
  //PetscViewerASCIIGetDescriptor(dtview, &dtfd); CHKERRQ(ierr);
  //PetscASCIIWrite(dtfd, &usr->par->dt, 1, PETSC_SCALAR);
  //PetscScalarView(1, &usr->par->dt, dtview);
  
  // Time loop  
  while ((ickpt <= maxckpt) && (istep<tstep) && (usr->par->t<=tmax))  {


    // Update face-wise permeability modification for the next time step
    ierr = UpdateFacePermeability(usr);CHKERRQ(ierr);


  jump:
    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt

    // set time step size for ADVDIFF
    ierr = FDPDEAdvDiffSetTimestep(fdphi,usr->par->dt);CHKERRQ(ierr);

    PetscPrintf(usr->comm,"# TIME CHECK POINT %d out of %d after %d steps: time %1.4e\n\n",ickpt, maxckpt, istep, usr->par->t);
    PetscPrintf(usr->comm,"# next check piont: %1.4e; distance between check points: %1.4e\n\n", tckpt, dtck);
    
    if (istep>0) {
      //one step forward to get f at the next step (extract velocity data within it)
      ierr = UpdateDF(usr->dmf, usr->fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(usr->dmf, usr->fprev, usr->f, usr->par->dt, usr); CHKERRQ(ierr);


      // 4th order runge-kutta

      Vec hk1, hk2, hk3, hk4, f, fprev;
      // allocate storage for hk1, hk2, f_bk, fprev_bk
      ierr = VecDuplicate(usr->f, &hk1); CHKERRQ(ierr);
      ierr = VecDuplicate(usr->f, &hk2); CHKERRQ(ierr);
      ierr = VecDuplicate(usr->f, &hk3); CHKERRQ(ierr);
      ierr = VecDuplicate(usr->f, &hk4); CHKERRQ(ierr);
      ierr = VecDuplicate(usr->f, &f); CHKERRQ(ierr);
      ierr = VecDuplicate(usr->f, &fprev); CHKERRQ(ierr);

      // copy f and fprev into the temporary working space
      ierr = VecCopy(usr->f, f); CHKERRQ(ierr);
      ierr = VecCopy(usr->fprev, fprev); CHKERRQ(ierr);
      
      // 1st stage - get h*k1 = f- fprev
      ierr = VecCopy(f, hk1); CHKERRQ(ierr);
      ierr = VecAXPY(hk1, -1.0, fprev);CHKERRQ(ierr);
      
      // 2nd stage - (t = t+0.5*dt, fprev = fprev + 0.5*hk1)
      ierr = VecCopy(usr->fprev, fprev); CHKERRQ(ierr);
      ierr = VecAXPY(fprev, 0.5, hk1); CHKERRQ(ierr);

      // correct time by half step
      usr->par->t -= 0.5*usr->par->dt;
      
      // update dfx and dfz and solve for the second stage
      ierr = UpdateDF(usr->dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(usr->dmf, fprev, f, usr->par->dt, usr);CHKERRQ(ierr);

      // get hk2
      ierr = VecCopy(f, hk2); CHKERRQ(ierr);
      ierr = VecAXPY(hk2, -1.0, fprev);CHKERRQ(ierr);
      
      // reset time
      usr->par->t += 0.5*usr->par->dt;

      // 3rd stage - (t = t+0.5*dt, fprev = fprev + 0.5*hk2)
      ierr = VecCopy(usr->fprev, fprev); CHKERRQ(ierr);
      ierr = VecAXPY(fprev, 0.5, hk2); CHKERRQ(ierr);

      // correct time by half step
      usr->par->t -= 0.5*usr->par->dt;
      
      // update dfx and dfz and solve for the second stage
      ierr = UpdateDF(usr->dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(usr->dmf, fprev, f, usr->par->dt, usr);CHKERRQ(ierr);

      // get hk3 and update the full step
      ierr = VecCopy(f, hk3); CHKERRQ(ierr);
      ierr = VecAXPY(hk3, -1.0, fprev);CHKERRQ(ierr);
      
      // reset time
      usr->par->t += 0.5*usr->par->dt;

      // 4th stage - (t = t+dt, fprev = fprev + hk3)
      ierr = VecCopy(usr->fprev, fprev); CHKERRQ(ierr);
      ierr = VecAXPY(fprev, 1.0, hk3); CHKERRQ(ierr);

      // correct time by half step
      usr->par->t -= usr->par->dt;

      // update dfx and dfz and solve for the second stage
      ierr = UpdateDF(usr->dmf, fprev, usr); CHKERRQ(ierr);
      ierr = ExplicitStep(usr->dmf, fprev, f, usr->par->dt, usr);CHKERRQ(ierr);

      // get hk4
      ierr = VecCopy(f, hk4); CHKERRQ(ierr);
      ierr = VecAXPY(hk4, -1.0, fprev);CHKERRQ(ierr);

      // reset time
      usr->par->t += usr->par->dt;
      
      //update the full step
      ierr = VecCopy(usr->fprev, usr->f); CHKERRQ(ierr);
      ierr = VecAXPY(usr->f, 1.0/6.0, hk1);CHKERRQ(ierr);
      ierr = VecAXPY(usr->f, 1.0/3.0, hk2);CHKERRQ(ierr);
      ierr = VecAXPY(usr->f, 1.0/3.0, hk3);CHKERRQ(ierr);
      ierr = VecAXPY(usr->f, 1.0/6.0, hk4);CHKERRQ(ierr);
      
      // check if hk1, hk2, hk3, hk4 are zeros or NANs
      PetscScalar hk1norm, hk2norm, hk3norm, hk4norm;
      ierr = VecNorm(hk1, NORM_1, &hk1norm);
      ierr = VecNorm(hk2, NORM_1, &hk2norm);
      ierr = VecNorm(hk3, NORM_1, &hk3norm);
      ierr = VecNorm(hk4, NORM_1, &hk4norm);
      PetscPrintf(usr->comm, "hk1norm=%g, hk2norm=%g, hk3norm=%g, hk4norm=%g \n", hk1norm, hk2norm, hk3norm,hk4norm);
      
      // destroy vectors after use
      ierr = VecDestroy(&f);CHKERRQ(ierr);
      ierr = VecDestroy(&fprev);CHKERRQ(ierr);
      ierr = VecDestroy(&hk1);CHKERRQ(ierr);
      ierr = VecDestroy(&hk2);CHKERRQ(ierr);
      ierr = VecDestroy(&hk3);CHKERRQ(ierr);
      ierr = VecDestroy(&hk4);CHKERRQ(ierr);
    }
    //---fluid solver
    //interpolate phase values on the face and edges before FDPDE solver
    ierr = UpdateCornerF(usr->dmf, usr->f, usr); CHKERRQ(ierr);
    ierr = UpdateVolFrac(usr->dmf, usr->f, usr); CHKERRQ(ierr);

    //update lithostatic pressure before fdpdesolve
    ierr = UpdatePStatic(usr->dmf, usr); CHKERRQ(ierr);

    ierr = UpdateYS(usr->dmPV, usr->xPV, usr->dmf, usr->Pst, usr->dmeps, usr->xtau_old, usr->xDP_old, usr); CHKERRQ(ierr);

    converged = PETSC_FALSE;
    while (!converged) {
      ierr = FDPDESolve(fdPV,&converged);CHKERRQ(ierr);

      if (!converged) {
        if (usr->par->dt > 1e-8) {



          //reset the time
          usr->par->t -= usr->par->dt;
          //change the time step size
          usr->par->dt *= 0.2;

          //block writing in of the solution at this refined step
          iwrt = PETSC_FALSE;

           //if the checkpoint is advanced at this step, reset the checkpoint
          if ((tckpt - dtck) > (usr->par->t + usr->par->dt )) {tckpt -= dtck; ickpt--;}

          PetscPrintf(PETSC_COMM_WORLD, "fdPV not converge, taking a time step size 5 times smaller, dt = %g \n", usr->par->dt);

          // return to the jump point, recompute the phase value according to the new time steps, and the time step size for porosity evolution
          goto jump;
        }
        else { break;} // terminate loop if dt is too small

      }
    }


    //StokesDarcy Solver
//    ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);

    // -- update cell-wise phase index
    ierr = CleanUpFPFN(usr->dmf,usr); CHKERRQ(ierr);
    ierr = CollectFPFN(usr->dmf,usr); CHKERRQ(ierr);
    ierr = VecCopy(usr->nfp, usr->fp); CHKERRQ(ierr);
    ierr = VecCopy(usr->nfn, usr->fn); CHKERRQ(ierr);

    // -- update xPV into the usrdata
    ierr = VecCopy(xPV, usr->xPV);CHKERRQ(ierr);
    ierr = VecDestroy(&xPV); CHKERRQ(ierr);

    // Porosity Solver, solve phi for the next time step
    ierr = FDPDESolve(fdphi,NULL);CHKERRQ(ierr);
    ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);

    //--check filter
    //check max(phi) and min(phi),
    PetscScalar phimax, phimin;
    ierr = VecMax(xphi,NULL,&phimax); CHKERRQ(ierr);
    ierr = VecMin(xphi,NULL,&phimin); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "BEFORE FILTER: Porosity field: Maximum of phi = %1.8f, Minimum of phi = %1.8f\n", phimax, phimin); CHKERRQ(ierr);

    ierr = VecCopy(xphi, phibkup); CHKERRQ(ierr);
    
    // filtering out phi<0 and phi>1
    //ierr = PorosityFilter(usr->dmphi, xphi, usr); CHKERRQ(ierr);

    //--check filter
    ierr = VecMax(xphi,NULL,&phimax); CHKERRQ(ierr);
    ierr = VecMin(xphi,NULL,&phimin); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "AFTER FILTER: Porosity field: Maximum of phi = %1.8f, Minimum of phi = %1.8f\n", phimax, phimin); CHKERRQ(ierr);


    // Porosity: copy new solution and coefficient to old
    ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphi,xphiprev);CHKERRQ(ierr);
    ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
    ierr = VecDestroy(&xphiprev);CHKERRQ(ierr);

    ierr = FDPDEGetCoefficient(fdphi,&dmphicoeff,&phicoeff);CHKERRQ(ierr);
    ierr = FDPDEAdvDiffGetPrevCoefficient(fdphi,&phicoeffprev);CHKERRQ(ierr);
    ierr = VecCopy(phicoeff,phicoeffprev);CHKERRQ(ierr);
    ierr = VecDestroy(&phicoeffprev);CHKERRQ(ierr);


    // integrate the plastic strain
  //  ierr = IntegratePlasticStrain(usr->dmeps, usr->strain, usr->xyield, usr); CHKERRQ(ierr);

    // calculate the damage-enhanced permeability
  //  ierr = DamageEnhancedPermeability(usr->dmeps, usr->Kdv, usr->strain, xphi, usr);
   
    // Update evpo
    ierr = VecCopy(usr->evp, usr->evpo); CHKERRQ(ierr);

 
    PetscPrintf(usr->comm,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // update phase value
    ierr = VecCopy(usr->f,usr->fprev); CHKERRQ(ierr);

   // Update xepso, the strain rates at the previous time step
   ierr = VecCopy(usr->xeps, usr->xepso); CHKERRQ(ierr);


    // write before changing time steps
    if (iwrt || istep == 0) {

      iwrt = PETSC_FALSE;
      
      // Output solution to file
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_solution_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmPV,usr->xPV,fout);CHKERRQ(ierr);
      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_strain_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xeps,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stress_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_dp_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xDP,fout);CHKERRQ(ierr);
      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_stressold_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xtau_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_dpold_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xDP_old,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_yield_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->xyield,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_lam_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->strain,fout);CHKERRQ(ierr);
      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_kd_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->Kdv,fout);CHKERRQ(ierr);
      
      /*
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_residual_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmPV,fd->r,fout);CHKERRQ(ierr);
      */
      ierr = FDPDEGetCoefficient(fdPV,&dmcoeff,&xcoeff);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_coefficient_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(dmcoeff,xcoeff,fout);CHKERRQ(ierr);
      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phase_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmf,usr->f,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_pstat_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmf,usr->Pst,fout);CHKERRQ(ierr);

      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phi_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmphi,xphi,fout);CHKERRQ(ierr);
      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phibk_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmphi,phibkup,fout);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_xkee_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmPV,usr->xkee,fout);CHKERRQ(ierr);
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_epsvp_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->epsvp,fout);CHKERRQ(ierr);

      
      ierr = PetscSNPrintf(fout,sizeof(fout),"%s_evp_ts%1.3d",usr->par->fname_out,ickpt);
      ierr = DMStagViewBinaryPython(usr->dmeps,usr->evp,fout);CHKERRQ(ierr);

   
  // PetscASCIIWrite(dtfd, &usr->par->dt, 1, PETSC_SCALAR);

    PetscScalarView(1, &usr->par->dt, dtview);
    }


    // Update xtau_old and xDP_old
    ierr = UpdateStressOld(usr->dmeps,usr);CHKERRQ(ierr);
    
    // Destroy some unused
    ierr = VecDestroy(&xphi); CHKERRQ(ierr);

    //check max(f) and min(f),
    PetscScalar fmax, fmin;
    ierr = VecMax(usr->f,NULL,&fmax); CHKERRQ(ierr);
    ierr = VecMin(usr->f,NULL,&fmin); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "Phase field: Maximum of f = %1.8f, Minimum of f = %1.8f\n", fmax, fmin); CHKERRQ(ierr);


    //check the size of time step
    PetscScalar xxmax, dtt, dtgap;
    PetscInt    imax, isize, isp;
    
    ierr = UpdateUMag(usr->dmPV, usr->xPV, usr); CHKERRQ(ierr);
    ierr = VecMax(usr->VMag,&imax, &xxmax); CHKERRQ(ierr);
    VecGetSize(usr->VMag,&isize);
    VecGetSize(usr->fp,&isp);

    usr->par->gamma = xxmax; CHKERRQ(ierr);
    
   /* 
    if (usr->par->gamma < 1e-5) {usr->par->gamma = 1e-5;}
    //change dt accordingly
    dtt = usr->par->H/nz/usr->par->gamma/4; //maximum time step allowed for boundedness
    if (dtt<1e-8) dtt = 1e-8;
*/

    // temporarily change as a fixed dtt
   /* 
if (usr->par->t < 1e-4) { dtt = 1e-5;}
   else if (usr->par->t < 2e-3) {dtt = 2e-5;}
   else {dtt = 3e-5;}
*/
/*
    else if (usr->par->t < 3e-4) {dtt = 2e-5;}
    else if (usr->par->t < 7e-4) {dtt = 4e-5;}
    else if (usr->par->t < 1.5e-3) {dtt = 8e-5;}
    else if (usr->par->t < 3.1e-3) {dtt = 1.6e-4;}
    else if (usr->par->t < 6.3e-3) {dtt = 3.2e-4;}
    else if (usr->par->t < 2e-2) {dtt = 5e-4;}
    else {dtt = dtck;}    
*/

	dtt =  dtck * usr->par->tscale ;

    if(usr->par->bcidx==12) {
	
	if (usr->par->t < 1e-4) {dtt=1e-5;}
	else if (usr->par->t < 1e-3) {dtt=1e-4;}
	else if (usr->par->t < 5e-3) {dtt=5e-4;}
	else {dtt = dtck;}

//    dtt = dtck;
}

    dtgap = tckpt - usr->par->t;   // gap between the current time and the next checkpoint

    if (dtgap <= 0) SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "dtgap is smaller or equal to zero, the next check points, tckpt, has not been updated properly");
    
    //check if too close to the check point, avoid left any gap smaller than 0.1*dtt for the next
    if (dtgap > 1.01*dtt) {usr->par->dt = dtt;}
    else {usr->par->dt = dtgap; ickpt++; tckpt += dtck; iwrt = PETSC_TRUE;}

    PetscPrintf(usr->comm, "Phase field: Maximum of U = %1.8f at i = %d of %d, %d  \n", xxmax, imax, isize, isp); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "Phase field: gamma = %1.8f\n", usr->par->gamma); CHKERRQ(ierr);

    PetscPrintf(usr->comm, "delta chceck: Kini = %1.8f, dini = %1.8f, delta = %1.8f\n", usr->par->Kini, usr->par->dini, usr->par->sdel); CHKERRQ(ierr);

    // increment timestep
    istep++;

  }

   // destroy the viewer
   PetscViewerDestroy(&dtview);
  // Destroy objects
  ierr = VecDestroy(&usr->f);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->fprev);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->dfx);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->dfz);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->volf);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xphi0);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->x3);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->Pst);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->fp);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->fn);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->nfp);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->nfn);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->VMag);CHKERRQ(ierr);
  
  ierr = VecDestroy(&usr->xeps);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xyield);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xtau_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xDP_old);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->YS);CHKERRQ(ierr);


  ierr = VecDestroy(&usr->xepso);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->epsvp);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->evp);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->evpo);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xkee);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->noise);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->strain);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->Kdv);CHKERRQ(ierr);
  
  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmf); CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmphi); CHKERRQ(ierr);
  
 //   ierr = VecDestroy(&phibkup); CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdPV);CHKERRQ(ierr);
  ierr = FDPDEDestroy(&fdphi);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// InputParameters
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputParameters"
PetscErrorCode InputParameters(UsrData **_usr)
{
  UsrData       *usr;
  Params        *par;
  PetscBag       bag;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Allocate memory to application context
  ierr = PetscMalloc1(1, &usr); CHKERRQ(ierr);

  // Get time, comm and rank
  usr->comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &usr->rank); CHKERRQ(ierr);

  // Create bag
  ierr = PetscBagCreate (usr->comm,sizeof(Params),&usr->bag); CHKERRQ(ierr);
  ierr = PetscBagGetData(usr->bag,(void **)&usr->par); CHKERRQ(ierr);
  ierr = PetscBagSetName(usr->bag,"UserParamBag","- User defined parameters -"); CHKERRQ(ierr);

  // Define some pointers for easy access
  bag = usr->bag;
  par = usr->par;

  // Initialize domain variables
  ierr = PetscBagRegisterInt(bag, &par->nx, 10, "nx", "Element count in the x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->nz, 10, "nz", "Element count in the z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xmin, 0.0, "xmin", "Start coordinate of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zmin, 0.0, "zmin", "Start coordinate of domain in z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->L, 1.2, "L", "Length of domain in x-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->H, 1.2, "H", "Height of domain in z-dir"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Hair, 0.2, "Hair", "Thickness of the sticky air"); CHKERRQ(ierr);

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->F4, 0.0, "F4", "Non-dimensional gravity for fluid 4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F3, 0.0, "F3", "Non-dimensional gravity for fluids 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F2, 0.0, "F2", "Non-dimensional gravity for fluids 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F1, 0.0, "F1", "Non-dimensional gravity for fluids 1"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->FS, 0.0, "FS", "Non-dimensional gravity for the phase 1"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->gth, 0.0, "gangle", "Angle between gravity and vertical axis"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->sth, 0.0, "sth", "Angle of the clockwise rotation of pure shear flow"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->delta, 0.0, "delta", "Angle of the clockwise rotation of pure shear flow"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor"); CHKERRQ(ierr);


  ierr = PetscBagRegisterScalar(bag, &par->p0, 0.0, "p0", "Injection pressure at the botom"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->pfs, 0.0, "pfs", "Static pressure at the free surface"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->dpfs, 0.0, "dpfs", "pressure increase from left to right"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dl, 0.3, "dl", "Relative length of unloading area on the top"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->hpeak, 0, "hpeak", "Relative elevation at the peak of initial topgraphic loading"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->iota, 0.0, "iota", "Coefficients to calculate the effective pressure, p_e = p_c + iota*pf"); CHKERRQ(ierr);
  
  // Viscosity
  ierr = PetscBagRegisterScalar(bag, &par->eta1, 1.0, "eta1", "Viscosity of the fluid 1"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta2, 1.0, "eta2", "Viscosity of the fluid 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta3, 1.0, "eta3", "Viscosity of the fluid 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->eta4, 1.0, "eta4", "Viscosity of the fluid 4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lam_v, 1.0e-1, "lam_v", "Factors for intrinsic visocisty, lam_v = eta/zeta"); CHKERRQ(ierr);
  // Plasticity
  ierr = PetscBagRegisterScalar(bag, &par->C1, 1e40, "C1", "Cohesion (fluid 1)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C2, 1e40, "C2", "Cohesion (fluid 2)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C3, 1e40, "C3", "Cohesion (fluid 3)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->C4, 1e40, "C4", "Cohesion (fluid 4)"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->angle, 0.0, "angle", "Friction angle"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->psi, 0.0, "psi", "Dilation angle"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->R_pe, 4.0, "R_pe", "R_pe = sigma_t/C"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->eta_vk, 1e-1, "eta_vk", "Shear viscosity of the parallel damping dashpot"); CHKERRQ(ierr);

  // Elasticity
  ierr = PetscBagRegisterScalar(bag, &par->G1, 1e40, "G1", "Shear elastic modulus, fluid 1"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->G2, 1e40, "G2", "Shear elastic modulus, fluid 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->G3, 1e40, "G3", "Shear elastic modulus, fluid 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->G4, 1e40, "G4", "Shear elastic modulus, fluid 4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->Z, 1e40, "Z", "Reference poro-elastic modulus"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->q, -0.5, "q", "Exponent of the porosity-dependent relation of poro-elastic modulus"); CHKERRQ(ierr);

  // squre of the ratio of compaction length to the global
  ierr = PetscBagRegisterScalar(bag, &par->R2_1, 1.0, "R2_1", "Square of the relative compaction scale: fluid 1"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R2_2, 1.0, "R2_2", "Sqaure of the relative compaction scale: fluid 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R2_3, 1.0, "R2_3", "Square of the relative compaction scale: fluid 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->R2_4, 1.0, "R2_4", "Square of the relative compaction scale: non in use"); CHKERRQ(ierr);

  // Time steps
  ierr = PetscBagRegisterScalar(bag, &par->dt, 0.01, "dt", "The size of time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tstep, 11, "tstep", "The maximum time steps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->tout,5, "tout", "Output every tout time step"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dtck, 0.1, "dtck", "The size between two check points in time"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->maxckpt, 1, "maxckpt", "Maximum number of check points"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tmax, 1.0, "tmax", "The maximum of dimensionless t"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tscale, 0.5, "tscale", "dt = tscale* dtck"); CHKERRQ(ierr);

  // Parameters for the phase field method
  ierr = PetscBagRegisterScalar(bag, &par->eps, 0.2, "eps", "epsilon in the kernel function"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->gamma, 1.0, "gamma", "gamma in the phase field method"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->vfopt, 3, "vfopt", "vfopt = 0,1,2,3"); CHKERRQ(ierr);


  // Initial porosity distribution
  ierr = PetscBagRegisterScalar(bag, &par->phi_m, 0.1, "phi_m", "Maximum porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phi_bg, 0.0001, "phi_bg", "background porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xsig, 0.1, "xsig", "length scale of Gaussian distribution - x direction"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zsig, 0.1, "zsig", "length scale of Gaussian distribution - z direction"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->xpeak, 0.5, "xpeak", "x location for the peak porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->zpeak, 0.0, "zpeak", "z location for the peak porosity"); CHKERRQ(ierr);

  // Time stepping and advection scheme for poro ADVDIFF
  ierr = PetscBagRegisterInt(bag, &par->ts_scheme,0, "ts_scheme", "Time stepping scheme 0-forward euler, 1-backward euler, 2-crank-nicholson"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->adv_scheme,0, "adv_scheme", "Advection scheme 0-upwind, 1-fromm"); CHKERRQ(ierr);

  // Boundaries
  ierr = PetscBagRegisterScalar(bag, &par->vi, 0, "vi", "outlet velocity on the right boundary"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->viz, 0, "viz", "outlet velocity on the right boundary, z component"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->bcidx, 0, "bcidx", "Choice of boundary conditions: 1, normal extension, 2, extension perpendicular to diagonal"); CHKERRQ(ierr);

  // Circular domain related
  ierr = PetscBagRegisterInt(bag, &par->air, 0, "air", "Initial phase field: 0, default layered structure, 1, circular domain"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->rd, 0.4, "rd", "radius of the circular domain"); CHKERRQ(ierr);


  // Parameters for strain softening
  ierr = PetscBagRegisterScalar(bag, &par->strain_max, 0.01, "strain_max", "the total plastic strain for softening"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->hca, 0.5, "hca", "the relative reduction for the friction angle because of strain softening"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->hcc, 0.5, "hcc", "the relative reduction for the cohesion because of strain softening"); CHKERRQ(ierr);

  // Depth-dependent parameters
  ierr = PetscBagRegisterScalar(bag, &par->erange, 1e4, "erange", "Relative range of viscosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->krange, 1e2, "krange", "Relative range of permeability"); CHKERRQ(ierr);

  // Crack enhanced permeability
  ierr = PetscBagRegisterScalar(bag, &par->ck, 10, "ck", "Max relative enhancement because of cracks"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->strainK, 0.01, "strainK", "Strain scale for the crack enhancement"); CHKERRQ(ierr);

  // Miscs
  ierr = PetscBagRegisterScalar(bag, &par->coeff_iso, 1.0/3.0, "coeff_iso", "Coefficients of the isotropic components"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->Nmax, 25, "Nmax", "Max Newton iteration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tf_tol, 1e-8, "tf_tol", "Function tolerance for solving yielding stresses"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->phicut, 1e-3, "phicut", "Cutoff porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->ncdl, 1, "ncdl", "Exponential power in cdl function"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->nap, 1, "nap", "Exponential power in ap function"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->alphaC, 1.0, "alphaC", "Coefficient in the smooth function of alpha"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->noise_max, 0.05, "noise_max", "Max magnitude of the relative noise"); CHKERRQ(ierr);

  // Controller for the evolution of porosity
  ierr = PetscBagRegisterBool(bag, &par->poroevo, PETSC_TRUE, "poroevo", "Controller for the evolution of porosity"); CHKERRQ(ierr);
  // Switch for the gaussian distribution or not
  ierr = PetscBagRegisterBool(bag, &par->gaussian, PETSC_TRUE, "gaussian", "Switch for the gaussian distribution or not"); CHKERRQ(ierr);

  // Switching interpolation type of porosity at faces and edges
  ierr = PetscBagRegisterInt(bag, &par->poromean, 0, "poromean", "Mean type to interpolate phi: 0, arithmetic, 1, geometric, 2, harmonic"); CHKERRQ(ierr);

  // Type of peameability model and parameters
  ierr = PetscBagRegisterInt(bag, &par->permeability, 0, "permeability", "Permeability model: 0, Kozeny-Carman, 1, poro-fracture"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->km, 1e12, "km", "The maximum permeability enhancement required to equivalent the poro-fracture flow at phi=1 to a channel flow"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->kd, 1, "kd", "Max Newton iteration"); CHKERRQ(ierr);

  // Interpolation type of permeability enhancement
  ierr = PetscBagRegisterInt(bag, &par->kenhance, 0, "kenhance", "Interpolation of permeability on the face: 0, linear, 1, geometric, 2, harmonic"); CHKERRQ(ierr);

  // For boundary conditions
  ierr = PetscBagRegisterScalar(bag, &par->sdel, 0.05, "sdel", "BC at the bottom: dpdz = (1-sdel)*DeltaRhoG"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->Kini, 0, "Kini", "Sum of scaled initial permeability"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->dini, 0.05, "dini", "Initial value to define dpdz, sucha sd dpdz = (1-dini)*DeltaRhoG"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->kscale, 1, "kscale", "Scale of permeability due to change of Dx, for poro-dyke"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->qin, 1, "qin", "Averaged flux rate Q/DeltaX, unit m/yr"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->K0, 1e-8, "K0", "Reference permeability, unit m2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->drhog, 4900, "drhog", "DeltaRho*G, Pa/m"); CHKERRQ(ierr);

// Anisotropic permeability enhancement 

  ierr = PetscBagRegisterScalar(bag, &par->keps_anis, 0.1, "keps_anis", "Parameters for anisotropic permeability enhancement, keps"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->km_anis, 9, "km_anis", "Max anisotropic permeability enhancement, km"); CHKERRQ(ierr);

  // Reference compaction viscosity
  par->zeta1 = par->eta1/par->lam_v;
  par->zeta2 = par->eta2/par->lam_v;
  par->zeta3 = par->eta3/par->lam_v;
  par->zeta4 = par->eta4/par->lam_v;
  
  par->plasticity = PETSC_TRUE;

  // other variables
  par->t = 0.0;

  // Input/output 
  ierr = PetscBagRegisterString(bag,&par->fname_out,FNAME_LENGTH,"out_solution","output_file","Name for output file, set with: -output_file <filename>"); CHKERRQ(ierr);

  // return pointer
  *_usr = usr;

  PetscFunctionReturn(0);
}

// ---------------------------------------
// InputPrintData
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "InputPrintData"
PetscErrorCode InputPrintData(UsrData *usr)
{
  char           date[30], *opts;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Get date
  ierr = PetscGetDate(date,30); CHKERRQ(ierr);

  // Get petsc command options
  ierr = PetscOptionsGetAll(NULL, &opts); CHKERRQ(ierr);

  // Print header and petsc options
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# Test_stokesdarcy2field_phasefield: %s \n",&(date[0]));
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");
  PetscPrintf(usr->comm,"# PETSc options: %s \n",opts);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Print usr bag
  ierr = PetscBagView(usr->bag,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscPrintf(usr->comm,"# --------------------------------------- #\n");

  // Free memory
  ierr = PetscFree(opts); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------
// FormCoefficient_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormCoefficient_PV"
PetscErrorCode FormCoefficient_PV(FDPDE fd, DM dm, Vec x, DM dmcoeff, Vec coeff, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  Vec            coefflocal, xepslocal, xtaulocal, xDPlocal, xyieldlocal, toldlocal, poldlocal, evplocal;
  Vec            xkeelocal;
  Vec            flocal, volflocal, fplocal, fnlocal, xlocal, pplocal, yslocal, nslocal, strainlocal, Kdlocal;
  DM             dmphi;
  Vec            xphi, xphilocal, xphi0, xphi0local;
  PetscScalar    phi0;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter, Nmax;
  PetscScalar    tf_tol;
  PetscScalar    ***c, ***xxs, ***xxy, ***xxp, ***xxevp;
  PetscScalar    dt, Z0, Z,r2local[4],Kphi[4], eta_v,zeta_v,eta_e,zeta_e,zeta_ve_dil;
  PetscScalar    eta_u, eta_d, zeta_u, zeta_d, F_u, F_d, C_d, C_u;
  PetscScalar    etaa[4], Fa[4], Ga[4], Ca[4], R2a[4];
  PetscScalar    angle, sina, Cc, Ct, eta_vk0, eta_vk, R_pe, coeff_iso, phicut, alphaC, nap, ncdl;
  PetscScalar    zmin, H, Hair, erange, krange;
  PetscScalar    hca, hcc, strain_max;  //relative reduction due to strain softening, accumulated strain for softening
  PetscScalar    ck, strainK, xstrainf[4];
  PetscErrorCode ierr;


  PetscFunctionBeginUser;

  zmin = usr->par->zmin;
  H = usr->par->H;

  dt = usr->par->dt;

  Z0 = usr->par->Z;

  // friction angle
  angle = usr->par->angle;
  sina = PetscSinScalar(angle);

  R_pe = usr->par->R_pe;

  coeff_iso = usr->par->coeff_iso; //coefficients of the isotropic components, 1/3 or 1/2
                                   //
  Nmax = usr->par->Nmax;
  tf_tol = usr->par->tf_tol;

  // angle between the gravity and vertical axis
  PetscScalar gcos, gsin;
  gcos = PetscCosScalar(usr->par->gth);
  gsin = PetscSinScalar(usr->par->gth);

  // strain softening related
  hca = usr->par->hca;
  hcc = usr->par->hcc;
  strain_max = usr->par->strain_max;

  // thickness of sticky air
  Hair = usr->par->Hair;

  // range of depth-dependent parameters
  erange = usr->par->erange;//1e4; //viscosity range
  krange = usr->par->krange;//1e2; //permeability range

  // permeability amplifier due to plastic strain
  ck = usr->par->ck;
  // strain scales to amplify the permeability due to plastic strain
  strainK = usr->par->strainK;

  // prepare arrays to store fluid parameters
  
  etaa[0] = 1.0;
  etaa[1] = usr->par->eta1;
  etaa[2] = usr->par->eta2;
  etaa[3] = usr->par->eta3;
  Fa[0] = 0.0;
  Fa[1] = usr->par->F1;
  Fa[2] = usr->par->F2;
  Fa[3] = usr->par->F3;
  Ga[0] = usr->par->G1;
  Ga[1] = usr->par->G1;
  Ga[2] = usr->par->G2;
  Ga[3] = usr->par->G3;
  Ca[0] = 1.0;
  Ca[1] = usr->par->C1;
  Ca[2] = usr->par->C2;
  Ca[3] = usr->par->C3;

  R2a[0] = 0.0;
  R2a[1] = usr->par->R2_1;
  R2a[2] = usr->par->R2_2;
  R2a[3] = usr->par->R2_3;

  phicut = usr->par->phicut;
  alphaC = usr->par->alphaC;
  ncdl = usr->par->ncdl;
  nap = usr->par->nap;

  PetscScalar Z_max = Z0*20, eta_min = 1e-3 *etaa[1];

  //reference porosity
  phi0 = usr->par->phi_0;

  // Get dm and solution vector for porosity
  dmphi = usr->dmphi;
  xphi  = usr->xphiprev;

  ierr = DMCreateLocalVector(dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

 
  xphi0  = usr->xphi0;

  ierr = DMCreateLocalVector(dmphi,&xphi0local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi0,INSERT_VALUES,xphi0local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi0,INSERT_VALUES,xphi0local);CHKERRQ(ierr);


  // viscosity in the parallel damping dashpot
  eta_vk0 = usr->par->eta_vk;

  // Effective shear and Compaction viscosity due to elasticity
  zeta_e = Z0*dt;

  // temporary use - to be delted
  eta_e = Ga[1]*dt;

  // phase field
  ierr = DMGetLocalVector(usr->dmf, &flocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->f, INSERT_VALUES, flocal); CHKERRQ(ierr);

  // volume fraction
  ierr = DMGetLocalVector(usr->dmf, &volflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->volf, INSERT_VALUES, volflocal); CHKERRQ(ierr);

  // indices of positive and negative fluids
  ierr = DMGetLocalVector(usr->dmf, &fplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->fp, INSERT_VALUES, fplocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmf, &fnlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->fn, INSERT_VALUES, fnlocal); CHKERRQ(ierr);

  // local vectors for noise
  ierr = DMGetLocalVector(usr->dmeps, &nslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->noise, INSERT_VALUES, nslocal); CHKERRQ(ierr);

  // solution
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);
  // Plastic Strain rates
  ierr = DMCreateLocalVector (usr->dmeps,&evplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,evplocal,&xxevp); CHKERRQ(ierr);

  // Get the face-wise permeability modification
  ierr = DMGetLocalVector(usr->dmPV, &xkeelocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xkee, INSERT_VALUES, xkeelocal); CHKERRQ(ierr);

  // Local vector for the plastic strain
  ierr = DMGetLocalVector(usr->dmeps, &strainlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->strain, INSERT_VALUES, strainlocal); CHKERRQ(ierr);
  
  // Local vector for the damage-enhanced permeability
  ierr = DMGetLocalVector(usr->dmeps, &Kdlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->Kdv, INSERT_VALUES, Kdlocal); CHKERRQ(ierr);
  
  // Stress_old
  ierr = DMGetLocalVector(usr->dmeps, &toldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &poldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xDP_old, INSERT_VALUES, poldlocal); CHKERRQ(ierr);

  // hydrostatic pressure
  ierr = DMGetLocalVector(usr->dmf, &pplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf, usr->Pst, INSERT_VALUES, pplocal); CHKERRQ(ierr);

  // upadte YS
  ierr = UpdateYS(usr->dmPV, usr->xPV, usr->dmf, usr->Pst, usr->dmeps, usr->xtau_old, usr->xDP_old, usr); CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(usr->dmeps, &yslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->YS, INSERT_VALUES, yslocal); CHKERRQ(ierr);

  // Local vectors
  ierr = DMCreateLocalVector (usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xDPlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);

  ierr = DMCreateLocalVector (usr->dmeps,&xyieldlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);



  // integrate the plastic strain
  ierr = IntegratePlasticStrain(usr->dmeps, usr->strain, usr->xyield, usr); CHKERRQ(ierr);

  // calculate the damage-enhanced permeability
  ierr = DamageEnhancedPermeability(usr->dmeps, usr->Kdv, usr->strain, xphi, usr);

  // Get domain corners
  ierr = DMStagGetCorners(dmcoeff, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmcoeff,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,RIGHT,&inext);CHKERRQ(ierr); 
  ierr = DMStagGetProductCoordinateLocationSlot(dmcoeff,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create coefficient local vector
  ierr = DMCreateLocalVector(dmcoeff, &coefflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray  (dmcoeff, coefflocal, &c); CHKERRQ(ierr);

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscInt idx;
      PetscScalar told_xx_e,told_zz_e,told_xz_e,told_II_e,pold,chis,chip,zeta;
      PetscScalar cs[4],told_xx[4],told_zz[4],told_xz[4],told_II[4];
      PetscScalar fp, fn;
      PetscInt    ifp, ifn;
      PetscScalar DeltaP, pstat, psdl;

      PetscScalar rvx[4], rvz[4]; //relative ratio of exx and ezz in Div(V), used for the permeability enhancement on certain directions. 

      // prepare the porosity data for:
      // phi: element
      // phif: left, right, down and up
      // phiv: dl, dr, ul, ur
      DMStagStencil pe[9];
      PetscScalar   phie[9], phi, phif[4],phiv[4], Kdc[9], phie0;
      pe[0].i = i;   pe[0].j = j;   pe[0].loc = ELEMENT;  pe[0].c = 0;
      pe[1].i = i-1; pe[1].j = j;   pe[1].loc = ELEMENT;  pe[1].c = 0;
      pe[2].i = i+1; pe[2].j = j;   pe[2].loc = ELEMENT;  pe[2].c = 0;
      pe[3].i = i;   pe[3].j = j-1; pe[3].loc = ELEMENT;  pe[3].c = 0;
      pe[4].i = i;   pe[4].j = j+1; pe[4].loc = ELEMENT;  pe[4].c = 0;
      pe[5].i = i-1; pe[5].j = j-1; pe[5].loc = ELEMENT;  pe[5].c = 0;
      pe[6].i = i+1; pe[6].j = j-1; pe[6].loc = ELEMENT;  pe[6].c = 0;
      pe[7].i = i-1; pe[7].j = j+1; pe[7].loc = ELEMENT;  pe[7].c = 0;
      pe[8].i = i+1; pe[8].j = j+1; pe[8].loc = ELEMENT;  pe[8].c = 0;

      if (i==0) {pe[1].i = pe[0].i; pe[5].i = pe[0].i; pe[7].i = pe[0].i;}
      if (j==0) {pe[3].j = pe[0].j; pe[5].j = pe[0].j; pe[6].j = pe[0].j;}
      if (i==Nx-1) {pe[2].i = pe[0].i; pe[6].i = pe[0].i; pe[8].i = pe[0].i;}
      if (j==Nz-1) {pe[4].j = pe[0].j; pe[7].j = pe[0].j; pe[8].j = pe[0].j;}

      ierr = DMStagVecGetValuesStencil(usr->dmphi,xphilocal,9,pe,phie); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(usr->dmeps,Kdlocal,9,pe,Kdc); CHKERRQ(ierr);

      // Get the initial porosity for calculating Kphi at the bottom
      ierr = DMStagVecGetValuesStencil(usr->dmphi,xphi0local,1,&pe[0],&phie0); CHKERRQ(ierr);
      phie0 = 1- phie0;

	//phie is used to calculate rheological parameter, make sure the solid fraction smaller than 1.
      PetscInt ii;
	for (ii=0; ii<9; ii++) {
	if (phie[ii] > 1-usr->par->phi_bg) {phie[ii]=1-usr->par->phi_bg;}
	}	



      phi = 1.0 - phie[0];
      if (usr->par->poromean == 0) {
	//arithmetic averaging	
      phif[0] = 1.0 - 0.5*(phie[0]+phie[1]);
      phif[1] = 1.0 - 0.5*(phie[0]+phie[2]);
      phif[2] = 1.0 - 0.5*(phie[0]+phie[3]);
      phif[3] = 1.0 - 0.5*(phie[0]+phie[4]);
      phiv[0] = 1.0 - 0.25*(phie[0]+phie[1]+phie[3]+phie[5]);
      phiv[1] = 1.0 - 0.25*(phie[0]+phie[2]+phie[3]+phie[6]);
      phiv[2] = 1.0 - 0.25*(phie[0]+phie[1]+phie[4]+phie[7]);
      phiv[3] = 1.0 - 0.25*(phie[0]+phie[2]+phie[4]+phie[8]);
	} else if (usr->par->poromean == 1) {
      // geometric averaging
	PetscInt ii;
      	PetscScalar phia[9];
	for (ii = 0; ii < 9; ii++) {phia[ii] = 1 - phie[ii];}
	phif[0] = pow(phia[0]*phia[1], 0.5);
	phif[1] = pow(phia[0]*phia[2], 0.5);
	phif[2] = pow(phia[0]*phia[3], 0.5);
	phif[3] = pow(phia[0]*phia[4], 0.5);
	phiv[0] = pow(phia[0]*phia[1]*phia[3]*phia[5], 0.25);
	phiv[1] = pow(phia[0]*phia[2]*phia[3]*phia[6], 0.25);
	phiv[2] = pow(phia[0]*phia[1]*phia[4]*phia[7], 0.25);
	phiv[3] = pow(phia[0]*phia[2]*phia[4]*phia[8], 0.25);
      } else if (usr->par->poromean ==2) {
	// harmonic averaging
      
	PetscInt ii;
      	PetscScalar phia[9];
	for (ii = 0; ii < 9; ii++) {phia[ii] = 1 - phie[ii];}
	phif[0] = 1/(1/phia[0] + 1/phia[1]);
	phif[1] = 1/(1/phia[0] + 1/phia[2]);
	phif[2] = 1/(1/phia[0] + 1/phia[3]);
	phif[3] = 1/(1/phia[0] + 1/phia[4]);
	phiv[0] = 1/(1/phia[0] + 1/phia[1] + 1/phia[3] + 1/phia[5]);
	phiv[1] = 1/(1/phia[0] + 1/phia[2] + 1/phia[3] + 1/phia[6]);
	phiv[2] = 1/(1/phia[0] + 1/phia[1] + 1/phia[4] + 1/phia[7]);
	phiv[3] = 1/(1/phia[0] + 1/phia[2] + 1/phia[4] + 1/phia[8]);
      } else {
	SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "Interpolation type for porosity at faces and edges is not defined.");
	}

	PetscScalar phiK[4];
	phiK[0] = 1.0 - PetscMin(phie[0], phie[1]);
	phiK[1] = 1.0 - PetscMin(phie[0], phie[2]);
	phiK[2] = 1.0 - PetscMin(phie[0], phie[3]);
	phiK[3] = 1.0 - PetscMin(phie[0], phie[4]);

	PetscScalar Kdf[4];
        Kdf[0] = PetscMax(Kdc[0], Kdc[1]);
        Kdf[1] = PetscMax(Kdc[0], Kdc[2]);
        Kdf[2] = PetscMax(Kdc[0], Kdc[3]);
        Kdf[3] = PetscMax(Kdc[0], Kdc[4]);


      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   epsII,exx,ezz,exz,txx,tzz,txz,tauII,epsII_dev, evpxx, evpzz, evpxz;
        PetscScalar   eta, evp[3];
        PetscScalar   ff, volf;

        // ratio of the solid phase
        PetscScalar phis;
        phis = 1 - phi;

        // get the phase values in the element
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf); CHKERRQ(ierr);

        // get the noise values in the element
        PetscScalar  ns;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,nslocal,1,&point,&ns); CHKERRQ(ierr);

        // get the fluid parameters for this cell
        //---------------
        ierr = DMStagVecGetValuesStencil(usr->dmf,fplocal,1,&point,&fp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,fnlocal,1,&point,&fn); CHKERRQ(ierr);
        ifp = (PetscInt)fp;
        ifn = (PetscInt)fn;


        PetscScalar  eta_factor, zeta_factor, Z_factor, eh, zp;

        zp = coordz[j][icenter];
        eh = PetscPowScalar(erange, (zp-zmin)/(H-Hair));

        eta_factor = PetscExpScalar(-27.0*phi)/phis;
        Z_factor = 1; //PetscPowScalar(phi, -0.5);


        eta_u = etaa[ifp]* eta_factor *eh;
        eta_d = etaa[ifn]* eta_factor *eh;

//	eta_e = Ga[1]*dt * pow((1-phi),2);

        if (eta_u < eta_min) eta_u = eta_min;
        if (eta_d < eta_min) eta_d = eta_min;

        Z = Z0* Z_factor;
        zeta_e = Z*dt;


        zeta_factor = 1/pow(phi, 1)/phis;//(1-phi);

        if (zeta_factor > 1e3) zeta_factor = 1e3;

        zeta_u = etaa[ifp] * zeta_factor *eh;
        zeta_d = etaa[ifn] * zeta_factor *eh;

        if (Z > Z_max) Z = Z_max;
        zeta_e = Z*dt;

        F_u = Fa[ifp];
        F_d = Fa[ifn];


        //noise on friction angle

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,nslocal,1,&point,&ns); CHKERRQ(ierr);

        PetscScalar xstrain, xsoft;
        point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,strainlocal,1,&point,&xstrain); CHKERRQ(ierr);
        xsoft = xstrain/strain_max;
        if (xsoft >= 1.0) xsoft = 1.0;

        C_u = Ca[ifp] * (1+ns) * (1 - hcc*xsoft ); // /phis;
        C_d = Ca[ifn] * (1+ns) * ( 1 - hcc*xsoft ); // /phis;


        angle = usr->par->angle * (1+ns) * ( 1 - hca*xsoft );
        sina = PetscSinScalar(angle);

        // second invariant of strain rate
        point.i = i; point.j = j; point.loc = ELEMENT; 
        point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xx_e); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,poldlocal,1,&point,&pold); CHKERRQ(ierr);
        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_zz_e); CHKERRQ(ierr);
        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_xz_e); CHKERRQ(ierr);
        point.c = 3;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,1,&point,&epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&told_II_e); CHKERRQ(ierr);

        // effective deviatoric strain rates
        PetscScalar exxp, ezzp, exzp, div13;
        div13 = (exx + ezz) * coeff_iso;
        exxp = ((exx-div13) + 0.5*told_xx_e/eta_e);
        ezzp = ((ezz-div13) + 0.5*told_zz_e/eta_e);
        exzp = (exz + 0.5*told_xz_e/eta_e);

/*
        // Get the old strain rates, and calculate permeability enhancement ratio based on it
        PetscScalar exxo[4], ezzo[4], exxf[4], ezzf[4],divf;
	DMStagStencil pt[4];
	pt[0].i = i; pt[0].j = j; pt[0].loc = DOWN_LEFT; pt[0].c = 0;
	pt[1].i = i; pt[1].j = j; pt[1].loc = DOWN_RIGHT; pt[1].c = 0;
	pt[2].i = i; pt[2].j = j; pt[2].loc = UP_LEFT; pt[2].c = 0;
	pt[3].i = i; pt[3].j = j; pt[3].loc = UP_RIGHT; pt[3].c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepsolocal,4,pt,exxo); CHKERRQ(ierr);
        pt[0].c = 1;
        pt[1].c = 1;
        pt[2].c = 1;
        pt[3].c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepsolocal,4,pt,ezzo); CHKERRQ(ierr);

	exxf[0] = 0.5*(exxo[0] + exxo[2]); //left
	exxf[1] = 0.5*(exxo[1] + exxo[3]); //right
	exxf[2] = 0.5*(exxo[0] + exxo[1]); //down
	exxf[3] = 0.5*(exxo[2] + exxo[3]); //UP
	
	ezzf[0] = 0.5*(ezzo[0] + ezzo[2]); //left
	ezzf[1] = 0.5*(ezzo[1] + ezzo[3]); //right
	ezzf[2] = 0.5*(ezzo[0] + ezzo[1]); //down
	ezzf[3] = 0.5*(ezzo[2] + ezzo[3]); //UP

//	divo = exxo + ezzo;

	// calculate rvx and rvz for permeability enhancement

	PetscInt iii;
	for(iii=0; iii<4; iii++) {
	  divf = exxf[iii] + ezzf[iii];	

	  if (divf > 1e-1 && usr->par->t > usr->par->dtck ) {
		
		rvx[iii] = exxf[iii]/divf;
		rvz[iii] = ezzf[iii]/divf;
	
	  }  
	  else {rvx[iii] = 0.5; rvz[iii] = 0.5;}
	}	
*/

        // second invariant of (eps + tau_old/(g*dt))
        epsII_dev = 0.5*(PetscPowScalar(exxp, 2) + PetscPowScalar(ezzp, 2));
        epsII_dev += PetscPowScalar(exzp, 2);
        epsII_dev = PetscPowScalar(epsII_dev, 0.5);

        // divp = (div - pold/zeta_e);
        PetscScalar divp;
        divp = ((exx+ezz) - pold/zeta_e);


        // extract the effective pressure from yslocal, and the pressure modifier
        PetscScalar pfterm;
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,yslocal,1,&point,&pstat); CHKERRQ(ierr);
        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,yslocal,1,&point,&pfterm); CHKERRQ(ierr);

        PetscScalar eta_ve, zeta_ve, txxt, tzzt, txzt, tIIt, pct;
        PetscScalar ap, stressP[2], stressA[5], stressSol[3], dotlam=0;

        // static pressure correction on the effective pressure, ap =(1- alpha)*pf*sin(angle)
        // P_e = (1-phi)*(\Delta P + ap);
        // if (phi > phicut) {ap = 0;}
        // else {ap = pfterm ;}
        ap = pfterm * (1 - PetscExpScalar(-PetscPowScalar(phicut/phi / alphaC,nap)));

        if (volf > 1e-8) {  // compute only UP fluids has a volume fraction greater than 1e-8

        // POSITIVE FLUIDS: compute eta and zeta
        eta_v = eta_u; zeta_v = zeta_u;
        Cc = C_u;
        Ct = Cc/R_pe;

        // eta, check trial stress
        eta_ve = eta_v*eta_e/(eta_v + eta_e);
        zeta_ve = zeta_v*zeta_e/(zeta_v + zeta_e);

        // compute the effective trial stresses
        txxt = 2*eta_ve*exxp;
        tzzt = 2*eta_ve*ezzp;
        txzt = 2*eta_ve*exzp;
        tIIt = PetscPowScalar(0.5*(txxt*txxt+tzzt*tzzt)+txzt*txzt, 0.5);
        pct = -zeta_ve * divp;

        // stress prediction
        stressP[0] = tIIt;
        stressP[1] = pct;

        PetscScalar cdl;
	cdl = PetscExpScalar(-PetscPowScalar(phicut/phi,ncdl));
        zeta_ve_dil = zeta_ve * cdl;

	//Ct = Ct * cdl;

        // get the eta_vk value -- a customised porosity dependency function
        eta_vk = eta_vk0;// * PetscExpScalar(-phi/phicut * alphaC) ;

        // get the deviatoric and isotropic stresses
        ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
        dotlam = stressSol[2]*volf;

	// Two types of eta
	PetscScalar etaY, etaN, etatmp;
	etaY = 0.5*stressSol[0]/epsII_dev * phis*volf;
	etaN = eta_ve *phis*volf;

        // effective viscosity
        if (epsII_dev > tf_tol && usr->par->plasticity  ) {
          etatmp =   etaY;      } else {
          etatmp =   etaN;      }
	eta = etatmp;
	/*
        if (phi>phicut && PetscAbs(divp) > tf_tol) {
          zeta = -stressSol[1]/divp * phis*volf;
        } else {
          zeta = zeta_ve *phis*volf;
        }
	*/
	
	zeta = zeta_ve * phis *volf; //Discard the concept of zeta_eff, use zeta_ve always
	psdl = (stressSol[1] - pct) *phis*volf; //psdl = DeltaP-DeltaP_ve 

	if (!usr->par->plasticity) {psdl=0;}

	// The relative ratio which will be used to calculate deviatoric plastic strain rates
	// and the isotropic plastic strain rates
	PetscScalar etaR, evpv;
	etaR = (1 - etatmp/etaN)*volf;
	if (PetscAbs(divp) > tf_tol) {
	evpv = (stressSol[1] - pct)/divp * volf;
	} else {
	evpv = 0;
	}

	evpxx = etaR * exxp + evpv/3; 
        evpzz = etaR * ezzp + evpv/3; 
	evpxz = etaR * exzp + evpv/3;	
	

        } else {  // if no UP fluid, initiate eta and zeta as zero for the computation of DOWN fluids.
          eta = 0.0;
          zeta = 0.0;
        }

        if (1-volf > 1e-8) { // compute if the DOWN fluids have a volume fraction greater than 1e-8
        // NEGATIVE FLUIDS: compute eta and zeta
        eta_v = eta_d; zeta_v = zeta_d;
        Cc = C_d;
        Ct = Cc/R_pe;

        // eta, check trial stress
        eta_ve = eta_v*eta_e/(eta_v + eta_e);
        zeta_ve = zeta_v*zeta_e/(zeta_v + zeta_e);

        // compute the effective trial stresses
        txxt = 2*eta_ve*exxp;
        tzzt = 2*eta_ve*ezzp;
        txzt = 2*eta_ve*exzp;
        tIIt = PetscPowScalar(0.5*(txxt*txxt+tzzt*tzzt)+txzt*txzt, 0.5);
        pct = -zeta_ve * divp;

        // stress prediction
        stressP[0] = tIIt;
        stressP[1] = pct;

        PetscScalar cdl;
        cdl = PetscExpScalar(-PetscPowScalar(phicut/phi,ncdl));
        zeta_ve_dil = zeta_ve * cdl;

	//Ct = Ct*cdl;

        // get the eta_vk value
        eta_vk = eta_vk0;// * PetscExpScalar(-phi/phicut * alphaC) ;

        // get the deviatoric and isotropic stresses
        ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
        dotlam += stressSol[2]*(1-volf);

	// Two types of eta
	PetscScalar etaY, etaN, etatmp;
	etaY = 0.5*stressSol[0]/epsII_dev * phis*(1-volf);
	etaN = eta_ve *phis*(1-volf);
        
	// effective viscosity
        if (epsII_dev > tf_tol && usr->par->plasticity) {
          etatmp = etaY;
        } else {
          etatmp = etaN;
        }
	eta += etatmp;
       /*
       	if (cdl > 1e-8 && PetscAbs(divp) > tf_tol) {
          zeta += -stressSol[1]/divp * phis*(1-volf);
        } else {
          zeta += zeta_ve *phis*(1-volf);
        }
	*/
	zeta += zeta_ve * phis *(1-volf); //Discard the concept of zeta_eff, use zeta_ve always
	psdl += (stressSol[1] - pct) *phis*(1-volf); //psdl = DeltaP-DeltaP_ve 

	if (!usr->par->plasticity) {psdl = 0;}

	// The relative ratio which will be used to calculate deviatoric plastic strain rates
	// and the isotropic plastic strain rates
	PetscScalar etaR, evpv;
	etaR = (1 - etatmp/etaN)*(1-volf);
	if (PetscAbs(divp) > tf_tol) {
	evpv = (stressSol[1] - pct)/divp * (1-volf);
	} else {
	evpv = 0;
	}
	evpxx += etaR * exxp + evpv/3; 
        evpzz += etaR * ezzp + evpv/3; 
	evpxz += etaR * exzp + evpv/3;	
	
	}

        // elastic stress evolution parameter
        chis = eta/eta_e;
        chip = zeta/zeta_e;


        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 1;

        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = eta;


        txx = 2*eta*exxp/phis;
        tzz = 2*eta*ezzp/phis;
        txz = 2*eta*exzp/phis;
        tauII = PetscPowScalar(0.5*(txx*txx + tzz*tzz + 2*txz*txz),0.5);  // this should be roughly equal to stressSol[0]

        // volumetric stress
        //DeltaP = -zeta*divp/phis;
	DeltaP = -zeta*divp/phis + psdl/phis;  //fix DeltaP according to the dilatancy
        
        // save stresses for output
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = dotlam; xxp[j][i][idx] = DeltaP;
	xxevp[j][i][idx] = evpxx;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz; xxy[j][i][idx] = eta; // save eta in dof=1 of usr->xyield
        xxevp[j][i][idx] = evpzz;
	ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz; xxy[j][i][idx] = chis; // save chis in dof = 2 of usr->xyield
        xxevp[j][i][idx] = evpxz;
	ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII;
      }

      { // A = eta (corner, c=0)
        // also compute cs = eta/(G*dt) (four corners, c =1)
        DMStagStencil point[4];
        PetscScalar   ff[4], volf[4], eta;
        PetscScalar   epsII[4],exx[4],ezz[4],exz[4],txx[4],tzz[4],txz[4],tauII[4],epsII_dev[4],poldv[4];
        PetscInt      ii;

        // second invariant of strain rate
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // collect phase values for the four corners
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,4,point,ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);
        
        
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,poldlocal,4,point,poldv); CHKERRQ(ierr);
                
        for (ii = 0; ii < 4; ii++) {point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,ezz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_zz); CHKERRQ(ierr);

        for (ii = 0; ii < 4; ii++) {point[ii].c = 2;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,exz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_xz); CHKERRQ(ierr);

        // second invariant of strain rate
        for (ii = 0; ii < 4; ii++) {point[ii].c = 3;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,xepslocal,4,point,epsII); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,4,point,told_II); CHKERRQ(ierr);

        // effective deviatoric strain rates
        PetscScalar exxp[4], ezzp[4], exzp[4], div13, phis[4], divp[4], dotlam[4] = {0};
        

        // second invariant of deviatoric strain rate
        for (ii = 0; ii < 4; ii++) {
          phis[ii] = 1 - phiv[ii];

          div13 = (exx[ii] + ezz[ii]) * coeff_iso;
          exxp[ii] = ((exx[ii]-div13) + 0.5*told_xx[ii]/eta_e);
          ezzp[ii] = ((ezz[ii]-div13) + 0.5*told_zz[ii]/eta_e);
          exzp[ii] = (exz[ii] + 0.5*told_xz[ii]/eta_e);
          // second invariant of (eps + tau_old/(g*dt))*(1-phi)
          epsII_dev[ii] = 0.5*(PetscPowScalar(exxp[ii], 2) + PetscPowScalar(ezzp[ii], 2));
          epsII_dev[ii] += PetscPowScalar(exzp[ii], 2);
          epsII_dev[ii] = PetscPowScalar(epsII_dev[ii], 0.5);
          divp[ii] = ((exx[ii]+ezz[ii]) - poldv[ii]/zeta_e);
        }

        
        // extract the effective pressure from yslocal, and the pressure modifier
        
        PetscScalar pfterm[4], xstrain[4], ns[4];
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;

        // accumulated strains
        ierr = DMStagVecGetValuesStencil(usr->dmeps, strainlocal, 4, point, xstrain); CHKERRQ(ierr); // plastic strain


        for(ii=0;ii<4;ii++){ point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,yslocal,4,point,pfterm); CHKERRQ(ierr);

        //noise data
        ierr = DMStagVecGetValuesStencil(usr->dmeps,nslocal,4,point,ns); CHKERRQ(ierr);


        // interpolate the total strains on the faces, LEFT, RIGHT, DOWN UP
        xstrainf[0] = 0.5*(xstrain[0] + xstrain[2]);
        xstrainf[1] = 0.5*(xstrain[1] + xstrain[3]);
        xstrainf[2] = 0.5*(xstrain[0] + xstrain[1]);
        xstrainf[3] = 0.5*(xstrain[2] + xstrain[3]);

                
        for (ii = 0; ii < 4; ii++) {

          PetscScalar eta_ve, zeta_ve, txxt, tzzt, txzt, tIIt, ap, pct, stressP[2], stressA[5], stressSol[3];

          PetscScalar  eta_factor, Z_factor, zeta_factor, eh, zp;

          PetscScalar xsoft;

          xsoft = xstrain[ii]/strain_max;
          if (xsoft >= 1.0) xsoft = 1.0;

          C_u = Ca[ifp] * (1+ns[ii]) * (1 - hcc*xsoft); // /phis[ii] ;
          C_d = Ca[ifn] * (1+ns[ii]) * (1 - hcc*xsoft); // /phis[ii] ;

          angle = usr->par->angle * (1+ns[ii])* ( 1 - hca*xsoft );
          sina = PetscSinScalar(angle);

          eta_factor = PetscExpScalar(-27.0*phiv[ii])/phis[ii];  //(1-phiv[ii]);
          Z_factor = 1; //PetscPowScalar(phiv[ii], -0.5);
          zeta_factor = 1 /pow(phiv[ii], 1)/phis[ii]; //(1-phiv[ii]);

          if (ii < 2) {
            zp = coordz[j][iprev];
            } else {
            zp = coordz[j][inext];
          }
          eh = PetscPowScalar(erange, (zp-zmin)/(H-Hair));

          if (zeta_factor >1e3) zeta_factor = 1e3;

          eta_u = etaa[ifp]* eta_factor *eh;
          eta_d = etaa[ifn]* eta_factor *eh;

          if (eta_u < eta_min) eta_u = eta_min;
          if (eta_d < eta_min) eta_d = eta_min;

          Z = Z0* Z_factor;
          zeta_e = Z*dt;

          zeta_u = etaa[ifp]*zeta_factor *eh;
          zeta_d = etaa[ifn]*zeta_factor *eh;

          if (Z > Z_max) Z = Z_max;
          zeta_e = Z*dt;

          // static pressure correction on the effective pressure, ap = (1-alpha)*pf*sin(angle)
          // P_e = (1-phi)*(\Delta P + ap);
//          if (phiv[ii] > phicut) {ap = 0;}
//          else {ap = pfterm[ii] ;}// zeta_u = 0; zeta_d = 0;}
          ap = pfterm[ii] * (1 - PetscExpScalar(-PetscPowScalar(phicut/phiv[ii] / alphaC,nap)));



          if (volf[ii] > 1e-8) {
          // POSITIVE FLUIDS: compute eta and zeta
          eta_v = eta_u; zeta_v = zeta_u;
          Cc = C_u;
          Ct = Cc/R_pe;

          // eta, check trial stress
          eta_ve = eta_v*eta_e/(eta_v + eta_e);
          zeta_ve = zeta_v*zeta_e/(zeta_v + zeta_e);

          // compute the effective trial stresses
          txxt = 2*eta_ve*exxp[ii];
          tzzt = 2*eta_ve*ezzp[ii];
          txzt = 2*eta_ve*exzp[ii];
          tIIt = PetscPowScalar(0.5*(txxt*txxt+tzzt*tzzt)+txzt*txzt, 0.5);
          pct = -zeta_ve*divp[ii];

          // stress prediction
          stressP[0] = tIIt;
          stressP[1] = pct;

          PetscScalar cdl;
          cdl =  PetscExpScalar(-PetscPowScalar(phicut/phiv[ii],ncdl));
          zeta_ve_dil = zeta_ve * cdl;

//	Ct = Ct * cdl;

          // get the eta_vk value
          eta_vk = eta_vk0;//* PetscExpScalar(-phiv[ii]/phicut * alphaC) ;
; //* (1 + 9*0.5*( 1 + PetscTanhScalar((zp - 0.75)/0.02  )  ));

          // get the deviatoric and isotropic stresses
          ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
          dotlam[ii] = stressSol[2] * volf[ii];

          // effective viscosity
          if (epsII_dev[ii] > tf_tol && usr->par->plasticity) {
            eta = 0.5*stressSol[0]/epsII_dev[ii] * phis[ii]*volf[ii];
          } else {
            eta = eta_ve *phis[ii]*volf[ii];
          }
          } else {  // if no UP fluid, initiate eta and zeta as zero for the computation of DOWN fluids.
            eta = 0.0;
          }

          if (1-volf[ii]>1e-8) {
          // Negative FLUIDS: compute eta and zeta
          eta_v = eta_d; zeta_v = zeta_d;
          Cc = C_d;
          Ct = Cc/R_pe;

          // eta, check trial stress
          eta_ve = eta_v*eta_e/(eta_v + eta_e);
          zeta_ve = zeta_v*zeta_e/(zeta_v + zeta_e);

          // compute the effective trial stresses
          txxt = 2*eta_ve*exxp[ii];
          tzzt = 2*eta_ve*ezzp[ii];
          txzt = 2*eta_ve*exzp[ii];
          tIIt = PetscPowScalar(0.5*(txxt*txxt+tzzt*tzzt)+txzt*txzt, 0.5);
          pct = -zeta_ve*divp[ii];


          // stress prediction
          stressP[0] = tIIt;
          stressP[1] = pct;

          PetscScalar cdl;
          cdl = PetscExpScalar(-PetscPowScalar(phicut/phiv[ii],ncdl));
          zeta_ve_dil = zeta_ve * cdl;

//	Ct = Ct*cdl;

          // get the eta_vk value
          eta_vk = eta_vk0; //* PetscExpScalar(-phiv[ii]/phicut * alphaC) ;
; //* (1 + 9*0.5*( 1 + PetscTanhScalar((zp - 0.75)/0.02  )  ));

          // get the deviatoric and isotropic stresses
          ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
          dotlam[ii] += stressSol[2] * (1-volf[ii]);

          // effective viscosity
          if (epsII_dev[ii] > tf_tol && usr->par->plasticity) {
            eta += 0.5*stressSol[0]/epsII_dev[ii] * phis[ii]*(1-volf[ii]);
          } else {
            eta += eta_ve *phis[ii]*(1-volf[ii]);
          }
          }


          // elastic stress evolution parameter
          cs[ii] = eta/eta_e;

          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, 0, &idx); CHKERRQ(ierr);
          c[j][i][idx] = eta;


          // deviatoric stress and its second invariant
          txx[ii] = 2*eta*exxp[ii]/phis[ii];
          tzz[ii] = 2*eta*ezzp[ii]/phis[ii];
          txz[ii] = 2*eta*exzp[ii]/phis[ii];
          tauII[ii] = PetscPowScalar(0.5*(txx[ii]*txx[ii] + tzz[ii]*tzz[ii] + 2.0*txz[ii]*txz[ii]),0.5);
        }

        // save stresses
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[0]; xxy[j][i][idx] = dotlam[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[0];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[0];

        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[1]; xxy[j][i][idx] = dotlam[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[1];
        ierr = DMStagGetLocationSlot(usr->dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[1];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[2]; xxy[j][i][idx] = dotlam[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[2];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[2];

        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx[3]; xxy[j][i][idx] = dotlam[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz[3];
        ierr = DMStagGetLocationSlot(usr->dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tauII[3];
      }


      // F2 and F3 are used in calculating B and D3 as well.
      PetscScalar  F0, F1, F2, F3;

      { // B = phi*F*ek - grad(chi_s*tau_old) + div(chi_p*dP_old) (edges, c=0)
        // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2)
        // hydrostatic pressure fix term according to free surface: dpstat/dx
	// EXTRA terms Div(psdl)
        DMStagStencil point[4];
        PetscScalar   rhs[4], volf[4];
        PetscInt      ii,jj;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // collect phase values for the four edges
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);


	F0 = -(F_u*volf[0] + F_d*(1.0-volf[0])) *gsin;
	F1 = -(F_u*volf[1] + F_d*(1.0-volf[1])) *gsin;
        F2 = -(F_u*volf[2] + F_d*(1.0-volf[2])) *gcos;
        F3 = -(F_u*volf[3] + F_d*(1.0-volf[3])) *gcos;
                

        ii = i - sx;
        jj = j - sz;
	
	PetscScalar PDL0, PDL1, PDL2, PDL3;

	PDL0 = psdl/dx[ii];
	PDL1 = -psdl/dx[ii];
	PDL2 = psdl/dz[jj];
	PDL3 = -psdl/dz[jj];

        // d(chi_s*tau_xy_old)/dz compute on the left and right
        // d(chi_s*tau_xy_old)/dx and gravity compute on the down boundary only
        // zero second derivatives of compaction stresses on the true boundaries
        
        if (i > 0) {
          rhs[0] = pstat/dx[ii];
	  rhs[0]+= 0.5*phif[0]*F0;
          rhs[0]+= -chis*told_xx_e/dx[ii];
          rhs[0]+= -0.5*(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
          rhs[0]+= chip * pold/dx[ii];
	  rhs[0]+= PDL0;
        } else {
          rhs[0] = phif[0]*F0; // dpstat/dx = 0 at the left boundary
          rhs[0]+= -0.5*((cs[3]*told_xx[3]+cs[1]*told_xx[1])-(cs[2]*told_xx[2]+cs[0]*told_xx[0]))/dx[ii];
          rhs[0]+= -(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
	  rhs[0]+= PDL0;
        }
        if (i < Nx-1) {
          rhs[1] = -pstat/dx[ii];
	  rhs[1]+= 0.5*phif[1]*F1;
          rhs[1]+= chis*told_xx_e/dx[ii];
          rhs[1]+= -0.5*(cs[3]*told_xz[3]-cs[1]*told_xz[1])/dz[jj];
          rhs[1]+= -chip*pold/dx[ii];
	  rhs[1]+= PDL1;
        } else {
          rhs[1] = phif[1]*F1;
          rhs[1]+= -0.5*((cs[3]*told_xx[3]+cs[1]*told_xx[1])-(cs[2]*told_xx[2]+cs[0]*told_xx[0]))/dx[ii];
          rhs[1]+= -(cs[3]*told_xz[3]-cs[1]*told_xz[1])/dz[jj];
	  rhs[1]+= PDL1;
        }
        if (j > 0) {
          rhs[2] = 0.5*phif[2]*F2;
          rhs[2]+= -chis*told_zz_e/dz[jj];
          rhs[2]+= -0.5*(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
          rhs[2]+= chip*pold/dz[jj];
	  rhs[2]+= PDL2;
        } else {
          rhs[2] = phif[2]*F2;
          rhs[2]+= -0.5*((cs[3]*told_zz[3]+cs[2]*told_zz[2])-(cs[1]*told_zz[1]+cs[0]*told_zz[0]))/dz[jj];
          rhs[2]+= -(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
	  rhs[2]+= PDL2;
        }
        if (j < Nz-1) {
          rhs[3] = 0.5*phif[3]*F3;
          rhs[3]+= chis*told_zz_e/dz[jj];
          rhs[3]+= -0.5*(cs[3]*told_xz[3]-cs[2]*told_xz[2])/dx[ii];
          rhs[3]+= -chip*pold/dz[jj];
	  rhs[3]+= PDL3;
        } else {
          rhs[3] = phif[3]*F3;
          rhs[3]+= -0.5*((cs[3]*told_zz[3]+cs[2]*told_zz[2])-(cs[1]*told_zz[1]+cs[0]*told_zz[0]))/dz[jj];
          rhs[3]+= -(cs[3]*told_xz[3]-cs[2]*told_xz[2])/dx[ii];
	  rhs[3]+= PDL3;
        }


        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] += rhs[ii];
        }
        
      }

      { // C = 0 (center, c=0)
        DMStagStencil point;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 0;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);
        c[j][i][idx] = 0.0;
      
      }

      { // D1 = zeta - 2/3*A (center, c=2)
        DMStagStencil point;
        PetscInt      idxA;
        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 2;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idx); CHKERRQ(ierr);

        point.i = i; point.j = j; point.loc = ELEMENT;  point.c = 1;
        ierr = DMStagGetLocationSlot(dmcoeff, point.loc, point.c, &idxA); CHKERRQ(ierr);
        c[j][i][idx] = zeta - 2.0*coeff_iso*c[j][i][idxA] ;

      }

      { // D2 = -R^2 * Kphi (edges, c=1)
        DMStagStencil point[4];
        PetscScalar   r2u, r2d, volf[4], kh, kc, zp[4],kee[4];
        PetscInt      ii;
/*
	PetscScalar   kee[4], kex[4], kez[4], krx,krz, keps=0.1, xexp, zexp; //permeability enhancement in x and z direction

	for (ii=0;ii<4;ii++) {
	krx = (rvz[ii] - 0.5)/keps;
	krz = (rvx[ii] - 0.5)/keps;
	xexp = PetscExpScalar(krx);
	zexp = PetscExpScalar(krz);

	kex[ii] = 1 + (3*xexp - 3/xexp)/(xexp + 3/xexp);
	kez[ii] = 1 + (3*zexp - 3/zexp)/(zexp + 3/zexp);
	}

	kee[0] = kex[0];
	kee[1] = kex[1];
	kee[2] = kez[2];
	kee[3] = kez[3];
*/
        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        //-----use the phase field to interpolate the value of R2
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);

        //-----LOAD the face-wise permeability modification
	point[0].c = 0;point[1].c = 0;point[2].c = 0;point[3].c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmPV,xkeelocal,4,point,kee); CHKERRQ(ierr);
        
	r2u = R2a[ifp];
        r2d = R2a[ifn];
        
        zp[0] = coordz[j][icenter];
        zp[1] = coordz[j][icenter];
        zp[2] = coordz[j][iprev];
        zp[3] = coordz[j][inext];

        for (ii = 0; ii < 4; ii++) {

          //depth-dependent ratio
          kh = PetscPowScalar(krange, (zp[ii]-zmin)/(H-Hair));

          //plastic-strain-dependent ratio
          //kh = kh* ( 1 + ck * ( 1- PetscExpScalar(-xstrainf[ii]/strainK)   )  );
          //kc =  (r2u*volf[ii] + r2d*(1-volf[ii]) ) * ( 1- PetscExpScalar(-xstrainf[ii]/strainK)   );
          kc = 0; //kc * PetscPowScalar(phif[ii],2)/PetscPowScalar(phi0,3);


          r2local[ii] = (r2u*volf[ii] + r2d*(1-volf[ii]) )*kh ;


          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          


         if (usr->par->permeability == 0) {
	  //Kozeny-Carman
          //Kphi[ii] = r2local[ii] * PetscPowScalar(phif[ii]/phi0, 3)* PetscPowScalar(1-phif[ii],-2);
          //Kphi[ii] = r2local[ii] * PetscPowScalar(phiK[ii]/phi0, 3)* PetscPowScalar(1-phiK[ii],-2);
          //

	  Kphi[ii] = r2local[ii] * PetscPowScalar(phi0, -3)* Permeability_porosity(phif[ii]); 


	  } else if (usr->par->permeability == 1) {
          // Customised K-phi relation obtained in the poro-fracture analysis
	  //Kphi[ii] = r2local[ii] * PetscPowScalar(phif[ii]/phi0, 3)* PetscPowScalar( usr->par->km, PetscPowScalar(phif[ii], usr->par->kd));
	  //Kphi[ii] = r2local[ii] * PetscPowScalar(phiK[ii]/phi0, 3)* PetscPowScalar( usr->par->km, PetscPowScalar(phiK[ii], usr->par->kd));

         
	  Kphi[ii] = r2local[ii] * PetscPowScalar(phi0, -3)* Permeability_porosity(phif[ii]) * PetscPowScalar( usr->par->km, PetscPowScalar(phif[ii], usr->par->kd));
 ; 

	  } else{
	SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "Permeability type is not set.");
	}


//          Kphi[ii] = Kphi[ii] + r2local[ii] * Kdf[ii] +  kc;

	
	Kphi[ii] = Kphi[ii] * kee[ii];

//	Kphi[ii] = r2local[ii] * kee[ii];

          c[j][i][idx] = - Kphi[ii];

        }
      }



      //PetscPrintf(PETSC_COMM_WORLD, "r2: %g, %g, %g, %g, %g, %g, %g, %g\n", r2local[0], r2local[1], r2local[2],r2local[3], Kphi[0], Kphi[1], Kphi[2], Kphi[3]);

      { // D3 = R^2 * Kphi * (F - dpstat/dx *ex)  (edges, c=2)
        DMStagStencil point[4];
        PetscScalar   dd3[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 2;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 2;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 2;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 2;

        /*
        dd3[0] = 0.0;
        dd3[1] = 0.0;
        dd3[2] = -F2;
        dd3[3] = -F3;
        */

        if (i > 0) {
          dd3[0] = -pstat/dx[ii] - 0.5*F0;
        } else {
          dd3[0] = -F0;  // dp_stat/dx = 0 at the left boundary
        }
        if (i < Nx-1) {
          dd3[1] = pstat/dx[ii] - 0.5*F1;
        } else {
          dd3[1] = -F1; //dp_stat/dx = 0 at the right boundary
        }
        if (j > 0) {
          dd3[2] = -0.5*F2;
        } else {
          dd3[2] = -F2;
        }
        if (j < Nz-1) {
          dd3[3] = -0.5*F3;
        } else {
          dd3[3] = -F3;
        }

        for (ii = 0; ii < 4; ii++) {
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          c[j][i][idx] += Kphi[ii]*dd3[ii];
        }

      }
    }
  }

  // release dx dz
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);  

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dmcoeff,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmcoeff,coefflocal,&c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmcoeff,coefflocal,INSERT_VALUES,coeff); CHKERRQ(ierr);
  ierr = VecDestroy(&coefflocal); CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(usr->dmeps,xtaulocal,&xxs); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xtaulocal,INSERT_VALUES,usr->xtau); CHKERRQ(ierr);
  ierr = VecDestroy(&xtaulocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,xDPlocal,&xxp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xDPlocal,INSERT_VALUES,usr->xDP); CHKERRQ(ierr);
  ierr = VecDestroy(&xDPlocal); CHKERRQ(ierr);
  
  ierr = DMStagVecRestoreArray(usr->dmeps,xyieldlocal,&xxy); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,xyieldlocal,INSERT_VALUES,usr->xyield); CHKERRQ(ierr);
  ierr = VecDestroy(&xyieldlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(usr->dmeps,evplocal,&xxevp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,evplocal,INSERT_VALUES,usr->epsvp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,evplocal,INSERT_VALUES,usr->epsvp); CHKERRQ(ierr);
  ierr = VecDestroy(&evplocal); CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&poldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &flocal);    CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &volflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &fplocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &fnlocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &pplocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,  &nslocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,  &yslocal);   CHKERRQ(ierr);


  ierr = DMRestoreLocalVector(usr->dmPV,&xkeelocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,  &xlocal);   CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dmphi, &xphilocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmphi, &xphi0local); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps, &strainlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps, &Kdlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



// ---------------------------------------
// FormBCList_PV_openside
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV_openside"
PetscErrorCode FormBCList_PV_openside(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc,i,j;
  PetscInt       sx, sz, nx, nz, Nx, Nz, iprev, icenter, inext;
  PetscScalar    *value_bc,*x_bc;
  BCType         *type_bc;
  PetscScalar    **coordx,**coordz;
  Vec            xlocal, toldlocal, paralocal, philocal, xkeelocal;
  PetscScalar    vi, L, H, Hair, val, phim;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;

  vi = usr->par->vi;
  L = usr->par->L;
  H = usr->par->H;
  Hair = usr->par->Hair;

  phim = usr->par->phi_m;


  PetscScalar xmin, zmin, dpfs, dl, xpeak, xsig;
  dl = usr->par->dl * L;
  dpfs = usr->par->dpfs;
  xmin = usr->par->xmin;
  zmin = usr->par->zmin;

  xpeak = usr->par->xpeak;
  xsig = usr->par->xsig;


// Get solution dm/vector
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);

  // Map global vectors to local domain
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &toldlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xtau_old, INSERT_VALUES, toldlocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmeps, &paralocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xyield, INSERT_VALUES, paralocal); CHKERRQ(ierr);
  
  ierr = DMGetLocalVector(usr->dmphi, &philocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi, usr->xphiprev, INSERT_VALUES, philocal); CHKERRQ(ierr);


  // Get the face-wise permeability modification
  ierr = DMGetLocalVector(usr->dmPV, &xkeelocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xkee, INSERT_VALUES, xkeelocal); CHKERRQ(ierr);
  // calculate delta at the current time step
/*
     DMStagStencil  point;
     PetscScalar    phis, phif, delta, sdel;

     sdel = usr->par->sdel;

     point.i = (PetscInt)((Nx-1)/2); point.j = 0;   point.loc = ELEMENT; point.c = 0;
     ierr = DMStagVecGetValuesStencil(usr->dmphi, philocal,1,&point,&phis); CHKERRQ(ierr);

     phif = 1 - phis;

     delta = sdel * PetscPowScalar(phif/phim, -3)*  PetscPowScalar((1-phif)/(1-phim), 2);
*/

     DMStagStencil  point;
     PetscScalar    phis, phif, delta, Kini, dini,kee, Know=0;

     Kini = usr->par->Kini;
     dini = usr->par->dini;

     for (i=0; i<Nx; i++) {

	 point.i = i; point.j = 0;   point.loc = ELEMENT; point.c = 0;
         ierr = DMStagVecGetValuesStencil(usr->dmphi, philocal,1,&point,&phis); CHKERRQ(ierr);
        
	point.loc = DOWN; 
	ierr = DMStagVecGetValuesStencil(usr->dmPV, xkeelocal,1,&point,&kee); CHKERRQ(ierr);

	if (!usr->par->plasticity) {kee = 1;}

         phif = 1 - phis;

	 Know += Permeability_porosity(phif) * kee;
     }

     delta = dini*Kini/Know;   //dini * Know/Kini;

     usr->par->sdel = delta;


//  PetscScalar sdel;
//  sdel = usr->par->sdel;

  
  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {

   for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][icenter]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, txx;
        point.i = 0; point.j = j;   point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&txx); CHKERRQ(ierr);

        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&eta); CHKERRQ(ierr);

        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&chis); CHKERRQ(ierr);



        value_bc[k] = 0.5/eta  * ( xx- chis * txx)  ;
        type_bc[k] = BC_NEUMANN_T;
      }
    }

//     value_bc[k] = -vi;
//    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {

for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][inext]) {
        DMStagStencil  point, pv[2];
        PetscScalar    eta, chis, txz, xv[2],dz;

        pv[0].i = 0; pv[0].j = j;   pv[0].loc = LEFT; pv[0].c = 0;
        pv[1].i = 0; pv[1].j = j+1;   pv[1].loc = LEFT; pv[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pv,xv); CHKERRQ(ierr);

        dz = coordz[j+1][icenter] - coordz[j][icenter];
        value_bc[k] =  -(xv[1]-xv[0])/dz ;
        type_bc[k] = BC_NEUMANN;
      }
    }

//    value_bc[k] = 0.0;
//    type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);


  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
/*
   for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][icenter]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, txx;
        point.i = Nx-1; point.j = j;   point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&txx); CHKERRQ(ierr);

        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&eta); CHKERRQ(ierr);

        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&chis); CHKERRQ(ierr);



        value_bc[k] =0.5/eta  *( xx- chis * txx)  ;
        type_bc[k] = BC_NEUMANN_T;
      }
    }
*/
      
   for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][icenter]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, txx;
        point.i = 0; point.j = j;   point.loc = LEFT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);


        value_bc[k] = -xx  ;
        type_bc[k] = BC_DIRICHLET;
      }
    }

//    value_bc[k] = vi;
//    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
  
       	  for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][inext]) {
        DMStagStencil  point, pv[2];
        PetscScalar    eta, chis, txz, xv[2],dz;

        pv[0].i = Nx-1; pv[0].j = j;   pv[0].loc = RIGHT; pv[0].c = 0;
        pv[1].i = Nx-1; pv[1].j = j+1;   pv[1].loc = RIGHT; pv[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pv,xv); CHKERRQ(ierr);

        dz = coordz[j+1][icenter] - coordz[j][icenter];
        value_bc[k] =  -(xv[1]-xv[0])/dz ;
        type_bc[k] = BC_NEUMANN;
      }
    }


/*
   for (j = sz; j < sz+nz; j++) {
      if (x_bc[2*k+1]==coordz[j][inext]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, txx;
        point.i = 0; point.j = j;   point.loc = UP; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);


        value_bc[k] = xx  ;
        type_bc[k] = BC_DIRICHLET;
      }
    }
*/

    //value_bc[k] = 0.0;
    //type_bc[k] = BC_DIRICHLET_STAG;//BC_NEUMANN;//
  }

  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);


  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
/*
if (k != (PetscInt)((n_bc-1)/2)) {
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][inext]) {
        DMStagStencil  pv[2];
        PetscScalar    xv[2],dx;

        pv[0].i = i; pv[0].j = 0;   pv[0].loc = DOWN; pv[0].c = 0;
        pv[1].i = i+1; pv[1].j = 0;   pv[1].loc = DOWN; pv[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pv,xv); CHKERRQ(ierr);


        dx = coordx[i+1][icenter] - coordx[i][icenter];
        value_bc[k] =  0.0;//-(xv[1]-xv[0])/dx;
        type_bc[k] = BC_NEUMANN;
      }
    }
} else {
    value_bc[k] = 0.0;  //vi/H;// vi * (x_bc[2*k+1]-zmin)/H;  //0.0; //-vi;
    type_bc[k] = BC_DIRICHLET;//BC_NEUMANN; //BC_DIRICHLET;//_STAG;
} */

	value_bc[k] = 0.0;
	type_bc[k] = BC_NEUMANN;
 
}
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
/*
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, tzz;
        point.i = i; point.j = 0;   point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);

        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&tzz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&eta); CHKERRQ(ierr);

        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&chis); CHKERRQ(ierr);


        value_bc[k] = 0.5/eta  * (xx - chis * tzz)  ;
        type_bc[k] = BC_NEUMANN_T;
      }

    }
*/
	  value_bc[k] = 0;
	  type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
// DOWN Boundary - P

  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
/*
  	  for (i = sx; i < sx+nx; i++) {
      		if (x_bc[2*k]==coordx[i][icenter] && PetscAbs(x_bc[2*k]-xpeak)<2*xsig ) {
      		value_bc[k] = usr->par->p0;
      		type_bc[k] = BC_DIRICHLET;
		}
	}
*/	
   //value_bc[k] = usr->par->p0;
  // type_bc[k] = BC_DIRICHLET;//_STAG;
//
/*
     DMStagStencil  point;
     PetscScalar    phis, phif, delta;

     point.i = i; point.j = 0;   point.loc = ELEMENT; point.c = 0;
     ierr = DMStagVecGetValuesStencil(usr->dmphi, philocal,1,&point,&phis); CHKERRQ(ierr);

     phif = 1 - phis;

     delta = sdel * PetscPowScalar(phif/phim, -3)*  PetscPowScalar((1-phif)/(1-phim), 2);
*/
		
   //value_bc[k] = (1 - usr->par->sdel) * usr->par->F1;   
if (k==(PetscInt)((Nx-1)/2)) {
   value_bc[k] = (1 - delta) * usr->par->F1;  // delta is modified to make sure a constant influx
} else {value_bc[k] = usr->par->F1;}
   type_bc[k] = BC_NEUMANN;//BC_DIRICHLET;//_STAG;

  }
  ierr = DMStagBCListInsertValues(bclist,'o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vx  - zero shear stress
  ierr = DMStagBCListGetValues(bclist,'n','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {

     for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][inext]) {
        DMStagStencil  point, pv[2];
        PetscScalar    eta, chis, txz, xv[2],dx;

        pv[0].i = i; pv[0].j = Nz-1;   pv[0].loc = UP; pv[0].c = 0;
        pv[1].i = i+1; pv[1].j = Nz-1;   pv[1].loc = UP; pv[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pv,xv); CHKERRQ(ierr);

        dx = coordx[i+1][icenter] - coordx[i][icenter];
        value_bc[k] =  -(xv[1]-xv[0])/dx ;
        type_bc[k] = BC_NEUMANN;
      }
    }


//if (k != (PetscInt)(n_bc/2 - 1)) {
/*
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][inext]) {
        DMStagStencil  pv[2];
        PetscScalar    xv[2],dx;

        pv[0].i = i; pv[0].j = 0;   pv[0].loc = DOWN; pv[0].c = 0;
        pv[1].i = i+1; pv[1].j = 0;   pv[1].loc = DOWN; pv[1].c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,2,pv,xv); CHKERRQ(ierr);


        dx = coordx[i+1][icenter] - coordx[i][icenter];
*/ 
/* 
      value_bc[k] =  0;//-(xv[1]-xv[0])/dx;
        type_bc[k] = BC_NEUMANN;
  //    }
  //  }
} ielse {

	
    for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][inext]) {
        DMStagStencil  pv;
        PetscScalar    xv,dx;

        pv.i = i; pv.j = Nz-1;   pv.loc = RIGHT; pv.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&pv,&xv); CHKERRQ(ierr);


    value_bc[k] = -xv;  
    type_bc[k] = BC_DIRICHLET;
}
}}*/ 
// value_bc[k] = 0;//vi/H;//vi * (x_bc[2*k+1]-zmin)/H;  //vi;
// type_bc[k] = BC_NEUMANN;//BC_DIRICHLET;
  }
  
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // UP Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'n','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {

   for (i = sx; i < sx+nx; i++) {
      if (x_bc[2*k]==coordx[i][icenter]) {
        DMStagStencil  point;
        PetscScalar    xx, eta, chis, tzz;
        point.i = i; point.j = Nz-1;   point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(dm,xlocal,1,&point,&xx); CHKERRQ(ierr);

        point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&tzz); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&eta); CHKERRQ(ierr);

        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&chis); CHKERRQ(ierr);


	PetscScalar dpp, xp;

	xp = x_bc[2*k];

	if ( PetscAbs(xp -xmin - 0.475*L) <= 0.5*dl) {dpp = dpfs;}
	else  {dpp = 0;}


        value_bc[k] =0.5/eta  * ( dpp +   xx- chis * tzz)  ;
        type_bc[k] = BC_NEUMANN_T;
      }
    }


//     value_bc[k] = 0;
//     type_bc[k] = BC_DIRICHLET;

//    value_bc[k] =  -2.0*vi*Hair/L;//0.0;//xx*1e6/2;
//    type_bc[k] = BC_DIRICHLET;//
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&paralocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi,&philocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPV,&xkeelocal); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}



// ---------------------------------------
// SetInitialField
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialField"
PetscErrorCode SetInitialField(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xlocal, fplocal, fnlocal;
  PetscInt       i,j, sx, sz, nx, nz, icenter;
  PetscScalar    eps;
  PetscScalar    ***xx, **coordx, **coordz;
  PetscScalar    ***fp, ***fn;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // some useful parameters
  eps = usr->par->eps;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector for the phase field, fp (fluid type at n+) and fn (fluid type at n-)
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &fplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, fplocal, &fp); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &fnlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, fnlocal, &fn); CHKERRQ(ierr);

 
 

  // Loop over local domain - set initial density and viscosity
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp, zp, fval = 0.0, xn=0.0, z1, z2;
      PetscInt      idx;


      if (usr->par->air == 0){

      z1 = 1e50;//usr->par->Hair;  // free surface
      z2 = -1e50;  // interface between high viscous and low viscous layer

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);

      if (zp > (z1+z2)/2) {
        xn =   z1 - zp; //z1 needs to be replaced with a curve describing the free surface.
        fp[j][i][idx] = 1.01;
        fn[j][i][idx] = 3.01;}
      else {
        xn = zp - z2;
        fp[j][i][idx] = 1.01;
        fn[j][i][idx] = 2.01;
      }
          
      fval = 0.5*(1 + PetscTanhScalar(xn/2.0/eps));
      }

      if (usr->par->air ==1) {

//	PetscScalar xct, zct, rr, xmin, zmin, L, H, rd;
/*
	xmin = usr->par->xmin;
	zmin = usr->par->zmin;
	H = usr->par->H;
	L = usr->par->L;
	xct = xmin + 0.5*L;    //x the center of domain
	zct = zmin + 0.5*H;    //z the center of domain
	rd = usr->par->rd;     //the radius of domain
	
	PetscScalar xp, zp, xp2, zp2;

        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        xp = coordx[i][icenter];
	zp = coordz[j][icenter];

	xp2 = pow((xp-xct),2);
	zp2 = pow((zp-zct),2);
	rr = pow(xp2+zp2, 0.5);

        ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);

	xn = rd - rr;
        fp[j][i][idx] = 1.01;
        fn[j][i][idx] = 2.01;
  */

	PetscScalar xmin, zmin, H, L, dl, hpeak;
        xmin = usr->par->xmin;
	zmin = usr->par->zmin;
	H = usr->par->H;
	L = usr->par->L;
	dl = usr->par->dl * L;
	hpeak = usr->par->hpeak * H;

      z1 = H - usr->par->Hair;  // free surface
      z2 = -1e50;  // interface between high viscous and low viscous layer

      

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      xp = coordx[i][icenter];
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);


      if (zp > (z1+z2)/2) {

// Gaussian distribution z = zmin + z1 + hpeak * exp( - ((x-xmin)/dl)^2)
// for simplicity, xn = z(xp) - zp;

	PetscScalar zg;

	zg = zmin + z1 + hpeak * PetscExpScalar(-pow((xp-xmin)/dl, 2));
	xn = zg - zp;



// triangular top loading
/*    
	//the initial free surface is prescribed as:
	// z = zmin + z1 + hpeak * (1 - (x-xmin)/dl), if x-xmin <= dl
	// z = zmin + z1,    if x-xmin > dl
	//
	// We find xn by the following rules:
	// Draw a straight line perpendicular to the triangular part of free surface which passing (xmin+dl, zmin+z1);
	// The straight line equation is z = k*x + c, k = dl/hpeak, c = (zmin+z1) - k * (xmin+dl)
	// If a point (xp, zp) is located on the left of this straight line, xn = ([hpeak - (xp-xmin)*hpeak/dl] - zp)/sqrt(k^2+1)
	// if this point is located on the right of this straight line, but xp < xmin+dl 
	// xn = sign(zmin+z1 - zp) * sqrt( (xp - (xmin+dl)  )**2 + (zp - (zmin+z1))**2)
	// if xp >= xmin+dl, xn = zmin+z1 - zp
	//
	
	PetscScalar x0, z0, xd, zd, kt, ct, kl, cl, zpt, zpl;

	x0 = xmin + dl;
	z0 = zmin + z1;
	xd = xp - x0;
	zd = zp - z0;
	kt = - hpeak/dl;
	ct = z0 + hpeak;
	kl = dl/hpeak;
	cl = z0 - kl*x0;


	if (xp < x0) {

	zpl = kl * xp + cl;

	if (zp > zpl) { zpt = kt*(xp-xmin)+ct; xn = (zpt - zp)/pow(kt*kt+1, 0.5);}
	else  {
		PetscInt sign;
		sign = PetscSign(z0-zp);

	}
	else {xn = z0 - zp;}  
*/	

 
        fp[j][i][idx] = 1.01;
        fn[j][i][idx] = 3.01;}
      else {
        xn = zp - z2;
        fp[j][i][idx] = 1.01;
        fn[j][i][idx] = 2.01;
      }
	
        
        fval = 0.5*(1 + PetscTanhScalar(xn/2.0/eps));

      }

      xx[j][i][idx] = fval;


    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,fplocal,&fp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,fplocal,INSERT_VALUES,usr->fp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,fplocal,INSERT_VALUES,usr->fp); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,fnlocal,&fn);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,fnlocal,INSERT_VALUES,usr->fn); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,fnlocal,INSERT_VALUES,usr->fn); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&fplocal); CHKERRQ(ierr);
  ierr = VecDestroy(&fnlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


// ---------------------------------------
// SetInitialPorosityProfile
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "SetInitialPorosityProfile"
PetscErrorCode SetInitialPorosityProfile(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec           xlocal;
  PetscInt      i,j, sx, sz, nx, nz, icenter;
  PetscScalar   phi_m, phi_bg, xsig0, zsig0, xpeak, zpeak;
  PetscScalar   ***xx, **coordx, **coordz;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Parameters
  phi_m  = usr->par->phi_m;
  phi_bg = usr->par->phi_bg;
  xsig0   = usr->par->xsig;
  zsig0   = usr->par->zsig;
  xpeak  = usr->par->xpeak;
  zpeak  = usr->par->zpeak;



  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);


  PetscScalar Kini = 0; // initial scaled permeability Kini=phi^3/(1-phi)^2 


  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   xp,zp, val, xsig, zsig;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;  

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);

      xp = coordx[i][icenter];
      zp = coordz[j][icenter];
      xsig = xsig0;
      zsig = zsig0;


      PetscScalar rr;
      rr = pow(pow(xp-xpeak,2) + pow((zp-zpeak)*xsig/zsig, 2) ,0.5);

      val = phi_bg + (phi_m-phi_bg) * PetscExpScalar( - pow(rr/xsig, 2) );

      if (usr->par->gaussian==0) {
	if (rr < 2*xsig) {val = phi_m;}
	else 		  {val = phi_bg;}
	}	

      if (usr->par->gaussian==0 && (usr->par->bcidx==0 || usr->par->bcidx==4) ) {
	if ( PetscAbs(xp-xpeak) < xsig && zpeak-zp >= 0) { val= phi_m;}
	else	{val = phi_bg;}
	}

      if (usr->par->gaussian==1 && usr->par->bcidx==4) {

	if (zpeak-zp > 0) { 
	rr = pow(pow((xp-xpeak)/xsig,2) ,0.5);
	val=phi_bg + (phi_m-phi_bg)*PetscExpScalar(- pow(rr, 2));
	} else {
      	rr = pow(pow((xp-xpeak)/xsig,2) + pow((zp-zpeak)/zsig, 2) ,0.5);
	val=phi_bg + (phi_m-phi_bg)*PetscExpScalar(- pow(rr, 2));
	}

	}	
     
      if (usr->par->bcidx ==2 || usr->par->bcidx==6) {
	if (rr < 2*xsig) {val = phi_m;}
      }
     
      if (usr->par->bcidx ==3 ) {
	PetscScalar xa, va;
	xa = xpeak + (zp - zpeak);
	va = phi_bg + (phi_m-phi_bg) * PetscExpScalar(-pow((zp-zpeak)/zsig,2));
	val = va * PetscExpScalar(-pow((xp-xa)/xsig,2));
      }

      //if (usr->par->bcidx ==7 || usr->par->bcidx==8 || usr->par->bcidx==9 || usr->par->bcidx==10 ) {
      if (usr->par->bcidx ==7 || usr->par->bcidx==8 ) {
	
      	//rr = pow(pow(xp-xpeak,2) + pow((zp-zpeak)*xsig/zsig, 2) ,0.5);


	PetscScalar xd, yd, d1, d2,lcos, lsin;
	xd = xp-xpeak;
	yd = zp-zpeak;
	lcos = PetscCosScalar(usr->par->gth);
	lsin = PetscSinScalar(usr->par->gth);
	d1 = PetscAbsScalar(xd*lcos - yd*lsin);
	d2 = PetscAbsScalar(xd*lsin + yd*lcos);

	rr = pow( pow(d1,2) + pow(d2*xsig/zsig  ,2), 0.5);


      	val = phi_bg + (phi_m-phi_bg) * PetscExpScalar( - pow(rr/(2*xsig), 4) );
      }
      
      /*
      PetscScalar x0;
      x0 = xpeak + (zp - zpeak);
      val = phi_bg + (phi_m - phi_bg) * PetscExpScalar(-pow((zp-zpeak)/zsig, 2 )) * PetscExpScalar( -pow((xp-x0)/xsig,2));  
*/
     // if (rr < 2*xsig) {val = phi_m;}
      //else           {val = phi_bg + (phi_m-phi_bg)*PetscExpScalar( - pow((rr-xsig)/0.05,2) );}
     // else           {val = phi_bg;}



      xx[j][i][idx] = 1 - val;

     if(j==0) { Kini += Permeability_porosity(val);}
    }
  }

  // Calculate the initial delta= Q/Sum(M)Dx/DRhoG = q/Sum(M)/DRhoG 
  // temporary solution, need to do it properly
  PetscScalar dini;
  dini = usr->par->qin/3600/24/365 / (Kini * usr->par->K0) / usr->par->drhog;
   
  //dini = dini / usr->par->kscale;


  // Insert the initial permeability into para
  usr->par->Kini = Kini; 
  usr->par->dini = dini;


  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


// ---------------------------------------
// IntegratePlasticStrain
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "IntegratePlasticStrain"
PetscErrorCode IntegratePlasticStrain(DM dm, Vec lam, Vec dlam, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  Vec            lamlocal, dlamlocal, devplocal, evplocal, evpolocal, evpo, evp, devp, dp, dplocal;
  PetscScalar    dt;
  PetscInt       i,j, sx, sz, nx, nz,Nx,Nz;
  PetscScalar    ***sp,***ep ;
  PetscErrorCode ierr;

  dt = usr->par->dt;

  evpo = usr->evpo;
  evp = usr->evp;
  devp = usr->epsvp;
  dp = usr->xDP;

  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  // Create local vector
  ierr = DMCreateLocalVector(dm, &lamlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,lam,INSERT_VALUES,lamlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,lam,INSERT_VALUES,lamlocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, lamlocal, &sp); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dlamlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, dlam, INSERT_VALUES, dlamlocal); CHKERRQ(ierr);


  // Create local vector for compoennets of plastic strain and strain rates, and compaction stress
  ierr = DMCreateLocalVector(dm, &evplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,evp,INSERT_VALUES,evplocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,evp,INSERT_VALUES,evplocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, evplocal, &ep); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &devplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, devp, INSERT_VALUES, devplocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &evpolocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, evpo, INSERT_VALUES, evpolocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, dp, INSERT_VALUES, dplocal); CHKERRQ(ierr);
  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);


  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[5];
      PetscInt      idx, ii;
      PetscScalar   val, dlamcell[5];


      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = DOWN_LEFT;  point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN_RIGHT; point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP_LEFT;    point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = UP_RIGHT;   point[4].c = 0;

      ierr = DMStagVecGetValuesStencil(dm,dlamlocal,5,point,dlamcell); CHKERRQ(ierr);

      for (ii=0; ii<2; ii++) {

         ierr = DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx); CHKERRQ(ierr);

         val = sp[j][i][idx] + dlamcell[ii] * dt;

         sp[j][i][idx] = val;

      }

      if (i==Nx-1) {
	ii=2; 
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx); CHKERRQ(ierr);
        val = sp[j][i][idx] + dlamcell[ii] * dt;
        sp[j][i][idx] = val;
	}

      if (j==Nz-1) {
	ii=3; 
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx); CHKERRQ(ierr);
        val = sp[j][i][idx] + dlamcell[ii] * dt;
        sp[j][i][idx] = val;
	}

      if (i==Nx-1 && j==Nz-1) {
	ii=4; 
        ierr = DMStagGetLocationSlot(dm, point[ii].loc, 0, &idx); CHKERRQ(ierr);
        val = sp[j][i][idx] + dlamcell[ii] * dt;
        sp[j][i][idx] = val;
	}
    

      //integrate to obtain plastic strain on ELEMENT, c = 0, 1, 2 for evpxx, evpzz and evpxz

	PetscScalar dpcell;
       ierr = DMStagVecGetValuesStencil(dm,dplocal,1,&point[0],&dpcell); CHKERRQ(ierr);
      for (ii=0; ii<3; ii++) {

	PetscScalar devpcell, evpocell;
         point[0].c = ii;
         ierr = DMStagVecGetValuesStencil(dm,evpolocal,1,&point[0],&evpocell); CHKERRQ(ierr);
         ierr = DMStagVecGetValuesStencil(dm,devplocal,1,&point[0],&devpcell); CHKERRQ(ierr);
         ierr = DMStagGetLocationSlot(dm, point[0].loc, ii, &idx); CHKERRQ(ierr);

         val = evpocell + devpcell * dt;
/*  Disable the control of dp
	// do the integration only when DeltaP<0
	 if (dpcell < 0) {
         val = evpocell + devpcell * dt;
	} else {val = evpocell;}
*/

         ep[j][i][idx] = val;
	}


	}
  }



  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dm,lamlocal,&sp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,lamlocal,INSERT_VALUES,lam); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,lamlocal,INSERT_VALUES,lam); CHKERRQ(ierr);
  ierr = VecDestroy(&lamlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,evplocal,&ep);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,evplocal,INSERT_VALUES,evp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,evplocal,INSERT_VALUES,evp); CHKERRQ(ierr);
  ierr = VecDestroy(&evplocal); CHKERRQ(ierr);
  
  ierr = DMRestoreLocalVector(dm, &dlamlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &evpolocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &devplocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &dplocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);


}

// ---------------------------------------
// Calculate damage-enhanced permeability
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "DamageEnhancedPermeability"
PetscErrorCode DamageEnhancedPermeability(DM dm, Vec Kd, Vec lam, Vec phi, void *ctx)
{
  UsrData        *usr = (UsrData*) ctx;
  Vec            Kdlocal, lamlocal, philocal;
  DMStagStencil  point;
  PetscInt       i,j, sx, sz, nx, nz,Nx,Nz, idx;
  PetscScalar    km, lamm;
  PetscScalar    ***kdarray;
  PetscErrorCode ierr;


  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm, &Kdlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Kd,INSERT_VALUES,Kdlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Kd,INSERT_VALUES,Kdlocal);CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, Kdlocal, &kdarray); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &lamlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, lam, INSERT_VALUES, lamlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(usr->dmphi, &philocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmphi, phi, INSERT_VALUES, philocal); CHKERRQ(ierr);
  
// Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  //load the parameters
  km = usr->par->ck;
  lamm = usr->par->strainK;

  // set the stencil.
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      PetscScalar   val, lamcell, phicell;

      point.i = i; point.j = j; point.loc = ELEMENT;    point.c = 0;

      ierr = DMStagVecGetValuesStencil(dm,lamlocal,1,&point,&lamcell); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(usr->dmphi,philocal,1,&point,&phicell); CHKERRQ(ierr);

      // phicell is the solid phase fraction originally, correct it
      phicell = 1 - phicell;

      // limit phicell in the range (phi_bg, 0.9], 0.9 is a random choice, change it later for consistency.
      if (phicell < usr->par->phi_bg) {phicell = usr->par->phi_bg;}
      if (phicell >0.9) {phicell = 0.9;}

      kdarray[j][i][idx] = km * ( 1 - PetscExpScalar(-lamcell/lamm )) * PetscPowScalar(phicell,3) * PetscPowScalar(1-phicell,-2);  

    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dm,Kdlocal,&kdarray);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,Kdlocal,INSERT_VALUES,Kd); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,Kdlocal,INSERT_VALUES,Kd); CHKERRQ(ierr);
  ierr = VecDestroy(&Kdlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &lamlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi, &philocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);


}

// ---------------------------------------
// UpdateStrainRates
// ---------------------------------------
PetscErrorCode UpdateStrainRates(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  DM             dmeps;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, ii;
  PetscInt       iprev, inext, icenter;
  PetscScalar    ***xx;
  PetscScalar    **coordx,**coordz;
  Vec            xeps, xepslocal,xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  dmeps = usr->dmeps;
  xeps  = usr->xeps;

  // Local vectors
  ierr = DMCreateLocalVector (dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, pointN[4];
      PetscScalar    epsIIc, exxc, ezzc, exzc, epsIIn[4], exxn[4], ezzn[4], exzn[4];

      // Strain rates: center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);

      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIc;

      // Strain rates: corner
      pointN[0].i = i; pointN[0].j = j; pointN[0].loc = DOWN_LEFT;  pointN[0].c = 0;
      pointN[1].i = i; pointN[1].j = j; pointN[1].loc = DOWN_RIGHT; pointN[1].c = 0;
      pointN[2].i = i; pointN[2].j = j; pointN[2].loc = UP_LEFT;    pointN[2].c = 0;
      pointN[3].i = i; pointN[3].j = j; pointN[3].loc = UP_RIGHT;   pointN[3].c = 0;
      ierr = DMStagGetPointStrainRates(dm,xlocal,4,pointN,epsIIn,exxn,ezzn,exzn); CHKERRQ(ierr);


      if (i==0) { // boundaries, down left
        ezzn[0] = ezzc;
        exxn[0] = exxc;
	exzn[0] = 0;//exzn[1];//0;//
      }

      if (i==Nx-1) { // boundaries, down right
        ezzn[1] = ezzc;
        exxn[1] = exxc;
	exzn[1] = 0; //exzn[0];//0;//
      }

      if (j==0) { // boundaries, down left
        exxn[0] = exxc;
        ezzn[0] = ezzc;
	exzn[0] = 0; //exzn[2];//0; //exzn[2];
      }

      if (j==Nz-1) { // boundaries, up left
        exxn[2] = exxc;
        ezzn[2] = ezzc;
	exzn[2] = 0; //exzn[0];//0; //
      }

      if ((i==Nx-1) && (j==Nz-1)) { // corner, up right
        exxn[3] = exxc;
        ezzn[3] = ezzc;
	exzn[3] = 0; //exzn[0];//0; //exzn[0];
      }


      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }

	// interpolate exzc, and calculate epsIIc
	exzc = 0.25*(exzn[0] + exzn[1] + exzn[2] + exzn[3]);
	epsIIc =PetscPowScalar(0.5*(exxc*exxc + ezzc*ezzc + 2.0*exzc*exzc),0.5);

      }


      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[0];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[0];

      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[1];
      ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[1];

      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[2];
      ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[2];

      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,0,&idx); CHKERRQ(ierr); xx[j][i][idx] = exxn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,1,&idx); CHKERRQ(ierr); xx[j][i][idx] = ezzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzn[3];
      ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIn[3];


      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xx[j][i][idx] = exzc;
      ierr = DMStagGetLocationSlot(dmeps,ELEMENT,3,&idx); CHKERRQ(ierr); xx[j][i][idx] = epsIIc;


    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps,xepslocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,xepslocal,INSERT_VALUES,xeps); CHKERRQ(ierr);
  ierr = VecDestroy(&xepslocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}



// ---------------------------------------
// MAIN
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "main"
int main (int argc,char **argv)
{
  UsrData         *usr;
  PetscLogDouble  start_time, end_time;
  PetscErrorCode  ierr;
    
  // Initialize application
  ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

  // Start time
  ierr = PetscTime(&start_time); CHKERRQ(ierr);
 
  // Load command line or input file if required
  ierr = PetscOptionsInsert(PETSC_NULL,&argc,&argv,NULL); CHKERRQ(ierr);

  // Input user parameters and print
  ierr = InputParameters(&usr); CHKERRQ(ierr);

  // Print user parameters
  ierr = InputPrintData(usr); CHKERRQ(ierr);

  // Numerical solution using the FD pde object
  ierr = StokesDarcy_Numerical(usr); CHKERRQ(ierr);

  // Destroy objects
  ierr = PetscBagDestroy(&usr->bag); CHKERRQ(ierr);
  ierr = PetscFree(usr);             CHKERRQ(ierr);

  // End time
  ierr = PetscTime(&end_time); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"# Total runtime: %g (sec) \n", end_time - start_time);
  PetscPrintf(PETSC_COMM_WORLD,"# --------------------------------------- #\n");
  
  
  // Finalize main
  ierr = PetscFinalize();
  return ierr;
}
