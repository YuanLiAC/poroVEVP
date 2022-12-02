// -------petsc----------------------
// Simulating the propagation of dyke in an extensional two-phase block with a uniform viscosity.
// StokesDarcy2field with a VEVP rheology and a hyperbolic yield surface
// Computational domain: extensional block with an initial melt blob along bottom
// ---------------------------------------
static char help[] = "Application to simulate the dyke propagation in an extensional two-phase block \n\n";

#include "poro-vevp.h"

// ---------------------------------------
// Application Context
// ---------------------------------------
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
"  UP: zero normal and tangential stress \n"
"  RIGHT: Vx=vi, dVz/dx=0\n"
"  BOTTOM: dVx/dz=0, Vz=0, p=0 \n";

const char coeff_description_phi[] =
"  << Porosity Coefficients (dimensionless) >> \n"
"  A = 1.0 \n"
"  B = 0 \n"
"  C = 0 \n"
"  u = [ux, uz] - StokesDarcy solid velocity \n";

const char bc_description_phi[] =
"  << Porosity BCs >> \n"
"  LEFT, RIGHT, DOWN, UP: Zero flux \n";



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
  Vec            xPV, xcoeff, xguess, xphi, xphiprev, phicoeff, phicoeffprev;
  PetscInt       nx, nz, istep = 0, tstep, ickpt = 0, maxckpt;
  PetscScalar    xmin, zmin, xmax, zmax, dtck, tckpt, tmax;
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
  
  PetscPrintf(usr->comm, "check point at the beginning\n");

  // 1. Stokes-Darcy: Create the FDPDE object, set the function and boundary conditions
  ierr = FDPDECreate(usr->comm,nx,nz,xmin,xmax,zmin,zmax,FDPDE_STOKESDARCY2FIELD,&fdPV);CHKERRQ(ierr);
  ierr = FDPDESetUp(fdPV);CHKERRQ(ierr);
  ierr = FDPDESetFunctionCoefficient(fdPV,FormCoefficient_PV,coeff_description_sd,usr); CHKERRQ(ierr);
  ierr = FDPDESetFunctionBCList(fdPV,FormBCList_PV,bc_description_sd,usr); CHKERRQ(ierr);
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

  // -- Create a vec to store the magnitude of velocity at every cell center
  //    use the same DM with fp and fn for the conveience to know VMag in different fluids.
  ierr = DMCreateGlobalVector(usr->dmPV, &usr->VMag); CHKERRQ(ierr);
  
  // -- Create a vector to store PV and phi in userdata
  ierr = FDPDEGetSolution(fdPV,&xPV);CHKERRQ(ierr);
  ierr = FDPDEGetSolution(fdphi,&xphi);CHKERRQ(ierr);
  ierr = VecDuplicate(xPV, &usr->xPV);CHKERRQ(ierr);
  ierr = VecDuplicate(xphi, &usr->xphiprev);CHKERRQ(ierr);
  ierr = VecDestroy(&xPV);CHKERRQ(ierr);
  ierr = VecDestroy(&xphi);CHKERRQ(ierr);


  // -- Create DM/vec for tau_old and DP_old, Initialise the two Vecs as zeros.
  ierr = VecZeroEntries(usr->xtau_old); CHKERRQ(ierr);
  ierr = VecZeroEntries(usr->xDP_old); CHKERRQ(ierr);

  // Initialise the phase field
  ierr = SetInitialField(usr->dmf,usr->f,usr);CHKERRQ(ierr);

  // Set initial porosity profile (t=0)
  ierr = FDPDEAdvDiffGetPrevSolution(fdphi,&xphiprev);CHKERRQ(ierr);
  ierr = SetInitialPorosityProfile(usr->dmphi,xphiprev,usr);CHKERRQ(ierr);
  ierr = VecCopy(xphiprev,usr->xphiprev);CHKERRQ(ierr);
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


  // output - initial state of the phase field
  ierr = PetscSNPrintf(fout,sizeof(fout),"%s_phase_initial",usr->par->fname_out);
  ierr = DMStagViewBinaryPython(usr->dmf,usr->f,fout);CHKERRQ(ierr);
  
  // FD SNES Solver
  PetscPrintf(usr->comm,"\n# SNES SOLVE #\n");
  
  // Time loop  
  while ((ickpt <= maxckpt) && (istep<tstep) && (usr->par->t<=tmax))  {

  jump:
    // Update time
    usr->par->t += usr->par->dt;  // computation start from t = dt

    // set time step size for ADVDIFF
    ierr = FDPDEAdvDiffSetTimestep(fdphi,usr->par->dt);CHKERRQ(ierr);

    PetscPrintf(usr->comm,"# TIME CHECK POINT %d out of %d after %d steps: time %1.4f\n\n",ickpt, maxckpt, istep, usr->par->t);
    PetscPrintf(usr->comm,"# next check piont: %1.4f; distance between check points: %1.4f\n\n", tckpt, dtck);
    
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
/* // temporarily disable auto refinement of time steps
      if (!converged) {

        //reset the time
          usr->par->t -= usr->par->dt;
          //change the time step size
          usr->par->dt *= 0.2;

          //block writing in of the solution at this refined step
          iwrt = PETSC_FALSE;

           //if the checkpoint is advanced at this step, reset the checkpoint
          if ((tckpt - dtck) > (usr->par->t + usr->par->dt )) {tckpt -= dtck; ickpt--;}

          PetscPrintf(PETSC_COMM_WORLD, "fdPV not converge, taking a time step size 5 times smaller, dt = %g \n", usr->par->dt);

          // return to the jump point
          goto jump;

      }
      */
    }


    //StokesDarcy Solver
    //ierr = FDPDESolve(fdPV,NULL);CHKERRQ(ierr);
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

    
    // filtering out phi<0 and phi>1
    ierr = PorosityFilter(usr->dmphi, xphi, usr); CHKERRQ(ierr);

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

    
    PetscPrintf(usr->comm,"# TIME: time = %1.12e dt = %1.12e \n",usr->par->t,usr->par->dt);

    // Update xtau_old and xDP_old
    ierr = UpdateStressOld(usr->dmeps,usr);CHKERRQ(ierr);

    ierr = VecCopy(usr->f,usr->fprev); CHKERRQ(ierr);

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
      
    }

    ierr = VecDestroy(&xphi); CHKERRQ(ierr);

    //check max(f) and min(f),
    PetscScalar fmax, fmin;
    ierr = VecMax(usr->f,NULL,&fmax); CHKERRQ(ierr);
    ierr = VecMin(usr->f,NULL,&fmin); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "Phase field: Maximum of f = %1.8f, Minimum of f = %1.8f\n", fmax, fmin); CHKERRQ(ierr);


    //check max(x) and min(x) - both face and center values are compared though, but pressure might be small if gravity <= 1 and eta << 1.
    PetscScalar xxmax, dtt, dtgap;
    PetscInt    imax, isize, isp;
    
    ierr = UpdateUMag(usr->dmPV, usr->xPV, usr); CHKERRQ(ierr);
    ierr = VecMax(usr->VMag,&imax, &xxmax); CHKERRQ(ierr);
    VecGetSize(usr->VMag,&isize);
    VecGetSize(usr->fp,&isp);

    usr->par->gamma = xxmax; CHKERRQ(ierr);
    if (usr->par->gamma < 1e-5) {usr->par->gamma = 1e-5;}

    /*
    //change dt accordingly
    dtt = usr->par->H/nz/usr->par->gamma/4; //maximum time step allowed for boundedness
    if (dtt<1e-8) dtt = 1e-8;
    */
    //temporary change for checking dependency on time step size
    dtt = 0.1*dtck;

    dtgap = tckpt - usr->par->t;   // gap between the current time and the next checkpoint

    if (dtgap <= 0) SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "dtgap is smaller or equal to zero, the next check points, tckpt, has not been updated properly");
    
    //check if too close to the check point, avoid left any gap smaller than 0.1*dtt for the next
    if (dtgap > 1.1*dtt) {usr->par->dt = dtt;}
    else {usr->par->dt = dtgap; ickpt++; tckpt += dtck; iwrt = PETSC_TRUE;}
    
    PetscPrintf(usr->comm, "Phase field: Maximum of U = %1.8f at i = %d of %d, %d  \n", xxmax, imax, isize, isp); CHKERRQ(ierr);
    PetscPrintf(usr->comm, "Phase field: gamma = %1.8f\n", usr->par->gamma); CHKERRQ(ierr);


    // increment timestep
    istep++;

  }

  // Destroy objects
  ierr = VecDestroy(&usr->f);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->fprev);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->dfx);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->dfz);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->volf);CHKERRQ(ierr);
  ierr = VecDestroy(&usr->xPV);CHKERRQ(ierr);

  ierr = VecDestroy(&usr->xphiprev);CHKERRQ(ierr);

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
  
  ierr = DMDestroy(&usr->dmPV);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmeps);CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmf); CHKERRQ(ierr);
  ierr = DMDestroy(&usr->dmphi); CHKERRQ(ierr);
  
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

  // Physical and material parameters
  ierr = PetscBagRegisterScalar(bag, &par->F4, 0.0, "F4", "Non-dimensional gravity for fluid 4"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F3, 0.0, "F3", "Non-dimensional gravity for fluids 3"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F2, 0.0, "F2", "Non-dimensional gravity for fluids 2"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->F1, 0.0, "F1", "Non-dimensional gravity for fluids 1"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->FS, 0.0, "FS", "Non-dimensional gravity for the solid phase"); CHKERRQ(ierr);
  
  ierr = PetscBagRegisterScalar(bag, &par->phi_0, 0.01, "phi_0", "Reference porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->n, 2.0, "n", "Porosity exponent"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->lambda, 0.0, "lambda", "Exponential melt weakening factor"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->pfs, 0.0, "pfs", "Static pressure at the free surface"); CHKERRQ(ierr);

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

  // Miscs
  ierr = PetscBagRegisterScalar(bag, &par->coeff_iso, 1.0/3.0, "coeff_iso", "Coefficients of the isotropic components"); CHKERRQ(ierr);
  ierr = PetscBagRegisterInt(bag, &par->Nmax, 25, "Nmax", "Max Newton iteration"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->tf_tol, 1e-8, "tf_tol", "Function tolerance for solving yielding stresses"); CHKERRQ(ierr);

  ierr = PetscBagRegisterScalar(bag, &par->alphaC, 1.0, "alphaC", "Coefficient in the smooth function of alpha"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->phicut, 1e-3, "phicut", "Cutoff porosity"); CHKERRQ(ierr);
  ierr = PetscBagRegisterScalar(bag, &par->etamin, 1e-3, "etamin", "Minimum shear viscosity"); CHKERRQ(ierr);

  // Controller for the evolution of porosity
  ierr = PetscBagRegisterBool(bag, &par->poroevo, PETSC_TRUE, "poroevo", "Controller for the evolution of porosity"); CHKERRQ(ierr);

  // Switch for the gaussian distribution or not
  ierr = PetscBagRegisterBool(bag, &par->gaussian, PETSC_TRUE, "gaussian", "Switch for the gaussian distribution or not"); CHKERRQ(ierr);


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
  Vec            coefflocal, xepslocal, xtaulocal, xDPlocal, xyieldlocal, toldlocal, poldlocal;
  Vec            flocal, volflocal, fplocal, fnlocal, xlocal, pplocal, yslocal;
  DM             dmphi;
  Vec            xphi, xphilocal;
  PetscScalar    phi0;
  PetscScalar    **coordx,**coordz;
  PetscInt       iprev, inext, icenter, Nmax;
  PetscScalar    tf_tol;
  PetscScalar    ***c, ***xxs, ***xxy, ***xxp;
  PetscScalar    dt, Z0, Z, r2local[4],Kphi[4],Kphi0,eta_v,zeta_v,eta_e,zeta_e,zeta_ve_dil;
  PetscScalar    eta_u, eta_d, zeta_u, zeta_d, F_u, F_d, C_d, C_u;
  PetscScalar    etaa[4], Fa[4], Ga[4], Ca[4], R2a[4];
  PetscScalar    angle, sina, Cc, Ct, eta_vk, R_pe, coeff_iso, phicut, alphaC;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

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


  PetscScalar etamin, zeta_max = 1e3 * etaa[1], Z_max = Z0*20;
  etamin = usr->par->etamin; 

  // reference porosity
  phi0 = usr->par->phi_0;
  Kphi0 = 1.0;

  // Get dm and solution vector for porosity
  dmphi = usr->dmphi;
  xphi  = usr->xphiprev;

  ierr = DMCreateLocalVector(dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dmphi,xphi,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

  // viscosity in the parallel damping dashpot
  eta_vk = usr->par->eta_vk;

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


  // solution
  ierr = DMGetLocalVector(dm, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);
  
  // Strain rates
  ierr = UpdateStrainRates(dm,x,usr); CHKERRQ(ierr);
  ierr = DMGetLocalVector(usr->dmeps, &xepslocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmeps, usr->xeps, INSERT_VALUES, xepslocal); CHKERRQ(ierr);

  
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
      PetscScalar DeltaP, pstat;

      // prepare the porosity data for:
      // phi: element
      // phif: left, right, down and up
      // phiv: dl, dr, ul, ur
      DMStagStencil pe[9];
      PetscScalar   phie[9], phi, phif[4],phiv[4];
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

      phi = 1.0 - phie[0];
      phif[0] = 1.0 - 0.5*(phie[0]+phie[1]);
      phif[1] = 1.0 - 0.5*(phie[0]+phie[2]);
      phif[2] = 1.0 - 0.5*(phie[0]+phie[3]);
      phif[3] = 1.0 - 0.5*(phie[0]+phie[4]);
      phiv[0] = 1.0 - 0.25*(phie[0]+phie[1]+phie[3]+phie[5]);
      phiv[1] = 1.0 - 0.25*(phie[0]+phie[2]+phie[3]+phie[6]);
      phiv[2] = 1.0 - 0.25*(phie[0]+phie[1]+phie[4]+phie[7]);
      phiv[3] = 1.0 - 0.25*(phie[0]+phie[2]+phie[4]+phie[8]);

      
      { // A = eta (center, c=1), and also compute chi_s and chi_p (center, c = 4,5)
        DMStagStencil point;
        PetscScalar   epsII,exx,ezz,exz,txx,tzz,txz,tauII,epsII_dev;
        PetscScalar   eta;
        PetscScalar   ff, volf;

        // ratio of the solid phase
        PetscScalar phis;
        phis = 1 - phi;

        // get the phase values in the element
        point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
        ierr = DMStagVecGetValuesStencil(usr->dmf,flocal,1,&point,&ff); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,1,&point,&volf); CHKERRQ(ierr);

        // get the fluid parameters for this cell
        //---------------
        ierr = DMStagVecGetValuesStencil(usr->dmf,fplocal,1,&point,&fp); CHKERRQ(ierr);
        ierr = DMStagVecGetValuesStencil(usr->dmf,fnlocal,1,&point,&fn); CHKERRQ(ierr);
        ifp = (PetscInt)fp;
        ifn = (PetscInt)fn;


        PetscScalar  eta_factor, Z_factor;

        eta_factor = PetscExpScalar(-27.0*phi)/(1-phi);
        Z_factor = PetscPowScalar(phi, -0.5);


        eta_u = etaa[ifp]* eta_factor;
        eta_d = etaa[ifn]* eta_factor;

        if (eta_u < etamin) eta_u = etamin;
        if (eta_d < etamin) eta_d = etamin;

        Z = Z0* Z_factor;
        zeta_e = Z*dt;

        zeta_u = etaa[ifp]/phi/(1-phi);
        zeta_d = etaa[ifn]/phi/(1-phi);

        if (zeta_u > zeta_max) zeta_u = zeta_max;
        if (zeta_d > zeta_max) zeta_d = zeta_max;
        if (Z > Z_max) Z = Z_max;
        zeta_e = Z*dt;

        F_u = Fa[ifp];
        F_d = Fa[ifn];
        C_u = Ca[ifp];
        C_d = Ca[ifn];


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

        // liquid pressure correction on the effective pressure, ap = alpha*pf*sin(angle)
        // P_e = (1-phi)*(\Delta P + ap);
        //if (phi > phicut) {ap = 0;}
        // else {ap = pfterm * sina/phis;}
        //ap = pfterm * sina/phis * PetscExpScalar(-PetscPowScalar(phi/phicut * alphaC, 1));
        //ap = pfterm * sina * (1 - PetscExpScalar(-PetscPowScalar(phicut/phi / alphaC, 1)));
        ap = pfterm * (1 - PetscExpScalar(-PetscPowScalar(phicut/phi / alphaC, 1)));

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
        cdl = PetscExpScalar(- PetscPowScalar(phicut/phi,1));
        zeta_ve_dil = zeta_ve * cdl;


        // get the deviatoric and isotropic stresses
        ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
        dotlam = stressSol[2]*volf;

        // effective viscosity
        if (epsII_dev > tf_tol) {
          eta = 0.5*stressSol[0]/epsII_dev * phis*volf;
        } else {
          eta = eta_ve *phis*volf;
        }
        if (phi>phicut && PetscAbs(divp) > tf_tol) {
          zeta = -stressSol[1]/divp * phis*volf;
        } else {
          zeta = zeta_ve *phis*volf;
        }
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
        cdl = PetscExpScalar(- PetscPowScalar(phicut/phi, 1));
        zeta_ve_dil = zeta_ve * cdl;

        // get the deviatoric and isotropic stresses
        ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
        dotlam += stressSol[2]*(1-volf);

        // effective viscosity
        if (epsII_dev > tf_tol) {
          eta += 0.5*stressSol[0]/epsII_dev * phis*(1-volf);
        } else {
          eta += eta_ve *phis*(1-volf);
        }
        if (cdl > 1e-8 && PetscAbs(divp) > tf_tol) {
          zeta += -stressSol[1]/divp * phis*(1-volf);
        } else {
          zeta += zeta_ve *phis*(1-volf);
        }
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
        DeltaP = -zeta*divp/phis;
        
        // save stresses for output
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,0,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txx; xxy[j][i][idx] = dotlam; xxp[j][i][idx] = DeltaP;
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,1,&idx); CHKERRQ(ierr); xxs[j][i][idx] = tzz; xxy[j][i][idx] = eta; // save eta in dof=1 of usr->xyield
        ierr = DMStagGetLocationSlot(usr->dmeps,ELEMENT,2,&idx); CHKERRQ(ierr); xxs[j][i][idx] = txz; xxy[j][i][idx] = chis; // save chis in dof = 2 of usr->xyield
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
        
        PetscScalar pfterm[4];
        point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = UP_LEFT;    point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP_RIGHT;   point[3].c = 0;
        //ierr = DMStagVecGetValuesStencil(usr->dmeps,yslocal,4,point,pp); CHKERRQ(ierr);
        for(ii=0;ii<4;ii++){ point[ii].c = 1;}
        ierr = DMStagVecGetValuesStencil(usr->dmeps,yslocal,4,point,pfterm); CHKERRQ(ierr);

                
        for (ii = 0; ii < 4; ii++) {
          PetscScalar eta_ve, zeta_ve, txxt, tzzt, txzt, tIIt, ap, pct, stressP[2], stressA[5], stressSol[3];

          PetscScalar  eta_factor, Z_factor;

          eta_factor = PetscExpScalar(-27.0*phiv[ii])/(1-phiv[ii]);
          Z_factor = PetscPowScalar(phiv[ii], -0.5);

          eta_u = etaa[ifp]* eta_factor;
          eta_d = etaa[ifn]* eta_factor;

          if (eta_u < etamin) eta_u = etamin;
          if (eta_d < etamin) eta_d = etamin;

          Z = Z0* Z_factor;
          zeta_e = Z*dt;

          zeta_u = etaa[ifp]/phiv[ii]/(1-phiv[ii]);
          zeta_d = etaa[ifn]/phiv[ii]/(1-phiv[ii]);

          if (zeta_u > zeta_max) zeta_u = zeta_max;
          if (zeta_d > zeta_max) zeta_d = zeta_max;
          if (Z > Z_max) Z = Z_max;
          zeta_e = Z*dt;

          // static pressure correction on the effective pressure, ap = alpha*pf*sin(angle)
          // P_e = (1-phi)*(\Delta P + ap);
          //if (phiv[ii] > phicut) {ap = 0;}
          //else {ap = pfterm[ii] * sina/phis[ii];}// zeta_u = 0; zeta_d = 0;}
//          ap = pfterm[ii] * sina/phis[ii] * PetscExpScalar(- PetscPowScalar(phiv[ii]/phicut * alphaC,1));
//          ap = pfterm[ii] * sina * (1 - PetscExpScalar(- PetscPowScalar(phicut/phiv[ii] / alphaC,1)));
          ap = pfterm[ii] * (1 - PetscExpScalar(- PetscPowScalar(phicut/phiv[ii] / alphaC,1)));

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
          cdl = PetscExpScalar(- PetscPowScalar(phicut/phiv[ii], 1));
          zeta_ve_dil = zeta_ve * cdl;

          // get the deviatoric and isotropic stresses
          ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
          dotlam[ii] = stressSol[2] * volf[ii];

          // effective viscosity
          if (epsII_dev[ii] > tf_tol) {
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
          cdl = PetscExpScalar(- PetscPowScalar(phicut/phiv[ii], 1));
          zeta_ve_dil = zeta_ve * cdl;

          // get the deviatoric and isotropic stresses
          ierr = VEVP_hyper_sol(Nmax, tf_tol, Cc, Ct, ap, angle, eta_ve, zeta_ve_dil, eta_vk, stressP, stressA, stressSol); CHKERRQ(ierr);
          dotlam[ii] += stressSol[2] * (1-volf[ii]);

          // effective viscosity
          if (epsII_dev[ii] > tf_tol) {
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
      PetscScalar  F2, F3;

      { // B = phi*F*ek - grad(chi_s*tau_old) + div(chi_p*dP_old) (edges, c=0)
        // (F = (rho^s-rho^f)*U*L/eta_ref * g*L/U^2)
        // hydrostatic pressure fix term according to free surface: dpstat/dx
        DMStagStencil point[4];
        PetscScalar   rhs[4], volf[4];
        PetscInt      ii,jj;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

        // collect phase values for the four edges
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);

        F2 = -(F_u*volf[2] + F_d*(1.0-volf[2]));
        F3 = -(F_u*volf[3] + F_d*(1.0-volf[3]));
                
        ii = i - sx;
        jj = j - sz;

        // d(chi_s*tau_xy_old)/dz compute on the left and right
        // d(chi_s*tau_xy_old)/dx and gravity compute on the down boundary only
        // zero second derivatives of compaction stresses on the true boundaries
        
        if (i > 0) {
          rhs[0] = pstat/dx[ii];
          rhs[0]+= -chis*told_xx_e/dx[ii];
          rhs[0]+= -0.5*(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
          rhs[0]+= chip * pold/dx[ii];
        } else {
          rhs[0] = 0.0;  // dp_stat/dx = 0 at the left boundary
          rhs[0] = -0.5*((cs[3]*told_xx[3]+cs[1]*told_xx[1])-(cs[2]*told_xx[2]+cs[0]*told_xx[0]))/dx[ii];
          rhs[0]+= -(cs[2]*told_xz[2]-cs[0]*told_xz[0])/dz[jj];
        }
        if (i < Nx-1) {
          rhs[1] = -pstat/dx[ii];
          rhs[1]+= chis*told_xx_e/dx[ii];
          rhs[1]+= -0.5*(cs[3]*told_xz[3]-cs[1]*told_xz[1])/dz[jj];
          rhs[1]+= -chip*pold/dx[ii];
        } else {
          rhs[1] = 0.0; //dp_stat/dx = 0 at the right boundary
          rhs[1] = -0.5*((cs[3]*told_xx[3]+cs[1]*told_xx[1])-(cs[2]*told_xx[2]+cs[0]*told_xx[0]))/dx[ii];
          rhs[1]+= -(cs[3]*told_xz[3]-cs[1]*told_xz[1])/dz[jj];
        }
        if (j > 0) {
          rhs[2] = 0.5*phif[2]*F2;
          rhs[2]+= -chis*told_zz_e/dz[jj];
          rhs[2]+= -0.5*(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
          rhs[2]+= chip*pold/dz[jj];
        } else {
          rhs[2] = phif[2]*F2;
          rhs[2]+= -0.5*((cs[3]*told_zz[3]+cs[2]*told_zz[2])-(cs[1]*told_zz[1]+cs[0]*told_zz[0]))/dz[jj];
          rhs[2]+= -(cs[1]*told_xz[1]-cs[0]*told_xz[0])/dx[ii];
        }
        if (j < Nz-1) {
          rhs[3] = 0.5*phif[3]*F3;
          rhs[3]+= chis*told_zz_e/dz[jj];
          rhs[3]+= -0.5*(cs[3]*told_xz[3]-cs[2]*told_xz[2])/dx[ii];
          rhs[3]+= -chip*pold/dz[jj];
        } else {
          rhs[3] = phif[3]*F3;
          rhs[3]+= -0.5*((cs[3]*told_zz[3]+cs[2]*told_zz[2])-(cs[1]*told_zz[1]+cs[0]*told_zz[0]))/dz[jj];
          rhs[3]+= -(cs[3]*told_xz[3]-cs[2]*told_xz[2])/dx[ii];
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
        PetscScalar   r2u, r2d, volf[4];
        PetscInt      ii;

        point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 1;
        point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 1;
        point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 1;
        point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 1;

        //-----use the phase field to interpolate the value of R2
        ierr = DMStagVecGetValuesStencil(usr->dmf,volflocal,4,point,volf); CHKERRQ(ierr);

        r2u = R2a[ifp];
        r2d = R2a[ifn];
        
        
        for (ii = 0; ii < 4; ii++) {

          r2local[ii] = r2u*volf[ii] + r2d*(1-volf[ii]);

          point[ii].c = 1;
          ierr = DMStagGetLocationSlot(dmcoeff, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);
          //Kphi[ii] = r2local[ii] * Kphi0*PetscPowScalar(phif[ii]/phi0, 3);
          Kphi[ii] = r2local[ii] * Kphi0*PetscPowScalar(phif[ii]/phi0, 3)* PetscPowScalar(1-phif[ii],-2);
          c[j][i][idx] = - Kphi[ii]; //-r2local[ii] * Kphi[ii];//

        }
      }

      { // D3 = R^2 * Kphi * (F - dpstat/dx * ex)  (edges, c=2)
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
          dd3[0] = -pstat/dx[ii];
        } else {
          dd3[0] = 0.0;  // dp_stat/dx = 0 at the left boundary
        }
        if (i < Nx-1) {
          dd3[1] = pstat/dx[ii];
        } else {
          dd3[1] = 0.0; //dp_stat/dx = 0 at the right boundary
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

  ierr = DMRestoreLocalVector(usr->dmeps,&xepslocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&poldlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &flocal);    CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &volflocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &fplocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &fnlocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf,  &pplocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,  &yslocal);   CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,  &xlocal);   CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dmphi, &xphilocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// FormBCList_PV
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "FormBCList_PV"
PetscErrorCode FormBCList_PV(DM dm, Vec x, DMStagBCList bclist, void *ctx)
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       k,n_bc,*idx_bc,i;
  PetscInt       sx, sz, nx, nz, Nx, Nz, iprev, icenter, inext;
  PetscScalar    *value_bc,*x_bc;
  BCType         *type_bc;
  PetscScalar    **coordx,**coordz;
  Vec            xlocal, toldlocal, paralocal;
  PetscScalar    vi;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  vi = usr->par->vi;


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

  // LEFT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'w','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = -vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // LEFT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'w','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);


  // RIGHT Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'e','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = vi;
    type_bc[k] = BC_DIRICHLET;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // RIGHT Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'e','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }

  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vx
  ierr = DMStagBCListGetValues(bclist,'s','-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=1; k<n_bc-1; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_NEUMANN;
  }
  ierr = DMStagBCListInsertValues(bclist,'-',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - Vz
  ierr = DMStagBCListGetValues(bclist,'s','|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {
    value_bc[k] = 0.0;
    type_bc[k] = BC_DIRICHLET;//
  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // DOWN Boundary - P
  ierr = DMStagBCListGetValues(bclist,'s','o',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);
  for (k=0; k<n_bc; k++) {

      value_bc[k] = 0;
      type_bc[k] = BC_DIRICHLET;

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

        point.i = i; point.j = Nz-1;   point.loc = ELEMENT; point.c = 1;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&eta); CHKERRQ(ierr);

        point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,paralocal,1,&point,&chis); CHKERRQ(ierr);

        point.i = i; point.j = Nz-1;   point.loc = UP_RIGHT; point.c = 2;
        ierr = DMStagVecGetValuesStencil(usr->dmeps,toldlocal,1,&point,&txz); CHKERRQ(ierr);

        dx = coordx[i+1][icenter] - coordx[i][icenter];
        value_bc[k] =  -(xv[1]-xv[0])/dx- chis/eta * txz  ;
        //PetscPrintf(PETSC_COMM_WORLD, "txz = %g, chis = %g, eta = %g, dv/dx= %g \n", txz, chis, eta, (xv[1]-xv[0])/dx);
        type_bc[k] = BC_NEUMANN;
      }
    }
//	value_bc[k] =  0;
//	type_bc[k] = BC_NEUMANN;
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

        value_bc[k] =0.5/eta  * (xx - chis * tzz)  ;
        type_bc[k] = BC_NEUMANN_T;
      }
    }
  
   // value_bc[k] = -2*vi*usr->par->H/usr->par->L  ; //(xx - chis * tzz)  ;
  //  type_bc[k] = BC_DIRICHLET;

  }
  ierr = DMStagBCListInsertValues(bclist,'|',0,&n_bc,&idx_bc,&x_bc,NULL,&value_bc,&type_bc);CHKERRQ(ierr);

  // restore
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&paralocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&toldlocal); CHKERRQ(ierr);

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
      PetscScalar   zp, fval = 0.0, xn=0.0, zcw;
      PetscInt      idx;


      zcw = 1.0e10;  // interface between crust and water

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;
      zp = coordz[j][icenter];

      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);

      xn = zcw - zp;
      fp[j][i][idx] = 1.01;
      fn[j][i][idx] = 3.01;
          
      fval = 0.5*(1 + PetscTanhScalar(xn/2.0/eps));
      
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

      if (usr->par->gaussian) {
        if (zp<zpeak) {
          rr = xp - xpeak;
          val = phi_bg + (phi_m-phi_bg)*PetscExpScalar(-pow(rr/xsig,2));
        }
        else {
          val = phi_bg + (phi_m-phi_bg) * PetscExpScalar( - (PetscPowScalar((xp-xpeak)/xsig,2) + PetscPowScalar((zp-zpeak)/zsig,2)));
        }
      }
      else {
        rr = PetscPowScalar(PetscPowScalar(xp-xpeak, 2) + PetscPowScalar(zp-zpeak, 2), 0.5);
        if (rr < xsig) { val = phi_m;}
        else           { val = phi_bg;}
      }

      xx[j][i][idx] = 1 - val;
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);

  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

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

      if (i==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[0] = ezzc;
        exxn[0] = exxc;
      }

      if (i==Nx-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        ezzn[1] = ezzc;
        exxn[1] = exxc;
      }

      if (j==0) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[0] = exxc;
        ezzn[0] = ezzc;
      }

      if (j==Nz-1) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[2] = exxc;
        ezzn[2] = ezzc;
      }

      if ((i==Nx-1) && (j==Nz-1)) { // boundaries
        pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
        ierr = DMStagGetPointStrainRates(dm,xlocal,1,&pointC,&epsIIc,&exxc,&ezzc,&exzc); CHKERRQ(ierr);
        exxn[3] = exxc;
        ezzn[3] = ezzc;
      }

      if ((i==0) || (i==Nx-1) || (j==0) || (j==Nz-1)) { // boundaries
        for (ii = 0; ii < 4; ii++) {
          epsIIn[ii] = PetscPowScalar(0.5*(exxn[ii]*exxn[ii] + ezzn[ii]*ezzn[ii] + 2.0*exzn[ii]*exzn[ii]),0.5);
        }
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
