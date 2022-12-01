#include "paper_vevp.h"


// ---------------------------------------
// UpdateStressOld
// ---------------------------------------
PetscErrorCode UpdateStressOld(DM dmeps, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            xtaulocal, xDPlocal, tauold_local,DPold_local;
  PetscScalar    ***xx, ***xxp;
  PetscInt       ic, ic1, ic2, sx, sz, nx, nz, i, j, Nx, Nz;
  PetscInt       idl, idr, iul, iur, idl1, idr1, iul1, iur1, idl2, idr2, iul2, iur2;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // Get local vectors from dmeps
  ierr = DMGetLocalVector(dmeps, &xtaulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmeps, usr->xtau, INSERT_VALUES, xtaulocal); CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmeps, &xDPlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmeps, usr->xDP, INSERT_VALUES, xDPlocal); CHKERRQ(ierr);


  // Create local vectors for the stress_old terms
  ierr = DMCreateLocalVector (dmeps,&tauold_local); CHKERRQ(ierr);
  ierr = DMCreateLocalVector (dmeps,&DPold_local); CHKERRQ(ierr);
  ierr = VecCopy(xtaulocal, tauold_local); CHKERRQ(ierr);
  ierr = VecCopy(xDPlocal, DPold_local); CHKERRQ(ierr);

  // create array from tauold_local and add the rotation terms
  ierr = DMStagVecGetArray(dmeps, tauold_local, &xx); CHKERRQ(ierr);

  // create array from dpold_local and add the rotation terms
  ierr = DMStagVecGetArray(dmeps, DPold_local, &xxp); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMStagGetGlobalSizes(dmeps, &Nx, &Nz,NULL);CHKERRQ(ierr);
  // Get indices for the four corners
  ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,  0, &idl); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT, 0, &idr); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,    0, &iul); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,   0, &iur); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,  1, &idl1); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT, 1, &idr1); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,    1, &iul1); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,   1, &iur1); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmeps,DOWN_LEFT,  2, &idl2); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,DOWN_RIGHT, 2, &idr2); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_LEFT,    2, &iul2); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,UP_RIGHT,   2, &iur2); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dmeps,ELEMENT,  0, &ic); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,ELEMENT,  1, &ic1); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dmeps,ELEMENT,  2, &ic2); CHKERRQ(ierr);

  // interpolating dpold on the vertices
  for (j=sz;j<sz+nz;j++){
    for (i=sx;i<sx+nx;i++){
      DMStagStencil pe[4];
      PetscScalar   dp[4];

      pe[0].i = i  ; pe[0].j = j  ; pe[0].loc = ELEMENT; pe[0].c = 0;
      pe[1].i = i-1; pe[1].j = j  ; pe[1].loc = ELEMENT; pe[1].c = 0;
      pe[2].i = i  ; pe[2].j = j-1; pe[2].loc = ELEMENT; pe[2].c = 0;
      pe[3].i = i-1; pe[3].j = j-1; pe[3].loc = ELEMENT; pe[3].c = 0;

      if (i==0) {pe[1].i = pe[0].i; pe[3].i = pe[0].i;}
      if (j==0) {pe[2].j = pe[0].j; pe[3].j = pe[0].j;}

      ierr = DMStagVecGetValuesStencil(dmeps,DPold_local,4,pe,dp); CHKERRQ(ierr);

      xxp[j][i][idl] = 0.25*(dp[0]+dp[1]+dp[2]+dp[3]);

      if (i==Nx-1) {xxp[j][i][idr] = 0.5*(dp[0]+dp[2]);}
      if (j==Nz-1) {xxp[j][i][iul] = 0.5*(dp[0]+dp[1]);}
      if (i==Nx-1 && j==Nz-1) {xxp[j][i][iur] = dp[0];}

    }
  }


  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps, tauold_local, &xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,tauold_local,INSERT_VALUES,usr->xtau_old); CHKERRQ(ierr);
  ierr = VecDestroy(&tauold_local); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dmeps, DPold_local, &xxp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (usr->dmeps,DPold_local,INSERT_VALUES,usr->xDP_old); CHKERRQ(ierr);
  ierr = VecDestroy(&DPold_local); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmeps,&xtaulocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmeps,&xDPlocal); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}



// ---------------------------------------
// UpdateUMag
// ---------------------------------------
PetscErrorCode UpdateUMag(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx;
  PetscScalar    ***xx;
  Vec            xlocal, umlocal, fplocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Local vectors
  ierr = DMCreateLocalVector (dm,&umlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,umlocal,&xx); CHKERRQ(ierr);

  ierr = DMGetLocalVector(usr->dmPV,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  // get local vector for fp
  ierr = DMGetLocalVector(usr->dmf,&fplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmf,usr->fp,INSERT_VALUES,fplocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);


  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  pointC, point[4];
      PetscScalar    u[4], ff;

      // center
      pointC.i = i; pointC.j = j; pointC.loc = ELEMENT; pointC.c = 0;
      ierr = DMStagVecGetValuesStencil(usr->dmf,fplocal,1,&pointC,&ff); CHKERRQ(ierr);

      // Strain rates: corner
      point[0].i = i; point[0].j = j; point[0].loc = LEFT;  point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = RIGHT; point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN;  point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP;    point[3].c = 0;

      ierr = DMStagVecGetValuesStencil(dm,xlocal,4,point,u); CHKERRQ(ierr);

      ierr = DMStagGetLocationSlot(dm,ELEMENT,0,&idx); CHKERRQ(ierr);
      xx[j][i][idx] = sqrt(PetscPowScalar((u[0]+u[1])/2, 2) + PetscPowScalar((u[2]+u[3])/2, 2) );
    }
  }


  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,umlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,umlocal,INSERT_VALUES,usr->VMag); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,umlocal,INSERT_VALUES,usr->VMag); CHKERRQ(ierr);
  ierr = VecDestroy(&umlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(usr->dmPV, &xlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmf, &fplocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update the static pressure
// ---------------------------------------
PetscErrorCode UpdatePStatic(DM dm, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       iul, iur, ic,idr;
  PetscScalar    ***pp;
  Vec            vflocal, fplocal, fnlocal, pplocal;
  PetscScalar    F[4], p0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  F[0] = 0;
  F[1] = usr->par->FS;
  F[2] = usr->par->FS;
  F[3] = 0;

  p0 = usr->par->pfs; // pressure at the free surface


  // volume fraction
  ierr = DMGetLocalVector(dm, &vflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->volf, INSERT_VALUES, vflocal); CHKERRQ(ierr);

  // indices of positive and negative fluids
  ierr = DMGetLocalVector(dm, &fplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fp, INSERT_VALUES, fplocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &fnlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fn, INSERT_VALUES, fnlocal); CHKERRQ(ierr);

  // Create coefficient local vector
  //  ierr = DMCreateLocalVector(dm, &pplocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &pplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->Pst, INSERT_VALUES, pplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray  (dm, pplocal, &pp); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get location slot
  ierr = DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, ELEMENT,    0, &ic ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr ); CHKERRQ(ierr);

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);

  // Get the coordinate - to customise the noise of static pressure
  PetscScalar **coordx, **coordz;
  PetscInt    iprev, inext, icenter;
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT,&iprev);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT,&inext);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // calculate the mass from top to the bottom
  // !!THE FOLLOWING IS NOT CORRECT FOR PARALLEL which add the total mass from all cells above
  for (j = sz+nz-1; j >= 0; j--) {
    for (i = 0; i<sx+nx; i++) {
      DMStagStencil  point[3], pt[3];
      PetscScalar    fp, fn, fu, fd, vf[3];
      PetscInt       ifp, ifn, ii, idx;

      point[0].i = i; point[0].j = j; point[0].loc = DOWN_LEFT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = DOWN_RIGHT;   point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = ELEMENT;    point[2].c = 0;

      pt[0].i = i; pt[0].j = j; pt[0].loc = LEFT;    pt[0].c = 0;
      pt[1].i = i; pt[1].j = j; pt[1].loc = RIGHT;   pt[1].c = 0;
      pt[2].i = i; pt[2].j = j; pt[2].loc = UP;    pt[2].c = 0;

      // get the fluid parameters for this cell
      //---------------
      ierr = DMStagVecGetValuesStencil(dm,fplocal,1,&point[2],&fp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,fnlocal,1,&point[2],&fn); CHKERRQ(ierr);
      ifp = (PetscInt)fp;
      ifn = (PetscInt)fn;

      fu = F[ifp];
      fd = F[ifn];


      if (j==Nz-1) {

        PetscScalar pnoise[3];
        for (ii=0;ii<3;ii++){
          pnoise[ii] = 0.0;
        }
        pp[j][i][iul] = p0 + pnoise[0];
        pp[j][i][iur] = p0 + pnoise[1];

        ierr = DMStagVecGetValuesStencil(dm,vflocal,3,pt,vf); CHKERRQ(ierr);
        for (ii=0; ii<3; ii++) {
          ierr = DMStagGetLocationSlot(dm, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);

          if (ii==2) {
            pp[j][i][idx] = p0 + 0.5*dz[j]*(fu * vf[ii] + fd * (1-vf[ii])) + pnoise[ii];
          }
          else {pp[j][i][idx] = p0 + dz[j]*(fu * vf[ii] + fd * (1-vf[ii])) + pnoise[ii];}
        }
      }
      else {
        ierr = DMStagVecGetValuesStencil(dm,vflocal,3,pt,vf); CHKERRQ(ierr);
        for (ii=0; ii<3; ii++) {
          ierr = DMStagGetLocationSlot(dm, point[ii].loc, point[ii].c, &idx); CHKERRQ(ierr);

          pp[j][i][idx] = pp[j+1][i][idx] + dz[j]*(fu * vf[ii] + fd * (1-vf[ii]));
        }
      }

    }
  }
  // release dx dz
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);


  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,pplocal,&pp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,pplocal,INSERT_VALUES,usr->Pst); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,pplocal,INSERT_VALUES,usr->Pst); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &pplocal ); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &fplocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &fnlocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &vflocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update the full liquid pressure: dof0 - hydrostatic pressure, dof1 - full liquid pressure
// ---------------------------------------
PetscErrorCode UpdateYS(DM dmPV, Vec x, DM dmf, Vec Pst, DM dmeps, Vec xTau, Vec xDP, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscScalar   ***pp, iota;
  Vec            pstlocal, xlocal, taulocal, dplocal, pplocal;
  Vec            xphilocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  iota = usr->par->iota;

  // porosity
  ierr = DMCreateLocalVector(usr->dmphi,&xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(usr->dmphi,usr->xphiprev,INSERT_VALUES,xphilocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(usr->dmphi,usr->xphiprev,INSERT_VALUES,xphilocal);CHKERRQ(ierr);

  // hydrostatic pressure
  ierr = DMGetLocalVector(dmf, &pstlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmf, Pst, INSERT_VALUES, pstlocal); CHKERRQ(ierr);

  // dynamic pressure in the liquid phase
  ierr = DMGetLocalVector(dmPV, &xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmPV, x, INSERT_VALUES, xlocal); CHKERRQ(ierr);

  // effective shear and volumetric stresses (lagged)
  ierr = DMGetLocalVector(dmeps, &taulocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmeps, xTau, INSERT_VALUES, taulocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmeps, &dplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dmeps, xDP, INSERT_VALUES,  dplocal); CHKERRQ(ierr);

  // local vector for the
  // dof 0: total pressure; dof 1: second invariant of shear stresses
  ierr = DMCreateLocalVector(dmeps,&pplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray  (dmeps,pplocal,&pp); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dmeps, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dmeps, &Nx,&Nz,NULL);CHKERRQ(ierr);


  for (j = sz; j<sz+nz; j++) {
    for (i = sx; i<sx+nx; i++) {
      DMStagStencil  point[5], pe[9];
      PetscScalar    pst[5], pf[9], dp[9], ppf[5], pdp[5];
      PetscScalar    phie[9], phi[5];
      PetscInt       ii, idx;

      //stencil
      point[0].i = i; point[0].j = j; point[0].loc = ELEMENT;    point[0].c = 0;
      point[1].i = i; point[1].j = j; point[1].loc = DOWN_LEFT;  point[1].c = 0;
      point[2].i = i; point[2].j = j; point[2].loc = DOWN_RIGHT; point[2].c = 0;
      point[3].i = i; point[3].j = j; point[3].loc = UP_LEFT;    point[3].c = 0;
      point[4].i = i; point[4].j = j; point[4].loc = UP_RIGHT;   point[4].c = 0;

      //hydrostatic pressure - element and vertex
      ierr = DMStagVecGetValuesStencil(dmf,pstlocal,5,point,pst); CHKERRQ(ierr);

      //dynamic pressure - element
      pe[0].i = i;   pe[0].j = j  ; pe[0].loc = ELEMENT;    pe[0].c = 0;
      pe[1].i = i-1; pe[1].j = j  ; pe[1].loc = ELEMENT;    pe[1].c = 0;
      pe[2].i = i+1; pe[2].j = j  ; pe[2].loc = ELEMENT;    pe[2].c = 0;
      pe[3].i = i;   pe[3].j = j-1; pe[3].loc = ELEMENT;    pe[3].c = 0;
      pe[4].i = i-1; pe[4].j = j-1; pe[4].loc = ELEMENT;    pe[4].c = 0;
      pe[5].i = i+1; pe[5].j = j-1; pe[5].loc = ELEMENT;    pe[5].c = 0;
      pe[6].i = i  ; pe[6].j = j+1; pe[6].loc = ELEMENT;    pe[6].c = 0;
      pe[7].i = i-1; pe[7].j = j+1; pe[7].loc = ELEMENT;    pe[7].c = 0;
      pe[8].i = i+1; pe[8].j = j+1; pe[8].loc = ELEMENT;    pe[8].c = 0;

      if (i==0)    {pe[1].i = i; pe[4].i=i; pe[7].i=i;}
      if (i==Nx-1) {pe[2].i = i; pe[5].i=i; pe[8].i=i;}
      if (j==0)    {pe[3].j = j; pe[4].j=j; pe[5].j=j;}
      if (j==Nz-1) {pe[6].j = j; pe[7].j=j; pe[8].j=j;}

      ierr = DMStagVecGetValuesStencil(dmPV,xlocal,9,pe,pf); CHKERRQ(ierr);

      ppf[0] = pf[0];
      ppf[1] = 0.25 * (pf[0]+pf[1]+pf[3]+pf[4]);
      ppf[2] = 0.25 * (pf[0]+pf[2]+pf[3]+pf[5]);
      ppf[3] = 0.25 * (pf[0]+pf[1]+pf[6]+pf[7]);
      ppf[4] = 0.25 * (pf[0]+pf[2]+pf[6]+pf[8]);

      //effective volumetric stresses
      ierr = DMStagVecGetValuesStencil(dmeps,dplocal,9,pe,dp); CHKERRQ(ierr);

      pdp[0] = dp[0];
      pdp[1] = 0.25 * (dp[0]+dp[1]+dp[3]+dp[4]);
      pdp[2] = 0.25 * (dp[0]+dp[2]+dp[3]+dp[5]);
      pdp[3] = 0.25 * (dp[0]+dp[1]+dp[6]+dp[7]);
      pdp[4] = 0.25 * (dp[0]+dp[2]+dp[6]+dp[8]);

      //porosity
      ierr = DMStagVecGetValuesStencil(usr->dmphi,xphilocal,9,pe,phie); CHKERRQ(ierr);

      phi[0] = phie[0];
      phi[1] = 0.25 * (phie[0]+phie[1]+phie[3]+phie[4]);
      phi[2] = 0.25 * (phie[0]+phie[2]+phie[3]+phie[5]);
      phi[3] = 0.25 * (phie[0]+phie[1]+phie[6]+phie[7]);
      phi[4] = 0.25 * (phie[0]+phie[2]+phie[6]+phie[8]);


      for (ii=0;ii<5;ii++) {
        PetscScalar  pf_local;

        ierr = DMStagGetLocationSlot(dmeps, point[ii].loc, 0, &idx); CHKERRQ(ierr);

        pf_local = (ppf[ii] + pst[ii]);
        pp[j][i][idx] = pst[ii];//phi[ii]*pdp[ii] + pf_local * iota;

        ierr = DMStagGetLocationSlot(dmeps, point[ii].loc, 1, &idx); CHKERRQ(ierr);
        pp[j][i][idx] = pf_local;

      }

    }
  }

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dmeps,pplocal,&pp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dmeps,pplocal,INSERT_VALUES,usr->YS); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dmeps,pplocal,INSERT_VALUES,usr->YS); CHKERRQ(ierr);
  ierr = VecDestroy(&pplocal); CHKERRQ(ierr);


  ierr = DMRestoreLocalVector(dmf  , &pstlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmPV , &xlocal  ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmeps, &taulocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmeps, &dplocal ); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmphi, &xphilocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


