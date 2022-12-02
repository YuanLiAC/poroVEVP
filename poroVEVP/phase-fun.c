#include "poro-vevp.h"


// static functions -
// - evaluate the volume fraction in a rectangular cell from the phase value of the center points on the top and bottom edges
static PetscScalar volf_2d(PetscScalar f1, PetscScalar f2, PetscScalar cc, PetscScalar ar)

{ PetscScalar tol, r10, r20, r1, r2, d1, d2, fchk, aa, result;

  tol = 1e-2;
  r10 = 2*f1 - 1.0;
  r20 = 2*f2 - 1.0;
  r1 = r10;
  r2 = r20;


  d1 = 2.0*cc*r1;
  d2 = 2.0*cc*r2;


  //1st check: is the interface too far away
  fchk = PetscAbs(d1+d2);
  aa   = sqrt(ar*ar + 1.0);
  if (fchk < aa ) {
    PetscScalar tx, tz;
    tx = (d1-d2);

    if (PetscAbs(tx)>1.0) {
      if (PetscAbs(tx)-1.0 > tol) {
        PetscPrintf(PETSC_COMM_WORLD, "f1 = %1.4f, f2 = %1.4f, d1= %1.4f, d2 = %1.4f, r10 = %1.4f, r20=%1.4f, cc = %1.4f, tx = %1.4f\n", f1, f2,d1, d2,r10, r20, cc, tx);
      }
      tx = tx/PetscAbs(tx);
    }

    tz = sqrt(1.0 - tx*tx);

    if (PetscAbs(tz)<tol) {
      //2.1 check: a horizontal line?
      if (d1+d2>=0) {result = PetscMin(0.5+(d1+d2), 1.0);}
      else          {result = 1.0 - PetscMin(0.5-(d1+d2), 1.0);}
    }
    else if (PetscAbs(tx)<tol) {
      //2.2 check: a vertical line?
      if (d1>=0) {result = PetscMin(0.5+d1/ar, 1.0);}
      else       {result = 1.0 - PetscMin(0.5-d1/ar, 1.0);}
    }
    else {
      //3 check: intersection with z = 0 and z = 1.0
      PetscScalar xb, xu, k0, k1, x0, x1;

      k0 = tz/tx;

      xb = 0.5*ar + d2/tz;
      xu = 0.5*ar + d1/tz;
      k1 = -k0*xb;

      if      (xu<=0.0) {x1 = 0.0;}
      else if (xu>=ar ) {x1 = ar; }
      else              {x1 = xu; }

      if      (xb<=0.0) {x0 = 0.0;}
      else if (xb>=ar ) {x0 = ar ;}
      else              {x0 = xb ;}

      result = (x1 - (0.5*k0*(x1*x1 - x0*x0) + k1*(x1-x0)))/ar;

      if (result <0 || result > 1.0) {
        PetscPrintf(PETSC_COMM_WORLD, "WRONG vvf, greater than 1 or smaller than zero, volf = %1.4f", result);}

    }
  }
  else if (f1>=0.5) {result = 1.0;} // line too far, only fluid 1
  else {result =0.0;}                 // line is too far away, only fluid 2

  return(result);
}


// ---------------------------------------
// Cleanup stage for FP, FN
// it should be called immediately after UpdateVolFrac
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CleanUpFPFN"
PetscErrorCode CleanUpFPFN(DM dm, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            vflocal, fplocal, fnlocal;
  PetscInt       i,j, sx, sz, nx, nz;
  PetscScalar    ***fp, ***fn;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create local vector for the volume fraction, fp (fluid type at n+) and fn (fluid type at n-)
  ierr = DMGetLocalVector(dm, &vflocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->volf, INSERT_VALUES, vflocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &fplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fp, INSERT_VALUES, fplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, fplocal, &fp); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &fnlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fn, INSERT_VALUES, fnlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, fnlocal, &fn); CHKERRQ(ierr);

  // Loop over local domain - clean up fp and fn
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point;
      PetscScalar   vf;
      PetscInt      idx;

      point.i = i; point.j = j; point.loc = ELEMENT; point.c = 0;

      ierr = DMStagVecGetValuesStencil(dm, vflocal, 1, &point, &vf); CHKERRQ(ierr);
      ierr = DMStagGetLocationSlot(dm, point.loc, point.c, &idx); CHKERRQ(ierr);
      if (vf<1e-8) {fp[j][i][idx] = 0.01;}
      if (1.0 - vf<1e-8) {fn[j][i][idx] = 0.01;}
    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dm,fplocal,&fp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,fplocal,INSERT_VALUES,usr->fp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,fplocal,INSERT_VALUES,usr->fp); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,fnlocal,&fn);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,fnlocal,INSERT_VALUES,usr->fn); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,fnlocal,INSERT_VALUES,usr->fn); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&vflocal); CHKERRQ(ierr);
  ierr = VecDestroy(&fplocal); CHKERRQ(ierr);
  ierr = VecDestroy(&fnlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Collection stage for FP, FN into NFP, NFN
// It should follow CleanUpFPFN, then copy NFP and NFN into FP and FN immediately afterwards
// ---------------------------------------
#undef __FUNCT__
#define __FUNCT__ "CollectFPFN"
PetscErrorCode CollectFPFN(DM dm, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  Vec            fplocal, fnlocal, nfplocal, nfnlocal;
  PetscInt       i,j, sx, sz, nx, nz, idx, Nx, Nz;
  PetscScalar    ***nfp, ***nfn;

  PetscErrorCode ierr;
  PetscFunctionBegin;

  // Get domain size
  ierr = DMStagGetGlobalSizes(dm,&Nx,&Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Create local vector for the volume fraction, fp (fluid type at n+) and fn (fluid type at n-)
  ierr = DMGetLocalVector(dm, &fplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fp, INSERT_VALUES, fplocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &fnlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm, usr->fn, INSERT_VALUES, fnlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &nfplocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, nfplocal, &nfp); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm, &nfnlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, nfnlocal, &nfn); CHKERRQ(ierr);

  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  // Loop over local domain - clean up fp and fn
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil point[9];
      PetscScalar   fp[9], fn[9];
      PetscInt      ii;

      point[0].i = i;   point[0].j = j;   point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i-1; point[1].j = j;   point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i+1; point[2].j = j;   point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i;   point[3].j = j-1; point[3].loc = ELEMENT; point[3].c = 0;
      point[4].i = i-1; point[4].j = j-1; point[4].loc = ELEMENT; point[4].c = 0;
      point[5].i = i+1; point[5].j = j-1; point[5].loc = ELEMENT; point[5].c = 0;
      point[6].i = i;   point[6].j = j+1; point[6].loc = ELEMENT; point[6].c = 0;
      point[7].i = i-1; point[7].j = j+1; point[7].loc = ELEMENT; point[7].c = 0;
      point[8].i = i+1; point[8].j = j+1; point[8].loc = ELEMENT; point[8].c = 0;

      if (i==0) {point[1].i=point[0].i; point[4].i=point[0].i; point[7].i=point[0].i;}
      if (i==Nx-1) {point[2].i=point[0].i; point[5].i=point[0].i; point[8].i=point[0].i;}
      if (j==0) {point[3].j=point[0].j; point[4].j=point[0].j; point[5].j=point[0].j;}
      if (j==Nz-1) {point[6].j=point[0].j; point[7].j=point[0].j; point[8].j=point[0].j;}


      ierr = DMStagVecGetValuesStencil(dm, fplocal, 9, point, fp); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm, fnlocal, 9, point, fn); CHKERRQ(ierr);

      nfp[j][i][idx] = fp[0];
      nfn[j][i][idx] = fn[0];

      for (ii=1; ii<9; ii++) {
        if ((PetscInt)nfp[j][i][idx] != (PetscInt)fp[ii] && (PetscInt)fp[ii] !=0) {
          if ((PetscInt)nfp[j][i][idx] ==0 ) {nfp[j][i][idx] = fp[ii];}
          else {
            PetscPrintf(usr->comm, "error region, cell i,j = %d, %d, fp=%g, %g, %g, %g, %g, %g, %g, %g, %g \n", i, j, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8]);
            SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "THREE FLUIDS IN 9 CELLS: positive fluid");
          }
        }
        if ((PetscInt)nfn[j][i][idx] != (PetscInt)fn[ii] && (PetscInt)fn[ii] !=0) {
          if ((PetscInt)nfn[j][i][idx] ==0 ) {nfn[j][i][idx] = fn[ii];}
          else {
            PetscPrintf(usr->comm, "error region, cell i,j = %d, %d, fp=%g, %g, %g, %g, %g, %g, %g, %g, %g \n", i, j, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5], fn[6], fn[7], fn[8]);
            SETERRQ(usr->comm, PETSC_ERR_ARG_NULL, "THREE FLUIDS IN 9 CELLS: negative fluid");
          }
        }
      }

    }
  }

  // Restore arrays, local vectors
  ierr = DMStagVecRestoreArray(dm,nfplocal,&nfp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,nfplocal,INSERT_VALUES,usr->nfp); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,nfplocal,INSERT_VALUES,usr->nfp); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,nfnlocal,&nfn);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,nfnlocal,INSERT_VALUES,usr->nfn); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,nfnlocal,INSERT_VALUES,usr->nfn); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&fplocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&fnlocal); CHKERRQ(ierr);
  ierr = VecDestroy(&nfplocal); CHKERRQ(ierr);
  ierr = VecDestroy(&nfnlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update dfdx and dfdz
// ---------------------------------------
PetscErrorCode UpdateDF(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       icenter, idx;
  PetscScalar    ***df1, ***df2;
  PetscScalar    **coordx,**coordz;
  Vec            dfxlocal, dfzlocal, xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Local vectors
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dfxlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, dfxlocal, &df1); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &dfzlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, dfzlocal, &df2); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    dx, dz, fval[4];

      // df/dx, df/dz: center
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i+1; point[1].j = j  ; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i  ; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j+1; point[3].loc = ELEMENT; point[3].c = 0;

      if ((i!=0) && (i!=Nx-1)) {
        dx = coordx[i+1][icenter] -  coordx[i-1][icenter];
      } else if (i == 0) {
        point[0].i = i; dx = coordx[i+1][icenter] - coordx[i][icenter];
      } else if (i == Nx-1) {
        point[1].i = i; dx = coordx[i][icenter] - coordx[i-1][icenter];
      }

      if ((j!=0) && (j!=Nz-1)) {
        dz = coordz[j+1][icenter] -  coordz[j-1][icenter];
      } else if (j == 0) {
        point[2].j = j; dz = coordz[j+1][icenter] - coordz[j][icenter];
      } else if (j == Nz-1) {
        point[3].j = j; dz = coordz[j][icenter] - coordz[j-1][icenter];
      }

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval); CHKERRQ(ierr);

      df1[j][i][idx] = (fval[1] - fval[0])/dx;
      df2[j][i][idx] = (fval[3] - fval[2])/dz;

    }
  }

  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,dfxlocal,&df1); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,dfxlocal,INSERT_VALUES,usr->dfx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,dfxlocal,INSERT_VALUES,usr->dfx); CHKERRQ(ierr);
  ierr = VecDestroy(&dfxlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,dfzlocal,&df2); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,dfzlocal,INSERT_VALUES,usr->dfz); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,dfzlocal,INSERT_VALUES,usr->dfz); CHKERRQ(ierr);
  ierr = VecDestroy(&dfzlocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------- */
PetscErrorCode ExplicitStep(DM dm, Vec xprev, Vec x, PetscScalar dt, void *ctx)
/* ------------------------------------------------------------------- */
{
  UsrData        *usr = (UsrData*)ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, idx, icenter;
  PetscScalar    gamma, eps;
  PetscScalar    **coordx,**coordz;
  PetscScalar    ***xx,***xxp;
  Vec            dfxlocal, dfzlocal, xlocal, xplocal;
  Vec            xVellocal;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  // User Parameter
  gamma = usr->par->gamma;
  eps = usr->par->eps;

  // create a dmPV and xPV in usrdata, copy data in and extract them here
  ierr = DMGetLocalVector(usr->dmPV, &xVellocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (usr->dmPV, usr->xPV, INSERT_VALUES, xVellocal); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  // Get global size
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);

  // Create local vector
  ierr = DMCreateLocalVector(dm,&xplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,xprev,INSERT_VALUES,xplocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,xprev,INSERT_VALUES,xplocal); CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&dfxlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,usr->dfx,INSERT_VALUES,dfxlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,usr->dfx,INSERT_VALUES,dfxlocal); CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&dfzlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin (dm,usr->dfz,INSERT_VALUES,dfzlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd (dm,usr->dfz,INSERT_VALUES,dfzlocal); CHKERRQ(ierr);

  // get array from xlocal
  ierr = DMStagVecGetArray(dm, xlocal, &xx); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, xplocal, &xxp); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);

  // get the location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT, 0, &idx); CHKERRQ(ierr);

  // Get the cell sizes
  PetscScalar *dx, *dz;
  ierr = DMStagCellSizeLocal_2d(dm, &nx, &nz, &dx, &dz); CHKERRQ(ierr);

  // loop over local domain and get the RHS value
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i < sx+nx; i++) {

      DMStagStencil point[5];
      PetscInt      ii,ix,iz;
      PetscScalar   fe[5], dfxe[5], dfze[5], gfe[5], c[5], fval = 0.0;

      ix = i - sx;
      iz = j - sz;

      point[0].i = i;   point[0].j = j;   point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i-1; point[1].j = j;   point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i+1; point[2].j = j;   point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i;   point[3].j = j-1; point[3].loc = ELEMENT; point[3].c = 0;
      point[4].i = i;   point[4].j = j+1; point[4].loc = ELEMENT; point[4].c = 0;

      // default zero flux on boundary
      if (i==0)    {point[1] = point[0];}
      if (i==Nx-1) {point[2] = point[0];}
      if (j==0)    {point[3] = point[0];}
      if (j==Nz-1) {point[4] = point[0];}

      ierr = DMStagVecGetValuesStencil(dm,dfxlocal,5,point,dfxe); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,dfzlocal,5,point,dfze); CHKERRQ(ierr);
      ierr = DMStagVecGetValuesStencil(dm,xplocal ,5,point,fe); CHKERRQ(ierr);

      for (ii=1; ii<5; ii++) {

        PetscScalar epsAlt;  //coefficients of anti-diffusion, center

        gfe[ii] = sqrt(dfxe[ii]*dfxe[ii]+dfze[ii]*dfze[ii]);

        if (gfe[ii] > 1e-10) {epsAlt = fe[ii]*(1-fe[ii])/gfe[ii];}
        else {epsAlt = eps;}

        c[ii] = epsAlt; //coefficients at the center

      }

      //diffusion terms
      fval = gamma*(eps * ((fe[2]+fe[1]-2*fe[0])/dx[ix]/dx[ix] + (fe[4]+fe[3]-2*fe[0])/dz[iz]/dz[iz]));

      //sharpen terms
      fval -= gamma* ( (c[2]*dfxe[2] - c[1]*dfxe[1])/(2.0*dx[ix]) + (c[4]*dfze[4]-c[3]*dfze[3])/(2.0*dz[iz]));


      { // velocity on the face and advection terms
        DMStagStencil pf[4];
        PetscScalar vf[4];

        pf[0].i = i; pf[0].j = j; pf[0].loc = LEFT;  pf[0].c = 0;
        pf[1].i = i; pf[1].j = j; pf[1].loc = RIGHT; pf[1].c = 0;
        pf[2].i = i; pf[2].j = j; pf[2].loc = DOWN;  pf[2].c = 0;
        pf[3].i = i; pf[3].j = j; pf[3].loc = UP;    pf[3].c = 0;

        ierr = DMStagVecGetValuesStencil(usr->dmPV,xVellocal,4,pf,vf); CHKERRQ(ierr);

        // central difference method
        fval -= 0.5*(vf[1]*(fe[2]+fe[0]) - vf[0]*(fe[1]+fe[0]))/dx[ix] + 0.5*(vf[3]*(fe[4]+fe[0]) - vf[2]*(fe[3]+fe[0]))/dz[iz];

        // the term to compensate compressiblity
        fval += fe[0] * ((vf[1]-vf[0])/dx[ix] + (vf[3]-vf[2])/dz[iz]);

      }

      xx[j][i][idx] = xxp[j][i][idx] + dt*fval;
    }
  }

  // reset sx, sz, nx, nz
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  // apply boundary conditions : zero flux
  if (sx==0) {
    for (j = sz; j<sz+nz; j++) {
    }
  }

  if (sx+nx==Nx) {
    for (j = sz; j<sz+nz; j++) {
    }
  }

  if (sz==0) {
    for (i = sx; i<sx+nx; i++) {
    }
  }

  if (sz+nz==Nz) {
    for (i = sx; i<sx+nx; i++) {
    }
  }

  // release dx dz
  ierr = PetscFree(dx);CHKERRQ(ierr);
  ierr = PetscFree(dz);CHKERRQ(ierr);

  // Restore arrays, local vectors
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  ierr = DMStagVecRestoreArray(dm,xplocal,&xxp);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xplocal,INSERT_VALUES,xprev); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xplocal,INSERT_VALUES,xprev); CHKERRQ(ierr);
  ierr = VecDestroy(&xplocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &dfxlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &dfzlocal); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(usr->dmPV, &xVellocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Interpolate corner and face values of f, for uniform grids only
// ---------------------------------------
PetscErrorCode UpdateCornerF(DM dm, Vec x, void *ctx)
{
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz;
  PetscInt       ic, il, ir, iu, id, idl, idr, iul, iur;
  PetscScalar    ***xx;
  Vec            xlocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  // Local vectors
  ierr = DMCreateLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm,xlocal,&xx); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT,    0, &ic ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, LEFT,       0, &il ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, RIGHT,      0, &ir ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN,       0, &id ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP,         0, &iu ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur); CHKERRQ(ierr);


  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[4];
      PetscScalar    fval[4];

      // collect the elements points around the down left corner
      point[0].i = i-1; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i  ; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i-1; point[2].j = j-1; point[2].loc = ELEMENT; point[2].c = 0;
      point[3].i = i  ; point[3].j = j;   point[3].loc = ELEMENT; point[3].c = 0;

      // fix the boundary cell
      if (i == 0)    {point[0].i = i; point[2].i = i;}
      if (j == 0)    {point[2].j = j; point[1].j = j;}

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 4, point, fval); CHKERRQ(ierr);

      xx[j][i][il]  = 0.5*(fval[3] + fval[0]); // left
      xx[j][i][id]  = 0.5*(fval[3] + fval[1]); // down
      xx[j][i][idl] = 0.25*(fval[0]+fval[1]+fval[2]+fval[3]); // downleft

      if (j==Nz-1) {
        xx[j][i][iu]  = xx[j][i][ic];
        xx[j][i][iul] = xx[j][i][il];
      }
      if (i==Nx-1) {
        xx[j][i][ir]  = xx[j][i][ic];
        xx[j][i][idr]  = xx[j][i][id];
      }
      if (i==Nx-1 && j==Nz-1) {
        xx[j][i][iur] = xx[j][i][ic];
      }
    }
  }

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,xlocal,&xx); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,xlocal,INSERT_VALUES,x); CHKERRQ(ierr);
  ierr = VecDestroy(&xlocal); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// ---------------------------------------
// Update volumefraction for fluid 1 within each cube between two vertically adjacent cell center
// ---------------------------------------
PetscErrorCode UpdateVolFrac(DM dm, Vec x, void *ctx)
{
  UsrData       *usr = (UsrData*) ctx;
  PetscInt       i, j, sx, sz, nx, nz, Nx, Nz, vfopt;
  PetscInt       ic, il, ir, iu, id, idl, iul, idr, iur, icenter, iprev, inext;
  PetscScalar    ***vvf, **coordx, **coordz;
  PetscScalar    eps;
  Vec            xlocal, vflocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  eps = usr->par->eps;
  vfopt = usr->par->vfopt;

  // Local vectors
  ierr = DMGetLocalVector(dm,&xlocal); CHKERRQ(ierr);
  ierr = DMGlobalToLocal (dm,x,INSERT_VALUES,xlocal); CHKERRQ(ierr);

  ierr = DMCreateLocalVector(dm, &vflocal); CHKERRQ(ierr);
  ierr = DMStagVecGetArray(dm, vflocal, &vvf); CHKERRQ(ierr);

  // Get domain corners
  ierr = DMStagGetGlobalSizes(dm, &Nx, &Nz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm, &sx, &sz, NULL, &nx, &nz, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  // Get location slot
  ierr = DMStagGetLocationSlot(dm, ELEMENT,    0, &ic ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, LEFT,       0, &il ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, RIGHT,      0, &ir ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN,       0, &id ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP,         0, &iu ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_LEFT,  0, &idl ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_LEFT,    0, &iul ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, DOWN_RIGHT, 0, &idr ); CHKERRQ(ierr);
  ierr = DMStagGetLocationSlot(dm, UP_RIGHT,   0, &iur ); CHKERRQ(ierr);

  // Get dm coordinates array
  ierr = DMStagGetProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,ELEMENT,&icenter);CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,LEFT   ,&iprev  );CHKERRQ(ierr);
  ierr = DMStagGetProductCoordinateLocationSlot(dm,RIGHT  ,&inext  );CHKERRQ(ierr);


  //if (sz==0) {sz++;nz--;}

  // Loop over local domain
  for (j = sz; j < sz+nz; j++) {
    for (i = sx; i <sx+nx; i++) {
      DMStagStencil  point[8];
      PetscScalar    ff[8], dx, dz, cc, ar;

      // collect the elements points around the down left corner
      point[0].i = i; point[0].j = j  ; point[0].loc = ELEMENT; point[0].c = 0;
      point[1].i = i; point[1].j = j-1; point[1].loc = ELEMENT; point[1].c = 0;
      point[2].i = i; point[2].j = j  ; point[2].loc = LEFT   ; point[2].c = 0;
      point[3].i = i; point[3].j = j-1; point[3].loc = LEFT   ; point[3].c = 0;

      point[4] = point[0]; point[4].loc = DOWN;
      point[5] = point[0]; point[5].loc = DOWN_LEFT;
      point[6] = point[0]; point[6].loc = UP;
      point[7] = point[0]; point[7].loc = UP_LEFT;

      if (j==0) {point[1].j = point[0].j; point[3].j = point[0].j;}

      ierr = DMStagVecGetValuesStencil(dm, xlocal, 8, point, ff); CHKERRQ(ierr);

      dz = coordz[j][inext] -  coordz[j  ][iprev];
      dx = coordx[i][inext  ] -  coordx[i  ][iprev  ];

      cc = eps/dz;
      ar = dx/dz;

      //diffuse
      if (vfopt == 0) {
        vvf[j][i][id] = ff[4];
        vvf[j][i][ic] = ff[0];
        vvf[j][i][il] = ff[2];
        vvf[j][i][idl]= ff[5];
      }

      //sharp: staggered
      if (vfopt ==1) {
        if (ff[4]>= 0.5) {vvf[j][i][id] = 1.0;}
        else             {vvf[j][i][id] = 0.0;}
        if (ff[0]>= 0.5) {vvf[j][i][ic] = 1.0;}
        else             {vvf[j][i][ic] = 0.0;}
        if (ff[2]>= 0.5) {vvf[j][i][il] = 1.0;}
        else             {vvf[j][i][il] = 0.0;}
        if (ff[5]>= 0.5) {vvf[j][i][idl]= 1.0;}
        else             {vvf[j][i][idl]= 0.0;}
      }


      //1d simplification
      if (vfopt ==2 ) {
        PetscScalar fftmp;
        fftmp = ff[4];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][id] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][id] = 0.0;}
        else    {vvf[j][i][id] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[0];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][ic] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][ic] = 0.0;}
        else    {vvf[j][i][ic] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[2];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][il] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][il] = 0.0;}
        else    {vvf[j][i][il] = 0.5 - 4.0*cc*(fftmp-0.5);}

        fftmp = ff[5];
        if      (fftmp >= 0.5+0.125/cc) {vvf[j][i][idl] = 1.0;}
        else if (fftmp <= 0.5-0.125/cc) {vvf[j][i][idl] = 0.0;}
        else    {vvf[j][i][idl] = 0.5 - 4.0*cc*(fftmp-0.5);}
      }

      //2d
      if (vfopt ==3) {
        vvf[j][i][ic] = volf_2d(ff[6], ff[4], cc, ar);
        vvf[j][i][il] = volf_2d(ff[7], ff[5], cc, ar);
        if (j>0) {
          vvf[j][i][id] = volf_2d(ff[0], ff[1], cc, ar);
          vvf[j][i][idl]= volf_2d(ff[2], ff[3], cc, ar);
        } else {
          vvf[j][i][id] = vvf[j][i][ic];
          vvf[j][i][idl] = vvf[j][i][il];
        }
      }
    }
  }

  // for nodes on the up and right boundaries
  if (sz+nz == Nz) {
    j = Nz-1;
    for (i = sx; i<sx+nx; i++) {
      vvf[j][i][iul] = vvf[j][i][il];
      vvf[j][i][iu]  = vvf[j][i][ic];
    }
  }
  if (sx+nx == Nx) {
    i = Nx-1;
    for (j = sz; j<sz+nz; j++) {
      vvf[j][i][idr] = vvf[j][i][id];
      vvf[j][i][ir]  = vvf[j][i][ic];
    }
  }
  if (sx+nx==Nx && sz+nz ==Nz) {
    i = Nx-1;
    j = Nz-1;
    vvf[j][i][iur] = vvf[j][i][ir];
  }

  // for nodes on the bottom boundary
  if (sz == 0) {
    j = 0;
    for (i = sx; i<sx+nx; i++) {vvf[j][i][idl] = vvf[j][i][il];}
    if (sx+nx==Nx) {i = Nx-1; vvf[j][i][idr] = vvf[j][i][ir];}
  }


  // Restore arrays
  ierr = DMStagRestoreProductCoordinateArraysRead(dm,&coordx,&coordz,NULL);CHKERRQ(ierr);

  // Restore and map local to global
  ierr = DMStagVecRestoreArray(dm,vflocal,&vvf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(dm,vflocal,INSERT_VALUES,usr->volf); CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd  (dm,vflocal,INSERT_VALUES,usr->volf); CHKERRQ(ierr);
  ierr = VecDestroy(&vflocal); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal ); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
