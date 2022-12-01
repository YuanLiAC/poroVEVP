#include "petsc.h"


static PetscScalar tau_ini(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar A5, PetscScalar xp)
{ PetscScalar A6, result;
  A6 = A3 + 1.0;
  result = -A5 + PetscPowScalar(A5*A5 + 4.0*A1*A3*A6*xp , 0.5 );
  result = 0.5*result/A6;
  return(result);
}

static PetscScalar ftau(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar xp, PetscScalar x)
{ PetscScalar aa, result;
  //aa = PetscPowScalar(x*x+A1*A1, 0.5);
  aa = PetscPowScalar(x*x+A1*A1, 0.5);
  //result = A3*xp*aa/x - (1+A3)*aa + A4;
  result = aa/x - (1/A3+1)*aa + A4/xp;
  return(result);
}

static PetscScalar dftau(PetscScalar A1, PetscScalar A2, PetscScalar A3, PetscScalar A4, PetscScalar xp, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x+A1*A1, 0.5);
  //result = A3*xp*(1/aa - aa/(x*x)) - (1+A3)*x/aa;
  result = (1/aa - aa/(x*x)) - (1/A3+1)*x/aa;
  return(result);
}

static PetscScalar dptau(PetscScalar xp[2], PetscScalar eve, PetscScalar zve, PetscScalar phi, PetscScalar A1, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + A1*A1, 0.5);
  result = xp[1] + zve/eve * (aa*xp[0]/x - aa) * PetscSinScalar(phi);
  return(result);
}

static PetscScalar lamtau(PetscScalar xp, PetscScalar eve, PetscScalar A1, PetscScalar x)
{ PetscScalar aa, result;
  aa = PetscPowScalar(x*x + A1*A1, 0.5);
  result = (aa*xp/x - aa)/eve;
  return(result);
}

PetscScalar ResF(PetscScalar A1, PetscScalar A2, PetscScalar evp, PetscScalar phi, PetscScalar x[3])
{ PetscScalar aa, result;
  aa = PetscPowScalar(x[0]*x[0] + A1*A1, 0.5);
  result = aa - (A2 + x[1]*PetscSinScalar(phi)) - evp*x[2];
  return(result);
}

PetscScalar ResT(PetscScalar A1, PetscScalar eve, PetscScalar xp[2], PetscScalar x[3])
{ PetscScalar aa, result;

  aa = PetscPowScalar(x[0]*x[0] + A1*A1, 0.5);
  result = x[0] - (xp[0] - eve*x[2]*x[0]/aa);
  return(result);
}

PetscScalar ResP(PetscScalar zve, PetscScalar phi, PetscScalar xp[2], PetscScalar x[3])
{ PetscScalar result;
  result = x[1] - (xp[1] + zve*x[2]*PetscSinScalar(phi));
  return(result);
}

// ---------------
// tau_a - prepare the constant parameter for the local Newton iteratives
// ----------------
static PetscErrorCode tau_a(PetscScalar c, PetscScalar ct, PetscScalar ap, PetscScalar phi, PetscScalar eve, PetscScalar zve, PetscScalar evp, PetscScalar xp[2], PetscScalar a[5])
{
  PetscFunctionBegin;
  a[0] = c*PetscCosScalar(phi) - ct*PetscSinScalar(phi);
  a[1] = c*PetscCosScalar(phi) + ap*PetscSinScalar(phi);
  a[2] = (zve*PetscPowScalar(PetscSinScalar(phi),2) + evp)/eve;
  a[3] = a[1] + xp[1] * PetscSinScalar(phi);
  a[4] = a[0]*(a[2] +1.0) - a[3] - a[2]*xp[0];
  PetscFunctionReturn(0);
}

// ---------------------------------------
//VEVP_hyper_tau: find the solution of tauII
// - a[5], five constants in the local Newton iterative, the output from tau_a
// - xp[2], prediction value of tauII = xp[0] and Dp = xp[1]
// - tf_tol, function tolerance of the local Newton iterative
// - Nmax,   the maximum newton iteratives
//
// Output:
// - xsol, the solution
// ---------------------------------------
static PetscScalar VEVP_hyper_tau(PetscScalar a[5], PetscScalar xp[2], PetscScalar tf_tol, PetscInt Nmax)
{
  PetscInt          ii=0;
  PetscScalar       xn,xn1, f, df, dx = 1e40;

  // prepare the initial guess
  xn = tau_ini(a[0], a[1], a[2], a[3], a[4], xp[0]);
  //xn1 = xn;

  // if a[2] (A3) is nonzero, do the rescaling
  if (PetscAbs(a[2]) > 1e-20) {
  // rescale xn and A1 by xp[0], A4 by A3
  xn = xn/xp[0];
  a[0] = a[0]/xp[0];
  a[3] = a[3]/a[2];

  xn1 = xn;

  f = ftau(a[0], a[1], a[2],a[3], xp[0], xn);

//  PetscPrintf(PETSC_COMM_WORLD, "CHECK xn = %1.6f, xn1 = %1.6f, f = %1.6f \n", xn, xn1, f);

  while (PetscAbs(f)> tf_tol && dx/xn > tf_tol && ii < Nmax) {

    df = dftau(a[0], a[1], a[2], a[3], xp[0], xn);
    dx = -f/df;

    while (PetscAbs(dx) > PetscAbs(xn)) {dx = 0.5*dx;} // if dx is too large, take half step

    xn1 = xn + dx;

//    PetscPrintf(PETSC_COMM_WORLD,"VEVP_hyper_tau: Iteration ii = %d, F = %1.6e, dF = %1.6e, xn = %1.6f, dx = %1.6f\n", ii, f, df, xn, dx);

    f = ftau(a[0], a[1], a[2],a[3], xp[0], xn1);
    xn = xn1;
    ii += 1;
  }

  if (ii>=Nmax) {
    PetscPrintf(PETSC_COMM_WORLD,"DIVERGENCE: VEVP_hyper_tau, max iteration reaches. ii = %d, Nmax = %d, F = %1.6e, dx = %1.6f, xn1=%1.6f \n", ii, Nmax, f, dx, xn1);
    PetscPrintf(PETSC_COMM_WORLD,"----------A1 = %1.6f, A2 = %1.6f, A3 = %1.6f, A4 = %1.6f, A5 =%1.6f \n ", a[0], a[1], a[2], a[3], a[4]);
    PetscPrintf(PETSC_COMM_WORLD,"----------tau_p = %1.6f, p_p = %1.6f", xp[0], xp[1]);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "FORCE STOPPED! VEVP_hyper_tau is divergent");
  }
  else {
//    PetscPrintf(PETSC_COMM_WORLD,"Convergence: VEVP_hyper_tau. Total iteration. ii = %d, F = %1.6e, xn1 = %1.6f \n", ii, f, xn1);
  }

  // scale back
  xn = xn*xp[0];
  a[0] = a[0]*xp[0];
  a[3] = a[3]*a[2];
  }
  else {
    // if a[2] (A3) is zero, tau = (A4**2 - a1**2)**0.5
    xn = PetscPowScalar(a[3]*a[3] - a[0]*a[0], 0.5);
  }

  return xn;
}



// drive code to return tau_II, dp and dotlam
// input: c, ct, ap, phi, eve, zve, evp, xp[0] = taup, x[1] = delta pp
// output: xsol[0] = tauii, xsol[1] = delta p, xsol[2] = dotlam
// case 1: f < 0, tauii = tp, deltap = pp, dotlam = 0.0
// case 2: f > 0 and xp[0] = 0, analytical solution with tauII =0
// case 3: f > 0 and xp[0] != 0, numerical solution, call VEVP_hyper_tau, dptau and lamtau
PetscErrorCode VEVP_hyper_sol(PetscInt Nmax, PetscScalar tf_tol,
                              PetscScalar c, PetscScalar ct, PetscScalar ap, PetscScalar phi, PetscScalar eve, PetscScalar zve, PetscScalar evp, PetscScalar xp[2],
                              PetscScalar a[5], PetscScalar xsol[3])
{ PetscScalar     ff;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  // prepare the constant parameter in the equation solving tauII
  ierr =  tau_a(c, ct, ap, phi, eve, zve, evp, xp, a); CHKERRQ(ierr);

  //PetscPrintf(PETSC_COMM_WORLD, "A1= %1.6f, A2=%1.6f, A3=%1.6f, A4=%1.6f, A5=%1.6f \n", a[0], a[1], a[2], a[3], a[4]);

  // check if it yields
  xsol[0] = xp[0];
  xsol[1] = xp[1];
  xsol[2] = 0.0;
  ff = ResF(a[0], a[1], evp, phi, xsol);

  // if yield, compute tauII, then dp and dotlam
  if (ff>0) {

    if (xsol[0] < tf_tol) {
      xsol[0] = 0.0;
      xsol[2] = ((a[0]-a[1]) - xp[1]*PetscSinScalar(phi))/(zve*PetscPowScalar(PetscSinScalar(phi),2) + evp);
      xsol[1] = xp[1] + zve * xsol[2] * PetscSinScalar(phi);
    } else {
      xsol[0] = VEVP_hyper_tau(a, xp, tf_tol, Nmax); CHKERRQ(ierr);
      xsol[1] = dptau(xp, eve, zve, phi, a[0], xsol[0]);
      xsol[2] = lamtau(xp[0], eve, a[0], xsol[0]);
    }

    //PetscPrintf(PETSC_COMM_WORLD, "TEST, tau = %1.6f, dp = %1.6f, lam = %1.6f\n", xsol[0], xsol[1], xsol[2]);
  }


  PetscFunctionReturn(0);
}



