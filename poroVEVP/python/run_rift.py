# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file for the refernece mode of the tectonic-scale model, non-dim values
# eta = 1e20 - 1e24. eta_vk = 1e19
# permeability enhancement 100x
fname = 'out_vevp_rift'

print('# --------------------------------------- #')
print('# Tectonic-scal model')
print('# --------------------------------------- #')


es = 1  # scaling for other parameters. eta scale = 1e21

n = 200
nx = 201
H = 1.0
Hair = 0.2
eps = H/n 
gamma = 1.0
vfopt = 3

xmin = -0.5
zmin = 0
L = 1
vi = 0.15768

dt = 5e-6

tstep = 100001

dtck = 1e-5 #time checkpoint
maxckpt = 350 # number of checkpoints
tmax = dtck * maxckpt

if dt >= dtck:
    dt = dtck/2.0;

tout = 1 
tout2 = 10

etamin = 0
phi0 = 1e-1
R = 0.31623 * es**0.5
lam = 0.0

angle = np.pi/6
R_pe = 4.0

coeff_iso = 1.0/3.0
psi = np.pi/18
iota = 0.0

eta1 = 1e-1
eta2 = 1e-1
eta3 = 1e-6
C1 = 15.768 /es
C2 = C1
C3 = C1
F1 = 154.5264 /es
F2 = F1
F3 = 0.0
G1 = 15768 /es
G2 = G1
G3 = G1

FS = F1*6

pfs = 0

R2_1 = R*R
R2_2 = R*R
R2_3 = 0

phi_m  = 0.5
phi_bg = 1e-12
xsig   = 0.02
zsig   = 0.06
xpeak  = 0.0
zpeak  = 0.25

Z = G1/10

eta_vk = 0.01 
zeta_vk = 200.0
phicut = 1e-4
alphaC = 1
tf_tol = 1e-8

lam_p = 1e40  
lam_v = phi0 
nh = 1.0

hca = 0  #relative softening of the friction angle
hcc = 0.5 #relative softening of cohesion
strain_max = 0.01 #1e40 #strain scale for strain softening
strainK = 0.01  #strain scale for the permeability enhancement

ck = 1e2 # maximum permeability enhancement
erange = 1e4 # relative viscosity range in depth
krange = 1 #1e2 # relative permeability range in depth


fname = fname + '_n'+str(n)


# solver parameters
phase = ' -eps '+str(eps)+' -gamma '+str(gamma) + ' -vfopt '+str(vfopt)
model = ' -L '+str(L)+' -H '+str(H) + ' -xmin '+str(xmin)+ ' -zmin ' + str(zmin) +\
    ' -vi '+str(vi)+ ' -Hair ' + str(Hair)+\
    ' -dt '+str(dt)+\
    ' -eta1 '+str(eta1) + ' -eta2 '+str(eta2) + ' -eta3 '+str(eta3) + \
    ' -C1 '+str(C1) + ' -C2 '+str(C2) + ' -C3 '+str(C3) +\
    ' -F1 ' + str(F1) + ' -F2 ' + str(F2) + ' -F3 ' + str(F3) + ' -FS ' + str(FS)+\
    ' -G1 ' + str(G1) + ' -G2 ' + str(G2) + ' -G3 ' + str(G3) +\
    ' -R2_1 ' + str(R2_1) + ' -R2_2 ' + str(R2_2) + ' -R2_3 ' + str(R2_3) +\
    ' -eta_vk ' + str(eta_vk) + ' -zeta_vk ' + str(zeta_vk)+\
    ' -angle ' + str(angle) +\
    ' -psi ' + str(psi) +\
    ' -iota ' + str(iota) +\
    ' -coeff_iso ' + str(coeff_iso) +\
    ' -R_pe ' + str(R_pe) +\
    ' -pfs ' + str(pfs) +\
    ' -dtck ' + str(dtck) + ' -maxckpt ' +str(maxckpt) +\
    ' -phicut ' + str(phicut) + ' -alphaC ' + str(alphaC) +\
    ' -tf_tol ' + str(tf_tol) +\
    ' -hca ' + str(hca) +  ' -hcc ' + str(hcc) + ' -strain_max ' + str(strain_max) +\
    ' -erange ' + str(erange) + ' -krange ' + str(krange) + ' -ck ' + str(ck) + ' -strainK ' + str(strainK)


newton = ' -pc_factor_mat_solver_type umfpack'+ \
  ' -pc_type lu'+ \
  ' -snes_linesearch_type bt'+ \
  ' -snes_linesearch_monitor'+ \
  ' -snes_atol 1e-8'+ \
  ' -snes_rtol 1e-10'+ \
  ' -snes_stol 1e-8'+ \
  ' -snes_max_it 10'+ \
  ' -snes_monitor'+ \
  ' -snes_view'+ \
  ' -snes_monitor_true_residual'+ \
  ' -ksp_monitor_true_residual'+ \
  ' -ksp_gmres_restart 300' + \
  ' -ksp_monitor_singular_value' + \
  ' -snes_converged_reason'+ \
  ' -ksp_converged_reason'+ \
  ' -python_snes_failed_report'+ \
  ' -ksp_type fgmres' + \
  ' -pc_factor_mat_ordering_type external'  #this one is used for petsc 3.14

#solver= ' -snes_mf_operator'
#solver = ' -fp_trap'
solver = ' '

sdpar = ' -R ' + str(R)+ \
  ' -phi_0 ' + str(phi0)+ \
  ' -lambda ' + str(lam)+ \
  ' -Z ' + str(Z) +\
  ' -lam_p '+str(lam_p)+ \
  ' -lam_v '+str(lam_v)+ \
  ' -etamin '+str(etamin)+ \
  ' -nh ' +str(nh)+ \
  ' -phi_m ' +str(phi_m)+ \
  ' -phi_bg ' +str(phi_bg)+ \
  ' -xsig ' +str(xsig)+ \
  ' -zsig ' +str(zsig)+ \
  ' -xpeak ' +str(xpeak) +\
  ' -zpeak ' +str(zpeak)
# Run test
# Forward euler
str1 = '../vevp_rift.app' + \
       ' -nx '+str(nx)+' -nz '+str(n)+' -tstep '+str(tstep) + ' -tmax '+str(tmax) +\
       ' -ts_scheme 2 -adv_scheme 1' +\
       newton + model + phase + solver + sdpar + \
       ' -output_file '+fname+' -tout '+str(tout)
print(str1)
os.system(str1)
