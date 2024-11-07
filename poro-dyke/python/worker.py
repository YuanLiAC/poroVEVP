# Import modules
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# Input file
fname = 'out_paper_porodyke'

print('# --------------------------------------- #')
print('# Buoyancy-driven dyke, to compare with LEFM')
print('# --------------------------------------- #')


ls = 1   #length scale
ts = 1e-3  # time scales, relative to 1myr
es = 1e5  # eta scaling with respect to 10^18
gaussian = 1

ks = 1
vs = ks**(1/3)

kscale = 1 # scaling to permeability, square of (Dx/0.01)

n = 500 #250 #125
nx = 61
H = 5 #2.5 #1.25
Hair = 0.0
eps = H/n #1.0/n #0.7/n
gamma = 1.0
vfopt = 3

xmin = -0.305
zmin = 0
L = 0.61
vi = 0 
viz = 0 

gangle = 0 #np.pi/4
sth = 0 #np.pi/8

dt = 5e-7

tstep = 100001#641#int(0.1/dt) + 1

dtck = 1e-5 /ts
maxckpt = 800
tmax = dtck * maxckpt

tscale = 0.025 /vs  #0.025 
sdel = 0.52418 *0.5 #0.051771 *0.5 # dpdz = (1-sdel)*F1, it gives q0 = 1m/yr at phim=0.2 and DeltaRhoG = 500*9.8

qin = 1 * 10

if dt >= dtck:
    dt = dtck/2.0;

tout = 1 #int((tstep-1)/4) #10
tout2 = 10

etamin = 0
phi0 = 1e-1
R = 0.25 * es**0.5 /ls * kscale**0.5 
lam = 0.0

angle = np.pi/6
R_pe = 4.0

coeff_iso = 1.0/3.0
psi = np.pi/18
iota = 0.0

eta1 = 1e7
eta2 = 0
eta3 = eta1
C1 = 1576.8 /es  /10 *ts
C2 = 1e40
C3 = 1e40
F1 = 618.1056 /es *ls *ts
F2 = 0.0
F3 = 0.0
G1 = 1576780 /es /10 *ts
G2 = G1
G3 = G1

FS = F1*6
p0 = 0 #C1

pfs =  0 #5*C1
dpfs = 0 #4*C1

R2_1 = R*R
R2_2 = 0
R2_3 = 0

phi_m  = 0.2 
phi_bg = 1e-12
xsig   = 0.0001
zsig   = 0.2
xpeak  = 0
zpeak  = 0

bcidx =  4

Z = G1 * 2

eta_vk = 1e-13 #1e-7 
zeta_vk = 200.0
phicut = 1e-6
alphaC = 1
tf_tol = 1e-8

lam_p = 1e40  #YC = (C_b or C_w)*lam_p
lam_v = phi0 #zeta_v = eta_v / lam_v
nh = 1.0

hca = 0  #relative softening of the friction angle
hcc = 0 #relative softening of cohesion
strain_max = 1e40 #strain scale for strain softening
strainK = 1e-2  #strain scale for the permeability enhancement

ck = 1e2 # maximum permeability enhancement
erange = 1 # relative viscosity range in depth
krange = 1 #1e2 # relative permeability range in depth

noise = 1e-40


# porosity means type
poromean = 0

# permeability model, (0, Kozeny-Carman, 1, customised, k = k0*phi^3 * km^(phi^kd), km = DeltaX^2/h0/3
permeability = 0
km = 1e8 *  (4*ls*L*1e3/nx)**2  /3.0
kd = 0.5

# Interpolation type for permeability enhancement on faces
kenhance = 1

# Parameters for anisotropic permeability
km_anis = 10  * ks
keps_anis = 0.05


# actual qin
#qin = qin0 * km_anis


fname = fname + '_n'+str(n)


# solver parameters
phase = ' -eps '+str(eps)+' -gamma '+str(gamma) + ' -vfopt '+str(vfopt)
model = ' -L '+str(L)+' -H '+str(H) + ' -xmin '+str(xmin)+ ' -zmin ' + str(zmin) +\
    ' -vi '+str(vi)+ ' -Hair ' + str(Hair)+ ' -viz '+ str(viz) +\
    ' -dt '+str(dt)+\
    ' -eta1 '+str(eta1) + ' -eta2 '+str(eta2) + ' -eta3 '+str(eta3) + \
    ' -C1 '+str(C1) + ' -C2 '+str(C2) + ' -C3 '+str(C3) +\
    ' -F1 ' + str(F1) + ' -F2 ' + str(F2) + ' -F3 ' + str(F3) + ' -FS ' + str(FS)+\
    ' -gangle ' + str(gangle)+ \
    ' -G1 ' + str(G1) + ' -G2 ' + str(G2) + ' -G3 ' + str(G3) +\
    ' -R2_1 ' + str(R2_1) + ' -R2_2 ' + str(R2_2) + ' -R2_3 ' + str(R2_3) +\
    ' -eta_vk ' + str(eta_vk) + ' -zeta_vk ' + str(zeta_vk)+\
    ' -angle ' + str(angle) +\
    ' -psi ' + str(psi) +\
    ' -iota ' + str(iota) +\
    ' -coeff_iso ' + str(coeff_iso) +\
    ' -R_pe ' + str(R_pe) +\
    ' -noise_max ' + str(noise) +\
    ' -pfs ' + str(pfs) + ' -dpfs ' + str(dpfs) +  ' -p0 ' + str(p0) +\
    ' -dtck ' + str(dtck) + ' -maxckpt ' +str(maxckpt) +\
    ' -phicut ' + str(phicut) + ' -alphaC ' + str(alphaC) +\
    ' -tf_tol ' + str(tf_tol) +\
    ' -hca ' + str(hca) +  ' -hcc ' + str(hcc) + ' -strain_max ' + str(strain_max) +\
    ' -erange ' + str(erange) + ' -krange ' + str(krange) + ' -ck ' + str(ck) + ' -strainK ' + str(strainK)+\
    ' -bcidx ' + str(bcidx) + ' -sth ' + str(sth) +\
    ' -permeability ' + str(permeability) + ' -km ' + str(km) + ' -kd ' + str(kd)  + \
    ' -poromean ' + str(poromean) + ' -gaussian ' + str(gaussian) + ' -kenhance ' + str(kenhance) + \
    ' -sdel ' + str(sdel) + ' -tscale ' + str(tscale) + \
    ' -km_anis ' + str(km_anis) + ' -keps_anis ' + str(keps_anis) + \
    ' -qin ' + str(qin) + ' -kscale ' + str(kscale)
    


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
  ' -ksp_type fgmres' + \
  ' -pc_factor_mat_ordering_type external'  #this one is used for petsc 3.14
#' -python_snes_failed_report' 

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
str1 = '../paper_porodyke.app' + \
       ' -nx '+str(nx)+' -nz '+str(n)+' -tstep '+str(tstep) + ' -tmax '+str(tmax) +\
       ' -ts_scheme 2 -adv_scheme 1' +\
       newton + model + phase + solver + sdpar + \
       ' -output_file '+fname+' -tout '+str(tout)
print(str1)
os.system(str1)
