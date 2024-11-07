# poroVEVP
Source codes for GJI manuscript poroVEVP; 
Source codes for GMD manuscript poro-dyke;
requirements [PETSc](https://petsc.org/release/) and [FD-PDE](https://github.com/apusok/FD-PDE)

## License: MIT License

# How to use

## Clone FD-PDE framework
`git clone https://github.com/apusok/FD-PDE FD-PDE`

## Install PETSc 3.14 as suggested in FD-PDE Readme.md 
`git clone -b release https://gitlab.com/petsc/petsc.git petsc` \
`cd petsc` \
`git checkout v3.14` \

Use `config` options as in `FD-PDE/Readme.md`.
<config,install> 

Also make sure [anaconda3](https://www.anaconda.com/products/distribution) (or other python3) is installed.

Set paths
`export PETSC_DIR=<PATH>` \
`export PYTHONPATH=<PATH_FDPDE>/utils:${PETSC_DIR}/lib/petsc/bin`

## Clone poroVEVP in `FD-PDE/`
`git clone https://github.com/YuanLiAC/poroVEVP poroVEVP`

## Compile poroVEVP 
`cd poroVEVP/poroVEVP` \
`make all`

## Run shear, tensile and rifting models in the GJI paper
In `FD-PDE/poroVEVP/poroVEVP/python` run with
* `python run_shear.py`
* `python run_tensile.py`
* `python run_rift.py`

