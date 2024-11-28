# poroVEVP
This repository contains poroVEVP model and its benchmark code. Two folders are:

* poroVEVP: the original poroVEVP model. Source codes for **GJI** manuscript [Y Li et al. 2023](https://doi.org/10.1093/gji/ggad173).
* poro-dyke: further benchmark of poroVEVP model by comparing with LEFM model. Source codes for **GMD** manuscript (submitted);


This repository has dependency: [PETSc](https://petsc.org/release/) and [FD-PDE](https://github.com/apusok/FD-PDE)

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

## To reproduce computations in [Y Li et al. 2023](https://doi.org/10.1093/gji/ggad173).

### Compile poroVEVP 
`cd poroVEVP/poroVEVP` \
`make all`

### Run shear, tensile and rifting models in the GJI paper
In `FD-PDE/poroVEVP/poroVEVP/python` run with
* `python run_shear.py`
* `python run_tensile.py`
* `python run_rift.py`


## To reproduce computations in GMD submission 

### Compile poro-dyke 
`cd poroVEVP/poro-dyke` \
`make all`

### Run shear, tensile and rifting models in GMD paper
In `FD-PDE/poroVEVP/poro-dyke/python` run with
* `python work.py`

### DATA extraction
Run Juypter notebook: `porodyke_loaddata.ipynb`

Example parameters have been added. Users need to modify parameters as needed.

### DATA process and produce figures for GMD paper

Run Juypter notebook: `PoroDyke_BMD.ipynb`



