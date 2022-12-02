# poroVEVP
source codes for paper poroVEVP, need PETSC and FDPDE

# How to use

## Install FD-PDE
https://github.com/apusok/FD-PDE 

## Install petsc 3.14 as suggested in FD-PDE Readme.md 
git clone -b release https://gitlab.com/petsc/petsc.git petsc \
cd petsc \
git checkout v3.14 \
<config,install> 

Also make sure anaconda3 (or other python3) is installed.

## Clone poroVEVP in FD-PDE/models

## Compile poroVEVP 
'make all' in FD-PDE/models/poroVEVP

## Run shear/tensile/rifting models in the GJI paper
Python scripts in FD-PDE/models/poroVEVP/python


