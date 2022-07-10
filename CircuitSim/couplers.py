#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 22:10:17 2022

"""
# Need to limit numpy & scipy use of multithreading
# to prepare for scqubits use of multiprocessing

# Multithreading broken in scqubits for CustomCircuits
# so the code below commented out for now.


# For multithreading in scqubits, run below to limit numpy and 
# scipy use of threads.
# import os

# NUM_THREADS = "1"

# os.environ["OMP_NUM_THREADS"] = NUM_THREADS
# os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
# os.environ["MKL_NUM_THREADS"] = NUM_THREADS
# os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
# os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import numpy as np
import scqubits as scqub
import SQcircuit as sqcirc
import scipy.constants as const
import matplotlib.pyplot as plt
from copy import deepcopy

# scqub.settings.NUM_CPUS = 1
# scqub.settings.MULTIPROC = 'pathos'


h = const.h
e = const.e
phi0 = h/(2*const.e)
phi0_red = phi0/2/np.pi
GHz = const.giga
MHz = const.mega
    
# SC qubits split junction transmon as CustomCircuit
tmon_yaml = """# transmon circuit
branches:
- ["JJ", 0, 1, EJ=12,10]
- ["JJ", 1, 0, EJ,10]
- ["C", 1, 0, EC=0.2]
"""

tmon = scqub.Circuit.from_yaml(tmon_yaml, from_file=False,
                               ext_basis="discretized")

spec = tmon.eigenvals()
print(f"Eigenvals = {spec-spec[0]}")

flux_array = np.linspace(-1,1,101)

fig, ax=tmon.plot_evals_vs_paramvals('Φ1', flux_array, 
                             evals_count=5, subtract_ground=True)

fig, ax=tmon.get_spectrum_vs_paramvals('Φ1', flux_array, 
                             evals_count=5, subtract_ground=True)

# Scqubits static capacitive
cap_coupled_tmon = """# transmon circuit
branches:
- ["JJ", 0, 1, EJ=12,10]
- ["JJ", 1, 0, EJ,10]
- ["C", 1, 0, EC=0.2]
- ["C", 1, 2, E_coup=2]
- ["JJ", 0, 2, EJ,10]
- ["JJ", 2, 0, EJ,10]
- ["C", 2, 0, EC]
"""


cap_tmons = scqub.Circuit.from_yaml(cap_coupled_tmon, from_file=False,
                               ext_basis="discretized")

flux1 = 'Φ1'
flux2 = 'Φ2'

setattr(cap_tmons, flux2, 0.4)


flux_array = np.linspace(0,0.45,101)

fig, ax=cap_tmons.plot_evals_vs_paramvals('Φ1', flux_array, 
                             evals_count=5, subtract_ground=True)

spec = cap_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                           evals_count=5,
                                           subtract_ground=True)

gap = np.abs(spec.energy_table[:,2]-spec.energy_table[:,1])

cap_list = np.linspace(0.4,4,26)
gap_array = []
for cap in cap_list:
    param_dict = {'E_coup':cap}
    cap_tmons.set_params(**param_dict)
    spec = cap_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                               evals_count=5,
                                               subtract_ground=True)

    gap = np.abs(spec.energy_table[:,2]-spec.energy_table[:,1])
    gap_array.append(np.min(gap))
 
caps = const.e**2/(2*cap_list*const.h*const.giga)
plt.figure()
plt.plot(caps/const.femto, np.array(gap_array)/2)
plt.xlabel('Coupling Cap (fF)')
plt.ylabel(r'g/2$\pi$ (GHz)')
plt.show()

# scqubits static inductive
# Doesn't work very well, I think 
# adjustments need to be made to Ej 
# to compensate for inductive coupling

ind_coupled_tmon = """# transmon circuit
branches:
- ["JJ", 2, 1, EJ=16,10]
- ["JJ", 1, 2, EJ,10]
- ["C", 1, 0, EC=0.2]
- ["L", 0, 2, E_coup=2]
- ["C", 0,2,1]
- ["JJ", 3, 2, EJ,10]
- ["JJ", 2, 3, EJ,10]
- ["C", 3, 0, EC]
"""

ind_tmons = scqub.Circuit.from_yaml(ind_coupled_tmon, from_file=False,
                               ext_basis="discretized")

flux1 = 'Φ1'
flux2 = 'Φ2'

setattr(ind_tmons, flux2, 0.1)

flux_array = np.linspace(0,.45,51)

fig, ax=ind_tmons.plot_evals_vs_paramvals('Φ1', flux_array, 
                             evals_count=5, subtract_ground=True)

spec = ind_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                           evals_count=5,
                                           subtract_ground=True)

gap = np.abs(spec.energy_table[:,2]-spec.energy_table[:,1])
print(np.min(gap))

ind_list = np.linspace(0.2,2.2,11)
gap_array = []
for ind in ind_list:
    param_dict = {'E_coup':ind}
    ind_tmons.set_params(**param_dict)
    spec = ind_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                               evals_count=5,
                                               subtract_ground=True)

    gap = np.abs(spec.energy_table[:,2]-spec.energy_table[:,1])
    gap_array.append(np.min(gap))
 
plt.figure()
plt.plot(ind_list, np.array(gap_array)/2)
plt.xlabel('INductive coupling energy')
plt.ylabel(r'g/2$\pi$ (GHz)')
plt.show()

#scqubits coupler sim
# Just taking a peak at the 
# coupler modes as its own device
coupler_yaml = """# coupler circuit
branches:
- ["JJ", 0, 1, EJc=36,10]
- ["JJ", 1, 0, EJc,10]
- ["C", 1, 0, EC2=0.1]
"""


coupler = scqub.Circuit.from_yaml(coupler_yaml, from_file=False,
                               ext_basis="discretized")


flux_array = np.linspace(0,1,101)

fig, ax=coupler.plot_evals_vs_paramvals('Φ1', flux_array, 
                             evals_count=2, subtract_ground=True)


# scqubits tunable capacitive
# Be careful using symbolic variables 
# for a circuit this size.
# Work arounds: remove any symbolic variables
# Add in a floating node (instead of 0, the gnd node)
# to bump number of explicit nodes up to 4.
# 4 node circuits forego symbolic matrix creation
# to save time, since they're expected to be huge.
tcap_coupled_tmon = """# transmon circuit
branches:
- ["JJ", 0, 1, 12,10]
- ["JJ", 1, 0, 12,10]
- ["C", 1, 0, 0.2]
- ["C", 1, 2, 5]
- ["JJ", 0, 2, 36,10]
- ["JJ", 2, 0, 36,10]
- ["C", 2, 0, 0.1]
- ["C", 2, 3, 5]
- ["JJ", 0, 3, 12,10]
- ["JJ", 3, 0, 12,10]
- ["C", 3, 0, 0.2]
"""

tcap_tmons = scqub.Circuit.from_yaml(tcap_coupled_tmon, from_file=False,
                               ext_basis="discretized")

flux1 = 'Φ1' #qubit 1
flux2 = 'Φ2'# coupler
flux3 = 'Φ3' # qubit2

setattr(tcap_tmons, flux3, 0.3)
setattr(tcap_tmons, flux2, 0.3)

flux_array = np.linspace(0.20,.4,51)

fig, ax=tcap_tmons.plot_evals_vs_paramvals('Φ1', flux_array, 
                             evals_count=5, subtract_ground=True)

spec = tcap_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                           evals_count=5,
                                           subtract_ground=True)

gap = spec.energy_table[:,2]-spec.energy_table[:,1]
print(np.min(gap)/2)

plt.figure(figsize=(10,8))
plt.plot(flux_array, spec.energy_table[:,2], label=r'|10$\rangle$')
plt.plot(flux_array, spec.energy_table[:,1], label=r'|01$\rangle$')
plt.xlabel('$\Phi$1 ($\Phi_0$)', fontsize=24)
plt.ylabel('Energy (hGHz)', fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(fontsize=22)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
plt.plot(flux_array, spec.energy_table[:,2]-spec.energy_table[:,1])
plt.xlabel('$\Phi$1 ($\Phi_0$)', fontsize=24)
plt.ylabel('Energy (hGHz)', fontsize=24)
plt.tick_params(labelsize=20)
plt.legend(fontsize=22)
plt.tight_layout()
plt.show()


flux_array = np.linspace(0.20,.4,51)
flux2_array = np.linspace(0,1,51)

setattr(tcap_tmons, flux3, 0.3)
setattr(tcap_tmons, flux2, 0)
setattr(tcap_tmons, flux1, 0)


gaps=np.zeros((len(flux2_array), len(flux_array)))
for idx,f2 in enumerate(flux2_array):
    setattr(tcap_tmons, flux2, f2)
    spec = tcap_tmons.get_spectrum_vs_paramvals(flux1,flux_array,
                                               evals_count=5,
                                               subtract_ground=True,
                                               num_cpus=1)

    gap = spec.energy_table[:,2]-spec.energy_table[:,1]
    gaps[idx,:]=gap
    
plt.figure(figsize=(10,8))
plt.plot(flux2_array,np.min(gaps, axis=1)/2, 'bo-', linewidth=3,
         markersize=10)
plt.ylabel('$g/2\pi$ (GHz)', fontsize=24)
plt.xlabel('Coupler Flux ($\Phi_0$)', fontsize=24)
plt.tick_params(labelsize=22)
plt.show()

plt.figure()
plt.plot(flux2_array, gaps[20,:])
plt.show()


# try with SQCircuit
#SQcircuit transmon-transmon cap coupled
init_flux1=0
init_flux2=0.4
C = 97 #fF
C_coup = 10 #fF
Ej = 24 #GHz
ng = 0.5
ncut = 30

sqcirc.set_unit_cap('fF')
sqcirc.set_unit_JJ('GHz')

loop1 = sqcirc.Loop(value=init_flux1)
loop2 = sqcirc.Loop(value=init_flux2)
C1 = sqcirc.Capacitor(value=C)
JJ1 = sqcirc.Junction(value=Ej/2, loops=[loop1])
JJ2 = sqcirc.Junction(value=Ej/2, loops=[loop1])

C2 = sqcirc.Capacitor(value=C)
JJ3 = sqcirc.Junction(value=Ej/2, loops=[loop2])
JJ4 = sqcirc.Junction(value=Ej/2, loops=[loop2])

C3 = sqcirc.Capacitor(value=C_coup)
elements1 = {(0,1): [C1,JJ1,JJ2],
             (1,2): [C3],
             (2,0): [C2,JJ3,JJ4]}


qubs = sqcirc.Circuit(elements1)
qubs.set_trunc_nums([ncut,ncut])
qubs.set_charge_offset(1,ng)
qubs.set_charge_offset(2,ng)


# Set number of eig vals and sweep flux param of tmon1 thru tmon2

n_eig = 6

tmon1_fluxes = np.linspace(0.35,0.45,101)


eigval_array = np.zeros((n_eig,len(tmon1_fluxes)))
for index, f in enumerate(tmon1_fluxes):
    loop1.set_flux(f)
    efreqs, _ = qubs.diag(n_eig=n_eig)
    eigval_array[:,index]=efreqs-efreqs[0]
    
plt.figure(figsize=(8,6))

plt.plot(tmon1_fluxes, eigval_array[1,:], label=r'|10$\rangle$')
plt.plot(tmon1_fluxes, eigval_array[2,:], label=r'|01$\rangle$')

plt.xlabel('Tmon1 Flux ($\Phi_0$)', fontsize=24)
plt.ylabel('Energies (GHz)', fontsize=24)
plt.tick_params(labelsize=22)
# plt.title(fr'Coupled T')
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()

cap_list = np.linspace(1,50,10)
tmon1_fluxes = np.linspace(0.35,0.45,21)

g_array = np.zeros_like(cap_list)
for cap_index, c in enumerate(cap_list):
    C3 = sqcirc.Capacitor(value=c)
    elements1 = {(0,1): [C1,JJ1,JJ2],
                 (1,2): [C3],
                 (2,0): [C2,JJ3,JJ4]}
    
    
    qubs = sqcirc.Circuit(elements1)
    qubs.set_trunc_nums([ncut,ncut])
    qubs.set_charge_offset(1,ng)
    qubs.set_charge_offset(2,ng)

    eigval_array = np.zeros((n_eig,len(tmon1_fluxes)))
    for index, f in enumerate(tmon1_fluxes):
        loop1.set_flux(f)
        efreqs, _ = qubs.diag(n_eig=n_eig)
        eigval_array[:,index]=efreqs-efreqs[0]
        
    g_array[cap_index] = np.min(eigval_array[2,:]-eigval_array[1,:])
    
plt.figure()
plt.plot(cap_list, 1000*g_array/2)
plt.title('g vs coupling cap', fontsize=26)
plt.ylabel('g/$2\pi$ (MHz)', fontsize=24)
plt.xlabel('Coupling Cap (fF)', fontsize=24)
plt.tick_params(labelsize=20)
plt.tight_layout()
plt.show()
    

# SQcircuit tmon-tmon inductively coupled

# try with SQCircuit
#SQcircuit transmon-transmon cap coupled
init_flux1=0
init_flux2=0.3
C = 97 #fF
L_coup = 200 #pH
M = 50
Ej = 24 #GHz
ng = 0
ncut = 30

sqcirc.set_unit_cap('fF')
sqcirc.set_unit_ind('pH')
sqcirc.set_unit_JJ('GHz')

loop1 = sqcirc.Loop(value=init_flux1)
loop2 = sqcirc.Loop(value=init_flux2)
C1 = sqcirc.Capacitor(value=C)
JJ1 = sqcirc.Junction(value=Ej/2, loops=[loop1])
JJ2 = sqcirc.Junction(value=Ej/2, loops=[loop1])

C2 = sqcirc.Capacitor(value=C)
JJ3 = sqcirc.Junction(value=Ej/2, loops=[loop2])
JJ4 = sqcirc.Junction(value=Ej/2, loops=[loop2])

L1 = sqcirc.Inductor(value=L_coup-M)
L2 = sqcirc.Inductor(L_coup-M)
mut = sqcirc.Inductor(value=M)
elements1 = {(0,1): [C1],
             (1,2): [JJ1,JJ2],
             (2,3): [L1],
             (3,0): [mut],
             (3,4): [L2],
             (4,5): [JJ3, JJ4],
             (5,0): [C2]}


qubs = sqcirc.Circuit(elements1)
qubs.set_trunc_nums([3,4,4,ncut,ncut])
qubs.set_charge_offset(4,ng)
qubs.set_charge_offset(5,ng)


# Set number of eig vals and sweep flux param of tmon1 thru tmon2

n_eig = 6

tmon1_fluxes = np.linspace(0.25,0.35,26)


eigval_array = np.zeros((n_eig,len(tmon1_fluxes)))
for index, f in enumerate(tmon1_fluxes):
    loop1.set_flux(f)
    efreqs, _ = qubs.diag(n_eig=n_eig)
    eigval_array[:,index]=efreqs-efreqs[0]
    
plt.figure(figsize=(8,6))

plt.plot(tmon1_fluxes, eigval_array[1,:], label=r'|10$\rangle$')
plt.plot(tmon1_fluxes, eigval_array[2,:], label=r'|01$\rangle$')

plt.xlabel('Tmon1 Flux ($\Phi_0$)', fontsize=24)
plt.ylabel('Energies (GHz)', fontsize=24)
plt.tick_params(labelsize=22)
# plt.title(fr'Coupled T')
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()

# SQcircuit with tcap coupled tmons
# I think something is wrong, this one takes a 
# loooooong time. Possibly missing some trick in
# SQcircuit to make it faster?

init_flux1=0
init_flux2=0
init_flux3=0.2
C = 97 #fF
C2 = 200 # fF
C_coup = 4
Ej = 24 #GHz
ng = 0
ncut = 30

sqcirc.set_unit_cap('fF')
sqcirc.set_unit_ind('pH')
sqcirc.set_unit_JJ('GHz')

loop1 = sqcirc.Loop(value=init_flux1)
loop2 = sqcirc.Loop(value=init_flux2)
loop3 = sqcirc.Loop(value=init_flux3)
C1 = sqcirc.Capacitor(value=C)
JJ1 = sqcirc.Junction(value=Ej/2, loops=[loop1])
JJ2 = sqcirc.Junction(value=Ej/2, loops=[loop1])

C2 = sqcirc.Capacitor(value=C2)
JJ3 = sqcirc.Junction(value=Ej, loops=[loop2])
JJ4 = sqcirc.Junction(value=Ej, loops=[loop2])

C3 = sqcirc.Capacitor(value=C)
JJ5 = sqcirc.Junction(value=Ej/2, loops=[loop3])
JJ6 = sqcirc.Junction(value=Ej/2, loops=[loop3])

Cc1 = sqcirc.Capacitor(value=C_coup)
Cc2 = sqcirc.Capacitor(value=C_coup)


elements1 = {(0,1): [C1,JJ1, JJ2],
             (1,2): [Cc1],
             (2,0): [C2, JJ3, JJ4],
             (2,3): [Cc2],
             (3,0): [C3, JJ5, JJ6]}


qubs = sqcirc.Circuit(elements1)
qubs.set_trunc_nums([ncut,ncut,ncut])
qubs.set_charge_offset(1,ng)
qubs.set_charge_offset(2,ng)
qubs.set_charge_offset(3,ng)


n_eig = 6

tmon1_fluxes = np.linspace(0,0.4,26)


eigval_array = np.zeros((n_eig,len(tmon1_fluxes)))
for index, f in enumerate(tmon1_fluxes):
    loop1.set_flux(f)
    efreqs, _ = qubs.diag(n_eig=n_eig)
    eigval_array[:,index]=efreqs-efreqs[0]
    
plt.figure(figsize=(8,6))

plt.plot(tmon1_fluxes, eigval_array[1,:], label=r'|10$\rangle$')
plt.plot(tmon1_fluxes, eigval_array[2,:], label=r'|01$\rangle$')

plt.xlabel('Tmon1 Flux ($\Phi_0$)', fontsize=24)
plt.ylabel('Energies (GHz)', fontsize=24)
plt.tick_params(labelsize=22)
# plt.title(fr'Coupled T')
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()
