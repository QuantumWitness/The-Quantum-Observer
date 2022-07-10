#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:10:22 2022

"""

import numpy as np
import scqubits as scqub
import SQcircuit as sqcirc
import scipy.constants as const
import matplotlib.pyplot as plt

h = const.h
e = const.e
phi0 = h/(2*e)
phi0_red = phi0/2/np.pi
GHz = const.giga
MHz = const.mega
#SC Qubits tunable transmon
EJmax = 24
EC = 0.200
d = 0
flux = 0
ng = 0
ncut = 30

tmon = scqub.TunableTransmon(EJmax, EC, d, flux, ng, ncut)

flux_array = np.linspace(-1,1,101)

fig, ax=tmon.plot_evals_vs_paramvals('flux', flux_array, 
                             evals_count=5, subtract_ground=True)
ax.set_ylabel('Energy (GHz)')
fig.tight_layout()

tmon_spec = tmon.get_spectrum_vs_paramvals('flux', flux_array,
                                           evals_count=5,
                                           subtract_ground=True)

def Em (m, Ej, Ec):
    E0 = -Ej + np.sqrt(8*Ej*Ec)/2 - Ec/4
    Em = -Ej + np.sqrt(8*Ej*Ec)*(m+0.5)-Ec*(6*m**2+6*m+3)/12
    return (Em-E0)/1e9

#SQcircuit transmon
init_flux=0
C = 97 #fF
Ej = 24 #GHz
ng = 0
ncut = 30

sqcirc.set_unit_cap('fF')
sqcirc.set_unit_JJ('GHz')

loop1 = sqcirc.Loop(value=init_flux)
C1 = sqcirc.Capacitor(value=C)
JJ1 = sqcirc.Junction(value=Ej/2, loops=[loop1])
JJ2 = sqcirc.Junction(value=Ej/2, loops=[loop1])

elements = {(0,1): [C1,JJ1,JJ2]}

tmon = sqcirc.Circuit(elements)
tmon.set_trunc_nums([ncut])
tmon.set_charge_offset(1,ng)
# get the first two eigenfrequencies and eigenvectors
n_eig=5
efreqs, evecs = tmon.diag(n_eig=n_eig)

# print the qubit frequency
print("qubit frequencies:", efreqs-efreqs[0])

flux_array = np.linspace(-1,1,101)

eigval_array = np.zeros((n_eig,len(flux_array)))
for index, f in enumerate(flux_array):
    loop1.set_flux(f)
    efreqs, _ = tmon.diag(n_eig=n_eig)
    eigval_array[:,index]=efreqs-efreqs[0]
    
plt.figure(figsize=(6,4))
for i in range(n_eig):
    plt.plot(flux_array, eigval_array[i,:], label=f'|{i}$\\rangle$')
    
plt.xlabel('Loop Flux ($\Phi_0$)')
plt.ylabel('Energies (GHz)')
plt.title(fr'Tmon Spectrum, $n_g$ = {ng}')
# plt.legend()
plt.tight_layout()
plt.show()

#compare
plt.figure(figsize=(8,4))
for i in range(n_eig):
    plt.plot(flux_array[40:60], eigval_array[i,40:60]-tmon_spec.energy_table[40:60,i], label=f'|{i}$\\rangle$')
    
plt.xlabel('Loop Flux ($\Phi_0$)')
plt.ylabel('Energy Diff (GHz)')
plt.title('SQcircuit  vs scQubits')
plt.legend()
plt.tight_layout()
plt.show()

