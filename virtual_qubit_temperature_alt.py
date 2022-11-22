#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigating virtual temperatures with tensor products
"""

import numpy as np
import scipy.constants as scc
import matplotlib.pyplot as plt
from virtual_qubit_utils import calc_Tv, sim_2qubit_alt

kB = scc.Boltzmann
h = scc.Planck
giga = scc.giga
Ea = 1
Eb = 10*Ea
energies= h*giga*np.array([Ea, Eb])


Ta = 0.05
Tb = 0.2
qubit_temps = [Ta, Tb] # Kelvin

max_time=10
num_timestep = 1001
times = np.linspace(0,max_time,num_timestep)


virtual_temperature, pops = sim_2qubit_alt(Ea, Eb, Ta, Tb,
                                           max_time=max_time,
                                           num_timestep=num_timestep)

final_temperatures = [(-energies[0])/(np.log(pops[-1,1]/pops[-1,0])*kB),(-energies[1])/(np.log(pops[-1,2]/pops[-1,0])*kB)]
print(f'Qubit A Temp = {final_temperatures[0]*1000} mK')
print(f'Qubit B Temp = {final_temperatures[1]*1000} mK')

virtual_temperature = -(energies[1]-energies[0])/(kB*np.log(pops[-1,2]/pops[-1,1]))
virt_temp_theory = (energies[1]-energies[0])/((energies[1]/Tb)-(energies[0]/Ta))
print(f'Virtual Qubit Temperature = {virtual_temperature*1000} mK')
print(f'Theoretical Tv = {virt_temp_theory*1000} mK')



plt.figure(figsize=(8,10))
for j in range(4):
    plt.plot(times,pops[:,j],label=f'$|{j}\\rangle$', linewidth=4)
plt.xlabel('Time (whatever)', fontsize=24)
plt.ylabel('Population', fontsize=24)
plt.title(f'$E_b/E_a$ = {Eb/Ea}, $T_a$={Ta*1000} mK, $T_b$={Tb*1000} mK', fontsize=24)
plt.tick_params(labelsize=22)
plt.tight_layout()
plt.legend(fontsize=24)
plt.grid()
plt.show()



Eb_list = np.logspace(0,2,26)*Ea


virtual_temps = np.zeros((len(Eb_list),))
for index,Eb in enumerate(Eb_list):
    
    Tv, _ = sim_2qubit_alt(Ea, Eb, Ta, Tb,
                           max_time=max_time,
                           num_timestep=num_timestep)
    
    virtual_temps[index]=Tv
    
#compare to theory
Eb_theory = np.linspace(1,100,1000)*Ea


Tv = [calc_Tv(Ea, E, Ta, Tb) for E in Eb_theory]

plt.figure(figsize=(10,8))
plt.semilogx(Eb_list/Ea,virtual_temps,'ro', markerfacecolor='none', markersize=10,
             label='QuTIP')
plt.semilogx(Eb_theory/Ea,Tv,color='xkcd:periwinkle',linewidth=3, label='Theory')
plt.ylabel('$T_v$ (K)', fontsize=24)
plt.xlabel('$E_b/E_a$', fontsize=24)
plt.tick_params(labelsize=22)
plt.ylim([-1,1])
plt.grid()
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()


