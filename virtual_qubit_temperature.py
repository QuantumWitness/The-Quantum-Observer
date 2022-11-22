#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculating virtual temperatures
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from virtual_qubit_utils import calc_Tv

kB = const.Boltzmann
h = const.Planck
giga = const.giga

Ea = 1
Eb = np.linspace(1,200,1000)*Ea

Ta = .05
Tb = .2


Tv = [calc_Tv(Ea, E, Ta, Tb) for E in Eb]


plt.figure(figsize=(10,8))
plt.semilogx(Eb/Ea, np.array(Tv), color='xkcd:periwinkle', linewidth=4,
             label='$T_v$')
plt.hlines(Ta, np.min(Eb/Ea),np.max(Eb/Ea), colors='b', linestyles='dashed', linewidth=4,
           label='$T_a$')
plt.hlines(Tb, np.min(Eb/Ea),np.max(Eb/Ea), colors='r', linestyles='dashed', linewidth=4,
           label='$T_b$')
plt.xlabel('$E_b/Ea$', fontsize=24)
plt.ylabel('$T_v$ (K)', fontsize=24)
plt.tick_params(labelsize=22)
plt.ylim([-1,1])
plt.tight_layout()
plt.legend(fontsize=24)
plt.grid()
plt.show()