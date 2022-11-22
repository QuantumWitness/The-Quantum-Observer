#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some functions useful for calculating virtual qubit temperatures or simulating
coupled qubits.

"""
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

kB = const.Boltzmann
h = const.Planck
giga = const.giga

def calc_Tv(Ea, Eb, Ta, Tb):
    """

    Parameters
    ----------
    Ea : float
        Energy of qubit A in hGHz
    Eb : float
        Energy of qubit B in hGHz
    Ta : float
        Temperature of qubit A in K
    Tb : float
        Temperature of qubit B in K

    Returns
    -------
    Tv: float
        Temperature of the virtual qubit

    """
    Eb = Eb*h*giga
    Ea = Ea*h*giga
    
    num = Eb-Ea
    denom = (Eb/(kB*Tb) - Ea/(kB*Ta))
    Tv = num/(denom*kB)
    
    return Tv

def sim_2qubit(Ea, Eb, Ta, Tb, max_time=10, num_timestep=1001, vqub=(1,2)):
    '''
    Calculate the virtual temperature from a composite Hamiltonian evolved in 
    time.
    
    Parameters
    ----------
    Ea : float
        Energy of qubit A in hGHz
    Eb : float
        Energy of qubit B in hGHz
    Ta : float
        Temperature of qubit A in K
    Tb : float
        Temperature of qubit B in K
    max_time: float, optional
        How long to run the simulation? Default is 10 time units. 
    num_timesteps: int
        What time resolution to use?
    vqub: tuple
        Which states make the virtual qubit?



    Returns
    -------
    virtual_temperature: float
        Calculated virtual temperature 
        
    pops: array-like
        The calculated state populations

    '''
    H_arr = np.identity(4)
    H_arr[0,0] = 0
    H_arr[1,1] = Ea
    H_arr[2,2] = Eb
    H_arr[3,3] = Ea + Eb
    
    H = Qobj(H_arr)
    
    energies= h*giga*np.array([Ea, Eb])
    
    qubit_temps = [Ta, Tb] # Kelvin
    decay_list = [1,1] #we normalize to decay rates, so can just pretend they're 1
    excite_list = [decay_rate * np.exp(-energy/(kB*qubit_T)) for decay_rate, energy, qubit_T in zip(decay_list, energies, qubit_temps)] 
    C_ops = Qobj(np.array([[0,np.sqrt(decay_list[0]),np.sqrt(decay_list[1]),0],
                           [np.sqrt(excite_list[0]),0,0,0],
                            [np.sqrt(excite_list[1]),0,0,0],
                            [0,0,0,0]]))
    times = np.linspace(0,max_time,num_timestep)
    init_state = ket2dm(basis(4,0))
    opts = Options(nsteps=10000)
    results = mesolve(H, init_state, times, c_ops=C_ops,options=opts)

    #First just check if we get the right final temperature
    pops = np.zeros((len(times),4))
    for index,state in enumerate(results.states):
        for j in range(4):
            pops[index,j] = (state*ket2dm(basis(4,j))).tr()
            
    virtual_temperature = -(energies[1]-energies[0])/(kB*np.log(pops[-1,vqub[1]]/pops[-1,vqub[0]]))

    return virtual_temperature, pops

def sim_2qubit_alt(Ea, Eb, Ta, Tb, max_time=10, num_timestep=1001, vqub=(1,2)):
    '''
    Same output as the function above, but creates the composite system 
    Hamiltonian from the tensor product of two single-qubit Hamiltonians.
    
    Parameters
    ----------
    Ea : float
        Energy of qubit A in hGHz
    Eb : float
        Energy of qubit B in hGHz
    Ta : float
        Temperature of qubit A in K
    Tb : float
        Temperature of qubit B in K
    max_time: float, optional
        How long to run the simulation? Default is 10 time units. 
    num_timesteps: int
        What time resolution to use?
    vqub: tuple
        Which states make the virtual qubit?
    pops: array-like
        The calculated state populations


    Returns
    -------
    virtual_temperature: float
        Calculated virtual temperature 

    '''
    H = tensor(sigmaz()*Ea/2, identity(2))+tensor(identity(2),(Eb/2)*sigmaz())

    energies= h*giga*np.array([Ea, Eb])
    
    qubit_temps = [Ta, Tb] # Kelvin
    decay_list = [1,1] #we normalize to decay rates, so can just pretend they're 1
    excite_list = [decay_rate * np.exp(-energy/(kB*qubit_T)) for decay_rate, energy, qubit_T in zip(decay_list, energies, qubit_temps)] 
    C_ops = tensor(Qobj(np.array([[0,1],
                                  [np.sqrt(excite_list[0]),0]])), identity(2))+\
            tensor(identity(2),Qobj(np.array([[0,1],
                                  [np.sqrt(excite_list[1]),0]])))

    times = np.linspace(0,max_time,num_timestep)
    init_state = tensor(basis(2,0),basis(2,0))
    opts = Options(nsteps=100000)
    results = mesolve(H, init_state, times, c_ops=C_ops,options=opts)

    #First just check if we get the right final temperature
    pops = np.zeros((len(times),4))
    for index,state in enumerate(results.states):
        pops[index,0] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,0)))).tr())
        pops[index,1] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,0)))).tr())
        pops[index,2] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,1)))).tr())
        pops[index,3] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,1)))).tr())
        
            
    virtual_temperature = -(energies[1]-energies[0])/(kB*np.log(pops[-1,vqub[1]]/pops[-1,vqub[0]]))

    return virtual_temperature, pops

def sim_3qubit(Ea, Eb, Ec, Ta, Tb, Tc, g,
                      max_time=100, num_timestep=1001, qubC=(0,4),vqub=(1,2)):
    '''

    Parameters
    ----------
    Ea : float
        Energy of qubit A in hGHz
    Eb : float
        Energy of qubit B in hGHz
    Ta : float
        Temperature of qubit A in K
    Tb : float
        Temperature of qubit B in K
    max_time: float, optional
        How long to run the simulation? Default is 10 time units. 
    num_timesteps: int
        What time resolution to use?
    qubC: tuple
        Which two states make the coupled qubit (Qubit C)
    vqub: tuple
        Which states make the virtual qubit?


    Returns
    -------
    qubit_temperature: float
        Calculated temperature of two states specified by qubC
        
    virtual_temperature: float
        Calculated temperature of two states specified by vqub
        
    pops: array-like
        The calculated state populations

    '''
    # Create the 3-qubit system hamiltonian
    H_sys = tensor(sigmaz()*Ea/2, identity(2),identity(2))+\
        tensor(identity(2),(Eb/2)*sigmaz(), identity(2))+\
        tensor(identity(2),identity(2),(Ec/2)*sigmaz())
    
    # Interaction hamiltonian from the paper
    H_int = g*(tensor(basis(2,0),basis(2,1),basis(2,0))*\
               tensor(basis(2,1),basis(2,0),basis(2,1)).dag()+\
               tensor(basis(2,1),basis(2,0),basis(2,1))*\
               tensor(basis(2,0),basis(2,1),basis(2,0)).dag())
    
    H = H_sys + H_int
    
    energies= h*giga*np.array([Ea, Eb, Ec])
    
    qubit_temps = [Ta, Tb, Tc] # Kelvin
    decay_list = np.array([1,1,0.001])
    excite_list = np.array([decay_rate * np.exp(-energy/(kB*qubit_T)) for decay_rate, energy, qubit_T in zip(decay_list, energies, qubit_temps)]) 
 
    C_op_a = tensor(Qobj(np.array([[0,1],
                                  [np.sqrt(excite_list[0]),0]])), identity(2),identity(2))
    C_op_b = tensor(identity(2),Qobj(np.array([[0,1],
                                  [np.sqrt(excite_list[1]),0]])), identity(2))
    C_op_c = tensor(identity(2), identity(2), Qobj(np.array([[0,decay_list[2]],
                                  [np.sqrt(excite_list[2]),0]])))
    
    C_ops = [C_op_a, C_op_b, C_op_c]
 
    times = np.linspace(0,max_time,num_timestep)
    init_state = tensor(basis(2,0),basis(2,0),basis(2,0))
    opts = Options(nsteps=10000)
    results = mesolve(H, init_state, times, c_ops=C_ops)
 
    pops = np.zeros((len(times),8))
    for index,state in enumerate(results.states):
        pops[index,0] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,0),basis(2,0)))).tr())
        pops[index,1] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,0),basis(2,0)))).tr())
        pops[index,2] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,1),basis(2,0)))).tr())
        pops[index,3] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,1),basis(2,0)))).tr())
        pops[index,4] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,0),basis(2,1)))).tr())
        pops[index,5] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,0),basis(2,1)))).tr())
        pops[index,6] = np.abs((state*ket2dm(tensor(basis(2,0),basis(2,1),basis(2,1)))).tr())
        pops[index,7] = np.abs((state*ket2dm(tensor(basis(2,1),basis(2,1),basis(2,1)))).tr())
        
             
    virtual_temperature = -(energies[1]-energies[0])/(kB*np.log(pops[-1,vqub[1]]/pops[-1,vqub[0]]))
    qubit_temperature = -(energies[2])/(kB*np.log(pops[-1,qubC[1]]/pops[-1,qubC[0]]))


    return qubit_temperature, virtual_temperature, pops