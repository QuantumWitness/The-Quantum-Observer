#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A bunch of functions to analyze IBMQ systems
and simulate virtual systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_qmap(architecture):
    """
    

    Parameters
    ----------
    architecture : string
        Which architecture do we need a mapping for?

    Returns
    -------
    qub_map : dict
        A dict which returns a list of nearest neighbors for each qubit
        
    ones, twos, threes : list
        List of qubits labelled with 1, 2, or 3 in the mapping found here:
            https://arxiv.org/pdf/2009.00781.pdf
            

    """
    if architecture == 'd3 hex':
        qubits = np.linspace(0,26,27)
        
        ones = [3,7,14,18,25]
        twos = [1,8,12,19,23]
        threes = [int(q) for q in qubits if q not in ones+twos]
        
        qub_map = {'Q0':[1],
                       'Q1':[0,2,4],
                       'Q2':[1,3],
                       'Q3':[2,5],
                       'Q4':[1,7],
                       'Q5':[3,8],
                       'Q6':[7],
                       'Q7':[4,6,10],
                       'Q8':[5,9,11],
                       'Q9':[8],
                       'Q10':[7,12],
                       'Q11':[8,14],
                       'Q12':[10,13,15],
                       'Q13':[12,14],
                       'Q14':[11,13,16],
                       'Q15':[12,18],
                       'Q16':[14,19],
                       'Q17':[18],
                       'Q18':[15,17,21],
                       'Q19':[16,20,22],
                       'Q20':[19],
                       'Q21':[18,23],
                       'Q22':[19,25],
                       'Q23':[21,24],
                       'Q24':[23,25],
                       'Q25':[22,24,26],
                       'Q26':[25]}
        
    elif architecture == 'd5 hex':
        qubits = np.linspace(0,64,65)
    
        ones = [2,6,13,17,21,29,33,37,41,45,49,56,60,64]
        twos = [0,4,8,15,19,23,27,31,35,43,47,51,58,62]
        threes = [int(q) for q in qubits if q not in ones+twos]

        qub_map = {'Q0':[1,10],
                       'Q1':[0,2],
                       'Q2':[1,3],
                       'Q3':[2,4],
                       'Q4':[3,5,11],
                       'Q5':[4,6],
                       'Q6':[5,7],
                       'Q7':[6,8],
                       'Q8':[7,9,12],
                       'Q9':[8],
                       'Q10':[0,13],
                       'Q11':[4,17],
                       'Q12':[8,21],
                       'Q13':[10,14],
                       'Q14':[13,15],
                       'Q15':[14,16,24],
                       'Q16':[15,17],
                       'Q17':[11,16,18],
                       'Q18':[17,19],
                       'Q19':[18,20,25],
                       'Q20':[19,21],
                       'Q21':[12,20,22],
                       'Q22':[21,23],
                       'Q23':[22,26],
                       'Q24':[15,29],
                       'Q25':[19,33],
                       'Q26':[23,37],
                       'Q27':[28,38],
                       'Q28':[27,29],
                       'Q29':[24,28,30],
                       'Q30':[29,31],
                       'Q31':[30,32,39],
                       'Q32':[31,33],
                       'Q33':[25,32,34],
                       'Q34':[33,35],
                       'Q35':[34,36,40],
                       'Q36':[35,37],
                       'Q37':[26,36],
                       'Q38':[27,41],
                       'Q39':[31,45],
                       'Q40':[35,49],
                       'Q41':[38,42],
                       'Q42':[41,43],
                       'Q43':[42,44,52],
                       'Q44':[43,45],
                       'Q45':[39,44,46],
                       'Q46':[45,47],
                       'Q47':[46,48,53],
                       'Q48':[47,49],
                       'Q49':[40,48,50],
                       'Q50':[49,51],
                       'Q51':[50,54],
                       'Q52':[43,56],
                       'Q53':[47,60],
                       'Q54':[51,64],
                       'Q55':[56],
                       'Q56':[52,55,57],
                       'Q57':[56,58],
                       'Q58':[57,59],
                       'Q59':[58,60],
                       'Q60':[53,59,61],
                       'Q61':[60,62],
                       'Q62':[61,63],
                       'Q63':[62,64],
                       'Q64':[54,63]}
    elif architecture == 'd7 hex':
        qubits = np.linspace(0,126,127)
    
        ones = [0,4,8,12,20,24,28,32,37,41,45,49,58,62,66,70,
                75,79,83,87,96,100,104,108,116,120,124]
        twos = [2,6,10,18,22,26,30,39,43,47,51,56,60,64,68,
                77,81,85,89,94,98,102,106,114,118,122,126]
        threes = [int(q) for q in qubits if q not in ones+twos]

        qub_map = {'Q0':[1,14],
                       'Q1':[0,2],
                       'Q2':[1,3],
                       'Q3':[2,4],
                       'Q4':[3,5,15],
                       'Q5':[4,6],
                       'Q6':[5,7],
                       'Q7':[6,8],
                       'Q8':[7,9,16],
                       'Q9':[8,10],
                       'Q10':[9,11],
                       'Q11':[10,12],
                       'Q12':[11,13,17],
                       'Q13':[12],
                       'Q14':[0,18],
                       'Q15':[4,22],
                       'Q16':[8,26],
                       'Q17':[12,30],
                       'Q18':[14,19],
                       'Q19':[18,20],
                       'Q20':[19,21,33],
                       'Q21':[20,22],
                       'Q22':[15,21,23],
                       'Q23':[22,24],
                       'Q24':[23,25,34],
                       'Q25':[24,26],
                       'Q26':[16,25,27],
                       'Q27':[26,28],
                       'Q28':[27,29,35],
                       'Q29':[28,30],
                       'Q30':[17,29,31],
                       'Q31':[30,32],
                       'Q32':[31,36],
                       'Q33':[20,39],
                       'Q34':[24,43],
                       'Q35':[28,47],
                       'Q36':[32,51],
                       'Q37':[38,52],
                       'Q38':[37,39],
                       'Q39':[33,38,40],
                       'Q40':[39,41],
                       'Q41':[40,42,53],
                       'Q42':[41,43],
                       'Q43':[34,42,44],
                       'Q44':[43,45],
                       'Q45':[44,46,54],
                       'Q46':[45,47],
                       'Q47':[35,46,48],
                       'Q48':[47,49],
                       'Q49':[48,50,55],
                       'Q50':[49,51],
                       'Q51':[36,50],
                       'Q52':[37,56],
                       'Q53':[41,60],
                       'Q54':[45,64],
                       'Q55':[49,68],
                       'Q56':[52,57],
                       'Q57':[56,58],
                       'Q58':[57,59,71],
                       'Q59':[58,60],
                       'Q60':[53,59,61],
                       'Q61':[60,62],
                       'Q62':[61,63,72],
                       'Q63':[62,64],
                       'Q64':[54,63,65],
                       'Q65':[64,66],
                       'Q66':[65,67,73],
                       'Q67':[66,68],
                       'Q68':[55,67,69],
                       'Q69':[68,70],
                       'Q70':[69, 74],
                       'Q71':[58,77],
                       'Q72':[62,81],
                       'Q73':[66,85],
                       'Q74':[70,89],
                       'Q75':[76,90],
                       'Q76':[75,77],
                       'Q77':[71,76,78],
                       'Q78':[77,79],
                       'Q79':[78,80,91],
                       'Q80':[79,71],
                       'Q81':[72,80,82],
                       'Q82':[81,83],
                       'Q83':[82,84,92],
                       'Q84':[83,85],
                       'Q85':[73,84,86],
                       'Q86':[85,87],
                       'Q87':[86,88,93],
                       'Q88':[87,89],
                       'Q89':[74,88],
                       'Q90':[75,94],
                       'Q91':[79,98],
                       'Q92':[83,102],
                       'Q93':[87,106],
                       'Q94':[90,95],
                       'Q95':[94,96],
                       'Q96':[95,97,109],
                       'Q97':[96,98],
                       'Q98':[91,97,99],
                       'Q99':[98,100],
                       'Q100':[99,101,110],
                       'Q101':[100,102],
                       'Q102':[92,101,103],
                       'Q103':[102,104],
                       'Q104':[103,105,111],
                       'Q105':[104,106],
                       'Q106':[93,105,107],
                       'Q107':[106,108],
                       'Q108':[107,112],
                       'Q109':[96,114],
                       'Q110':[100,118],
                       'Q111':[104,122],
                       'Q112':[108,126],
                       'Q113':[114],
                       'Q114':[109,113,115],
                       'Q115':[114,116],
                       'Q116':[115,117],
                       'Q117':[116,118],
                       'Q118':[110,117,119],
                       'Q119':[118,120],
                       'Q120':[119,121],
                       'Q121':[120,122],
                       'Q122':[111,121,123],
                       'Q123':[122,124],
                       'Q124':[123,125],
                       'Q125':[124,126],
                       'Q126':[112,125]}
    else:
        raise ValueError("Bad input string")

    return qub_map,ones,twos,threes    

def add_collision(collision_list, q1=None,q2=None,q3=None, collision_type=1):
    """
    Appends collision details to a list
    
    Collision Rules from: 
    https://arxiv.org/pdf/2009.00781.pdf
    or
    https://arxiv.org/pdf/2012.08475.pdf

    Collisions occur if
        1. abs(f01_j - f01_k)  < 17 MHz; j,k Nearest Neighbors
        2. abs(f02_j -2*f01_k) <  4 MHz; control qubit j, target qubit k
        3. abs(f01_j - f12_k)  < 30 MHz; Nearest Neighbors
        4. f01_k < f12_j OR f01_j < f01_k; control qubit j, target qubit k
        5. abs(f01_i - f01_k)  < 17 MHz; j control to i and/or k, NN to both
        6. abs(f01_i - f12_k) < 25 MHz;
           abs(f12_i - f01_k) < 25 MHz; j control to i and/or k, NN to both
        7. abs(f02_j - (f01_i+f01_k)) < 17 MHz; j control to i and/or k, NN to both


    Parameters
    ----------
    collision_list : list
        The list of collisions to which we will append .
    q1 : string, 
        First qubit in the interaction. Expected to be of the form
        'Q##'
    q2 : String, 
        Second qubit in the interaction. Expected to be of the form
        'Q##'
    q3 : string, optional
        Third qubit in the interaction. Expected to be of the form
        'Q##'. This one only exists for conditions 5-7, which include a 
        control qubit along with its targets. 
    collision_type : int, optional
        Whi. The default is 1.

    Returns
    -------
    None.

    """
    
    if q3 is not None:
        collision_list.append([[q1,q2,q3],collision_type])
    else:
        collision_list.append([[q1,q2],collision_type])

def check_collisions(cal_data=None, architecture=None,
                     collisions=None):
    """
    Collision checker for IBMQ systems.
    
    Collision Rules from: 
    https://arxiv.org/pdf/2009.00781.pdf
    or
    https://arxiv.org/pdf/2012.08475.pdf

    Collisions occur if
        1. abs(f01_j - f01_k)  < 17 MHz; j,k Nearest Neighbors
        2. abs(f02_j -2*f01_k) <  4 MHz; control qubit j, target qubit k
        3. abs(f01_j - f12_k)  < 30 MHz; Nearest Neighbors
        4. f01_k < f12_j OR f01_j < f01_k; control qubit j, target qubit k
        5. abs(f01_i - f01_k)  < 17 MHz; j control to i and/or k, NN to both
        6. abs(f01_i - f12_k) < 25 MHz;
           abs(f12_i - f01_k) < 25 MHz; j control to i and/or k, NN to both
        7. abs(f02_j - (f01_i+f01_k)) < 17 MHz; j control to i and/or k, NN to both

    Control qubits are always threes
    Threes always see 2 target qubits

    Parameters
    ----------
    cal_data : pandas Dataframe
        Calibration data from an IBMQ system.
        Indexed on qubit ('Q##'), with 'Frequency (GHz)' and
        'Anharmonicity (GHz)'
    architecture : string
        What is the qubit connectivity? Currently only supports
        d=5 heavy hex (65 qubits)
        Once more mappings are added, will expect input like
        'd3 hex'
        'd5 hex'
        'd7 hex'
    collisions :

    Returns
    -------
    collisions : pandas Dataframe
    Data frame of collisions with columns
    'Qubit 1', 'Qubit 2', 'Collision Type'

    """
    # Check that an architecture has been supplied
    # We need to know which are control, flag, and ancillae
    if architecture is not None:
        qub_map,ones,twos,threes = get_qmap(architecture)
    
    if collisions is None:
        collisions = []
    
    #Step through qubits in the system and 
    
    freq = 'Frequency (GHz)'
    anharm = 'Anharmonicity (GHz)'
    
    for index,key in enumerate(qub_map.keys()):
        '''
        Each key is a qubit
        If it's a 1  or 2, check rules 1,3
        If it's a 3, check the rest of the rules
        
        In this way, I think we never double count collisions
        
        Record collisions in form:
            Q1 | Q2 | COLLISION TYPE
        '''
        
        f01_j = cal_data.loc[key][freq]
        anharm_j = cal_data.loc[key][anharm]
        f12_j = f01_j + anharm_j 
        f02_j = 2*f01_j + anharm_j
        
        if (index in ones) or (index in twos):
            # Check rules 1, 3
            for NN in qub_map[key]:
                nn_qubit = f'Q{NN}'
                f01_k = cal_data.loc[nn_qubit][freq]
                anharm_k = cal_data.loc[nn_qubit][anharm]
                f12_k = f01_k + anharm_k
                
                #condition1
                if np.abs(f01_j - f01_k)  < 0.017:
                    add_collision(collisions,q1=key,
                                  q2=nn_qubit, collision_type=1)
                # condition3
                if np.abs(f01_j - f12_k)  < 0.03:
                    add_collision(collisions,q1=key,
                                  q2=nn_qubit, collision_type=3)
                        
        else:
            #check other rules
            for NN in qub_map[key]:
                nn_qubit = f'Q{NN}'
                f01_k = cal_data.loc[nn_qubit][freq]
                anharm_k = cal_data.loc[nn_qubit][anharm]
                f12_k = f01_k + anharm_k
                
                # check condition 2,4 
                if np.abs(f02_j - 2*f01_k) < 0.004:
                    add_collision(collisions,q1=key,q2=nn_qubit,
                                  collision_type=2)
                    
                # if f01_j < f01_k:
                    # add_collision(collisions,q1=key,q2=nn_qubit,
                    #               collision_type=4)
                    
            
            #check conditions 5,6,7
            # do these checks ONLY if there are 2 target qubits
            # in these mappings, control qubits NEVER have more than 
            # 2 targets/nearest neighbors, so these are straight forward
            if len(qub_map[key]) > 1:
                nn1 = qub_map[key][0]
                nn2 = qub_map[key][1]
                nn_qubit1 = f'Q{nn1}'
                nn_qubit2 = f'Q{nn2}'
                
                f01_i = cal_data.loc[nn_qubit1][freq]
                anharm_i = cal_data.loc[nn_qubit1][anharm]
                f12_i = f01_i + anharm_i
                
                f01_k = cal_data.loc[nn_qubit2][freq]
                anharm_k = cal_data.loc[nn_qubit2][anharm]
                f12_k = f01_k + anharm_k
                
                
                
                # #condition 5
                # if np.abs(f01_i - f01_k) < 0.017:
                #     add_collision(collisions, q1=nn_qubit1, q2=nn_qubit2,
                #                   q3=key, collision_type=5)
                
                # #condition 6
                # if np.abs(f01_i-f12_k) < 0.025 or np.abs(f12_k-f01_i) < 0.025:
                #     add_collision(collisions, q1=nn_qubit1, q2=nn_qubit2,
                #                   q3=key, collision_type=6)
                     
                # #condition 7
                # if np.abs(f02_j - f01_k - f01_i) < 0.017:
                #     add_collision(collisions, q1=nn_qubit1, q2=nn_qubit2,
                #                   q3=key, collision_type=7)
            
    
    return collisions

def ibm_check_collisions(cal_data=None, architecture=None,
                         collisions=None, freq_window=0.03):
    """
    Check collisions using the IBM frequency window method 
    as in their paper.

    Parameters
    ----------
    cal_data : TYPE, optional
        DESCRIPTION. The default is None.
    architecture : TYPE, optional
        DESCRIPTION. The default is None.
    collisions : TYPE, optional
        DESCRIPTION. The default is None.
    freq_window : TYPE, optional
        DESCRIPTION. The default is 0.03.

    Returns
    -------
    collisions

    """
    
    if architecture is not None:
        qub_map,ones,twos,threes = get_qmap(architecture)
    
    if collisions is None:
        collisions = []
    
    #Step through qubits in the system and 
    
    freq = 'Frequency (GHz)'
    
    targets = [5.05, 5.12, 5.19]
    
    for f01 in cal_data[freq]:
        if f01 < np.min(targets) - freq_window or f01 > np.max(targets) + freq_window:
            add_collision(collisions,q1='Q1',q2='q1',
                          collision_type=1)
        elif f01 > targets[0]+freq_window and f01 < targets[1]-freq_window:
            add_collision(collisions,q1='Q1',q2='q1',
                          collision_type=1)
        elif f01 > targets[1]+freq_window and f01 < targets[2]-freq_window:
            add_collision(collisions,q1='Q1',q2='q1',
                          collision_type=1)
            
    return collisions
            

def create_dummy_system(architecture, sigma_f=0.132):
    """
    Function to create a dummy system in the chosen architecture.
    Sigma_f is the standard deviation of the frequencies in this system.
    As far as I can tell, there are three target frequencies in IBM heavy 
    hex architectures, 5.05 GHz, 5.12 GHz, 5.19 GHz
    
    We'll assign these in order to ones, twos, and threes per the paper.
    
    

    Parameters
    ----------
    architecture : string
        Which architecture should we use?
        Acceptable options: 'd3 hex', 'd5 hex', 'd7 hex'
    sigma_f : double, optional
        The standard deviation of the frequencies. The default is 0.132 GHz,
        corresponding to non-LASIQ tuned qubits.

    Returns
    -------
    dummy_system : pandas Dataframe
        DataFrame indexed by qubit with columns'Frequency (GHz)'
        and 'Anharmonicity (GHz)'

    """
    freq_1 = 5.05
    freq_2 = 5.12
    freq_3 = 5.19
    
    eta = -0.328
    sigma_eta = 0.0023
    
    qub_map,ones,twos,threes = get_qmap(architecture)
    
    f01 = np.zeros((len(qub_map),1))
    anharm = np.random.normal(loc=eta,scale=sigma_eta,size=(len(qub_map),1))
    
    for index,qubit in enumerate(qub_map.keys()):
        if index in ones:
            f01[index] = np.random.normal(loc=freq_1, scale=sigma_f)
        elif index in twos:
            f01[index] = np.random.normal(loc=freq_2, scale=sigma_f)
        else:
            f01[index] = np.random.normal(loc=freq_3, scale=sigma_f)
            
    data = np.column_stack((f01,anharm))
    
    dummy_system = pd.DataFrame(data=data, index=qub_map.keys(),
                                columns=['Frequency (GHz)','Anharmonicity (GHz)'] )
            
    return dummy_system
    
    
def run_collision_sims(architecture, sigma_f=0.132,
                       num_sims=1000, plot=True, use_IBM_check=False):
    """
    Run many collision checks on dummy systems with specified sigma_f

    Parameters
    ----------
    architecture : string
        'd3 hex', 'd5 hex', or 'd7 hex'
    sigma_f : float, optional
        Standard deviation of frequency distributions. 
        The default is 0.132 GHz
    num_sims : int, optional
        How many simulations to run. The default is 1000.
    plot : boolean, optional
        Plot results as histogram? The default is True.
    use_IBM_check: boolean, optional
        Use the IBM frequency window to check collisions. Default false.

    Returns
    -------
    num_collisions : array-like
        Number of collisions per run in an array

    """
    num_collisions = np.zeros((num_sims,1))
    for i in range(num_sims):
        dummy_system = create_dummy_system(architecture, sigma_f=sigma_f)
        if use_IBM_check:
            colls = ibm_check_collisions(cal_data=dummy_system,
                                         architecture=architecture)
        else:
            colls = check_collisions(cal_data=dummy_system,
                                     architecture=architecture)
        num_collisions[i] = len(colls)
        
    if plot:
        plt.figure(figsize=(9,9))
        plt.hist(num_collisions, bins=num_sims//100, density=True)
        plt.vlines(np.mean(num_collisions), 0,np.max(num_collisions)/np.sum(num_collisions),
                   color='xkcd:sky blue',
                   label=f'Mean collisions={np.mean(num_collisions)}')
        plt.xlabel("Collisions", fontsize=24)
        plt.title(architecture, fontsize=26)
        plt.tick_params(labelsize=22)
        plt.legend(prop={'size': 16})
        plt.tight_layout()
        plt.show()
        
    return num_collisions
        






