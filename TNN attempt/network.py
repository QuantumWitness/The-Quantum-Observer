'''

Trying to implement a Thermodynamic Neural Network 
ala T. Hylton 
arXiv: 1906.01678v4

TODO: Remove limit on magnitude of weights
TODO: Not sure that we need output charge compartments. 
TODO: Allow user specified initialization instead 
        of random state selection.
TODO: Have a continuous option for state_space?
TODO: Expand connectivity options
TODO: Allow network regions with different temperatures?
TODO: Implement selection functions beyond the Heaviside function.
DONE: Implement node state update
TODO: Initialize Network with pre-allocated energy history array
TODO: Implement fixed nodes (external biases)
'''
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite


class Network:

    def __init__(self, size, num_states=2, connectivity='NN', temperature=(1,1)):
        """
        Initialize a network with Node class as the
        node object. 

        For this initial implementation will assume NN connectivity
        and periodic boundaries. To conform to the TNN description
        of input/output charge channels, we'll implement an undirected
        graph, because edges will define channels. Nodes will have
        distinct compartments for + and - input/output charges.

        Although we can choose initial random states, we don't know
        the energy of each node until will initialize the edge weights
        and calculate charges. 

        

        size: tuple
            Should be an ordered pair indicating the size of
            the NN

        num_state = int
            How many states in the state space? We'll choose in the 
            interval [-1,1]. Default is 2
            
        connectivity: string
            A string indicating the type of connectivity expected.
            Currently valid values:
            'NN': to make a nearest neighbor (2D) graph.
            'bipartite': to make a 2D bipartite graph. If no kwargs are supplied,
                         will default to NN. Otherwise takes connectivity from kwargs.
            'custom': to make a custom graph. Must supply a graph object in kwargs.
                      Graph object should have integer nodes with the properties 
                      shown below in the __init__ function.
            

        temperature: float or tuple
            The 'temperature' of the network (in energy units). If 
            a float is supplied, then nodes and edges will have
            the same temperature. If a tuple is supplied, the first
            element will be node temperature, and the 2nd element
            will be edge temperature. Default is 1.


        """
        self._size = size
        self._state_space = np.linspace(-1,1,num_states)
        if isinstance(temperature, float):
            self.node_temperature = temperature
            self.edge_temperature = temperature
        elif isinstance(temperature, tuple):
            self.node_temperature = temperature[0]
            self.edge_temperature = temperature[1]
        # this is kind of a kluge, but 1000 pts should be fine enough
        # for most purposes. Also need to confirm that these are
        # the correct bounds for the weights.
        self._weight_space = np.linspace(-1*self.edge_temperature,self.edge_temperature,1000)
        
        self._energy_series = []

        self._state = np.zeros((np.prod(self._size),))
        self._fixed_nodes = []

        # build the correct graph
        # TODO: Implement bipartite
        if connectivity == 'NN':
            self._graph = nx.grid_2d_graph(size[0],size[1],periodic=True, create_using=nx.MultiGraph)
            self._graph = nx.convert_node_labels_to_integers(self._graph)
            self._connectivity = connectivity
        elif connectivity == 'bipartite':
            #assumes bipartite next-nearest neighbor (16 connections per node)
            # even sizes are OK for periodic boundaries
            # but odd are not.
            if size[0]%2 == 0 and size[1]%2 == 0:
                self._graph = nx.grid_2d_graph(size[0],size[1],periodic=True, create_using=nx.MultiGraph)
            else:
                self._graph = nx.grid_2d_graph(size[0],size[1],periodic=False, create_using=nx.MultiGraph)
            self._graph = nx.convert_node_labels_to_integers(self._graph)
            self._connectivity = connectivity
            nnn = self._get_NNN_edges()
            self._graph.add_edges_from(nnn)
            # create bipartiteness
            # and add next-nearest neighbors
            for node in self._graph:
                x, y = divmod(node,size[0])   # Convert single integer back into (i,j) coordinates.
                self._graph.nodes[node]["bipartite"] = (x + y) % 2
            

            assert(bipartite.is_bipartite(self._graph))

        elif connectivity == 'custom':
            graph = kwargs.get('graph', None)
            if graph is not None:
                self._graph = graph
                self._connectivity = connectivity
            else:
                raise ValueError('Must supply graph object in kwargs if connectivity is custom')
        else:
            raise NotImplementedError('Only NN connectivity is currently supported')
        
        #iterate through nodes to initialize them
        node_states=np.random.randint(0,num_states,size=np.prod(size))
        for i,(n,s) in enumerate(zip(self._graph.nodes, node_states)):
      
            self._graph.nodes[n]['state']= self._state_space[s]
            self._graph.nodes[n]['energy']= 0
            self._graph.nodes[n]['energy_series'] = []
            self._graph.nodes[n]['input'] = np.zeros((2,2))
            self._graph.nodes[n]['output'] = np.zeros((2,2))
            self._graph.nodes[n]['edges'] = np.zeros((2,2)) #need a place to store edge compartment values
            self._graph.nodes[n]['weight_tracker'] = {'w++':[],'w+-':[],'w-+':[],'w--':[]}
            self._graph.nodes[n]['fixed'] = False # flag for fixed nodes
        self.__record_state()
        #iterate through edges to initialize them

        # original TNN paper samples from a Gaussian, but I'm built different
        # Right now each pair of adjacent nodes will have one edge
        # We will add another edge with the same weight.
        # Thus the two edges will represent the input and output channels
        # for a node.
        # I think the 0 index edge should be 'toward' the smaller index,
        # and 1 index goes 'away' from the smaller index.
        # We will always require that they have the same weight.
        #not sure what the weights should be
        #going to let them range from[-1,1]
        min_weight = np.min(self._weight_space)
        max_weight = np.max(self._weight_space)
        edge_weights = np.random.default_rng().uniform(min_weight,max_weight,size=(len(self._graph.edges)))
        for e,w in zip(self._graph.edges, edge_weights):
            
            #set first edge weight
            #should charges be randomly set?
            # Based on my understanding of the paper, each edge needs
            # to have input and output charges.
            self._graph.edges[e]['weight'] = w
            self._graph.edges[e]['fixed'] = False # flag for fixed edges
            self._graph.edges[e]['input charges'] = np.zeros((2,2))
            self._graph.edges[e]['output charges'] = np.zeros((2,2))

        for node in self._graph:
            for nbr in self._graph[node]:
                # check if second edge already exists
                if self._graph.has_edge(node,nbr,1):
                    continue
                # create second edge and set weight
                else:
                    w = self._graph.edges[node,nbr,0]['weight']
                    self._graph.add_edge(node,nbr,weight=w)
                    self._graph.edges[node,nbr,1]['input charges'] = np.zeros((2,2))
                    self._graph.edges[node,nbr,1]['output charges'] = np.zeros((2,2))
                    self._graph.edges[node,nbr,1]['fixed'] = False # flag for fixed edges

    def _get_NNN_edges(self):
        '''
        Adds edges to a node to connect it to its next nearest neighbors.
        Will only add edges if they don't already exist.
        For bipartite graphs, will search for next-nearest neighbors
        in the opposite set.

        Assumes neighbors are already connected
        
        return set from which to add edges.
        '''
        nnn = set()
        for node in self._graph:
            for nbr in self._graph[node]:
                
                for nbr_nbr in self._graph[nbr]:
                    if self._connectivity=='NN':
                        nnn.add((node,nbr_nbr,0))
                    elif self._connectivity == 'bipartite':
                        for nbr_nbr_nbr in self._graph[nbr_nbr]:
                            if nbr_nbr_nbr not in self._graph[node]:
                                nnn.add((node,nbr_nbr_nbr,0))
                    else:
                        raise NotImplementedError('Only NN and bipartite \
                                                connectivities currently supported.')
        return nnn

    def set_bias_nodes(self, bias_nodelist):
        """
        A function to set the state of nodes to fixed values.
        Will also iterate through all edges attached to bias nodes and fix them.

        ARGUMENTS
        ---------
        bias_nodelist: list
            A list of tuples of the form (node, state, weight) that will 
            define which nodes should be fixed. The state should be an
            element of self.state_space. 
        """

        for node, state, weight in bias_nodelist:
            self._graph.nodes[node]['state'] = state
            self._graph.nodes[node]['fixed'] = True
            self._fixed_nodes.append(node)
            for nbr in self._graph[node]:
                self._graph.edges[node,nbr,0]['fixed'] = True
                self._graph.edges[node,nbr,0]['weight'] = weight
                self._graph.edges[node,nbr,1]['fixed'] = True
                self._graph.edges[node,nbr,1]['weight'] = weight

# DONE: update input and output charge functs need to be refactored to 
#       update single node

    def __update_edge_input_charge(self, edge):
        '''
        Calculates q_ij for a given edge.

        q_ij = e_ij * w_ij

        Where e_ij is the state of the node at the end of the edge.
        Charge must be placed into the correct compartment.
        '''
        w_ij = self._graph.edges[edge]['weight']

        #Done: Check for max charge
        #DONE: This indexing is not as clever as I though, it doesn't
        #       work. Fix it.


        if edge[2] == 0:
            #update toward node with smaller index
            #use potential of node with larger index
            # use our compare_nodes function to determine which node
            # is 'larger'
            larger_index = self.get_greater_node(edge[0],edge[1])
            e_ij = self._graph.nodes[larger_index]['state']
            self._graph.edges[edge]['input charges'][int(e_ij<0),int(w_ij<0)]+=e_ij*w_ij
            
        

        elif edge[2]==1:
            # update toward node with larger index
            # so use potential of node with smaller index
            smaller_index = self.get_lesser_node(edge[0],edge[1])
            e_ij = self._graph.nodes[smaller_index]['state']
            self._graph.edges[edge]['input charges'][int(e_ij<0),int(w_ij<0)]+=e_ij*w_ij


    def __update_edge_output_charge(self, edge):
        '''
        Calculates p_ij for a given edge.

        p_ij = e_ij * w_ij

        Where e_ij will be determined by edge[2].
        Charge must be placed into the correct compartment.
        Note that in the paper, the output charge compartment for node j,
        p_j, is defined by its coupled node i. So we need to know the state
        of node i to calculate p_j. (section 2.2.2)
        '''
        w_ij = self._graph.edges[edge]['weight']
        larger_index = self.get_greater_node(edge[0],edge[1])
        smaller_index = self.get_lesser_node(edge[0],edge[1])


        if edge[2] == 0:
            #update toward node with smaller index
            #use potential of node with LARGER index
            e_j = self._graph.nodes[larger_index]['state']
            e_i = self._graph.nodes[smaller_index]['state']
            self._graph.edges[edge]['output charges'][int(e_i<0),int(w_ij<0)]+=e_j*w_ij

        elif edge[2]==1:
            # update toward node with larger index
            # so use potential of node with SMALLER index
            e_j = self._graph.nodes[smaller_index]['state']
            e_i = self._graph.nodes[larger_index]['state']
            self._graph.edges[edge]['output charges'][int(e_i<0),int(w_ij<0)]+=e_j*w_ij

        
    def __enforce_edge_charge_limits(self, edge):
        """
        Make sure edges don't accumulate too much charge.
        
        Checks to see if edge has property `fixed` set to True.
        In that case, will do nothing, because we won't enforce charge limits, I think?

        """
        if self._graph.edges[edge]['fixed']:
            return
        
        max_charge = np.max(self._state_space)
        min_charge = np.min(self._state_space)


        self._graph.edges[edge]['output charges'] = np.where(self._graph.edges[edge]['output charges']>max_charge,
                                                                    max_charge, self._graph.edges[edge]['output charges'])
        self._graph.edges[edge]['output charges'] = np.where(self._graph.edges[edge]['output charges']<min_charge,
                                                                    min_charge, self._graph.edges[edge]['output charges'])
        self._graph.edges[edge]['input charges'] = np.where(self._graph.edges[edge]['input charges']>max_charge,
                                                                    max_charge, self._graph.edges[edge]['input charges'])
        self._graph.edges[edge]['input charges'] = np.where(self._graph.edges[edge]['input charges']<min_charge,
                                                                    min_charge, self._graph.edges[edge]['input charges'])


    def __update_node_input_charges(self, node):
        """
        Based on the current network state and edge weights,
        calculate input charge compartments
        for a node.

        Compartment charge updates are described in 
        section 2.2.2 of the paper referenced above.

        We're updating the property of the nodes called
        G.nodes[n]['input']

        Where input charges are a 2x2 array of floats
        The input array corresponds to q_j in the paper.

        q_j =  [[q++, q+-],
                [q-+, q--]]]

        """

        
        for nbr in self._graph[node]:
            index = self.compare_nodes(nbr,node)

            self._graph.nodes[node]['input'] += self._graph.edges[node,nbr,index]['input charges']

        
    def __update_node_output_charges(self,node):
        """
        Based on the current network state and edge weights,
        calculate output charge compartments
        for each node.

        Compartment charge updates are described in 
        section 2.2.2 of the paper referenced above.

        We're updating the property of the nodes called
        G.nodes[n]['output']

        Where each of these proprties is a 2x2 array of floats
        The input array corresponds to q_j in the paper.

        p_j =  [[p++, p+-],
                [p-+, p--]]]

        """

        for nbr in self._graph[node]:
            index = self.compare_nodes(node,nbr)
            self._graph.nodes[node]['output'] += self._graph.edges[node,nbr,index]['output charges']

    def __zero_node_charges(self,node):
        """
        Sets the input and output charge compartments
        of a node to 0.
        """
        self._graph.nodes[node]['input'] = np.zeros((2,2))
        self._graph.nodes[node]['output'] = np.zeros((2,2))

    def __update_node_edge_compartments(self, node):
        """
        UPDATE: might not need this function
        Based on the current network state and edge weights,
        calculate edge weight compartments
        for a node.

        Edge compartment values are described in 
        section 2.2.2 of the paper referenced above.

        We're updating the property of the nodes called
        G.nodes[n]['edges']

        Where each of these proprties is a 2x2 array of floats
        w_j =  [[w++, w+-],
                [w-+, w--]]]

        During TNN operation, we'll be updating compartment weights,
        we need to keep track of which edges contribute to which compartment.
        """ 
        # We'll need to wipe the edge compartments clean
        # before we update them

        self._graph.nodes[node]['edges'] = np.zeros((2,2))
        self._graph.nodes[node]['weight_tracker'] = {'w++':[],'w+-':[],'w-+':[],'w--':[]}   

        

        for nbr in self._graph[node]:
            which_edge = int(self.compare_nodes(nbr, node)) # 0 if nbr > node, 1 if nbr < node
            e_i = self._graph.nodes[nbr]['state']
            w_ij = self._graph.edges[(node,nbr,which_edge)]['weight']
            edge = (node,nbr,which_edge)
            if e_i>0:
                if w_ij>0:
                    #update w++
                    self._graph.nodes[node]['edges'][0,0] += w_ij
                    self._graph.nodes[node]['weight_tracker']['w++'].append(edge)
                else:
                    #update w+-
                    self._graph.nodes[node]['edges'][0,1] += w_ij
                    self._graph.nodes[node]['weight_tracker']['w+-'].append(edge)

                

            else:
                if w_ij > 0:
                    # update w-+
                    self._graph.nodes[node]['edges'][1,0] += w_ij
                    self._graph.nodes[node]['weight_tracker']['w-+'].append(edge)

                else:
                    #update w--
                    self._graph.nodes[node]['edges'][1,1] += w_ij
                    self._graph.nodes[node]['weight_tracker']['w--'].append(edge)

        

    def __calculate_network_energy(self):
        """
        Calculates total network energy by
        calculating and summing over node energies.
        """
        network_energy = 0

        for node in self._graph:
            node_energy = self.get_node_energy(node)
            network_energy += node_energy

        self._energy = network_energy

        return network_energy

    def __calculate_node_energy(self, node, state=None):
        """
        Calculates energy of a node using Eq 5
        in Section 2.2.3 of reference above.
        Sets the 'energy' property of the node to 
        the result of this calculation.

        H_j = -4 * [(f_j_+)^2 * (q_++ * p_+- + q_+- * p_++) +\
                    (f_j_-)^2 * (q_-- * p_-+ + q_-+ * p_--)]

        where q, p are the input and output charge compartments,
        respectively. They are 2x2 arrays with elements defined like this:
        [[++, +-],
         [-+, --]]

         f_j is the Heaviside step function, h, in the paper.
         f_j_(+/-)(e_j) = h (-/+ e_j)
         Where e_j is the state of node j.

         I think in principle you can use any logistic-ish function
         as f. I think a more ~*thermodynamic*~ way to do this is 
         to use some function of the network temperature, like
         the Fermi-Dirac distribution (if we're talking about 'charge')

        ARGUMENTS
        ---------

        node: networkx.Node
            Node object defining the node on which we are 
            calculating the energy

        state: float
            Element of self.state_space or None. If None,
            then function will use current node state. Otherwise
            pretend node has user provided state. The latter
            option useful for sampling the Boltzmann distribution 
            during node update.

        RETURNS
        -------
        node_energy: float
            Energy of node calculated as in function
            description.

        """
        if state is not None and state in self._state_space:
            e_j = state
        else:
            e_j = self._graph.nodes[node]['state']


        q = self._graph.nodes[node]['input']
        p = self._graph.nodes[node]['output']

        first_term = np.heaviside(-e_j, 0.5)**2 * (q[0,0]*p[0,1] + q[0,1]*p[0,0])
        second_term = np.heaviside(e_j, 0.5)**2 * (q[1,1]*p[1,0] + q[1,0]*p[1,1])

        node_energy = -4. * (first_term + second_term)

        self._graph.nodes[node]['energy']=node_energy

        return node_energy

    def update_energy_series(self):

        self._energy_series.append(self.__calculate_network_energy())

    def __update_node_energy_series(self, node):
        self._graph.nodes[node]['energy_series'].append(self.get_node_energy(node))

    def update_network_state(self):
        """
        Updates all node states, network weights
        should this function even be here? 
        """
        # DONE: This block should move to `update_network_state`
        for edge in self._graph.edges:
            self.__update_edge_input_charge(edge)
            self.__update_edge_output_charge(edge)
            self.__enforce_edge_charge_limits(edge)
            

        for node in self._graph:
            self.__update_node_state(node)
        
        self.__record_state()


    def __update_node_state(self, node):
        """
        Function to update node state.
        TODO: Figure out what actually is supposed to happen
                to output compartment charges.
        Algorithm:
        1. Update input charges
        2. Sample node state from Boltzman distribution
            as per Eq 17 in section 2.2.5 of reference.
        3. Select state
        4. Recalculate node energy 
        5. If update is IRREVERSIBLE:
        6. 'Communicates its state to its connected nodes'- what does this mean?
        7. Update edge weights per section 2.2.4
        8. Dissipate residual charge in selected compartment per eq 16
    
        This function should be called with node charges zero'd out.
        Need to call node input and output charges before starting the uodate.

        08/31/2023: Decided that it's dumb to have changes communicated in the middle of 
        a time step. The weights will be changed if the update is irreversible, but
        the changes won't be communicated until the next time step. Similarly, a
        reversible update won't result in changes to charge on the edges until 
        all nodes have been updated. This simplifies a lot of accounting. 

        09/06/2023: Checks for bias node and does not update if node is fixed.
        """
        if self._graph.nodes[node]['fixed']:
            return self._graph.nodes[node]['state'], 'FIXED'
        # Done: This block should move to `__update_node_state`
        self.__update_node_input_charges(node)
        self.__update_node_output_charges(node)
        self.__update_node_edge_compartments(node)


        # Calculate energy of each possible state
        node_energy_list = np.array([self.__calculate_node_energy(node,state=state) for state in self._state_space])
        
        #calculate probability of each possible state
        #note that we need to subtract the ground state energy 
        #from each energy to avoid positive exponents
        node_energy_list -= np.min(node_energy_list)
        probs = np.exp(-node_energy_list/self.node_temperature)
        probs /= np.sum(probs)

        new_state = np.random.choice(self._state_space, p=probs)

        #set new state, energy
        self._graph.nodes[node]['state']=new_state
        self.__calculate_node_energy(node)
        new_energy = self.get_node_energy(node)
        self._graph.nodes[node]['energy_series'].append(new_energy) 


        #check for reversibility
        if len(self._graph.nodes[node]['energy_series']) > 10:
            node_history = self._graph.nodes[node]['energy_series'][-10:]
        else:
            node_history = self._graph.nodes[node]['energy_series']

        if np.std(node_history) <= self.node_temperature:
            # update is irreversible
            update_type = 'IRREV'
            #DONE: update weights and dissipate charges
            self.__update_edge_weights(node)
        else:
            #update is reversible
            #nothing further need be done
            update_type = 'REV'

        #zero node charges
        self.__zero_node_charges(node)
        

        return new_state, update_type


    def __update_edge_weights(self, node):
        """
        I have read the relevant section in the paper (2.2.4)
        multiple times and STILL don't understand how the 
        weights are supposed to be updated. The implementation
        in the Thermodynamic Neural Network repo 
        (synapse_v21.py, Real1 class) does not really
        appear to match what is written in the paper in a way 
        I can understand.

        OK, I think I did it.
        Will sample and update the edge weights according to
        eq 14 from 2.2.4 of reference.
        Will then dissipate charge in selected compartments. 
        """
        if self._graph.nodes[node]['fixed']:
            return

        resid_charge = np.zeros(shape=(2,2)) #residual charge in each compartment
        ei_arr = np.zeros(shape=(2,2)) #aggregated compartment input potentials
        e_j = self.get_node_state(node)
        abs_ej = np.abs(e_j)
        num_nbrs = len(self._graph[node])
        input_charges = self._graph.nodes[node]['input']
        weight_compartments = self._graph.nodes[node]['edges']

        for nbr in self._graph[node]:
            which_edge = int(self.compare_nodes(node, nbr)) # 0 if nbr < node, 1 if nbr > node
            e_i = self.get_node_state(nbr)
            w_ij = self.get_single_edge_weight(node, nbr, which_edge)
            
            ei_index = int(e_i<0)
            w_index = int(w_ij<0)

            resid_charge[ei_index,w_index] = np.heaviside((-1)**(1-ei_index) * e_j, 0.5) *\
                                             (input_charges[ei_index,w_index] - e_j *\
                                              weight_compartments[ei_index,(w_index+1)%2])
            ei_arr[ei_index, w_index] += np.abs(e_i)


        Delta = ei_arr + np.ones(shape=(2,2)) * num_nbrs * abs_ej

        eta = (ei_arr @ resid_charge) / Delta
        mu = ((num_nbrs * abs_ej * np.ones(shape=(2,2))) @ resid_charge) / Delta

        # now we need to sample from a normal distribution
        # gonna write this the dumb way for now
        # Since exponentiating Eq 14 gives us independent gaussians
        # we can just do the exponent and update in the for-loop
        # how do we define the delta w_ij?
        
        # TODO: this needs to be in its own function
        # TODO: Compartments can, in principle, have multiple edges. Update should be 
        #       evenly distributed over them.
        for key in self._graph.nodes[node]['weight_tracker']:
            for edge in self._graph.nodes[node]['weight_tracker'][key]:
                if key == 'w_++':
                    r,c = 0,0
                elif key == 'w_+-':
                    r,c = 0,1
                elif key == 'w_-+':
                    r,c = 1,0
                elif key == 'w_--':
                    r,c = 1,1
                else:
                    #some compartments will be empty
                    continue
                e_j = self._graph.nodes[node]['state']
                nbr = edge[0] if edge[1] == node else edge[1]
                e_i = self._graph.nodes[nbr]['state']
                w_ij = self._graph.edges[edge]['weight']
                #TODO: need to define dels in a more sane way.
                dels = self._weight_space - w_ij
                energies = self._calc_delta_energy(e_i, e_j, dels, Delta, resid_charge, r,c)
                del_wij = dels[self.boltzmann_sample(energies, self.edge_temperature)]

                # update the weight and make sure to adjust for mean/variance explosion
                # don't forget to do this for both 0 and 1 edge indexes
                new_weight = (w_ij + del_wij)*np.sqrt(1/(1+num_nbrs/(self.edge_temperature*2*(w_ij+del_wij)**2)))
                self._graph.edges[edge]['weight'] = new_weight
                self._graph.edges[edge[0],edge[1],int(not edge[2])]['weight'] = new_weight

                # dissipate charge in compartment
                if not np.heaviside(e_j,0.5):
                    # need to determine which edge compartment to dissipate charge from
                    # We need to dissipate input and output charges, i think
                    # so we need to know which node is larger
                    if self.compare_nodes(node,nbr): #node < nbr
                        self._graph.edges[edge[0],edge[1],0]['input charges'] = 0
                        self._graph.edges[edge[0],edge[1],1]['output charges'] = 0

                    else:
                        self._graph.edges[edge[0],edge[1],1]['input charges'] = 0
                        self._graph.edges[edge[0],edge[1],0]['output charges'] = 0


  
    def _calc_delta_energy(self, e_i, e_j, dels, Delta, resid_charge, r,c):
        """
        Calculates the energy difference for a given
        set of weights.

        ARGUMENTS
        ---------
        e_i: float
            Input potential of the node
        e_j: float
            Output potential of the node
        dels: np.array
            Array of weight differences
        Delta: np.array
            Array of weight sums
        resid_charge: np.array
            Array of residual charges
        r: int
            Row index of the weight compartment
        c: int
            Column index of the weight compartment

        RETURNS
        -------
        energies: np.array
            Array of energies for the given weight differences
        """
        state_sum_sqr = e_j**2 + e_i**2
        first_term = dels
        second_term = e_i*resid_charge[r,c]*e_i**2/(np.abs(e_i)*Delta[r,c])
        third_term = e_j*resid_charge[r,int(not c)]*e_j**2/(np.abs(e_j)*Delta[r,int(not c)])

        energy = state_sum_sqr*(first_term - (second_term + third_term)/(state_sum_sqr))**2

        return energy


    def boltzmann_sample(self, energies, temperature, k_b=1.0):
        """
        Samples from a Boltzmann distribution
        with energies and temperature.

        Energies should be a list of floats.

        We will usually take k_B = 1.0, but user can modify.

        Will return index of energy selection, not the energy itself.

        ARGUMENTS
        ---------
        energies: list of floats
            Energies to sample from
        temperature: float
            Temperature of the system
        k_b: float
            Boltzmann constant. Default is 1.0
        
        RETURNS
        -------
        state: int
            Index of the state selected
        """
        beta = 1.0 / (k_b * temperature)
        prob = np.exp(-beta * energies)
        prob /= np.sum(prob)
    
        # Select a state based on the weighted probabilities
        state = np.random.choice(len(energies), p=prob)
        return state


    def __record_state(self):

        for i,(_,n_state) in enumerate(self._graph.nodes.data('state')):
            self._state[i] = n_state
        
    def compare_nodes(self,node_a, node_b):
        # Determines whether node_a is 'less than' node_b
        # using modular arithmetic.
        # This should help do node comparisons in a way that
        # is consistent with the periodic boundary conditions.
        
        N = len(self._graph)
        return (node_a - node_b) % N < (node_b - node_a) % N
    
    def get_greater_node(self, node_a, node_b):
        return node_a if self.compare_nodes(node_b, node_a) else node_b
    
    def get_lesser_node(self, node_a, node_b):
        return node_a if self.compare_nodes(node_a, node_b) else node_b

    def set_node_temperature(self, temperature):
        self.node_temperature = temperature

    def set_edge_temperature(self, temperature):
        self.edge_temperature = temperature

    def set_node_state(self, node, state):
        if state not in self._state_space:
            raise ValueError('State must be an element of self._state_space')
        self._graph.nodes[node]['state'] = state

    def get_graph(self):

        return self._graph

    def get_state(self):
        """
        Returns the State of the Network
        as an array with shape self._size
        """

        return np.reshape(self._state, newshape=self._size)

    def get_edge_weights(self):
        """
        Returns list of edge weights
        as a tuple of the form
        [(n1,n2,w_12), ..., (ni,nj,w_ij),...]
        """
        edge_weights = []
        for e_w in self._graph.edges.data('weight'):
            edge_weights.append(e_w)
        return edge_weights

    def get_energy(self):

        return self._energy
    
    def get_energy_series(self):
        return self._energy_series    
    def get_node_energy(self, node):

        return self._graph.nodes[node]['energy']

    def get_node_state(self, node):
        
        return self._graph.nodes[node]['state']

    def get_single_edge_weight(self, node, nbr, index):

        return self._graph[node][nbr][index]['weight']

    def get_single_edge_charge(self, node, nbr, index):
        '''
        Returns the charge on an edge. 
        '''

        
        return self._graph[node][nbr][index]['charge']
    
    def get_node_temperature(self):
        return self.node_temperature
    
    def get_edge_temperature(self):
        return self.edge_temperature
        


