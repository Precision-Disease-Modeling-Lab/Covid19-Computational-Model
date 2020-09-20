# covid simulation

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import random
from scipy import stats
from tqdm import tqdm
import seaborn as sns
import pickle
import os


contry = 'Italy'
infection_p0 = 0.2
mean_num_of_neighbors = 15
num_Of_age_groups=4




if contry=='Israel' : 
    mean_family_size=3.32 # Israel
    infection_by_age =   {1: {'age_group': [0,14], 'precentage' : 27.73,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.8, 
                            'death_rate' : 0.0},
                        2: {'age_group': [15,34],  'precentage' : 27.0,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.6, 
                            'death_rate' : 0.0015},
                        3: {'age_group': [35,54],  'precentage' : 22.8,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.4, 
                            'death_rate' : 0.01},
                        4: {'age_group': [55,100], 'precentage' : 19.33,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]),
                            'asymptomatic_rate': 0.2, 
                            'death_rate' : 0.24}
                        }

elif contry=='Italy': 
    mean_family_size=2.38  # Italy
    infection_by_age =   {1: {'age_group': [0,14], 'precentage' : 13.6,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.8, 
                            'death_rate' : 0.0},
                        2: {'age_group': [15,24],  'precentage' : 9.61,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.6, 
                            'death_rate' : 0.0015},
                        3: {'age_group': [25,54],  'precentage' : 41.82,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]), 
                            'asymptomatic_rate' : 0.4, 
                            'death_rate' : 0.01},
                        4: {'age_group': [55,100], 'precentage' : 34.98,  
                            'p' : dict([(x,0.1) for x in range(1,num_Of_age_groups+1)]),
                            'asymptomatic_rate': 0.2, 
                            'death_rate' : 0.24}
                        }
## build_population

def is_free_node(G,node): 
    return 'group' not in G.nodes[node]


def find_free_neighbors(G, seed): 
    return [x for x in nx.all_neighbors(G,seed) if is_free_node(G,x)] 


def find_free_node(G, min_neighbors=2):
    free_node_list=[x for x,y in G.nodes(data=True) if 'group' not in y]
    random.shuffle(free_node_list)
    
    for node in free_node_list: 
        if len(find_free_neighbors(G, node)) >= min_neighbors:
            return node 

    return None  

## Age dist
class MixtureModel(stats.rv_continuous):
    def __init__(self, submodels, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.p = [x/sum(p) for x in p]

    def _pdf(self, x):
        print(self.p)
        pdf = self.p[0] * self.submodels[0].pdf(x)
        for idx, submodel in enumerate(self.submodels[1:]):
            pdf += self.p[idx+1] * submodel.pdf(x)
        #pdf /= len(self.submodels)
        return pdf

    def rvs(self, size):
        #submodel_choices = np.random.uniform(len(self.submodels), size=size)
        submodel_choices = np.random.choice(len(self.submodels), size=size, p=self.p )
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def age_draw(samples_in, num_of_elements, age_range = [0,100]): 
    selected_ages = []
    numOfReties=0
    while len(selected_ages) < num_of_elements: 
        sample = np.random.choice(samples_in)
        numOfReties +=1
        if numOfReties % 100 == 0 : 
            age_range[1] = 1.1 * age_range[1]
            
        if (sample > age_range[0]) & (sample <= age_range[1]): 
            samples_in.remove(sample)
            selected_ages.append(sample)
            
    if numOfReties > 100: 
        pass
        #print('update age to : ', round(age_range[1]))
    return selected_ages


def match_neighbors_per_age_group(G, seed, add_new=True, verbose=False): 
    if verbose: 
        print('node:{}'.format(seed))
    age_group=find_age_group(G.nodes[seed]['age'])
    
    # from all non-family members, find those with different age  and remove connections
    non_family_list=find_non_family_members(G, seed)
    members_to_remove=[x for x in non_family_list if find_age_group(G.nodes[x]['age']) != age_group]
    if verbose: 
        print('{} conection were removed.'.format(len(members_to_remove)))
    for m in members_to_remove: 
        #print('remove edge: {0}-{1}'.format(seed,m))
        G.remove_edge(seed,m)

    if add_new: 
        # find non-family node with the same age_group and add new edges between them
        potential_neighbors=[x for x,y in G.nodes(data=True) if (find_age_group(y['age'])==age_group)]
        # filter out if within the same family
        if 'group' in G.nodes[seed]:
            potential_neighbors=filter_per_attrib(G, potential_neighbors, 'group',G.nodes[seed]['group'],eq=False)
        
        # keeps num of connections within statisics
        #num_of_connections_to_add = int(np.random.normal(loc=mean_num_of_neighbors,scale=4) - len(find_family_members(G, seed)))
        num_of_connections_to_add = len(members_to_remove)
        
        if (len(potential_neighbors) > num_of_connections_to_add) & (num_of_connections_to_add > 0):
            if verbose: 
                print('{} conection were added.'.format(num_of_connections_to_add))
            for m in np.random.choice(potential_neighbors, size=num_of_connections_to_add, replace=False): 
                #print('add edge: {0}-{1}'.format(seed,m))
                G.add_edge(seed,m)
        else: 
            print('not enough neighbors')
        


def rematch_neighbors(G, seed): 

    age_group=find_age_group(G.nodes[seed]['age'])
    total_neighbors = list(nx.all_neighbors(G,seed))

    family_size= len(find_family_members(G,seed))
    rand_num_of_neighbors = int(np.random.normal(loc=mean_num_of_neighbors ,scale=4))

    if  len(total_neighbors) > rand_num_of_neighbors:
        #print('remove connections')
        num_of_connection_to_remove =len(total_neighbors) - rand_num_of_neighbors-family_size

        potential_neighbors_to_remove = [x for x in total_neighbors 
                                    if (x not in find_family_members(G,seed)) & 
                                        (find_age_group(G.nodes[x]['age'])) != age_group]

        # print('\n total_neighbors:', len(total_neighbors), 
        # '\n rand_num_of_neighbors:',rand_num_of_neighbors, 
        # '\n num_of_connection_to_remove:',num_of_connection_to_remove, 
        # '\n family_size:',family_size) 
        # print(potential_neighbors_to_remove)

        if (len(potential_neighbors_to_remove) >=  num_of_connection_to_remove) & (num_of_connection_to_remove>0):
            connection_to_remove = np.random.choice(potential_neighbors_to_remove, size=num_of_connection_to_remove, replace=False)
            for m in connection_to_remove: 
                G.remove_edge(seed,m)
            

    elif len(total_neighbors) < rand_num_of_neighbors : 
        #print('add connections')
        potential_neighbors_to_add=[x for x,y in G.nodes(data=True) 
                                    if  (find_age_group(y['age'])==age_group) &
                                        (x not in find_family_members(G,seed))]
 
        num_of_connections_to_add = rand_num_of_neighbors - len(total_neighbors) 
        
        if (len(potential_neighbors_to_add) > num_of_connections_to_add) & (num_of_connections_to_add > 0):
            connections_to_add = np.random.choice(potential_neighbors_to_add, size=num_of_connections_to_add, replace=False)
            #print(connections_to_add)
            for m in connections_to_add: 
                G.add_edge(seed,m)
      
     
            
            

## Simulation
def update_infection_table(p_general, p_diag, p_eldery=None):
    for n in range(1,num_Of_age_groups+1): 
        for m in range(1,num_Of_age_groups+1):
            infection_by_age[n]['p'][m]=p_general

    for k in range(1,num_Of_age_groups+1): 
        infection_by_age[k]['p'][k]=p_diag
    if p_eldery != None: 
        print(p_eldery)
        for q in range(1,num_Of_age_groups): 
            infection_by_age[num_Of_age_groups]['p'][q]=p_eldery
            infection_by_age[q]['p'][num_Of_age_groups]=p_eldery
            

            
            



def find_age_group(age_in):
    age_to_group_dict={}
    for k,x in infection_by_age.items(): 
        age_range=x['age_group']
        for age in range(age_range[0], age_range[1]+1): 
            age_to_group_dict[age] = k

    return age_to_group_dict[int(age_in)]


def find_infection_probability(age1, age2): 
    """
    find infection probability between two age groups
    """
    g1=find_age_group(age1)
    g2=find_age_group(age2)
    n, m = (num_Of_age_groups,num_Of_age_groups)
    if (g1 <= n) & (g2 <= m): 
        return infection_by_age[g1]['p'][g2]
    else: 
        return 0.0
    
    
def find_family_members(G, seed, status=None): 
    family_list=[]
    if 'group' not in G.nodes[seed]: 
        return family_list

    for node in nx.neighbors(G,seed):
        try: 
            if G.nodes[node]['group'] == G.nodes[seed]['group']: 
                
                if None != status: # include filter by status
                    if G.nodes[node]['status']==status: 
                        family_list.append(node)
                else: 
                    family_list.append(node) 
        except: 
            pass
    return family_list
    
    
def find_non_family_members(G, seed, status=None):
    lst=[]
    for node in nx.neighbors(G,seed): 
        if node not in find_family_members(G, seed):
            
            if None != status: # include filter by status
                if G.nodes[node]['status']==status: 
                    lst.append(node)
            else: 
                lst.append(node) 
        
    return lst
    
    
def binary_choise(p): 
        return np.random.choice([0, 1], size=1, p=[1-p, p])[0]  

    
    
def select_attr_per_time(G, attr, t): 
    lst=[]
    for node, data in G.nodes(data=True):
        try: 
            if  G.nodes[node]['exposed_time'] + G.nodes[node][attr] == t: 
                lst.append(node)
        except:
            pass
    return lst    
    
    
    
def filter_per_attrib(G, nodes_list, attr, val, eq = True):
    lst=[]
    for node in nodes_list:
        try:
            if eq: 
                if G.nodes[node][attr] == val: 
                    lst.append(node)
            else: 
                if G.nodes[node][attr] != val: 
                    lst.append(node)
        except:
            pass
    return lst  


def select_per_attr(G, attr, val, eq = True): 
    return filter_per_attrib(G, G.nodes(), attr=attr, val=val, eq = eq)    
    
def status_list(G, status, details=False): 
    if details :
        return [(x,y) for x,y in G.nodes(data=True) if y['status']==status]
    else: 
        return [x for x,y in G.nodes(data=True) if y['status']==status]    
    
    
    
def infection(G, seed, t, verbose=False):
    
    # close family infection with probability
    if 'group' in G.nodes[seed]: 
        if verbose:
            #pass
            print('family infection part ({})'.format(G.nodes[seed]['group']))
        for member in find_family_members(G, seed, status='S'): 
            if binary_choise(infection_p0):
                update_infection_details(G, seed, member, t, verbose=verbose)

                
    # non-familty  neighbors infections  
    non_family_neighbors= find_non_family_members(G, seed, status='S')
    for member in non_family_neighbors: 
        p = find_infection_probability(G.nodes[seed]['age'], G.nodes[member]['age'])
        if binary_choise(p):
            update_infection_details(G, seed, member, t, verbose=verbose) 
            
            
   
def update_infection_details(G, src, dst, t, verbose=False): 
    if G.nodes[dst]['status'] != 'S': 
        if verbose:
            print('dest node is not S state')
        return 
    
    # find the age group of the dst node
    age_group=find_age_group(G.nodes[dst]['age'])
     
    # uodate to Expose State
    G.nodes[dst]['status'] = 'E'
    G.nodes[dst]['contigaious'] = True
    G.nodes[dst]['exposed_time'] = t

    # get incubation time
    G.nodes[dst]['incubation_time'] =  int(round(stats.weibull_min.rvs(2.3, loc=0, scale=6.4))) 
    
    # get recovry time
    recovery_time=int(round(np.random.normal(loc=28, scale=14)))
    if recovery_time < 1: 
        # incase recovery time is negative
        G.nodes[dst]['recovry_time'] = 1
    else :
        G.nodes[dst]['recovry_time'] = recovery_time

    # if become asymptomatic
    if binary_choise(infection_by_age[age_group]['asymptomatic_rate']): 
        G.nodes[dst]['asymptomatic'] = True
    else: 
        # assume stop contigaious once symptoms shown
        G.nodes[dst]['asymptomatic'] = False
    

    # trace origin of infection (nice to have)
    if True: 
        if src >= 0 : 
            if 'origin' in G.nodes[src] :
                src_origin = G.nodes[src]['origin'][:]
                src_origin.append(src)
                G.nodes[dst]['origin'] = src_origin
        else:
             G.nodes[dst]['origin'] = [src]
        if verbose:
            print('{0} -> {1} '.format(src, dst))    
    
    
    
def run_simulation(G, T_max=100, verbose = False):

    ## set patient(s) zero
    numOfPatientZero=5
    for _ in range(numOfPatientZero): 
        seed = np.random.choice(G.nodes())
        # G.nodes[seed]['status']='E'
        # G.nodes[seed]['origin']=[-1]

        # # patient contigaious at infection event
        # G.nodes[seed]['contigaious']=True
        update_infection_details(G, -1, seed, t=-1)

    logStatus={}
    for t in tqdm(range(T_max)): 
        #print('---- Day {} ----'.format(t))

        # find all pateunt which incubaion ends today ( @ t)
        incubation_time_ends_list=select_attr_per_time(G, 'incubation_time',t)
        if incubation_time_ends_list: 
            # Once incubation ends
            if verbose: 
                print('found {} new paitents which incubation time ends'.format(len(incubation_time_ends_list)))
            for node in incubation_time_ends_list: 
                # incubation take affect only if exposed
                if ( G.nodes[node]['status'] == 'E' ):
                    if G.nodes[node]['asymptomatic']:
                        # if patient about to become asymptomatic he will remain in the E compartment and still contigaious
                        G.nodes[node]['status'] = 'E'
                        G.nodes[node]['contigaious'] = True
                    else: # if becaome symptomatic, "exposed" turn to "infected" and patient put into quarantine and stop being contigaious
                        G.nodes[node]['contigaious'] = False
                        G.nodes[node]['status'] = 'I'

        # check for recovries
        recovery_list=select_attr_per_time(G, 'recovry_time',t)
        if recovery_list:
            if verbose: 
                print('found {} new recovries'.format(len(recovery_list)))
            for node in recovery_list: 
                age_group= find_age_group(G.nodes[node]['age'])

                if binary_choise(infection_by_age[age_group]['death_rate']): 
                    G.nodes[node]['status'] = 'D'
                else :
                    G.nodes[node]['status'] = 'R'        
                G.nodes[node]['contigaious'] = False

        # infection happend for all 'contigaious' patients
        for seed in select_per_attr(G, 'contigaious', True): 
            infection(G, seed, t, verbose=verbose)


        # log results
        logStatus[t] = dict(list(map(lambda x: (x, status_list(G, status=x)), ['S','E','I','R','D'])))
    return  logStatus
        
    
    
    
    
    
def get_status_count(logStatus, t): 
    return list(map(lambda x : len(logStatus[t][x]),  ['S','E','I','R', 'D']))   
    