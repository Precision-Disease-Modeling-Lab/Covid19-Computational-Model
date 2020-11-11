N=10010
type_of_network = 3
from covid_simulation import * 


type_of_network_dict = {1: 'Erdős-Rényi graph', 2: 'Barabási–Albert graph', 3: 'Watts–Strogatz small-world graph'}

type_of_network_dict[type_of_network]
data_dir = os.path.join('../data','data_' + str(type_of_network)+'_'+ str(N))

if not os.path.exists(data_dir): 
    os.mkdir(data_dir)


# create empty graph
if type_of_network == 1:  
    G = nx.fast_gnp_random_graph(n=N, p=mean_num_of_neighbors/(N-1))
elif type_of_network ==2 : 
    G = nx.barabasi_albert_graph(n=N,m=mean_num_of_neighbors)
elif type_of_network == 3 : 
    G = nx.watts_strogatz_graph(n=N,k=mean_num_of_neighbors, p=0.5)

# Set all nodes to 'S' status
for node in G.nodes: 
    G.nodes[node]['status'] = 'S'


# Group into families
family_size_list=sorted([int(np.round(x)) for x in np.random.normal(loc=mean_family_size, scale=1.0, size=int(N/mean_family_size))], reverse=True)





for group_index, family_size in enumerate(family_size_list): 
    seed = find_free_node(G, min_neighbors=family_size)
    if not seed: 
        print('no free node was found for family size of {}'.format(family_size))
        continue 
        
    neigbors=find_free_neighbors(G,seed)   
    if neigbors: 
        G.nodes[seed]['group'] = group_index
        selected_nodes=list(np.random.choice(neigbors, size=family_size, replace=False))
        selected_nodes.append(seed)    
        for node in selected_nodes: 
            if 'group' in G.nodes[node]: 
                #print('!!!! group was already y assigned !!!!')
                pass
            G.nodes[node]['group'] = group_index

nx.write_gpickle(G, os.path.join(data_dir,'G.p'))            


# Assign age per distribution
p=[x['precentage'] for k, x in infection_by_age.items()]
p /= np.sum(p)

# normal dist
age_stats=[(np.mean(x['age_group']), np.diff(x['age_group'])[0]/2) for k, x in infection_by_age.items()]

age_distribution_model=MixtureModel( submodels=[stats.norm(x[0], x[1]) for x in age_stats], p=p)

age_samples = list(np.random.choice([x for x in age_distribution_model.rvs(round(1.01 * N), ) if (x>0) & (x<100)], size=round(1.001 * N)))


# assign age for families
family_list=set([y['group'] for node, y in G.nodes(data=True) if 'group' in y])
for group_idx in family_list: 
    family=select_per_attr(G, 'group', group_idx)
    family_len=len(family)
    if family_len > 2 : 
        for idx, age in enumerate(age_draw(age_samples, 2, age_range = [20,60])): 
            y=family.pop(0) # take the first element from famility list and assgin random age
            G.nodes[y]['age'] = age
            #print('({0}):{1}->{2}'.format(group_idx,y,age))
        
        for idx, age in enumerate(age_draw(age_samples, family_len - 2, age_range = [0,20])): 
            y=family.pop(0) # take the first element from famility list and assgin random age
            G.nodes[y]['age'] = age        
            #print('({0}):{1}->{2}'.format(group_idx,y,age))
            
    elif family_len == 2: 
        for idx, age in enumerate(age_draw(age_samples, family_len, age_range = [55,100])): 
            y=family.pop(0) # take the first element from famility list and assgin random age
            G.nodes[y]['age'] = age  
            #print('({0}):{1}->{2}'.format(group_idx,y,age))
    else: 
        for idx, age in enumerate(age_draw(age_samples, family_len, age_range = [20,100])): 
            y=family.pop(0) # take the first element from famility list and assgin random age
            G.nodes[y]['age'] = age            
            #print('({0}):{1}->{2}'.format(group_idx,y,age))



# assign age for the remains: 
non_aged_list=[node for node, y in G.nodes(data=True) if 'age' not in y]
for node in non_aged_list:    
    age=age_draw(age_samples, 1)[0]
    G.nodes[node]['age']=age


# save into pickle
nx.write_gpickle(G, os.path.join(data_dir,'G_aged.p'))

## Set equi-age connections
G=nx.read_gpickle(os.path.join(data_dir,'G_aged.p'))
nodes_list=list(range(N))
random.shuffle(nodes_list) # make sure changes are done randomlly
for node in nodes_list: 
    match_neighbors_per_age_group(G, node)

# save into pickle
nx.write_gpickle(G, os.path.join(data_dir,'G_equi_aged.p'))

## Set equi-age for eldery group connections
G=nx.read_gpickle(os.path.join(data_dir,'G_aged.p'))

for node in G.nodes(): 
    if find_age_group(G.nodes[node]['age']) == 4:  
        rematch_neighbors(G, node)

# save into pickle
nx.write_gpickle(G, os.path.join(data_dir,'G_aged_eldery_grouped.p'))