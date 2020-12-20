import functools
import itertools
import math
import numpy as np
import random

import utils

POP_SIZE = 100  # population size
REPEATS = 1000  # number of runs of algorithm (should be at least 10)
MAX_GEN = 5000  # maximum number of generations (premature end, so fine ...)


CX_PROB = 0.8  # crossover probability -- ok
MUT_PROB = 0.2  # mutation probability -- ok
MUT_MAX_LEN = 10  # maximum lenght of the swapped part -- not used ...

INPUT = 'inputs/tsp_std.in'  # the input file
OUT_DIR = 'tsp' # output directory for logs
EXP_ID = 'default' # the ID of this experiment (used to create log names)

k=2


def generate_combinations(individual, node1, node2, node3, locations__):

    # pre = fitness(individual, locations__).objective
    """
    This method generate 8 possible combinations from a list of cities
    :param individual: List of cities
    :param node1: [a, b]
    :param node2: [c, d]
    :param node3: [e, f]
    :return: Best combination i.e. list of cities, tour cost
    """
    """Combo 1: Same as the original node : Everything till Node 1 -> Node 1 to Node2 -> Node2 to Node 3 -> Node3 
    to Everything 
    """
    combo_1 = individual[:node1[0] + 1] + individual[node1[1]:node2[0]+1]     + individual[node2[1]: node3[0]+1]     + individual[node3[1]:]
    combo_2 = individual[:node1[0] + 1] + individual[node1[1]:node2[0]+1]     + individual[node3[0]: node2[1]-1: -1] + individual[node3[1]:]
    combo_3 = individual[:node1[0] + 1] + individual[node2[0]:node1[1]-1: -1] + individual[node2[1]: node3[0]+1]     + individual[node3[1]:]
    combo_4 = individual[:node1[0] + 1] + individual[node2[0]:node1[1]-1: -1] + individual[node3[0]: node2[1]-1: -1] + individual[node3[1]:]
    combo_5 = individual[:node1[0] + 1] + individual[node2[1]:node3[0]+1]     + individual[node1[1]:node2[0]+1]      + individual[node3[1]:]
    combo_6 = individual[:node1[0] + 1] + individual[node2[1]:node3[0]+1]     + individual[node2[0]:node1[1]-1: -1]  + individual[node3[1]:]
    combo_7 = individual[:node1[0] + 1] + individual[node3[0]:node2[1]-1: -1] + individual[node1[1]:node2[0]+1]      + individual[node3[1]:]
    combo_8 = individual[:node1[0] + 1] + individual[node3[0]:node2[1]-1: -1] + individual[node2[0]:node1[1]-1: -1]  + individual[node3[1]:]

    combinations_array = [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8]
    distances_array = list(map(lambda x: fitness(x, locations__).objective, combinations_array))
    min_distance = int(np.argmin(distances_array))
    return combinations_array[min_distance], distances_array[min_distance]
    # self.random_solution = np.array(combinations_array[min_distance])
    # self.total_cost = distances_array[min_distance]
fff = 1
def k_opt_3(route, locations):
    global fff
    if fff == 1:
        print("\tk opt 3")
        fff+=1
    """
    3 OPT Local search
    Generates all possible valid combinations.
    Runs a for loop for each combination obtained above and generates 7 different combinations
    possible after 3 OPT move. Selects the one with minimum tour cost
    :param route: list of cities
    :return: updated list of cities , tour_cost
    """
    all_combinations = list(itertools.combinations(range(len(route)), 3))
    """This generates all possible sorted routes and hence eliminating the need of for loop and then sorting it 
    and hence avoiding duplicates 
    """
    # Select any random city including first and last city
    random_city = np.random.randint(low=0, high=len(route))
    # Keep only valid combinations, i.e combinations containing the random selected city
    all_combinations = list(filter(lambda x: random_city in x, all_combinations))
    # Remove consecutive numbers to avoid overlaps and invalid cities
    # all_combinations = list(filter(lambda x: x[1] != x[0] + 1 and x[2] != x[1] + 1, all_combinations))

    for idx, item in enumerate(all_combinations):
        """
        Run for every combination generated above.
        a,c,e = x,y,z  # Generated in the combination
        d,e,f = x+1, y+1, z+1  # To form the edge
        """
        # print('Iteration count is {} and item a, c, e is {}' .format(idx, item))
        a1, c1, e1 = item
        b1, d1, f1 = a1 + 1, c1 + 1, e1 + 1

        """The above generates the edge. The edge is sent to generate 7 possible combinations and the best one is 
        selected and applied to the global solution
        """
        route, _ = generate_combinations(route, [a1, b1], [c1, d1], [e1, f1], locations)

    # distance = calc_tour_cost(route)
    return route #, distance


# reads the input set of values of objects
def read_locations(filename):
    locations = []
    with open(filename) as f:
        for l in f.readlines():
            tokens = l.split(' ')
            locations.append((float(tokens[0]), float(tokens[1])))
    return locations

@functools.lru_cache(maxsize=None) # this enables caching of the values
def distance(loc1, loc2):
    # based on https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [loc1[1], loc1[0], loc2[1], loc2[0]])
    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371.01 * c
    return km

# the fitness function
def fitness(ind, cities):

    # quickly check that ind is a permutation
    num_cities = len(cities)
    assert len(ind) == num_cities
    assert sum(ind) == num_cities*(num_cities - 1)//2

    dist = 0
    for a, b in zip(ind, ind[1:]):
        dist += distance(cities[a], cities[b])

    dist += distance(cities[ind[-1]], cities[ind[0]])

    return utils.FitObjPair(fitness=-dist,
                            objective=dist)

# creates the individual (random permutation)
def create_ind(ind_len):
    ind = list(range(ind_len))
    random.shuffle(ind)
    return ind

city_cache = None
key_val_cache = {}

def distances_from(idx_city, cities):
    global city_cache
    global key_val_cache

    if city_cache is None:
        city_cache = np.zeros([len(cities), len(cities)])
        for i, i_city in enumerate(cities):
            for j, j_city in enumerate(cities):
                if i != j:
                    city_cache[i, j] = distance(i_city, j_city)
                else:
                    city_cache[i, j] = np.infty
        for key in range(city_cache.shape[0]):
            key_val_cache[key] = np.argsort(city_cache[key])
    return key_val_cache[idx_city]

aaa = -1
def create_ind_kostra(ind_len, cities):

    if np.random.random() < 0.5:
        return create_ind(ind_len)

    global aaa
    aaa+=1
    first = aaa if aaa < ind_len else np.random.randint(0, ind_len)
    # first = random.randrange(0, ind_len)

    result = [first]
    while True:
        if len(result) == ind_len:
            break
        closest_cities = distances_from(result[-1], cities)
        for i in closest_cities:
            if i not in result:
                result.append(i)
                break
    return result


# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


# the tournament selection
def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(pop[p1][:])
        else:
            selected.append(pop[p2][:])

    return selected


# implements the order crossover of two individuals
def order_cross(p1, p2):
    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    start = min(point1, point2)
    end = max(point1, point2)

    # swap the middle parts
    o1mid = p2[start:end]
    o2mid = p1[start:end]

    # take the rest of the values and remove those already used
    restp1 = [c for c in p1[end:] + p1[:end] if c not in o1mid]
    restp2 = [c for c in p2[end:] + p2[:end] if c not in o2mid]

    o1 = restp1[-start:] + o1mid + restp1[:-start]
    o2 = restp2[-start:] + o2mid + restp2[:-start]

    return o1, o2

default = False
def edge_recombination(p1, p2):
    adj = {}

    def add_to_dic(dic, key, val):
        if key in dic:
            dic[key].append(val)
        else:
            dic[key] = [val]

    def add_p_to_dic(dic, p):
        for idx, i in enumerate(p):
            if 0<idx and idx+1 < len(p):
                add_to_dic(dic, i, p[idx - 1])
                add_to_dic(dic, i, p[idx + 1])
            elif idx == 0:
                add_to_dic(dic, i, p[-1])
                add_to_dic(dic, i, p[idx + 1])
            elif idx +1 == len(p):
                add_to_dic(dic, i, p[idx - 1])
                add_to_dic(dic, i, p[0])
            else:
                raise BaseException("not possible")

    # Step 1
    add_p_to_dic(adj, p1)
    add_p_to_dic(adj, p2)
    global default
    default = False

    # Step 2
    def select_min_key(dic, curr_k, ignored):
        global default
        min_cnt = None
        for adj_k in dic[curr_k]:
            if adj_k in ignored:  # skip already used
                continue
            v = set(dic[adj_k])   # we count used ...

            # use the min, but not zero ! :-D
            tmp = len(set(dic[adj_k]) - set(ignored))
            if min_cnt is None or 0 < tmp <= min_cnt:
                min_cnt = tmp

        if min_cnt is not None:
            adj_keys = []
            for adj_k in dic[curr_k]:
                if adj_k in ignored:  # skip already used
                    continue
                tmp = len(set(dic[adj_k]) - set(ignored))
                if tmp == min_cnt:
                    adj_keys.append(adj_k)

            # Select randomly (prefer edges common to both ...)
            # print(adj_keys)
            probs = [adj_keys.count(x)/len(adj_keys) for x in set(adj_keys)]
            result = np.random.choice(list(set(adj_keys)), p=probs)
            if result not in ignored:
                return result

            # or at least possibly...
            for k in adj_keys:
                if k not in ignored:
                    return k
        # else return some random...
        else:
            # if default == False:
            #     print("default... {}".format(len(p1)- len(ignored)))
            #     default = True
            not_used = list(set(list(range(len(p1)))) - set(ignored))
            return np.random.choice(not_used)


    def fill_recursively(adj, curr_key, infant):
        if len(infant) == len(p1):
            return

        # select
        next_key = select_min_key(adj, curr_k=curr_key, ignored=infant)
        if next_key is not None:
            infant.append(next_key)
            fill_recursively(adj, next_key, infant)
        return
    # run it ...
    result = []
    first_k = np.random.choice(list(range(len(p1))))
    fill_recursively(adj, first_k, result)
    return result


def k_opt(p, locations):
    global k
    if k == 2:
        return k_opt_2(p, locations)
    elif k==3:
        return k_opt_3(p, locations)
    else:
        raise NotImplementedError("only k2 and k3 supported")


def k_opt_2(p, locations):
    def get_cities(p, idx):
        c1 = p[idx]
        c2 = p[idx+1] if idx+1 < len(p) else p[0]
        return locations[c1], locations[c2]
    reduct = 0
    best_idx_b, best_idx_c = None, None
    # pre = fitness(p, locations)

    for i in range(len(p)):
        for j in range(i+2, len(p)):
            # A---B>>>>>>>>>>>>>C--D
            a, b = get_cities(p, i)
            c, d = get_cities(p, j)
            # A---C<<<<<<<<<<<<B---D
            a_b__c_d = distance(a, b) + distance(c, d)
            a_c__b_d = distance(a, c) + distance(b, d)

            # update ...
            if a_b__c_d - a_c__b_d > reduct:
                reduct = a_b__c_d - a_c__b_d
                best_idx_b, best_idx_c = i+1, j

    if best_idx_b is not None:
        # reverse ...
        # i+1 .. nutne max pred predposledni...
        begin = p[0:best_idx_b]  # nevcetne b

        # j+1 muze byt len.... v takovem pripade je end prazdny
        end = p[best_idx_c+1:] if best_idx_c+1 < len(p) else []
        reverse_part = p[best_idx_b: best_idx_c+1]  # python je s tim ok a da to i kdyz c+1 pretece..
        result = begin + [x for x in reversed(reverse_part)] + end
        # assert len(result) == len(p) == len(set(result))
        # after = fitness(result, locations)
        # assert abs(pre.objective - reduct - after.objective) < 1
        return result
    return p


# implements the swapping mutation of one individual
def swap_mutate(p, max_len):
    source = random.randrange(1, len(p) - 1)
    dest = random.randrange(1, len(p))
    lenght = random.randrange(1, min(max_len, len(p) - source))

    o = p[:]
    move = p[source:source+lenght]  # up to 10 numbers...
    o[source:source + lenght] = []  # remove ?
    if source < dest:
        dest = dest - lenght  # we removed `lenght` items - need to recompute dest

    o[dest:dest] = move  # insert ..

    return o

# applies a list of genetic operators (functions with 1 argument - population)
# to the population
def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            cross_res = cross(p1, p2)
            if isinstance(cross_res, tuple):
                (o1, o2) = cross_res
                off.append(o1)
                off.append(o2)
            else:
                off.append(cross_res)
        else:
            o1, o2 = p1[:], p2[:]
            off.append(o1)
            off.append(o2)

    # if we had crossover returning only single child, we need to fill it ...
    idxs = list(range(len(pop)))
    while len(off) < len(pop):

        p1 = pop[np.random.choice(idxs)]
        p2 = pop[np.random.choice(idxs)]
        cross_res = cross(p1, p2)
        off.append(cross_res)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    global k
    evals = 0
    same_for = 0
    bes_res = 999999
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            res = log.add_gen(fits_objs, evals)

            if abs(res[0] - res[1]) < 1 and k == 2:
                k = 3

            elif abs(res[0] - res[1]) < 1 and k == 3:
                print("fast break: {}".format(res[2]))
                break

        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)

        pop = offspring[:-1] + [max(list(zip(fits, pop)), key = lambda x: x[0])[1]]

    return pop


if __name__ == '__main__':
    # read the locations from input
    locations = read_locations(INPUT)

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind_kostra, ind_len=len(locations), cities=locations)
    fit = functools.partial(fitness, cities=locations)

    # todo -- change
    xover = functools.partial(crossover, cross=edge_recombination, cx_prob=CX_PROB)

    # todo -- change ( note that i use max_len also for k_opt, but there i pass locations...  max_len=MUT_MAX_LEN
    mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=functools.partial(k_opt, locations=locations))

    # we can use multiprocessing to evaluate fitness in parallel
    import multiprocessing
    pool = multiprocessing.Pool()

    import matplotlib.pyplot as plt

    # run the algorithm `REPEATS` times and remember the best solutions from 
    # last generations
    best_inds = []
    for run in range(REPEATS):
        print("run: {}".format(run))
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run,
                        write_immediately=True, print_frequency=5)
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        best_template = '{individual}'
        with open('resources/kmltemplate.kml') as f:
            best_template = f.read()

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            f.write(str(bi))

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best.kml', 'w') as f:
            bi_kml = [f'{locations[i][1]},{locations[i][0]},5000' for i in bi]
            bi_kml.append(f'{locations[bi[0]][1]},{locations[bi[0]][0]},5000')
            f.write(best_template.format(individual='\n'.join(bi_kml)))

        # if we used write_immediately = False, we would need to save the 
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    for i, bi in enumerate(best_inds):
        print(f'Run {i}: difference = {fit(bi).objective}')

    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)

    # read the summary log and plot the experiment
    evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
    plt.figure(figsize=(12, 8))
    utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
    plt.legend()
    plt.show()

    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned')
