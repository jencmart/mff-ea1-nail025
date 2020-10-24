import pprint
import random
import numpy as np
import functools

import utils
import multiprocessing
import matplotlib.pyplot as plt
import itertools as it

K = 10  # number of piles
DONE = False


# reads the input set of values of objects
def read_weights(filename):
    with open(filename) as f:
        return list(map(int, f.readlines()))


# computes the bin weights
# - bins are the indices of bins into which the object belongs
def bin_weights(weights, bins):
    bw = [0] * K
    for w, b in zip(weights, bins):
        bw[b] += w
    return bw


# the fitness function
def fitness_minmax(ind, weights):
    bw = bin_weights(weights, ind)
    return utils.FitObjPair(fitness=1 / (max(bw) - min(bw) + 1), objective=max(bw) - min(bw))


def fitness_root(ind, weights):
    bw = bin_weights(weights, ind)
    mu = sum(bw) / K
    fitness = 1 / (np.sum(np.sqrt(np.abs(np.array(bw) - mu))) + 1)
    return utils.FitObjPair(fitness=fitness, objective=max(bw) - min(bw))


def fitness_abs(ind, weights):
    bw = bin_weights(weights, ind)
    mu = sum(bw) / K
    fitness = 1 / (np.sum(np.abs(np.array(bw) - mu)) + 1)
    return utils.FitObjPair(fitness=fitness, objective=max(bw) - min(bw))


def fitness_pow2(ind, weights):
    bw = bin_weights(weights, ind)
    mu = sum(bw) / K
    fitness = 1 / (np.sum(np.pow(np.abs(np.array(bw) - mu, 2))) + 1)
    return utils.FitObjPair(fitness=fitness, objective=max(bw) - min(bw))


# creates the individual
def create_ind(ind_len):
    return [random.randrange(0, K) for _ in range(ind_len)]


# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


# the roulette wheel selection
def roulette_wheel_selection(pop, fits, k):
    return random.choices(pop, fits, k=k)


# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = p1[:point] + p2[point:]
    o2 = p2[:point] + p1[point:]
    return o1, o2


# implements the "bit-flip" mutation of one individual
def flip_mutate(individual, prob, upper, weights):
    for idx, i in enumerate(individual):
        if random.random() < prob:
            individual[idx] = random.randrange(0, upper)
    return individual


# implements the "bit-flip" mutation of one individual
def better_mutate(individual, prob, upper, weights):
    for _ in range(int(len(individual) / 2)):
        if random.random() < prob:

            # Sample single "Over average" hromadka and single "Under average" hromadka
            bw = np.asarray(bin_weights(weights, individual))
            mu = np.sum(bw) / K
            over = (bw - mu).clip(min=0)
            under = (mu - bw).clip(min=0)
            over_s = np.sum(over)
            under_s = np.sum(under)
            if over_s == 0 or under_s == 0:
                return individual
            over_bw_id = np.random.choice(np.arange(K), p=over / over_s)
            under_bw_id = np.random.choice(np.arange(K), p=under / under_s)

            # Create list of weights (real hromadky), but also save indices
            np_individual = np.asarray(individual)
            np_weights = np.asarray(weights)

            # Create "real hromadky" + save indices
            # over_indexes = np.where(np_individual == over_bw_id, True, False)
            # idxs_over = (np.argwhere(over_indexes)).flatten()
            # over_bw = np_weights[over_indexes]
            # under_indexes = np.where(np_individual == under_bw_id, True, False)
            # idxs_under = (np.argwhere(under_indexes)).flatten()
            # under_bw = np_weights[under_indexes]

            over_bw, under_bw, idxs_over, idxs_under = [], [], [], []
            for idx, i in enumerate(individual):
                if i == over_bw_id:
                    over_bw.append(weights[idx])
                    idxs_over.append(idx)
                if i == under_bw_id:
                    under_bw.append(weights[idx])
                    idxs_under.append(idx)
            over_bw = np.asarray(over_bw)
            under_bw = np.asarray(under_bw)
            # print(over_bw)
            # print("------------")
            # print(over_bw2)
            # assert np.array_equal(over_bw, over_bw2) == True, "dfasf"


            # Now find the BEST pair from both Hromadky w.r.t smallest distance from average
            over_bw_grid = np.repeat(np.expand_dims(over_bw, axis=0), under_bw.size, axis=0).T
            under_bw_grid = np.repeat(np.expand_dims(under_bw, axis=0), over_bw.size, axis=0)
            diff = under_bw_grid - over_bw_grid
            new_distance = np.abs(np.sum(over_bw) - mu + diff) + np.abs(np.sum(under_bw) - mu - diff)
            flat_index = np.argmin(new_distance)

            idx_over = idxs_over[flat_index % over_bw.size]
            idx_under = idxs_under[flat_index % under_bw.size]



            # Finally, perform the swap
            individual[idx_over], individual[idx_under] = individual[idx_under], individual[idx_over]

    return individual


# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate_op, mut_prob):
    return [mutate_op(individual) if random.random() < mut_prob else individual[:] for individual in pop]


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
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off


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
def evolutionary_algorithm(pop, POP_SIZE, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]

    return pop


def perform_experiment_and_log(experiment, BASE_OUT_DIR, OUT_DIR, EXP_ID, REPEATS, POP_SIZE, MAX_GEN, weights):
    cr_ind = functools.partial(create_ind, ind_len=len(weights))

    FITNESS = experiment["fitness"]
    CX_PROB = experiment["cx_p"]
    MUT_PROB = experiment["mut_p"]
    MUT_FLIP_PROB = experiment["mut_flip_p"]

    CROSSOVER = experiment["crossover"]
    MUTATION = experiment["mutation"]

    MUT_IMPL = experiment["mutation_impl"]
    CROSS_IMLP = experiment["crossover_impl"]

    fit = functools.partial(FITNESS, weights=weights)

    # Crossover
    xover = functools.partial(CROSSOVER,
                              cross=CROSS_IMLP,
                              cx_prob=CX_PROB)

    # Mutation
    mut = functools.partial(MUTATION,
                            mut_prob=MUT_PROB,
                            mutate_op=functools.partial(MUT_IMPL, prob=MUT_FLIP_PROB, upper=K, weights=weights)
                            )
    # run the algorithm `REPEATS` times and remember the best solutions from
    # last generations
    best_inds = []
    for run in range(REPEATS):
        # initialize the log structure
        log = utils.Log(OUT_DIR, EXP_ID, run, write_immediately=True, print_frequency=5)
        # create population
        pop = create_pop(POP_SIZE, cr_ind)
        # run evolution - notice we use the pool.map as the map_fn
        pop = evolutionary_algorithm(pop, POP_SIZE, MAX_GEN, fit, [xover, mut], roulette_wheel_selection,
                                     map_fn=pool.map, log=log)
        # remember the best individual from last generation, save it to file
        bi = max(pop, key=fit)
        best_inds.append(bi)

        with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
            for w, b in zip(weights, bi):
                f.write(f'{w} {b}\n')

        # if we used write_immediately = False, we would need to save the
        # files now
        # log.write_files()

    # print an overview of the best individuals from each run
    with open(f'{BASE_OUT_DIR}/all.best', 'a') as f:
        for i, bi in enumerate(best_inds):
            # exp_id, i , objective, bin_weights
            s = f'{EXP_ID},{i},{fit(bi).objective},{bin_weights(weights, bi)}'
            print(s)
            f.write(s)
            f.write("\n")

    # write summary logs for the whole experiment
    utils.summarize_experiment(OUT_DIR, EXP_ID)


if __name__ == '__main__':
    # read the weights from input
    WEIGHTS = read_weights('inputs/partition-hard.txt')
    BASE_OUT_DIR = 'partition_new'  # output directory for logs

    # we can use multiprocessing to evaluate fitness in parallel
    pool = multiprocessing.Pool()

    REPEATS = 20  # number of runs of algorithm (should be at least 10)
    max_gen = 200  # maximum number of generations
    pop_size = 1000
    # experiments = [{"fitness": fitness_minmax, "cx_p": 0.8, "mut_p": 0.2, "mut_flip_p": 0.1, "rename": "Default"},
    #                {"fitness": fitness_abs, "cx_p": 0.2, "mut_p": 0.3, "mut_flip_p": 0.01, "rename": "Best"},
    #                {"fitness": fitness_abs, "cx_p": 0.8, "mut_p": 0.2, "mut_flip_p": 0.1},
    #                {"fitness": fitness_root, "cx_p": 0.8, "mut_p": 0.2, "mut_flip_p": 0.1},
    #                {"fitness": fitness_pow2, "cx_p": 0.8, "mut_p": 0.2, "mut_flip_p": 0.1},
    #                ]
    experiments = [
        # {"fitness": fitness_minmax, "mutation_impl": flip_mutate, "crossover_impl": one_pt_cross, "cx_p": 0.8, "mut_p": 0.2, "mut_flip_p": 0.1, "rename": "Default"},
        {"fitness": fitness_abs, "mutation_impl": better_mutate, "crossover_impl": one_pt_cross, "cx_p": 0.8,
         "mut_p": 0.2, "mut_flip_p": 0.1, "rename": "Better Mutate"},

    ]
    # variants = {
    #     "fitness": [fitness_minmax, fitness_abs, fitness_root, fitness_pow2],
    #     "cx_p": [0.2, 0.5, 0.8],  # crossover prob
    #     "mut_p": [0.01, 0.05, 0.1, 0.2, 0.3],  # mutation prob
    #     "mut_flip_p": [0.01, 0.05, 0.1, 0.2, 0.3],  # prob of flipping during mutation
    #     "pop_s": [1000, 2000, 5000, 10000]  # population size
    # }
    # varNames = sorted(variants)
    # experiments = [dict(zip(varNames, prod)) for prod in it.product(*(variants[varName] for varName in varNames))]
    EXP_ID = ""

    to_plot_dirs = []
    to_plot_exps = []
    renames = {}
    for experiment in experiments:
        experiment["crossover"] = crossover
        experiment["mutation"] = mutation

        cx_prob = experiment["cx_p"]
        mut_prob = experiment["mut_p"]
        mut_flip_prob = experiment["mut_flip_p"]

        EXP_ID = experiment["fitness"].__str__().split(" ")[1][8:] \
                 + "::" + str(cx_prob) \
                 + "::" + str(mut_prob) \
                 + "::" + str(mut_flip_prob) \
                 + "::" + str(pop_size)

        OUT_DIR = BASE_OUT_DIR + '/' + EXP_ID

        perform_experiment_and_log(experiment, BASE_OUT_DIR, OUT_DIR, EXP_ID, REPEATS, pop_size, max_gen, WEIGHTS)

        # evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID)
        # plt.figure(figsize=(12, 8))
        # utils.plot_experiment(evals, lower, mean, upper, legend_name='Default settings')
        # plt.legend()
        # plt.savefig(f'{BASE_OUT_DIR}/{EXP_ID}.png')
        to_plot_dirs.append(OUT_DIR)
        to_plot_exps.append(EXP_ID)
        if "rename" in experiment:
            renames[EXP_ID] = "(" + experiment["rename"] + ") " + EXP_ID

    utils.plot_experiments2(to_plot_dirs, to_plot_exps, renames)
    plt.show()
    # plt.savefig(f'final_results.png')
