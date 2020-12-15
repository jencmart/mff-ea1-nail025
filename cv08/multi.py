import copy
import functools
import numpy as np
import operator
import random

import moo_functions as mf
import multi_utils as mu
import utils


class MultiIndividual:

    def __init__(self, x):
        self.x = x
        self.fitness = None
        self.ssc = None
        self.front = None


# creates the individual
def create_ind(ind_len):
    return MultiIndividual(np.random.uniform(0, 1, size=(ind_len,)))


# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


# the tournament selection (roulette wheell would not work, because we can have
# negative fitness)
def tournament_selection_NSGA2(pop, k):
    selected = []
    for i in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if (pop[p1].front, -pop[p1].ssc) < (pop[p2].front, -pop[p2].ssc):  # lexicographic comparison
            selected.append(copy.deepcopy(pop[p1]))
        else:
            selected.append(copy.deepcopy(pop[p2]))

    return selected


def nsga2_select(pop, k):
    fronts = mu.divide_fronts(pop)
    selected = []
    for i, f in enumerate(fronts):
        mu.assign_ssc_hyper_volume(f)
        for ind in f:
            ind.front = i
        if len(selected) + len(f) <= k:
            selected += f
        else:
            break

    assert len(selected) <= k
    assert len(f) + len(selected) >= k

    if len(selected) != k:
        # f is now the front that did not fit fully
        selected += list(sorted(f, key=lambda x: -x.ssc))[:k - len(selected)]

    assert len(selected) == k

    return selected


# implements the one-point crossover of two individuals
def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1.x))
    p1 = copy.deepcopy(p1)
    p2 = copy.deepcopy(p2)
    o1 = np.append(p1.x[:point], p2.x[point:])
    o2 = np.append(p2.x[:point], p1.x[point:])
    p1.x = o1
    p2.x = o2
    return p1, p2


# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class Mutation:

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, ind, population):
        a = ind.x + self.step_size * np.random.normal(size=ind.x.shape)
        np.clip(a, 0, 1, ind.x)
        return ind
    def episodeEnd(self):
        pass


class MutationGradient():
    def __init__(self, step_size=0.8, decay=1):
        self.f = step_size
        self.decay = decay

    def __call__(self, ind, population):
        i1 = population[np.random.randint(low=0, high=len(population))]  # individual 1
        i2 = population[np.random.randint(low=0, high=len(population))]  # individual 2
        a = ind.x + self.f * (i1.x - i2.x)   # posunu ve smeru rozdilu
        np.clip(a, 0, 1, ind.x)
        return ind

    def episodeEnd(self):
        self.f = self.f*self.decay


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
            o1, o2 = copy.deepcopy(p1), copy.deepcopy(p2)
        off.append(o1)
        off.append(o2)
    return off


# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    x = [mutate(p, pop) if random.random() < mut_prob else copy.deepcopy(p) for p in pop]
    mutate.episodeEnd()
    return x


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
#   mutate_ind - reference to the class to mutate an individual - can be used to
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, pop_size, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None,
                           opt_hv=np.product(mu.HYP_REF)):
    evals = 0
    for G in range(max_gen):

        if G == 0:
            fits_objs = list(map_fn(fitness, pop))
            for ind, fit in zip(pop, fits_objs):
                ind.fitness = fit
            evals += len(pop)
            fronts = mu.divide_fronts(pop)
            for i, f in enumerate(fronts):
                mu.assign_ssc_hyper_volume(f)
                for ind in f:
                    ind.front = i

        if log:
            log.add_multi_gen(pop, evals, opt_hv)

        mating_pool = mate_sel(pop, pop_size)
        offspring = mate(mating_pool, operators)
        fits_objs = list(map_fn(fitness, offspring))
        for ind, fit in zip(offspring, fits_objs):
            ind.fitness = fit
        evals += len(offspring)
        pop = nsga2_select(pop + offspring, pop_size)

    return pop

def main():
    DIMENSION = 10  # dimension of the problems
    POP_SIZE = 100  # population size
    MAX_GEN = 50  # maximum number of generations
    REPEATS = 10  # number of runs of algorithm (should be at least 10)

    OUT_DIR = 'multi'  # output directory for logs
    EXP_ID = 'sscHyperVolume::mutGradient'  # the ID of this experiment (used to create log names)


    current_best = 10000
    best_stuff = {
        'CX_PROB': 0.1669818096306694,
        'MUT_PROB': 0.562295986934051,
        'MUT_STEP': 0.5325860595068554,
        'MUT_DECAY': 0.998200916123056
    }

    EXPERS = 1
    for i in range(EXPERS):
        print("RUN [{}]/[{}]: ".format(i+1, EXPERS))
        if i == 0:
            CX_PROB = best_stuff["CX_PROB"]
            MUT_PROB = best_stuff["MUT_PROB"]
            MUT_STEP = best_stuff["MUT_STEP"]
            MUT_DECAY = best_stuff["MUT_DECAY"]
        else:
            CX_PROB = np.random.uniform(low=0.1, high=0.7)   # crossover probability
            MUT_PROB = np.random.uniform(low=0.3, high=0.9)  # mutation probability
            MUT_STEP = np.random.uniform(low=0.1, high=0.9)  # size of the mutation steps
            MUT_DECAY = np.random.uniform(low=0.9, high=1)

        # and create functions with required signatures
        cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
        # we will run the experiment on a number of different functions
        fit_names = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']

        total_avg = 0
        for fit_name in fit_names:
            print("\t{}: ".format(fit_name), end="")
            fit = mf.get_function_by_name(fit_name)
            opt_hv = mf.get_opt_hypervolume(fit_name)

            # todo -- mutation Gradient
            mutate_ind = MutationGradient(step_size=MUT_STEP, decay=MUT_DECAY)
            mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)

            # Crossover ...
            xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)

            # run the algorithm `REPEATS` times and remember the best solutions from
            # last generations

            best_inds = []
            for run in range(REPEATS):
                print("{}/{}, ".format(run + 1, REPEATS), end="")
                # initialize the log structure
                log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name, run,
                                write_immediately=True, print_frequency=5)
                # create population
                pop = create_pop(POP_SIZE, cr_ind)
                # run evolution - notice we use the pool.map as the map_fn
                pop = evolutionary_algorithm(pop, MAX_GEN, POP_SIZE, fit, [xover, mut], tournament_selection_NSGA2, mutate_ind,
                                             map_fn=map, log=log, opt_hv=opt_hv)
                # remember the best individual from last generation, save it to file
                best_inds.append(mu.hypervolume(pop))

                # if we used write_immediately = False, we would need to save the
                # files now
                # log.write_files()
            print("")
            # print an overview of the best individuals from each run
            avg = 0
            for i, bi in enumerate(best_inds):
                # print(f'Run {i}: objective = {opt_hv - bi}')
                avg += opt_hv - bi
            # print("\t\tfit_name: {} avg: {:.3f}".format(fit_name, avg / len(best_inds)))
            total_avg += avg/len(best_inds)

        total_avg = total_avg/len(fit_names)
        print("\tTotal avg: {:.3f}".format(total_avg))
        if total_avg < current_best:
            current_best = total_avg
            print("BEST AVG = {}".format(current_best))

            best_stuff["CX_PROB"] = CX_PROB
            best_stuff["MUT_PROB"] = MUT_PROB
            best_stuff["MUT_STEP"] = MUT_STEP
            best_stuff["MUT_DECAY"] = MUT_DECAY

            # write summary logs for the whole experiment
            for fit_name in fit_names:
                utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)

    print("----------------------------------------")
    print("----------------------------------------")
    print("----------------------------------------")
    for k, v in best_stuff.items():
        print("{} = {}".format(k, v))


if __name__ == '__main__':
    main()

# {'CX_PROB': 0.1669818096306694, 'MUT_PROB': 0.562295986934051, 'MUT_STEP': 0.5325860595068554, 'MUT_DECAY': 0.998200916123056}

# BEST AVG = 6.33000325298815