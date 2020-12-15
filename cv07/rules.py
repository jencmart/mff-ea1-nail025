import copy
import csv
import functools
import random

from collections import namedtuple, defaultdict

import numpy as np

import utils

# a rule is a list of conditions (one for each attribute) and the predicted class
Rule = namedtuple('Rule', ['conditions', 'cls', 'weight'])

# the following 3 classes implement simple conditions, the call method is used 
# to match the condition against a value
class LessThen:

    def __init__(self, threshold, lb, ub):
        self.params = np.array([threshold])
        self.lb = lb
        self.ub = ub

    def boundary(self):
        return (self.ub - self.lb)*self.params[0] + self.lb

    def __call__(self, value):
        return value <= self.boundary()
    
    def __str__(self):
        return " <= " + str(self.boundary())


class Between:
    def __init__(self, threshold, lb, ub):
        self.params = np.array(threshold)
        self.lb = lb
        self.ub = ub

    # params == 1 --> ub
    # params == 0 --> lb
    def boundary(self):
        b1 = (self.ub - self.lb) * self.params[0] + self.lb
        b2 = (self.ub - self.lb) * self.params[1] + self.lb
        if b1 < b2:
            return b1, b2
        else:
            return b2, b1

    def __call__(self, value):
        b = self.boundary()
        return b[0] <= value <= b[1]

    def __str__(self):
        b = self.boundary()
        return str(b[0]) + " <= " + str(b[1])


class GreaterThen:

    def __init__(self, threshold, lb, ub):
        self.params = np.array([threshold])
        self.lb = lb  # ve slopci x_3 je min hodnota 11
        self.ub = ub  # ve sloupci x_3 je max hodnota 70

    # params == 1 --> ub
    # params == 0 --> lb
    def boundary(self):
        return (self.ub - self.lb)*self.params[0] + self.lb

    def __call__(self, value):
        return value >= self.boundary()

    def __str__(self):
        return " >= " + str(self.boundary())


class Any:

    def __init__(self, lb, ub):
        self.params = np.array([])
        self.lb = lb
        self.ub = ub

    def __call__(self, value):
        return True

    def __str__(self):
        return " * "


def create_single_condition(low, high):
    choices = [LessThen(random.random(), low, high),
               GreaterThen(random.random(), low, high),
               Between([random.random(), random.random()], low, high),
               Any(low, high)
               ]
    c = np.random.choice(choices, p=[1 / 4, 1 / 4, 1 / 4, 1 / 4])
    return c


# generate a single random rule - defines the probabilities of different
# conditions in the initial population
def create_rule(num_attrs, num_classes, lb, ub):
    conditions = []
    for i in range(num_attrs):
        c = create_single_condition(lb[i], ub[i])
        conditions.append(c)

    return Rule(conditions=conditions, cls=random.randrange(0, num_classes), weight=np.random.uniform())


# creates the individual - list of rules
def create_ind(max_rules, num_attrs, num_classes, lb, ub):
    ind_len = random.randrange(1, max_rules)
    return [create_rule(num_attrs, num_classes, lb, ub) for i in range(ind_len)]


# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


# uses an individual to predict a single instance - the rules in the individual
# vote for the final class
def classify_instance(ind, attrs):
    votes = defaultdict(int)
    for rule in ind:
        if all([cond(a) for cond, a in zip(rule.conditions, attrs)]):
            votes[rule.cls] += 1*rule.weight  # todo -- now we are weighting ...
    
    best_class = None
    best_votes = -1
    for k, v in votes.items():
        if v > best_votes:
            best_votes = v
            best_class = k

    if best_class == None:
        best_class = 0

    return best_class


# computes the accuracy of the individual on a given dataset
def accuracy(ind, data):
    data_x, data_y = data

    correct = 0
    for attrs, target in zip(data_x, data_y):
        if classify_instance(ind, attrs) == target:
            correct += 1
    
    return correct/len(data_y)


# computes the fitness (accuracy on training data) and objective (error rate
# on testing data)
def fitness(ind, train_data, test_data):
    return utils.FitObjPair(fitness=accuracy(ind, train_data), 
                            objective=1-accuracy(ind, test_data))


# the tournament selection
def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(copy.deepcopy(pop[p1]))
        else:
            selected.append(copy.deepcopy(pop[p2]))

    return selected


# implements a uniform crossover for individuals with different lenghts
def cross(p1, p2):
    o1, o2 = [], []
    for r1, r2 in zip(p1, p2):
        if random.random() < 0.5:
            o1.append(copy.deepcopy(r1))
            o2.append(copy.deepcopy(r2))
        else:
            o1.append(copy.deepcopy(r2))
            o2.append(copy.deepcopy(r1))
   
    # individuals can have different lenghts
    l = min(len(p1), len(p2))
    rest = o1[l:] + o2[l:]
    for r in rest:
        if random.random() < 0.5:
            o1.append(copy.deepcopy(r))
        else:
            o2.append(copy.deepcopy(r))

    return o1, o2


def weight_mutate(p, mut_weight_prob_chage, mut_weight_sigma):
    p = copy.deepcopy(p)
    o = []
    for r in p:
        if random.random() < mut_weight_prob_chage:
            new_w = r.weight + mut_weight_sigma*np.random.randn()
        else:
            new_w = r.weight
        o.append(Rule(conditions=r.conditions, cls=r.cls, weight=new_w))
    return o


# class mutation - changes the predicted class for a given rule
def y_mutate(p, num_classes, mut_cls_prob_change):
    p = copy.deepcopy(p)
    o = []
    for r in p:
        o_cls = r.cls
        if random.random() < mut_cls_prob_change:
            o_cls = random.randrange(0, num_classes)   
        o.append(Rule(conditions=r.conditions, cls=o_cls, weight=r.weight))
    return o


# mutation changing the threshold in conditions in an individual
def cond_mutate(p, mut_cond_sigma):
    o = copy.deepcopy(p)
    for r in o:
        for c in r.conditions:
            c.params += mut_cond_sigma*np.random.randn(*c.params.shape)
    return o


def whole_cond_mutate(p, mut_whole_cond_prob_change):
    p = copy.deepcopy(p)
    o = []

    # for 1 rule ...
    for r in p:
        new_conditions = []
        for old_c in r.conditions:
            if random.random() < mut_whole_cond_prob_change:
                # Create new condition
                new_c = create_single_condition(old_c.lb, old_c.ub)
                # Reuse old threshold
                if new_c.params.shape[0] > 0 and old_c.params.shape[0] > 0:
                    new_c.params[0] = old_c.params[0]
                new_conditions.append(new_c)
            else:
                new_conditions.append(old_c)

        o.append(Rule(conditions=new_conditions, cls=r.cls, weight=r.weight))
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
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off


# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)
def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]


# reads data in a csv file
def read_data(filename):
    data_x = []
    data_y = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            attrs = line[:-1]
            target = line[-1]
            data_x.append(list(map(float, attrs)))
            data_y.append(int(target))

    return (np.array(data_x), np.array(data_y))

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
def evolutionary_algorithm(pop, pop_size, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, pop_size)
        offspring = mate(mating_pool, operators)
        pop = offspring[1:] + [pop[max(enumerate(fits), key=lambda x: x[1])[0]]]

    return pop


def main(INPUT_FILE):

    # todo - not change
    MAX_GEN = 50  # maximum number of generations
    REPEATS = 10  # number of runs of algorithm (should be at least 10)
    OUT_DIR = INPUT_FILE[:-4] + "_results"  #'rules'  # output directory for logs

    # todo - maybe change ??
    MAX_RULES = 15  # maximum number of rules in an individual
    POP_SIZE = 100  # population size

    # todo - change per experiment
    # INPUT_FILE = 'winequality-white.csv'
    EXP_ID = 'between::weight::mutWeight::mutWholeCond'  # the ID of this experiment (used to create log names)

    current_best = 100
    best_stuff = {'CX_PROB': 0.42804296292896027,
                 'MUT_CLS_PROB': 0.21661746207575233,
                 'MUT_CLS_PROB_CHANGE': 0.19439091432281305,
                 'MUT_WHOLE_COND_PROB': 0.13656056536261027,
                 'MUT_WHOLE_COND_PROB_CHANGE': 0.06541206317573847,
                 'MUT_WEIGHT_PROB': 0.342560669294847,
                 'MUT_WEIGHT_PROB_CHANGE': 0.0800282027592091,
                 'MUT_WEIGHT_SIGMA': 0.2397772686910034,
                 'MUT_COND_PROB': 0.15994937585132796,
                 'MUT_COND_SIGMA': 0.28829087857065017
                 }

    EXPERS  = 1
    for i in range(EXPERS):

        CX_PROB = np.random.uniform(low=0.3, high=0.8)  # crossover probability
        MUT_CLS_PROB = np.random.uniform(low=0.05, high=0.5)  # probability of class changing mutationm
        MUT_CLS_PROB_CHANGE = np.random.uniform(low=0.01, high=0.3)  # probability of changing target class in mutation
        MUT_WHOLE_COND_PROB = np.random.uniform(low=0.05, high=0.4)
        MUT_WHOLE_COND_PROB_CHANGE = np.random.uniform(low=0.05, high=0.4)
        MUT_WEIGHT_PROB = np.random.uniform(low=0.05, high=0.4)
        MUT_WEIGHT_PROB_CHANGE = np.random.uniform(low=0.05, high=0.4)
        MUT_WEIGHT_SIGMA = np.random.uniform(low=0.05, high=0.4)
        MUT_COND_PROB = np.random.uniform(low=0.05, high=0.4)  # probabilty of condition changing mutation
        MUT_COND_SIGMA = np.random.uniform(low=0.05, high=0.6)  # step size of condition changing mutation
        print("RUN [{}]/[{}]: ".format(i+1, EXPERS), end="")

        # read the data
        data = read_data('inputs/' + INPUT_FILE)

        num_attrs = len(data[0][0])
        num_classes = max(data[1]) + 1

        # make training and testing split
        perm = np.arange(len(data[1]))
        np.random.shuffle(perm)
        n_train = 2 * len(data[1]) // 3

        train_x, test_x = data[0][perm[:n_train]], data[0][perm[n_train:]]
        train_y, test_y = data[1][perm[:n_train]], data[1][perm[n_train:]]

        # count the lower and upper bounds
        lb = np.min(train_x, axis=0)  # min pro kazdy sloupecek ... pro kazde x_i
        ub = np.max(train_x, axis=0)  # max pro kazdy sloupecek .. pro kazde x_i

        train_data = (train_x, train_y)
        test_data = (test_x, test_y)

        # use `functool.partial` to create fix some arguments of the functions
        # and create functions with required signatures
        cr_ind = functools.partial(create_ind, max_rules=MAX_RULES,
                                   num_attrs=num_attrs, num_classes=num_classes,
                                   lb=lb, ub=ub)
        xover = functools.partial(crossover, cross=cross, cx_prob=CX_PROB)

        # Y mutate
        cls_mutate = functools.partial(y_mutate, num_classes=num_classes, mut_cls_prob_change=MUT_CLS_PROB_CHANGE)
        mut_cls = functools.partial(mutation, mutate=cls_mutate, mut_prob=MUT_CLS_PROB)

        # rule threshold mutate
        c_mutate = functools.partial(cond_mutate, mut_cond_sigma=MUT_COND_SIGMA)
        mut_cond = functools.partial(mutation, mutate=c_mutate, mut_prob=MUT_COND_PROB)

        # Whole condition mutate .. todo - new
        wc_mutate = functools.partial(whole_cond_mutate, mut_whole_cond_prob_change=MUT_WHOLE_COND_PROB_CHANGE)
        mut_whole_cond = functools.partial(mutation, mutate=wc_mutate, mut_prob=MUT_WHOLE_COND_PROB)

        # Weight mutate .. todo - new
        w_mut = functools.partial(weight_mutate, mut_weight_prob_chage=MUT_WEIGHT_PROB_CHANGE, mut_weight_sigma=MUT_WEIGHT_SIGMA)
        mut_weight = functools.partial(mutation, mutate=w_mut, mut_prob=MUT_WEIGHT_PROB)

        fit = functools.partial(fitness, train_data=train_data, test_data=test_data)

        # run the algorithm `REPEATS` times and remember the best solutions from
        # last generations

        import multiprocessing

        pool = multiprocessing.Pool(8)
        map_fn = pool.map

        best_inds = []
        for run in range(REPEATS):
            print("{}/{}, ".format(run+1, REPEATS), end="")
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID, run, write_immediately=True, print_frequency=1)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, POP_SIZE, MAX_GEN, fit, [xover, mut_cls, mut_cond, mut_whole_cond, mut_weight],
                                         tournament_selection, map_fn=map_fn, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)

            # if we used write_immediately = False, we would need to save the
            # files now
            # log.write_files()
        print("")

        # print an overview of the best individuals from each run
        sum = 0
        for i, bi in enumerate(best_inds):
            sum += fit(bi).objective

        avg = sum / len(best_inds)
        if avg < current_best:
            current_best = avg

            # print it
            for i, bi in enumerate(best_inds):
                print(f'Run {i}: objective = {fit(bi).objective}')
            print("BEST AVG = {}".format(current_best))

            # write summary logs for the whole experiment
            utils.summarize_experiment(OUT_DIR, EXP_ID)

            best_stuff['CX_PROB'] = CX_PROB
            best_stuff['MUT_CLS_PROB'] = MUT_CLS_PROB
            best_stuff['MUT_CLS_PROB_CHANGE'] = MUT_CLS_PROB_CHANGE
            best_stuff['MUT_WHOLE_COND_PROB'] = MUT_WHOLE_COND_PROB
            best_stuff['MUT_WHOLE_COND_PROB_CHANGE'] = MUT_WHOLE_COND_PROB_CHANGE
            best_stuff['MUT_WEIGHT_PROB'] = MUT_WEIGHT_PROB
            best_stuff['MUT_WEIGHT_PROB_CHANGE'] = MUT_WEIGHT_PROB_CHANGE
            best_stuff['MUT_WEIGHT_SIGMA'] = MUT_WEIGHT_SIGMA
            best_stuff['MUT_COND_PROB'] = MUT_COND_PROB
            best_stuff['MUT_COND_SIGMA'] = MUT_COND_SIGMA

    for k, v in best_stuff.items():
        print("{} = {}".format(k, v))


if __name__ == '__main__':
    main('iris.csv')
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    main('winequality-white.csv')

# BEST AVG = 0.028
# best_vals = {'CX_PROB': 0.42804296292896027, 'MUT_CLS_PROB': 0.21661746207575233, 'MUT_CLS_PROB_CHANGE': 0.19439091432281305, 'MUT_WHOLE_COND_PROB': 0.13656056536261027, 'MUT_WHOLE_COND_PROB_CHANGE': 0.06541206317573847, 'MUT_WEIGHT_PROB': 0.342560669294847, 'MUT_WEIGHT_PROB_CHANGE': 0.0800282027592091, 'MUT_WEIGHT_SIGMA': 0.2397772686910034, 'MUT_COND_PROB': 0.15994937585132796, 'MUT_COND_SIGMA': 0.28829087857065017}
