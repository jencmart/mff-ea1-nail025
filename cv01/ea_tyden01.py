import random
import pprint
import numpy as np
import matplotlib.pyplot as plt


def fitness_onemax(ind):
    return sum(ind)


def fitness_cnt_correct_pos(ind, start_with=1):
    return sum([1 if (idx % 2 == 0 and val == start_with) or ((idx-1) % 2 == 1-start_with and val == 0) else 0 for idx, val in enumerate(ind)])


def fitness_cnt_changes(ind, start_with=1):
    return sum([1 if (idx == 0 and val == start_with) or (val != ind[idx-1]) else 0 for idx, val in enumerate(ind)])


class SGA:
    def __init__(self, dimension, population_size, max_generations, mut_prob):
        self.dimension = dimension
        self.population_size = population_size
        self.max_generations = max_generations
        self.mut_prob = mut_prob

    def create_ind(self):
        return [random.randint(0, 1) for _ in range(self.dimension)]

    def create_random_population(self):
        return [self.create_ind() for _ in range(self.population_size)]

    def select(self, pop, fits):  # p_i = f_i/sum(f_j) # ruletova selekce
        return random.choices(pop, fits, k=self.population_size)

    def mutation(self, pop):
        def mutate(ind, mut_prob):
            o = []
            for bit in ind:
                if random.random() < mut_prob:
                    o.append(1 - bit)
                else:
                    o.append(bit)
            return o
        return [mutate(ind, self.mut_prob) for ind in pop]

    def crossover(self, pop):
        def cross(p1, p2, dimension):
            # 01000|10101 -> 01000|00110
            # 10101|00110 -> 10101|10101
            point = random.randint(0, dimension - 1)
            o1 = p1[:point] + p2[point:]
            o2 = p2[:point] + p1[point:]
            return o1, o2
        offspring = []
        for p1, p2 in zip(pop[::2], pop[1::2]):
            o1, o2 = cross(p1, p2, self.dimension)
            offspring.append(o1)
            offspring.append(o2)
        return offspring

    def mate(self, pop):
        return

    def evolutionary_algorithm(self, fitness_f, leave_best=True):
        pop = self.create_random_population()
        log = []
        for G in range(self.max_generations):
            # 1. spocitam fitness
            # fits = list(map(fitness, pop)) # namapuje fci fitness na kazdy prvek z pop
            calc_fitness = [fitness_f(i) for i in pop]

            # pro kazdou generaci pridej
            log.append((max(calc_fitness),  # best_fitness
                        sum(calc_fitness)/self.population_size,   # avg fitness
                        (G+1)*self.population_size))  # COUNTS

            # 2. selection
            mating_pool = self.select(pop, calc_fitness)
            # 3. krizeni -> 4. mutace
            offspring = self.mutation(self.crossover(pop))
            # 5. nova populace z offsprings
            # lst[:] pomoci indexovani muzu delat shallow copy listu - trik
            # prvni potomek pryc, zachovej nejlepsiho jedince z populace
            if leave_best:
                pop = offspring[1:] + [max(pop, key=fitness_f)]
            else:
                pop = offspring[:]

        return pop, log


# Zkuste změnit některé parametry algoritmu (např. pravděpodobnost mutace nebo křížení) a podívejte se, co se stane.

# Napište mi, co vše jste zkusili.
#    10 vet ....
#    muzete poslat i zdrojak nejake specialni funkce

# pust algoritmus 15 x a udelej prumer z behu ....
# prvi a treti quartil


def average_sga(sga, fitness):
    logs = []
    for _ in range(RUNS_FOR_AVG):
        # 100 x (best_fitness, avg_fitness)
        pop, log = sga.evolutionary_algorithm(fitness)
        logs.append(log)

    # plot
    max_from_each_gen = np.array([[l[0] for l in log] for log in logs])  # RUNS_FOR_AVG x CNT_GENERATIONS
    mu = np.mean(max_from_each_gen, axis=0)
    # x_axis = np.array(([[l[2] for l in log] for log in logs])[0])  # 50, 100, 150, 200, 250,
    x_axis = np.array(range(CNT_GENERATIONS))
    return x_axis, mu, max_from_each_gen


def plot_it(ax, res, c='b'):
    tmp, = ax.plot(res[0], res[1], c)
    ax.fill_between(res[0],
                     np.quantile(res[2], axis=0, q=0.25,),
                     np.quantile(res[2], axis=0, q=0.75,),
                     alpha=0.2, color=c)
    return tmp


if __name__ == "__main__":
    DIMENSION = 25
    POP_SIZE = 50
    CNT_GENERATIONS = 100
    RUNS_FOR_AVG = 10

    # PLOT
    rows = 3
    cols = 2

    res_correct_pos_all = []
    res_cnt_changes_all = []
    mut_prob = [0.01, 0.05, 0.1,
                0.2, 0.3, 0.4]

    for i in mut_prob:
        MUT_PROB = i
        sga = SGA(DIMENSION, POP_SIZE, CNT_GENERATIONS, MUT_PROB)
        res_correct_pos_all.append(average_sga(sga, fitness_cnt_correct_pos))
        res_cnt_changes_all.append(average_sga(sga, fitness_cnt_changes))

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(5,10))

    for i in range(rows):
        for j in range(cols):
            axs = ax[i, j]
            red = plot_it(axs, res_correct_pos_all[i*cols + j], c='r')
            blue = plot_it(axs, res_cnt_changes_all[i*cols +j], c='b')
            axs.set(xlabel="Generations", ylabel='AVG Fitness')
            axs.set_title('MutProb:{}'.format(mut_prob[i*cols + j]))

            axs.legend([red, blue], ['F: # Correct Pos', 'F: # Changes'])
    fig.suptitle("Dim:{}, PopSize:{}, CntGen:{}".format(DIMENSION, POP_SIZE, CNT_GENERATIONS), fontsize=14)
    plt.show()
