# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt

if __name__ == "__main__":
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
    algos = [
             # ["const::one_pt", 1000],   # -- Defautl
             # ["lin_decay::one_pt", 1000], # Lin dec
             # ["rule_20::one_pt", 1000],  # Rule 20
             # ["scalable::one_pt", 1000], # -- single set of sigmas....
             # ["scale_ind::one_pt", 1500],  #  scalable
             # ["scale_rotate_ind::one_pt", 2000], # rotatable

            # ------ Better mutation w.r.t. funct == weighted or none ----
            #  ["const::weighted", None],
            #  ["lin_decay::weighted", None],
            #  ["rule_20::weighted", None],
            #  ["scalable::weighted", None],  # -- single set of sigmas....
            #  ["scale_ind::none", 4500],
            #  ["scale_rotate_ind::none", 2500],

             ["scale_ind::none::M,10L", 500],
             ['scale_rotate_ind::none::M,10L', 500],
             # ["scale_ind::one_pt::M,10L", 5000],
             # ["scale_rotate_ind::one_pt::M,10L", 500],

             ]

    fig, axs = plt.subplots(2, 3, figsize=(24, 16))  # figsize=(9, 5)

    for idx, funct in enumerate(fit_names):
        i = idx%3
        j = int(idx/3)
        ax = axs[j,i]
        for i in algos:
            ax.set_yscale('log')

            algo = i[0]
            # xlim = 20000 # i[1]
            xlim=None
            ylim=None
            # if funct == "f06" or funct == "f08" or funct == "f10":
            #     ylim= 100
            name = algo + "." + funct
            utils.plot_experiments(ax, 'continuous', [name], rename_dict=None, xlim=xlim, ylim=ylim, title="Function {}".format(funct))  # prefix, exp ids
    axs[1,2].set_visible(False)
    handles, lbls = axs[0,0].get_legend_handles_labels()
    labels = []
    for l in lbls:
        labels.append(l.split("::")[0])
    fig.legend(handles, labels,   bbox_to_anchor=(0.4, 0.3, 0.4, 0.3), loc='lower right')
    fig.suptitle("Various mutations, no crossover, |offsprings| = 10*|pop_size| ", fontsize=20)
    plt.show()

# {"name": "scale_rotate_ind::none::M,10L", "mutation": RotatableIndividual, "crossover": no_cross,  "num": 10}, # todo - done
# {"name": "scale_ind::none::M,10L", "mutation": MutationScalableIndividual, "crossover": no_cross, "num": 10},
#
# {"name": "scale_rotate_ind::none", "mutation": RotatableIndividual, "crossover": no_cross, "num": 1},
# {"name": "scale_ind::none", "mutation": MutationScalableIndividual, "crossover": no_cross, "num": 1},
#
# {"name": "const::one_pt", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1},
# {"name": "lin_decay::one_pt", "mutation": MutationLinearDecay, "crossover": one_pt_cross, "num": 1},
# {"name": "rule_20::one_pt", "mutation": MutationRule20, "crossover": one_pt_cross, "num": 1},
# {"name": "scalable::one_pt", "mutation": MutationScalable, "crossover": one_pt_cross, "num": 1},
#
# {"name": "const::weighted", "mutation": MutationConstant, "crossover": weighted_cross, "num": 1},
# {"name": "lin_decay::weighted", "mutation": MutationLinearDecay, "crossover": weighted_cross, "num": 1},
# {"name": "rule_20::weighted", "mutation": MutationRule20, "crossover": weighted_cross, "num": 1},
# {"name": "scalable::weighted", "mutation": MutationScalable, "crossover": weighted_cross, "num": 1},
#
# {"name": "scale_ind::one_pt::M,10L", "mutation": MutationScalableIndividual, "crossover": one_pt_cross, "num": 10},
# {"name": "scale_rotate_ind::one_pt::M,10L", "mutation": RotatableIndividual, "crossover": one_pt_cross, "num": 10},
#
# {"name": "scale_ind::one_pt", "mutation": MutationScalableIndividual, "crossover": one_pt_cross,
#  "num": 1},
# {"name": "scale_rotate_ind::one_pt", "mutation": RotatableIndividual, "crossover": one_pt_cross,
#  "num": 1},