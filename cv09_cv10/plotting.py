# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils2

import matplotlib.pyplot as plt

if __name__ == "__main__":

    fit_names = ['ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6']

    algos = [ ['default', None],
               ["nn::er::2opt", None],
    ]
    folder = 'tsp'

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
            name = algo + "." + funct

            utils2.plot_experiments(ax, folder, [name], rename_dict=None, xlim=xlim, ylim=ylim, title="Function {}".format(funct))  # prefix, exp ids
    axs[1,2].set_visible(False)
    handles, lbls = axs[0,0].get_legend_handles_labels()
    labels = []
    for l in lbls:
        labels.append(l[:-4])
    fig.legend(handles, labels,   bbox_to_anchor=(0.4, 0.3, 0.4, 0.3), loc='lower right')
    fig.suptitle("Default x HyperVolume+DiffMut", fontsize=20)
    plt.savefig("result.png")

