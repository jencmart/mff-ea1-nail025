# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils2

import matplotlib.pyplot as plt

if __name__ == "__main__":

    fit_names = ['']

    algos = [ ['default', None],
               ["between::weight::mutWeight::mutWholeCond", None],
    ]

    name = "winequality-white"
    folder = name + '_results'
    figure_name = name + ".png"

    fig, axs = plt.subplots(1, 1, figsize=(16, 16))  # figsize=(9, 5)
    for idx, funct in enumerate(fit_names):
        i = idx%3
        j = int(idx/3)
        ax = axs
        for i in algos:
            ax.set_yscale('log')

            algo = i[0]
            # xlim = 20000 # i[1]
            xlim=None
            ylim=None
            name = algo  #  + "." + funct
            if name == "default":
                dic = {name : "between::weight::mutWeight::mutWholeCond"}
            else:
                dic = {name: "default"}
            utils2.plot_experiments(ax, folder, [name], rename_dict=None, xlim=xlim, ylim=ylim, title="Function {}".format(funct))  # prefix, exp ids
    # axs[1, 2].set_visible(False)
    handles, lbls = axs.get_legend_handles_labels()
    labels = ["between::weight::mutWeight::mutWholeCond", "default"]
    # for l in lbls:
        # labels.append(l[:-4])
    fig.legend(handles, labels,   bbox_to_anchor=(0.4, 0.3, 0.4, 0.3), loc='lower right')
    fig.suptitle("Default x Improved classification", fontsize=20)
    plt.savefig(figure_name)
