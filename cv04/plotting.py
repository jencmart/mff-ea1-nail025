# use and edit this file to make all the plots you need - it is generally easier
# than plotting directly after the run of the algorithm

import utils

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # {"name": "const::one_pt",
    # {"name": "differential::0.8::0.9",
    # {"name": "differential::0.8::0.9::decay", .... .997},
    # {"name": "differential::0.8::0.9::from4"
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
    algos = [
               ["const::one_pt", None], # -- default
               ["differential::0.8::0.9", None],
               ["differential::0.8::0.9::decay", None],
               ["differential::0.8::0.9::from4", None],
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
            # xlim= 50000
            name = algo + "." + funct
            utils.plot_experiments(ax, 'differential', [name], rename_dict=None, xlim=xlim, ylim=ylim, title="Function {}".format(funct))  # prefix, exp ids
    axs[1,2].set_visible(False)
    handles, lbls = axs[0,0].get_legend_handles_labels()
    labels = []
    for l in lbls:
        labels.append(l[:-4])
    fig.legend(handles, labels,   bbox_to_anchor=(0.4, 0.3, 0.4, 0.3), loc='lower right')
    fig.suptitle("Differential Evolution", fontsize=20)
    plt.show()

