# call trained models

# test called models using simulated validation data

# and also plot the results here
import os
import numpy as np
from scipy import stats
from scipy.stats import gamma
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker


def root_mean_squared_error(pred_pre: np.ndarray, true_pre: np.ndarray):
    # exclude nans
    pred_post = pred_pre[~np.isnan(pred_pre)]
    true_post = true_pre[~np.isnan(pred_pre)]
    return ((pred_post - true_post) ** 2).mean() ** 0.5


def get_rho(pred_pre: np.ndarray, true_pre: np.ndarray):
    # exclude nans
    pred_post = pred_pre[~np.isnan(pred_pre)]
    true_post = true_pre[~np.isnan(pred_pre)]
    return stats.spearmanr(true_post, pred_post)[0]


def get_proportion_from_gamma(shape, scale, N, bin_from=0, bin_to=np.inf):
    dist = gamma(shape, scale=scale / (2 * N))
    return dist.cdf(bin_to) - dist.cdf(bin_from)


def plot_accuracy_single(
    x,
    y,
    size=(8, 2, 20),
    x_label="Simulated",
    y_label="Inferred",
    log=False,
    r2=None,
    rho=None,
    rmse=None,
    c=None,
    title=None,
):
    """
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    rmse: rmse score for x and y
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    """
    plt.clf()
    font = {"size": size[2]}
    plt.rc("font", **font)
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect("equal", "box")

    # plot data points in a scatter plot
    if c is None:
        plt.scatter(x, y, s=size[0] * 2**3, alpha=0.8)  # 's' specifies dots size
    else:  # condition to add color bar
        # plt.scatter(x, y, c=c, vmax=5, s=size[0]*2**3, alpha=0.8)
        plt.scatter(x, y, c=c, s=size[0] * 2**3, alpha=0.8)
        # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        # cbar.ax.set_title(r'$\frac{T}{ν}$')
        # cbar.ax.set_title('true\nmean', fontsize=15)
        cbar.ax.set_title("true\nscale", fontsize=15)

    # axis label texts
    plt.xlabel(x_label, labelpad=size[2] / 2)
    plt.ylabel(y_label, labelpad=size[2] / 2)

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        plt.ylim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        # plt.xticks(ticks=[1e-2, 1e0, 1e2])
        # plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    elif max(x + y) > 0.4:  # shape
        plt.xlim([min(x + y) - 0.1, max(x + y) + 0.1])
        plt.ylim([min(x + y) - 0.1, max(x + y) + 0.1])
    else:
        plt.xlim([min(x + y) - 0.01, max(x + y) + 0.01])
        plt.ylim([min(x + y) - 0.01, max(x + y) + 0.01])
    plt.tick_params("both", length=size[2] / 2, which="major")

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (0.5, 0.5), linewidth=size[1] / 2, color="black", zorder=-100)

    # plot scores if specified
    if r2 is not None:
        plt.text(
            0.25,
            0.82,
            "\n\n" + r"$R^{2}$: " + str(round(r2, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    if rho is not None:
        plt.text(
            0.25,
            0.82,
            "ρ: " + str(round(rho, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    if rmse is not None:
        plt.text(
            0.7,
            0.08,
            "rmse: " + str(round(rmse, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=size[2],
            transform=ax.transAxes,
        )
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va="top")
    plt.tight_layout()


def plot_all_gamma_results(model, test_in, test_out, outdir: str):
    
    # make outdir if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    pred = model.predict(np.array(test_in))
    pred_scale = 10 ** pred[:, 0]
    # pred_shape = 10**pred[:, 1]
    pred_shape = 10 ** (pred[:, 1] * -1) # this is for if use abs value
    # pred_shape = pred[:, 1]

    # pred_mean = [(scale * shape)/12378 for scale, shape in zip(pred_scale, pred_shape)]
    pred_all_proportions = []
    for proportion in [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, np.inf)]:
        pred_proportion = [
            get_proportion_from_gamma(
                shape, scale, N=7778, bin_from=proportion[0], bin_to=proportion[1]
            )
            for scale, shape in zip(pred_scale, pred_shape)
        ]
        pred_all_proportions.append(pred_proportion)


    true_scale = 10 ** test_out[:, 0]
    # true_shape = 10**test_out[:, 1]
    true_shape = 10 ** (test_out[:, 1] * -1)
    # true_shape = test_out[:, 1]
    true_mean = [(scale * shape) / 12378 for scale, shape in zip(true_scale, true_shape)]
    true_all_proportions = []
    for proportion in [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, np.inf)]:
        true_proportion = [
            get_proportion_from_gamma(
                shape, scale, N=7778, bin_from=proportion[0], bin_to=proportion[1]
            )
            for scale, shape in zip(true_scale, true_shape)
        ]
        true_all_proportions.append(true_proportion)

    # calculate rmse in log for scale
    rmse_scale = root_mean_squared_error(pred[:, 0], test_out[:, 0])
    rmse_shape = root_mean_squared_error(pred_shape, true_shape)
    # rmse_mean = root_mean_squared_error(np.array(pred_mean), np.array(true_mean))
    all_rmse_proportions = [
        root_mean_squared_error(np.array(pred_proportion), np.array(true_proportion))
        for pred_proportion, true_proportion in zip(
            pred_all_proportions, true_all_proportions
        )
    ]
    # print(all_rmse_proportions)

    rho_scale = stats.spearmanr(true_scale, pred_scale)[0]
    rho_shape = stats.spearmanr(true_shape, pred_shape)[0]
    # rho_mean = stats.spearmanr(true_mean, pred_mean)[0]
    all_rho_proportions = [
        get_rho(np.array(pred_proportion), np.array(true_proportion))
        for true_proportion, pred_proportion in zip(
            true_all_proportions, pred_all_proportions
        )
    ]

    # plot shape
    plot_accuracy_single(
        list(true_shape),
        list(pred_shape),
        log=True,
        #  c=np.log10(true_scale),
        size=[6, 2, 20],
        rho=rho_shape,
        rmse=rmse_shape,
        title="shape",
    )
    plt.savefig(f"{outdir}/shape.png", transparent=True, dpi=150)

    # plot scale
    plot_accuracy_single(
        list(true_scale),
        list(pred_scale),
        size=[6, 2, 20],
        log=True,
        #  c=np.log10(true_scale),
        rho=rho_scale,
        rmse=rmse_scale,
        title="scale",
    )
    plt.savefig(f"{outdir}/scale.png", transparent=True, dpi=150)

    # plot proportions based on gamma params
    f = mticker.ScalarFormatter(useMathText=True)

    for i, proportion in enumerate(
        [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, np.inf)]
    ):
        plt.clf()
        fig = plt.figure()
        p_from = (
            "${}$".format(f.format_data(proportion[0]))
            if proportion[0] not in (0, np.inf)
            else proportion[0]
        )
        p_to = (
            "${}$".format(f.format_data(proportion[1]))
            if proportion[1] not in (0, np.inf)
            else proportion[1]
        )
        plot_accuracy_single(
            true_all_proportions[i],
            pred_all_proportions[i],
            # c=true_mean,
            # c=np.log10(true_scale),
            # c=np.log10(abs(true_scale-pred_scale)),
            # c=((np.log10(true_scale)-np.log10(pred_scale))**2)**0.5,
            size=[6, 2, 20],
            rho=all_rho_proportions[i],
            rmse=all_rmse_proportions[i],
            title=f"{p_from}≤|s|≤{p_to}",
        )
        plt.savefig(f"{outdir}/mean_{proportion[0]:.0e}_to_{proportion[1]:.0e}.png", transparent=True, dpi=150)


def plot_LD_results(model, test_in, test_out, outdir: str):
    # make outdir if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    pred = model.predict(np.array(test_in))

    for i, param in enumerate(['a', 'b', 'c']):
    # calculate rmse in log for scale
        rmse = root_mean_squared_error(pred[:, i], test_out[:, i])
        rho = stats.spearmanr(pred[:, i], test_out[:, i])[0]

        plot_accuracy_single(
            list(test_out[:, i]),
            list(pred[:, i]),
            log=False,
            size=[6, 2, 20],
            rho=rho,
            rmse=rmse,
            title=param,
        )
        plt.savefig(f"{outdir}/{param}.png", transparent=True, dpi=150)
