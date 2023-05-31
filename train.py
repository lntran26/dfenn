import matplotlib.ticker as mticker
import pickle
from stdpopsim import ext
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import gamma


def ts_to_tensor(data: dict):
    """
    Input: dictionary of params:ts
    Output: dictionary of params:tensor,
    """
    # separate values and keys
    tensor_list = []
    for ts in data.values():
        # get haplotype matrix
        haps = ts.genotype_matrix().T

        selection_coeffs = [
            ext.selection_coeff_from_mutation(ts, mut) for mut in ts.mutations()
        ]
        positions = [variant.site.position for variant in ts.variants()]

        # get position in haps that is neutral
        neu_positions = []
        for i, (_, s) in enumerate(zip(positions, selection_coeffs)):
            if s == 0:
                neu_positions.append(i)

        fixed_positions = []
        dim_1, dim_2 = haps.copy(), haps.copy()
        for idx in range(haps.shape[1]):
            # get position in haps that have fixed (1s) snp
            if np.all(haps[:, idx] == 1):
                fixed_positions.append(idx)
            if idx in neu_positions:
                dim_1[:, idx][np.where(haps[:, idx] == 1)] = 0
            else:
                dim_2[:, idx][np.where(haps[:, idx] == 1)] = 0

        snp_tensor = np.stack((dim_1, dim_2), axis=-1)
        # drop columns that are all 1s (fixed)
        new_tensor = np.delete(snp_tensor, fixed_positions, 1)
        # if have position vector in the future will have to
        # remove the fixed positions accordingly as well

        # tensor_list.append(snp_tensor)
        tensor_list.append(new_tensor)

    return dict(zip(list(data.keys()), tensor_list))


def prep_tensor(data: dict, max_snps: int):
    X_input, y_label = [], []
    # crop or filter
    for param in data:
        tensor = data[param]
        if tensor.shape[1] >= max_snps:
            tensor = tensor[:, :max_snps, :]
            X_input.append(tensor)

            # y_label.append(param[:-1])  # exclude seed

            # # change mean to abs log scale
            # log_mean = np.log10(param[0])
            # y_label.append((log_mean, param[1]))
            # change mean to scale (log scale)
            scale_log = np.log10(12378*param[0]/param[1])
            y_label.append((scale_log, param[1]))
        else:
            pass

    return np.array(X_input), np.array(y_label)


def create_dfe_cnn(input_shape: tuple, n_outputs: int):

    model = Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=5, strides=2,
                            input_shape=input_shape, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Conv1D(filters=16, kernel_size=2,
              strides=2, activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(20, 1)))
    model.add(layers.AveragePooling2D(pool_size=(1, 4)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(n_outputs, activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    return model, {}


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


# load data
# max_snps = 200
max_snps = 300
# subfix = 'scale_2'

# train_data = pickle.load(open(f'data_varied_1B08/train_data_1B08_varied_{subfix}.pickle', 'rb'))
# train_in, train_out = prep_tensor(ts_to_tensor(train_data), max_snps)
# pickle.dump(train_in, open(f'data_varied_1B08/train_in_{subfix}_{max_snps}','wb'))
# pickle.dump(train_out, open(f'data_varied_1B08/train_out_{subfix}_{max_snps}','wb'))

# test_data = pickle.load(open(f'data_varied_1B08/test_data_1B08_varied_{subfix}.pickle', 'rb'))
# test_in, test_out = prep_tensor(ts_to_tensor(test_data), max_snps)
# pickle.dump(test_in, open(f'data_varied_1B08/test_in_{subfix}_{max_snps}','wb'))
# pickle.dump(test_out, open(f'data_varied_1B08/test_out_{subfix}_{max_snps}','wb'))

# test_data = pickle.load(open(f'data_varied_1B08/test_data_1B08_varied_{subfix}.pickle', 'rb'))
# test_in, test_out = prep_tensor(ts_to_tensor(test_data), max_snps)
# pickle.dump(test_in, open(f'data_varied_1B08/test_in_{subfix}_{max_snps}','wb'))
# pickle.dump(test_out, open(f'data_varied_1B08/test_out_{subfix}_{max_snps}','wb'))

# test_data = pickle.load(open(f'data_varied_1B08/test_data_1B08_varied.pickle', 'rb'))
# test_in, test_out = prep_tensor(ts_to_tensor(test_data), max_snps)
# pickle.dump(test_in, open(f'data_varied_1B08/test_in_no_scale_wider_param_range_{max_snps}','wb'))
# pickle.dump(test_out, open(f'data_varied_1B08/test_out_no_scale_wider_param_range_{max_snps}','wb'))

# train_in = pickle.load(open(f'data_two_epoch/train_in_{max_snps}_{subfix}','rb'))
# train_out = pickle.load(open(f'data_two_epoch/train_out_{max_snps}_{subfix}','rb'))

subfix = 'scale_2'
test_in = pickle.load(open(f'data_varied_1B08/test_in_{subfix}_{max_snps}','rb'))
test_out = pickle.load(open(f'data_varied_1B08/test_out_{subfix}_{max_snps}','rb'))
print(f'test_in_len: {len(list(test_in))}')
print(f'test_in_shape: {test_in.shape}')
print(f'test_out_len: {len(list(test_out))}')

# subfix = 'no_scale_wider_param_range'
# test_in = pickle.load(
#     open(f'data_varied_1B08/test_in_{subfix}_{max_snps}', 'rb'))
# test_out = pickle.load(
#     open(f'data_varied_1B08/test_out_{subfix}_{max_snps}', 'rb'))
# print(f'test_in_len: {len(list(test_in))}')
# print(f'test_out_len: {len(list(test_out))}')

# # request a model
# input_shape = train_in.shape[1:]
# n_outputs = 2
# model, kwargs = create_dfe_cnn(input_shape, n_outputs)

# # set training data, epochs and validation data
# kwargs.update(x=train_in, y=train_out, batch_size=10,
#               epochs=30, validation_data=(test_in, test_out))

# # call fit, including any arguments supplied alongside the model
# callback = callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model.fit(**kwargs, callbacks=[callback])
# # model.fit(**kwargs)


# # save trained model
# pickle.dump(model, open(f'data_varied_1B08/trained_model_{subfix}_{max_snps}','wb'))

# subfix = 'scale_2'
# model = pickle.load(
#     open(f"data_varied_1B08/trained_model_{subfix}_{max_snps}", "rb"))
model = pickle.load(open(f"data/trained_model_{max_snps}_scale_shape", "rb"))


# # make sure error is low enough
# rmse = root_mean_squared_error(model.predict(test_in), test_out)
# print("\n{:.1f} RMSE for toy CNN on toy sample".format(rmse))

# plot


def plot_accuracy_single(x, y, size=(8, 2, 20), x_label="Simulated",
                         y_label="Inferred", log=False, r2=None,
                         rho=None, rmse=None, c=None, title=None):
    '''
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    rmse: rmse score for x and y
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    '''
    font = {'size': size[2]}
    plt.rc('font', **font)
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal', 'box')

    # plot data points in a scatter plot
    if c is None:
        plt.scatter(x, y, s=size[0]*2**3, alpha=0.8)  # 's' specifies dots size
    else:  # condition to add color bar
        plt.scatter(x, y, c=c, vmax=5, s=size[0]*2**3, alpha=0.8)
        # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$')

    # axis label texts
    plt.xlabel(x_label, labelpad=size[2]/2)
    plt.ylabel(y_label, labelpad=size[2]/2)

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        # plt.xticks(ticks=[1e-2, 1e0, 1e2])
        # plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    elif max(x+y) > 0.4:  # shape
        plt.xlim([min(x+y)-0.1, max(x+y)+0.1])
        plt.ylim([min(x+y)-0.1, max(x+y)+0.1])
    else:
        plt.xlim([min(x+y)-0.01, max(x+y)+0.01])
        plt.ylim([min(x+y)-0.01, max(x+y)+0.01])
    plt.tick_params('both', length=size[2]/2, which='major')

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (0.5, 0.5),
               linewidth=size[1]/2, color='black', zorder=-100)

    # plot scores if specified
    if r2 is not None:
        plt.text(0.25, 0.82, "\n\n" + r'$R^{2}$: ' + str(round(r2, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)
    if rho is not None:
        plt.text(0.25, 0.82, "ρ: " + str(round(rho, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)
    if rmse is not None:
        plt.text(0.7, 0.08, "rmse: " + str(round(rmse, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va='top')
    plt.tight_layout()


def get_proportion_from_gamma(shape, scale, N, bin_from=0, bin_to=np.inf):
    dist = gamma(shape, scale=scale/(2*N))
    return dist.cdf(bin_to) - dist.cdf(bin_from)


pred = model.predict(test_in)
pred_scale = 10**pred[:, 0]
pred_shape = pred[:, 1]

# pred_mean = [(scale * shape)/12378 for scale, shape in zip(pred_scale, pred_shape)]
pred_all_proportions = []
for proportion in [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3),
                   (1e-3, 1e-2), (1e-2, np.inf)]:
    pred_proportion = [get_proportion_from_gamma(shape, scale, N=7778,
                                                 bin_from=proportion[0],
                                                 bin_to=proportion[1]) for scale, shape in zip(pred_scale, pred_shape)]
    pred_all_proportions.append(pred_proportion)


true_scale = 10**test_out[:, 0]
true_shape = test_out[:, 1]
# true_mean = [(scale * shape)/12378 for scale,
#              shape in zip(true_scale, true_shape)]
true_all_proportions = []
for proportion in [(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3),
                   (1e-3, 1e-2), (1e-2, np.inf)]:
    true_proportion = [get_proportion_from_gamma(shape, scale, N=7778,
                                                 bin_from=proportion[0],
                                                 bin_to=proportion[1]) for scale, shape in zip(true_scale, true_shape)]
    true_all_proportions.append(true_proportion)

# calculate rmse in log for scale
rmse_scale = root_mean_squared_error(pred[:, 0], test_out[:, 0])
rmse_shape = root_mean_squared_error(pred_shape, true_shape)
# rmse_mean = root_mean_squared_error(np.array(pred_mean), np.array(true_mean))
all_rmse_proportions = [root_mean_squared_error(
    np.array(pred_proportion), np.array(true_proportion))
    for pred_proportion, true_proportion in
    zip(pred_all_proportions, true_all_proportions)]
# print(all_rmse_proportions)

rho_scale = stats.spearmanr(true_scale, pred_scale)[0]
rho_shape = stats.spearmanr(true_shape, pred_shape)[0]
# rho_mean = stats.spearmanr(true_mean, pred_mean)[0]
all_rho_proportions = [get_rho(np.array(pred_proportion),
                               np.array(true_proportion))
                       for true_proportion, pred_proportion in
                       zip(true_all_proportions, pred_all_proportions)]
# print(all_rho_proportions)


# plot

plot_accuracy_single(list(true_scale), list(pred_scale), size=[6, 2, 20],
                     log=True,
                     rho=rho_scale,
                     rmse=rmse_scale,
                     title="scale")

# plt.savefig(f"plots/two_epoch_varied_1B08/scale_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/scale_train_{subfix}_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
plt.savefig(f"plots/two_epoch_varied_1B08/scale_train_no_demog_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/scale_train_no_demog_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/scale_train_{subfix}_test_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/scale_train_{subfix}_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(
#     f"plots/two_epoch_varied_1B08/scale_train_{subfix}_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(
#     f"plots/two_epoch_varied_1B08/scale_train_no_demog_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)
plt.clf()

plot_accuracy_single(list(true_shape), list(pred_shape), size=[6, 2, 20], rho=rho_shape,
                     rmse=rmse_shape, title="shape")

# plt.savefig(f"plots/two_epoch_varied_1B08/shape_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/shape_train_{subfix}_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
plt.savefig(f"plots/two_epoch_varied_1B08/shape_train_no_demog_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/shape_train_no_demog_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/shape_train_{subfix}_test_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/shape_train_{subfix}_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(
#     f"plots/two_epoch_varied_1B08/shape_train_{subfix}_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(
#     f"plots/two_epoch_varied_1B08/shape_train_no_demog_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)
plt.clf()

# plot_accuracy_single(true_mean, pred_mean, size=[6, 2, 20], rho=rho_mean,
#                      rmse=rmse_mean,
#                      title="mean"
#                      )

# plt.savefig(f"plots/two_epoch_varied_1B08/mean_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_{subfix}_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_no_demog_test_scale_2_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_no_demog_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_{subfix}_test_{subfix}_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_{subfix}_test_scale_10_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(f"plots/two_epoch_varied_1B08/mean_train_{subfix}_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)
# plt.savefig(
#     f"plots/two_epoch_varied_1B08/mean_train_no_demog_test_no_scale_wider_param_range_{max_snps}.png", transparent=True, dpi=150)

f = mticker.ScalarFormatter(useMathText=True)

for i, proportion in enumerate([(0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3),
                                (1e-3, 1e-2), (1e-2, np.inf)]):
    plt.clf()
    p_from = "${}$".format(f.format_data(proportion[0])) if proportion[0] not in (
        0, np.inf) else proportion[0]
    p_to = "${}$".format(f.format_data(proportion[1])) if proportion[1] not in (
        0, np.inf) else proportion[1]
    plot_accuracy_single(true_all_proportions[i],
                         pred_all_proportions[i],
                         size=[6, 2, 20], rho=all_rho_proportions[i],
                         rmse=all_rmse_proportions[i],
                         title=f"{p_from}≤|s|≤{p_to}"
                         )
    # plt.savefig(
    #     f"plots/two_epoch_varied_1B08/proportions/mean_train_no_demog_test_no_scale_wider_param_range_{max_snps}_{proportion[0]:.0e}_to_{proportion[1]:.0e}.png", transparent=True, dpi=150)
    # plt.savefig(
    #     f"plots/two_epoch_varied_1B08/proportions/mean_train_{subfix}_test_no_scale_wider_param_range_{max_snps}_{proportion[0]:.0e}_to_{proportion[1]:.0e}.png", transparent=True, dpi=150)
    # plt.savefig(
    #     f"plots/two_epoch_varied_1B08/proportions/mean_train_{subfix}_test_{subfix}_{max_snps}_{proportion[0]:.0e}_to_{proportion[1]:.0e}.png", transparent=True, dpi=150)
    plt.savefig(
        f"plots/two_epoch_varied_1B08/proportions/mean_train_no_demog_test_{subfix}_{max_snps}_{proportion[0]:.0e}_to_{proportion[1]:.0e}.png", transparent=True, dpi=150)