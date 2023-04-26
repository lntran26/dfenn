import pickle
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean() ** 0.5

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
    elif  max(x+y) > 0.4: # shape
        plt.xlim([min(x+y)-0.1, max(x+y)+0.1])
        plt.ylim([min(x+y)-0.1, max(x+y)+0.1])
    else:
        plt.xlim([min(x+y)-0.01, max(x+y)+0.01])
        plt.ylim([min(x+y)-0.01, max(x+y)+0.01])
    plt.tick_params('both', length=size[2]/2, which='major')

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (0.5, 0.5), linewidth=size[1]/2, color='black', zorder=-100)

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
    
# load dadi inference
# bestfits_list = sorted(glob.glob("inference/all_two_epoch/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_two_epoch_2/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_two_epoch_3/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_snm_equil/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_snm_equil_2/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_snm_equil_3/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_two_epoch_small_chunks_100/*DFE.bestfits*"))
# bestfits_list = sorted(glob.glob("inference/all_two_epoch_small_chunks_1000/*DFE.bestfits*"))
bestfits_list = sorted(glob.glob("inference/all_two_epoch_reprocessed/*DFE.bestfits*"))

dadi_pred = []

finished_fs_id = []
converged_fs_id = []

for fname in bestfits_list:
    # get id of fs that finishes
    # finished_fs_id.append(int(fname.split('_')[1].split('.')[0]))
    fs_id = int(fname.split('.')[0].split('_')[-1])
    finished_fs_id.append(fs_id)
    
    file = open(fname, "r")
    for line in file:
        if "Converged" in line:
            converged_fs_id.append(fs_id)
        if line[0] != '#':
            nums = line.split()
            x = [float(num) for num in nums][1:] # remove ll (first)
            dadi_pred.append(x)
            break
# print(f'FS with bestfits file: {finished_fs_id}')

not_finished_fs_id = []
not_converged_fs_id = []

# for i in range(10):
for i in range(50):
    if i not in finished_fs_id:
        not_finished_fs_id.append(i)
    if i not in converged_fs_id and i in finished_fs_id:
        not_converged_fs_id.append(i)

print(f'FS without bestfits file: {not_finished_fs_id}')
print(f'FS without converged results:{not_converged_fs_id}')

pred = np.array(dadi_pred)
# pred = np.array(dadi_pred[:10])
pred_shape = pred[:, 0]
pred_scale = pred[:, 1]
pred_theta = pred[:, 2]
pred_mean = pred_scale * pred_shape / (pred_theta / (0.08 * (2.31/(1 + 2.31))))
# print(pred_shape)
# print(pred_scale)
# print(pred_mean)

# load true
test_data = pickle.load(open('data/test_data.pickle', 'rb')) # scale in log 10
true_shape = []
true_scale = []
true_mean = []

for i, param in enumerate(list(test_data.keys())):
# for i, param in enumerate(list(test_data.keys())[:10]):
    if i not in not_finished_fs_id: # exclude data set that dadi doesn't have results for
        true_shape.append(param[1])
        true_scale.append(12378*abs(param[0])/param[1])
        true_mean.append(param[0])
# print(np.array(true_shape))
# print(np.array(true_scale))
# print(np.array(true_mean))

# for fname in bestfits_list:
#     idx = int(fname.split('_')[1].split('.')[0])

#     file = open(fname, "r")
#     for line in file:
#         if line[0] != '#':
#             nums = line.split()
#             # x = [float(num) for num in nums][1:-1]
#             x = [float(num) for num in nums][1:]
#             # remove ll (first) and theta (last)
#             if idx in [1, 20, 30]:
#                 print(f'Pred p for fs_{idx}: shape={x[0]}, scale(log)={np.log10(x[1])}, theta={x[2]}.')
#             dadi_pred.append(x)
#             break
# pred = np.array(dadi_pred)
# pred_shape = pred[:, 0]
# pred_scale = np.log10(pred[:, 1]) # scale in log 10
# # pred_scale = pred[:, 1]
# pred_theta = pred[:, 2]
# # population-scaled Nes is pred_scale * pred_shape (mean expected)
# # divide this mean/product by Ne (calculated from theta) to get 
# # non-population-scaled mean
# # 0.08 = 4 mu L = 4 * 1e-8 * 2e6, (2.31/(1 + 2.31) is portion of Lns from L
# pred_mean = pred_scale * pred_shape / (pred_theta / (0.08 * (2.31/(1 + 2.31)))))

# # print(pred_mean)

# # load true
# test_data = pickle.load(open('data/test_data.pickle', 'rb'))
# true_shape = []
# true_scale = []
# true_mean = []
# for i, param in enumerate(list(test_data.keys())):
#     if i not in [2,6,13,18,21,40]:
#         true_shape.append(param[1])
#         true_scale.append(np.log10(12378*abs(param[0])/param[1]))
#         true_mean.append(param[0])
#         if i in [1, 20, 30]:
#             print(f'True p for fs_{i}: shape={param[1]}, scale(log)={np.log10(12378*abs(param[0])/param[1])}, mean={param[0]}.')
# # print(np.array(true_mean))


rmse_scale = root_mean_squared_error(pred_scale, true_scale)
rmse_shape = root_mean_squared_error(pred_shape, true_shape)
rmse_mean = root_mean_squared_error(pred_mean, true_mean)
rho_scale = stats.spearmanr(true_scale, pred_scale)[0]
rho_shape = stats.spearmanr(true_shape, pred_shape)[0]
rho_mean = stats.spearmanr(true_mean, pred_mean)[0]

plot_accuracy_single(list(true_scale), list(pred_scale), size=[6, 2, 20], rho=rho_scale,
                     rmse=rmse_scale, log=True,
                     title="scale"
                     )

# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_2/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_3/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil/dadi_scale_new.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_2/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_3/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_1000/dadi_scale.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_scale_10.png", transparent=True, dpi=150)
plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_reprocessed/dadi_scale.png", transparent=True, dpi=150)
plt.clf()

plot_accuracy_single(list(true_mean), list(pred_mean), size=[6, 2, 20], rho=rho_mean,
                     rmse=rmse_mean, 
                     title="mean"
                     )

# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_2/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_3/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil/dadi_mean_new.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_2/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_3/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_1000/dadi_mean.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_mean_10.png", transparent=True, dpi=150)
plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_reprocessed/dadi_mean.png", transparent=True, dpi=150)
plt.clf()

plot_accuracy_single(list(true_shape), list(pred_shape), size=[6, 2, 20], rho=rho_shape,
                     rmse=rmse_shape, title="shape")

# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_2/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_3/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil/dadi_shape_new.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_2/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_snm_equil_3/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_1000/dadi_shape.png", transparent=True, dpi=150)
# plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_small_chunks_100/dadi_shape_10.png", transparent=True, dpi=150)
plt.savefig(f"plots/dadi_dfe_accuracy_two_epoch_reprocessed/dadi_shape.png", transparent=True, dpi=150)
    