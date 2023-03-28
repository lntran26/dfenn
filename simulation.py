import os, io, subprocess
import dadi
from matplotlib import pyplot as plt
import multiprocess as mp

def runSLiM(args):
    mean, shape, seed = args
    command = f"slim -d MEAN={mean} -d SHAPE={shape} -d SEED={seed} base.slim"

    # Use subprocess to capture output without intermediate file on disk
    p = subprocess.check_output(command, shell=True, text=True)
    with open(f"simulations/{str(args)}.txt", "w") as text_file:
        print(p, file=text_file)

    # process SLiM output to plot ASF
    try:
        pop_outputs = p.split('#OUT')[1:]
        # Read parts of string output as if they were a file
        fids = [io.StringIO(_) for _ in pop_outputs]

        dd_syn,ns = dadi.Misc.dd_from_SLiM_files(fids, mut_types=['m1'])
        # Return the beginning of the "files" to process the nonsynonymous
        # mutations
        [_.seek(0) for _ in fids]
        dd_non,ns = dadi.Misc.dd_from_SLiM_files(fids, mut_types=['m2'])

        popids = range(len(ns))
        fs_syn = dadi.Spectrum.from_data_dict(dd_syn, popids, ns)
        fs_non = dadi.Spectrum.from_data_dict(dd_non, popids, ns)

        return fs_syn, fs_non
    except:
        # If we failed in parsing, print the SLiM output, for debugging.
        print('Failed to process SLiM output.')
        print(p[:int(1e3)])


arg_list = []
for seed in [123,231,312]:
    for gamma in [(-0.0131483, 0.186), (-0.05, 0.01), (-0.005, 0.5)]:
        param = gamma[0], gamma[1], seed
        arg_list.append(param)

with mp.Pool() as pool:
    fs_list = pool.map(runSLiM, arg_list)

def plot_fs(fs_syn, fs_non, plot_title, plot_dir, file_name):
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()
    x = range(len(fs_syn))
    ax.plot(x, fs_syn, label=f"Syn ({int(fs_syn.sum())} SNPs)", alpha=0.6, zorder=3, marker='o', color="blue")
    ax.plot(x, fs_non, label=f"Non-syn ({int(fs_non.sum())} SNPs)", alpha=0.6, marker='s',color="red", zorder=2.5)
    ax.legend(loc = 'upper right', fontsize=18)
    ax=plt.xticks(ticks=[1,3,5,7,9])
    plt.tick_params('both', length=7, which='major')
    plt.xlabel("Frequency", fontsize=15, labelpad=10)
    plt.ylabel("Count", fontsize=15, labelpad=10)
    plt.title(plot_title, fontsize=15)
    fig.tight_layout()
    plt.savefig(f"{plot_dir}/{file_name}", transparent=True, dpi=150)

for i, (fs_pair, arg) in enumerate(zip(fs_list, arg_list)):
    fs_syn = fs_pair[0]
    fs_non = fs_pair[1]
    title = str(arg)
    directory = "plots"
    fname = f"fs_{i+1}.png"
    plot_fs(fs_syn, fs_non, title, directory, fname)
