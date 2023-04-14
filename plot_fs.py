"""module for processing ts to syn and non syn SFS and plot"""
import pickle
from matplotlib import pyplot as plt
import numpy as np
import stdpopsim
import allel

ts_list = pickle.load(open('simulations/archived_15_10MB_sim/ts_list.pickle','rb'))
# ts_list = pickle.load(open('data/15_small_set/ts_list.pickle', 'rb'))
# test_data = pickle.load(open('data/test_data.pickle', 'rb'))


def ts_adjust_intervals(input_ts):
    """Processing for ts simulation that include non-exonic element
    and start from the middle of chromosome which require adjustment
    of exonic interval positions"""

    # load exon_intervals
    species = stdpopsim.get_species("HomSap")
    exons = species.get_annotations("ensembl_havana_104_exons")
    exon_intervals = exons.get_chromosome_annotations("chr20")

    # adjust the exon_intervals to the simulated contig
    shifted_exon_intervals = exon_intervals - 10e6
    left = shifted_exon_intervals[shifted_exon_intervals <= 10e6]
    right = left[left > 0]
    adjusted_intervals = np.reshape(right, (-1, 2))

    # make exonic tree sequence
    adjusted_exon_ts = input_ts.keep_intervals(adjusted_intervals)

    return adjusted_exon_ts


def ts_to_two_fs(exon_ts):
    """Processing an exonic ts into a synonymous and a non-synonymous SFS"""

    # get selection coeffs
    selection_coeffs = [
        stdpopsim.ext.selection_coeff_from_mutation(exon_ts, mut) for mut in exon_ts.mutations()
    ]

    # get hap positions for neu vs non_neu sites
    neu_positions = []
    non_neu_positions = []
    for i, s in enumerate(selection_coeffs):
        neu_positions.append(i) if s == 0 else non_neu_positions.append(i)

    # Extract neutral positions haplotypes
    haps = exon_ts.genotype_matrix()
    haps_neu = haps[neu_positions, :]
    haps_non_neu = haps[non_neu_positions, :]

    # make neutral / non-neutral SFS
    sfs_neu = allel.sfs(allel.HaplotypeArray(
        haps_neu).count_alleles()[:, 1])[1:]
    # exclude fixed SNP
    if sfs_neu.shape[0] == exon_ts.sample_size:
        sfs_neu = sfs_neu[:-1]

    sfs_non_neu = allel.sfs(allel.HaplotypeArray(
        haps_non_neu).count_alleles()[:, 1])[1:]
    # exclude fixed SNP
    if sfs_non_neu.shape[0] == exon_ts.sample_size:
        sfs_non_neu = sfs_non_neu[:-1]

    return sfs_neu, sfs_non_neu


def plot_fs(fs_syn, fs_non, plot_title, plot_dir, file_name):

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots()
    x_syn = range(1, len(fs_syn)+1)
    ax.plot(x_syn, fs_syn, label=f"Syn ({int(fs_syn.sum())} SNPs)",
            alpha=0.6, zorder=3, marker='o', color="blue")
    x_non = range(1, len(fs_non)+1)
    ax.plot(x_non, fs_non, label=f"Non-syn ({int(fs_non.sum())} SNPs)",
            alpha=0.6, marker='s', color="red", zorder=2.5)
    ax.legend(loc='upper right', fontsize=18)
    ax = plt.xticks(ticks=[1, 3, 5, 7, 9])
    plt.tick_params('both', length=7, which='major')
    plt.xlabel("Frequency", fontsize=15, labelpad=10)
    plt.ylabel("Count", fontsize=15, labelpad=10)
    plt.title(plot_title, fontsize=15)
    fig.tight_layout()
    plt.savefig(f"{plot_dir}/{file_name}", transparent=True, dpi=150)


for i, ts in enumerate(ts_list):
    dfe_params = ts.metadata['stdpopsim']['DFEs'][1]['mutation_types'][1]['distribution_args']
    exon_ts = ts_adjust_intervals(ts)
    fs_syn, fs_non = ts_to_two_fs(exon_ts)
    title = str(dfe_params)
    directory = "plots/test_fs"
    fname = f"fs_{i+1:02d}.png"
    plot_fs(fs_syn, fs_non, title, directory, fname)
