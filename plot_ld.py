#!/usr/bin/env python

#SBATCH --job-name=plot_dfe_cnn_test_LD
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=250gb
#SBATCH --time=00:30:00

import pickle
import allel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

### run using dfe-cnn env
# get LD for 1 ts function
def get_LD_from_ts(ts):

    n_bins=19
    haps = ts.genotype_matrix()
    n_SNP, _ = haps.shape
    gaps = (2 ** np.arange(0, np.log2(n_SNP), 1)).astype(int)
    size_chr = ts.sequence_length # 100 MB
    distance_bins = np.logspace(2, np.log10(size_chr), n_bins)
    distance_bins = np.insert(distance_bins, 0, [0])
        
    selected_snps = []
    for gap in gaps:

        snps = np.arange(0, n_SNP, gap) + np.random.randint(0, (n_SNP - 1) % gap + 1)

        # non overlapping contiguous pairs
        # snps=[ 196, 1220, 2244] becomes
        # snp_pairs=[(196, 1220), (1221, 2245)]
        snp_pairs = np.unique([((snps[i] + i) % n_SNP, (snps[i + 1] + i) % n_SNP) for i in range(len(snps) - 1)], axis=0)
        snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]
        last_pair = snp_pairs[-1]
        max_value = n_SNP - gap - 1

        while len(snp_pairs) <= min(300, max_value):
            #count += 1
            #if count % 10 == 0:
                #print(">>  " + str(gap) + " - " + str(len(np.unique(snp_pairs, axis=0))) + " -- "+ str(len(snps) - 1) + "#" + str(count))
            #remainder = (n_SNP - 1) % gap if (n_SNP - 1) % gap != 0 else (n_SNP - 1) // gap
            random_shift =  np.random.randint(1, n_SNP) % n_SNP
            new_pair = (last_pair + random_shift) % n_SNP
            snp_pairs = np.unique(np.concatenate([snp_pairs,
                                                    new_pair.reshape(1, 2) ]), axis=0)
            last_pair = new_pair

            snp_pairs = snp_pairs[snp_pairs[:, 0] < snp_pairs[:, 1]]
        selected_snps.append(snp_pairs)
    
    pos_vec = np.array([variant.site.position for variant in ts.variants()]).astype(int)

    ld = pd.DataFrame()
    for i, snps_pos in enumerate(selected_snps):

        sd = pd.DataFrame((np.diff(pos_vec[snps_pos])), columns=["snp_dist"])

        sd["dist_group"] = pd.cut(sd.snp_dist, bins=distance_bins)
        sr = [allel.rogers_huff_r(snps) ** 2 for snps in haps[snps_pos]]
        sd["r2"] = sr
        sd["gap_id"] = i
        ld = pd.concat([ld, sd])
        
        ld_df = ld.sort_values("dist_group").dropna().groupby('dist_group').mean()
        
    return ld_df['r2']

# multiprocessing for get LD
def get_LD_from_ts_list(ts_list, ncpu=None):
    with Pool(processes=ncpu) as pool:
        ld_list = pool.map(get_LD_from_ts, ts_list)
    return ld_list
    
def plot_LD_from_ts(ld_list, fname, title):
    plt.figure(figsize=(10,6))
    for ld in ld_list:
        y = ld
        x = ld.index.astype(str)
        plt.plot(x, y)
    x_labels = [str(int(ld.index[i].right)) for i in range(len(ld.index))]
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i, label in enumerate(labels):
        if i % 2 == 0:
            labels[i] = x_labels[i]
        else:
            labels[i] = ''
    # plt.xticks([x_labels[i] for i in [0,2,4,6,8,10,12,14,16,18]], visible=True, rotation="horizontal")
    ax.set_xticklabels(labels)
    plt.xlabel("Distance")
    plt.ylabel("r2")
    plt.title(title)
    plt.savefig(fname, bbox_inches='tight')

file_list = ['test_data.pickle','test_data_1B08_varied_scale_2.pickle','test_data_1B08_varied.pickle','test_data_1B08_varied_scale_10.pickle']

for filename in file_list:
    test_d = pickle.load(
        open('data/' + filename, 'rb'))
    all_ld = get_LD_from_ts_list(list(test_d.values()))
    outfile = 'plots/LD/' + filename + '.png'
    plot_title = 'test set: ' + filename
    plot_LD_from_ts(all_ld, outfile, plot_title)
    
    
    