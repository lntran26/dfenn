"""module for processing simulated test tree sequence into synonymous
and non-synonymous SFS that can be used by dadi-cli for DFE inference
Modification: adding up SNP from simulation chunks"""
import pickle
import allel
import dadi
import stdpopsim
import numpy as np


def hap_to_dadi_fs(hap, sample_size):
    # convert from hap to allel fs
    fs = allel.sfs(allel.HaplotypeArray(hap).count_alleles()[:, 1])[1:]
    # exclude fixed SNP
    if fs.shape[0] == sample_size:
        fs = fs[:-1]

    # patch 0s to end of fs
    if fs.shape[0] < sample_size - 1:
        for _ in range(sample_size - 1 - fs.shape[0]):
            fs = list(fs)
            fs.append(0)        

    # process to make dadi fs
    fs = list(fs)
    fs.insert(0, 0)
    fs.append(0)
    fs = dadi.Spectrum(fs)

    return fs


def ts_to_two_dadi_fs(exon_ts):
    """Processing an exonic ts into a synonymous and a non-synonymous SFS"""

    # get selection coeffs
    selection_coeffs = [
        stdpopsim.ext.selection_coeff_from_mutation(exon_ts, mut) for mut in exon_ts.mutations()
    ]
    
    if len(selection_coeffs) == 0: # make empty spectrum if no mutation
        sfs_neu, sfs_non_neu = dadi.Spectrum([0]*(exon_ts.sample_size + 1)), dadi.Spectrum([0]*(exon_ts.sample_size + 1))

        
    else:
        # get hap positions for neu vs non_neu sites, 
        neu_positions = []
        non_neu_positions = []
        # for i, s in enumerate(selection_coeffs):
        #     neu_positions.append(i) if s == 0 else non_neu_positions.append(i)
        
        # Mapping mutation type IDs to class of mutation (e.g., neutral, non-neutral)
        class_muts = {}
        for dfe in exon_ts.metadata["stdpopsim"]["DFEs"]:
            for mt in dfe["mutation_types"]:
                mid = mt["slim_mutation_type_id"]
                if not mid in class_muts:
                    class_muts[mid] = "neutral" if mt["is_neutral"] else "non_neutral"
        
        for j, s in enumerate(exon_ts.sites()):
            mut_hits = []
            for m in s.mutations:
                for md in m.metadata["mutation_list"]:
                    mut_hits.append(md["mutation_type"])
                    if set(class_muts[md["mutation_type"]]) == set("neutral"):
                        neu_positions.append(m.site)
                    if set(class_muts[md["mutation_type"]]) == set("non_neutral"):
                        non_neu_positions.append(m.site)
    
        # Extract neutral positions haplotypes
        haps = exon_ts.genotype_matrix()
        
        try:
            haps_neu = haps[neu_positions, :]
            haps_non_neu = haps[non_neu_positions, :]
        except:
            print(haps.shape)
            print(non_neu_positions)
            print(len(selection_coeffs))
            print(exon_ts)
    
        # make neutral / non-neutral SFS
        if haps_neu.shape[0] == 0:
            sfs_neu = dadi.Spectrum([0]*(exon_ts.sample_size + 1))
        else:
            sfs_neu = hap_to_dadi_fs(haps_neu, exon_ts.sample_size)
        if haps_non_neu.shape[0] == 0:
            sfs_non_neu = dadi.Spectrum([0]*(exon_ts.sample_size + 1))
        else:
            sfs_non_neu = hap_to_dadi_fs(haps_non_neu, exon_ts.sample_size)

    return sfs_neu, sfs_non_neu


# read in pickle dictionary test data file
# test_data = pickle.load(open('data/test_data_small_chunks_100_1_to_2_of_10.pickle', 'rb'))
test_data = pickle.load(open('data/test_data_small_chunks_100.pickle', 'rb'))

n = 1
true = []

for param in test_data:
    ts = test_data[param]
    neu_fs_small, non_neu_fs_small = ts_to_two_dadi_fs(ts)
    
    if n == 1: # first case, define neu_fs, non_neu_fs
        neu_fs, non_neu_fs = np.zeros(neu_fs_small.shape), np.zeros(non_neu_fs_small.shape)
        
    neu_fs += neu_fs_small
    non_neu_fs += non_neu_fs_small
    
    if n % 100 == 0: # reset fs sum every 100 ts, output added FS
        i = int(n / 100 - 1)
        # output current added FS
        dadi.Spectrum(neu_fs).to_file(f'data/dadi_test/small_chunks_100/fs_{i}_syn')
        dadi.Spectrum(non_neu_fs).to_file(f'data/dadi_test/small_chunks_100/fs_{i}_nonsyn')
        # reset fs sum
        neu_fs, non_neu_fs = np.zeros(neu_fs_small.shape), np.zeros(non_neu_fs_small.shape)
        # save current true param
        true.append(param)
    
    n += 1

# print(true)
# pickle.dump(true, open('data/dadi_test/small_chunks_100/true_small_chunks_100_1_to_2_of_10.pickle','wb'))
pickle.dump(true, open('data/dadi_test/small_chunks_100/true_small_chunks_100.pickle','wb'))
