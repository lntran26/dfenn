"""module for processing simulated test tree sequence into synonymous
and non-synonymous SFS that can be used by dadi-cli for DFE inference"""
import os
import pickle
import allel
import dadi
import stdpopsim


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


def ts_to_two_dadi_fs(exon_ts, max_snps: int):
    """Processing an exonic ts into a synonymous and a non-synonymous SFS"""

    # get selection coeffs
    selection_coeffs = [
        stdpopsim.ext.selection_coeff_from_mutation(exon_ts, mut) for mut in exon_ts.mutations()
    ]

    # get hap positions for neu vs non_neu sites, 
    # excluding position outside max_snp
    neu_positions = []
    non_neu_positions = []
    
    # for i, s in enumerate(selection_coeffs[: max_snps]):
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
    haps = exon_ts.genotype_matrix()[: max_snps, :]
    haps_neu = haps[neu_positions, :]
    haps_non_neu = haps[non_neu_positions, :]

    # make neutral / non-neutral SFS

    sfs_neu = hap_to_dadi_fs(haps_neu, exon_ts.sample_size)
    sfs_non_neu = hap_to_dadi_fs(haps_non_neu, exon_ts.sample_size)

    return sfs_neu, sfs_non_neu


# read in pickle dictionary test data file
# test_data = pickle.load(open('data/test_data.pickle', 'rb'))  # 50 set
# test_data = pickle.load(open('data/test_data_1B08.pickle', 'rb')) 
test_data = pickle.load(open('data/test_data_1T12.pickle', 'rb'))

for i, ts in enumerate(test_data.values()):
    for max_snps_count in ['all']:
        max_snps = ts.genotype_matrix(
        ).shape[0] if max_snps_count == 'all' else max_snps_count

        neu_fs, non_neu_fs = ts_to_two_dadi_fs(ts, max_snps)

        # if not os.path.exists(f'data/dadi_test/{max_snps_count}_reprocessed'):
        #     os.makedirs(f'data/dadi_test/{max_snps_count}_reprocessed')
        # if not os.path.exists(f'data/dadi_test/two_epoch/{max_snps_count}'):
        #     os.makedirs(f'data/dadi_test/two_epoch/{max_snps_count}')
        if not os.path.exists(f'data/dadi_test/three_epoch/{max_snps_count}'):
            os.makedirs(f'data/dadi_test/three_epoch/{max_snps_count}')

        # save individual SFS to file, different dir for different SNP size
        # neu_fs.to_file(f'data/dadi_test/{max_snps_count}_reprocessed/fs_{i}_syn')
        # non_neu_fs.to_file(f'data/dadi_test/{max_snps_count}_reprocessed/fs_{i}_nonsyn')
        # neu_fs.to_file(f'data/dadi_test/two_epoch/{max_snps_count}/fs_{i}_syn')
        # non_neu_fs.to_file(f'data/dadi_test/two_epoch/{max_snps_count}/fs_{i}_nonsyn')
        neu_fs.to_file(f'data/dadi_test/three_epoch/{max_snps_count}/fs_{i}_syn')
        non_neu_fs.to_file(f'data/dadi_test/three_epoch/{max_snps_count}/fs_{i}_nonsyn')
        
        
        
        
        