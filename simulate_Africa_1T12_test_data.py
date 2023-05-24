#!/usr/bin/env python

#SBATCH --job-name=dfe_sim_Africa_1T12_2e6_contig
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
#SBATCH --time=48:00:00

import stdpopsim
import tskit
import pickle
import multiprocess as mp
import numpy as np

def simulate_ts(args):
    gamma_mean, gamma_shape, seed = args
    # species, demography, contig, sample settings
    species = stdpopsim.get_species("HomSap")
    # generic demography
    # model = stdpopsim.PiecewiseConstantSize(species.population_size)
    # specify demography
    model = species.get_demographic_model("Africa_1T12")
    # select specific chromosome and region
    # contig = species.get_contig("chr20", left=10e6, right=11e6)
    contig = species.get_contig(length=2e6)
    # sampling 10 diploid genomes
    # samples = {"pop_0": 10}
    samples = {"AFR": 10}
    # choose engine
    engine = stdpopsim.get_engine("slim")
    # DFE setting
    dfe = species.get_dfe("Gamma_K17")
    # change dfe distribution for negative mutation types
    dfe.mutation_types[1].distribution_args = [-gamma_mean, gamma_shape]
    # # apply DFE to exons only
    # exons = species.get_annotations("ensembl_havana_104_exons")
    # exon_intervals = exons.get_chromosome_annotations("chr20")
    # contig.add_dfe(intervals=exon_intervals, DFE=dfe)
    contig.add_dfe(intervals=np.array([[0, int(contig.length)]]), DFE=dfe)

    # simulate tree sequence
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        slim_scaling_factor=1,
        slim_burn_in=10)
    # save output tree sequence
    # ts.dump(f"simulations/{str(args)}.trees")

    return ts
    

def get_dfe_params(n_gammas, n_reps, seed=None):
    # set random seed
    np.random.seed(seed)
    
    params_list = []
    for _ in range(n_gammas):
        gamma_mean = (np.random.random() + 0.1) / 20
        gamma_shape = (np.random.random() + 0.02) / 2
        for seed in range(n_reps):
            params_list.append((gamma_mean, gamma_shape, seed))
        
    return params_list

arg_list_train = get_dfe_params(100, 5, seed=1)
arg_list_test = get_dfe_params(50, 1, seed=100)
# arg_list_train = get_dfe_params(10, 5, seed=1)
# arg_list_test = get_dfe_params(5, 1, seed=100)


# parallelize simulations
with mp.Pool() as pool:
    test = pool.map(simulate_ts, arg_list_test)
test_dict = dict(zip(arg_list_test, test))
# pickle.dump(test_dict, open('data/test_data.pickle', 'wb'))
pickle.dump(test_dict, open('data/test_data_1T12.pickle', 'wb'))
    
with mp.Pool() as pool:
    train = pool.map(simulate_ts, arg_list_train)
train_dict = dict(zip(arg_list_train, train))
pickle.dump(train_dict, open('data/train_data_1T12.pickle', 'wb'))
