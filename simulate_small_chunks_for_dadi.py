#!/usr/bin/env python

#SBATCH --job-name=dfe_sim_exon_small_chunks
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
#SBATCH --time=72:00:00

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
    model = stdpopsim.PiecewiseConstantSize(species.population_size)
    # select a contig length
    contig = species.get_contig(length=2e3)
    # sampling 10 diploid genomes
    samples = {"pop_0": 10}
    # choose engine
    engine = stdpopsim.get_engine("slim")
    # DFE setting
    dfe = species.get_dfe("Gamma_K17")
    # change dfe distribution for negative mutation types
    dfe.mutation_types[1].distribution_args = [-gamma_mean, gamma_shape]
    contig.add_dfe(intervals=np.array([[0, int(contig.length)]]), DFE=dfe)

    # simulate tree sequence
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        slim_scaling_factor=1,
        slim_burn_in=10)

    return ts
    
test_data = pickle.load(open('data/test_data.pickle', 'rb'))

arg_list_test = []
for param in list(test_data.keys()):
    gamma_mean = param[0]
    gamma_shape = param[1]
    for seed in range(1000):
       arg_list_test.append((gamma_mean, gamma_shape, seed))

# parallelize simulations
with mp.Pool() as pool:
    test = pool.map(simulate_ts, arg_list_test)
test_dict = dict(zip(arg_list_test, test))
pickle.dump(test_dict, open('data/test_data_small_chunks.pickle', 'wb'))
    