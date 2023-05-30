#!/usr/bin/env python

#SBATCH --job-name=dfe_sim_varied_demog_2e6_contig_1B08_test_small
#SBATCH --output=outfiles/%x-%j.out
#SBATCH --error=outfiles/%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lnt@arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem=50gb
#SBATCH --time=10:00:00

import stdpopsim
import tskit
import pickle
import multiprocess as mp
import numpy as np

def simulate_ts(args):
    # load dfe and demography args
    gamma_mean, gamma_shape, seed, time, present_size = args
    # species, demography, contig, sample settings
    species = stdpopsim.get_species("HomSap")
    # specify demography
    model = species.get_demographic_model("Africa_1B08")
    # alter demographic parameter value
    model.model.events[0].time = time
    model.model.populations[0].initial_size = present_size
    # specify contig length
    contig = species.get_contig(length=2e6)
    # sampling 10 diploid genomes
    samples = {"African_Americans": 10}
    # choose engine
    engine = stdpopsim.get_engine("slim")
    # DFE setting
    dfe = species.get_dfe("Gamma_K17")
    # change dfe distribution for negative mutation types
    dfe.mutation_types[1].distribution_args = [-gamma_mean, gamma_shape]
    # apply DFE to the whole contig
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
        T = np.random.random() * 1.4 + 0.1
        nu = np.random.random() * 2
        initial_size = 7778
        time = 2*initial_size*T
        present_size = initial_size*10**nu
        for seed in range(n_reps):
            params_list.append((gamma_mean, gamma_shape, seed, time, present_size))
    return params_list

arg_list_train = get_dfe_params(5, 1, seed=1)
arg_list_test = get_dfe_params(5, 1, seed=100)

# parallelize simulations
with mp.Pool() as pool:
    test = pool.map(simulate_ts, arg_list_test)
test_dict = dict(zip(arg_list_test, test))
pickle.dump(test_dict, open('data/test_data_1B08_varied_test_small.pickle', 'wb'))
    
with mp.Pool() as pool:
    train = pool.map(simulate_ts, arg_list_train)
train_dict = dict(zip(arg_list_train, train))
pickle.dump(train_dict, open('data/train_data_1B08_varied_test_small.pickle', 'wb'))
