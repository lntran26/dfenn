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
    
    print(model)

    # simulate tree sequence
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        slim_scaling_factor=1,
        slim_burn_in=10,
        dry_run=True)

    return ts
    
def get_dfe_params(n_gammas, n_reps, seed=None):
    # set random seed
    np.random.seed(seed)
    params_list = []
    for _ in range(n_gammas):
        gamma_mean = (np.random.random() + 0.1) / 20
        gamma_shape = (np.random.random() + 0.02) / 2
        T = np.random.random() * 0.99 + 0.01 # not too long past
        nu = np.random.random() * 3 - 1 # decrease bottleneck size
        initial_size = 7778
        time = int(2*initial_size*T)
        present_size = int(initial_size*10**nu)
        for seed in range(n_reps):
            params_list.append((gamma_mean, gamma_shape, seed, time, present_size))
    return params_list

arg_list_train = get_dfe_params(5, 1, seed=1)
arg_list_test = get_dfe_params(5, 1, seed=100)

# parallelize simulations
with mp.Pool() as pool:
    test = pool.map(simulate_ts, arg_list_test)
test_dict = dict(zip(arg_list_test, test))
print('test_dict:')
for param in test_dict:
    print(param)
    
with mp.Pool() as pool:
    train = pool.map(simulate_ts, arg_list_train)
train_dict = dict(zip(arg_list_train, train))
print('train_dict:')
for param in train_dict:
    print(param)
    
