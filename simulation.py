import stdpopsim
import tskit
import pickle
import multiprocess as mp

def simulate_ts(args):

    gamma_mean, gamma_shape, seed = args

    # species, demography, contig, sample settings
    species = stdpopsim.get_species("HomSap")
    # generic demography
    model = stdpopsim.PiecewiseConstantSize(species.population_size)
    # select specific chromosome and region
    contig = species.get_contig("chr20", left=10e6, right=20e6)
    # contig = species.get_contig(length=5e6)
    # sampling 5 diploid genomes
    samples = {"pop_0": 5}

    # choose engine
    engine = stdpopsim.get_engine("slim")

    # DFE setting
    dfe = species.get_dfe("Gamma_K17")

    # change dfe distribution for negative mutation types
    dfe.mutation_types[1].distribution_args = [gamma_mean, gamma_shape]

    # apply DFE to exons only
    exons = species.get_annotations("ensembl_havana_104_exons")
    exon_intervals = exons.get_chromosome_annotations("chr20")
    contig.add_dfe(intervals=exon_intervals, DFE=dfe)

    # simulate tree sequence
    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        slim_scaling_factor=1,
        slim_burn_in=10,
    )
    # save output tree sequence
    ts.dump(f"simulations/{str(args)}.trees")

    return ts

# TO-DO: function to generate gamma values
arg_list = []
for seed in [1, 2, 3]:
    for gamma in [(-0.0131483, 0.186), (-0.05, 0.186), (-0.005, 0.186),
                  (-0.0131483, 0.01), (-0.0131483, 0.5)]:
        param = gamma[0], gamma[1], seed
        arg_list.append(param)

# parallelize simulations
with mp.Pool() as pool:
    ts_list = pool.map(simulate_ts, arg_list)
pickle.dump(ts_list, open('simulations/ts_list.pickle', 'wb'))
