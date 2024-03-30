import pandas as pd
import random
from gillespy2.core import Model, Species, Reaction, Parameter
from gillespy2 import ODESolver
from data_preparation import *

def run_simulations(max_runs=300):
    for i in range(1, max_runs + 1):
        run_sim(i)
        print(i)
    print('Simulation runs completed.')

def run_sim(i):
    rates_file = get_rates_to_list(all_genes)
    print('Rates added.')
    species = get_concentrations_to_list(all_genes, source)
    species_df = pd.DataFrame(species).drop_duplicates(subset=[0], keep="last")
    species = species_df.values.tolist()
    print('Species added.')

    class MichaelisMenten(Model):
        def __init__(self, parameter_values=None):
            Model.__init__(self, name="Michaelis_Menten")
            parameters = [Parameter(name=rate[0], expression=rate[1]) for rate in rates_file]
            self.add_parameter(parameters)
            species_list = [Species(name=spec[0], initial_value=spec[1], mode='continuous', allow_negative_populations=True) for spec in species]
            self.add_species(species_list)
            reactions = get_edges_to_list(link1, link2)
            reaction_list = [Reaction(name=r[0], reactants=r[1], products=r[2], rate=r[3]) for r in reactions]
            self.add_reaction(reaction_list)
            self.timespan(np.linspace(0, 100, 101))

    model = MichaelisMenten()
    results = model.run(solver=ODESolver)
    print('Simulation completed.')

    results_dict = dict(results[0])
    final_results_df = pd.DataFrame.from_dict(results_dict)
    final_results_df = final_results_df.loc[:, (final_results_df != 0).any(axis=0)].drop(columns=['time'])
    print('Results saved.')
    final_results_df.to_csv('.../results/SIM/results{}.csv'.format(i))
