#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:02:09 2024

@author: nado
"""

# folder chapter in code 

import os
import sys
import numpy as np
import pandas as pd
import random
from random import sample
import ast
import time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

from gillespy2.core import Model, Species, Reaction, Parameter
from gillespy2 import  ODESolver

# Load data
links_df = pd.read_csv('.../data/edges_list', index_col=0, sep=',')
source_df = pd.read_csv('.../data/source_list.csv', index_col=0, sep=',')
source = list(source_df.index)

# Define random parameters
random_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
random_concentration = list(range(10, 100))

# Extract link data
link_new_g1 = list(links_df['0'])
link_new_g2 = list(links_df['1'])
group_targets_done = links_df.groupby('0')
link1, link2 = [], []

# Group all targets for each source
for group_key, group_data in group_targets_done:
    link1.append(group_key)
    link2.append(list(group_data['1']))

# Get all genes
all_genes = list(set(links_df['0']) | set(links_df['1']))

# Functions to generate data
def get_rates_to_list(genes):
    rates = []
    for gene in genes:
        rates.append(random.choice(random_rate))
    return [('rate{}'.format(i), rate) for i, rate in enumerate(rates)]

def get_concentrations_to_list(genes, source_genes):
    non_source_genes = [x for x in genes if x not in source_genes]
    num_non_source_genes = len(non_source_genes)
    zero_concentration = [0.0] * num_non_source_genes
    random_num_sources = random.randint(1, min(5, len(source_genes)))
    random_sources = random.sample(source_genes, random_num_sources)
    source_concentration = random.choices(random_concentration, k=random_num_sources)
    leftover_sources = [x for x in source_genes if x not in random_sources]
    zero_leftover = [0.0] * len(leftover_sources)
    gene_list = non_source_genes + random_sources + leftover_sources
    concentration_list = zero_concentration + source_concentration + zero_leftover
    return list(zip(gene_list, concentration_list))

def get_edges_to_list(g1, g2):
    edges = []
    for i in range(len(g1)):
        try:
            link_name = g1[i] + 'link'
            relation_in = {globals()[g1[i]]: 1}
            relation_out = {globals()[j]: 1 for j in g2[i]}
            temp_rate = globals()['rate{}'.format(i)]
            edges.append([link_name, relation_in, relation_out, temp_rate])
        except IndexError:
            continue
    return edges

# Main simulation function
def run_simulations(max_runs=300):
    for i in range(1, max_runs + 1):
        run_sim(i)
        print(i)
    print('Simulation runs completed.')

# Simulation runner
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

run_simulations()