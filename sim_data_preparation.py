import pandas as pd
import random

# Constants
EDGES_DATA_PATH = '.../data/edges_list'
SOURCE_DATA_PATH = '.../data/source_list.csv'
RANDOM_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONCENTRATION_RANGE = list(range(10, 100))

# Read data
edges_df = pd.read_csv(EDGES_DATA_PATH, index_col=0, sep=',')
source_df = pd.read_csv(SOURCE_DATA_PATH, index_col=0, sep=',')

# Extract source genes
source_genes = list(source_df.index)

# Extract link information
links_from = list(edges_df['0'])
links_to = list(edges_df['1'])
grouped_targets = edges_df.groupby('0')
targets_from, targets_to = [], []

# Group links
for group_key, group_data in grouped_targets:
    targets_from.append(group_key)
    targets_to.append(list(group_data['1']))

# All genes in the network
all_genes = list(set(edges_df['0']) | set(edges_df['1']))

# Generate random rates for genes
def generate_random_rates(genes):
    rates = []
    for gene in genes:
        rates.append(random.choice(RANDOM_RATES))
    return [(f'rate{i}', rate) for i, rate in enumerate(rates)]

# Generate random concentrations for genes
def generate_random_concentrations(genes, source_genes):
    non_source_genes = [x for x in genes if x not in source_genes]
    num_non_source_genes = len(non_source_genes)
    zero_concentration = [0.0] * num_non_source_genes
    random_num_sources = random.randint(1, min(5, len(source_genes)))
    random_sources = random.sample(source_genes, random_num_sources)
    source_concentration = random.choices(CONCENTRATION_RANGE, k=random_num_sources)
    leftover_sources = [x for x in source_genes if x not in random_sources]
    zero_leftover = [0.0] * len(leftover_sources)
    gene_list = non_source_genes + random_sources + leftover_sources
    concentration_list = zero_concentration + source_concentration + zero_leftover
    return list(zip(gene_list, concentration_list))

# Generate edges
def generate_edges(from_genes, to_genes):
    edges = []
    for i in range(len(from_genes)):
        try:
            link_name = f'{from_genes[i]}link'
            relation_in = {globals()[from_genes[i]]: 1}
            relation_out = {globals()[j]: 1 for j in to_genes[i]}
            temp_rate = globals()[f'rate{i}']
            edges.append([link_name, relation_in, relation_out, temp_rate])
        except IndexError:
            continue
    return edges
