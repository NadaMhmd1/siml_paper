import pandas as pd
import random

links_df = pd.read_csv('.../data/edges_list', index_col=0, sep=',')
source_df = pd.read_csv('.../data/source_list.csv', index_col=0, sep=',')
source = list(source_df.index)

random_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
random_concentration = list(range(10, 100))

link_new_g1 = list(links_df['0'])
link_new_g2 = list(links_df['1'])
group_targets_done = links_df.groupby('0')
link1, link2 = [], []

for group_key, group_data in group_targets_done:
    link1.append(group_key)
    link2.append(list(group_data['1']))

all_genes = list(set(links_df['0']) | set(links_df['1']))

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
