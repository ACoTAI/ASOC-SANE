from random import choice, choices, sample
from math import ceil, floor
from copy import deepcopy


class EvolutionOperation(object):
    def __init__(self, evolution_config):
        self.evolution_config = evolution_config

    def add_cell(self, genotype_list):
        chosen_genotypes = sample(genotype_list,
                                  k=int(ceil(len(genotype_list) * (self.evolution_config['add_cell_prob'] / 100.0))))

        for genotype in chosen_genotypes:
            chosen_organ_type = choices(genotype.ann_config['organ_types'],
                                        weights=self.evolution_config['organ_prob'], k=1)[0]

            in_cell_key = choice([cell_key for cell_key in genotype.data_path[chosen_organ_type]
                                 if cell_key != -1])

            chosen_cell_type = choice(list(genotype.ann_config[chosen_organ_type+'_cell_types'].keys()))

            new_cell_key = max(genotype.genotype_dict[chosen_organ_type].keys()) + 1

            candidate_cell_list = [cell for cell in genotype.genotype_dict[chosen_organ_type].values()
                                   if cell['type'] == chosen_cell_type]

            ie_index = genotype.data_path[chosen_organ_type].index(in_cell_key)
            out_cell_key = genotype.data_path[chosen_organ_type][ie_index+1]
            if len(candidate_cell_list) > 0:
                new_cell = deepcopy(choice(candidate_cell_list))
                new_cell['oe'] = out_cell_key
            else:
                new_cell = {'type': chosen_cell_type,
                            'oe': out_cell_key,
                            'cm': [chosen_cell_type,
                                   genotype.ann_config[chosen_organ_type + '_cell_types'][chosen_cell_type][0]],
                            'am': genotype.ann_config[chosen_organ_type + '_cell_types'][chosen_cell_type][1],
                            }
            genotype.genotype_dict[chosen_organ_type][new_cell_key] = new_cell

            genotype.genotype_dict[chosen_organ_type][in_cell_key]['oe'] = new_cell_key
            genotype.data_path[chosen_organ_type].insert(ie_index+1, new_cell_key)

            print(f'genotype {genotype.genotype_key}')
            print(f'cell key {new_cell_key}, position {(in_cell_key, new_cell_key, out_cell_key)}, '
                  f'organ {chosen_organ_type}, cell type {chosen_cell_type}')
            print()

            genotype.genotype_dict['fitness'] = 0.0

    def modify_cell(self, genotype_list):
        chosen_genotypes = sample(genotype_list,
                                  k=int(ceil(len(genotype_list) * (self.evolution_config['modify_cell_prob'] / 100.0))))

        for genotype in chosen_genotypes:
            cell_info_list = []
            for organ in genotype.ann_config['organ_types']:
                cell_info_list += [[cell_key, cell, organ] for cell_key, cell in genotype.genotype_dict[organ].items()
                                   if cell['type'] not in ['input', 'output']]

            if len(cell_info_list) > 0:
                chosen_cell_info = choice(cell_info_list)
                print(f'genotype {genotype.genotype_key}, organ {chosen_cell_info[2]}, cell key {chosen_cell_info[0]}')

                chosen_cell = chosen_cell_info[1]
                if chosen_cell['type'] == 'conv':
                    attr_list = [i for i in range(len(self.evolution_config['conv_attr_prob']))]
                    chosen_attr = choices(attr_list, weights=self.evolution_config['conv_attr_prob'], k=1)[0]
                    if chosen_attr < 4:
                        print(chosen_cell['cm'], '->')
                        chosen_cell['cm'][1][chosen_attr] += \
                            self.evolution_config['conv_attr_growth_factor'][chosen_attr]
                        print(chosen_cell['cm'])
                    else:
                        print(chosen_cell['am'], '->')
                        if 'maxpool' in chosen_cell['am']:
                            chosen_cell['am'].remove('maxpool')
                        else:
                            chosen_cell['am'].append('maxpool')
                        print(chosen_cell['am'])

                elif chosen_cell['type'] == 'linear':
                    print(chosen_cell['cm'], '->')
                    chosen_cell['cm'][1] += self.evolution_config['linear_attr_growth_factor']
                    print(chosen_cell['cm'])

                elif chosen_cell['type'] == 'convtrans':
                    attr_list = [i for i in range(len(self.evolution_config['convtrans_attr_prob']))]
                    chosen_attr = choices(attr_list, weights=self.evolution_config['convtrans_attr_prob'], k=1)[0]

                    print(chosen_cell['cm'], '->')
                    chosen_cell['cm'][1][chosen_attr] += \
                        self.evolution_config['convtrans_attr_growth_factor'][chosen_attr]
                    print(chosen_cell['cm'])

                elif chosen_cell['type'] == 'convlstm':
                    print(chosen_cell['cm'], '->')
                    chosen_cell['cm'][1] += self.evolution_config['convlstm_attr_growth_factor']
                    print(chosen_cell['cm'])

                elif chosen_cell['type'] == 'res':
                    attr_list = [i for i in range(len(self.evolution_config['res_attr_prob']))]
                    chosen_attr = choices(attr_list, weights=self.evolution_config['res_attr_prob'], k=1)[0]

                    print(chosen_cell['cm'], '->')
                    chosen_cell['cm'][1][chosen_attr] += \
                        self.evolution_config['res_attr_growth_factor'][chosen_attr]
                    print(chosen_cell['cm'])

                print()

                genotype.genotype_dict['fitness'] = 0.0

    def crossover(self, genotype_list):
        chosen_genotypes = sample(genotype_list,
                                  k=int(ceil(len(genotype_list) * (self.evolution_config['crossover_prob']/100.0))))

        crossover_list_length = floor(len(chosen_genotypes) / 2)

        crossover_tuples = [[chosen_genotypes[i], chosen_genotypes[i+crossover_list_length]]
                            for i in range(crossover_list_length)]

        for parent in crossover_tuples:
            chosen_organ = choice(parent[0].ann_config['organ_types'])
            organ0 = deepcopy(parent[0].genotype_dict[chosen_organ])
            organ1 = deepcopy(parent[1].genotype_dict[chosen_organ])

            parent[0].genotype_dict[chosen_organ] = organ1
            parent[1].genotype_dict[chosen_organ] = organ0

            parent[0].get_data_path()
            parent[1].get_data_path()

            print(f'crossover parents: {parent[0].genotype_key, parent[1].genotype_key}')

            parent[0].genotype_dict['fitness'] = 0.0
            parent[1].genotype_dict['fitness'] = 0.0

        print()
