import yaml


class Genotype(object):
    def __init__(self, genotyp_key, genotype_file, ann_config, use_old_key=False):

        with open(genotype_file, encoding='ascii', errors='ignore') as f:
            self.genotype_dict = yaml.safe_load(f)
        f.close()

        self.genotype_key = genotyp_key
        if use_old_key:
            self.genotype_key = self.genotype_dict['genotype_key']

        self.ann_config = ann_config
        self.data_path = None
        self.get_data_path()

    def get_data_path(self):
        self.data_path = {}
        for organ in self.ann_config['organ_types']:
            self.data_path[organ] = [0]
            current_cell = 0
            while current_cell != -1:
                next_cell = self.genotype_dict[organ][current_cell].get('oe')
                self.data_path[organ].append(next_cell)
                current_cell = next_cell

    def print_architecture(self, output_file, print_fitness=False):
        print('genotype key', self.genotype_key, file=output_file)
        print('genotype dict:', file=output_file)
        for organ in self.ann_config['organ_types']:
            for cell_key in self.data_path[organ]:
                if self.genotype_dict[organ][cell_key]['type'] != 'input':
                    print(self.genotype_dict[organ][cell_key], file=output_file)

        if print_fitness:
            print('fitness:', self.genotype_dict['fitness'], file=output_file)
            if self.genotype_dict['final_fitness'] > 0.0:
                print('final fitness:', self.genotype_dict['final_fitness'], file=output_file)

    def save_genotype(self, generation):
        genotype_file_name = './genotypes/genotype-' + str(generation) + '-' + str(self.genotype_key) + '.yaml'
        self.genotype_dict['genotype_key'] = self.genotype_key
        with open(genotype_file_name, 'w') as file:
            yaml.safe_dump(self.genotype_dict, file, sort_keys=False)
        file.close()

