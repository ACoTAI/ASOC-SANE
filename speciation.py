from itertools import count


class GenotypeDistance(object):
    def __init__(self, distance_coefficient):
        self.distance_coefficient = distance_coefficient
        self.distances = {}

    def get_distance(self, this_genotype, another_genotype):
        distance = self.distances.get((this_genotype.genotype_key, another_genotype.genotype_key))
        if distance is None:
            distance = self.compute_distance(this_genotype, another_genotype)
            self.distances[this_genotype.genotype_key, another_genotype.genotype_key] = distance
            self.distances[another_genotype.genotype_key, this_genotype.genotype_key] = distance
        return distance

    def compute_distance(self, this_genotype, another_genotype):
        distance = 0
        for organ in this_genotype.ann_config['organ_types']:
            this_cell_keys = this_genotype.data_path[organ]
            another_cell_keys = another_genotype.data_path[organ]

            for cell_key in this_cell_keys:
                if cell_key in another_cell_keys:
                    if this_genotype.genotype_dict[organ][cell_key]['type'] != \
                            another_genotype.genotype_dict[organ][cell_key]['type']:
                        distance += 1
                else:
                    distance += 1

            for cell_key in another_cell_keys:
                if cell_key not in this_cell_keys:
                    distance += 1

        return distance * self.distance_coefficient


class Species(object):
    def __init__(self, species_key):
        self.species_key = species_key
        self.representation = None
        self.members = {}
        self.fitness = 0.0

    def update(self, fitness, representation):
        self.fitness = fitness
        self.representation = representation

    def output_species_info(self, output_file):
        print(f'species key - {self.species_key}: ', file=output_file)
        print(f'species fitness: {self.fitness}', file=output_file)
        print(f'species members: {list(self.members.keys())}', file=output_file)
        print('species representative member info: ', file=output_file)
        self.representation.print_architecture(output_file=output_file)
        print('', file=output_file)


class SpeciesSet(object):
    def __init__(self, evolution_config):
        self.evolution_config = evolution_config
        self.species_dict = {}
        self.distances = GenotypeDistance(self.evolution_config['distance_coefficient'])

    def speciate(self, genotypes):
        self.species_dict = {}
        species_counter = count(1)
        # speciate remain genotypes
        for genotype in genotypes.values():
            min_distance = self.evolution_config['compatibility_threshold']
            target_species_key = None

            # compute distance between genotype with each representation, find candidates
            for species_key, species in self.species_dict.items():
                distance = self.distances.get_distance(genotype, species.representation)

                if distance < min_distance:
                    min_distance = distance
                    target_species_key = species_key

            if target_species_key is not None:
                self.species_dict[target_species_key].members[genotype.genotype_key] = genotype
            else:
                # one genotype have no match candidates, new species and be its representation
                species_key = next(species_counter)
                self.species_dict[species_key] = Species(species_key)
                self.species_dict[species_key].representation = genotype
                self.species_dict[species_key].members[genotype.genotype_key] = genotype

    def output_species_dict_info(self, output_file):
        print('|---------- species info ----------|', file=output_file)
        for species in self.species_dict.values():
            species.output_species_info(output_file)
        print('', file=output_file)

