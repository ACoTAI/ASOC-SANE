import os
import time
from math import ceil
from itertools import count
from copy import deepcopy
from random import sample

from genotype import Genotype
from phenotype import construct_ann
from speciation import SpeciesSet
from train_ann import TrainANN
from evolution_operations import EvolutionOperation


class SANEExecuion(object):
    def __init__(self, evolution_config, train_config, ann_config, from_scratch=True,
                 start_generation=1, npi=1, tpg=1, s_t=10, best_genotype_key=None, train_epoch=10):
        self.evolution_config = evolution_config
        self.train_config = train_config
        self.ann_config = ann_config

        self.from_scratch = from_scratch
        self.generation = start_generation
        self.population = {}

        self.evolution = EvolutionOperation(evolution_config)
        self.speciation = SpeciesSet(evolution_config)

        if self.from_scratch:
            for i in range(1, self.evolution_config['genotype_number_init'] + 1):
                new_genotype = Genotype(i, evolution_config['init_genotype'], ann_config)
                self.population[i] = new_genotype
            self.NpI = self.evolution_config['NpI_init']
            self.TpG = self.evolution_config['TpG_init']
            self.species_threshold = self.evolution_config['species_number_limit']
            self.best_genotype = None
        else:
            self.load_population('./genotypes/', self.generation)
            self.NpI = npi
            self.TpG = tpg
            self.species_threshold = s_t
            self.best_genotype = self.population[best_genotype_key]

        self.train_epoch = train_epoch

        self.genotype_indexer = count(max([genotype.genotype_key for genotype in self.population.values()]) + 1)

    def evolve_population(self, data_train, data_valid=None, total_time=0):
        total_time = total_time
        while self.generation <= self.evolution_config['generation_limit']:
            start_time = time.time()
            print(f'|========== Generation {self.generation} ==========|\n')
            if self.from_scratch:
                # [Duplication]
                print('|---------- Generate Offsprings ----------|')
                offsprings = {}
                for i in range(self.NpI):
                    print(f'generate offsprings index: {i + 1}\n')
                    for genotype in self.population.values():
                        new_genotype = deepcopy(genotype)
                        new_genotype.genotype_key = next(self.genotype_indexer)
                        offsprings[new_genotype.genotype_key] = new_genotype
                offspring_list = [genotype for genotype in offsprings.values()]

                # [Evolution]
                print('|---------- Take Mutation ----------|')
                for i in range(self.TpG):
                    print(f'add cell - search index : {i+1}')
                    self.evolution.add_cell(offspring_list)
                for i in range(self.TpG):
                    print(f'modify cell - search index : {i+1}')
                    self.evolution.modify_cell(offspring_list)
                for i in range(self.TpG):
                    print(f'crossover cell - search index : {i+1}')
                    self.evolution.crossover(offspring_list)

                self.population.update(offsprings)

                # Save genotypes
                for genotype in self.population.values():
                    genotype.save_genotype(self.generation)

            # [Speciation]
            self.speciation.speciate(self.population)

            # [Training]
            training_list = []
            for species in self.speciation.species_dict.values():
                member_list = [member for member in species.members.values()]
                training_list += sample(member_list,
                                       k=ceil(len(member_list) * (self.train_config['train_rate'] / 100.0)))

            for genotype in training_list:
                print(f'genotype {genotype.genotype_key}({training_list.index(genotype)+1}-{len(training_list)})')
                if genotype.genotype_dict['fitness'] == 0.0:
                    ann = construct_ann(genotype, self.train_config['input_size'])
                    if ann.is_vaild:
                        print(ann.phenotype)
                        train = TrainANN(self. train_config, genotype.ann_config['ann_type'], ann)
                        train.train_ann(data_train)
                        fitness = train.valid_ann(data_valid)
                        genotype.genotype_dict['fitness'] = fitness
                        print(f'individual {genotype.genotype_key} fitness: {fitness}\n')
                        genotype.genotype_dict['fitness_history'].append(fitness)

                        if genotype.genotype_dict['fitness'] > self.train_config['fitness_threshold']:
                            print(f'individual {genotype.genotype_key} reached fitness threshold...')
                            train.train_ann(data_train, scheduler=True)
                            final_fitness = train.valid_ann(data_valid)
                            genotype.genotype_dict['final_fitness'] = final_fitness
                            print(f'individual {genotype.genotype_key} final fitness {final_fitness}\n')

                            if self.species_threshold > self.evolution_config['species_number_limit_floor']:
                                self.species_threshold -= 1

                        genotype.save_genotype(self.generation)
                        if genotype.genotype_dict['final_fitness'] > self.train_config['final_fitness_threshold']:
                            f = open(self.evolution_config['output_file'], 'a')
                            print(f'\nThe SATISFIED individual:', file=f)
                            genotype.print_architecture(f, True)
                            print(ann.phenotype, file=f)
                            f.close()
                            return
                    else:
                        self.population.pop(genotype.genotype_key)
                        print(f'{genotype.genotype_key} is not vaild, removed from population\n')
                else:
                    print('individual ', genotype.genotype_key, 'fitness:', genotype.genotype_dict['fitness'], '\n')

            sorted_genotypes = sorted(self.population.values(), key=lambda g: g.genotype_dict['fitness'], reverse=True)

            if self.best_genotype is None or \
                    sorted_genotypes[0].genotype_dict['fitness'] > self.best_genotype.genotype_dict['fitness']:
                self.best_genotype = deepcopy(sorted_genotypes[0])
                self.TpG = self.evolution_config['TpG_init']
            else:
                if self.TpG < self.NpI:
                    self.TpG += self.evolution_config['TpG_step']
                else:
                    self.TpG = self.evolution_config['TpG_init']
                    if self.NpI < self.evolution_config['NpI_limit']:
                        self.NpI += self.evolution_config['NpI_step']

            # update reserved species fitness
            for species_key, species in self.speciation.species_dict.items():
                species_representation = \
                    sorted([member for member in species.members.values()],
                           key=lambda m: m.genotype_dict['fitness'], reverse=True)[0]
                self.speciation.species_dict[species_key].update(species_representation.genotype_dict['fitness'],
                                                                 species_representation)

            # [Selection]
            # control species number
            current_genotype_number = 0
            if len(self.speciation.species_dict) > self.species_threshold:
                sorted_species = sorted(self.speciation.species_dict.values(),
                                        key=lambda s: s.fitness, reverse=True)
                new_species_dict = {}
                for i in range(self.species_threshold):
                    new_species_dict[sorted_species[i].species_key] = sorted_species[i]
                self.speciation.species_dict = new_species_dict
                for species in self.speciation.species_dict.values():
                    current_genotype_number += len(species.members)

            # control each species member number
            if current_genotype_number == 0:
                current_genotype_number = len(self.population)
            if current_genotype_number > self.evolution_config['genotype_number_limit']:
                for species in self.speciation.species_dict.values():
                    reserved_number = ceil(len(species.members) / current_genotype_number *
                                           self.evolution_config['genotype_number_limit'])
                    sorted_members = sorted(species.members.values(),
                                            key=lambda m: m.genotype_dict['fitness'], reverse=True)
                    new_members = {}
                    for i in range(reserved_number):
                        new_members[sorted_members[i].genotype_key] = sorted_members[i]
                    species.members = new_members

            # construct population for next generation
            next_population = {}
            for species in self.speciation.species_dict.values():
                for genotype in species.members.values():
                    next_population[genotype.genotype_key] = genotype
            self.population = next_population

            # calculate time
            end_time = time.time()
            total_time += (end_time - start_time)
            print(f'time so far: {total_time}\n')

            # output evolution information in current generation
            f = open(self.evolution_config['output_file'], 'a')
            print(f'|========== generation {self.generation} ==========|', file=f)
            print(f'NpI - {self.NpI}; TpG - {self.TpG}', file=f)
            self.speciation.output_species_dict_info(f)
            print(f'time so far: {total_time}\n', file=f)
            f.close()

            self.from_scratch = 1
            self.generation += 1

        return self.best_genotype

    def load_population(self, genotype_dir, generation_index):
        genotype_files = os.listdir(genotype_dir)
        for genotype_file in genotype_files:
            if len(str(generation_index)) == 1:
                genotype_generation = genotype_file[9]
            else:
                genotype_generation = genotype_file[9:11]

            if genotype_generation == str(generation_index):
                new_genotype = Genotype(0, genotype_dir+genotype_file, self.ann_config, use_old_key=True)
                self.population[new_genotype.genotype_key] = new_genotype

