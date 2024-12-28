import numpy as np
import torch
from ga.fitness import apply_pixel_constraints

def integer_mutation(offspring, ga_instance):

    # # Apply pixel cleaning operation before mutation
    # offspring = pixel_cleaning_operation(offspring)

    for chromosome in offspring:
        # Num of genes to mutate
        num_genes = len(chromosome)
        num_mutations = int(num_genes * ga_instance.mutation_percent_genes / 100.0)
        mutation_indices = np.random.choice(num_genes, size=num_mutations, replace=False)

        # Flip each selected gene
        for mi in mutation_indices:
            chromosome[mi] = np.random.randint(ga_instance.init_range_low, ga_instance.init_range_high + 1)
            # Flip from 0 to any int, or any int to zero or another int
    return offspring



def custom_mutation(offspring, ga_instance, pixel_std, pixel_constraint_weight, max_perturbation_magnitude, input_batch, apply_pixel_constraints=True,):
    for chromosome in offspring:        
        # Calculate the number of genes to mutate
        num_genes = len(chromosome)
        num_mutations = int(num_genes * ga_instance.mutation_percent_genes / 100.0)
        # print(f"Number of mutations: {num_mutations}")

        # Select random genes to mutate
        mutation_indices = np.random.choice(num_genes, size=num_mutations, replace=False)

        mutation_values = np.random.uniform(ga_instance.random_mutation_min_val, 
                                            ga_instance.random_mutation_max_val, size=num_mutations)

        chromosome[mutation_indices] += mutation_values

        # Apply constraints after mutation
        perturbation = torch.tensor(chromosome).float().reshape(input_batch.shape[1:])
        if apply_pixel_constraints:
            perturbation = apply_pixel_constraints(perturbation, pixel_std, pixel_constraint_weight, max_perturbation_magnitude)

        chromosome[:] = perturbation.flatten().numpy()

    return offspring