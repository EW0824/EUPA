import numpy as np


########
# Pixel Cleaning Step
########

def pixel_cleaning_operation(offspring, cleaning_probability=0.1):
    for i, chromosome in enumerate(offspring):

        # Pre cleaning
        # print(f"Chromosome {i} before: {chromosome[:10]}")

        for g_index in range(len(chromosome)):
            # For every pixel in the perturbation, have a change to be set to zero
            if np.random.rand() < cleaning_probability:
                chromosome[g_index] = 0.0

        # Post cleaning
        # print(f"Chromosome {i} after: {chromosome[:10]}")
    return offspring
