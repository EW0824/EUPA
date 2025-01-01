import numpy as np
import torch
from ga.components.crossover import pixel_cleaning_operation
from ga.utils import visualize_images_batch

def on_crossover_func(ga_instance, offspring, config):
    # print("Number of zeros before pixel cleaning in crossover:", torch.sum(torch.tensor(offspring == 0)))
    cleaning_probability =  config["crossover"]["pixel_cleaning_probability"]
    offspring = pixel_cleaning_operation(offspring, cleaning_probability)
    return offspring

def on_generation_func(ga_instance, input_batch, top_perturbations, config):

    print(f"\nGeneration {ga_instance.generations_completed} completed with fitness: {ga_instance.last_generation_fitness}")

    # print(f"Input batch starting label: {labels[0]}")

    # New batch every generation
    # input_batch, labels = next(iter(dataloader))
    # input_batch, labels = input_batch.to(device), labels.to(device)
    # print(f"New batch loaded with first label: {labels[0]}")
    
    # Print the best fitness for this generation
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best Fitness = {best_solution_fitness}\n")

    # Possibly check partial genes in best solution
    snippet = best_solution[:10]
    print(f"Best solution snippet: {snippet}")

    num_zeros = np.count_nonzero(np.isclose(best_solution, 0.0))
    print(f"Best solution zeros count: {num_zeros}")

    # # Print out samples from the top 3 perturbations
    # top_fitness = sorted(ga_instance.last_generation_fitness)[-3:]
    # for i, perturbation in enumerate(top_fitness):
    #     print(f"Top {i+1} perturbation: {torch.norm(perturbation).item()}")

    best_perturbation = torch.tensor(best_solution).float().reshape(input_batch.shape[1:])
    top_perturbations.append(best_perturbation)
    print(f"Best perturbation magnitude: {torch.norm(best_perturbation).item()}")

    ########
    # VISUALIZATION
    ########
    if config["visualization"]["visualize"] and ga_instance.generations_completed % config["visualization"]["visualize_every"] == 0:
        print(f"Visualizing")
        # get the current best perturbation
        visualize_images_batch(input_batch, best_perturbation)

    # print(f"Generation {ga_instance.generations_completed}: Current Fitness: Best Fitness = {ga_instance.best_solution()[1]}")



def calculate_dynamic_probs(ga_instance):

    current_gen = ga_instance.generations_completed
    total_gens = ga_instance.num_generations
    frac = current_gen / total_gens

    user_data = ga_instance.user_data
    cross_start = user_data["cross_start"]
    cross_end = user_data["cross_end"]
    mut_start = user_data["mut_start"]
    mut_end = user_data["mut_end"]

    cross_prob = cross_start - frac * (cross_end - cross_start)
    mut_prob = mut_start - frac * (mut_end - mut_start)

    return cross_prob, mut_prob