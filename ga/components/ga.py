import numpy as np
import torch
from numpy import linalg
from ga.components.crossover import apply_pixel_cleaning
from ga.utils import visualize_images_batch, compute_norm, compute_misclassification, compute_avg_mse, compute_confidence_score

########
# ON CROSSOVER
########

def on_crossover_func(ga_instance, offspring, config):
    # print("Number of zeros before pixel cleaning in crossover:", torch.sum(torch.tensor(offspring == 0)))
    
    norms = linalg.norm(offspring, axis=1)

    for i, chromosome in enumerate(offspring):
        if norms[i] > ga_instance.user_data["epsilon"]:
            cleaning_probability =  config["crossover"]["pixel_cleaning_probability"]
            offspring[i] = apply_pixel_cleaning(chromosome, cleaning_probability)
    return offspring


########
# ON GENERATION
########

def on_generation_func(ga_instance, dataloader, model, device, config, metrics_log):

    input_batch = ga_instance.user_data["input_batch"]
    labels = ga_instance.user_data["labels"]
    top_perturbations = ga_instance.user_data["top_perturbations"]
    input_batch, labels = input_batch.to(device), labels.to(device)

    ########
    # LOGGING
    ########
    gen = ga_instance.generations_completed
    print(f"\n--- Generation {gen} Completed ---")
    # print(f"\nGeneration {ga_instance.generations_completed} completed with fitness scores: {ga_instance.last_generation_fitness}")

    # Best solution
    best_sol, best_fit, _ = ga_instance.best_solution()
    best_perturb = torch.tensor(best_sol).float().reshape(input_batch.shape[1:]).float().to(device)
    print(f"Best Fitness = {best_fit}")

    # # Possibly check partial genes in best solution
    # snippet = best_sol[:10]
    # print(f"Best solution snippet: {snippet}")
    # num_zeros = np.count_nonzero(np.isclose(best_sol, 0.0))
    # print(f"Best solution zeros count: {num_zeros}")

    # 1. Norm
    norm_val = compute_norm(best_perturb)
    print(f"Best perturbation norm: {norm_val}")

    # 2. MSE
    mse_val = compute_avg_mse(input_batch, best_perturb)
    print(f"Best perturbation MSE: {mse_val}")

    # 3. Misclassification rate
    misclassification_score = compute_misclassification(model, input_batch, labels, best_perturb)
    print(f"Misclassification score: {misclassification_score}")


    # 4. Confidence score
    confidence_score = compute_confidence_score(model, input_batch, labels)
    print(f"Confidence score: {confidence_score}")

    # Logging the metrics
    metrics_log["gen"].append(gen)
    metrics_log["norm"].append(norm_val)
    metrics_log["avg_mse"].append(mse_val)
    metrics_log["misclassification"].append(misclassification_score)
    metrics_log["confidence"].append(confidence_score)


    ########
    # SWITCHING BATCHES
    ########
    current_gen = ga_instance.generations_completed
    if current_gen % 4 == 0:
        input_batch, labels = next(iter(dataloader))
        input_batch, labels = input_batch.to(device), labels.to(device)
        ga_instance.user_data["input_batch"] = input_batch
        ga_instance.user_data["labels"] = labels
        print(f"New batch loaded with first label: {labels[0]}")

    # print(f"Input batch starting label: {labels[0]}")

    # New batch every generation
    # input_batch, labels = next(iter(dataloader))
    # input_batch, labels = input_batch.to(device), labels.to(device)
    


    ########
    # VISUALIZATION
    ########
    if config["visualization"]["visualize"] and ga_instance.generations_completed % config["visualization"]["visualize_every"] == 0:
        print(f"Visualizing")
        # get the current best perturbation
        visualize_images_batch(input_batch, best_perturb)

    ########
    # MODIFY DYNAMIC PROBABILITIES
    ########

    cross_prob, mut_prob = calculate_dynamic_probs(ga_instance, config)
    ga_instance.crossover_probability = cross_prob
    ga_instance.mutation_probability = mut_prob
    print(f"New crossover & mutation probabilities: {cross_prob}, {mut_prob}")

    # print(f"Generation {ga_instance.generations_completed}: Current Fitness: Best Fitness = {ga_instance.best_sol()[1]}")


########
# DYNAMIC PROBABILITIES
########

def calculate_dynamic_probs(ga_instance, config):

    current_gen = ga_instance.generations_completed
    total_gens = ga_instance.num_generations
    frac = current_gen / total_gens

    # user_data = ga_instance.user_data
    cross_start = config["crossover"]["p_crossover_init"]
    cross_end = config["crossover"]["p_crossover_end"]
    mut_start = config["mutation"]["p_mutation_init"]
    mut_end = config["mutation"]["p_mutation_end"]

    cross_prob = cross_start - frac * (cross_start - cross_end)
    mut_prob = mut_start - frac * (mut_start - mut_end)

    return cross_prob, mut_prob