
import numpy as np
import torch
from ga.utils import compute_norm, compute_misclassification, compute_avg_mse


def constrained_fitness_func(ga_instance, solution, solution_idx, model, epsilon_init, epsilon_end, penalty_factor):

    input_batch = ga_instance.user_data["input_batch"]
    labels = ga_instance.user_data["labels"]

    # 1) Convert chromosome to perturbation
    perturbation = torch.tensor(solution, device=input_batch.device, dtype=torch.float32).float().reshape(input_batch.shape[1:]) # [channel, height, width] (3*224*224) instead of (64*3*224*224)

    # 2) Calculate misclassification
    misclassification_score = compute_misclassification(model, input_batch, labels, perturbation)

    # 3) Compute visibility measure (xi)
    xi = compute_norm(perturbation)

    # print(f"Misclassification score: {misclassification_score}")
    # print(f"Epsilon: {epsilon}")
    # print(f"Xi: {xi}")
    # print(f"Penalty factor: {penalty_factor}")
    # print(f"xi-epsilon: {max(xi - epsilon, 0)}")

    # 4) Calculate epsilon (exponential decay)
    epsilon = epsilon_init * (epsilon_end / epsilon_init) ** (ga_instance.generations_completed / ga_instance.num_generations)
    ga_instance.user_data["epsilon"] = epsilon

    # 5) Enforce epsilon constraint
    if xi <= epsilon:
        fitness = misclassification_score

    else:
        # Penalty function instead of strict cutoff
        # penalty_factor = 1e-3
        # l = lambda_0 * (1 + solution_idx)
        fitness = misclassification_score - penalty_factor * max((xi - epsilon), 0)

    # print(f"Fitness: {misclassification_score}-{penalty_factor * max((xi - epsilon), 0)}={fitness}")

    return fitness


# def apply_pixel_constraints(perturbation, pixel_std, pixel_constraint_weight, max_perturbation_magnitude):
#     # Make sure the perturbation is within the pixel standard deviation
#     lower_bound = -pixel_constraint_weight * pixel_std
#     upper_bound = pixel_constraint_weight * pixel_std
#     # print(f"Perturbation before clamping: {torch.norm(perturbation).item()}")
#     perturbation = torch.clamp(perturbation, lower_bound, upper_bound)
#     print(f"Perturbation after clamping: {torch.norm(perturbation).item()}")

#     # Also apply global constraints
#     perturbation_magnitude = torch.norm(perturbation).item()

#     # print(f"Perturbation magnitude after channel-wise clamping: {perturbation_magnitude}")
#     if perturbation_magnitude > max_perturbation_magnitude:
#         perturbation = perturbation * (max_perturbation_magnitude / perturbation_magnitude)
#     print(f"Perturbation magnitude after global clamping: {torch.norm(perturbation).item()}")

#     return perturbation
