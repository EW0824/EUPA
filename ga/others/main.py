import pygad
import numpy as np
import torch
import os
from ga.components.fitness import constrained_fitness_func
from ga.components.model import load_model
from ga.utils import get_dataloader, visualize_image_perturbation,visualize_image_perturbation_batch, visualize_perturbation, compute_pixel_statistics, config
# from nn.models.googlenet import create_googlenet

########
# PARAMS
########
num_generations =num_generations=config["ga"]["num_generations"]
num_parents_mating=config["ga"]["num_parents_mating"]
sol_per_pop=config["ga"]["sol_per_pop"]
init_range_low=config["ga"]["init_range_low"]
init_range_high=config["ga"]["init_range_high"]
mutation_percent_genes=config["ga"]["mutation_percent_genes"]

model_type = config["model"]["model_type"]
batch_size = config["model"]["batch_size"]

visualize = config["visualization"]["visualize"]
visualize_every = config["visualization"]["visualize_every"]


########
# MODEL
########

model = load_model(model_type)
# input_tensor = torch.randn(1, 3, 224, 224)  # Random example
# original_label = 123


########
# LOADING
########

# A single image
# input_image_path = "nn/data/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG"
# input_tensor = preprocess_image(input_image_path)
# original_label = 0

# Load
image_dir = os.path.join(os.getcwd(), "nn/data/imagenet/val")
dataloader = get_dataloader(batch_size, image_dir)
# Compute the pixel mean and standard deviation for each pixel across the entire dataset - for constrained fitness func
pixel_mean, pixel_std = compute_pixel_statistics(dataloader)

# First batch
input_batch, original_labels = next(iter(dataloader))

# Collect top perturbations from each generation
top_perturbations = []


########
# GA CONFIG
########


def fitness_wrapper(ga_instance, solution, solution_idx):
    return constrained_fitness_func(
        ga_instance, solution, solution_idx, pixel_std, model, input_batch, original_labels
    )

def on_generation(ga_instance):

    print(f"\nGeneration {ga_instance.generations_completed} completed with fitness: {ga_instance.last_generation_fitness}")

    input_batch, original_labels = next(iter(dataloader))
    print(f"New batch loaded with first label: {original_labels[0]}")
    
    # Print the best fitness for this generation
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best Fitness = {best_solution_fitness}\n")
    top_perturbations.append(best_solution)

    ########
    # VISUALIZATION
    ########
    if visualize and ga_instance.generations_completed % visualize_every == 0:
        print(f"Visualizing")
        # get the current best perturbation
        perturbation = torch.tensor(best_solution).float().reshape(input_batch.shape)
        visualize_image_perturbation_batch(input_batch, perturbation)

    # print(f"Generation {ga_instance.generations_completed}: Current Fitness: Best Fitness = {ga_instance.best_solution()[1]}")


ga_instance = pygad.GA(
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    sol_per_pop=sol_per_pop,
    num_genes=input_batch.numel(),
    gene_type=float,
    init_range_low=init_range_low,
    init_range_high=init_range_high,
    mutation_percent_genes=mutation_percent_genes,
    fitness_func=fitness_wrapper,
    on_generation=on_generation,
)

ga_instance.run()

# After the run
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Best solution fitness: {solution_fitness}") # Might be the penultimate best fitness??

########
# CALCULATING UNIVERSAL PERTURBATION
########
if len(top_perturbations) > 0:
    universal_perturbation = torch.mean(torch.stack(top_perturbations), dim=0)
    print(f"Universal perturbation created.")
    visualize_perturbation(universal_perturbation)

########
# GETTING THE METRICS
########

# Evaluate the model without the universal perturbation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f"Accuracy without perturbation: {accuracy}")

# Evaluate the model with the universal perturbation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        perturbed_images = images + universal_perturbation
        perturbed_images = torch.clamp(perturbed_images, 0, 1) # Ensure the pixel values are between 0 and 1
        outputs = model(perturbed_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()