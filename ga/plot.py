import matplotlib.pyplot as plt


def plot_time_to_convergence(metrics_log, threshold=50.0):
    
    times = metrics_log["time"]
    norms = metrics_log["norm"]
    
    plt.plot(times, norms, label="Norm over time")
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"{threshold} threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Perturbation Norm")
    plt.title("Time to Convergence for Norm < 50")
    plt.legend()
    plt.show()

    # You can also find the time at which norm < threshold for the first time:
    for t, n in zip(times, norms):
        if n < threshold:
            print(f"Reached norm < {threshold} at t={t:.2f} seconds")
            break







def plot_snr(snrs, title="SNR for Each Image After Attack"):
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, marker='x', linestyle='-', color='r')
    plt.xlabel("Image Index")
    plt.ylabel("SNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_psnr(psnrs, title="Peak SNR for Each Image After Attack"):
    plt.figure(figsize=(10, 6))
    plt.plot(psnrs, marker='o', linestyle='-', color='b')
    plt.xlabel("Image Index")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_norm(metrics_log, title="Perturbation Norm Over Generations"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["gen"], metrics_log["norm"], label="Perturbation Norm", color='y')
    plt.xlabel("Generation")
    plt.ylabel("Perturbation Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_avg_mse(metrics_log, title="Average MSE Over Generations"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["gen"], metrics_log["avg_mse"], label="Average MSE", color='m')
    plt.xlabel("Generation")
    plt.ylabel("Average MSE")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_misclassification(metrics_log, title="Misclassification Rate (on each batch) Over Generations"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["gen"], metrics_log["misclassification"], label="Attack Success Rate", color='c')
    plt.xlabel("Generation")
    plt.ylabel("Misclassification Rate")
    plt.xticks(range(0, len(metrics_log["gen"]), 4))
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confidence_score(metrics_log, title="Average Confidence Score Over Generations"):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["gen"], metrics_log["confidence"], label="Confidence Score", color='g')
    plt.xlabel("Generation")
    plt.ylabel("Average Confidence Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()