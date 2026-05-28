import matplotlib.pyplot as plt


def plot_losses(history,
                outpath="loss_curves.png"):

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Total loss
    axs[0,0].plot(history["train_total"], marker='o', label="Train")
    axs[0,0].plot(history["val_total"], marker='.', label="Val")
    axs[0,0].set_title("Total Loss")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].grid(True); axs[0,0].legend()

    # Similarity loss
    axs[0,1].plot(history["train_sim"], marker='o', label="Train")
    axs[0,1].plot(history["val_sim"], marker='.', label="Val")
    axs[0,1].set_title("Similarity Loss")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Loss")
    axs[0,1].grid(True); axs[0,1].legend()

    # Regularizer loss
    axs[1,0].plot(history["train_reg"], marker='o', label="Train")
    axs[1,0].plot(history["val_reg"], marker='.', label="Val")
    axs[1,0].set_title("Regularizer Loss")
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("Loss")
    axs[1,0].grid(True); axs[1,0].legend()

    # Cavity loss
    axs[1,1].plot(history["train_cav"], marker='o', label="Train")
    axs[1,1].plot(history["val_cav"], marker='.', label="Val")
    axs[1,1].set_title("Cavity Loss")
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("Loss")
    axs[1,1].grid(True); axs[1,1].legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved loss curves to {outpath}")