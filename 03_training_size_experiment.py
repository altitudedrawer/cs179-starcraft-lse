import os
import random
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions import constraints
import matplotlib.pyplot as plt


# =========================
# Basic settings
# =========================

SEED = 179

NUM_STEPS = 2000
LEARNING_RATE = 0.03
PRIOR_SCALE = 1.0

COLUMN_NAMES = [
    "date",
    "player1",
    "result1",
    "score",
    "player2",
    "result2",
    "race1",
    "race2",
    "version",
    "mode",
]


# =========================
# Reproducibility
# =========================

def set_seed(seed):
    """
    Set random seeds so the experiment is more reproducible.
    Pyro SVI has randomness, so fixing the seed helps keep results stable.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


# =========================
# Load and prepare data
# =========================

def load_matches(train_path="train.csv", valid_path="valid.csv"):
    """
    Load the StarCraft match data.
    The CSV files do not have headers, so I manually add column names.
    """

    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    valid_df = pd.read_csv(valid_path, header=None, names=COLUMN_NAMES)

    # outcome = 1 means player1 wins.
    # outcome = 0 means player1 loses, so player2 wins.
    train_df["outcome"] = (train_df["result1"] == "[winner]").astype(int)
    valid_df["outcome"] = (valid_df["result1"] == "[winner]").astype(int)

    return train_df, valid_df


def build_player_mapping(train_df, valid_df):
    """
    Build a mapping from player name to integer ID.

    I include players from both the training subset and the validation set.
    If a validation player never appears in training, its skill will stay
    close to the prior mean, which is a reasonable fallback.
    """

    all_players = pd.concat([
        train_df["player1"],
        train_df["player2"],
        valid_df["player1"],
        valid_df["player2"],
    ]).unique()

    player_to_id = {player: idx for idx, player in enumerate(all_players)}

    return player_to_id


def encode_matches(df, player_to_id):
    """
    Convert player names into integer IDs and return PyTorch tensors.
    """

    p1_ids = df["player1"].map(player_to_id).values
    p2_ids = df["player2"].map(player_to_id).values
    outcomes = df["outcome"].values

    p1_tensor = torch.tensor(p1_ids, dtype=torch.long)
    p2_tensor = torch.tensor(p2_ids, dtype=torch.long)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float32)

    return p1_tensor, p2_tensor, outcome_tensor


# =========================
# Pyro model and guide
# =========================

def model(p1_ids, p2_ids, outcomes, num_players):
    """
    Latent skill model.

    Each player has one hidden skill value.
    For a match between player1 and player2, the win probability depends on
    the difference between their skill values.
    """

    skills = pyro.sample(
        "skills",
        dist.Normal(
            torch.zeros(num_players),
            PRIOR_SCALE * torch.ones(num_players)
        ).to_event(1)
    )

    skill_diff = skills[p1_ids] - skills[p2_ids]
    win_probs = torch.sigmoid(skill_diff)

    with pyro.plate("matches", len(outcomes)):
        pyro.sample("obs", dist.Bernoulli(probs=win_probs), obs=outcomes)


def guide(p1_ids, p2_ids, outcomes, num_players):
    """
    Variational guide.

    I approximate the posterior distribution over player skills using
    independent Normal distributions.
    """

    skill_loc = pyro.param(
        "skill_loc",
        torch.zeros(num_players)
    )

    skill_scale = pyro.param(
        "skill_scale",
        0.5 * torch.ones(num_players),
        constraint=constraints.positive
    )

    pyro.sample(
        "skills",
        dist.Normal(skill_loc, skill_scale).to_event(1)
    )


# =========================
# Evaluation helpers
# =========================

def accuracy_from_probs(probs, outcomes):
    """
    Convert probabilities into winner predictions.
    If prob >= 0.5, predict player1 wins.
    """

    preds = (probs >= 0.5).astype(int)
    return np.mean(preds == outcomes)


def log_loss_from_probs(probs, outcomes):
    """
    Compute log loss for probabilistic predictions.
    Lower log loss means better probability estimates.
    """

    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)

    losses = -(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return np.mean(losses)


def predict_with_skill_means(p1_ids, p2_ids):
    """
    Use learned posterior mean skills to predict validation matches.
    """

    skill_loc = pyro.param("skill_loc").detach()

    skill_diff = skill_loc[p1_ids] - skill_loc[p2_ids]
    probs = torch.sigmoid(skill_diff)

    return probs.cpu().numpy()


# =========================
# Train one model
# =========================

def run_one_training_size(train_df_full, valid_df, fraction):
    """
    Train the same Pyro SVI model using a fraction of the training data.
    Then evaluate it on the same validation set.
    """

    set_seed(SEED)

    pyro.clear_param_store()

    total_rows = len(train_df_full)
    train_rows = int(total_rows * fraction)

    train_df = train_df_full.iloc[:train_rows].copy()

    player_to_id = build_player_mapping(train_df, valid_df)
    num_players = len(player_to_id)

    train_p1, train_p2, train_y = encode_matches(train_df, player_to_id)
    valid_p1, valid_p2, valid_y = encode_matches(valid_df, player_to_id)

    optimizer = Adam({"lr": LEARNING_RATE})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print("\n====================================")
    print(f"Training size: {int(fraction * 100)}%")
    print(f"Training rows used: {train_rows}")
    print(f"Number of players: {num_players}")
    print("====================================")

    final_avg_loss = None

    for step in range(NUM_STEPS):
        loss = svi.step(train_p1, train_p2, train_y, num_players)
        avg_loss = loss / len(train_y)

        if step % 200 == 0 or step == NUM_STEPS - 1:
            print(f"Step {step:4d} | Average ELBO loss per match: {avg_loss:.4f}")

        final_avg_loss = avg_loss

    valid_probs = predict_with_skill_means(valid_p1, valid_p2)
    valid_outcomes = valid_y.cpu().numpy().astype(int)

    acc = accuracy_from_probs(valid_probs, valid_outcomes)
    log_loss = log_loss_from_probs(valid_probs, valid_outcomes)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss: {log_loss:.4f}")

    return {
        "training_fraction": fraction,
        "training_percent": int(fraction * 100),
        "training_rows": train_rows,
        "num_players": num_players,
        "num_steps": NUM_STEPS,
        "final_avg_elbo_loss": final_avg_loss,
        "accuracy": acc,
        "log_loss": log_loss,
    }


# =========================
# Plot Figure 2
# =========================

def plot_training_size_results(results_df):
    """
    Make one figure with two side-by-side panels:
    accuracy and log loss as training data size changes.
    """

    os.makedirs("figures", exist_ok=True)

    x = results_df["training_percent"].values
    accuracy = results_df["accuracy"].values
    log_loss = results_df["log_loss"].values

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    line_color = "forestgreen"

    # -------------------------
    # Panel (a): Accuracy
    # -------------------------
    ax1 = axes[0]
    ax1.plot(x, accuracy, marker="o", color=line_color)
    ax1.set_title("(a) Accuracy", fontsize=11)
    ax1.set_xlabel("Training Data Used (%)")
    ax1.set_ylabel("Validation Accuracy")
    ax1.set_xticks(x)

    # Add labels near points.
    for xi, yi in zip(x, accuracy):
        ax1.text(xi, yi + 0.002, f"{yi:.4f}", ha="center", va="bottom", fontsize=8)

    # -------------------------
    # Panel (b): Log Loss
    # -------------------------
    ax2 = axes[1]
    ax2.plot(x, log_loss, marker="o", color=line_color)
    ax2.set_title("(b) Log Loss", fontsize=11)
    ax2.set_xlabel("Training Data Used (%)")
    ax2.set_ylabel("Validation Log Loss")
    ax2.set_xticks(x)

    # Add labels near points.
    for xi, yi in zip(x, log_loss):
        ax2.text(xi, yi + 0.001, f"{yi:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/figure2_training_size_effect.png", dpi=300)
    plt.show()

    print("\nSaved figure:")
    print("figures/figure2_training_size_effect.png")


# =========================
# Main function
# =========================

def main():
    train_df_full, valid_df = load_matches()

    print("Original train shape:", train_df_full.shape)
    print("Original valid shape:", valid_df.shape)

    # Full experiment: 25%, 50%, 75%, and 100%.
    training_fractions = [0.25, 0.50, 0.75, 1.00]

    all_results = []

    for fraction in training_fractions:
        result = run_one_training_size(train_df_full, valid_df, fraction)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv("training_size_results.csv", index=False)

    print("\nTraining size experiment results:")
    print(results_df)

    print("\nSaved results to training_size_results.csv")

    plot_training_size_results(results_df)


if __name__ == "__main__":
    main()