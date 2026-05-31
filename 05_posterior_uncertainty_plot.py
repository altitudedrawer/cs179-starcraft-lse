import random
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions import constraints

# We only use matplotlib for drawing, NO scipy needed at all!
import matplotlib.pyplot as plt
import math

# =========================================================================
# 1. Basic Settings & Reproducibility
# =========================================================================

# Setting seeds so our results and plots are exactly the same every time we run it
SEED = 179
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)

# Set to None to run the full training dataset as requested by the professor!
TRAIN_LIMIT = None

NUM_STEPS = 2000
LEARNING_RATE = 0.03
PRIOR_SCALE = 1.0  # This is the standard deviation \sigma_0 for our N(0, 1) prior

# The StarCraft dataset doesn't have a header, so we manually define the column names
COLUMN_NAMES = [
    "date", "player1", "result1", "score", "player2", "result2",
    "race1", "race2", "version", "mode"
]

# =========================================================================
# 2. Data Loading & Data Engineering
# =========================================================================

def load_matches(train_path="train.csv", valid_path="valid.csv"):
    """
    Loads the CSV files and maps who won each match into a binary outcome.
    outcome = 1 means player1 won.
    outcome = 0 means player2 won.
    """
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    valid_df = pd.read_csv(valid_path, header=None, names=COLUMN_NAMES)

    train_df["outcome"] = (train_df["result1"] == "[winner]").astype(int)
    valid_df["outcome"] = (valid_df["result1"] == "[winner]").astype(int)

    return train_df, valid_df

def build_player_mapping(train_df, valid_df):
    """
    Creates a unique mapping dictionary to convert player string names into numerical IDs.
    """
    all_players = pd.concat([
        train_df["player1"], train_df["player2"],
        valid_df["player1"], valid_df["player2"]
    ]).unique()

    player_to_id = {player: idx for idx, player in enumerate(all_players)}
    id_to_player = {idx: player for player, idx in player_to_id.items()}

    return player_to_id, id_to_player

def encode_matches(df, player_to_id):
    """
    Maps player names inside a DataFrame to Torch Tensors so Pyro can read them.
    """
    p1_ids = df["player1"].map(player_to_id).values
    p2_ids = df["player2"].map(player_to_id).values
    outcomes = df["outcome"].values

    p1_tensor = torch.tensor(p1_ids, dtype=torch.long)
    p2_tensor = torch.tensor(p2_ids, dtype=torch.long)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float32)

    return p1_tensor, p2_tensor, outcome_tensor

# =========================================================================
# 3. Probabilistic Graphical Model (PGM) Architecture
# =========================================================================

def model(p1_ids, p2_ids, outcomes, num_players):
    """
    The Generative Model with a standard normal prior N(0, 1) over skills.
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
    The Variational Distribution q(s) defining learnable mu and sigma parameters.
    """
    skill_loc = pyro.param("skill_loc", torch.zeros(num_players))
    skill_scale = pyro.param(
        "skill_scale",
        0.5 * torch.ones(num_players),
        constraint=constraints.positive
    )

    pyro.sample("skills", dist.Normal(skill_loc, skill_scale).to_event(1))

# =========================================================================
# 4. Evaluation Helper Functions
# =========================================================================

def accuracy_from_probs(probs, outcomes):
    preds = (probs >= 0.5).astype(int)
    return np.mean(preds == outcomes)

def log_loss_from_probs(probs, outcomes):
    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)
    losses = -(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return np.mean(losses)

def predict_with_skill_means(p1_ids, p2_ids):
    skill_loc = pyro.param("skill_loc").detach()
    skill_diff = skill_loc[p1_ids] - skill_loc[p2_ids]
    probs = torch.sigmoid(skill_diff)
    return probs.cpu().numpy()

# =========================================================================
# 5. Pure Python Gaussian PDF Formula (Replaces Scipy!)
# =========================================================================

def pure_gaussian_pdf(x_array, mu, sigma):
    """
    This is our pure Python/Numpy function that calculates the standard Gaussian formula.
    It takes an array of x points and manually computes the bell curve values.
    NO external libraries needed!
    """
    # Part 1: 1 / (sigma * sqrt(2 * pi))
    coefficient = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    
    # Part 2: -0.5 * ((x - mu) / sigma)^2
    exponent = -0.5 * ((x_array - mu) / sigma) ** 2
    
    # Combined: coefficient * e^(exponent)
    return coefficient * np.exp(exponent)

def plot_uncertainty_comparison(train_df, player_to_id):
    """
    Finds the most active and least active player, extracts their parameters,
    and calls our custom pure_gaussian_pdf function to output the final plot image.
    """
    print("\n[Plotting Engine] Analyzing player match frequencies...")
    
    # Combine player1 and player2 columns to count total matches played per person
    all_match_players = pd.concat([train_df["player1"], train_df["player2"]])
    match_counts = all_match_players.value_counts()
    
    # 1. Identify the 'Frequent Pro' (Player with the maximum number of matches)
    frequent_player = match_counts.index[0]
    frequent_count = match_counts.iloc[0]
    
    # 2. Identify the 'Rare Rookie' (Player with exactly 1 match)
    rare_players = match_counts[match_counts == 1]
    if len(rare_players) > 0:
        rare_player = rare_players.index[0]
    else:
        rare_player = match_counts.index[-1]
    rare_count = match_counts.loc[rare_player]
    
    # Find their mapped numerical Pyro internal tensor IDs
    freq_id = player_to_id[frequent_player]
    rare_id = player_to_id[rare_player]
    
    # Extract the optimized variational parameters (mu and sigma) from Pyro Param Store
    skill_loc = pyro.param("skill_loc").detach().cpu().numpy()
    skill_scale = pyro.param("skill_scale").detach().cpu().numpy()
    
    mu_freq, sigma_freq = skill_loc[freq_id], skill_scale[freq_id]
    mu_rare, sigma_rare = skill_loc[rare_id], skill_scale[rare_id]
    
    print(f"-> Selected Frequent Pro: {frequent_player} ({frequent_count} matches) | mu={mu_freq:.4f}, sigma={sigma_freq:.4f}")
    print(f"-> Selected Rare Rookie: {rare_player} ({rare_count} match) | mu={mu_rare:.4f}, sigma={sigma_rare:.4f}")
    
    # Setup matplotlib canvas
    plt.figure(figsize=(8, 5), dpi=300)
    
    # Dynamically establish plot grid bounds on the x-axis
    x_min = min(mu_freq - 3 * sigma_freq, mu_rare - 3 * sigma_rare)
    x_max = max(mu_freq + 3 * sigma_freq, mu_rare + 3 * sigma_rare)
    x = np.linspace(x_min, x_max, 500)
    
    # CRITICAL: We call our manual function here instead of scipy!
    pdf_freq = pure_gaussian_pdf(x, mu_freq, sigma_freq)
    pdf_rare = pure_gaussian_pdf(x, mu_rare, sigma_rare)
    
    # Plotting Curve 1: The Frequent Pro (Narrow, tall and sharp)
    plt.plot(x, pdf_freq, label=f'Frequent Player ({frequent_player}, {frequent_count} matches)', color='teal', linewidth=2.5)
    plt.fill_between(x, pdf_freq, alpha=0.15, color='teal')
    
    # Plotting Curve 2: The Rare Rookie (Wide, short and flat)
    plt.plot(x, pdf_rare, label=f'Rare Player ({rare_player}, {rare_count} match)', color='crimson', linewidth=2.5, linestyle='--')
    plt.fill_between(x, pdf_rare, alpha=0.15, color='crimson')
    
    # Format and label the final academic figure
    plt.title("Posterior Skill Distributions: Frequent vs. Rare Player", fontsize=12, fontweight='bold')
    plt.xlabel("Latent Skill Value ($s$)", fontsize=10)
    plt.ylabel("Probability Density (PDF)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    
    # Output the image to your workspace folder
    plt.tight_layout()
    plt.savefig("figure3_uncertainty_comparison.png")
    plt.close()
    print("[Plotting Engine] Done! Chart successfully saved as 'figure3_uncertainty_comparison.png'")

# =========================================================================
# 6. Main Execution Pipeline
# =========================================================================

def main():
    pyro.clear_param_store()

    # Load data from workspace directory
    train_df, valid_df = load_matches()

    print("Original train shape:", train_df.shape)
    print("Original valid shape:", valid_df.shape)

    if TRAIN_LIMIT is not None:
        train_df_used = train_df.iloc[:TRAIN_LIMIT].copy()
    else:
        train_df_used = train_df.copy()

    print("Training rows used:", len(train_df_used))

    # Construct name indexing mappings
    player_to_id, id_to_player = build_player_mapping(train_df_used, valid_df)
    num_players = len(player_to_id)
    print("Total unique network nodes (players):", num_players)

    # Encode pandas columns to tensors
    train_p1, train_p2, train_y = encode_matches(train_df_used, player_to_id)
    valid_p1, valid_p2, valid_y = encode_matches(valid_df, player_to_id)

    # Set up the Adam optimizer and the SVI loss routine (ELBO)
    optimizer = Adam({"lr": LEARNING_RATE})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print("\n[Pyro Training] Running Stochastic Variational Inference (SVI)...")
    for step in range(NUM_STEPS):
        loss = svi.step(train_p1, train_p2, train_y, num_players)

        if step % 100 == 0 or step == NUM_STEPS - 1:
            avg_loss = loss / len(train_y)
            print(f"Step {step:4d} | Average ELBO loss per match: {avg_loss:.4f}")

    print("\n[Evaluation] Calculating metrics on validation set...")
    valid_probs = predict_with_skill_means(valid_p1, valid_p2)
    valid_outcomes = valid_y.cpu().numpy().astype(int)

    acc = accuracy_from_probs(valid_probs, valid_outcomes)
    log_loss = log_loss_from_probs(valid_probs, valid_outcomes)

    print("\n==============================")
    print("  Pyro SVI Model Final Metrics")
    print("==============================")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss: {log_loss:.4f}")

    # Export metrics metadata to csv for later bar chart formatting
    results = pd.DataFrame([{
        "model": "Pyro SVI Latent Skill Model",
        "accuracy": acc,
        "log_loss": log_loss,
        "train_rows_used": len(train_df_used),
        "num_steps": NUM_STEPS,
    }])
    results.to_csv("pyro_results.csv", index=False)
    print("\n[Export] Saved validation summary metrics to 'pyro_results.csv'")

    # Save ranking lists sorted by learned latent mean values
    skill_loc = pyro.param("skill_loc").detach().cpu().numpy()
    skill_df = pd.DataFrame({
        "player_id": list(range(num_players)),
        "player": [id_to_player[i] for i in range(num_players)],
        "estimated_skill_mean": skill_loc,
    }).sort_values("estimated_skill_mean", ascending=False)
    
    skill_df.to_csv("learned_skills.csv", index=False)
    print("[Export] Mapped skills saved to 'learned_skills.csv'")

    # Trigger our newly embedded Bayesian plotting pipeline automatically
    plot_uncertainty_comparison(train_df_used, player_to_id)

if __name__ == "__main__":
    main()