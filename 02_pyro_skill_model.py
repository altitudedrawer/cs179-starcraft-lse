import random
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions import constraints


# =========================
# Basic settings
# =========================

SEED = 179
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)

# First run: use a smaller training subset to check that everything works.
# Later, change TRAIN_LIMIT to None to use the full training set.
TRAIN_LIMIT = None

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
    Build a player name -> integer ID mapping.

    I include players from both train and validation.
    If a validation player never appears in training, the model will keep
    that player's skill close to the prior mean, which is a reasonable default.
    """

    all_players = pd.concat([
        train_df["player1"],
        train_df["player2"],
        valid_df["player1"],
        valid_df["player2"],
    ]).unique()

    player_to_id = {player: idx for idx, player in enumerate(all_players)}
    id_to_player = {idx: player for player, idx in player_to_id.items()}

    return player_to_id, id_to_player


def encode_matches(df, player_to_id):
    """
    Convert player names into integer IDs and return tensors for Pyro.
    """

    p1_ids = df["player1"].map(player_to_id).values
    p2_ids = df["player2"].map(player_to_id).values
    outcomes = df["outcome"].values

    p1_tensor = torch.tensor(p1_ids, dtype=torch.long)
    p2_tensor = torch.tensor(p2_ids, dtype=torch.long)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float32)

    return p1_tensor, p2_tensor, outcome_tensor


# =========================
# Pyro latent skill model
# =========================

def model(p1_ids, p2_ids, outcomes, num_players):
    """
    Latent skill model.

    Each player has one latent skill value.
    The probability that player1 beats player2 depends on:

        skill[player1] - skill[player2]

    This is similar to the idea behind ELO / TrueSkill,
    but here we learn the skills using Pyro SVI.
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

    We approximate the posterior skill distribution with independent
    Normal distributions for all players.

    loc controls the estimated skill mean.
    scale controls the uncertainty for each player's skill.
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
    Convert probabilities into predictions.
    If prob >= 0.5, predict player1 wins.
    """

    preds = (probs >= 0.5).astype(int)
    return np.mean(preds == outcomes)


def log_loss_from_probs(probs, outcomes):
    """
    Compute log loss for probability predictions.
    I clip probabilities to avoid log(0).
    """

    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)

    losses = -(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return np.mean(losses)


def predict_with_skill_means(p1_ids, p2_ids):
    """
    Use the learned posterior mean of each skill to predict validation matches.
    """

    skill_loc = pyro.param("skill_loc").detach()

    skill_diff = skill_loc[p1_ids] - skill_loc[p2_ids]
    probs = torch.sigmoid(skill_diff)

    return probs.cpu().numpy()


# =========================
# Main experiment
# =========================

def main():
    pyro.clear_param_store()

    train_df, valid_df = load_matches()

    print("Original train shape:", train_df.shape)
    print("Original valid shape:", valid_df.shape)

    # Use a subset for the first run, so we can debug faster.
    if TRAIN_LIMIT is not None:
        train_df_used = train_df.iloc[:TRAIN_LIMIT].copy()
    else:
        train_df_used = train_df.copy()

    print("Training rows used:", len(train_df_used))

    player_to_id, id_to_player = build_player_mapping(train_df_used, valid_df)
    num_players = len(player_to_id)

    print("Number of players:", num_players)

    train_p1, train_p2, train_y = encode_matches(train_df_used, player_to_id)
    valid_p1, valid_p2, valid_y = encode_matches(valid_df, player_to_id)

    optimizer = Adam({"lr": LEARNING_RATE})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print("\nTraining Pyro SVI latent skill model...")

    for step in range(NUM_STEPS):
        loss = svi.step(train_p1, train_p2, train_y, num_players)

        if step % 100 == 0 or step == NUM_STEPS - 1:
            avg_loss = loss / len(train_y)
            print(f"Step {step:4d} | Average ELBO loss per match: {avg_loss:.4f}")

    print("\nEvaluating on validation set...")

    valid_probs = predict_with_skill_means(valid_p1, valid_p2)
    valid_outcomes = valid_y.cpu().numpy().astype(int)

    acc = accuracy_from_probs(valid_probs, valid_outcomes)
    loss = log_loss_from_probs(valid_probs, valid_outcomes)

    print("\nPyro SVI Latent Skill Model Results")
    print("-----------------------------------")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss: {loss:.4f}")

    # Save the model result so we can combine it with baseline_results.csv later.
    results = pd.DataFrame([
        {
            "model": "Pyro SVI Latent Skill Model",
            "accuracy": acc,
            "log_loss": loss,
            "train_rows_used": len(train_df_used),
            "num_steps": NUM_STEPS,
        }
    ])

    results.to_csv("pyro_results.csv", index=False)
    print("\nSaved results to pyro_results.csv")

    # Save learned skill means for possible discussion or debugging.
    skill_loc = pyro.param("skill_loc").detach().cpu().numpy()

    skill_df = pd.DataFrame({
        "player_id": list(range(num_players)),
        "player": [id_to_player[i] for i in range(num_players)],
        "estimated_skill_mean": skill_loc,
    })

    skill_df = skill_df.sort_values("estimated_skill_mean", ascending=False)
    skill_df.to_csv("learned_skills.csv", index=False)

    print("Saved learned skills to learned_skills.csv")
    print("\nTop 10 estimated players by skill:")
    print(skill_df.head(10))


if __name__ == "__main__":
    main()