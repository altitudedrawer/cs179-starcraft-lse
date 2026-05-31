import pandas as pd
import numpy as np
import random
import math


# =========================
# Basic settings
# =========================

SEED = 179
random.seed(SEED)
np.random.seed(SEED)

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
# Load and clean data
# =========================

def load_matches(train_path="train.csv", valid_path="valid.csv"):
    """
    Load train and validation CSV files.
    The original CSV files do not have headers, so I manually add column names.
    """

    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    valid_df = pd.read_csv(valid_path, header=None, names=COLUMN_NAMES)

    return train_df, valid_df


def add_outcome_column(df):
    """
    Convert the winner/loser text into a simple numeric outcome.

    outcome = 1 means player1 wins.
    outcome = 0 means player1 loses, so player2 wins.
    """

    df = df.copy()
    df["outcome"] = (df["result1"] == "[winner]").astype(int)

    return df


# =========================
# Evaluation helpers
# =========================

def accuracy_from_probs(probs, outcomes):
    """
    Convert predicted probabilities into winner predictions.
    If prob >= 0.5, predict player1 wins.
    Otherwise, predict player2 wins.
    """

    preds = (probs >= 0.5).astype(int)
    accuracy = np.mean(preds == outcomes)

    return accuracy


def log_loss_from_probs(probs, outcomes):
    """
    Compute log loss for probabilistic predictions.

    I clip probabilities so that log(0) will not happen.
    Smaller log loss means better probability predictions.
    """

    eps = 1e-12
    probs = np.clip(probs, eps, 1 - eps)

    losses = -(outcomes * np.log(probs) + (1 - outcomes) * np.log(1 - probs))
    return np.mean(losses)


# =========================
# Baseline 1: Random baseline
# =========================

def random_baseline(valid_df):
    """
    Random baseline.

    This baseline gives each match a 0.5 probability that player1 wins.
    It is a simple lower-bound comparison.
    """

    probs = np.full(len(valid_df), 0.5)
    outcomes = valid_df["outcome"].values

    acc = accuracy_from_probs(probs, outcomes)
    loss = log_loss_from_probs(probs, outcomes)

    return acc, loss


# =========================
# Baseline 2: Historical win-rate baseline
# =========================

def compute_player_win_rates(train_df):
    """
    Compute each player's historical win rate from the training data.

    For every match:
    - player1 gets one game played
    - player2 gets one game played
    - the winner gets one win
    """

    wins = {}
    games = {}

    for _, row in train_df.iterrows():
        p1 = row["player1"]
        p2 = row["player2"]
        outcome = row["outcome"]

        # Make sure both players exist in the dictionaries.
        if p1 not in wins:
            wins[p1] = 0
            games[p1] = 0
        if p2 not in wins:
            wins[p2] = 0
            games[p2] = 0

        # Both players played one game.
        games[p1] += 1
        games[p2] += 1

        # outcome = 1 means player1 won; otherwise player2 won.
        if outcome == 1:
            wins[p1] += 1
        else:
            wins[p2] += 1

    win_rates = {}

    for player in games:
        win_rates[player] = wins[player] / games[player]

    return win_rates


def win_rate_baseline(valid_df, win_rates):
    """
    Historical win-rate baseline.

    For a validation match, compare the two players' training win rates.
    If player1 has a higher win rate, predict player1 is more likely to win.
    If player2 has a higher win rate, predict player1 is less likely to win.

    If a player did not appear in training, I use 0.5 as the default win rate.
    """

    probs = []

    for _, row in valid_df.iterrows():
        p1 = row["player1"]
        p2 = row["player2"]

        p1_rate = win_rates.get(p1, 0.5)
        p2_rate = win_rates.get(p2, 0.5)

        # Simple probability rule:
        # If player1 has a better historical win rate, give player1 a higher probability.
        # This is not a full probabilistic skill model, just a baseline.
        if p1_rate > p2_rate:
            prob_p1_wins = 0.75
        elif p1_rate < p2_rate:
            prob_p1_wins = 0.25
        else:
            prob_p1_wins = 0.5

        probs.append(prob_p1_wins)

    probs = np.array(probs)
    outcomes = valid_df["outcome"].values

    acc = accuracy_from_probs(probs, outcomes)
    loss = log_loss_from_probs(probs, outcomes)

    return acc, loss


# =========================
# Main function
# =========================

def main():
    train_df, valid_df = load_matches()

    train_df = add_outcome_column(train_df)
    valid_df = add_outcome_column(valid_df)

    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)

    print("\nRunning baselines...")

    random_acc, random_loss = random_baseline(valid_df)

    win_rates = compute_player_win_rates(train_df)
    win_rate_acc, win_rate_loss = win_rate_baseline(valid_df, win_rates)

    print("\nBaseline Results on Validation Set")
    print("----------------------------------")
    print(f"Random Baseline Accuracy:   {random_acc:.4f}")
    print(f"Random Baseline Log Loss:   {random_loss:.4f}")

    print(f"\nWin-rate Baseline Accuracy: {win_rate_acc:.4f}")
    print(f"Win-rate Baseline Log Loss: {win_rate_loss:.4f}")

    # Save results for later figures and report writing.
    results = pd.DataFrame([
        {
            "model": "Random Baseline",
            "accuracy": random_acc,
            "log_loss": random_loss,
        },
        {
            "model": "Historical Win-rate Baseline",
            "accuracy": win_rate_acc,
            "log_loss": win_rate_loss,
        },
    ])

    results.to_csv("baseline_results.csv", index=False)
    print("\nSaved results to baseline_results.csv")


if __name__ == "__main__":
    main()