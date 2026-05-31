import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Load result files
# =========================
# This file combines the baseline results and the Pyro SVI result
# into one figure with two side-by-side subplots.

baseline_df = pd.read_csv("baseline_results.csv")
pyro_df = pd.read_csv("pyro_results.csv")

# Keep only the columns needed for plotting.
baseline_df = baseline_df[["model", "accuracy", "log_loss"]]
pyro_df = pyro_df[["model", "accuracy", "log_loss"]]

results_df = pd.concat([baseline_df, pyro_df], ignore_index=True)

# Use short names to make the figure cleaner.
results_df["short_label"] = ["Random", "Win-rate", "Pyro SVI"]

# Save the combined table for later report writing if needed.
results_df.to_csv("combined_results.csv", index=False)

print("Combined results:")
print(results_df)


# =========================
# Create output folder
# =========================

os.makedirs("figures", exist_ok=True)


# =========================
# Helper function
# =========================

def add_value_labels(ax, bars, values, offset):
    """
    Add numeric labels above each bar.
    This makes the figure easier to read in the final PDF.
    """
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.text(x, y + offset, f"{value:.4f}", ha="center", va="bottom", fontsize=9)


# =========================
# Make one figure with two side-by-side panels
# =========================

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

# Green color for this version.
bar_color = "forestgreen"


# -------------------------
# Panel (a): Accuracy
# -------------------------
ax1 = axes[0]

bars1 = ax1.bar(
    results_df["short_label"],
    results_df["accuracy"],
    width=0.50,
    color=bar_color
)

ax1.set_ylabel("Validation Accuracy")
ax1.set_xlabel("Model")
ax1.set_title("(a) Accuracy", fontsize=11)
ax1.set_ylim(0.45, 0.71)

add_value_labels(ax1, bars1, results_df["accuracy"], offset=0.003)


# -------------------------
# Panel (b): Log Loss
# -------------------------
ax2 = axes[1]

bars2 = ax2.bar(
    results_df["short_label"],
    results_df["log_loss"],
    width=0.50,
    color=bar_color
)

ax2.set_ylabel("Validation Log Loss")
ax2.set_xlabel("Model")
ax2.set_title("(b) Log Loss", fontsize=11)
ax2.set_ylim(0.55, 0.71)

add_value_labels(ax2, bars2, results_df["log_loss"], offset=0.0015)


# =========================
# Final layout and save
# =========================
# I do not put a big title inside the figure,
# because the report caption will already explain it.

plt.tight_layout()
plt.savefig("figures/figure1_model_comparison_green.png", dpi=300)
plt.show()

print("\nSaved figure:")
print("figures/figure1_model_comparison_green.png")
print("Saved combined table:")
print("combined_results.csv")