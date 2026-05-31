# cs179-starcraft-lse

CS 179 Project Code: Latent Skill Estimation from StarCraft Match Outcomes Using Pyro SVI

Main environment:
Python 3.11
Required packages: numpy, pandas, matplotlib, torch, pyro-ppl

To install packages:
pip install -r requirements.txt

Run order:

1. python 01_baseline.py
   Output:
   baseline_results.csv

2. python 02_pyro_skill_model.py
   Output:
   pyro_results.csv
   learned_skills.csv

3. python 03_training_size_experiment.py
   Output:
   training_size_results.csv
   figures/figure2_training_size_effect.png

4. python 04_plot_results_figure1.py
   Output:
   combined_results.csv
   figures/figure1_model_comparison_green_largefont.png

5. python 05_posterior_uncertainty_plot.py
   Output:
   figure3_uncertainty_comparison.png

Notes:
- train.csv and valid.csv should be in the same folder as the scripts.
- SVI steps = 2000.
- Learning rate = 0.03.
- Random seed = 179.
- load_data.py is included as provided/reference code. The main scripts directly load the CSV files and build the player ID mapping.
