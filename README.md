# stochastichs_assignment

# Port of Rotterdam - Stochastic Simulation Model

This repository contains the Python codebase for the stochastic modeling and simulation of daily cargo throughput at the Port of Rotterdam. The project utilizes a compound Poisson process and a Monte Carlo simulation engine to evaluate baseline operations and assess congestion risks under various operational conditions.

## Project Structure

* **`Data/`**: Contains the historical operational records (`rotterdam_dataset_37.xlsx`) used to calibrate the models.


* **`figures/`**: Directory where the generated plots from the Exploratory Data Analysis (EDA) are saved.


* **`Analysis_volume.py` & `datanalysis.py**`: Scripts for Exploratory Data Analysis, outlier detection, and goodness-of-fit testing (Lognormal, Gamma, Weibull) to determine the optimal probability distributions for cargo volumes.


* **`main.py`**: The master execution script that runs the baseline model and all three operational scenarios sequentially, outputting the standardized evaluation metrics.
* **`model/`**: The core directory containing the simulation engine and scenario configurations:


* `base.py`: The Basic Scenario utilizing a homogeneous Poisson process and Lognormal distributions for all vessel types.


* `model.py`: The Optimized Model acting as the statistical control, replacing the Bulk Lognormal fit with a Method of Moments Gamma fit.


* `newTerminalScenario.py`: **Scenario 1** (New Terminal Expansion) evaluating an increased capacity threshold of 800,000 tons/day.


* `largerVesselScenario.py`: **Scenario 2** (Fleet Upscaling) increasing the arrival rates of Tanker and Bulk vessels by 20%.


* `seasonEffectsScenario.py`: **Scenario 3** (Time-Varying Seasonality) introducing non-homogeneous daily and monthly arrival multipliers.





## Prerequisites

The simulation engine relies on standard data science and statistical libraries. Ensure you have Python 3.x installed along with the following packages:

```bash
pip install numpy pandas scipy matplotlib seaborn openpyxl

```

## How to Run

### 1. Run the Full Simulation Suite

To run the baseline models and all three scenarios (producing the exact metrics used in the report), execute the master script from the project root:

```bash
python main.py

```

This will sequentially output the Annual Cumulative Metrics ($T=365$), Daily Throughput Breakdown ($t=1$), and Congestion Probabilities for each configuration.

### 2. Run Individual Scenarios

If you wish to run a specific model or scenario in isolation, you can execute its script directly from the `model/` directory:

```bash
python model/largerVesselScenario.py

```

### 3. Run the Exploratory Data Analysis

To regenerate the histograms, Q-Q plots, tail plots, and goodness-of-fit metrics (AIC/BIC/KS-Stats) used to justify the distribution choices, run the analysis script from the project root:

```bash
python Analysis_volume.py

```

Note: The generated plots will automatically be saved into the `figures/` folder.