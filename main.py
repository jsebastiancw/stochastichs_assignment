#script to run the models and the scenarios. Take into account that since there is no initial seed,
# the results will be slightly different each time the script is run.
import numpy as np
import pandas as pd

from model.base import BasicScenarioModel
from model.model import VesselArrivalModel
from model.newTerminalScenario import new_terminal_scenario
from model.largerVesselScenario import simulate_larger_vessel_scenario
from model.seasonEffectsScenario import simulate_seasonal_effects_scenario


def run_base_model():
    model = BasicScenarioModel()
    simulation_data = model.simulate_scenario(days=365, num_simulation=10000)

    print("----- BASE MODEL RESULTS -----")
    print(simulation_data["metrics_df"].to_string(index=False))

    daily_cargo = simulation_data["Daily_cargo"]
    daily_vessels = simulation_data["Daily_vessels"]
    s_matrix = simulation_data["S_total_matrix"]
    num_runs = s_matrix.shape[0]

    daily_cargo_ci = 1.96 * np.std(np.mean(s_matrix, axis=1), ddof=1) / np.sqrt(num_runs)

    cong_days_650 = np.sum(s_matrix > 650000, axis=1)
    cong_prob_650 = np.mean(cong_days_650 / 365)
    cong_prob_650_ci = 1.96 * np.std(cong_days_650 / 365, ddof=1) / np.sqrt(num_runs)

    cong_days_800 = np.sum(s_matrix > 800000, axis=1)
    cong_prob_800 = np.mean(cong_days_800 / 365)

    #code copied from the basic scenario file
    print("\n--- BASE MODEL DAILY & CONGESTION METRICS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.2f} ± {daily_cargo_ci:,.2f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.2f} tons")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.2f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.2f} tons")
    print(f"Congestion Probability (>650k tons): {cong_prob_650:.4%} ± {cong_prob_650_ci:.4%} ({np.mean(cong_days_650):.2f} days/year)")
    print(f"Congestion Probability (>800k tons): {cong_prob_800:.4%} ({np.mean(cong_days_800):.2f} days/year)\n")


def run_our_model():
    model = VesselArrivalModel()
    metrics = model.simulate_scenario(days=365, num_simulation=10000)

    print("--- OUR MODEL ANNUAL RESULTS ---")
    print(metrics["metrics_df"].to_string(index=False))

    daily_cargo = metrics["Daily_cargo"]
    daily_vessels = metrics["Daily_vessels"]
    s_matrix = metrics["S_total_matrix"]
    num_runs = s_matrix.shape[0]

    daily_cargo_ci = 1.96 * np.std(np.mean(s_matrix, axis=1), ddof=1) / np.sqrt(num_runs)
    cong_days_650 = np.sum(s_matrix > 650000, axis=1)
    cong_prob = np.mean(cong_days_650 / 365)
    cong_prob_ci = 1.96 * np.std(cong_days_650 / 365, ddof=1) / np.sqrt(num_runs)
    cong_days_800 = np.sum(s_matrix > 800000, axis=1)

    #code copied from the seasonal effects scenario file
    print("\n--- OUR MODEL DAILY & CONGESTION RESULTS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.2f} ± {daily_cargo_ci:,.2f}")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.2f}")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.2f}")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.2f}")
    print(f"Congestion Probability (>650k tons): {cong_prob:.4%} ± {cong_prob_ci:.4%} ({np.mean(cong_days_650):.2f} days/year)")
    print(f"Congestion Probability (>800k tons): {np.mean(cong_days_800 / 365):.4%} ({np.mean(cong_days_800):.2f} days/year)\n")


if __name__ == "__main__":
    # 1. Run the Basic Scenario (All Lognormal)
    run_base_model()

    # 2. Run Our Proposed Model (Gamma Bulk Fit)
    run_our_model()

    # 3. Run Scenario 1: New Terminal Expansion
    new_terminal_scenario()

    # 4. Run Scenario 2: Larger Vessels (+20% Tanker & Bulk)
    simulate_larger_vessel_scenario()

    # 5. Run Scenario 3: Seasonal Effects (Weekly & Monthly)
    simulate_seasonal_effects_scenario()