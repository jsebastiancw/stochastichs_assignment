import numpy as np
import pandas as pd
from scipy import stats
from model.model import VesselArrivalModel


def new_terminal_scenario():
    """
    Function to simulate a new terminal scenario with increased capacity.
    """
    #Setup the model and run the simulation
    model = VesselArrivalModel()
    simulation_data = model.simulate_scenario(days=365, num_simulation=10000)

    # Get the daily cargo and vessel data
    daily_cargo = simulation_data["Daily_cargo"]
    daily_vessels = simulation_data["Daily_vessels"]
    s_matrix = simulation_data["S_total_matrix"]
    num_runs = s_matrix.shape[0]
    total_simulated_days = len(daily_cargo)

    # Defining the capacities:
    old_capacity = 650000
    new_capacity = 800000

    # Calculate the congestions events and the probabilities for both scenarios:
    congestion_events_old = np.sum(daily_cargo > old_capacity)
    congestion_events_new =  np.sum(daily_cargo > new_capacity)
    old_congestion_prob = congestion_events_old / total_simulated_days
    new_congestion_prob = congestion_events_new / total_simulated_days

    # Confidence intervals and expected days/year across runs
    daily_cargo_ci = 1.96 * np.std(np.mean(s_matrix, axis=1), ddof=1) / np.sqrt(num_runs)
    cong_days_old = np.sum(s_matrix > old_capacity, axis=1)
    cong_days_new = np.sum(s_matrix > new_capacity, axis=1)
    old_prob_ci = 1.96 * np.std(cong_days_old / 365, ddof=1) / np.sqrt(num_runs)
    new_prob_ci = 1.96 * np.std(cong_days_new / 365, ddof=1) / np.sqrt(num_runs)

    print("\n--- NEW TERMINAL SCENARIO ANNUAL RESULTS ---")
    print(simulation_data["metrics_df"].to_string(index=False))

    pd.options.display.float_format = '{:,.4f}'.format
    print("\n--- NEW TERMINAL SCENARIO RESULTS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} ± {daily_cargo_ci:,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Baseline Congestion Probability (>650k tons): {old_congestion_prob:.4%} ± {old_prob_ci:.4%} ({np.mean(cong_days_old):.2f} days/year)")
    print(f"New Terminal Congestion Probability (>800k tons): {new_congestion_prob:.4%} ± {new_prob_ci:.4%} ({np.mean(cong_days_new):.2f} days/year)")
    print(f"Relative Congestion Reduction: {((old_congestion_prob - new_congestion_prob) / old_congestion_prob):.2%}")

if __name__ == "__main__":
    new_terminal_scenario()