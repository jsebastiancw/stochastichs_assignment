import numpy as np
import pandas as pd
from scipy import stats
from model import VesselArrivalModel


class larger_vessel_model(VesselArrivalModel):
    """
    Class to simulate a scenario with larger vessels arriving at the terminal.
    """
    def __init__(self, capacity_increase=1.2):
        super().__init__()
        self.base_lam_tanker *= capacity_increase
        self.base_lam_cargo *= capacity_increase

def simulate_larger_vessel_scenario():
    """
    Function to simulate a scenario with larger vessels arriving at the terminal.
    """
    
    print("\n --- LARGER VESSEL SCENARIO: 20% increase in LARGER VESSELS ---")
    model = larger_vessel_model(capacity_increase=1.2)
    simulation_data = model.simulate_scenario(days=365, num_simulation=10000)

    daily_cargo = simulation_data["Daily_cargo"]
    daily_vessels = simulation_data["Daily_vessels"]
    s_matrix = simulation_data["S_total_matrix"]
    num_runs = s_matrix.shape[0]
    total_simulated_days = len(daily_cargo)
    congestion_probability = np.sum(daily_cargo > 650000) / total_simulated_days

    daily_cargo_ci = 1.96 * np.std(np.mean(s_matrix, axis=1), ddof=1) / np.sqrt(num_runs)
    cong_days_650 = np.sum(s_matrix > 650000, axis=1)
    cong_prob_ci = 1.96 * np.std(cong_days_650 / 365, ddof=1) / np.sqrt(num_runs)
    cong_days_800 = np.sum(s_matrix > 800000, axis=1)

    print("\n--- LARGER VESSEL SCENARIO ANNUAL RESULTS ---")
    print(simulation_data["metrics_df"].to_string(index=False))

    pd.options.display.float_format = '{:,.4f}'.format
    print("\n--- LARGER VESSEL SCENARIO RESULTS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} ± {daily_cargo_ci:,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Congestion Probability (>650k tons): {congestion_probability:.4%} ± {cong_prob_ci:.4%} ({np.mean(cong_days_650):.2f} days/year)")
    print(f"Congestion Probability (>800k tons): {np.mean(cong_days_800 / 365):.4%} ({np.mean(cong_days_800):.2f} days/year)")

if __name__ == "__main__":
    simulate_larger_vessel_scenario()