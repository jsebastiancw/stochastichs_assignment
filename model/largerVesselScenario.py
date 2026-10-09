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
    total_simulated_days = len(daily_cargo)

    congestion_probability = np.sum(daily_cargo > 650000) / total_simulated_days

    pd.options.display.float_format = '{:,.4f}'.format

    print("--- LARGER VESSEL SCENARIO RESULTS ---")
    print(f"Congestion Probability: {congestion_probability:.4%}")  
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Congestion Probability (>650k tons): {congestion_probability:.4%}")


if __name__ == "__main__":
    simulate_larger_vessel_scenario()