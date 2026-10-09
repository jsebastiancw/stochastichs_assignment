import numpy as np
import pandas as pd
from scipy import stats
from model import VesselArrivalModel


def new_terminal_scenario():

    """
    Function to simulate a new terminal scenario with increased capacity.

    """

    #Setup the model and run the simulation
    model = VesselArrivalModel()
    simulation_data = model.simulate_scenario(days=365, num_simulation=10000)

    # Get the daily cargo and vessel data
    daily_cargo = simulation_data["Daily_cargo"]
    total_simulated_days = len(daily_cargo)

    # Defining the capacities:
    old_capacity = 650000
    new_capacity = 800000

    # Calculate the congestions events and the probabilities for both scenarios:

    congestion_events_old = np.sum(daily_cargo > old_capacity)
    congestion_events_new =  np.sum(daily_cargo > new_capacity)

    old_congestion_prob = congestion_events_old / total_simulated_days
    new_congestion_prob = congestion_events_new / total_simulated_days

    pd.options.display.float_format = '{:,.4f}'.format

    print("\n--- NEW TERMINAL SCENARIO RESULTS ---")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Baseline Congestion Probability (>650k tons): {old_congestion_prob:.4%}")
    print(f"New Terminal Congestion Probability (>800k tons): {new_congestion_prob:.4%}")
    print(f"Relative Congestion Reduction: {((old_congestion_prob - new_congestion_prob) / old_congestion_prob):.2%}")

if __name__ == "__main__":
    new_terminal_scenario()
