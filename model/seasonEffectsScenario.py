import numpy as np
import pandas as pd
from scipy import stats
from model import VesselArrivalModel

class season_effects_model(VesselArrivalModel):
    """
    Class to simulate a scenario with seasonal effects on vessel arrivals.
    """
    
    # Initialize the model with both daily (day-of-week) and monthly seasonal multipliers
    def __init__(self):
        super().__init__()
        # seasonality multipliers based on day of the week using the dataset analysis
        inter_arrival_hours = [1.567, 1.557, 1.580, 1.635, 1.578, 1.570, 1.606]
        self.daily_multipliers = [(24 / hours) / 15.14 for hours in inter_arrival_hours]

        # monthly seasonality multipliers
        monthly_seasonal_factors = [1237, 1154, 1412, 1400, 1476, 1430, 1488, 1524, 1371, 1457, 1406, 1243]
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.monthly_multipliers = []
        for count, days in zip(monthly_seasonal_factors, days_in_month):
           monthly_daily_average = count / (days * 3)
           self.monthly_multipliers.append(monthly_daily_average / 15.14)

    # Simulate the seasonal effects scenario
    def simulate_seasonal_effects_scenario(self, days=365, num_simulation=10000):
        results = {
            "Total_vessels": np.zeros(num_simulation),
            "Total_cargo": np.zeros(num_simulation),
            "Daily_cargo": np.zeros(num_simulation*days),
            "Daily_vessels": np.zeros(num_simulation*days),
            "S_total_matrix": np.zeros((num_simulation, days))
        }

        annual_N_container = np.zeros(num_simulation)
        annual_N_tanker = np.zeros(num_simulation)
        annual_N_cargo = np.zeros(num_simulation)

        annual_S_container = np.zeros(num_simulation)
        annual_S_tanker = np.zeros(num_simulation)
        annual_S_cargo = np.zeros(num_simulation)

        dist_container = stats.lognorm(s=self.sig_container, scale=np.exp(self.mu_container))
        dist_tanker = stats.lognorm(s=self.sig_tanker, scale=np.exp(self.mu_tanker))
        dist_cargo = stats.gamma(a=self.alpha_b, scale=1/self.beta_b)

        #map each day to its corresponding month
        month_map = []
        for month_indx, days_in_month in enumerate([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]):
            month_map.extend([month_indx] * days_in_month)

        day_index = 0
        for i in range(num_simulation):
            cargo_container_total, cargo_tanker_total, cargo_cargo_total = 0,0,0
            vessels_container, vessels_tanker, vessels_cargo = 0, 0, 0

            for day in range(days):
                day_of_week = day % 7
                month_of_year = month_map[day]

                #combine both seasonality effects
                weekly_multiplier = self.daily_multipliers[day_of_week]
                monthly_multiplier = self.monthly_multipliers[month_of_year]
                combined_multiplier = weekly_multiplier * monthly_multiplier

                n_container = stats.poisson.rvs(mu=self.base_lam_container * combined_multiplier)
                n_tanker = stats.poisson.rvs(mu=self.base_lam_tanker * combined_multiplier) 
                n_cargo = stats.poisson.rvs(mu=self.base_lam_cargo * combined_multiplier)

                vessels_container += n_container
                vessels_tanker += n_tanker   
                vessels_cargo += n_cargo

                today_container = np.sum(dist_container.rvs(size=n_container)) if n_container > 0 else 0
                today_tanker = np.sum(dist_tanker.rvs(size=n_tanker)) if n_tanker > 0 else 0
                today_cargo = np.sum(dist_cargo.rvs(size=n_cargo)) if n_cargo > 0 else 0

                cargo_container_total += today_container
                cargo_tanker_total += today_tanker
                cargo_cargo_total += today_cargo
                
                results["Daily_cargo"][day_index] = today_container + today_tanker + today_cargo
                results["Daily_vessels"][day_index] = n_container + n_tanker + n_cargo
                results["S_total_matrix"][i, day] = today_container + today_tanker + today_cargo
                day_index += 1

            annual_N_container[i] = vessels_container
            annual_N_tanker[i] = vessels_tanker
            annual_N_cargo[i] = vessels_cargo
            results["Total_vessels"][i] = vessels_container + vessels_tanker + vessels_cargo

            annual_S_container[i] = cargo_container_total
            annual_S_tanker[i] = cargo_tanker_total
            annual_S_cargo[i] = cargo_cargo_total
            results["Total_cargo"][i] = cargo_container_total + cargo_tanker_total + cargo_cargo_total

        annual_arrays = [
            annual_N_container, annual_N_tanker, annual_N_cargo, results["Total_vessels"],
            annual_S_container, annual_S_tanker, annual_S_cargo, results["Total_cargo"]
        ]

        metrics = {
            "Variable": ["N_container", "N_tanker", "N_cargo", "N_total", "S_container", "S_tanker", "S_cargo", "S_total"],
            "Mean": [np.mean(x) for x in annual_arrays],
            "95% CI": [1.96 * np.std(x, ddof=1) / np.sqrt(num_simulation) for x in annual_arrays],
            "Std": [np.std(x, ddof=1) for x in annual_arrays],
            "95th Percentile": [np.percentile(x, 95) for x in annual_arrays]
        }
        results["metrics_df"] = pd.DataFrame(metrics)

        return results


def simulate_seasonal_effects_scenario():
    """
    Function to simulate a scenario with seasonal effects on vessel arrivals.
    """
    print("\n --- SEASONAL EFFECTS SCENARIO: WEEKLY AND MONTHLY VARIATIONS ---")
    model = season_effects_model()
    simulation_data = model.simulate_seasonal_effects_scenario(days=365, num_simulation=10000)

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

    print("\n--- SEASONAL EFFECTS SCENARIO ANNUAL RESULTS ---")
    print(simulation_data["metrics_df"].to_string(index=False))

    pd.options.display.float_format = '{:,.4f}'.format
    print("\n--- SEASONAL EFFECTS SCENARIO RESULTS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} ± {daily_cargo_ci:,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Congestion Probability (>650k tons): {congestion_probability:.4%} ± {cong_prob_ci:.4%} ({np.mean(cong_days_650):.2f} days/year)")
    print(f"Congestion Probability (>800k tons): {np.mean(cong_days_800 / 365):.4%} ({np.mean(cong_days_800):.2f} days/year)")

if __name__ == "__main__":
    simulate_seasonal_effects_scenario()