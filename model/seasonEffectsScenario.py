import numpy as np
import pandas as pd
from scipy import stats
from model import VesselArrivalModel

class season_effects_model(VesselArrivalModel):

    """
    Class to simulate a scenario with seasonal effects on vessel arrivals.
    """
    
    # Initialize the model with seasonal multipliers
    def __init__(self):
        super().__init__()
        monthly_seasonal_factors = [1237, 1154, 1412, 1400, 1476, 1430, 1488, 1524, 1371, 1457, 1406, 1243]
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        self.monthly_multipliers = []
        for count, days in zip(monthly_seasonal_factors, days_in_month):
           monthly_daily_average = count / (days * 3)
           self.monthly_multipliers.append(monthly_daily_average)

    # Simulate the seasonal effects scenario
    def simulate_seasonal_effects_scenario(self, days=365, num_simulation=10000):

        results = {
            "Total_vessels": np.zeros(num_simulation),
            "Total_cargo": np.zeros(num_simulation),
            "Daily_cargo": np.zeros(num_simulation*days),
            "Daily_vessels": np.zeros(num_simulation*days)

        }

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

                n_container = stats.poisson(mu=self.base_lam_container * combined_multiplier).rvs()
                n_tanker = stats.poisson(mu=self.base_lam_tanker * combined_multiplier).rvs() 
                n_cargo = stats.poisson(mu=self.base_lam_cargo * combined_multiplier).rvs()


                vessels_container += n_container
                vessels_tanker += n_tanker   
                vessels_cargo += n_cargo

                today_container = np.sum(dist_container.rvs(size=n_container)) if n_container > 0 else 0
                today_tanker = np.sum(dist_tanker.rvs(size=n_tanker)) if n_tanker > 0 else 0
                today_cargo = np.sum(dist_cargo.rvs(size=n_cargo)) if n_cargo > 0 else 0

                cargo_container_total += today_container
                cargo_tanker_total += today_tanker
                cargo_cargo_total += today_cargo
                
                # FIX BUG 1: Record daily values INSIDE the day loop, using day_index
                results["Daily_cargo"][day_index] = today_container + today_tanker + today_cargo
                results["Daily_vessels"][day_index] = n_container + n_tanker + n_cargo
                day_index += 1

            # FIX BUG 3: Record annual simulation totals
            results["Total_vessels"][i] = vessels_container + vessels_tanker + vessels_cargo
            results["Total_cargo"][i] = cargo_container_total + cargo_tanker_total + cargo_cargo_total

        return results

def simulate_seasonal_effects_scenario():

    """
    Function to simulate a scenario with seasonal effects on vessel arrivals.
    """

    print("\n --- SEASONAL EFFECTS SCENARIO: WEEKLY AND MONTHLY VARIATIONS ---")
    model = season_effects_model()
    simulation_data = model.simulate_seasonal_effects_scenario(days=365, num_simulation=10000)

    daily_cargo = simulation_data["Daily_cargo"]
    total_simulated_days = len(daily_cargo)

    congestion_probability = np.sum(daily_cargo > 650000) / total_simulated_days

    pd.options.display.float_format = '{:,.4f}'.format

    print("--- SEASONAL EFFECTS SCENARIO RESULTS ---")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.0f} tons")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.0f} tons")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.0f} tons")
    print("-" * 50)
    print(f"Congestion Probability (>650k tons): {congestion_probability:.4%}")


if __name__ == "__main__":
    simulate_seasonal_effects_scenario()
