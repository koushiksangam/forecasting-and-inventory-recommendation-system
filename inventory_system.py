import pandas as pd
from prophet import Prophet
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress Prophet's informational and future warnings to keep output clean.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# --- 1. Load and Peek at Data ---
print("--- Step 1: Load and Peek at Data ---")

# Attempt to load the consumption data; exit if file is missing.
try:
    bar_data = pd.read_csv('Consumption Dataset.xlsx - Dataset.csv')
    print("Consumption dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Consumption Dataset.xlsx - Dataset.csv' not found. Check file path.")
    exit()

# Display first few rows to understand structure.
print("\nFirst few rows of raw data:")
print(bar_data.head())

# Show column types and non-null counts for initial assessment.
print("\nData column types and non-null counts:")
bar_data.info()

# Confirm no missing values in initial load.
print("\nMissing values check (raw data):")
print(bar_data.isnull().sum())

# --- 2. Clean and Structure Data ---
print("\n--- Step 2: Clean and Structure Data ---")

# Standardize column names: lowercase, replace spaces with underscores, remove parentheses.
bar_data.columns = bar_data.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()

# Map original column names to logical names for inventory system.
bar_data = bar_data.rename(columns={
    'date_time_served': 'recorded_at',    # Event timestamp
    'bar_name': 'outlet_name',            # Location where item was served
    'brand_name': 'item_variant',         # Specific item (e.g., brand of rum)
    'consumed_ml': 'volume_consumed'      # Quantity consumed, in milliliters
})

# Convert timestamp column to datetime objects for time-series operations.
bar_data['recorded_at'] = pd.to_datetime(bar_data['recorded_at'])

# Sort data by time, then location, then item for consistent processing.
bar_data = bar_data.sort_values(by=['recorded_at', 'outlet_name', 'item_variant']).reset_index(drop=True)

# Aggregate consumption to daily totals per item per location.
# This ensures each unique combination of date, outlet, and item has one total consumption entry.
daily_item_consumption = bar_data.groupby(['recorded_at', 'outlet_name', 'item_variant'])['volume_consumed'].sum().reset_index()

# Display aggregated data's initial rows.
print("\nAggregated daily consumption data head:")
print(daily_item_consumption.head())

# Report unique outlets and items found.
print(f"\nUnique outlets found: {daily_item_consumption['outlet_name'].nunique()}")
print(f"Unique item variants found: {daily_item_consumption['item_variant'].nunique()}")

# --- 3. Forecast Future Demand ---
print("\n--- Step 3: Forecast Future Demand ---")

# Define period for demand prediction.
FORECAST_WEEKS = 2 
# Set a split point for training: historical data excluding the last few weeks.
TRAIN_DATA_END = daily_item_consumption['recorded_at'].max() - pd.Timedelta(weeks=FORECAST_WEEKS * 2) 

# Prepare a structure to hold all demand predictions.
all_demand_forecasts = pd.DataFrame()

# Identify all unique combinations of outlet and item to forecast individually.
unique_product_locations = daily_item_consumption[['outlet_name', 'item_variant']].drop_duplicates()

print(f"\nInitiating demand forecasts for {len(unique_product_locations)} specific product-location pairings...")

# Iterate through each unique product-location pair for forecasting.
for _, combo in unique_product_locations.iterrows():
    outlet = combo['outlet_name']
    item = combo['item_variant']

    # Filter consumption data for the current outlet and item.
    current_item_data = daily_item_consumption[(daily_item_consumption['outlet_name'] == outlet) & 
                                               (daily_item_consumption['item_variant'] == item)].copy()

    # Prophet model requires 'ds' (datestamp) and 'y' (value) columns.
    current_item_data = current_item_data[['recorded_at', 'volume_consumed']].rename(columns={
        'recorded_at': 'ds', 
        'volume_consumed': 'y'
    })

    # Skip if insufficient historical data for model training.
    if len(current_item_data) < 2:
        continue

    # Prepare training data up to the defined split point.
    training_set = current_item_data[current_item_data['ds'] <= TRAIN_DATA_END]

    # Skip if training set is too small.
    if len(training_set) < 2:
        continue

    # Initialize Prophet model.
    # Multiplicative seasonality fits demand where variations grow with baseline.
    demand_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False, # Daily seasonality often too noisy unless very specific events drive it.
        seasonality_mode='multiplicative'
    )
    demand_model.fit(training_set) # Train the model.

    # Create a future dataframe for prediction covering the forecast horizon.
    future_dates = demand_model.make_future_dataframe(periods=FORECAST_WEEKS * 7, freq='D')

    # Generate predictions.
    predicted_demand = demand_model.predict(future_dates)

    # Extract the relevant forecast period.
    forecast_start_date = daily_item_consumption['recorded_at'].max() + pd.Timedelta(days=1)
    forecast_end_date = forecast_start_date + pd.Timedelta(weeks=FORECAST_WEEKS) - pd.Timedelta(days=1)
    
    forecast_period_predictions = predicted_demand[(predicted_demand['ds'] >= forecast_start_date) & 
                                                   (predicted_demand['ds'] <= forecast_end_date)]

    # Sum daily predictions over the forecast horizon, ensuring non-negative results.
    total_forecasted_volume = max(0, forecast_period_predictions['yhat'].sum())

    # Add this forecast to the collective forecasts dataframe.
    all_demand_forecasts = pd.concat([all_demand_forecasts, pd.DataFrame([{
        'outlet_name': outlet,
        'item_variant': item,
        'predicted_volume': total_forecasted_volume,
        'prediction_start': forecast_start_date,
        'prediction_end': forecast_end_date
    }])], ignore_index=True)

print("\nSample of generated demand forecasts:")
print(all_demand_forecasts.head())
print(f"Total product-location forecasts completed: {len(all_demand_forecasts)}")


# --- 4. Determine Optimal Inventory Levels (Par Levels) ---
print("\n--- Step 4: Determine Optimal Inventory Levels (Par Levels) ---")

# Key assumption: 95% service level desired. This dictates the safety stock.
TARGET_SERVICE_LEVEL = 0.95  
# Z-score corresponding to the target service level (for a normal distribution).
Z_SCORE_FOR_SERVICE = norm.ppf(TARGET_SERVICE_LEVEL) 
# Duration of the replenishment cycle in days, influencing safety stock.
REPLENISH_CYCLE_DAYS = FORECAST_WEEKS * 7 

# Calculate historical average and standard deviation of daily consumption for each item.
item_historical_stats = daily_item_consumption.groupby(['outlet_name', 'item_variant'])['volume_consumed'].agg(['mean', 'std']).reset_index()
item_historical_stats.rename(columns={'mean': 'avg_daily_volume', 'std': 'std_daily_volume'}, inplace=True)

# Combine forecasts with historical consumption statistics.
inventory_levels = pd.merge(all_demand_forecasts, item_historical_stats, on=['outlet_name', 'item_variant'], how='left')

# Handle items with no historical variability (std_daily_volume would be NaN).
inventory_levels['std_daily_volume'] = inventory_levels['std_daily_volume'].fillna(0)

# Calculate safety stock: Z-score * std dev of demand during replenishment cycle.
# std dev during cycle = daily_std_dev * sqrt(cycle_days).
inventory_levels['safety_buffer_volume'] = Z_SCORE_FOR_SERVICE * inventory_levels['std_daily_volume'] * np.sqrt(REPLENISH_CYCLE_DAYS)

# Round up safety stock to ensure whole units.
inventory_levels['safety_buffer_volume'] = np.ceil(inventory_levels['safety_buffer_volume']).astype(int) 

# Calculate recommended par level: forecasted demand + safety stock.
inventory_levels['recommended_par_level'] = (
    inventory_levels['predicted_volume'] + inventory_levels['safety_buffer_volume']
)

# Ensure par levels are positive whole numbers.
inventory_levels['recommended_par_level'] = np.ceil(inventory_levels['recommended_par_level']).astype(int)
inventory_levels['recommended_par_level'] = inventory_levels['recommended_par_level'].apply(lambda x: max(0, x))

print("\nSample of calculated inventory recommendations (par levels):")
print(inventory_levels.head())


# --- 5. Simulate Inventory Performance ---
print("\n--- Step 5: Simulate Inventory Performance ---")

# Define simulation duration.
SIMULATION_DURATION_WEEKS = 8 
# Factor to set initial inventory at the start of simulation.
STARTING_STOCK_FACTOR = 1.0 

# Calculate average historical weekly demand to set initial stock for simulation.
historical_weekly_sum = daily_item_consumption.groupby(['outlet_name', 'item_variant', pd.Grouper(key='recorded_at', freq='W')])['volume_consumed'].sum().reset_index()
average_weekly_historical = historical_weekly_sum.groupby(['outlet_name', 'item_variant'])['volume_consumed'].mean().reset_index()
average_weekly_historical.rename(columns={'volume_consumed': 'avg_weekly_historical_demand'}, inplace=True)

# Merge historical averages with inventory recommendations to set initial stock levels.
simulation_state = pd.merge(inventory_levels, average_weekly_historical, on=['outlet_name', 'item_variant'], how='left')
simulation_state['current_stock_on_hand'] = (simulation_state['avg_weekly_historical_demand'].fillna(0) * STARTING_STOCK_FACTOR).astype(int)

# List to collect results from each simulated week.
weekly_performance_records = []

print(f"\nInitiating {SIMULATION_DURATION_WEEKS}-week inventory simulation...")

# Loop through each week of the simulation.
for current_week in range(1, SIMULATION_DURATION_WEEKS + 1):
    print(f"\n--- Simulating Week {current_week} ---")
    for idx, item_row in simulation_state.iterrows():
        # Extract relevant details for the current item-location.
        outlet = item_row['outlet_name']
        item = item_row['item_variant']
        par_level = item_row['recommended_par_level']
        stock_on_hand = item_row['current_stock_on_hand']
        avg_daily_vol = item_row['avg_daily_volume']
        std_daily_vol = item_row['std_daily_volume']

        # Simulate actual daily demand for 7 days, ensuring non-negative values.
        daily_simulated_demand = np.random.normal(avg_daily_vol, std_daily_vol, 7)
        daily_simulated_demand[daily_simulated_demand < 0] = 0 
        weekly_simulated_demand = int(np.ceil(daily_simulated_demand.sum()))

        # Determine actual consumption, stockouts, and lost sales.
        if stock_on_hand >= weekly_simulated_demand:
            actual_volume_sold = weekly_simulated_demand
            had_stockout = 0
            unmet_demand_volume = 0
        else:
            actual_volume_sold = stock_on_hand
            had_stockout = 1 
            unmet_demand_volume = weekly_simulated_demand - stock_on_hand

        # Calculate stock after fulfilling demand.
        stock_after_sales = stock_on_hand - actual_volume_sold

        # Simulate replenishment: bring stock up to par level (assuming immediate arrival for simplicity).
        # In a real system, orders would be placed earlier based on lead time.
        order_to_place = max(0, par_level - stock_after_sales)
        new_stock_level = stock_after_sales + order_to_place

        # Check for overstocking: if current stock is significantly above par.
        is_overstocked = 1 if new_stock_level > par_level * 1.1 else 0 # 10% buffer for overstock check.

        # Record this week's performance for the item.
        weekly_performance_records.append({
            'simulation_week': current_week,
            'outlet_name': outlet,
            'item_variant': item,
            'starting_stock': stock_on_hand,
            'simulated_weekly_demand': weekly_simulated_demand,
            'actual_volume_sold': actual_volume_sold,
            'stock_after_sales': stock_after_sales,
            'replenishment_order': order_to_place,
            'ending_stock': new_stock_level,
            'recommended_par_level': par_level,
            'had_stockout': had_stockout,
            'unmet_demand_volume': unmet_demand_volume,
            'is_overstocked': is_overstocked
        })

        # Update stock for the next simulation week.
        simulation_state.loc[idx, 'current_stock_on_hand'] = new_stock_level

# Convert the list of weekly records into a DataFrame.
simulation_results_summary = pd.DataFrame(weekly_performance_records)

print("\nSample of simulation results:")
print(simulation_results_summary.head())

# --- 6. Analyze Simulation Outcomes ---
print("\n--- Step 6: Analyze Simulation Outcomes ---")

# Sum up key metrics across the entire simulation.
total_stockout_events = simulation_results_summary['had_stockout'].sum()
total_lost_revenue_volume = simulation_results_summary['unmet_demand_volume'].sum()
total_overstock_events = simulation_results_summary['is_overstocked'].sum()

print(f"\nTotal stockout events across simulation: {total_stockout_events}")
print(f"Total volume of lost sales (units) during simulation: {total_lost_revenue_volume}")
print(f"Total overstocked events during simulation: {total_overstock_events}")

# Calculate the overall service level achieved.
total_potential_volume = simulation_results_summary['simulated_weekly_demand'].sum()
total_actual_volume_sold = simulation_results_summary['actual_volume_sold'].sum()

# Prevent division by zero if no demand occurred (unlikely).
final_service_level = (total_actual_volume_sold / total_potential_volume) * 100 if total_potential_volume > 0 else 0
print(f"Overall Service Level Achieved: {final_service_level:.2f}% (Target: {TARGET_SERVICE_LEVEL*100:.0f}%)")

# --- 7. Visualize Key Performance Indicators ---

# Plot weekly stockout occurrences.
plt.figure(figsize=(12, 5))
sns.lineplot(data=simulation_results_summary.groupby('simulation_week')['had_stockout'].sum().reset_index(), 
             x='simulation_week', y='had_stockout', marker='o')
plt.title('Weekly Stockout Events Over Simulation')
plt.xlabel('Simulation Week')
plt.ylabel('Number of Stockout Incidents')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot weekly lost sales volume.
plt.figure(figsize=(12, 5))
sns.lineplot(data=simulation_results_summary.groupby('simulation_week')['unmet_demand_volume'].sum().reset_index(), 
             x='simulation_week', y='unmet_demand_volume', marker='o')
plt.title('Weekly Lost Sales Volume Over Simulation')
plt.xlabel('Simulation Week')
plt.ylabel('Lost Sales (Volume in ml)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot total inventory level over time.
plt.figure(figsize=(12, 5))
sns.lineplot(data=simulation_results_summary.groupby('simulation_week')['ending_stock'].sum().reset_index(), 
             x='simulation_week', y='ending_stock', marker='o')
plt.title('Total Inventory Volume Across All Locations Over Simulation')
plt.xlabel('Simulation Week')
plt.ylabel('Total Inventory (Volume in ml)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()