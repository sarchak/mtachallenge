import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# 1. Setup and Configuration
st.set_page_config(page_title="MTA Express Bus Capacity Optimization", layout="wide")

st.title("MTA Express Bus Capacity Optimization")

# README section
st.markdown("""
## About This App

This Streamlit application is designed to analyze and optimize the capacity utilization of MTA Express Bus routes. Here's what you can do with this app:

1. **System Overview**: 
   - View a heatmap of hourly load percentages across all routes
   - Analyze borough-level capacity utilization during peak and off-peak hours

2. **Route Comparison**: 
   - Compare average load percentages across all routes
   - Identify overloaded and underutilized routes

3. **Peak Time Analysis**: 
   - Focus on defined peak hours to spot critical capacity issues
   - Visualize load patterns during busiest times

4. **Bus Reallocation Simulation**: 
   - Run an automated algorithm to optimize bus allocation
   - Set target load percentages and adjust maximum allowed changes
   - View the impact of reallocation on route capacities

5. **Resource Planning**: 
   - Calculate additional buses needed to meet target load percentages
   - Identify routes that still need attention after reallocation

Use the interactive elements below to explore different aspects of the MTA Express Bus system and simulate optimization strategies.
""")

# 2. Data Loading and Validation
try:
    df = pd.read_csv('mta_data.csv')
    required_columns = ['Week', 'Day Type', 'Borough', 'Route', 'Direction', 'Hour', 'Load Percentage', 'Trips with APC']
    
    if all(col in df.columns for col in required_columns):
        st.success("MTA data loaded and validated successfully!")
        
        # 3. Data Preprocessing
        df['Week'] = pd.to_datetime(df['Week'])
        df['Hour'] = df['Hour'].astype(int)
        
        # Calculate average load percentage for each Route and Hour combination
        avg_load = df.groupby(['Route', 'Hour'])['Load Percentage'].mean().reset_index()
        
        # Create pivot table
        pivot_table = avg_load.pivot(index='Route', columns='Hour', values='Load Percentage')
        
        # 4. Visualizations
        st.header("Hourly Load Percentage Heatmap")

        # Create three columns
        col1, col2, col3 = st.columns([1,3,1])

        # Use the middle column for the heatmap
        with col2:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_table, cmap="RdYlGn_r", ax=ax)
            plt.title("Average Load Percentage by Route and Hour")
            plt.xlabel("Hour")
            plt.ylabel("Route")
            st.pyplot(fig)
        
        # Directional Analysis at Borough Level
        st.header("Directional Analysis by Borough")

        # Group data by Borough, Direction, and Hour
        borough_direction_data = df.groupby(['Borough', 'Direction', 'Hour'])['Load Percentage'].mean().reset_index()

        # Create a dropdown to select borough
        selected_borough = st.selectbox("Select a Borough", df['Borough'].unique())

        # Filter data for the selected borough
        borough_data = borough_direction_data[borough_direction_data['Borough'] == selected_borough]

        # Create a line plot
        fig_line = px.line(borough_data, x='Hour', y='Load Percentage', color='Direction',
                           title=f"Average Load Percentage by Direction for {selected_borough}")
        fig_line.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Load Percentage (%)")
        st.plotly_chart(fig_line)

        # Create a box plot
        fig_box = px.box(borough_data, x='Direction', y='Load Percentage',
                         title=f"Distribution of Load Percentage by Direction for {selected_borough}")
        fig_box.update_layout(xaxis_title="Direction", yaxis_title="Load Percentage (%)")
        st.plotly_chart(fig_box)

        # Calculate and display summary statistics
        summary_stats = borough_data.groupby('Direction')['Load Percentage'].agg(['mean', 'median', 'min', 'max']).reset_index()
        summary_stats = summary_stats.round(2)
        st.subheader("Summary Statistics")
        st.table(summary_stats)

        # Identify peak hours for each direction
        peak_hours = borough_data.loc[borough_data.groupby('Direction')['Load Percentage'].idxmax()]
        st.subheader("Peak Hours by Direction")
        st.table(peak_hours[['Direction', 'Hour', 'Load Percentage']])

        # Calculate directional imbalance
        directional_avg = summary_stats.set_index('Direction')['mean']
        if 'NB' in directional_avg.index and 'SB' in directional_avg.index:
            imbalance = abs(directional_avg['NB'] - directional_avg['SB'])
            st.subheader("Directional Imbalance")
            st.write(f"The average load imbalance between northbound and southbound is {imbalance:.2f}%")
            
            if imbalance > 10:
                st.warning("There is a significant imbalance between northbound and southbound traffic. Consider adjusting resources.")
        elif 'EB' in directional_avg.index and 'WB' in directional_avg.index:
            imbalance = abs(directional_avg['EB'] - directional_avg['WB'])
            st.subheader("Directional Imbalance")
            st.write(f"The average load imbalance between eastbound and westbound is {imbalance:.2f}%")
            
            if imbalance > 10:
                st.warning("There is a significant imbalance between eastbound and westbound traffic. Consider adjusting resources.")
        
        # Borough-Level Analysis
        st.header("Borough-Level Analysis")
        df['Peak'] = df['Hour'].apply(lambda x: 'Peak' if x in [7, 8, 9, 16, 17, 18] else 'Off-Peak')
        borough_data = df.groupby(['Borough', 'Peak'])['Load Percentage'].mean().reset_index()
        fig = px.bar(borough_data, x='Borough', y='Load Percentage', color='Peak', barmode='group',
                     title="Average Load Percentage by Borough during Peak and Off-Peak Hours")
        st.plotly_chart(fig)
        
        # 5. Reallocation Logic
        if st.button("Optimize Bus Allocation"):
            st.header("Bus Allocation Optimization")
            
            # Identify overloaded and underutilized routes
            overloaded = avg_load[avg_load['Load Percentage'] > 80].groupby('Route').size()
            underutilized = avg_load[avg_load['Load Percentage'] < 20].groupby('Route').size()
            
            # Simple reallocation logic (for demonstration purposes)
            reallocation = pd.DataFrame({
                'Route': pd.concat([overloaded.index, underutilized.index]),
                'Original Load': pd.concat([overloaded, underutilized]),
                'Adjustment': pd.concat([pd.Series(1, index=overloaded.index), pd.Series(-1, index=underutilized.index)]),
            })
            reallocation['New Load'] = reallocation['Original Load'] + reallocation['Adjustment']
            
            st.subheader("Suggested Reallocation")
            st.dataframe(reallocation)
            
            # 6. Optimization Output
            st.subheader("Interactive Resource Adjustment")
            for idx, row in reallocation.iterrows():
                new_value = st.slider(f"Adjust buses for Route {row['Route']}", 
                                      min_value=0, max_value=int(row['Original Load'])*2, 
                                      value=int(row['New Load']))
                reallocation.loc[idx, 'User Adjusted'] = new_value
            
            reallocation['Impact'] = reallocation['User Adjusted'] - reallocation['Original Load']
            st.dataframe(reallocation)
            
            # Visualization of adjustments
            fig = px.bar(reallocation, x='Route', y=['Original Load', 'User Adjusted'], 
                         title="Impact of Bus Reallocation", barmode='group')
            st.plotly_chart(fig)
            
            # 7. Export and Reporting
            csv = reallocation.to_csv(index=False)
            st.download_button(
                label="Download Reallocation Report as CSV",
                data=csv,
                file_name="bus_reallocation_report.csv",
                mime="text/csv",
            )
    else:
        st.error("The MTA data file is missing required columns. Please check the file and try again.")
except Exception as e:
    st.error(f"An error occurred while processing the MTA data file: {str(e)}")

# 8. Future Integrations (placeholder)
st.sidebar.header("Future Integrations")
st.sidebar.info("Real-time data updates and seasonal adjustments coming soon!")

# New section for route comparison and optimization
st.header("Route Comparison and Optimization")

# Calculate average load percentage for each route
route_avg_load = df.groupby('Route')['Load Percentage'].mean().sort_values(ascending=False).reset_index()

# Create a bar chart for average load percentage by route
fig_route_comparison = px.bar(route_avg_load, x='Route', y='Load Percentage',
                              title="Average Load Percentage by Route",
                              labels={'Load Percentage': 'Average Load Percentage (%)'},
                              color='Load Percentage',
                              color_continuous_scale='RdYlGn_r')
fig_route_comparison.update_layout(xaxis_title="Route", yaxis_title="Average Load Percentage (%)")
st.plotly_chart(fig_route_comparison)

# Identify routes that need optimization
overloaded_routes = route_avg_load[route_avg_load['Load Percentage'] > 40]
underutilized_routes = route_avg_load[route_avg_load['Load Percentage'] < 20]

st.subheader("Routes Needing Optimization")
col1, col2 = st.columns(2)

with col1:
    st.write("Overloaded Routes (>40% average load):")
    st.dataframe(overloaded_routes)

with col2:
    st.write("Underutilized Routes (<20% average load):")
    st.dataframe(underutilized_routes)

# Hourly load heatmap for all routes
st.subheader("Hourly Load Percentage Heatmap for All Routes")
hourly_route_load = df.groupby(['Route', 'Hour'])['Load Percentage'].mean().reset_index()
hourly_route_pivot = hourly_route_load.pivot(index='Route', columns='Hour', values='Load Percentage')

fig_heatmap = go.Figure(data=go.Heatmap(
    z=hourly_route_pivot.values,
    x=hourly_route_pivot.columns,
    y=hourly_route_pivot.index,
    colorscale='RdYlGn_r',
    colorbar=dict(title='Load Percentage')
))
fig_heatmap.update_layout(
    title="Average Load Percentage by Route and Hour",
    xaxis_title="Hour of Day",
    yaxis_title="Route"
)
st.plotly_chart(fig_heatmap)

# New section for bus reallocation simulation
# st.header("Bus Reallocation Simulation")

# # Calculate initial bus allocation based on average load
# initial_allocation = route_avg_load.copy()
# initial_allocation['Buses'] = np.ceil(initial_allocation['Load Percentage'] / 100 * 10)  # Assume 10 buses per 100% load
# initial_allocation['Buses'] = initial_allocation['Buses'].astype(int)

# # Create sliders for each route
# st.subheader("Adjust Bus Allocation")
# adjusted_allocation = initial_allocation.copy()
# total_initial_buses = initial_allocation['Buses'].sum()

# for idx, row in initial_allocation.iterrows():
#     adjusted_buses = st.slider(f"Buses for {row['Route']} (Initial: {row['Buses']})", 
#                                min_value=0, 
#                                max_value=int(row['Buses']*2), 
#                                value=int(row['Buses']))
#     adjusted_allocation.loc[idx, 'Adjusted Buses'] = adjusted_buses

# # Calculate new load percentages
# adjusted_allocation['New Load Percentage'] = (adjusted_allocation['Load Percentage'] * 
#                                               adjusted_allocation['Buses'] / 
#                                               adjusted_allocation['Adjusted Buses'])

# Display results
# st.subheader("Reallocation Results")
# fig = go.Figure()
# fig.add_trace(go.Bar(x=adjusted_allocation['Route'], 
#                      y=adjusted_allocation['Load Percentage'],
#                      name='Original Load %'))
# fig.add_trace(go.Bar(x=adjusted_allocation['Route'], 
#                      y=adjusted_allocation['New Load Percentage'],
#                      name='New Load %'))
# fig.update_layout(title="Impact of Bus Reallocation",
#                   xaxis_title="Route",
#                   yaxis_title="Load Percentage",
#                   barmode='group')
# st.plotly_chart(fig)

# # Display allocation changes
# st.subheader("Allocation Changes")
# allocation_changes = adjusted_allocation[['Route', 'Buses', 'Adjusted Buses', 'Load Percentage', 'New Load Percentage']]
# allocation_changes['Bus Change'] = allocation_changes['Adjusted Buses'] - allocation_changes['Buses']
# allocation_changes['Load % Change'] = allocation_changes['New Load Percentage'] - allocation_changes['Load Percentage']
# st.dataframe(allocation_changes.style.format({
#     'Load Percentage': '{:.2f}%',
#     'New Load Percentage': '{:.2f}%',
#     'Load % Change': '{:.2f}%'
# }))

# # Summary statistics
# total_adjusted_buses = adjusted_allocation['Adjusted Buses'].sum()
# st.subheader("Summary")
# st.write(f"Total initial buses: {total_initial_buses}")
# st.write(f"Total adjusted buses: {total_adjusted_buses}")
# st.write(f"Change in total buses: {total_adjusted_buses - total_initial_buses}")

# # Identify routes still needing attention
# st.subheader("Routes Still Needing Attention")
# overloaded = adjusted_allocation[adjusted_allocation['New Load Percentage'] > 80]
# underutilized = adjusted_allocation[adjusted_allocation['New Load Percentage'] < 20]

# col1, col2 = st.columns(2)
# with col1:
#     st.write("Overloaded Routes (>80% new load):")
#     st.dataframe(overloaded[['Route', 'New Load Percentage']])
# with col2:
#     st.write("Underutilized Routes (<20% new load):")
#     st.dataframe(underutilized[['Route', 'New Load Percentage']])

st.header("Peak Time Route Optimization")

# Define peak hours (e.g., morning and evening rush hours)
peak_hours = [7, 8, 9, 16, 17, 18]

# Filter data for peak hours
peak_data = df[df['Hour'].isin(peak_hours)]

# Calculate average load percentage for each route during peak hours
peak_route_load = peak_data.groupby('Route')['Load Percentage'].mean().sort_values(ascending=False).reset_index()

# Visualize peak time route loads
fig_peak = px.bar(peak_route_load, x='Route', y='Load Percentage',
                  title="Average Load Percentage by Route during Peak Hours",
                  color='Load Percentage',
                  color_continuous_scale='RdYlGn_r')
st.plotly_chart(fig_peak)

# Identify overloaded and underutilized routes
overloaded_routes = peak_route_load[peak_route_load['Load Percentage'] > 50]
underutilized_routes = peak_route_load[peak_route_load['Load Percentage'] < 20]

st.subheader("Routes Needing Optimization")
col1, col2 = st.columns(2)
with col1:
    st.write("Overloaded Routes (>50% average load):")
    st.dataframe(overloaded_routes)
with col2:
    st.write("Underutilized Routes (<20% average load):")
    st.dataframe(underutilized_routes)

# Automated bus reallocation
st.subheader("Automated Bus Reallocation")

# Initial bus allocation (assuming 10 buses per 100% load)
peak_route_load['Initial Buses'] = np.ceil(peak_route_load['Load Percentage'] / 10).astype(int)
total_buses = peak_route_load['Initial Buses'].sum()

# Function to reallocate buses
def reallocate_buses(df, total_buses, target_load=70, max_adjustment=0.5):
    df = df.copy()
    df['Needed Buses'] = np.ceil(df['Load Percentage'] / target_load * df['Initial Buses']).astype(int)
    
    # Limit the maximum adjustment to a percentage of initial buses
    max_increase = df['Initial Buses'] * (1 + max_adjustment)
    max_decrease = df['Initial Buses'] * (1 - max_adjustment)
    df['Needed Buses'] = df['Needed Buses'].clip(max_decrease, max_increase).astype(int)
    
    while df['Needed Buses'].sum() > total_buses:
        # Remove buses from the least loaded route that still has more than its minimum
        removable = df[df['Needed Buses'] > max_decrease]
        if not removable.empty:
            idx = removable['Load Percentage'].idxmin()
            df.loc[idx, 'Needed Buses'] -= 1
        else:
            break
    
    while df['Needed Buses'].sum() < total_buses:
        # Add buses to the most loaded route that's still under its maximum
        addable = df[df['Needed Buses'] < max_increase]
        if not addable.empty:
            idx = addable['Load Percentage'].idxmax()
            df.loc[idx, 'Needed Buses'] += 1
        else:
            break
    
    df['New Load Percentage'] = df['Load Percentage'] * df['Initial Buses'] / df['Needed Buses']
    return df

# User inputs for reallocation parameters
target_load = st.slider("Target Load Percentage", min_value=30, max_value=90, value=70)
max_adjustment = st.slider("Maximum Bus Adjustment (%)", min_value=10, max_value=100, value=50) / 100

# Perform reallocation
reallocated = reallocate_buses(peak_route_load, total_buses, target_load, max_adjustment)

# Display reallocation results
st.subheader("Reallocation Results")
fig_realloc = go.Figure()
fig_realloc.add_trace(go.Bar(x=reallocated['Route'], y=reallocated['Load Percentage'], name='Original Load %'))
fig_realloc.add_trace(go.Bar(x=reallocated['Route'], y=reallocated['New Load Percentage'], name='New Load %'))
fig_realloc.update_layout(title="Impact of Bus Reallocation", barmode='group')
st.plotly_chart(fig_realloc)

# Display allocation changes
st.subheader("Allocation Changes")
allocation_changes = reallocated[['Route', 'Initial Buses', 'Needed Buses', 'Load Percentage', 'New Load Percentage']]
allocation_changes['Bus Change'] = allocation_changes['Needed Buses'] - allocation_changes['Initial Buses']
allocation_changes['Load % Change'] = allocation_changes['New Load Percentage'] - allocation_changes['Load Percentage']
allocation_changes['Relative Bus Change %'] = (allocation_changes['Bus Change'] / allocation_changes['Initial Buses'] * 100).round(2)

st.dataframe(allocation_changes.style.format({
    'Load Percentage': '{:.2f}%',
    'New Load Percentage': '{:.2f}%',
    'Load % Change': '{:.2f}%',
    'Relative Bus Change %': '{:.2f}%'
}))

# Highlight significant changes
significant_changes = allocation_changes[abs(allocation_changes['Relative Bus Change %']) > 20]
if not significant_changes.empty:
    st.subheader("Significant Changes (>20% bus change)")
    st.dataframe(significant_changes)

# Summary statistics
st.subheader("Summary")
st.write(f"Total buses: {total_buses}")
st.write(f"Buses reallocated: {abs(allocation_changes['Bus Change']).sum() // 2}")  # Divide by 2 as each move is counted twice
st.write(f"Average load before reallocation: {reallocated['Load Percentage'].mean():.2f}%")
st.write(f"Average load after reallocation: {reallocated['New Load Percentage'].mean():.2f}%")

# Identify routes still needing attention
st.subheader("Routes Still Needing Attention")
still_overloaded = reallocated[reallocated['New Load Percentage'] > 80]
still_underutilized = reallocated[reallocated['New Load Percentage'] < 20]

col1, col2 = st.columns(2)
with col1:
    st.write("Overloaded Routes (>80% new load):")
    st.dataframe(still_overloaded[['Route', 'New Load Percentage']])
with col2:
    st.write("Underutilized Routes (<20% new load):")
    st.dataframe(still_underutilized[['Route', 'New Load Percentage']])