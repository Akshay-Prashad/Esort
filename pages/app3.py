import streamlit as st
from Eco_model.data_loader import generate_market_data, get_base_recycling_costs
from Eco_model.models import EconomicModel, RecyclingCalculator
from Eco_model.visualization import Visualizer
import pandas as pd
# Configure page
st.set_page_config(page_title="E-Waste Economic Model", layout="wide")
st.title("Economic Descriptive Modeling Module")
st.markdown("""
**Objective**: Determine whether identified components should be resold or recycled based on market trends, 
component conditions, and estimated profitability.
""")

# Initialize models and load data
economic_model = EconomicModel()
recycling_calc = RecyclingCalculator()
market_df = generate_market_data()

# --- Market Data Section ---
st.header("1. Market Data Integration")
with st.expander("View Raw Market Data"):
    st.dataframe(market_df)

# Show price trends
st.subheader("Historical Price Trends")
selected_component = st.selectbox("Select Component", market_df['Component'].unique())
trend_fig = Visualizer.plot_price_trends(market_df, selected_component)
if trend_fig:
    st.plotly_chart(trend_fig, use_container_width=True)

# --- Resale Value Prediction ---
st.header("2. Resale Value Prediction")
col1, col2 = st.columns(2)

with col1:
    component_type = st.selectbox("Component Type", 
                                ['GPU', 'CPU', 'RAM', 'Motherboard', 'HDD', 'SSD'])
    brand = st.selectbox("Brand", ['NVIDIA', 'AMD', 'Intel', 'Samsung', 'Western Digital', 'ASUS'])
    age = st.slider("Age (years)", 0, 10, 2)

with col2:
    condition = st.select_slider("Condition", 
                               options=['Poor', 'Fair', 'Good', 'Excellent'])
    capacity = st.number_input("Capacity (GB)", min_value=1, max_value=2000, value=256)
    demand_level = st.select_slider("Market Demand", 
                                  ['Low', 'Medium', 'High'])

if st.button("Estimate Resale Value"):
    estimated_value = economic_model.predict_resale_value(
        component_type, condition, demand_level, age, capacity
    )
    st.session_state['estimated_value'] = estimated_value
    st.success(f"Estimated Resale Value: ${estimated_value:,.2f}")

# --- Recycling Cost Estimation ---
st.header("3. Recycling Cost Estimation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Recycling Costs")
    cost_df = pd.DataFrame.from_dict(get_base_recycling_costs(), orient='index', columns=['Cost ($)'])
    st.dataframe(cost_df.style.format("${:.2f}"))

with col2:
    st.subheader("Additional Cost Factors")
    labor_cost = st.number_input("Labor Cost ($/unit)", min_value=0.0, value=5.0)
    energy_cost = st.number_input("Energy Cost ($/unit)", min_value=0.0, value=2.0)
    hazardous = st.checkbox("Contains hazardous materials (+$3.00)")

if st.button("Estimate Recycling Cost"):
    total_recycling_cost = recycling_calc.calculate_cost(
        component_type, labor_cost, energy_cost, hazardous
    )
    st.session_state['total_recycling_cost'] = total_recycling_cost
    st.info(f"Total Estimated Recycling Cost: ${total_recycling_cost:,.2f}")

# --- Decision Support ---
st.header("4. Comparative Analysis & Decision Support")
if 'estimated_value' in st.session_state and 'total_recycling_cost' in st.session_state:
    estimated_value = st.session_state['estimated_value']
    total_recycling_cost = st.session_state['total_recycling_cost']
    profit_margin = estimated_value - total_recycling_cost
    decision = "Resell" if profit_margin > 0 else "Recycle"
    color = "green" if profit_margin > 0 else "red"
    
    st.markdown(f"### Recommendation: :{color}[{decision}]")
    st.markdown(f"- Resale Value: ${estimated_value:,.2f}")
    st.markdown(f"- Recycling Cost: ${total_recycling_cost:,.2f}")
    st.markdown(f"- Profit Margin: ${profit_margin:,.2f}")
    
    st.subheader("Component Decision Table")
    st.table(Visualizer.show_decision_table(
        component_type, estimated_value, total_recycling_cost, profit_margin, decision
    ))
    
    st.plotly_chart(
        Visualizer.plot_comparison(estimated_value, total_recycling_cost),
        use_container_width=True
    )
else:
    st.warning("Please estimate both resale value and recycling costs first")

