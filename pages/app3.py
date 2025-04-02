import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# Configure page
st.set_page_config(page_title="E-Waste Economic Model", layout="wide")
st.title("Economic Descriptive Modeling Module")
st.markdown("""
**Objective**: Determine whether identified components should be resold or recycled based on market trends, 
component conditions, and estimated profitability.
""")

# --- Data Section ---
st.header("1. Market Data Integration")

@st.cache_data
def load_market_data():
    """Simulate loading real-time market data"""
    components = ['GPU', 'CPU', 'RAM', 'Motherboard', 'HDD', 'SSD']
    brands = ['NVIDIA', 'AMD', 'Intel', 'Samsung', 'Western Digital', 'ASUS']
    
    # Generate synthetic market data
    data = []
    for _ in range(100):
        component = np.random.choice(components)
        brand = np.random.choice(brands)
        age = np.random.randint(0, 5)
        condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'])
        price = np.random.uniform(10, 500)
        date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        data.append({
            'Component': component,
            'Brand': brand,
            'Age (years)': age,
            'Condition': condition,
            'Price ($)': round(price, 2),
            'Date': date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(data)

market_df = load_market_data()

# Show raw data
with st.expander("View Raw Market Data"):
    st.dataframe(market_df)

# Show price trends
st.subheader("Historical Price Trends")
selected_component = st.selectbox("Select Component", market_df['Component'].unique())
component_df = market_df[market_df['Component'] == selected_component]

if not component_df.empty:
    fig = px.line(component_df, x='Date', y='Price ($)', 
                 title=f"Price Trend for {selected_component}",
                 hover_data=['Brand', 'Condition'])
    st.plotly_chart(fig, use_container_width=True)

# --- Resale Value Prediction ---
st.header("2. Resale Value Prediction")

# Simulate trained Random Forest model
@st.cache_resource
def train_model():
    # In a real app, this would be pre-trained with actual data
    model = RandomForestRegressor(n_estimators=100)
    return model

model = train_model()

# Prediction interface
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

# Simulate prediction
if st.button("Estimate Resale Value"):
    # In a real app, this would use the actual model
    base_values = {
        'GPU': 200, 'CPU': 100, 'RAM': 50, 
        'Motherboard': 80, 'HDD': 30, 'SSD': 60
    }
    condition_multiplier = {
        'Poor': 0.3, 'Fair': 0.6, 'Good': 0.85, 'Excellent': 1.0
    }
    demand_multiplier = {
        'Low': 0.7, 'Medium': 1.0, 'High': 1.3
    }
    age_penalty = 0.9 ** age
    
    estimated_value = (base_values[component_type] * 
                      condition_multiplier[condition] * 
                      demand_multiplier[demand_level] * 
                      age_penalty * 
                      (1 + np.log(capacity/256)))
    
    st.success(f"Estimated Resale Value: ${estimated_value:,.2f}")

# --- Recycling Cost Estimation ---
st.header("3. Recycling Cost Estimation")

recycling_costs = {
    'GPU': 15.00,
    'CPU': 8.00,
    'RAM': 5.00,
    'Motherboard': 12.00,
    'HDD': 7.00,
    'SSD': 10.00
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Base Recycling Costs")
    cost_df = pd.DataFrame.from_dict(recycling_costs, orient='index', columns=['Cost ($)'])
    st.dataframe(cost_df.style.format("${:.2f}"))

with col2:
    st.subheader("Additional Cost Factors")
    labor_cost = st.number_input("Labor Cost ($/unit)", min_value=0.0, value=5.0)
    energy_cost = st.number_input("Energy Cost ($/unit)", min_value=0.0, value=2.0)
    hazardous = st.checkbox("Contains hazardous materials (+$3.00)")

# Calculate total recycling cost
if component_type in recycling_costs:
    total_recycling_cost = (recycling_costs[component_type] + 
                           labor_cost + energy_cost + 
                           (3 if hazardous else 0))
    st.info(f"Total Estimated Recycling Cost: ${total_recycling_cost:,.2f}")

# --- Decision Support ---
st.header("4. Comparative Analysis & Decision Support")

if 'estimated_value' in locals() and 'total_recycling_cost' in locals():
    profit_margin = estimated_value - total_recycling_cost
    
    decision = "Resell" if profit_margin > 0 else "Recycle"
    color = "green" if profit_margin > 0 else "red"
    
    st.markdown(f"""
    ### Recommendation: :{color}[{decision}]
    - Resale Value: ${estimated_value:,.2f}
    - Recycling Cost: ${total_recycling_cost:,.2f}
    - Profit Margin: ${profit_margin:,.2f}
    """)
    
    # Decision table
    st.subheader("Component Decision Table")
    decision_data = {
        'Component': [component_type],
        'Resale Value': [f"${estimated_value:,.2f}"],
        'Recycling Cost': [f"${total_recycling_cost:,.2f}"],
        'Margin': [f"${profit_margin:,.2f}"],
        'Decision': [decision]
    }
    st.table(pd.DataFrame(decision_data))
    
    # Visualization
    fig = px.bar(
        x=['Resale Value', 'Recycling Cost'],
        y=[estimated_value, total_recycling_cost],
        labels={'x': 'Metric', 'y': 'Amount ($)'},
        title='Economic Comparison',
        color=['Resale Value', 'Recycling Cost'],
        color_discrete_map={'Resale Value':'green', 'Recycling Cost':'red'}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please estimate both resale value and recycling costs first")

# --- How to Run ---
st.markdown("""
### How to Run This App
1. Install requirements: `pip install streamlit pandas numpy plotly scikit-learn`
2. Save this script as `economic_model.py`
3. Run: `streamlit run economic_model.py`
""")