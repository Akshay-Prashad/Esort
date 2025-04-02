import plotly.express as px
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_price_trends(df, component):
        """Generate price trend plot for selected component"""
        component_df = df[df['Component'] == component]
        if not component_df.empty:
            return px.line(
                component_df, 
                x='Date', 
                y='Price ($)', 
                title=f"Price Trend for {component}",
                hover_data=['Brand', 'Condition']
            )
        return None
    
    @staticmethod
    def plot_comparison(resale_value, recycling_cost):
        """Generate comparison bar chart"""
        return px.bar(
            x=['Resale Value', 'Recycling Cost'],
            y=[resale_value, recycling_cost],
            labels={'x': 'Metric', 'y': 'Amount ($)'},
            title='Economic Comparison',
            color=['Resale Value', 'Recycling Cost'],
            color_discrete_map={'Resale Value':'green', 'Recycling Cost':'red'}
        )
    
    @staticmethod
    def show_decision_table(component, resale_value, recycling_cost, margin, decision):
        """Generate decision table dataframe"""
        return pd.DataFrame({
            'Component': [component],
            'Resale Value': [f"${resale_value:,.2f}"],
            'Recycling Cost': [f"${recycling_cost:,.2f}"],
            'Margin': [f"${margin:,.2f}"],
            'Decision': [decision]
        })