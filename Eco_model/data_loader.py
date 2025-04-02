import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_market_data():
    """Generate synthetic market data for components"""
    components = ['GPU', 'CPU', 'RAM', 'Motherboard', 'HDD', 'SSD']
    brands = ['NVIDIA', 'AMD', 'Intel', 'Samsung', 'Western Digital', 'ASUS']
    
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

def get_base_recycling_costs():
    """Return base recycling costs dictionary"""
    return {
        'GPU': 15.00,
        'CPU': 8.00,
        'RAM': 5.00,
        'Motherboard': 12.00,
        'HDD': 7.00,
        'SSD': 10.00
    }

def get_component_base_values():
    """Return base resale values for components"""
    return {
        'GPU': 200, 'CPU': 100, 'RAM': 50, 
        'Motherboard': 80, 'HDD': 30, 'SSD': 60
    }