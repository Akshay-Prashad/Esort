import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Eco_model.data_loader import get_component_base_values

class EconomicModel:
    def __init__(self):
        self.model = self._init_model()
        
    def _init_model(self):
        """Initialize the machine learning model"""
        return RandomForestRegressor(n_estimators=100)
    
    def predict_resale_value(self, component_type, condition, demand_level, age, capacity):
        """Predict resale value using business rules (simulated)"""
        base_values = get_component_base_values()
        condition_multiplier = {
            'Poor': 0.3, 'Fair': 0.6, 'Good': 0.85, 'Excellent': 1.0
        }
        demand_multiplier = {
            'Low': 0.7, 'Medium': 1.0, 'High': 1.3
        }
        age_penalty = 0.9 ** age
        
        estimated_value = (
            base_values[component_type] * 
            condition_multiplier[condition] * 
            demand_multiplier[demand_level] * 
            age_penalty * 
            (1 + np.log(capacity/256)))
        
        return round(estimated_value, 2)

class RecyclingCalculator:
    @staticmethod
    def calculate_cost(component_type, labor_cost, energy_cost, hazardous):
        """Calculate total recycling cost"""
        from Eco_model.data_loader import get_base_recycling_costs
        recycling_costs = get_base_recycling_costs()
        
        if component_type in recycling_costs:
            return round(recycling_costs[component_type] + labor_cost + energy_cost + (15 if hazardous else 0), 2)
        return 0.0