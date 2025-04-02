import pandas as pd
import ast
import os

def load_rules(file_path='apriori_rules.csv'):
    """Load association rules from CSV file"""
    if not os.path.exists(file_path):
        print(f"Error: Rules file '{file_path}' not found.")
        return None
        
    try:
        rules_df = pd.read_csv(file_path)
        
        # Convert string representations of lists to actual lists
        for col in ['antecedents', 'consequents']:
            if rules_df[col].dtype == object:
                try:
                    rules_df[col] = rules_df[col].apply(ast.literal_eval)
                except:
                    # If literal_eval fails, try a different approach
                    rules_df[col] = rules_df[col].str.strip('()').str.split(',')
        
        return rules_df
    except Exception as e:
        print(f"Error loading rules: {e}")
        return None

def get_recommendations(selected_components, rules_df, min_confidence=0.5, min_lift=1.2, max_recommendations=5):
    if rules_df is None or rules_df.empty:
        return {}
        
    # Filter rules by confidence and lift
    filtered_rules = rules_df[(rules_df['confidence'] >= min_confidence) & 
                            (rules_df['lift'] >= min_lift)]
    
    recommendations = {}
    
    # Find rules where antecedents match selected components
    for _, rule in filtered_rules.iterrows():
        antecedents = set(rule['antecedents'])
        
        # Check if any of the selected components are in the antecedents
        if any(component in antecedents for component in selected_components):
            consequents = rule['consequents']
            
            # Add each consequent as a recommendation if not already selected
            for item in consequents:
                if item not in selected_components and item not in recommendations:
                    recommendations[item] = {
                        'confidence': rule['confidence'],
                        'lift': rule['lift']
                    }
                    
                # Stop once we reach the maximum number of recommendations
                if len(recommendations) >= max_recommendations:
                    break
        
        if len(recommendations) >= max_recommendations:
            break
            
    return recommendations

def print_recommendations(recommendations):
    """Print recommendations in a user-friendly format"""
    if not recommendations:
        print("\nNo recommendations found based on your selections.")
        return
        
    print("\n=== Recommended Components ===")
    print(f"{'Component':<30} {'Confidence':<12} {'Lift':<10}")
    print("-" * 52)
    
    for component, metrics in recommendations.items():
        print(f"{component:<30} {metrics['confidence']:<12.2f} {metrics['lift']:<10.2f}")

def run_recommendation_system():
    print("Loading PC component association rules...")
    rules_df = load_rules()
    
    if rules_df is None:
        return
        
    print(f"Loaded {len(rules_df)} association rules.")
    
    # Extract unique components from rules
    unique_components = set()
    for idx, row in rules_df.iterrows():
        unique_components.update(row['antecedents'])
        unique_components.update(row['consequents'])
    
    components_list = sorted(list(unique_components))
    
    while True:
        print("\n=== PC Component Recommendation System ===")
        print("Available components:")
        
        # Display components with numbers for easier selection
        for i, component in enumerate(components_list, 1):
            print(f"{i}. {component}")
        
        # Get user input
        print("\nEnter the numbers of components you have selected (comma-separated)")
        print("Example: 1,3,5")
        print("Or type 'q' to quit")
        
        user_input = input("> ").strip()
        
        if user_input.lower() == 'q':
            break
            
        try:
            # Parse user selections
            selected_indices = [int(idx.strip()) - 1 for idx in user_input.split(',')]
            selected_components = [components_list[idx] for idx in selected_indices 
                                if 0 <= idx < len(components_list)]
            
            if not selected_components:
                print("No valid components selected.")
                continue
                
            print(f"\nYou selected: {', '.join(selected_components)}")
            
            # Get recommendations
            recommendations = get_recommendations(selected_components, rules_df)
            print_recommendations(recommendations)
            
        except ValueError:
            print("Invalid input. Please enter component numbers separated by commas.")


run_recommendation_system()

