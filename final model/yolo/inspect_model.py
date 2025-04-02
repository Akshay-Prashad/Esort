import torch
import os
import sys

def inspect_model(model_path):
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        return
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model file
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print("\nModel loaded successfully!")
        
        # Print the keys in the model dictionary
        print("\nModel dictionary keys:")
        if isinstance(model_dict, dict):
            for key in model_dict.keys():
                print(f"  - {key}")
        else:
            print(f"  Note: Model is not a dictionary, but a {type(model_dict)}")
        
        # Print detailed information about each key's content
        print("\nDetailed content analysis:")
        if isinstance(model_dict, dict):
            for key, value in model_dict.items():
                print(f"\nKey: {key}")
                
                if isinstance(value, torch.nn.Module):
                    print(f"  Type: torch.nn.Module")
                    print(f"  Structure: {value}")
                elif isinstance(value, dict):
                    print(f"  Type: dictionary with {len(value)} items")
                    print(f"  Sub-keys: {', '.join(list(value.keys())[:5])}{'...' if len(value) > 5 else ''}")
                elif isinstance(value, torch.Tensor):
                    print(f"  Type: Tensor")
                    print(f"  Shape: {value.shape}")
                    print(f"  Dtype: {value.dtype}")
                else:
                    print(f"  Type: {type(value)}")
        
        # If 'model' key is missing but we might have alternatives
        if isinstance(model_dict, dict) and 'model' not in model_dict:
            print("\nThe 'model' key is missing. Possible alternatives:")
            for key in model_dict.keys():
                if isinstance(model_dict[key], torch.nn.Module) or (
                    isinstance(model_dict[key], dict) and any(k.startswith('model') for k in model_dict[key].keys())):
                    print(f"  - {key} could be used as a model")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Default path or from command line
    model_path = "saved_models/best_yolov8_model.pt"
    
    # Allow command line override
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    inspect_model(model_path)

