import random
import pandas as pd


cpu = ["Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"]
gpu = ["NVIDIA RTX 3060", "NVIDIA RTX 3070", "AMD RX 6700 XT"]
ram = ["Corsair 16GB", "G.Skill 16GB", "Kingston 32GB"]
storage = ["Samsung SSD 1TB", "WD HDD 2TB", "Crucial SSD 500GB"]
motherboard = ["ASUS B550", "MSI X570", "Gigabyte B450"]
power_supply = ["Corsair 650W", "EVGA 750W", "Cooler Master 850W"]

def generate_synthetic_data(num_transactions=100):
    transactions = []
    for _ in range(num_transactions):
        transaction = [
            random.choice(cpu),
            random.choice(gpu),
            random.choice(ram),
            random.choice(storage),
            random.choice(motherboard),
            random.choice(power_supply)
        ]
        transactions.append(transaction)
    return transactions

transactions = generate_synthetic_data(200)
df = pd.DataFrame({"TransactionID": range(1, 201), "PurchasedParts": transactions})


df.to_csv("alternate method\pc_market_data.csv", index=False)

print("PC market data saved.")
