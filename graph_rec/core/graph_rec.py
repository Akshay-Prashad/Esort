import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import List, Dict, Optional

class EnhancedPCPartGraph:
    def __init__(self, data_file: str = "pc_market_data.csv"):
        self.graph = nx.Graph()
        self.data_file = data_file
        self.transactions = self.load_data()
        self.build_graph()
        self.category_weights = {"CPU": 1.5, "GPU": 1.3, "RAM": 1.2, "Motherboard": 1.4, "Storage": 1.1, "Other": 1.0}

    def load_data(self) -> List[List[str]]:
        """Load and parse transaction data with robust error handling"""
        try:
            df = pd.read_csv(self.data_file)
            df = df.dropna(subset=["PurchasedParts"])
            transactions = df["PurchasedParts"].apply(
                lambda x: [part.strip().strip("'\"") 
                         for part in x.strip("[]").split(",") 
                         if part.strip()]
            ).tolist()
            return [t for t in transactions if t]
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def build_graph(self):
        """Build weighted graph tracking part co-occurrence frequencies"""
        for transaction in self.transactions:
            for i in range(len(transaction)):
                for j in range(i+1, len(transaction)):
                    if self.graph.has_edge(transaction[i], transaction[j]):
                        self.graph[transaction[i]][transaction[j]]["weight"] += 1
                    else:
                        self.graph.add_edge(transaction[i], transaction[j], weight=1)

    def get_part_category(self, part_name: str) -> str:
        """Categorize parts based on name patterns"""
        part_name = part_name.lower()
        if any(x in part_name for x in ["ryzen", "core i", "xeon", "cpu"]):
            return "CPU"
        elif any(x in part_name for x in ["rtx", "gtx", "radeon", "gpu", "graphics"]):
            return "GPU"
        elif any(x in part_name for x in ["ddr", "ram", "memory"]):
            return "RAM"
        elif any(x in part_name for x in ["b550", "x570", "motherboard", "mainboard"]):
            return "Motherboard"
        elif any(x in part_name for x in ["ssd", "hdd", "nvme", "storage"]):
            return "Storage"
        return "Other"

    def get_recommendations(self, purchased_parts: List[str], top_n: int = 3, min_weight: int = 1) -> List[str]:
        """Get enhanced recommendations considering weights and categories"""
        recommendations = {}
        
        for part in purchased_parts:
            if part in self.graph:
                for neighbor, data in self.graph[part].items():
                    if (neighbor not in purchased_parts and 
                        data.get("weight", 0) >= min_weight and
                        self.check_compatibility(part, neighbor)):
                        
                        category = self.get_part_category(neighbor)
                        score = data["weight"] * self.category_weights.get(category, 1.0)
                        recommendations[neighbor] = recommendations.get(neighbor, 0) + score
        
        return [item[0] for item in sorted(recommendations.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:top_n]]

    def check_compatibility(self, part1: str, part2: str) -> bool:
        """Basic compatibility checking between parts"""
        cat1 = self.get_part_category(part1)
        cat2 = self.get_part_category(part2)
        
        # Compatibility rules
        if cat1 == "CPU" and cat2 == "CPU":
            return False
        if cat1 == "Motherboard" and cat2 == "Motherboard":
            return False
        return True

    def add_user_transaction(self, parts: List[str]):
        """Add new transaction and update the graph"""
        if not parts:
            return
        
        self.transactions.append(parts)
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                if self.graph.has_edge(parts[i], parts[j]):
                    self.graph[parts[i]][parts[j]]["weight"] += 1
                else:
                    self.graph.add_edge(parts[i], parts[j], weight=1)

    def visualize_graph(self, highlight_nodes: Optional[List[str]] = None):
        """Enhanced visualization with categories and weights"""
        plt.figure(figsize=(14, 10))
        
        # Node styling
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            category = self.get_part_category(node)
            if category == "CPU":
                node_colors.append("lightcoral")
            elif category == "GPU":
                node_colors.append("lightgreen")
            elif category == "RAM":
                node_colors.append("lightblue")
            elif category == "Motherboard":
                node_colors.append("violet")
            elif category == "Storage":
                node_colors.append("gold")
            else:
                node_colors.append("lightgray")
            
            node_sizes.append(2000 if highlight_nodes and node in highlight_nodes else 1200)
        
        # Edge styling
        edge_weights = [self.graph[u][v].get("weight", 1)*0.3 for u, v in self.graph.edges()]
        
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        nx.draw(self.graph, pos, with_labels=True, 
                node_color=node_colors, node_size=node_sizes,
                edge_color="gray", width=edge_weights,
                font_size=8, font_weight="bold")
        
        plt.title("Enhanced PC Part Compatibility Graph")
        plt.tight_layout()
        plt.show()

    def save_state(self, filename: str = "pc_part_graph.pkl"):
        """Save the current graph state"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'graph': self.graph,
                    'transactions': self.transactions
                }, f)
            print(f"Graph state saved to {filename}")
        except Exception as e:
            print(f"Error saving graph state: {e}")

    def load_state(self, filename: str = "pc_part_graph.pkl"):
        """Load a saved graph state"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.transactions = data['transactions']
            print(f"Graph state loaded from {filename}")
        except Exception as e:
            print(f"Error loading graph state: {e}")
            self.graph = nx.Graph()
            self.transactions = []