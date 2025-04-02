import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class EnhancedPCPartGraph:
    def __init__(self, data_file: str = "pc_market_data.csv"):
        self.graph = nx.Graph()
        self.data_file = data_file
        self.transactions = self.load_data()
        self.build_graph()
        self._initialize_category_weights()
        self._setup_compatibility_rules()

    def _initialize_category_weights(self):
        self.category_weights = {
            "CPU": 1.5, "GPU": 1.3, "RAM": 1.2,
            "Motherboard": 1.4, "Storage": 1.1,
            "PSU": 1.0, "Case": 1.0, "Cooler": 1.0, "Other": 1.0
        }

    def _setup_compatibility_rules(self):
        self.incompatible_pairs = [
            ("intel", "am[0-9] motherboard", "Intel CPUs require Intel-compatible motherboards"),
            ("amd", "lga[0-9]+", "AMD CPUs require AMD-compatible motherboards"),
            ("ddr4", "ddr[235] motherboard", "DDR4 RAM requires DDR4-compatible motherboard"),
            ("atx psu", "sfx case", "ATX PSUs don't fit in SFX cases"),
            ("itx motherboard", "atx case", "ITX motherboards may have mounting issues in ATX cases")
        ]
        self.special_compatible = [("ram", "ram"), ("ssd", "ssd"), ("hdd", "hdd")]

    def load_data(self) -> List[List[str]]:
        try:
            df = pd.read_csv(self.data_file)
            df = df.dropna(subset=["PurchasedParts"])
            transactions = []
            for parts_list in df["PurchasedParts"]:
                try:
                    parts = [
                        part.strip().strip("'\"") 
                        for part in parts_list.strip("[]").split(",") 
                        if part.strip()
                    ]
                    if parts:
                        transactions.append(parts)
                except (AttributeError, TypeError):
                    continue
            return transactions
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def build_graph(self):
        for transaction in self.transactions:
            cleaned_parts = [self._normalize_part_name(p) for p in transaction]
            for i in range(len(cleaned_parts)):
                for j in range(i+1, len(cleaned_parts)):
                    part1, part2 = cleaned_parts[i], cleaned_parts[j]
                    if not part1 or not part2 or part1 == part2:
                        continue
                    if self.graph.has_edge(part1, part2):
                        self.graph[part1][part2]["weight"] += 1
                    else:
                        self.graph.add_edge(part1, part2, weight=1)

    def _normalize_part_name(self, part_name: str) -> str:
        if not isinstance(part_name, str):
            return ""
        return part_name.strip().lower()

    def get_part_category(self, part_name: str) -> str:
        part_lower = self._normalize_part_name(part_name)
        category_patterns = {
            "CPU": ["ryzen", "core i[0-9]", "xeon", "cpu", "processor"],
            "GPU": ["rtx[ _-]?[0-9]", "gtx[ _-]?[0-9]", "radeon", "gpu", "graphics"],
            "RAM": ["ddr[0-9]", "ram", "memory"],
            "Motherboard": ["b[0-9]{3}", "x[0-9]{3}", "motherboard", "mainboard", "mobo"],
            "Storage": ["ssd", "hdd", "nvme", "m.2", "storage"],
            "PSU": ["power supply", "psu", "[0-9]+w"],
            "Case": ["case", "chassis", "tower", "atx", "itx", "matx"]
        }
        for category, patterns in category_patterns.items():
            if any(self._matches_pattern(part_lower, pattern) for pattern in patterns):
                return category
        return "Other"

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        try:
            import re
            return bool(re.search(pattern, text, re.IGNORECASE))
        except:
            return pattern.lower() in text.lower()

    def get_recommendations(self, purchased_parts: List[str], top_n: int = 5, min_weight: int = 1) -> List[str]:
        """Returns only part names without scores"""
        scored_recs = self._get_scored_recommendations(purchased_parts, top_n, min_weight)
        return [part for part, _ in scored_recs]

    def _get_scored_recommendations(self, purchased_parts: List[str], top_n: int, min_weight: int) -> List[Tuple[str, float]]:
        recommendations = defaultdict(float)
        purchased_normalized = [self._normalize_part_name(p) for p in purchased_parts]
        
        for part in purchased_parts:
            part_normalized = self._normalize_part_name(part)
            if part_normalized not in self.graph:
                continue
            for neighbor, data in self.graph[part_normalized].items():
                if (neighbor in purchased_normalized or 
                    data.get("weight", 0) < min_weight or
                    not self.check_compatibility(part, neighbor)):
                    continue
                category = self.get_part_category(neighbor)
                score = data["weight"] * self.category_weights.get(category, 1.0)
                recommendations[neighbor] += score
        
        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def check_compatibility(self, part1: str, part2: str) -> bool:
        part1_normalized = self._normalize_part_name(part1)
        part2_normalized = self._normalize_part_name(part2)
        for pattern1, pattern2 in self.special_compatible:
            if (self._matches_pattern(part1_normalized, pattern1) and 
                self._matches_pattern(part2_normalized, pattern2)):
                return True
        for pattern1, pattern2, _ in self.incompatible_pairs:
            if (self._matches_pattern(part1_normalized, pattern1) and 
                self._matches_pattern(part2_normalized, pattern2)) or \
               ((self._matches_pattern(part1_normalized, pattern2) and 
                self._matches_pattern(part2_normalized, pattern1))):
                return False
        return True

    def get_compatibility_reason(self, part1: str, part2: str) -> str:
        part1_normalized = self._normalize_part_name(part1)
        part2_normalized = self._normalize_part_name(part2)
        for pattern1, pattern2 in self.special_compatible:
            if (self._matches_pattern(part1_normalized, pattern1)) and \
               (self._matches_pattern(part2_normalized, pattern2)):
                return "Multiple instances of this component type are supported"
        for pattern1, pattern2, reason in self.incompatible_pairs:
            if ((self._matches_pattern(part1_normalized, pattern1)) and 
                 self._matches_pattern(part2_normalized, pattern2)) or \
                ((self._matches_pattern(part1_normalized, pattern2) and 
                 self._matches_pattern(part2_normalized, pattern1))):
                return reason
        return "Compatible based on available data"

    def visualize_graph(self, highlight_parts: List[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """Enhanced to show edge weights"""
        if not self.graph.nodes():
            print("Graph is empty - nothing to visualize")
            return
            
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Draw nodes and edges
        node_colors = [self._get_category_color(node) for node in self.graph.nodes()]
        node_sizes = [2000 if highlight_parts and node in highlight_parts else 1000 for node in self.graph.nodes()]
        edge_widths = 0.3 
        
        nx.draw(self.graph, pos, 
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color="black",
                width=edge_widths,
                font_size=8,
                font_weight="bold")
        
        # Add edge weight labels
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        self._add_category_legend()
        plt.title("PC Part Compatibility Graph (Edge Weights Shown)")
        plt.tight_layout()
        return plt.gcf()

    def _get_category_color(self, category: str) -> str:
        color_map = {
            "CPU": "lightcoral", "GPU": "lightgreen", "RAM": "lightblue",
            "Motherboard": "violet", "Storage": "gold", "PSU": "orange",
            "Case": "lightgray", "Other": "silver"
        }
        return color_map.get(category, "silver")

    def _add_category_legend(self):
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='CPU', markerfacecolor='lightcoral', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='GPU', markerfacecolor='lightgreen', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='RAM', markerfacecolor='lightblue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Motherboard', markerfacecolor='violet', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Storage', markerfacecolor='gold', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    def save_state(self, filename: str = "pc_part_graph.pkl"):
        try:
            with open(filename, 'wb') as f:
                pickle.dump({'graph': self.graph, 'transactions': self.transactions}, f)
            print(f"Graph state saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving graph state: {e}")
            return False

    def load_state(self, filename: str = "pc_part_graph.pkl"):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.transactions = data['transactions']
            print(f"Graph state loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading graph state: {e}")
            self.graph = nx.Graph()
            self.transactions = []
            return False

    def add_user_transaction(self, parts: List[str]):
        if not parts:
            return
            
        cleaned_parts = [self._normalize_part_name(p) for p in parts if p]
        self.transactions.append(cleaned_parts)
        
        for i in range(len(cleaned_parts)):
            for j in range(i+1, len(cleaned_parts)):
                part1, part2 = cleaned_parts[i], cleaned_parts[j]
                if part1 == part2:
                    continue
                if self.graph.has_edge(part1, part2):
                    self.graph[part1][part2]["weight"] += 1
                else:
                    self.graph.add_edge(part1, part2, weight=1)