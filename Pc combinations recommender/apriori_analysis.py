import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AprioriAnalysis:
    def __init__(self, filename="pc_market_data.csv"):
        self.filename = filename
        self.output_filename = "apriori_ready.csv"
    
    def prepare_data(self):
        """Transform dataset into a format suitable for Apriori."""
        df = pd.read_csv(self.filename)
        df['TransactionID'] = df['Source'] + '_' + df['Model'].astype(str)
        transaction_df = df.groupby("TransactionID")["Component"].apply(lambda x: list(x)).reset_index()
        
        # Convert list to a string representation for proper CSV formatting
        transaction_df["Component"] = transaction_df["Component"].apply(lambda x: str(x))
        
        transaction_df.to_csv(self.output_filename, index=False)
        print(f"Apriori-ready dataset saved to {self.output_filename}")
    
    def run_apriori(self):
        """Run Apriori for Market Basket Analysis."""
        df = pd.read_csv(self.output_filename)
        
        # Convert string representation of lists back to actual lists
        df["Component"] = df["Component"].apply(eval)
        
        transactions = df["Component"].tolist()
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_trans, min_support=0.005, use_colnames=True)
        print("Frequent Itemsets Found:")
        print(frequent_itemsets)
        
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering the min_support threshold.")
            return
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x))
        rules["consequents"] = rules["consequents"].apply(lambda x: list(x))
        rules.to_csv("apriori_rules.csv", index=False)
        print("Apriori rules saved to apriori_rules.csv")
    
if __name__ == "__main__":
    analysis = AprioriAnalysis()
    analysis.prepare_data()
    analysis.run_apriori()
