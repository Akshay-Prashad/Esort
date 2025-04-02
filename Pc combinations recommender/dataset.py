import time
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from mlxtend.frequent_patterns import apriori, association_rules

class PCMarketAnalysis:
    def __init__(self):
        """Initialize Google Trends."""
        self.pytrends = TrendReq(hl='en-US', tz=360)

    def get_popularity(self, items):
        """Fetch search popularity data from Google Trends."""
        batch_size = 5
        result_df = None
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            self.pytrends.build_payload(batch, timeframe='today 12-m', geo='', gprop='')
            batch_data = self.pytrends.interest_over_time()
            batch_data = batch_data.drop(columns=['isPartial'], errors='ignore')
            result_df = batch_data if result_df is None else pd.concat([result_df, batch_data], axis=1)
            time.sleep(1)  # Avoid hitting rate limits
        
        return result_df if result_df is not None else pd.DataFrame()
    
    def get_ebay_prices(self, item):
        """Scrape eBay for item prices using BeautifulSoup."""
        url = f'https://www.ebay.com/sch/i.html?_nkw={item.replace(" ", "+")}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return [(item, 0)]
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        elements = soup.select('.s-item')
        
        for elem in elements:
            try:
                title = elem.select_one('.s-item__title').text
                price = elem.select_one('.s-item__price').text.replace('$', '').replace(',', '').strip()
                if price and price.replace('.', '').isdigit():
                    data.append((title, float(price)))
            except:
                continue
        
        return data if data else [(item, 0)]
    
    def get_pcpartpicker_prices(self, item):
        """Scrape PCPartPicker for pricing using BeautifulSoup."""
        url = f'https://pcpartpicker.com/search/?q={item.replace(" ", "%20")}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            return [(item, 0)]
        
        soup = BeautifulSoup(response.text, 'html.parser')
        data = []
        elements = soup.select('.search_results .tr__product')
        
        for elem in elements:
            try:
                title = elem.select_one('.td__name a').text
                price = elem.select_one('.td__price').text.replace('$', '').replace(',', '').strip()
                if price and price.replace('.', '').isdigit():
                    data.append((title, float(price)))
            except:
                continue
        
        return data if data else [(item, 0)]

    def collect_data(self, items, filename="pc_market_data.csv"):
        """Fetch popularity + price data and save to CSV."""
        trends_data = self.get_popularity(items)
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Component", "Popularity", "Source", "Model", "Price"])
            
            for item in items:
                popularity = trends_data[item].iloc[-1] if item in trends_data.columns else 0
                for source, fetch_method in [("eBay", self.get_ebay_prices), ("PCPartPicker", self.get_pcpartpicker_prices)]:
                    for model, price in fetch_method(item):
                        writer.writerow([item, popularity, source, model, price])
        
        print(f"Data saved to {filename}")
    
    def run_apriori(self, filename="pc_market_data.csv"):
        """Run Apriori for Market Basket Analysis."""
        df = pd.read_csv(filename)
        df['TransactionID'] = df['Source'] + '_' + df['Model'].astype(str)
        transactions = df.groupby("TransactionID")["Component"].apply(list).tolist()
        
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_trans = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_trans, min_support=0.01, use_colnames=True)
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering the min_support threshold.")
            return
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules.to_csv("apriori_rules.csv", index=False)
        print("Apriori rules saved to apriori_rules.csv")

# Define PC components to track
components = ["CPU", "Motherboard", "Power Supply", "Memory", "GPU", "SSD", "RAM"]

# Run the scraper and Apriori analysis
scraper = PCMarketAnalysis()
scraper.collect_data(components)
scraper.run_apriori()
