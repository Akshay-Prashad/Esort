import time
import csv
import random
import pandas as pd
from pytrends.request import TrendReq
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class PCMarketScraper:
    def __init__(self):
        """Initialize Google Trends and Selenium driver."""
        self.pytrends = TrendReq(hl='en-US', tz=360)
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        self.driver = webdriver.Chrome(service=Service(executable_path="Pc combinations recommender\chromedriver.exe"), options=options)
    
    def get_popularity(self, items):
        """Fetch search popularity data from Google Trends."""
        batch_size = 5
        result_df = pd.DataFrame()
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            self.pytrends.build_payload(batch, timeframe='today 12-m', geo='', gprop='')
            batch_data = self.pytrends.interest_over_time()
            if batch_data.empty:
                print(f"Warning: No data retrieved for {batch}")
                continue
            batch_data = batch_data.drop(columns=['isPartial'], errors='ignore')
            result_df = batch_data if result_df.empty else pd.concat([result_df, batch_data], axis=1)
            time.sleep(random.uniform(10, 30))  # Random delay to prevent rate limiting
        return result_df if not result_df.empty else pd.DataFrame(columns=items)
    
    def get_prices(self, item, url, item_selector, title_selector, price_selector):
        """Scrape eBay and PCPartPicker for prices."""
        self.driver.get(url.format(item.replace(" ", "%20")))
        try:
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, item_selector)))
        except:
            print(f"Warning: No data found for {item} on {url}")
            return [(item, 0)]
        
        data = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, item_selector)
        for elem in elements:
            try:
                title = elem.find_element(By.CSS_SELECTOR, title_selector).text
                price = elem.find_element(By.CSS_SELECTOR, price_selector).text.replace('$', '').replace(',', '').strip()
                if price and price.replace('.', '').isdigit():
                    data.append((title, float(price)))
            except:
                continue
        return data if data else [(item, 0)]

    def collect_data(self, items, filename="pc_market_data.csv"):
        """Fetch popularity + price data and save to CSV."""
        trends_data = self.get_popularity(items)
        ebay_url = 'https://www.ebay.com/sch/i.html?_nkw={}'
        pcpartpicker_url = 'https://pcpartpicker.com/search/?q={}'
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Component", "Popularity", "Source", "Model", "Price"])
            for item in items:
                popularity = trends_data[item].iloc[-1] if item in trends_data.columns else 0
                print(f"{item} - Popularity: {popularity}")
                for source, url, item_selector, title_selector, price_selector in [
                    ("eBay", ebay_url, '.s-item', '.s-item__title', '.s-item__price'),
                    ("PCPartPicker", pcpartpicker_url, '.search_results .tr__product', '.td__name a', '.td__price')
                ]:
                    for model, price in self.get_prices(item, url, item_selector, title_selector, price_selector):
                        writer.writerow([item, popularity, source, model, price])
        print(f"Data saved to {filename}")
    
    def close(self):
        """Close the Selenium driver."""
        self.driver.quit()

if __name__ == "__main__":
    components = ["CPU", "Motherboard", "Power Supply", "Memory", "GPU", "SSD", "RAM"]
    scraper = PCMarketScraper()
    scraper.collect_data(components)
    scraper.close()
