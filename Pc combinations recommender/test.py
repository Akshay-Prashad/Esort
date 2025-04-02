import time
import csv
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class PCMarketScraper:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")  
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = uc.Chrome(service=Service(executable_path="chromedriver.exe"), options=options)

    def get_trending_topics(self):
        """Fetch trending search topics from Google Trends using Selenium."""
        url = "https://trends.google.com/trends/trendingsearches/daily"
        self.driver.get(url)
        time.sleep(5)

        # Scroll down to ensure all trends load
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.summary-text"))
            )
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div.summary-text")
            trends = [el.text.strip() for el in elements if el.text.strip()]
            print("Trending Topics:", trends)
            return trends
        except:
            print("Failed to retrieve Google Trends data.")
            return []

    def get_ebay_prices(self, item):
        """Scrape eBay for prices using Selenium."""
        url = 'https://www.ebay.com/sch/i.html?_nkw={}'.format(item.replace(" ", "+"))
        self.driver.get(url)
        time.sleep(5)

        try:
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.s-item')))
        except:
            print(f"Warning: No data found for {item} on eBay")
            return [(item, 0)]

        data = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, '.s-item')
        for elem in elements:
            try:
                title = elem.find_element(By.CSS_SELECTOR, '.s-item__title').text
                price = elem.find_element(By.CSS_SELECTOR, '.s-item__price').text.replace('$', '').replace(',', '').strip()
                if price and price.replace('.', '').isdigit():
                    data.append((title, float(price)))
            except:
                continue
        return data if data else [(item, 0)]

    def collect_data(self, filename="pc_market_data.csv"):
        """Fetch Google Trends & eBay price data and save to CSV."""
        trending_topics = self.get_trending_topics()
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Component", "Source", "Model", "Price"])
            for item in trending_topics:
                print(f"Fetching prices for: {item}")
                for source, fetch_method in [("eBay", self.get_ebay_prices)]:
                    for model, price in fetch_method(item):
                        writer.writerow([item, source, model, price])
        print(f"Data saved to {filename}")

    def close(self):
        """Close the Selenium driver."""
        self.driver.quit()

if __name__ == "__main__":
    scraper = PCMarketScraper()
    scraper.collect_data()
    scraper.close()
