from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from typing import List

class PCMarketScraper:
    def __init__(self, headless: bool = True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
    
    def scrape_newegg(self, url: str = "https://www.newegg.com/todays-best-deals", max_items: int = 15) -> List[str]:
        """Scrape Newegg for popular PC parts"""
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                                options=self.options)
        parts = []
        
        try:
            driver.get(url)
            time.sleep(3)
            
            elements = driver.find_elements(By.CSS_SELECTOR, ".item-title")
            parts = [elem.text.strip() for elem in elements[:max_items]] if elements else []
            
        except Exception as e:
            print(f"Error scraping Newegg: {e}")
        finally:
            driver.quit()
        
        return parts
    
    def scrape_ebay(self, url: str = "https://www.ebay.com/deals/computers-tablets", max_items: int = 15) -> List[str]:
        """Scrape eBay for popular PC parts"""
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), 
                                options=self.options)
        parts = []
        
        try:
            driver.get(url)
            time.sleep(3)
            
            elements = driver.find_elements(By.CSS_SELECTOR, ".ebayui-ellipsis-2")
            parts = [elem.text.strip() for elem in elements[:max_items]] if elements else []
            
        except Exception as e:
            print(f"Error scraping eBay: {e}")
        finally:
            driver.quit()
        
        return parts
    
    def scrape_market_data(self) -> pd.DataFrame:
        """Scrape both Newegg and eBay and return combined DataFrame"""
        newegg_parts = self.scrape_newegg()
        ebay_parts = self.scrape_ebay()
        
        data = []
        if newegg_parts:
            data.extend([("Newegg", part) for part in newegg_parts])
        if ebay_parts:
            data.extend([("eBay", part) for part in ebay_parts])
        
        return pd.DataFrame(data, columns=["Source", "PC Part"])

    def update_market_data(self, filename: str = "pc_market_data.csv"):
        """Update the market data CSV file with new scraped data"""
        try:
            new_data = self.scrape_market_data()
            if not new_data.empty:
                # Read existing data and append new data
                try:
                    existing_data = pd.read_csv(filename)
                    combined_data = pd.concat([existing_data, new_data]).drop_duplicates()
                except FileNotFoundError:
                    combined_data = new_data
                
                combined_data.to_csv(filename, index=False)
                print(f"Market data updated in {filename}")
                return True
        except Exception as e:
            print(f"Error updating market data: {e}")
        
        return False