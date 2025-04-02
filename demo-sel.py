from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_ebay_price(item):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(f'https://www.ebay.com/sch/i.html?_nkw={item.replace(" ", "+")}')
    time.sleep(3)
    
    prices = []
    elements = driver.find_elements(By.CSS_SELECTOR, '.s-item__price')
    for elem in elements:
        text = elem.text.replace('$', '').replace(',', '').strip()
        if text and text.replace('.', '').isdigit():
            prices.append(float(text))
    
    driver.quit()
    return min(prices) if prices else 0

def get_amazon_price(item):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(f'https://www.amazon.com/s?k={item.replace(" ", "+")}')
    time.sleep(3)
    
    prices = []
    elements = driver.find_elements(By.CSS_SELECTOR, '.a-price-whole')
    for elem in elements:
        text = elem.text.replace(',', '').strip()
        if text.isdigit():
            prices.append(float(text))
    
    driver.quit()
    return min(prices) if prices else 0

# Example usage
items = ['CPU', 'Motherboard', 'Power Supply', 'Memory']
prices = {item: {'eBay': get_ebay_price(item), 'Amazon': get_amazon_price(item)} for item in items}

print(prices)