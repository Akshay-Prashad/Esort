import requests
from bs4 import BeautifulSoup

# Function to scrape eBay for product price
def get_ebay_price(item):
    url = f'https://www.ebay.com/sch/i.html?_nkw={item.replace(" ", "+")}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    prices = [float(p.text.replace('$', '').replace(',', '')) for p in soup.select('.s-item__price') if '$' in p.text]
    return min(prices) if prices else 0

# Function to scrape Amazon for product price
def get_amazon_price(item):
    url = f'https://www.amazon.com/s?k={item.replace(" ", "+")}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    prices = [float(p.text.replace('$', '').replace(',', '')) for p in soup.select('.a-price-whole') if p.text.strip()]
    return min(prices) if prices else 0

# Example usage
items = ['CPU', 'Motherboard', 'Power Supply', 'Memory']
prices = {item: {'eBay': get_ebay_price(item), 'Amazon': get_amazon_price(item)} for item in items}

print(prices)