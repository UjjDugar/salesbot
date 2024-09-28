from bs4 import BeautifulSoup
import requests
import csv

file_path = 'index.html'

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

soup = BeautifulSoup(content, 'html.parser')

best_seller_items = soup.find_all('div', class_="product_list_holder format")[1].find_all('div', class_='list_item')

best_sellers_info = []
for item in best_seller_items:
    game_name = item.find('div', class_='product_name').text.strip()
    price = item.find('span').text.strip()
    link = item.find('a')['href']
    if link[:5] != 'https': link = 'https://www.simplygames.com/' + link

    response = requests.get(link)

    if response.status_code != 200:
        print(f"Failed to retrieve the page {link}")
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    description_div = soup.find('div', class_='description')
    
    if description_div:
        description = description_div.get_text(strip=True)
    else:
        description = None
    
    best_sellers_info.append({
        'game_name': game_name,
        'price': price,
        'link': link,
        'description': description
    })

csv_file_path = 'ps4_best_sellers.csv'

# Write the extracted data to a CSV file
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['game_name', 'price', 'link', 'description'])
    writer.writeheader()
    for info in best_sellers_info:
        writer.writerow(info)


if False:
    for info in best_sellers_info:
        print(f"Game: {info['game_name']}")
        print(f"Price: {info['price']}")
        print(f"Link: {info['link']}")
        print(f"Description: {info['description']}")
        print('-' * 20)

