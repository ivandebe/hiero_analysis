from bs4 import BeautifulSoup
import requests

# url = "https://thesaurus-linguae-aegyptiae.de/sentence/IBUBd0JOxWGMe0fRglQQpsC0KAw"
# url = "https://thesaurus-linguae-aegyptiae.de/sentence/IBUBdyfLB0Bxp0vzrxDu36HPx7o"
url = "https://thesaurus-linguae-aegyptiae.de/sentence/IBUBd9iiCAb3WEvWqLdCcj5TxmA"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract all text from the body tag
body_text = soup.body.get_text(separator='\n', strip=True)

# Save to text file
with open('body_content_1.txt', 'w', encoding='utf-8') as file:
    file.write(body_text)   