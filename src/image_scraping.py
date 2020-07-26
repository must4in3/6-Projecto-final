from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests

def getPage(url):
    '''
    This function show a basic way to make a get requests used Selenium
    First line to to avoid this error:
    WebDriverException: Message: 'chromedriver' executable needs to be available in the path.
    driver.execute_script(..) to scroll the page in Selenium untill the bottom
    '''
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    html = driver.page_source
    soup = BeautifulSoup(html,features="lxml")
    driver.quit()
    return soup


def get_image(df):
    count = 0
    for j in range(0,200,2):
        for _, url in enumerate(df[j]):
            try:
                if 'jpg' in url:
                    img_data = requests.get(url).content
                    with open(f'inputs/building_dataset/{count}.jpg', 'wb') as handler:
                        handler.write(img_data)
                        count += 1
                        break
            except:
                pass