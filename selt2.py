import time
from selenium import webdriver

#from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys

s = Service(executable_path=r'/home/vboxuser/src/geckodriver')
o = webdriver.FirefoxOptions()
#o.add_argument('--headless')
o.add_argument('--no-sandbox')
o.add_argument('--disable-dev-shm-usage')
#o.add_argument('window-size=1200x600')
driver = webdriver.Firefox(service=s, options=o)


driver.maximize_window()
driver.get("https://www.python.org")
print(driver.title)


driver.close()