import time
from selenium import webdriver

#from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

s = Service(executable_path=r'/usr/bin/chromedriver')
o = webdriver.ChromeOptions()
#o.add_argument('--headless')
o.add_argument('--no-sandbox')
#o.add_argument('--disable-dev-shm-usage')
#o.add_argument('window-size=1200x600')
driver = webdriver.Chrome(service=s, options=o)


#driver.maximize_window()
#driver.get("https://www.python.org")
driver.get("https://www.google.com/")
print(driver.title)

sb=driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/textarea')
sb.send_keys("typing")
print(sb)

'''sb=driver.find_element_by_name("q")
sb.clear()
sb.send_keys("getting started with python")
sb.send_keys(Keys.RETURN)
print(driver.current_url)
driver.close()'''
driver.close()