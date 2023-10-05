from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
#from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager

#s = Service(executable_path=r'/usr/bin/chromedriver')
o = webdriver.ChromeOptions()
#o.add_argument('--headless')
o.add_argument('--no-sandbox')
#o.add_argument('--disable-dev-shm-usage')
#o.add_argument('window-size=1200x600')
o.add_experimental_option("detach", True)
#driver = webdriver.Chrome(service=s, options=o)
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

#driver.maximize_window()
driver.get("https://www.python.org")
print(driver.title)
#time.sleep(20)


'''sb=driver.find_element_by_name("q")
sb.clear()
sb.send_keys("getting started with python")
sb.send_keys(Keys.RETURN)
print(driver.current_url)
driver.close()'''
#driver.close()