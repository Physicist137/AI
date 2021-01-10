from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
import time


def get_webdriver(headless=True):
	profile = webdriver.FirefoxProfile()
	profile.set_preference('intl.accept_languages', 'en-US, en')

	if headless == True:
		opts = Options()
		opts.set_headless()
		assert opts.headless
		driver = webdriver.Firefox(firefox_profile=profile, options=opts)

	else:
		driver = webdriver.Firefox(firefox_profile=profile)


	return driver


def open_google_images(driver):
	driver.get('https://google.com')

	# Change language to english, if it is not english.
	try: driver.find_element_by_link_text('English').click()
	except: pass

	# Go to google image
	driver.find_element_by_partial_link_text('Imag').click()


def scroll_shim(passed_in_driver, object):
	x = object.location['x']
	y = object.location['y']
	scroll_by_coord = 'window.scrollTo(%s,%s);' % (
		x,
		y
	)
	scroll_nav_out_of_way = 'window.scrollBy(0, -120);'
	passed_in_driver.execute_script(scroll_by_coord)
	passed_in_driver.execute_script(scroll_nav_out_of_way)


def image_search(driver, query, depth=1, see_more=False, timeout=10, initial_id=0):
	driver.find_element_by_name('q').send_keys(query, Keys.ENTER)
	element = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CLASS_NAME, "mM5pbd")))
	time.sleep(1)
	driver.get(driver.current_url)

	page_url = driver.current_url
	photos = driver.find_elements_by_class_name('mM5pbd')

	results = []

	i = initial_id
	for photo in photos:
		actions = ActionChains(driver)
		scroll_shim(driver, photo)
		actions.move_to_element(photo).perform()
		#photo.click()
		photo.send_keys(Keys.ENTER)

		main = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CLASS_NAME, "n3VNCb")))

		src = main.get_attribute('src')
		while (not src.startswith('http')) or (src.startswith('https://encrypted-tbn0.gstatic.com/images?')):
			time.sleep(0.5)
			src = main.get_attribute('src')

		url = driver.current_url
		alt = main.get_attribute('alt')
		src = main.get_attribute('src')
		driver.find_element_by_class_name('hm60ue').click()
		time.sleep(1)

		data = {
			"id" : i,
			"level" : 1,
			"alt" : alt,
			"src" : src,
			"url" : url,
		}

		results.append(data)
		i += 1


	return results


