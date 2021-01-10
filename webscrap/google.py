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


def query_image_from_google(query, driver, timeout=60):
	driver.find_element_by_name('q').send_keys(query, Keys.ENTER)
	element = WebDriverWait(driver, timeout) \
		.until(EC.presence_of_element_located((By.CLASS_NAME, "mM5pbd")))

	time.sleep(1)
	driver.get(driver.current_url)


def get_images_from_google(driver, level=1, amount=48, \
timeout=60, init_id=0, from_id=0):
	# Get google page url.
	page_url = driver.current_url

	# Get the specified amount of photos.
	photos = driver.find_elements_by_class_name('mM5pbd')[:amount]

	results = []

	i = init_id
	
	for photo in photos:
		actions = ActionChains(driver)
		scroll_shim(driver, photo)
		actions.move_to_element(photo).perform()
		#photo.click()
		photo.send_keys(Keys.ENTER)

		main = WebDriverWait(driver, timeout) \
			.until(EC.presence_of_element_located((By.CLASS_NAME, "n3VNCb")))

		src = main.get_attribute('src')
		while (not src.startswith('http')) or \
		(src.startswith('https://encrypted-tbn0.gstatic.com/images?')):
			time.sleep(0.5)
			src = main.get_attribute('src')

		url = driver.current_url
		alt = main.get_attribute('alt')
		src = main.get_attribute('src')
		driver.find_element_by_class_name('hm60ue').click()
		time.sleep(1)

		data = {
			"id" : i,
			"from" : from_id,
			"level" : level,
			"page" : page_url,
			"alt" : alt,
			"src" : src,
			"url" : url,
		}

		results.append(data)
		i += 1


	return results


# Collected images at each level will be: amount^level.
# Total amount of collected images will be O(amount^(depth+1)).
# Due to the insane exponential nature depending on depth, use with great care.
def image_search(query, driver, depth=1, amount=48, timeout=60):
	# Make the query.
	query_image_from_google(query, driver, timeout)

	# Get images in the first level
	current_level = 1
	results = get_images_from_google(
		driver = driver,
		level = 1,
		amount = amount,
		timeout = timeout,
		init_id = 1
	)

	# Get images from the next levels.
	current_level += 1

	while current_level <= depth:
		init_id = results[-1]['id'] + 1
		level_results = []
		for result in results:
			if result['level'] == current_level - 1:
				# Go to webpage.
				driver.get(result['url'])

				# Get see more link text element.
				# see_more = driver.find_element_by_link_text('See more')
				see_more = WebDriverWait(driver, timeout).until(
					EC.presence_of_element_located(
						(By.LINK_TEXT, 'See more')
					)
				)

				# Load page.
				# see_more.click()
				driver.get(see_more.get_attribute('href'))
				time.sleep(1)

				# Get all the images.
				intermediate = get_images_from_google(
					driver = driver,
					level = current_level,
					amount = amount,
					timeout = timeout,
					init_id = init_id,
					from_id = result['id']
				)

				init_id = intermediate[-1]['id'] + 1
				level_results += intermediate


		results += level_results
		current_level += 1
	

	return results



