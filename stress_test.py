# USAGE
# python stress_test.py

# import the necessary packages
from threading import Thread
import requests
import time

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://13.57.50.47:5000/"
IMAGE_PATH = "./images/protest_01.jpg"

# initialize the number of requests for the stress test along with
# the sleep amount between requests
NUM_REQUESTS = 5
SLEEP_COUNT = 0.05

def call_predict_endpoint(n):
	# load the input image and construct the payload for the request
	image = open(IMAGE_PATH, "rb").read()
	payload = {"image": ('protest_01.jpg',image,'multipart/form-data')}

	# submit the request
	r = requests.post('http://127.0.0.1:8000/Segment', files=payload).json()

	# ensure the request was sucessful
	if r["success"]:
		print("[INFO] thread {} OK".format(n))

	# otherwise, the request failed
	else:
		print("[INFO] thread {} FAILED".format(n))

# loop over the number of threads
for i in range(0, NUM_REQUESTS):
	# start a new thread to call the API
	t = Thread(target=call_predict_endpoint, args=(i,))
	t.daemon = True
	t.start()
	time.sleep(SLEEP_COUNT)

# insert a long sleep so we can wait until the server is finished
# processing the images
print("finished threads")
time.sleep(300)
print("finished timer")