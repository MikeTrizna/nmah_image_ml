import os
from pathlib import Path
import json

import time

from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

model_start = time.perf_counter()

model = ResNet50(weights='imagenet')

model_time = time.perf_counter() - model_start
print(model_time)

image_list = os.listdir('thumbnails')

keras_results = []
for image_name in image_list:
	if image_name.endswith('.jpg'):
		try:
			start_time = time.perf_counter()
			image_id = image_name.split('.')[0]
			image_path = os.path.join('thumbnails',image_name)

			img = image.load_img(image_path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			preds = model.predict(x)

			prediction_labels = decode_predictions(preds, top=3)[0]
			prediction_list = []
			for _, label_name, confidence in prediction_labels:
				prediction_dict = {'label': label_name,
				                    'confidence': str(confidence)}
				prediction_list.append(prediction_dict)
			elapsed_time = time.perf_counter() - start_time
			keras_result = {'image_id':image_id,
			                'predictions':prediction_list,
			                'elapsed_time': elapsed_time}
			keras_results.append(keras_result)
		except:
			print(image_id)

with open('keras_results_resnet.json','w') as keras_out:
	json.dump(keras_results, keras_out, indent=2)
