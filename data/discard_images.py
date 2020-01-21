from PIL import Image
import glob
import os

if __name__ == '__main__':

	SUBSETS = ['deviantart', 'wikiart']

	for subset in SUBSETS:
		print(subset)
		_dir = os.path.join(subset, 'images')
		discard_dir = os.path.join(subset, 'images_discard')
		os.makedirs(discard_dir, exist_ok=True)

		image_files = glob.glob(os.path.join(_dir, '*'))

		counter = 0 
		for image_file in image_files: 
			img = Image.open(image_file) 
			if img.mode != 'RGB': 
				new_image_file = image_file.replace('images', 'images_discard') 
				os.rename(image_file, new_image_file) 
				counter += 1 
		print(counter, 'files moved to', discard_dir)
