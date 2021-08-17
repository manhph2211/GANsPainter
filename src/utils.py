import cv2
import matplotlib.pyplot as plt 
import os
import glob
import random
import json 
from sklearn import model_selection


def plot(test_monet_img,test_photo_img):
	assert test_monet_img.shape == test_photo_img.shape
	try:
		test_monet_img,test_photo_img = test_monet_img.detach().cpu().numpy(), test_photo_img.detach().cpu().numpy()
	except:
		pass
	plt.subplot(1,2,1)
	plt.imshow(test_monet_img)
	plt.axis("off")
	plt.title("Monet image")
	plt.subplot(1,2,2)
	plt.imshow(test_photo_img)
	plt.axis("off")
	plt.title("Normal image")
	plt.show()


def read_json(file):
	with open(file,'r') as f:
		data = json.load(f)
	return data 


def write_json(file,data):
	with open(file,'w') as f:
		json.dump(data,f)


def split(data_folder = '../data',test_size = 0.2):
	monet_data_paths = glob.glob(os.path.join(data_folder,'monet_jpg/*.jpg'))
	photo_data_paths = glob.glob(os.path.join(data_folder,'photo_jpg/*.jpg'))
	train_monet_data_paths, val_monet_data_paths = model_selection.train_test_split(monet_data_paths, test_size=test_size, shuffle=True)
	train_photo_data_paths, val_photo_data_paths = model_selection.train_test_split(photo_data_paths, test_size=test_size, shuffle=True)
	write_json('../data/data.json',{'train':[train_monet_data_paths,train_photo_data_paths],'val':[val_monet_data_paths,val_photo_data_paths],'full':[monet_data_paths,photo_data_paths]})


if __name__ == '__main__':

	monet_data_paths = glob.glob('../data/monet_jpg/*.jpg')
	photo_data_paths = glob.glob('../data/photo_jpg/*.jpg')

	print("There are ", len(monet_data_paths) , "monet images") # 300
	print("There are ", len(photo_data_paths) , "photo images") # 7038

	test_monet_img = cv2.imread(random.choice(monet_data_paths))
	test_photo_img = cv2.imread(random.choice(photo_data_paths))

	plot(test_monet_img,test_photo_img)
	# ------------------------------------------------------------------

	split()

