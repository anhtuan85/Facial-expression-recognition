from PIL import Image
import cv2
import torch
from torchvision import transforms
from vgg import VGG
from datasets import FER2013
from utils import eval, detail_eval
from face_detect.haarcascade import haarcascade_detect
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--trained_model", default = "model_state.pth.tar", type= str,
				help = "Trained state_dict file path to open")
ap.add_argument("--model_name", default= "VGG19",type= str, help = "name model")
ap.add_argument("--input", type= str, help= "Input path for detect")
ap.add_argument("--output", type= str, help = "Output path to save")
ap.add_argument("--mode", type= str, help = "mtcnn or haarcascade")
args = ap.parse_args()

mode = args.mode
assert mode in {"mtcnn", "haarcascade"}

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
crop_size= 44
image_path = args.input
#Load model
trained_model = torch.load(args.trained_model)
print("Load weight model with {} epoch".format(trained_model["epoch"]))

model = VGG(args.model_name)
model.load_state_dict(trained_model["model_weights"])
model.to(device)
model.eval()

transform_test = transforms.Compose([
		transforms.TenCrop(crop_size),
		transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
		])

def detect():
	original_image = cv2.imread(image_path)
	if mode == "haarcascade":
		gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
		faces = haarcascade_detect.face_detect(gray_image)
	else:
		detector = MTCNN()
		result = detector.detect_faces(original_image)
		faces = []
		for person in result:
			faces.append(person["box"])
	if faces != []:
		for (x, y, w, h) in faces:
			roi = original_image[y:y+h, x:x+w]
			roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
			roi_gray = cv2.resize(roi_gray, (48, 48))
			
			roi_gray = Image.fromarray(np.uint8(roi_gray))
			inputs = transform_test(roi_gray)
			
			ncrops, c, ht, wt = np.shape(inputs)
			inputs = inputs.view(-1, c, ht, wt)
			inputs = inputs.to(device)
			outputs = model(inputs)
			outputs = outputs.view(ncrops, -1).mean(0)
			_, predicted = torch.max(outputs, 0)
			expression = classes[int(predicted.cpu().numpy())]
			
			cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
			
			text = "{}".format(expression)
			
			cv2.putText(original_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
	cv2.imwrite(args.output, original_image)
if __name__ == '__main__':
	detect()
	print("Done!")