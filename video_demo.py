from PIL import Image
import cv2
import torch
from torchvision import transforms
from vgg import VGG
from datasets import FER2013
from utils import eval, detail_eval
import numpy as np
import argparse
from mtcnn.mtcnn import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--trained_model", default = "model_state.pth.tar", type= str,
				help = "Trained state_dict file path to open")
ap.add_argument("--model_name", default= "VGG19",type= str, help = "name model")
ap.add_argument("--input", type= str, help= "Input path video to detect")
ap.add_argument("--output", type= str, help = "Output path to save")
ap.add_argument("--save_fps", default = 24, type= int, help = "FPS for save output")
args = ap.parse_args()

classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
crop_size= 44
image_path = args.input

#Load model
trained_model = torch.load(args.trained_model)

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

def detect_video():
	vs = cv2.VideoCapture(image_path)
	writer = cv2.VideoWriter(args.output, 0x7634706d, args.save_fps, (int(vs.get(4)), int(vs.get(3))), True)
	(W, H) = (None, None)
	
	detector = MTCNN()
	while True:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		result = detector.detect_faces(frame)
		faces = []
		for person in result:
			faces.append(person["box"])
		
		if faces != []:
			for (x, y, w, h) in faces:
				roi = frame[y:y+h, x:x+w]
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
				
				cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
				
				text = "{}".format(expression)
				
				cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
		#Write video
		writer.write(frame)
	writer.release()
if __name__ == "__main__":
	detect_video()