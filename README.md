# Facial-expression-recognition
Facial expression recognition using Pytorch on FER2013 dataset, achieving accuracy 72.53% (state of the art: 75.2%)

## Installation

* Clone this repository (only support Python 3+)
* Download FER2013 dataset in [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
* Download VGG19 pretrained ([Google Drive](https://drive.google.com/file/d/15qjJjOEWvzcAvJIwZfl7nFlk4Nw76VG1/view?usp=sharing))
* Install requirements:
```
pip install -r requirements.txt
```

## FER2013 Dataset

The data consists of 48x48 pixel grayscale images of faces, 7 class (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples. The validation set consists of 3,589 examples. The test set consists of 3,589 examples.

## Training

Run file `train.py`:
```
python train.py --dataset_root path/to/file/fer2013.csv --model_name "VGG19" --checkpoint path/to/the/checkpoint --bs ... --lr ...
```

## Evaluation

Run file `eval.py`:
```
python eval.py --dataset_root path/to/file/fer2013.csv --trained_model path/to/the/trained/model
```
Example:
```
python eval.py --dataset_root ./fer2013/fer2013.csv --trained_model model_state.pth.tar
```

## Performance

Model VGG19 achieved 72.53% accuracy on test set (state of the art 75.2%: [paper](https://arxiv.org/pdf/1612.02903.pdf))
Class-wise accuracy:

|  Class      |   Accuracy   |
| :---------: | :----------: |
|    Angry    |    65.78     |
|   Disgust   |    72.77     |
|    Fear     |    55.49     |
|    Happy    |    89.87     |
|    Sad      |    62.69     |
|  Surprise   |    82.69     |
|   Neutral	  |    70.77     |

## Face Detection

* [x] Haar Cascades
* [x] MTCNN (from link [here](https://github.com/ipazc/mtcnn))

## Demo

Predict image, run `image_demo.py`:
```
python image_demo.py --trained_model path/to/the/trained/model --mode "haarcascade"(or "mtcnn") --input path/to/input/image --output path/to/output/image
```
Example:
```
python image_demo.py --trained_model model_state.pth.tar --mode "haarcascade" --input ./input.jpg --output ./out.jpg
```

```
python image_demo.py --trained_model model_state.pth.tar --mode "mtcnn" --input ./input.jpg --output ./out.jpg
```

With video, run `video_demo.py`:
```
python video_demo.py --trained_model path/to/the/trained/model --input path/to/input/video --output path/to/output/video --save_fps 24
```

![alt text](https://github.com/anhtuan85/Facial-expression-recognition/blob/master/images/out1.png)

![alt text](https://github.com/anhtuan85/Facial-expression-recognition/blob/master/images/out3.jpg)

Some example in folder ```images```

## TODO
I hope to complete the to-do list in the near future:
* [x] Improve model face detection and classifier
* [x] Demo with video
