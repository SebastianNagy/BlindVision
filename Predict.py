import torch
import torchvision
from torchvision import transforms as T
from PIL import Image
from threading import Thread
import cv2
import matplotlib.pyplot as plt
import pyttsx3

left_x = 0
right_x = 0
model = None
engine = None
talk_thread = None
detected_classes = []
detected_boxes = []
img_shape = []
positions = []
speakstring = ""

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = T.Compose([T.ToTensor()])  # Defining PyTorch Transform
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def config_model():
    global model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()


def init_tts():
    global engine
    engine = pyttsx3.init()
    # Set properties _before_ you add things to say
    engine.setProperty('rate', 150)  # Speed percent (can go over 100)
    engine.setProperty('volume', 0.9)  # Volume 0-1


def talk():
    global speakstring
    try:
        build_text_for_speech()
        engine.say(speakstring)
        engine.runAndWait()
        speakstring = ""
    except RuntimeError:
        return


def run_talk():
    global talk_thread
    talk_thread = Thread(target=talk)
    talk_thread.start()


def get_prediction(image, threshold):
  global img_shape, left_x, right_x
  #img = Image.open(img_path) # Load the image
  img = Image.fromarray(image)
  img_shape = img.size
  left_x = img_shape[0] / 3  # + img_shape[0] / 10
  right_x = (img_shape[0] / 3) * 2  # - img_shape[0] / 10
  img = transform(img).float().to(device)  # Apply the transform to the image

  pred = model([img])  # Pass the image to the model

  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().detach().numpy())]  # Get the Prediction Score
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
  pred_score = list(pred[0]['scores'].cpu().detach().numpy())

  try:
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
  except IndexError:
    pred_boxes = []

  return pred_boxes, pred_class, pred_score


def object_detection_api(frame, img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    global detected_classes, detected_boxes
    if img_path != "":
        img = cv2.imread(img_path)  # Read image with cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    else:
        img = frame

    boxes, pred_cls, pred_score = get_prediction(img, threshold)  # Get predictions
    detected_classes = pred_cls
    detected_boxes = boxes

    for box_i in range(len(boxes)):
        cv2.rectangle(img, boxes[box_i][0], boxes[box_i][1], color=(0, 255, 0), thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[box_i] + " " + str(round(pred_score[box_i], 2)), boxes[box_i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)  # Write the prediction class
    cv2.line(img, (int(left_x), 0), (int(left_x), img_shape[1]), (0, 0, 0), 2)
    cv2.line(img, (int(right_x), 0), (int(right_x), img_shape[1]), (0, 0, 0), 2)

    #plt.figure(figsize=(20, 30))  # Display the output image
    #plt.imshow(img)
    #plt.xticks([])
    #plt.yticks([])
    result = img_path[:img_path.find(".")]
    cv2.imwrite("Results/"+result+"_modified.jpg", img)
    #plt.show()
    return img


def build_text_for_speech():
    global speakstring
    for i in range(len(detected_classes)):
        obj_xpos = (detected_boxes[i][0][0]+detected_boxes[i][1][0])/2
        if obj_xpos <= left_x:
            positions.append("left")
            text = "to your left, "
        elif obj_xpos >= right_x:
            positions.append("right")
            text = "to your right, "
        else:
            positions.append("center")
            text = "in front of you, "
        speakstring = speakstring + f"{detected_classes[i]} is {text}"
    #print(speakstring)
    #print(positions)


#init_tts()
#config_model()

#img_path = "PredictData/my_img2.jpg"
#object_detection_api(img_path, threshold=0.8)

#run_talk()
#cv2.destroyAllWindows()
