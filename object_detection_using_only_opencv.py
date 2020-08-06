import cv2
import numpy as np
import time
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
Tsize = 320
prob_tresh = 0.5
nms_threshold = 0.2
ModelConfigFile = "yolov3.cfg"
ModelWeightFile = "yolov3.weights"
model = cv2.dnn.readNetFromDarknet(ModelConfigFile,ModelWeightFile)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classfile= "coco.names"
with open(classfile,'r') as f:
    classnames = f.read().rsplit('\n')
def findbox(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    prob = []
    for output in outputs:
        for det in output:
            rest = det[5:]
            classId = np.argmax(rest)
            conf = rest[classId]
            if conf>= prob_tresh:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int(det[0]*wT-w/2),int(det[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                prob.append(float(conf))
    indices = cv2.dnn.NMSBoxes(bbox,prob,prob_tresh,nms_threshold=nms_threshold)
    for index in indices:
        index = index[0]
        x,y,w,h = bbox[index][0],bbox[index][1],bbox[index][2],bbox[index][3]
        conf = int(prob[index]*100)
        classId = classIds[index]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classnames[classIds[index]].upper()} {int(prob[index]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)  
cap = cv2.VideoCapture(0)
time.sleep(3)
while(cap.isOpened()):
    flag,img = cap.read()
    if flag==False:
        break
    else:
        blob = cv2.dnn.blobFromImage(img,1/255,(Tsize,Tsize),[0,0,0],1,crop=False)
        outputlayers = model.getUnconnectedOutLayersNames()
        model.setInput(blob)
        outputs = model.forward(outputlayers)
        findbox(outputs,img)
        out.write(img)
        cv2.imshow("output",img)
        k = cv2.waitKey(1)
        if k==27:
            break
cap.release()    
out.release()
cv2.destroyAllWindows()
