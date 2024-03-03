import os
from django.shortcuts import render
from .forms import *
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
import json
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.utils import timezone
import cv2
import threading
from .models import *
from .send_email import send_email
from datetime import datetime
##########
import cv2 
from yolox.tracker.byte_tracker import BYTETracker, STrack 
from onemetric.cv.utils.iou import box_iou_batch 
from dataclasses import dataclass 
from supervision.detection.core import Detections, BoxAnnotator 
from typing import List 
import numpy as np 
from ultralytics import YOLO 
import supervision as sv
import easyocr
from supervision.detection.polygon_zone import PolygonZone, PolygonZoneAnnotator 
from supervision.draw.color import ColorPalette, Color
from supervision.draw.utils import draw_polygon
from .FunctionLibrary import *
import time
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerUser
from telethon import TelegramClient
import asyncio
PTime = 0

@dataclass(frozen=True) 
class BYTETrackerArgs: 
    track_thresh: float = 0.25 
    track_buffer: int = 30 
    match_thresh: float = 0.8 
    aspect_ratio_thresh: float = 3.0 
    min_box_area: float = 1.0 
    mot20: bool = False 

def detections2boxes(detections: Detections) -> np.ndarray: 
    return np.hstack(( 
        detections.xyxy, 
        detections.confidence[:, np.newaxis] 
    )) 

def tracks2boxes(tracks: List[STrack]) -> np.ndarray: 
    return np.array([ 
        track.tlbr 
        for track 
        in tracks 
    ], dtype=float) 
 
def match_detections_with_tracks( 
    detections: Detections,  
    tracks: List[STrack] 
) -> Detections: 
    if not np.any(detections.xyxy) or len(tracks) == 0: 
        return np.empty((0,)) 
 
    tracks_boxes = tracks2boxes(tracks=tracks) 
    iou = box_iou_batch(tracks_boxes, detections.xyxy) 
    track2detection = np.argmax(iou, axis=1) 
     
    tracker_ids = [None] * len(detections) 
     
    for tracker_index, detection_index in enumerate(track2detection): 
        if iou[tracker_index, detection_index] != 0: 
            tracker_ids[detection_index] = tracks[tracker_index].track_id 
 
    return tracker_ids 


MODEL2 = "plate.pt"
model2=YOLO(MODEL2)
model2.predict(source="0", show=False, stream=True, classes=0)  
model2.fuse() 

MODEL = "yolov8s.pt"
model=YOLO(MODEL)
model.predict(source="0", show=False, stream=True, classes=2)  
model.fuse() 

CLASS_NAMES_DICT = model.model.names 
CLASS_ID = [2]

byte_tracker = BYTETracker(BYTETrackerArgs()) 

cnt = 0
# VIDEO = ""
# video_info = sv.VideoInfo.from_video_path(VIDEO)


box_annotator = BoxAnnotator(thickness=1, text_scale=0.5, text_thickness=1)
tracker_ids = []
ind = 0
##########
def process_frame(frame: np.ndarray, i) -> np.ndarray:
    start_time = time.time()
    global ind
    results = model(frame)
    detections = Detections( 
        xyxy=results[0].boxes.xyxy.cpu().numpy(), 
        confidence=results[0].boxes.conf.cpu().numpy(), 
        class_id=results[0].boxes.cls.cpu().numpy().astype(int) 
    )    
    # detections = detections[detections.class_id==CLASS_ID]
    tracks = byte_tracker.update( 
        output_results=detections2boxes(detections=detections), 
        img_info=frame.shape, 
        img_size=frame.shape 
    ) 
    
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks) 
    detections.tracker_id = np.array(tracker_id) 
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool) 
    detections.filter(mask=mask, inplace=True) 
    labels = [ 
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" 
        for _, confidence, class_id, tracker_id 
        in detections 
    ]  
    for car in tracker_ids:
        if car not in tracker_id:
            tracker_ids.remove(car)
            ind -= 1
    myX = myY = 50
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels) 
    for i in range(int(len(detections.xyxy))):
        # print(i)
        x1 = int(detections.xyxy[i][0])
        x2 = int(detections.xyxy[i][2])
        y1 = int(detections.xyxy[i][1])
        y2 = int(detections.xyxy[i][3])
        cropped = frame[y1:y2, x1:x2]
        scale_percent = 2.20    
        newwidth = int(cropped.shape[0] * scale_percent )
        newheight = int(cropped.shape[1] * scale_percent )
        dim = (newheight, newwidth)
        frame2 = cv2.resize(cropped,dim,interpolation=cv2.INTER_AREA)   
     
        results2 = model2(frame2)  
        detections2 = Detections( 
            xyxy=results2[0].boxes.xyxy.cpu().numpy(), 
            confidence=results2[0].boxes.conf.cpu().numpy(), 
            class_id=results2[0].boxes.cls.cpu().numpy().astype(int) 
        )    
        detections2 = detections2[detections2.class_id==0]
        if(len(detections2.xyxy)>0):
        # for j in detections2.xyxy:
            xx1 = int(detections2.xyxy[0][0])
            xx2 = int(detections2.xyxy[0][2])
            yy1 = int(detections2.xyxy[0][1])
            yy2 = int(detections2.xyxy[0][3])
            croppedplate = frame2[yy1:yy2,xx1:xx2]
            # scale_percent = 2.20 # percent of original size
            scale_percent2 = 4.0    
            newwidth2 = int(croppedplate.shape[0] * scale_percent2 )
            newheight2 = int(croppedplate.shape[1] * scale_percent2 )
            dim2 = (newheight2, newwidth2)
            frame3 = cv2.resize(croppedplate,dim2,interpolation=cv2.INTER_AREA) 
            frameSave = frame3
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_RGB2GRAY)
            cv2.imwrite('plate.jpg', frame3)
            reader = easyocr.Reader(['en'])
            bounds = reader.readtext('plate.jpg')
            platetext=''
            for k in bounds:
                platetext += (k[-2])
            print(platetext)
            if len(platetext) == 7 and tracker_id[i] not in tracker_ids:
                # if(tracker_id[ind] not in tracker_ids):
                tracker_ids.append(tracker_id[i])                
                filename = str(tracker_ids[ind])
                filename += ".jpg"
                ind += 1
                cv2.imwrite(filename, frameSave)
            fps = 29.97
            if(len(tracker_ids)>0):
            # else:
                for l in range(len(tracker_ids)):
                    neshe = np.searchsorted(detections.tracker_id, l)
                    newX = detections.xyxy[l][0]
                    newY = detections.xyxy[l][1]
                    SpeedEstimatorTool=SpeedEstimator([newX,newY],fps)
                    speed=SpeedEstimatorTool.estimateSpeed()
                    newfilename = str(tracker_ids[l])
                    newfilename += ".jpg"
                    img = cv2.imread(newfilename)
                    img_h = img.shape[0]
                    img_w = img.shape[1]
                    img_h = int(img_h/scale_percent)
                    img_w = int(img_w/scale_percent)
                    dimImg = (img_w, img_h)
                    img = cv2.resize(img, dimImg, interpolation = cv2.INTER_AREA)
                    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                    imgroi = frame[myY:myY+img_h, myX:myX+img_w]
                    imgroi[np.where(mask)] = 0
                    imgroi += img
                    myY += img_h
                    myY += 20
                    cv2.putText(frame, str(speed)+ "km/h", (myX, myY+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
                    if speed>42:
                        shtraf = str(tracker_ids[l])
                        shtraf += "SHTRAF.jpg"
                        file_path = 'C:/Users/hackaton/Desktop/Hackathon/Hackathon/media/num_pics'
                        cv2.imwrite(os.path.join(file_path , shtraf), img)
                        # cv2.imwrite(shtraf, img)

                        shtrafFrame = str(tracker_ids[l])
                        shtrafFrame += "FRAMEshtraf.jpg"
                        file_path = 'C:/Users/hackaton/Desktop/Hackathon/Hackathon/media/num_pics'
                        cv2.imwrite(os.path.join(file_path , shtrafFrame), frame)
                        # cv2.imwrite(shtrafFrame, frame)
                        if Fine.objects.filter(accident_img=os.path.join(file_path , shtrafFrame)).exists() == False:
                            newfine = Fine(fine_date = timezone.now(), speed = int(speed), accident_img=os.path.join(file_path , shtrafFrame), number_img=os.path.join(file_path , shtraf), driver=Driver.objects.get(id=999))
                            newfine.save()
                    if len(tracker_ids)!=1:
                        myY += img_h
                        myY += 20
                myY = 50
                    # myX += img_w
                    # myX += 20
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.imwrite("Nomergo.jpg", frame)
        
    return frame

@gzip.gzip_page
def home(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def index(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect("/") 
    else:
        form = VideoForm
    return render(request, 'index.html', {'form':form})

def fine(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            waytovid = 'media/files/' + str(form.cleaned_data['file'])
            VIDEO = waytovid
            print(waytovid)
                        
            video_info = sv.VideoInfo.from_video_path(VIDEO)
            print(video_info)
            sv.process_video(source_path=VIDEO, target_path=f"result.mp4", callback=process_frame)
            # видео тут уже загружено дальше можно с ней работать
            return HttpResponseRedirect("fine") 
    else:
        form = VideoForm
    context = {
        'tablepo': Fine.objects.filter(check=False),
        'validpo': Fine.objects.filter(check=True),
        'form': form
    }
    return render(request, 'fine.html', context)

def concrete_fine(request, id):
    if request.method == 'POST':
        form = ValidFineForm(request.POST)
        num = ''
        if form.is_valid():
            print('form is valid')
            num = str(form.cleaned_data['number'])
            if Driver.objects.filter(number=num).exists():
                print('Профиль водителя с данным номером найден')
                driv = Driver.objects.get(number=num)
                latest = Fine.objects.last()
                latest.driver = driv
                latest.save()
            #после сформирования профиля водителя
            try:
                api_id = '22853963'
                api_hash = 'f20e4d9cd98113673a977e407c954055'
                token = '6100262544:AAEMvz7zd9h0cnti9IRp2czPj1y380bRtkY'
                message = "You got a fine!"

                phone = '+77089285850'
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                client = TelegramClient(phone, api_id, api_hash, loop=loop)
                # client = TelegramClient('session', api_id, api_hash)
                client.connect()

                if not client.is_user_authorized():

                    client.send_code_request(phone)
                    
                    client.sign_in(phone, input('Введите код: '))


                try:
                    userid = 822866684
                    # userid=int(Driver.objects.get(number=num).email)
                    print('userid', userid)
                    receiver = InputPeerUser(userid, 0)

                    client.send_message(receiver, message, parse_mode='html')
                except Exception as e:
                    print(e)

                client.disconnect()
            except Exception as e:
                print('e2', e)
            fine = Fine.objects.get(id=id)	
            if str(form.cleaned_data['choice_fine']) == 'Оштрафовать':	
                fine.valid = True
                # send_email(fine.driver.email, message)
            
            fine.check = True
            fine.save()
    else:
        form = ValidFineForm()
    context = {
        'my_fine': Fine.objects.get(id=id),
        'speed_addon' : Fine.objects.get(id=id).speed - 40,
        'form': form
    }
    return render(request, 'concrete_fine.html', context)
