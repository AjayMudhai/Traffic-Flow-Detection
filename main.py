import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf

import time
import colorsys
import imutils
from moviepy.editor import VideoFileClip
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
import random
from YOLO.utils_class import Utilities
from YOLO.yolo import *


class MainClass:
    def __init__(self):
        self.utils=Utilities()
        self.input_size=416
        self.score_threshold=0.6
        self.iou_threshold=0.45
        self.max_cosine_distance = 0.7
        self.nn_budget = None
        self.times=[]
        
        self.video_path   = '/Users/ajaymudhai/Desktop/SL/Videos/NVR_ch1_main_20200207140000_20200207143000.asf'  #### Enter video path for video file
        class_name_path="model_data/coco/coco.names"
        self.NUM_CLASS = self.utils.read_class_names(class_name_path)
        self.key_list = list(self.NUM_CLASS.keys()) 
        self.val_list = list(self.NUM_CLASS.values())
        self.Track_only=['person','car','bicycle','motorbike','bus','truck']

        self.load_yolo()
        self.load_tracker()
        self.load_video()

        self.traffic_direction_x=0
        self.traffic_direction_y=0
        self.traffic_movement={}
        self.vehicle_movement={}
        self.diff_x=0
        self.diff_y=0
    





        
    def load_yolo(self):
        Darknet_weights = "model_data/yolov3.weights"
        self.yolo = Create_Yolo(input_size=416)
        self.utils.load_yolo_weights(self.yolo, Darknet_weights)

    def load_tracker(self):
        model_filename = '/Users/ajaymudhai/Desktop/SL/model_data/mars1-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine",self.max_cosine_distance,self.nn_budget)
        self.tracker = Tracker(metric)

    def load_video(self):
        self.vid = cv2.VideoCapture(self.video_path)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
      
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        self.totalFrames = int(self.vid.get(prop))
        print('........................................ \n')
        print('Total Frames in Video : {}'.format(self.totalFrames))
        print('......................................... \n')
      
      

    def detectTrack(self,img):
       
        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            pass

        image_data = self.utils.img_preprocessing(np.copy(original_image))
        image_data = tf.expand_dims(image_data, 0)


        pred_bbox = self.yolo.predict(image_data)
        
     

       

       
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
    


       
        bboxes = self.utils.box_postprocessing(pred_bbox, original_image)
        bboxes = self.utils.nms(bboxes)
       

         # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(self.Track_only) !=0 and self.NUM_CLASS[int(bbox[5])] in self.Track_only or len(self.Track_only) == 0:
                boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(self.NUM_CLASS[int(bbox[5])])

        boxes = np.array(boxes) 
        
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(self.encoder(original_image, boxes))
      
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
        self.tracker.predict()
        self.tracker.update(detections)


        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = self.key_list[self.val_list.index(class_name)] # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function


  
        self.traffic_flow_direction(tracked_bboxes)

  
        image = self.draw_rect(original_image,tracked_bboxes)
      

        return image

   

           
    
    
    def draw_rect(self,image,bboxes):   
       
        num_classes = len(self.NUM_CLASS)
        image_h, image_w, _ = image.shape
        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = (0,255,0)
            bbox_thick = 1
           
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

    
            

          
          

       
            score_str = " "+str(score)
         

            if(abs(self.diff_x)<2 and abs(self.diff_y)<2):
                bbox_color=(255,255,255)


            elif(abs(self.diff_x)>abs(self.diff_y)):
                if(self.vehicle_movement[score][2]*self.diff_x>0):  ##### To test if vehicle and traffic flow is in same direction
                    bbox_color=(0,255,0)
                else:
                    bbox_color=(0,0,255)
            else:
                if(self.vehicle_movement[score][3]*self.diff_y>0):  ##### To test if vehicle and traffic flow is in same direction
                    bbox_color=(0,255,0)
                else:
                    bbox_color=(0,0,255)



     

      

            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)
           
                
            

           
            




            label = "{}".format(self.NUM_CLASS[class_ind]) + score_str

            
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
           
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

      
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale,(255,255,255), bbox_thick, lineType=cv2.LINE_AA)
            
        return image


    
    def traffic_flow_direction(self,bboxes):
        for i, bbox in enumerate(bboxes):
            
            coor = np.array(bbox[:4], dtype=np.int32)
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
            center_x=(x1+x2)/2
            center_y=(y1+y2)/2
            score = bbox[4]
            if (score not in self.vehicle_movement):
           
                self.vehicle_movement[score]=[center_x,center_y,0,0]
                # print(self.vehicle_movement)
            else:

                prev_coor=self.vehicle_movement[score]
                prev_x,prev_y=prev_coor[0],prev_coor[1]
                vehicle_diff_x,vehicle_diff_y=prev_coor[2],prev_coor[3]
                vehicle_diff_x+=center_x-prev_x
                vehicle_diff_y+=center_y-prev_y
              
                self.vehicle_movement[score]=[center_x,center_y,vehicle_diff_x,vehicle_diff_y]
                # print(self.vehicle_movement)
                self.diff_x+=center_x-prev_x
                self.diff_y+=center_y-prev_y

                





                
           
         
            


                


            
            

        
       


  

    def capture_frame(self):
        result = cv2.VideoWriter('result_test.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10,(1920,1080)) 

        frame_count=0
        while frame_count<self.totalFrames:
            ret,frame = self.vid.read()
            
            if ret:
                frame=self.detectTrack(frame)
                frame_count+=1
                percent_comp=(frame_count/self.totalFrames)*100
                result.write(frame)
                # print('Frames Completed : {}/{}     {} %'.format(frame_count,self.totalFrames,percent_comp))
                cv2.imshow('frame',frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
                




if __name__=='__main__':
    main=MainClass()
    main.capture_frame()
    





