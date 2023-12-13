
from utils import cv2_safe_read
import pandas as pd
import cv2
import numpy as np
import os
import json

class Visualizer:
    def __init__(self):
        """
        A basic tool to visualize videos based on given dataframes
        @functionalities:
        1. draw_bboxes (with or without IDs)
        2. draw_gaze_general
        3. draw_gaze_patterns
        4. draw_gaze_heatmaps 
        5. draw_emotion_curve
        6. draw_focus_curve
        """
        self._frame_list = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # set the codec
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.circle_thickness = 5
        self.line_thickness = 2
        self.font_size = 1
        self.color_dict = {
            'general': (0,255,0), # green
            'teacher':  (0,255,0), # green
            'student': (0,0,255), # red
            'object_color': (255,0,0) # blue
        }
        self.w = -1
        self.h = -1

    def update_colors(self, new_color_dict:dict):
        self.color_dict = new_color_dict

    def set_output_width_height(self, width: int, height:int):
        '''
        Set the width/height of the output video to a given value
        '''
        self.w = width
        self.h = height

    def load_frames_from_dir(self, frame_dir:str, compression=0.5):
        """
        load all frames from a given directory
        """

        framels = ['%s/%s'%(frame_dir, f) for f in os.listdir(frame_dir) if f[-4:]=='.jpg']
        framels.sort()

        self.load_frames_from_list(framels, compression)

    def load_frames_from_list(self, frame_ls:enumerate, compression=0.5):
        """
        load frames by a given list
        @ self._frame_list: keys are path to the frames, values are frames
        """
        initial_frame = cv2_safe_read(frame_ls[0])

        if self.w<0:
            # update values by the given frames
            h, w, _ = initial_frame.shape
            h, w = map(int, [h*compression, w*compression])
            self.h = h 
            self.w = w

        for f in range(len(frame_ls)):
            frame_path = frame_ls[f]
            frame = cv2_safe_read('%s\\%06d.jpg'%(frame_path))
            self._frame_list[f] = cv2.resize(frame, (w,h)) if compression<1 else frame

    def draw_bboxes(self, annotations:dict, color_by_id=None, write_id=False):
        """
        draw bboxes on all frames in self._frame_list by given annotations
        inputs:
        @annotations: dictionary[image_path] = [xmin, ymin, xmax, ymax], values normalized in 0-1
        @color_by_id: the ID (specific color) of the box, if None, will use general_color
        @write_id: write the given id of on the left top of the bounding box
        """
        if color_by_id:
            if not color_by_id in self.color_dict.keys():
                print("[ERROR] The chosen color id is not in color_dict of this Visualizer, will use general color to draw.")
                print("Please update the color_dict.")
                draw_color = self.color_dict["general"]
            else:
                draw_color = self.color_dict[color_by_id]
        else:
            write_id = False
            draw_color = self.color_dict["general"]

        for img_path in annotations.keys():
            xmin, ymin, xmax, ymax = annotations[img_path]
            xmin, ymin, xmax, ymax = map(int, [xmin*self.w, ymin*self.h, xmax*self.w, ymax*self.h])   
            try:
                frame = self._frame_list[img_path]   
            except Exception as e:
                print("[ERROR] The image %s is not loaded in the frame list, please recheck!"%img_path)
                continue
            frame = self._draw_individual_bbox(img_path, (xmin,ymin, xmax, ymax), draw_color) # Draw head box
            if write_id:
                frame = cv2.putText(frame, str(color_by_id), (xmin+5, ymax+20), self.font, self.font_size, color_by_id, self.line_thickness)
            
            self._frame_list[img_path] = frame

    def _draw_individual_bbox(self, img_path:str, bbox:tuple, color:tuple):
        """
        draw bounding boxes on given img_path
        """
        xmin, ymin, xmax, ymax = bbox
        frame = self._frame_list[img_path]
        frame  = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, self.line_thickness) 
        self._frame_list[img_path] = frame
        return frame
    
    def draw_gaze_general(self, annotations:dict, color_by_id=None, write_pattern=False, draw_pattern_illustr=False,
                          illustr_path=r'D:\ShanghaiASD_project\ShanghaiASD\Misc'):
        """
        draw gaze line point on all frames in self._frame_list by given annotations
        inputs:
        @annotations: dictionary[image_path] = [xmin, ymin, xmax, ymax, gaze_x, gaze_y, patterns], values normalized in 0-1
        @color_by_id: the ID (specific color) of the box, if None, will use general_color
        @write_id: write the given id around the head boxes
        @write_pattern: write the gaze pattern around the given head boxes 
        @draw_pattern_illustr: draw the gaze pattern illustration on the right top corner of the video
        @illustr_path: path to gaze pattern illustration figures.
        """

        if color_by_id:
            if not color_by_id in self.color_dict.keys():
                print("[ERROR] The chosen color id is not in color_dict of this Visualizer, will use general color to draw.")
                print("Please update the color_dict.")
                draw_color = self.color_dict["general"]
            else:
                draw_color = self.color_dict[color_by_id]
        else:
            draw_color = self.color_dict["general"]

        if draw_pattern_illustr:
            illustr_path = illustr_path
            assert os.path.exists(illustr_path)
            
        for img_path in annotations.keys():
            xmin, ymin, xmax, ymax, gaze_x, gaze_y, pattern = annotations[img_path]
            xmin, ymin, xmax, ymax, gaze_x, gaze_y = map(int, [xmin*self.w, ymin*self.h, xmax*self.w, ymax*self.h, \
                                                               gaze_x*self.w, gaze_y*self.h])   
            try:
                frame = self._frame_list[img_path]   
            except Exception as e:
                print("[ERROR] The image %s is not loaded in the frame list, please recheck!"%img_path)
                continue
            frame = self._draw_individual_bbox(img_path, (xmin,ymin, xmax, ymax), draw_color) # Draw head box
            center_x, center_y = map(int,[(xmax+xmin)/2, (ymin+ymax)/2])
            frame = cv2.circle(frame, (gaze_x, gaze_y), self.circle_thickness, draw_color, -1) # Draw gaze point
            frame = cv2.line(frame, (center_x, center_y), (gaze_x, gaze_y), draw_color, self.line_thickness) # Draw line from head center to gaze point
            
            if write_pattern: # Write Gaze Pattern
                frame  = cv2.putText(frame, str(pattern), (xmin+5, ymax+20), self.font, self.font_size, draw_color, self.line_thickness)
            
            if (draw_pattern_illustr): # Draw based on the given gaze pattern
                fig = cv2_safe_read('%s/%s_figure.png'%(illustr_path, pattern.lower()))
                fig_h, fig_w, _ = fig.shape
                frame[:fig_h, -fig_w:, :] = fig
                frame = cv2.rectangle(frame, (self.w-fig_w, 0), (self.w, fig_h), draw_color, self.line_thickness)
            self._frame_list[img_path] = frame
    
    def load_emotion(self, emotion_dir):
        self.label_list = pd.read_csv(emotion_dir,sep=',',header=None).values
        self.neutral_list=[]
        self.angry_list=[]
        self.disgust_list=[]
        self.fear_list=[]
        self.happy_list=[]
        self.sad_list=[]
        self.surprise_list=[]
        last_item = 8*['0.143']
        for i in range(len(self.label_list)):
            item = self.label_list[i]
            if 'frameID' not in item:
                if type(item[1]) == float: #遇到null都置0
                    item = last_item
                self.neutral_list.append(float(item[1]))
                self.angry_list.append(float(item[2]))
                self.disgust_list.append(float(item[3]))
                self.fear_list.append(float(item[4]))
                self.happy_list.append(float(item[5]))
                self.sad_list.append(float(item[6]))
                self.surprise_list.append(float(item[7]))
                last_item = item

    def draw_emotion_curve(self,bar_height=90,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=0.5):
        color_list=[(255,0,0),(255,165,0),(128,128,0),(0,255,0),(0,191,243),(0,0,255),(233,0,233)]
        #draw curve
        for frame_num in self._frame_list.keys():
            i = frame_num-1
            img = self._frame_list[frame_num]

            #curve param
            curve_h = bar_height
            curve_thick = 2

            #padding
            img = cv2.copyMakeBorder(img,0,curve_h+10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            im_h = img.shape[0]
            im_w = img.shape[1]
            self.h = im_h
            self.w = im_w

            #draw curve
            word_w = 140
            space = (im_w-word_w)/len(self._frame_list.keys())

            for j in range(i):
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.neutral_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.neutral_list[j+1]*curve_h)-5),color_list[0],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.angry_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.angry_list[j+1]*curve_h)-5),color_list[1],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.disgust_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.disgust_list[j+1]*curve_h)-5),color_list[2],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.fear_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.fear_list[j+1]*curve_h)-5),color_list[3],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.happy_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.happy_list[j+1]*curve_h)-5),color_list[4],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.sad_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.sad_list[j+1]*curve_h)-5),color_list[5],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.surprise_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.surprise_list[j+1]*curve_h)-5),color_list[6],curve_thick)
            
            #为了不让字出格子，使其逐渐往左偏移的宽度（像素数），最大70
            all_num=len(self.neutral_list) #所有帧的数量
            shift_wid = int(70.0*(frame_num-all_num*0.8)/(all_num*0.2)) if frame_num > all_num*0.8 else 0

            cv2.putText(img,'neutral',(int((i)*space)+word_w-shift_wid,im_h-int(self.neutral_list[i]*curve_h)-5),font,font_scale,color_list[0],2)
            cv2.putText(img,'angry',(int((i)*space)+word_w-shift_wid,im_h-int(self.angry_list[i]*curve_h)-5),font,font_scale,color_list[1],2)
            cv2.putText(img,'disgust',(int((i)*space)+word_w-shift_wid,im_h-int(self.disgust_list[i]*curve_h)-5),font,font_scale,color_list[2],2)
            cv2.putText(img,'fear',(int((i)*space)+word_w-shift_wid,im_h-int(self.fear_list[i]*curve_h)-5),font,font_scale,color_list[3],2)
            cv2.putText(img,'happy',(int((i)*space)+word_w-shift_wid,im_h-int(self.happy_list[i]*curve_h)-5),font,font_scale,color_list[4],2)
            cv2.putText(img,'sad',(int((i)*space)+word_w-shift_wid,im_h-int(self.sad_list[i]*curve_h)-5),font,font_scale,color_list[5],2)
            cv2.putText(img,'surprise',(int((i)*space)+word_w-shift_wid,im_h-int(self.surprise_list[i]*curve_h)-5),font,font_scale,color_list[6],2)

            cv2.putText(img,'Emotion',(30,im_h-int(curve_h/2)),font,font_scale,(0,0,0),2)
            self._frame_list[frame_num] = img
    
    def draw_focus_curve(self,bar_height=90,teacher_curve_color=(0,255,0),student_curve_color=(0,0,255),font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=0.5):
        #draw curve
        for frame_num in self._frame_list.keys():
            i = frame_num-1
            img = self._frame_list[frame_num]

            #curve param
            curve_h = bar_height
            curve_thick = 2

            #padding
            img = cv2.copyMakeBorder(img,0,curve_h*2+10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            im_h = img.shape[0]
            im_w = img.shape[1]
            self.h = im_h
            self.w = im_w

            #draw curve
            word_w = 140
            space = (im_w-word_w)/len(self._frame_list.keys())

            for j in range(i):
                #mark if the student is looking at the camera
                if self.prob_list_student[j]>0.5 and self.prob_list_student[j+1]>0.5:
                    cv2.fillConvexPoly(img,
                    np.array([[int(j*space)+word_w,im_h-curve_h-5],[int(j*space)+word_w,im_h-5],[int((j+1)*space)+word_w,im_h-5],[int((j+1)*space)+word_w,im_h-curve_h-5]]),
                    (200,200,255))

                #student curve
                if self.prob_list_student[j]>0 and self.prob_list_student[j+1]>0:
                    cv2.line(img,(int(j*space)+word_w,im_h-int(self.prob_list_student[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.prob_list_student[j+1]*curve_h)-5),student_curve_color,curve_thick)

                #mark if the teacher is looking at the camera
                if self.prob_list_teacher[j]>0.5 and self.prob_list_teacher[j+1]>0.5:
                    cv2.fillConvexPoly(img,
                    np.array([[int(j*space)+word_w,im_h-curve_h*2-10],[int(j*space)+word_w,im_h-curve_h-10],[int((j+1)*space)+word_w,im_h-curve_h-10],[int((j+1)*space)+word_w,im_h-curve_h*2-10]]),
                    (200,255,200))

                #teacher curve
                if self.prob_list_teacher[j]>0 and self.prob_list_teacher[j+1]>0:
                    cv2.line(img,(int(j*space)+word_w,im_h-int(self.prob_list_teacher[j]*curve_h)-curve_h-10),(int((j+1)*space)+word_w,im_h-int(self.prob_list_teacher[j+1]*curve_h)-curve_h-10),teacher_curve_color,curve_thick)

            #the probability=0.5 line
            cv2.line(img,(word_w,im_h-int(curve_h/2)),(int(i*space)+word_w,im_h-int(curve_h/2)),student_curve_color,1)
            cv2.line(img,(word_w,im_h-int(curve_h/2)-curve_h-5),(int(i*space)+word_w,im_h-int(curve_h/2)-curve_h-5),teacher_curve_color,1)

            #text
            cv2.putText(img,'probability=0.5',(word_w,im_h-int(curve_h/2)-15),font,font_scale,student_curve_color,1)
            cv2.putText(img,'probability=0.5',(word_w,im_h-int(curve_h/2)-curve_h-15),font,font_scale,teacher_curve_color,1)
            cv2.putText(img,'teacher looking',(0,im_h-int(curve_h/2)-curve_h-15),font,font_scale,teacher_curve_color,2)
            cv2.putText(img,'at the camera',(0,im_h-int(curve_h/2)-curve_h+15),font,font_scale,teacher_curve_color,2)
            cv2.putText(img,'student looking',(0,im_h-int(curve_h/2)-15),font,font_scale,student_curve_color,2)
            cv2.putText(img,'at the camera',(0,im_h-int(curve_h/2)+15),font,font_scale,student_curve_color,2)

            self._frame_list[frame_num] = img
    
    def _dict2list(self,in_dict):
        out_list = []
        for i in range(1,len(self._frame_list.keys())+1):
            if str(i) in in_dict:
                out_list.append(in_dict[str(i)])
            else:
                out_list.append(-1)
        return out_list

    def load_focus_prob(self, student_prob_dir, teacher_prob_dir):
        #get the probability of looking at the camera from the json file
        with open(student_prob_dir,'r') as f:
            prob_dict = json.load(f)
        self.prob_list_student = self._dict2list(prob_dict)

        with open(teacher_prob_dir,'r') as f:
            prob_dict = json.load(f)
        self.prob_list_teacher = self._dict2list(prob_dict)

    def generate_output_vid(self, output_vid_path:str, fps = 30):
        """
        generate output video
        @output_vid_path: path to output video
        """
        out = cv2.VideoWriter(output_vid_path, self.fourcc, fps, (self.w, self.h))
        for img_path in self._frame_list.keys():
            frame = self._frame_list[img_path] 
            # output to video
            out.write(frame) 
        out.release()

    def empty_frames(self):
        self._frame_list = {}