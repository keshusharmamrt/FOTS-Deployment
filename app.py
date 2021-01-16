#https://github.com/Poseyy/StreamlitDemos/tree/master/Streamlit_Upload
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import numpy as np
import threading
import shutil
import pandas as pd
import os
import math
import csv
import cv2
import time
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
import random
from PIL import Image
PAGE_CONFIG = {"page_title":"FOTS:Scene Text Parsing","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

########################## CODE FOR SOME FUNCTIONS NEEDED TO CONVERT SCORE & GEO MAPS TO BOXES ############################

def sort_poly(p):
  min_axis = np.argmin(np.sum(p, axis=1))
  p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
  if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
    return p
  else:
    return p[[0, 3, 2, 1]]
def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    '''
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)

def restore_rectangle_rbox(origin, geometry):
    ''' Resotre rectangle tbox'''
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)



def generate_roiRotatePara(box, angle, expand_w = 60):
    '''Generate all ROI Parameterts'''
    p0_rect, p1_rect, p2_rect, p3_rect = box
    cxy = (p0_rect + p2_rect) / 2.
    size = np.array([np.linalg.norm(p0_rect - p1_rect), np.linalg.norm(p0_rect - p3_rect)])
    rrect = np.concatenate([cxy, size])

    box=np.array(box)

    points=np.array(box, dtype=np.int32)
    xmin=np.min(points[:,0])
    xmax=np.max(points[:,0])
    ymin=np.min(points[:,1])
    ymax=np.max(points[:,1])
    bbox = np.array([xmin, ymin, xmax, ymax])
    if np.any(bbox < -expand_w):
        return None
    
    rrect[:2] -= bbox[:2]
    rrect[:2] -= rrect[2:] / 2
    rrect[2:] += rrect[:2]

    bbox[2:] -= bbox[:2]

    rrect[::2] = np.clip(rrect[::2], 0, bbox[2])
    rrect[1::2] = np.clip(rrect[1::2], 0, bbox[3])
    rrect[2:] -= rrect[:2]
    
    return bbox.astype(np.int32), rrect.astype(np.int32), - angle

def restore_roiRotatePara(box):
    rectange, rotate_angle = sort_rectangle(box)
    return generate_roiRotatePara(rectange, rotate_angle)




def get_boxes(score_map,geo_map):
  score_map_thresh=0.5
  box_thresh=0.1 
  nms_thres=0.2
  if len(score_map.shape) == 4:
    score_map = score_map[0, :, :, 0]
    geo_map = geo_map[0, :, :, :]
  # filter the score map
  xy_text = np.argwhere(score_map > score_map_thresh)
  # sort the text boxes via the y axis
  xy_text = xy_text[np.argsort(xy_text[:, 0])]
  # restore
  text_box_restored = restore_rectangle(xy_text[:, ::-1], geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
  boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
  boxes[:, :8] = text_box_restored.reshape((-1, 8))
  boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
  boxes = nms_locality(boxes.astype(np.float64), nms_thres)
  # boxes = np.concatenate([boxes, _boxes], axis=0)

  # here we filter some low score boxes by the average score map, this is different from the orginal paper
  for i, box in enumerate(boxes):
    mask = np.zeros_like(score_map, dtype=np.uint8)
    cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32), 1)
    boxes[i, 8] = cv2.mean(score_map, mask)[0]
    if i==4:
      break
  if len(boxes)>0:
    boxes = boxes[boxes[:, 8] > box_thresh]
  boxes[:,:8:2] = np.clip(boxes[:,:8:2], 0, 512 - 1)
  boxes[:,1:8:2] = np.clip(boxes[:,1:8:2], 0, 512 - 1)  
  res = []
  result = []
  if len(boxes)>0:
    for box in boxes:
      box_ =  box[:8].reshape((4, 2))
      if np.linalg.norm(box_[0] - box_[1]) < 8 or np.linalg.norm(box_[3]-box_[0]) < 8:
        continue
      result.append(box_)
  res.append(np.array(result, np.float32))   

  box_index = []
  brotateParas = []
  filter_bsharedFeatures = []
  for i in range(len(res)):
    rotateParas = []
    rboxes=res[i]
    for j, rbox in enumerate(rboxes):
      para = restore_roiRotatePara(rbox)
      if para and min(para[1][2:]) > 8:
        rotateParas.append(para)
        box_index.append((i, j))
  return rotateParas      


#Sorting a rectangle to get all point in clockwies manner
# https://github.com/Pay20Y/FOTS_TF
# https://github.com/yu20103983/FOTS
# https://github.com/Masao-Taketani/FOTS_OCR
def sort_rectangle(poly):
    '''sort the four coordinates of the polygon, points in poly should be sorted clockwise'''
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            #this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle

CHAR_VECTOR = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÉ´-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
NUM_CLASSES = len(CHAR_VECTOR) 
char_index={}
index_char={}
for i,val in enumerate(CHAR_VECTOR):
  index_char[i+1]=val
  char_index[val]=i+1


#################################################3
## CODE TO GET ORIGNAL MODEL
@st.cache
def load_main_models():
  # This is deconv layer that we have in our text detection branch
  class Deconv(tf.keras.layers.Layer):
    def __init__(self,name="deconv"):
      super().__init__(name)
      self.inp_size=0
      self.conv=None
      self.upsample=None
      self.bn=None
    def build(self,imshape):
      self.inp_size=imshape
      self.bn=tf.keras.layers.BatchNormalization()
      self.conv=tf.keras.layers.Conv2D(filters=self.inp_size[-1]//2,kernel_size=3,padding='same',activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12),use_bias=False)
      self.upsample=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',)
    def call(self,X):
      x1=self.upsample(X)
      x1=self.conv(x1)
      x1=self.bn(x1)
      x1=tf.keras.activations.relu(x1)
      return x1
  resnet=tf.keras.applications.ResNet50(input_shape=(512,512,3),include_top=False,weights='imagenet')
  tf.keras.backend.clear_session()
  layers=resnet.layers
  x1,x2,x3,x4=None,None,None,None
  for i in range(len(layers)):
    x=layers[i]
    if x.name=='pool1_pool':
      x1=x
    if x.name=='conv3_block1_1_conv':
      x2=x
    if x.name=='conv4_block1_1_conv':
      x3=x   
    if x.name=='conv5_block3_2_conv':
      x4=x  
  #  input_1 ,conv1_relu
  d=x4.output
  d=Deconv('deconv1')(d)
  d=tf.keras.layers.add([d,x3.output])

  d=Deconv('deconv2')(d)
  d=tf.keras.layers.add([d,x2.output])

  d=Deconv('deconv3')(d)
  d=tf.keras.layers.add([d,x1.output])
  d=tf.keras.layers.BatchNormalization()(d)
  d=Deconv('deconv4')(d)
  d=Deconv('deconv5')(d)
  score=tf.keras.layers.Conv2D(1,kernel_size=3,padding='same',activation='sigmoid')(d)

  # Used this beacause sigmoid gives values in range of 0-1(as mentioned in git repository)
  geo_map=tf.keras.layers.Conv2D(4,kernel_size=3,padding='same',activation='sigmoid')(d)*512
  #Angles are assumed to be between [-45 to 45]
  angle_map=(tf.keras.layers.Conv2D(1,kernel_size=3,padding='same',activation='sigmoid')(d)-0.5)*np.pi/2
  out=tf.concat([score,geo_map,angle_map],axis=3)
  detector=tf.keras.Model(resnet.input,out,name='detector')

  for layers in resnet.layers:
    layers.trainable=False 
  detector.load_weights('detector_best.h5')
  
  
  #Text Recognition Model
  #Here I have changed the architecture a bit as mentioned in FOTS paper
  inputs = tf.keras.layers.Input(name='the_input', shape=(64,128,3), dtype='float32')  

  inner = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs) 
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='max1')(inner)  

  inner = tf.keras.layers.Conv2D(64, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='max2')(inner)  

  inner = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='max3')(inner)  

  inner = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv6')(inner)   
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name='max4')(inner)  

  inner = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', name='con7')(inner) 
  inner = tf.keras.layers.BatchNormalization()(inner)
  inner = tf.keras.layers.Activation('relu')(inner)
  inner = tf.keras.layers.Reshape(target_shape=((64,512)), name='reshape')(inner)  
  inner = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 

  out=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32,return_sequences=True,go_backwards=True))(inner)
  out=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128,return_sequences=True,go_backwards=True))(out)
  x=tf.keras.layers.Dense(100)(out)#Here we hve given 100 bcz vocab size is 99 and 1 extra is for blank symbol
  x=tf.keras.activations.softmax(x)
  recognizer=tf.keras.models.Model(inputs,x)
  
  
  
  recognizer.load_weights('recognizer_best.h5')
  size=os.path.getsize('detector_best.h5')+os.path.getsize('recognizer_best.h5')
  return detector,recognizer,size

## CODE TO GET DYNAMIC POST TRAINING QUNATIZATION MODEL
@st.cache
def load_dynamic_quantized_models():
  detector_quantized=tf.lite.Interpreter(model_path="quantized_detector_dynamic.tflite")
  detector_quantized.allocate_tensors()
  # Get input and output tensors.
  input_details_detector = detector_quantized.get_input_details()
  output_details_detector = detector_quantized.get_output_details()
  
  #recognizer
  recognizer_quantized = tf.lite.Interpreter(model_path="quantized_recognized_dynamic.tflite")
  recognizer_quantized.allocate_tensors()
  # Get input and output tensors.
  input_details_recognizer = recognizer_quantized.get_input_details()
  output_details_recognizer = recognizer_quantized.get_output_details()
  size=os.path.getsize('quantized_detector_dynamic.tflite')+os.path.getsize('quantized_recognized_dynamic.tflite')
  return (detector_quantized,input_details_detector,output_details_detector),(recognizer_quantized,input_details_recognizer,output_details_recognizer),size

## CODE TO GET FLOAT16 POST TRAINING QUANTIZATION
@st.cache
def load_float16_quantized_models():
  detector_quantized = tf.lite.Interpreter(model_path=".quantized_detector_float16.tflite")
  detector_quantized.allocate_tensors()
  # Get input and output tensors.
  input_details_detector = detector_quantized.get_input_details()
  output_details_detector = detector_quantized.get_output_details()
  
  #Recognizer
  recognizer_quantized = tf.lite.Interpreter(model_path="quantized_recognized_float16.tflite")
  recognizer_quantized.allocate_tensors()
  # Get input and output tensors.
  input_details_recognizer = recognizer_quantized.get_input_details()
  output_details_recognizer = recognizer_quantized.get_output_details()  
  size=os.path.getsize('quantized_detector_float16.tflite')+os.path.getsize('quantized_recognized_float16.tflite')
  return (detector_quantized,input_details_detector,output_details_detector),(recognizer_quantized,input_details_recognizer,output_details_recognizer),size





## MAIN FUNCTION
def main():
  st.title("FOTS: Scene Text Parsing")
  menu =["Orignal Model","Dynamic Post Training Qunatized Model","Float16 Post Training Qunatized Model"]
  choice = st.selectbox('Choose Model you want...',menu)
  uploaded_file = st.file_uploader("Choose Image for Text Parsing...", type=["jpg","jpeg","png"])
  btn=st.button('Predict')
  if uploaded_file is not None:
    img = Image.open(uploaded_file)
    image=np.array(img)
    image=cv2.resize(image,(512,512))
    ## ORIGNAL MODEL
    if choice== 'Orignal Model' and btn:
      st.subheader('Orignal Model')
      detector,recognizer,size=load_main_models()
      img=image.copy()
      start_time=time.time()
      ii=detector.predict(np.expand_dims(img,axis=0))
      score_map=ii[0][:,:,0]
      geo_map=ii[0][:,:,1:]
      for ind in [0,1,2,3,4]:
        geo_map[:,:,ind]*=score_map
      rotateParas=get_boxes(score_map,geo_map)
      txt=[]
      pts=[]
      if len(rotateParas) > 0:
        for num in range(len(rotateParas)):
          text=""
          out=rotateParas[num][0]
          crop=rotateParas[num][1]
          points=np.array([[out[0],out[1]],[out[0]+out[2],out[1]],[out[0]+out[2],out[1]+out[3]],[out[0],out[1]+out[3]]])
          angle=rotateParas[num][2] 
          img1=tf.image.crop_to_bounding_box(img,out[1],out[0],out[3],out[2])
          img2=tf.keras.preprocessing.image.random_rotation(img1,angle)
          img2=tf.image.crop_to_bounding_box(img2,crop[1],crop[0],crop[3],crop[2]).numpy()
          img2=cv2.resize(img2,(128,64))
          img2=cv2.detailEnhance(img2)
          ii=recognizer.predict(np.expand_dims(img2,axis=0))
          arr=tf.keras.backend.ctc_decode(ii,np.ones((1),'int8')*64,)
          for val in arr[0][0].numpy()[0]:
            if val==-1:
              break
            else:
              text+=index_char[val]
          txt.append(text)
          pts.append(points)
      for i in range(len(txt)):
        cv2.polylines(img,[pts[i]],isClosed=True,color=(255,255,0),thickness=2)
        cv2.putText(img,txt[i],(pts[i][0][0],pts[i][0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
      end_time=time.time()
      st.image(img)
      st.subheader("Latency="+str(end_time-start_time)+" seconds") 
      st.subheader('Model Size='+str(size/2**20)+'MB')

    ## DYNAMIC POST TRAINING QUANTIZATION  
    elif choice=='Dynamic Post Training Qunatized Model' and btn:
      st.subheader('Dynamic Post Training Qunatized Model')
      det,rec,size=load_dynamic_quantized_models()
      start_time=time.time()
      img=image.copy()
      img1=img.astype('float32')
      det[0].set_tensor(det[1][0]['index'],np.expand_dims(img1,axis=0))
      det[0].invoke()
      ii=det[0].get_tensor(det[2][0]['index'])
      score_map=ii[0][:,:,0]
      geo_map=ii[0][:,:,1:]


      for ind in [0,1,2,3,4]:
        geo_map[:,:,ind]*=score_map

      rotateParas=get_boxes(score_map,geo_map)
      txt=[]
      pts=[]
      if len(rotateParas) > 0:
        for num in range(len(rotateParas)):
          text=""
          out=rotateParas[num][0]
          crop=rotateParas[num][1]
          points=np.array([[out[0],out[1]],[out[0]+out[2],out[1]],[out[0]+out[2],out[1]+out[3]],[out[0],out[1]+out[3]]])
          angle=rotateParas[num][2] 
          img1=tf.image.crop_to_bounding_box(img,out[1],out[0],out[3],out[2])
          img2=tf.keras.preprocessing.image.random_rotation(img1,angle)
          img2=tf.image.crop_to_bounding_box(img2,crop[1],crop[0],crop[3],crop[2]).numpy()
          img2=cv2.resize(img2,(128,64))
          img2=cv2.detailEnhance(img2)
          img2=img2.astype('float32')

          rec[0].set_tensor(rec[1][0]['index'],np.expand_dims(img2,axis=0))
          rec[0].invoke()
          ii=rec[0].get_tensor(rec[2][0]['index'])
          arr=tf.keras.backend.ctc_decode(ii,np.ones((1),'int8')*64)
          for val in arr[0][0].numpy()[0]:
            if val==-1:
              break
            else:
              text+=index_char[val]
          txt.append(text)
          pts.append(points)
      for i in range(len(txt)):
        cv2.polylines(img,[pts[i]],isClosed=True,color=(255,255,0),thickness=2)
        cv2.putText(img,txt[i],(pts[i][0][0],pts[i][0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
      end_time=time.time()
      st.image(img)
      st.subheader("Latency="+str(end_time-start_time)+" seconds")
      st.subheader('Dynamic Quantized Model Size='+str(size/2**20)+'MB')
    
    ## FLOAT16 POST TRAINING QUANTIZATION
    else:
      if btn:
        st.subheader('Float16 Qunatization')
        det,rec,size=load_float16_quantized_models()
        start_time=time.time()
        img=image.copy()
        img1=img.astype('float32')
        det[0].set_tensor(det[1][0]['index'],np.expand_dims(img1,axis=0))
        det[0].invoke()
        ii=det[0].get_tensor(det[2][0]['index'])
        score_map=ii[0][:,:,0]
        geo_map=ii[0][:,:,1:]


        for ind in [0,1,2,3,4]:
          geo_map[:,:,ind]*=score_map

        rotateParas=get_boxes(score_map,geo_map)
        txt=[]
        pts=[]
        if len(rotateParas) > 0:
          for num in range(len(rotateParas)):
            text=""
            out=rotateParas[num][0]
            crop=rotateParas[num][1]
            points=np.array([[out[0],out[1]],[out[0]+out[2],out[1]],[out[0]+out[2],out[1]+out[3]],[out[0],out[1]+out[3]]])
            angle=rotateParas[num][2] 
            img1=tf.image.crop_to_bounding_box(img,out[1],out[0],out[3],out[2])
            img2=tf.keras.preprocessing.image.random_rotation(img1,angle)
            img2=tf.image.crop_to_bounding_box(img2,crop[1],crop[0],crop[3],crop[2]).numpy()
            img2=cv2.resize(img2,(128,64))
            img2=cv2.detailEnhance(img2)
            img2=img2.astype('float32')

            rec[0].set_tensor(rec[1][0]['index'],np.expand_dims(img2,axis=0))
            rec[0].invoke()
            ii=rec[0].get_tensor(rec[2][0]['index'])
            arr=tf.keras.backend.ctc_decode(ii,np.ones((1),'int8')*64)
            for val in arr[0][0].numpy()[0]:
              if val==-1:
                break
              else:
                text+=index_char[val]
            txt.append(text)
            pts.append(points)
        for i in range(len(txt)):
          cv2.polylines(img,[pts[i]],isClosed=True,color=(255,255,0),thickness=2)
          cv2.putText(img,txt[i],(pts[i][0][0],pts[i][0][1]),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
        end_time=time.time()
        st.image(img)
        st.subheader("Latency="+str(end_time-start_time)+" seconds")
        st.subheader('Float16 Quantized Model Size='+str(size/2**20)+'MB')
     
if __name__ == '__main__':
  main()