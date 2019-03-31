#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

if tf.__version__ < '1.10.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

input_video = "./input_images_and_videos/first_phase_p.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')
detection_graph, category_index = backbone.set_model('rfcn_resnet101_coco_11_06_2017')
# detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03')

#object_counting_api.object_counting(input_video, detection_graph, category_index, 0) # for counting all the objects, disabled color prediction

#object_counting_api.object_counting(input_video, detection_graph, category_index, 1) # for counting all the objects, enabled color prediction


targeted_objects = "traffic light" # (for counting targeted objects) change it with your targeted objects
# targeted_objects = "my_traffic_light"
fps = 50 # change it with your input video fps
width = 1920 # change it with your input video width
height = 1080 # change it with your input vide height
is_color_recognition_enabled = 0

object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting

#object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
