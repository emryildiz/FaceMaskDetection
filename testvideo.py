#Paketleri ekliyoruz
import os
import cv2
import numpy as np
import tensorflow as tf
import sys


sys.path.append("..")


from utils import label_map_util
from utils import visualization_utils as vis_util

#Modelimizin kayıttaki yerini giriyoruz
MODEL_NAME = 'model'

#Test videomuzun konumunu belirtiyoruz
VIDEO_NAME = 'testvideo.mp4'

# Dosyanın bulunduğu klasörü bir değişkene alıyoruz
CWD_PATH = os.getcwd()

#Modelimizi değişkene alıyoruz
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

#Labelmap dosyamızı değişkene alıyoruz
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

#Videomuzun konumunu alıyoruz
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Sınıf sayımızı giriyoruz Maskeli-Maskesiz
NUM_CLASSES = 2



#Label mapımızı yüklüyoruz
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Modelimizi yüklüyoruz
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


#Girdi değişkenini tanımlıyoruz
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


#Çizilecek olan kutucuğun değişkenini tanımlıyoruz
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

#Skor değişkenini tanımlıyoruz
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

#sınıf "Maskeli-Makesiz" değişkenini tanımlıyoruz
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

#Kaç adet surat olduğunu belirten değişkeni tanımlıyoruz
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Videomuzu açıyoruz
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened()):

    #videodan bir kare alıyoruz
    ret, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Modelimizi videodan aldığı bir kare üzerinde uyguluyoruz
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Sonuçları frame değişkenine alıyoruz
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # Frame değişkeninimizi cv2 kütüphanesiyle gösteriyoruz
    cv2.imshow('Object detector', frame)

    # 'q' ya basarak çıkış yapıyoruz
    if cv2.waitKey(1) == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
