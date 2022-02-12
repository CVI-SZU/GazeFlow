import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
import tqdm

from utils import *


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, required=True, help='Path to MPIIGaze orignal dir')
parser.add_argument('-s', type=str, default='mpiigaze-norm-rgb.tfrecords',
                    help='Path to save results, is a tfrecords file')
args = parser.parse_args()

full_dict = {}

face_model_path = '6 points-based face model.mat'
face_model = get_facemodel(face_model_path)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

print('start normalize data...')
for person_id in tqdm.tqdm(os.scandir(args.p)):
    calibration_fp = os.path.join(person_id.path, 'Calibration/Camera.mat')
    cameraMatrix = get_cameraMatrix(calibration_fp)
    for day_id in tqdm.tqdm(os.scandir(person_id.path)):
        if 'Calibration' not in day_id.name:
            annotation_file = os.path.join(day_id.path, 'annotation.txt')
            annotation_dict = get_annotation_dict_in_day(annotation_file)
            for img_id in range(len(annotation_dict)):
                img_name = f'{img_id + 1:04}.jpg'
                img_fp = os.path.join(day_id.path, img_name)
                annotation = annotation_dict[img_name]
                result = get_eyes_image(img_fp, annotation, cameraMatrix, face_model, clahe=clahe)
                full_dict[f'{person_id.name}/{day_id.name}/{img_name}'] = result
                
def gen():
    for key , v in full_dict.items():
        tmp = np.concatenate([v['gaze'], v['pose']], axis=-1).tolist()
        tmp.append(v['position']) 
        yield (v['img'], tmp)

tfdataset = tf.data.Dataset.from_generator(
     gen,
     (tf.uint8, tf.float32))

def image_example(image_string, label):
    image_shape = image_string.shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _bytes_feature(tf.io.serialize_tensor(label)),
        'image': _bytes_feature(tf.io.encode_jpeg(image_string).numpy()),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

print(f'start save file at {args.s}...')
with tf.io.TFRecordWriter(args.s) as writer:
    for img, label in tqdm.tqdm(tfdataset):
        tf_example = image_example(img, label)
        writer.write(tf_example.SerializeToString())

