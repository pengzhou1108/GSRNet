from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import scipy.io as sio
import tensorflow as tf
import glob
import numpy as np
import random
import threading
import pdb
import cv2
import sys
import re
tf.app.flags.DEFINE_string('train_dir', '../../dataset/CASIA2/', 'Training image file')
tf.app.flags.DEFINE_string('train_dir2', '', 'Training label file')
tf.app.flags.DEFINE_string('dataset', 'train_all_single_seg.txt', 'Training image/label file')
tf.app.flags.DEFINE_string('mask_dir', '../../dataset/casia2_mask/', 'Training label file')
tf.app.flags.DEFINE_string('test_dir', './test', 'Testing label file')
tf.app.flags.DEFINE_string('output_directory', './train_casia2_pair_bgr_300',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 128,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 16,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 16,
                            'Number of threads to preprocess the images.')
#tf.app.flags.DEFINE_string('image_dir', '../../dataset/filter_tamper/', 'Directory of image patches')
tf.app.flags.DEFINE_string('image_dir_google', '../patch_googlenet/log/train_lr001_google_coco_dynamic_new/train/', 'Directory of image patches')
tf.app.flags.DEFINE_string('image_dir_deeplab', './output/deeplab/train/', 'Directory of image patches')


FLAGS = tf.flags.FLAGS

writer=tf.python_io.TFRecordWriter(FLAGS.output_directory+'.tfrecords')
work_dir=FLAGS.train_dir
suffix='.jpg'
image_size=300
def _is_png(filename):
    """Determine if a file contains a PNG format image.
        Args:
        filename: string, path of the image file.
        Returns:
        boolean indicating if the image is a PNG.
        """
    return '.png' in filename



def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))




def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
def _convert_to_example( image, tamper, mask, labels,shape,name):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    face_id: string, name of the face where the patch is from
    
  Returns:
    Example proto
  """

  example = tf.train.Example(features=tf.train.Features(feature={
                                                          'shape': _bytes_feature(shape),
                                                          'label': _int64_feature(labels),
                                                          'name': _bytes_feature(name),
                                                          'image': _bytes_feature(image),
                                                          'tamper': _bytes_feature(tamper),
                                                          'mask': _bytes_feature(mask)}))
  return example
def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.

  filename=filename.split(' ')[0]
  suffix={'jpg','png'}

  if os.path.isfile(FLAGS.train_dir + filename):
    context=filename.split('_')[5]
    au_name=glob.glob(FLAGS.train_dir+'Au_'+context[:3]+'_'+context[3:]+'*')[0]
    image_data = cv2.imread(au_name)[:,:,:3]
    tamper_data = cv2.imread(FLAGS.train_dir + filename)[:,:,:3]



    mask_data=cv2.imread(FLAGS.mask_dir + os.path.splitext(filename)[0]+'.png',cv2.IMREAD_UNCHANGED)
    image_data=cv2.resize(image_data,(image_size,image_size))
    tamper_data=cv2.resize(tamper_data,(image_size,image_size))
    mask_data=((cv2.resize(mask_data,(image_size,image_size))>50)*255).astype(np.uint8)

    image=image_data
    tamper = tamper_data
    # Check that image converted to RGB
    assert len(image.shape) == 3

    shape=np.array(image.shape, np.int32)

  elif os.path.isfile(FLAGS.train_dir2 + filename):
    context=filename.split('_')[5]
    au_name=glob.glob(FLAGS.train_dir+'Au_'+context[:3]+'_'+context[3:]+'*')[0]

    image_data = cv2.imread(au_name)[:,:,:3]
    tamper_data = cv2.imread(FLAGS.train_dir2 + filename)[:,:,:3]


    mask_data=cv2.imread(FLAGS.mask_dir + os.path.splitext(filename)[0]+'.png',cv2.IMREAD_UNCHANGED)
    #image_data_google=cv2.resize(image_data_google,(image_size,image_size))
    image_data=cv2.resize(image_data,(image_size,image_size))
    tamper_data=cv2.resize(tamper_data,(image_size,image_size))
    mask_data=((cv2.resize(mask_data,(image_size,image_size))>50)*255).astype(np.uint8)

    image=image_data
    tamper = tamper_data

    assert len(image.shape) == 3

    shape=np.array(image.shape, np.int32)
  else:
    print(filename)
      



  return image.tobytes(), tamper.tobytes(), mask_data.tobytes(), shape.tobytes(), filename.encode()
def _find_image_files(data_file,dataset):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the image data set resides in JPEG files located in
      the following directory structure.
        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg
      where 'dog' is the label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
  image_name=[]
  label=[]
  num=0

  with open(os.path.join(data_file,dataset)) as f:
    for line in f:
      image_name.append(line)
      label.append(int(1))
      name = (' ').join([line.split(' ')[-1].strip(),line.split(' ')[-1]])
      image_name.append(name)
      label.append(int(1))



    
  
  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.


  shuffled_index = list(range(len(image_name)))

  random.seed(12345)
  random.shuffle(shuffled_index)
  images=[image_name[i]for i in shuffled_index]
  labels=[label[i]for i in shuffled_index]
  return images,labels
def _process_image_files_batch(coder, thread_index, ranges, images,labels):
    k=0
    for i in range(len(images)):
      try:
        image_data, tamper_data, mask,shape,name=_process_image(images[i], coder)
      except Exception as e:
        print(e)
        continue

      #example = _convert_to_example(image_data,
                                  #labels[i],
                                  #height, width)

      example = _convert_to_example(image_data,
                                tamper_data,
                                mask,
                                labels[i],
                                shape,name)
      writer.write(example.SerializeToString())
    writer.close()

def _process_dataset(data_file,dataset):
  """Process a complete data set and save it as a TFRecord.
  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  #print(data_file)
  coder=ImageCoder()
  threads=[]
  ranges=[1]
  coord=tf.train.Coordinator()
  images, labels = _find_image_files(data_file,dataset)
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, images,labels)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)
  coord.join(threads)


def main(unused_argv):
    #assert not FLAGS.train_shards % FLAGS.num_threads, (
    #'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    #assert not FLAGS.test_shards % FLAGS.num_threads, (
    #'Please make the FLAGS.num_threads commensurate with '
    #'FLAGS.test_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  # Run it!
  #_process_dataset('test-patch-pairs', FLAGS.test_label,
  #                 FLAGS.test_shards)
  print('start')
  _process_dataset(FLAGS.train_dir,FLAGS.dataset)
  print('finished')
if __name__ == '__main__':
  tf.app.run()
