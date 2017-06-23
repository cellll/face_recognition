from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import align.detect_face
import time
import cv2

class FACE:
    
    def __init__(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)
                print('Loading feature extraction model')
                facenet.load_model('/root/data/models/20170511-185253/')
                
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                
                self.classifier = os.path.expanduser('/root/data/models/cfier.pkl')
               
                with open(self.classifier, 'rb') as infile:
                    (self.model, self.classes_names) = pickle.load(infile)
            
                print('Loaded classifier model from file "%s"' % self.classifier)
        
        
    def asdf(self):
        print (self.images_placeholder)
        print (self.embeddings)
        print (self.phase_train_placeholder)
        
        
    def crop_image(self, img):
        start = time.time()

        minsize = 8 # minimum size of face
        threshold = [ 0.5, 0.5, 0.5 ]  # three steps's threshold
        factor = 0.1 # scale factor    
        #img = misc.imread(img)[:,:,0:3]
        img = img[:,:,0:3]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)

        if not bounding_boxes.size :
            #print("""empty""")
            return None
        nrof_faces = bounding_boxes.shape[0]
        
        scaled_arr=[]
        det = bounding_boxes[:,0:4]
        img_size = np.asarray(img.shape)[0:2]
        bbs = np.zeros((nrof_faces,4), dtype=np.int32)

        for i in range(nrof_faces):
          
            det[i] = np.squeeze(det[i])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[i][0]-32/2, 0)
            bb[1] = np.maximum(det[i][1]-32/2, 0)
            bb[2] = np.minimum(det[i][2]+32/2, img_size[1])
            bb[3] = np.minimum(det[i][3]+32/2, img_size[0])
            
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
            
            bbs[i] = bb
            scaled_arr.append(scaled)
            

        scaled_arr = np.asarray(scaled_arr)

        print ('crop time : {}'.format(time.time()-start))

        return scaled_arr, bbs   
        
    def getEmb(self, img):
        
        s_time = time.time()
        result = self.crop_image(img)
        if result is None:
            return "no"
        else:
            aligned_img_arr = result[0]
            self.bbox = result[1]
            #aligned_img, bb = self.crop_image(img)
        
        images = self.load_data(aligned_img_arr)
        feed_dict = {self.images_placeholder:images, self.phase_train_placeholder: False}
        
        self.emb = self.sess.run(self.embeddings, feed_dict = feed_dict)
        print ('Embeddings extraction time : {}'.format(time.time()-s_time))
        
    def load_data(self, img_arr):
        nrof_faces = img_arr.shape[0]
        
        images = np.zeros((nrof_faces,160,160,3))
        for i in range(nrof_faces):
            img = img_arr[i]
            mean = np.mean(img)
            std = np.std(img)
            std_adj = np.maximum(std, 1.0/np.sqrt(img.size))
            img = np.multiply(np.subtract(img, mean), 1/std_adj)
            images[i,:,:,:] = img
        
        return images
        
    def classification(self, img):
        sc_time=time.time()
        if self.getEmb(img) is not "no":
        
            for idx in range(len(self.emb)):
                rep = self.emb[idx].reshape(-1, 128)
                predictions = self.model.predict_proba(rep)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
		
                bl = (self.bbox[idx][0], self.bbox[idx][3])
                tr = (self.bbox[idx][2], self.bbox[idx][1])
                cv2.rectangle(img, bl, tr, color=(153,255,204), thickness=3)
		
                #print('**************%s: %.3f' % (self.classes_names[best_class_indices[0]], best_class_probabilities[0]))
                name = self.classes_names[best_class_indices[0]]
		if best_class_probabilities > 0.5:
	            cv2.putText(img, name, (self.bbox[idx][0], self.bbox[idx][3]+ 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,255), thickness=2)
        print ("elapsed time : {}".format(time.time()-sc_time))
        return img    
        
