import tensorflow as tf
import numpy as np
import cv2
import sys
import time
import pafy

imsize = (512, 512)
pb_dir = './ssd_inception_v2_coco_2017_11_17/frozen_inference_graph.pb'
im_output = './Test/images/'
num_pred = 30
density = {'car': 2, 'bus': 4, 'truck': 4, 'person': 0.5, 'bicycle': 0.5, 'motorcycle': 1}
batch_size = 1
frames_to_skip = 0
frame_w = frame_h = 512


vid_inp_east = sys.argv[1]
vid_inp_south = sys.argv[2]
vid_inp_west = sys.argv[3]
vid_inp_north = sys.argv[4]


def draw_boxes(image_fed, bbox, classes):

	for i in range(best_boxes_roi.shape[0]):
		im = np.reshape(image_fed[i], (frame_w, frame_h, 3))

		for j in range(num_pred):
			if best_boxes_scores[i][j] > 0.15:
				x = int(best_boxes_roi[i][j][1])
				y = int(best_boxes_roi[i][j][0])
				x_max = int(best_boxes_roi[i][j][3])
				y_max = int(best_boxes_roi[i][j][2])

				cv2.rectangle(im, (x,y), (x_max,y_max), (0,255,0), 2)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(im, labels[int(classes[i][j])], (x,y), font, 1e-3*frame_h, (255,0,0), 2)
				#cv2.imshow('Output',im)
		if i == 0:
			cv2.imwrite(im_output+'east_bbox.png', im)
		if i == 1:
			cv2.imwrite(im_output+'west_bbox.png', im)
		if i == 2:
			cv2.imwrite(im_output+'north_bbox.png', im)
		if i == 3:
			cv2.imwrite(im_output+'south_bbox.png', im)


#loading the prtrained model
graph = tf.Graph()
with graph.as_default():
	with tf.io.gfile.GFile(pb_dir, 'rb') as file:
		graph_def = tf.compat.v1.GraphDef()
		graph_def.ParseFromString(file.read())
		tf.import_graph_def(graph_def, name='')

		img = graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = graph.get_tensor_by_name('detection_scores:0')
		num_detections = graph.get_tensor_by_name('num_detections:0')
		detection_classes = graph.get_tensor_by_name('detection_classes:0')
		sess = tf.compat.v1.Session(graph=graph)



labels = []
with open('./labels.txt', 'r') as file:
	for line in file.read().splitlines():
		a = line.split()#.readline()
		a = a[-1]
		#label = label.replace('\n', '')
		a = str(a)
		labels.append(a)


video_reader_east = cv2.VideoCapture(vid_inp_east)


video_reader_west = cv2.VideoCapture(vid_inp_west)


video_reader_north = cv2.VideoCapture(vid_inp_north)


video_reader_south = cv2.VideoCapture(vid_inp_south)


try:
	while True:
		density_score = [0, 0, 0, 0]
		image_bat = []

		for j in range(batch_size):
			ret, image = video_reader_east.read()
			if ret==False:
				break
			image = cv2.resize(image, imsize)
			image_bat.append(image)
			cv2.imwrite(im_output + 'east.png', image)

			ret, image = video_reader_west.read()
			image = cv2.resize(image, imsize)
			image_bat.append(image)
			cv2.imwrite(im_output + 'west.png', image)

			ret, image = video_reader_north.read()
			image = cv2.resize(image, imsize)
			image_bat.append(image)
			cv2.imwrite(im_output + 'north.png', image)

			ret, image = video_reader_south.read()
			image = cv2.resize(image, imsize)
			image_bat.append(image)
			cv2.imwrite(im_output + 'south.png', image)

		for k in range(frames_to_skip):
			video_reader_east.grab()
			video_reader_west.grab()
			video_reader_north.grab()
			video_reader_south.grab()

		image_batch = np.asarray(image_bat)
		feed_dict = {img: image_batch}
		# print('Images read\nEvaluating....')
		tick = time.time()
		y_p_boxes, y_p_scores, y_p_num_detections, y_p_classes = sess.run([detection_boxes,
																		   detection_scores,
																		   num_detections,
																		   detection_classes], feed_dict=feed_dict)
		#print('\n\n', 'Time taken: ', time.time() - tick)

		best_boxes_roi = []
		best_boxes_scores = []
		best_boxes_classes = []
		for i in range(y_p_boxes.shape[0]):
			temp = y_p_boxes[i, :num_pred] * frame_h
			best_boxes_roi.append(temp)
			best_boxes_scores.append(y_p_scores[i, :num_pred])
			best_boxes_classes.append(y_p_classes[i, :num_pred])
		best_boxes_roi = np.asarray(best_boxes_roi)
		best_boxes_scores = np.asarray(best_boxes_scores)
		best_boxes_classes = np.asarray(best_boxes_classes)

		draw_boxes(image_batch, best_boxes_roi, best_boxes_classes)

		for i in range(4):
			for j in range(num_pred):
				if (best_boxes_scores[i][j] > 0.15) and (labels[int(best_boxes_classes[i][j])] in density.keys()):
					density_score[i] += density[labels[int(best_boxes_classes[i][j])]]

		east = density_score[0]
		west = density_score[1]
		north = density_score[2]
		south = density_score[3]

		density_score[0] = east
		density_score[1] = south
		density_score[2] = west
		density_score[3] = north

		with open('val.txt', 'w') as file:
			file.write(str(density_score))

		print("Density_Score", "-------------------------", density_score)
except:
	print("Video_Ended")


video_reader_east.release()
video_reader_west.release()
video_reader_north.release()
video_reader_south.release()