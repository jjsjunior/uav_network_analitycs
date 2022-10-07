import csv
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import  utils
import numpy as np
import  os
import tensorflow as tf
import statistics
from os.path import isfile, join

PERSON_CLASS_COCO_DS = 0

num_classes = 80
input_size_car_detect = 416
graph_car_detect = tf.Graph()

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
return_tensors = None

class MetricIndicator:

	def __init__(self):
		self.precision = 0
		self.recall = 0
		self.true_positive = 0
		self.false_positive = 0
		self.false_negative = 0
		self.labeled_samples_total = 0


	def precision_recall(self):
		return self.precision_recall_base(self.true_positive, self.false_positive, self.false_negative)

	def precision_recall_base(self, true_positive, false_positive, false_negative):
		precicion = 0
		recall = 0
		if (true_positive + false_positive)!=0:
			precicion = true_positive / (true_positive + false_positive)
		if (true_positive + false_negative) != 0:
			recall = true_positive / (true_positive + false_negative)
		return precicion, recall

	def imprimir_precision_recall(self, precision, recall, class_object):
		print('object class: %s | precision:  %.2f  | recall: %.2f ' % (class_object, precision, recall))

	def imprimir_precision_recall_all(self):
		precision, recall = self.precision_recall()
		self.imprimir_precision_recall(precision, recall, 'total')


	def imprimir_fpn(self, true_positive, false_positive, false_negative, class_object):
		print('object class: %s | true positives: %s | false positives: %s | false negatives: %s  ' % (class_object, true_positive, false_positive, false_negative))

	def imprimir_total_amostras(self):
		print('total amostras: %s | total amostras carros: %s | total amostras moto: %s  ' % (self.labeled_samples_total, self.labeled_samples_total_car, self.labeled_samples_total_moto))

	def imprimir_probabilidades(self):
		if len(self.true_positive_probability)>0 and len(self.false_positive_probability)>0:
			print('True positives:  %f media  %f std  | False positives %f media %f std ' %
				  (statistics.mean(self.true_positive_probability), statistics.stdev(self.true_positive_probability),
				   statistics.mean(self.false_positive_probability), statistics.stdev(self.false_positive_probability)))

	def imprmir_ftn_all(self):
		# self.imprimir_total_amostras()
		print('total de amostras: %i' % self.labeled_samples_total)
		self.imprimir_fpn(self.true_positive, self.false_positive, self.false_negative, 'total')



def calc_iou(gt_bbox, pred_bbox):
	'''
	This function takes the predicted bounding box and ground truth bounding box and
	return the IoU ratio
	'''
	x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
	x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
	if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
		raise AssertionError("Ground Truth Bounding Box is not correct")
	if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
		raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)
	# if the GT bbox and predcited BBox do not overlap then iou=0
	if x_bottomright_gt < x_topleft_p:
		# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
		return 0.0
	if y_bottomright_gt < y_topleft_p:  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
		return 0.0
	if x_topleft_gt > x_bottomright_p:  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
		return 0.0
	if y_topleft_gt > y_bottomright_p:  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
		return 0.0
	GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
	Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)
	x_top_left = np.max([x_topleft_gt, x_topleft_p])
	y_top_left = np.max([y_topleft_gt, y_topleft_p])
	x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
	y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
	intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)
	union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
	return intersection_area / union_area



def carregar_ground_truth(caminho_arquivo_anotacoes):
	# caminho_completo_arquivo = '/media/jones/datarec/deteccao_pessoas/5g_dataset_person_detect/annotations.xml'
	annotation_tree = ET.parse(caminho_arquivo_anotacoes)
	tag_raiz = annotation_tree.getroot()
	dict_bbox_gt = {}
	for indice_tag_image, image_tag in enumerate(tag_raiz.findall('image')):
		nome_frame = image_tag.get('name')
		boxes = image_tag.findall('box')
		lista_bbox_gt_frame = []
		for box in boxes:
			bbox_plate = [int(float(box.get('xtl'))), int(float(box.get('ytl'))), int(float(box.get('xbr'))), int(float(box.get('ybr'))), 0, 0, 0]
			top_left = float(box.get('xtl')), float(box.get('ytl'))
			bottom_right = float(box.get('xbr')), float(box.get('ybr'))
			lista_bbox_gt_frame.append((top_left[0],top_left[1], bottom_right[0], bottom_right[1]))
			# lista_bbox_gt_frame.append(bbox_plate)
			print(image_tag.tag, image_tag.attrib, box.get('label'))
		dict_bbox_gt[nome_frame] = np.array(lista_bbox_gt_frame)
	return dict_bbox_gt


def evaluate_precision_recall(ground_truth_frame: list, predictions_bbox_frame: list, iou_threshold: int = 0.6):
	matched_ground_truth_indexes = []
	true_positive = 0
	for index_pred, pred_bbox in enumerate(predictions_bbox_frame):
		for index_gt, gt_bbox in enumerate(ground_truth_frame):
			iou = calc_iou(gt_bbox, pred_bbox)
			if iou >= iou_threshold and index_gt not in matched_ground_truth_indexes:
				true_positive += 1
				matched_ground_truth_indexes.append(index_gt)
	false_negative = len(ground_truth_frame) - true_positive
	false_positive = len(predictions_bbox_frame) - true_positive
	return true_positive, false_positive, false_negative


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-cba ' ,'--caminho_base_arquivos' ,type=str, default ='/media/jones/datarec/deteccao_pessoas/5g_dataset_person_detect/rate_frames' ,help='')
	parser.add_argument('-car ', '--nome_arquivo_resultados', type=str, default='resultados.csv', help='')
	parser.add_argument('-cmy ', '--caminho_modelo_yolo', type=str, default='resultados.csv', help='')
	parser.add_argument('-caa ', '--caminho_arquivo_anotacoes', type=str, default='resultados.csv', help='')
	args = parser.parse_args()
	caminho_base_arquivos = args.caminho_base_arquivos
	nome_arquivo_resultados = args.nome_arquivo_resultados
	caminho_modelo_yolo = args.caminho_modelo_yolo
	caminho_arquivo_anotacoes = args.caminho_arquivo_anotacoes
	indicadores_validacao = MetricIndicator()
	dict_bbox_gt = carregar_ground_truth(caminho_arquivo_anotacoes)
	session_car_detection = tf.Session(graph=graph_car_detect)
	global return_tensors
	pb_car_detect_file = caminho_modelo_yolo
	return_tensors = utils.read_pb_return_tensors(graph_car_detect, pb_car_detect_file, return_elements)
	img_path_list = [f for f in os.listdir(caminho_base_arquivos) if isfile(join(caminho_base_arquivos, f))]
	resultados_list = []
	for img_path in img_path_list:
		original_image = cv2.imread(join(caminho_base_arquivos, img_path))
		original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		org_img_shape = original_image.shape[:2]
		# input_size_car_detect = 416
		input_size_car_detect = 3840
		image_data = utils.image_preporcess(np.copy(original_image), [input_size_car_detect, input_size_car_detect])
		image_data = image_data[np.newaxis, ...]
		pred_sbbox, pred_mbbox, pred_lbbox = session_car_detection.run(
			[return_tensors[1], return_tensors[2], return_tensors[3]],
			feed_dict={return_tensors[0]: image_data})
		pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
									np.reshape(pred_mbbox, (-1, 5 + num_classes)),
									np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

		bboxes_cars = utils.postprocess_boxes(pred_bbox, org_img_shape, input_size_car_detect, 0.35)
		# bboxes = utils.nms(bboxes, 0.25, method='nms')
		bboxes_cars = utils.nms_np_bboxes(bboxes_cars, 0.45, method='nms')
		bbox_red_color = (255, 0, 0)
		predictions_bbox_frame = []
		seg_threshold = 0.5
		if len(bboxes_cars) > 0:
			for indice_bbox, bbox_car in enumerate(bboxes_cars):
				# top_left_car, bottom_right_car = (int(bbox_car[0]), int(bbox_car[1])), (int(bbox_car[2]), int(bbox_car[3]))
				top_left_car, bottom_right_car = (float(bbox_car[0]), float(bbox_car[1])), (float(bbox_car[2]), float(bbox_car[3]))
				class_detect_object_coco = int(bbox_car[5])
				if class_detect_object_coco == PERSON_CLASS_COCO_DS:
					predictions_bbox_frame.append((top_left_car[0], top_left_car[1], bottom_right_car[0], bottom_right_car[1]))
					# cv2.rectangle(original_image, top_left_car, bottom_right_car, bbox_red_color, 1)  # filled

		# cv2.rectangle(image_nd, (top_left_abs[0], top_left_abs[1]), (bottom_right_abs[0], bottom_right_abs[1]),
		# 			  bbox_color_gt, thickness)  # filled
		# image = Image.fromarray(original_image)
		# image.save('/media/jones/datarec/deteccao_pessoas/amostras/amostra.jpg')
		ground_truth_frame = dict_bbox_gt[img_path.split('/')[-1]]
		true_positive_frame, false_positive_frame, false_negative_frame = evaluate_precision_recall(ground_truth_frame, predictions_bbox_frame, seg_threshold)
		print('true_positive_frame {} false_positive_frame {} false_negative_frame {}'.format(true_positive_frame, false_positive_frame, false_negative_frame))
		resultados_list.append([img_path.split('/')[-1], true_positive_frame, false_positive_frame, false_negative_frame])
		indicadores_validacao.true_positive += true_positive_frame
		indicadores_validacao.false_positive += false_positive_frame
		indicadores_validacao.false_negative += false_negative_frame
	indicadores_validacao.imprmir_ftn_all()
	indicadores_validacao.imprimir_precision_recall_all()
	with open(nome_arquivo_resultados, 'w') as resultados_file:
		csv_writer = csv.writer(resultados_file, delimiter=';')
		csv_writer.writerow(['amostra', 'true_positive_frame', 'false_positive_frame', 'false_negative_frame'])
		csv_writer.writerows(resultados_list)



# caminho_base_arquivos_raw = '/media/jones/datarec/deteccao_pessoas/5g_dataset_person_detect/images'
# main(caminho_base_arquivos_raw, 'resultados_imagem_original.csv')
# caminho_base_arquivos_raw = '/media/jones/datarec/deteccao_pessoas/5g_dataset_person_detect/loss_frames'
# main(caminho_base_arquivos_raw, 'resultados_loss_frames.csv')
if __name__ == '__main__':
	main()
