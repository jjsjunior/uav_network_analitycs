from __future__ import print_function

import datetime

from tqdm import tqdm
##work resnet
# from tensorflow import keras
#wwwork efficientnet
import efficientnet.tfkeras
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import glob
from PIL import Image
import sys
import argparse
import matplotlib.pyplot as plt
import statistics

width = 450
height = 450
channel = 3

# melhor resultado efficientnet
# saved_models/char_recog_ceia_efficientnet_B7_model.056.h5

#
# def get_char_labels_idx_array():
#     list_labels = []
#     for idx in range(0, 34):
#         list_labels.append(get_label(idx))
#     return np.array(list_labels)


def load_data(base_directory_dataset):
    images = np.array([]).reshape(0, height, width, channel)
    directory_dataset_raw = os.path.abspath(os.path.join(base_directory_dataset,'original_frames_450'))
    filelist_raw = glob.glob(directory_dataset_raw + '/*.jpg')
    labels_0 = np.zeros(len(filelist_raw))
    images_raw = np.array([np.array(Image.open(fname)) for fname in filelist_raw])
    directory_dataset_disturb = os.path.abspath(os.path.join(base_directory_dataset, 'loss_frames_450'))
    filelist_disturb = glob.glob(directory_dataset_disturb + '/*.jpg')
    labels_1 = np.ones(len(filelist_disturb))
    images_disturb = np.array([np.array(Image.open(fname)) for fname in filelist_disturb])
    images = np.append(images, images_raw, axis=0)
    images = np.append(images, images_disturb, axis=0)
    labels = np.append(labels_0, labels_1, axis=0)
    test_set = filelist_raw
    test_set.extend(filelist_disturb)
    return images, labels, test_set



def atualizar_contagem_erros(erros_dict, classe_prevista_erro, true_class):
    if true_class in erros_dict:
        if classe_prevista_erro not in erros_dict[true_class]:
            erros_dict[true_class] = {classe_prevista_erro: 0}
        erros_dict[true_class][classe_prevista_erro] = erros_dict[true_class][classe_prevista_erro]+1
    else:
        erros_dict[true_class] = {classe_prevista_erro: 1}


def subtract_pixel_mean(images):
    x_images_mean = np.mean(images, axis=0)
    images -= x_images_mean
    return images


def get_label(class_idx : int) -> str:
    if class_idx==0:
        return 'original'
    return 'corrompida'


def get_char_labels_idx_array():
    list_labels = ['original', 'corrompida']
    return np.array(list_labels)


def main(args):
    dir_images_test = args.dir_images_test
    images, labels, test_set = load_data(dir_images_test)
    # images = subtract_pixel_mean(images)
    model = keras.models.load_model(args.model_path)
    batch_size = 64
    nrof_samples = len(images)
    qtd_steps = int(nrof_samples / batch_size)+1
    erros_true_label_dict  = {}
    erros_mismatch_label_dict  = {}
    qtd_acertos = 0
    y_true_label_array = np.zeros((nrof_samples,))
    y_predited_label_array = np.zeros((nrof_samples,))
    tempo_batch_list = []
    for i in tqdm(range(qtd_steps)):
        idx_start = i*batch_size
        idx_end = idx_start + batch_size
        idx_end = nrof_samples if nrof_samples < idx_end else idx_end
        x_batch = images[idx_start:idx_end]
        y_batch = labels[idx_start:idx_end]
        x_batch = x_batch.astype('float32') / 255
        tempo_inicio = datetime.datetime.now()
        y_predicted_batch = model.predict(x_batch)
        tempo_inferencia = (datetime.datetime.now() -tempo_inicio).total_seconds()
        tempo_batch_list.append(tempo_inferencia)
        y_predicted_classes = y_predicted_batch.argmax(axis=-1)
        y_predited_label_array[idx_start:idx_end] = y_predicted_classes
        y_true_label_array[idx_start:idx_end] = y_batch
        # y_predited_label_array[idx_start:idx_end] = get_label_from_array(y_predicted_classes)
        # y_true_label_array[idx_start:idx_end] = get_label_from_array(y_batch)
        for idx, predicted_class in enumerate(y_predicted_classes):
            if int(predicted_class)!=int(y_batch[idx]):
                print('prediction error')
                print(test_set[idx_start+idx])
                y_predicted_label = get_label(int(predicted_class))
                y_true_label = get_label(int(y_batch[idx]))
                atualizar_contagem_erros(erros_true_label_dict, y_predicted_label, y_true_label)
                atualizar_contagem_erros(erros_mismatch_label_dict, y_true_label, y_predicted_label)
            else:
                qtd_acertos +=1
    possible_labels = range(0, 2)
    possible_labels = [float(possible) for possible in possible_labels]
    labels_not_predicted = [y_pred for y_pred in y_predited_label_array if y_pred not in possible_labels]
    print('labels_not_predicted: %s' % str(labels_not_predicted))
    true_labels_not_used = [y_pred for y_pred in y_true_label_array if y_pred not in possible_labels]
    print('true_labels_not_used: %s' % str(true_labels_not_used))
    name_labels_array = get_char_labels_idx_array()
    # char_confusion_matrix = confusion_matrix(y_true_label_array, y_predited_label_array,labels=name_labels_array)
    char_confusion_matrix = confusion_matrix(y_true_label_array, y_predited_label_array, labels = possible_labels)
    disp_conf_matrix = ConfusionMatrixDisplay(confusion_matrix=char_confusion_matrix, display_labels=name_labels_array)
    print('total amostras: %i | acc: %f | acertos: %i' % (nrof_samples, qtd_acertos/nrof_samples,  qtd_acertos))
    print('True labels errors')
    print_errors(erros_true_label_dict)
    print('False labels errors')
    print('estatistica de tempo. total: {} | media: {} | maximo: {} | minimo: {}'.format(
        sum(tempo_batch_list), statistics.mean(tempo_batch_list), max(tempo_batch_list), min(tempo_batch_list)))
    print_errors(erros_mismatch_label_dict)
    disp_conf_matrix.plot()
    plt.show()
    print('xpto')



def print_errors(errors_dict):
    for key in errors_dict.keys():
        print('true label: %s ' % key)
        list_erros_true_label = list(errors_dict[key].items())
        list_erros_true_label.sort(key=lambda x: x[1])
        for key_predited, value_predited in list_erros_true_label:
            print('predicted label: %s | qtd: %i' % (key_predited, value_predited))







def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_test', type=str)
    parser.add_argument('--model_path', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))