# UAV Network Analytics
<img src="readmeimg.png"/>
Libraries dependancies:
  <ul>
  <li>Tensorflow</li>
  <li>Numpy</li>
  <li>cv2</li>
  <li>imutils</li>
  </ul>
  
  <strong>You can run the demo by running "python3 finalPrototype.py"</strong>
  
  <p>In Yolo training folder, there are some cfg file, weights, python code we used to train our 2 yolos</p>
  <p>In CNN training folder, there is the python code we used to train our CNN for character recognition</p>
  <p>You can donwload pb files, yolo.weights and datasets here : https://drive.google.com/drive/folders/17gxw7tv7jy3KgJFhQiHX0IilYObFbIJp?usp=sharing </p>
 <p> More informations : https://medium.com/@theophilebuyssens/license-plate-recognition-using-opencv-yolo-and-keras-f5bfe03afc65 </p>    
 ### Criação de ambiente conda 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#criar ambiente python 3.7.10  

`conda create --prefix ../auv/env python=3.7.10`

#ativar ambiente 
`conda activate ../auv/env`
#instalar tensorflow 

pip install tensorflow-gpu==1.14.0

#instalar keras 

pip install keras==2.2.4 

pip install opencv-python==3.4.2.17  
pip install opencv-contrib-python==3.4.2.17  
pip install Cython --install-option="--no-cython-compile"   

#ir para a pasta do darkflow pra instala-lo:
## https://github.com/thtrieu/darkflow  
## procedimentos pra correcao de problema na instalacao local do darkflow:
####https://github.com/TheophileBuy/LicensePlateRecognition/issues/2
####Just build the Cython extensions in place. NOTE: If installing this way you will have to use ./flow in the cloned darkflow directory instead of flow as darkflow is not installed globally.
`python3 setup.py build_ext --inplace`  
####Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
`pip install -e .`   
`Install with pip globally`  
`pip install .`  

###cd /media/jones/datarec/lpr/fontes/ocr/darkflow/darkflow APENAS COMENTADO, NAO UTILIZAR    
pip install .  
pip install imutils  
pip install -Iv h5py==2.10.0  
pip install --upgrade Pillow  



### lembretes:
problema ao definir o LR. encontrada issue no proprio repo:  
Darkflow does not use the learning rate in .cfg. Use --lr instead.  
`https://github.com/thtrieu/darkflow/issues/515#issuecomment-356474112`

### comear a treinar a partir de um checkpoint
* Transformar os arquivos de ckp para o formator protobuf (.pb)

`https://github.com/thtrieu/darkflow/issues/869#issuecomment-412757194`
```
python flow.py
--model
cfg/yolo-character.cfg
--load
-1
--savepb
```
* Definir o parâmetro load no train-character.py (utiliza o ultimo checkpoint):  
`"load": -1`

## DS-xpto
* descrição dataset  




Modelo | loss   | acc   
------------ |--------| -------------   
xpto | 2.67   | 0.8517  
char_recog_ceia_2 | 0.9905 | 0.9024 
Ceia_ResNet20v1_model | 0.5065 | 0.93399  
char_recog_ceia_3 | 0.85   | 0.9123
ceia_char_recog_2_ResNet20v1_model.086 | 0.4451 | **0.94283**  
ceia_char_recog_3_ResNet29v2_model | 0.4534 | 0.9412  
ceia_char_recog_4_ResNet29v2_model | 0.3613 | 0.9409  
ceia_char_recog_5_ResNet29v2_model | 0.3486 | 0.9442
ceia_char_recog_7_gaussian_ResNet29v2_model.088.h5 | 0.29   | 0.9541