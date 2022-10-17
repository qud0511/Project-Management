# 파일명: image_classification_train_sub.py

# Imports
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import logging
import base64 
import io
from PIL import Image

logging.info(f'[hunmin log] tensorflow ver : {tf.__version__}')
logging.getLogger('PIL').setLevel(logging.WARNING)

# 사용할 gpu 번호를 적는다.
# os.environ["CUDA_VISIBLE_DEVICES"]='0'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logging.info('[hunmin log] gpu set complete')
        logging.info('[hunmin log] num of gpu: {}'.format(len(gpus)))
    
    except RuntimeError as e:
        logging.info('[hunmin log] gpu set failed')
        logging.info(e)


def exec_train(tm):
    
    logging.info('[hunmin log] the start line of the function [exec_train]')
    
    logging.info('[hunmin log] tm.train_data_path : {}'.format(tm.train_data_path))
    
    # 저장 파일 확인
    list_files_directories(tm.train_data_path)
    
    ###########################################################################
    ## 1. 데이터셋 준비(Data Setup)
    ###########################################################################
    
    my_path = os.path.join(tm.train_data_path, 'dataset') + '/'
    
    # 카테고리
    dataset=['ant','apple', 'bus', 'butterfly', 'cup', 'envelope','fish', 'giraffe', 'lightbulb','pig']
    dataset_num= len(dataset) #10

    # 경로에 있는 numpy를 load하고 dataset_numpy list에 추가한다. 
    dataset_numpy = []
    for i in range (dataset_num):
        ad = my_path + str(dataset[i]) +'.npy'
        dataset_numpy.append(np.load(ad))
   
    logging.info('[hunmin log] : (image_number, image_size)')
    
    for i in range (dataset_num):
        logging.info('[hunmin log] : {}'.format(dataset_numpy[i].shape))
    
        
    np.set_printoptions(linewidth=116)
    # dataset_numpy[5] 가 envelope numpy 이다.    
    logging.info('[hunmin log] envelope : ')
    logging.info('{}'.format(dataset_numpy[5][0]))
    
    ###########################################################################
    ## 2. 데이터 전처리(Data Preprocessing)
    ###########################################################################

    # 카테고리별로 같은 수의 이미지를 훈련시키기 위해 훈련시키고자 하는 이미지의 개수를 정해준다.
    idx = 1000
    
    # 데이터 정규화 (Normalization) & 데이터 합치기 & 레이블 생성
    # X: 입력 이미지 배열 데이터
    # Y: 정답 레이블 데이터
    # 정규화 및 정답 레이블 생성
    X = np.array([data_numpy[:idx, :]/255. for data_numpy in dataset_numpy]).astype('float32')
    X = X.reshape(-1, 28*28)
    Y = np.array([i for i in range(10) for j in range(idx)]).astype('float32')

    # 훈련 & 평가 데이터셋 생성
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    # 모델 훈련에 사용할 수 있는 형태로 변경
    # X의 값을 [samples][pixels][width][height] 형태로 reshape한다.
    X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')
    X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')
    
    # reshape된 결과 확인 및 원래 배열의 형태와 비교
    logging.info('[hunmin log] X_train : {}'.format(X_train.shape))
    logging.info('[hunmin log] X_train_cnn : {}'.format(X_train_cnn.shape))

    
    # Y의 배열에 one-hot-encoding 진행
    Y_train_cnn = utils.to_categorical(Y_train)
    Y_test_cnn = utils.to_categorical(Y_test)
    num_classes = Y_test_cnn.shape[1] # class는 총 10개이다.

    # encoding된 결과 확인 및 원래 배열의 형태와 비교
    logging.info('[hunmin log] Y_train : {}'.format(Y_train.shape))
    logging.info('[hunmin log] Y_train_cnn : {}'.format(Y_train_cnn.shape))
    logging.info('[hunmin log] class number : {}'.format(num_classes))
    
    
    
    ###########################################################################
    ## 3. 학습 모델 훈련(Train Model)
    ###########################################################################

    # 모델 구축 (Build Model)
    # 이미지 분류를 위해 아주 간단한 CNN 모델을 Keras를 이용하여 구축하고자 한다.
    
    # 단일 gpu 혹은 cpu학습
    if len(gpus) < 2:
        model = model_build_and_compile(num_classes)
    # multi-gpu
    else:
        strategy = tf.distribute.MirroredStrategy()
        logging.info('[hunmin log] gpu devices num {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = model_build_and_compile(num_classes)
    
    # 사용자 입력 파라미터
    batch_size = int(tm.param_info['batch_size'])
    epochs = int(tm.param_info['epoch'])
    
    # gpu에 따른 batch_size 설정
    batch_size = batch_size * len(gpus) if len(gpus) > 0 else batch_size

    # 모델 학습 (Train Model)
    history = model.fit(X_train_cnn, Y_train_cnn, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        validation_split=0.1, 
                        verbose=0, 
                        callbacks=[LossAndErrorPrintingCallback()]
                       )
    
    # 모델 평가 (Evaluate Model)
    loss, acc = model.evaluate(X_test_cnn, Y_test_cnn, verbose=0, callbacks=[LossAndErrorPrintingCallback()])
    
    logging.info('[hunmin log] loss : {}'.format(loss))
    logging.info('[hunmin log] acc : {}'.format(acc))
    

    ###########################################################################
    ## 플랫폼 시각화
    ###########################################################################  

    plot_metrics(tm, history, model, X_test_cnn, Y_test_cnn)

    
    
    ###########################################################################
    ## 학습 모델 저장
    ###########################################################################
    
    logging.info('[hunmin log] tm.model_path : {}'.format(tm.model_path))
    model.save(os.path.join(tm.model_path, 'cnn_model.h5'))
    
    # 저장 파일 확인
    list_files_directories(tm.model_path)
    
    logging.info('[hunmin log]  the finish line of the function [exec_train]')
    


def exec_init_svc(im):

    logging.info('[hunmin log] im.model_path : {}'.format(im.model_path))
    
    # 저장 파일 확인
    list_files_directories(im.model_path)
    
    ###########################################################################
    ## 학습 모델 준비
    ########################################################################### 
    
    # load the model
    model = load_model(os.path.join(im.model_path, 'cnn_model.h5'))
    
    return {'model' : model}



def exec_inference(df, params, batch_id):
    
    ###########################################################################
    ## 4. 추론(Inference)
    ###########################################################################
    
    logging.info('[hunmin log] the start line of the function [exec_inference]')
    
    ## 학습 모델 준비
    model = params['model']
    logging.info('[hunmin log] model.summary() :')
    model.summary(print_fn=logging.info)
    
    dataset=['ant','apple', 'bus', 'butterfly', 'cup', 'envelope','fish', 'giraffe', 'lightbulb','pig']
    
    # image preprocess
    img_base64 = df.iloc[0, 0]
    image_bytes = io.BytesIO(base64.b64decode(img_base64))
    image = Image.open(image_bytes).convert('L')
    image = image.resize((28, 28))
    image = np.invert(image).astype('float32')/255.
    image = image.reshape(-1, 28, 28 , 1)
    
    # data predict
    y_pred = model.predict(image)
    y_pred_idx=np.argmax(y_pred, axis=1)
    
    # inverse transform
    result = {'inference' : dataset[y_pred_idx[0]]}
    logging.info('[hunmin log] result : {}'.format(result))

    return result



# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))



###########################################################################
## exec_train(tm) 호출 함수 
###########################################################################

# for epoch, loss
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        #logging.info("For epoch {}, loss is {:.2f}, acc is {:.2f}.".format(batch, logs.get('loss'), logs.get('acc')))
        logging.info('[hunmin log] For epoch {}, loss is {:.2f}.'.format(batch+1, logs['loss']))

def model_build_and_compile(num_classes):
    #모델 구축
    model = keras.Sequential(
        [
            layers.Input(shape=(28,28,1)),
            layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.Dropout(0.25),
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax")
        ]
    )
    logging.info('[hunmin log] model.summary() :')
    model.summary(print_fn=logging.info)
    
    # 모델 컴파일 (Compile Model)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model
    
# 시각화
def plot_metrics(tm, history, model, x_test, y_test):
    from sklearn.metrics import confusion_matrix
    
    accuracy_list = history.history['accuracy']
    loss_list = history.history['loss']
    
    for step, (acc, loss) in enumerate(zip(accuracy_list, loss_list)):
        metric={}
        metric['accuracy'] = acc
        metric['loss'] = loss
        metric['step'] = step
        tm.save_stat_metrics(metric)

    predict_y = np.argmax(model.predict(x_test), axis = 1).tolist()
    actual_y = np.argmax(y_test, axis = 1).tolist()
    
    eval_results={}
    eval_results['predict_y'] = predict_y
    eval_results['actual_y'] = actual_y
    eval_results['accuracy'] = history.history['val_accuracy'][-1]
    eval_results['loss'] = history.history['val_loss'][-1]

    # calculate_confusion_matrix(eval_results)
    eval_results['confusion_matrix'] = confusion_matrix(actual_y, predict_y).tolist()
    tm.save_result_metrics(eval_results)
    logging.info('[hunmin log] accuracy and loss curve plot for platform')
    