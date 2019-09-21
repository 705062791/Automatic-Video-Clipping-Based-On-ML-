import cv2
import csv
import copy
import numpy as np
import random
from sklearn import tree
import joblib

def ReadData(csv_file,flag='float'):

    print('start reading {}'.format(csv_file))
    result = []
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if flag == 'float':
                result.append([float(i) for i in row])
            else:
                result.append([int(i) for i in row])

    f.close()
    print('read {} over'.format(csv_file))
    return result

def ExtractFeatureTestLandMark(video_file,landmark_num,video_feature_position_file):
    test_video_capture = cv2.VideoCapture(video_file)

    #视频信息
    frame_num = int(test_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(test_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(test_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    random_sample_position = [360,360,360,360,360]

    #random_sample_position = random.sample(list(range(350,frame_num-350)),landmark_num)
    #random_sample_position.sort()

    #读取特征点位置信息
    feature_position = ReadData(video_feature_position_file,'int')
    feature_num = int(len(feature_position)/landmark_num)
    
    frame_buffer = []
    test_video_feature = []
    for i in range(landmark_num):
        #缓存frame
        del frame_buffer
        frame_buffer = []
        test_video_capture.set(cv2.CAP_PROP_POS_FRAMES,random_sample_position[i]-350)
        single_landmark_feature = [0 for i in range(feature_num*3)]
        for n in range(750):
            
            flag,frame = test_video_capture.read()
            frame_buffer.append(frame)

        
        for j in range(feature_num):
            [frame_index_1,frame_index_2,x_1,x_2,y_1,y_2] = feature_position[i*feature_num+j]

            frame_1 = frame_buffer[frame_index_1+350]
            frame_2 = frame_buffer[frame_index_2+350]

            [b_1,g_1,r_1] = frame_1[x_1][y_1]
            [b_2,g_2,r_2] = frame_2[x_2][y_2]

            single_landmark_feature[j*3+0] = int(r_1) - int(r_2)
            single_landmark_feature[j*3+1] = int(g_1) - int(g_2)
            single_landmark_feature[j*3+2] = int(b_1) - int(b_2)

        print('get {}th feature'.format(i))
        test_video_feature.append(copy.deepcopy(np.array(single_landmark_feature)))
        del single_landmark_feature

    return [test_video_feature,random_sample_position]

def ExtractFeature(video_file,landmark_num,video_feature_position_file):
    test_video_capture = cv2.VideoCapture(video_file)

    #视频信息
    frame_num = int(test_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(test_video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(test_video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    random_sample_position = random.sample(list(range(350,frame_num-350)),landmark_num)
    random_sample_position.sort()
    #读取特征点位置信息
    feature_position = ReadData(video_feature_position_file,'int')
    feature_num = int(len(feature_position)/landmark_num)
    
    frame_buffer = []
    test_video_feature = []
    for i in range(landmark_num):
        #缓存frame
        del frame_buffer
        frame_buffer = []
        test_video_capture.set(cv2.CAP_PROP_POS_FRAMES,random_sample_position[i]-350)
        single_landmark_feature = [0 for i in range(feature_num*3)]
        for n in range(750):
            
            flag,frame = test_video_capture.read()
            frame_buffer.append(frame)

        
        for j in range(feature_num):
            [frame_index_1,frame_index_2,x_1,x_2,y_1,y_2] = feature_position[i*feature_num+j]

            frame_1 = frame_buffer[frame_index_1+350]
            frame_2 = frame_buffer[frame_index_2+350]

            [b_1,g_1,r_1] = frame_1[x_1][y_1]
            [b_2,g_2,r_2] = frame_2[x_2][y_2]

            single_landmark_feature[j*3+0] = int(r_1) - int(r_2)
            single_landmark_feature[j*3+1] = int(g_1) - int(g_2)
            single_landmark_feature[j*3+2] = int(b_1) - int(b_2)

        print('get {}th feature'.format(i))
        test_video_feature.append(copy.deepcopy(np.array(single_landmark_feature)))
        del single_landmark_feature

    return [test_video_feature,random_sample_position]
            
def RunTest(Regressor,test_video_feature):
    result = []

    for i in range(len(test_video_feature)):
        clip_data = test_video_feature[i]
        predict_result = Regressor[i].predict([clip_data])
        result.append(predict_result.tolist())

    return result



#if __name__ == '__main__':



#    video_file = 'D:/data/无人机拍摄/mp4/'
#    video_trian_feature_file = 'D:/data/训练数据/video_feature/'
#    video_feature_position_file = 'D:/data/训练数据/feature_position/feature_position.csv'
#    video_name = ['DJI_0027','DJI_0028','DJI_0034','DJI_0035','DJI_0036']
#    shift_model_file = 'D:/data/训练数据/shift_model/'
#    scale_model_file = 'D:/data/训练数据/scale_model/'
#    video_model_name = ['landmark_1.model','landmark_2.model','landmark_3.model','landmark_4.model','landmark_5.model']
    
#    #读取训练数据
#    video_trian_feature = []
    
#    for i in range(len(video_name)):
#        video_trian_feature.append(np.array(ReadData(video_trian_feature_file + video_name[i] + '.csv')))




#    #处理data
#    video_num = len(video_name)
#    landmark_num = 5
#    sample_num = int(len(video_trian_feature[0])/landmark_num)
#    feature_num = len(video_trian_feature[0][0]) - 2
    
#    train_data = []
#    for i in range(landmark_num):
#        single_landmark_feature = np.zeros((sample_num*video_num ,feature_num + 2))
#        for j in range(len(video_name)):
#            single_landmark_feature[j*sample_num:(j+1)*sample_num,:] = np.copy(video_trian_feature[j][i*sample_num:(i+1)*sample_num,:])

#        train_data.append(single_landmark_feature)
#        del single_landmark_feature

#    #针对每一个landmark进行训练

#    Regressor_shift = []
#    Regressor_scale = []

#    for i in range(landmark_num):
#        print('start training {}th landmark'.format(i))

#        data = np.copy(train_data[i]) 
#        X = data[:,0:7500]
#        Y_shift = data[:,7500:7501]
#        Y_scale = data[:,7501:7502]
   
#        clf_shift = tree.DecisionTreeRegressor(max_depth=3)
#        clf_shift = clf_shift.fit(X, Y_shift)

#        clf_scale = tree.DecisionTreeRegressor(max_depth=3)
#        clf_scale = clf_scale.fit(X, Y_scale)
        
#        #保存model
#        joblib.dump(clf_shift,shift_model_file+video_model_name[i])
#        joblib.dump(clf_scale,scale_model_file+video_model_name[i])

#        Regressor_shift.append(copy.deepcopy(clf_shift))
#        Regressor_scale.append(copy.deepcopy(clf_scale))
#        print('training {}th landmark over'.format(i))
#        del clf_shift
#        del clf_scale

#    #输入测试样本
#    [test_video_feature,random_sample_position] = ExtractFeatureTestLandMark(video_file+'DJI_0027.mp4',landmark_num,video_feature_position_file)
#    #测试
#    result_shift = RunTest(Regressor_shift,test_video_feature)
#    resutl_scale = RunTest(Regressor_scale,test_video_feature)

#    print('偏移量：  ')
#    print(result_shift)
#    print('缩放比例：')
#    print(resutl_scale)
#    print('起始位置：')
#    print(random_sample_position)


if __name__ == '__main__':
    video_file = 'D:/data/无人机拍摄/mp4/'
    video_trian_feature_file = 'D:/data/训练数据/video_feature/'
    shift_model_file = 'D:/data/训练数据/shift_model/'
    scale_model_file = 'D:/data/训练数据/scale_model/'
    video_feature_position_file = 'D:/data/训练数据/feature_position/feature_position.csv'
    video_name = ['DJI_0027','DJI_0028','DJI_0034','DJI_0035','DJI_0036']
    video_model_name = ['landmark_1.model','landmark_2.model','landmark_3.model','landmark_4.model','landmark_5.model']

    video_num = len(video_name)
    landmark_num = 5
    sample_num = 1201
    feature_num = 2500

    #读取model
    Regressor_shift = []
    Regressor_scale = []

    for i in range(landmark_num):
        Regressor_shift.append(joblib.load(shift_model_file+video_model_name[i]))
        Regressor_scale.append(joblib.load(scale_model_file+video_model_name[i]))

    #输入测试样本
    [test_video_feature,random_sample_position] = ExtractFeature(video_file+'DJI_0036.mp4',landmark_num,video_feature_position_file)
    #测试
    result_shift = RunTest(Regressor_shift,test_video_feature)
    resutl_scale = RunTest(Regressor_scale,test_video_feature)

    print('偏移量：  ')
    print(result_shift)
    print('缩放比例：')
    print(resutl_scale)
    print('起始位置：')
    print(random_sample_position)

    shifted_position = []
    for i in range(len(random_sample_position)):
        shifted_position.append(random_sample_position[i]+result_shift[i][0])
    print('偏移后位置')
    print(shifted_position)