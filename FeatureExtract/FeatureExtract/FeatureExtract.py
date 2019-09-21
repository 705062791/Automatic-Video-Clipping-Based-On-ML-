import cv2
import csv
import copy
import numpy as np
import random

def RandomBuildFeatureIndex(clip_num,single_clip_feature_num,width,height):
	#在700帧的范围里随机 n 对特征点 由于有6段所以总共随机6 * n 对特征点 
	#输出：两个帧号 两个像素点*n*6
    all_clip_feature_position = []
    for i in range(clip_num):
        single_clip_feature_list = []

        for j in range(single_clip_feature_num):
            frame_index_pair = random.sample(list(range(-350,350)),2)

            pixal_x_pair = random.sample(list(range(width)),2)
            pixal_y_pair = random.sample(list(range(height)),2)

            single_clip_feature_list.append(copy.deepcopy(frame_index_pair+pixal_x_pair+pixal_y_pair))
            
            del frame_index_pair
            del pixal_x_pair
            del pixal_y_pair

        all_clip_feature_position.append(single_clip_feature_list)
        del single_clip_feature_list

    return all_clip_feature_position

def ReadCsv(csv_file,flag):
    result = []
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if flag == 'int':
                result.append([int(i) for i in row])
            elif flag == 'float':
                result.append([float(i) for i in row])
            else:
                result.append([str(i) for i in row])


    f.close()
    return result



def SaveAsCsv(data,save_file):

    print('start writing')
    with open(save_file,'w',newline = '')as f:
      f_csv = csv.writer(f)
      f_csv.writerows(data.tolist())

    print('writing over')


def RandomBuildFeatureIndex_numpy(clip_num,single_clip_feature_num,width,height):

    all_clip_feature_position = np.zeros((single_clip_feature_num*clip_num,6),int)

    for i in range(single_clip_feature_num*clip_num):
        frame_index_pair = random.sample(list(range(-350,350)),2)
        pixal_x_pair = random.sample(list(range(height)),2)
        pixal_y_pair = random.sample(list(range(width)),2)

        all_clip_feature_position[i] = np.asarray(frame_index_pair+pixal_x_pair+pixal_y_pair)


    return all_clip_feature_position


#怎么样在保证速度的同时减少对内存的消耗呢？
def ExtractFeatureSingleVideo(video_file,landmark_info,sample_number,all_clip_feature_position,single_clip_feature_num):
    
    video_capture = cv2.VideoCapture(video_file)
    #获得视频数据
    video_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    #随机样本 均匀取值
    #样本的值统一设置为100帧
    if sample_number*1>video_frame:
        print("too much sample in a video! please reduce the number of sample")
        return 0
    #等间距取
    #sample_center_postion = list(range(0+350,video_frame-350))[::int((video_frame-700)/(sample_number-1)-1)]
    sample_center_postion = random.sample(list(range(0+350,video_frame-350)),sample_number)
    sample_center_postion.sort()

    interval = int((sample_number-1)/3)
    interval_list = []
    #预缓存间隔 将视频分成3段

    

    for i in list(range(sample_number))[interval:sample_number:interval]:
        interval_list.append([sample_center_postion[i-interval]-350,sample_center_postion[i]+350])
    
    video_frame_buffer = [[] for i in range(video_frame)]

    #all_single_clip_sample_feature = np.empty([sample_number*single_clip_feature_num,5],float)
    all_single_clip_sample_feature = np.empty([5*sample_number,3*single_clip_feature_num+2],float)



    for i in range(sample_number):
        #读取 分3段读取
        if i%(interval) == 0 and i!=(sample_number-1):
            del video_frame_buffer
            video_frame_buffer = [[] for i in range(video_frame)]
            print('delete frame buffer')
            video_capture.set(cv2.CAP_PROP_POS_FRAMES,interval_list[int(i/(interval))][0])
            for n in range(interval_list[int(i/(interval))][0],interval_list[int(i/interval)][1]):
                if n%100 == 0:
                    print('frame {} read'.format(n))
                flag,video_frame_buffer[n] = video_capture.read()

        for clip_index in range(5):   
            print('clip_index {} sample {} '.format(clip_index,i))
            for j in range(single_clip_feature_num):
               #读取feature pixal的位置
               
                  

               [frame_num_1,frame_num_2,x_1,x_2,y_1,y_2] = all_clip_feature_position[clip_index*single_clip_feature_num+j]

               #io 读取视频帧
               frame_1 = []
               frame_2 = []

               if len(video_frame_buffer[sample_center_postion[i]+frame_num_1]) == 0:
                   video_capture.set(cv2.CAP_PROP_POS_FRAMES,sample_center_postion[i]+frame_num_1)
                   flag,frame_1 = video_capture.read()
                   video_frame_buffer[sample_center_postion[i]+frame_num_1] = copy.deepcopy(frame_1)
               else:
                   frame_1 = video_frame_buffer[sample_center_postion[i]+frame_num_1]
           
               if len(video_frame_buffer[sample_center_postion[i]+frame_num_2]) == 0:
                   video_capture.set(cv2.CAP_PROP_POS_FRAMES,sample_center_postion[i]+frame_num_2)
                   flag,frame_2 = video_capture.read()
                   video_frame_buffer[sample_center_postion[i]+frame_num_2] = copy.deepcopy(frame_2)
               else:
                   frame_2 = video_frame_buffer[sample_center_postion[i]+frame_num_2]  

           
               [b_1,g_1,r_1] = frame_1[x_1][y_1]
               [b_2,g_2,r_2] = frame_2[x_2][y_2]

               
               all_single_clip_sample_feature[clip_index*sample_number+i][j*3+0] = int(r_1) - int(r_2)
               all_single_clip_sample_feature[clip_index*sample_number+i][j*3+1] = int(g_1) - int(g_2)
               all_single_clip_sample_feature[clip_index*sample_number+i][j*3+2] = int(b_1) - int(b_2)

        

    del video_frame_buffer

    for clip_index in range(5):
        for i in range(sample_number):
            landmark_position = int((int(landmark_info[clip_index][1])+int(landmark_info[clip_index][2]))/2)
            landmark_length = int(int(landmark_info[clip_index][2])-int(landmark_info[clip_index][1]))
            #shift = sample_center_postion[i] - landmark_position
            shift = landmark_position - sample_center_postion[i]
            scale = landmark_length/100
            print('shift {} scale {} index {}'.format(shift,scale,clip_index*sample_number+i))
            all_single_clip_sample_feature[clip_index*sample_number+i][3*single_clip_feature_num] = shift
            all_single_clip_sample_feature[clip_index*sample_number+i][3*single_clip_feature_num+1] = scale

    print(all_single_clip_sample_feature.shape)
    return {'clip_index':clip_index,'all_single_clip_sample_feature':all_single_clip_sample_feature}

if __name__ == '__main__':
    
    #随机特征位置

    #all_clip_feature_position = RandomBuildFeatureIndex_numpy(5,2500,1280,720)
    

    feature_position_file = 'D:/data/训练数据/feature_position/feature_position.csv'
    video_path = 'D:/data/无人机拍摄/mp4/'
    landmark_file = 'D:/data/训练数据/landmark/landmark.csv'
    video_feature_save_file = 'D:/data/训练数据/video_feature/'
    single_clip_feature_num = 2500
    sample_number = 1201

    #特征点位置
    all_clip_feature_position = ReadCsv(feature_position_file,'int')
    
    #landmark
    landmark_info = ReadCsv(landmark_file,'str')
    landmark_info_uniform = []

    temp = []
    for i in range(int(len(landmark_info)/5)):
        del temp
        temp = []
        for j in range(5):
            temp.append(landmark_info[i*5+j])

        landmark_info_uniform.append(temp)

    video_name = []

    for i in range(len(landmark_info_uniform)):
        video_name.append(landmark_info_uniform[i][0][0])


    for i in range(len(video_name)):
        ExtractFeature_result = ExtractFeatureSingleVideo(video_path+video_name[i]+'.mp4'\
                                                          ,landmark_info_uniform[i]\
                                                          ,sample_number\
                                                          ,all_clip_feature_position\
                                                          ,single_clip_feature_num)
        #保存样本数据
        SaveAsCsv(ExtractFeature_result['all_single_clip_sample_feature'],video_feature_save_file+video_name[i]+'.csv')
        #保存feature position
        #SaveAsCsv(all_clip_feature_position,'feature_position.csv')
    

