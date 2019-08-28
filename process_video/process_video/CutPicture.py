import cv2
import copy
import os  


def CutPicture(video_file,save_dir):
    video_read = cv2.VideoCapture(video_file)

    video_frame = video_read.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(int(video_frame)):
        flag,frame = video_read.read()
        if flag:
         cut_frame = copy.deepcopy(frame[10:80,20:160])
         cv2.imwrite(save_dir+str(i)+'.jpg',cut_frame)

    print(save_dir)



if __name__ == '__main__':
    
    for root,dir,files in os.walk('D:/无人机拍摄/第二次剪辑/马铁铮/'):
        #print(dir)
        #print(files)
        #print(root)
        for i in files[5:9]:
            print(root+i)
            print(root+dir[0]+'/'+i[0:len(i)-4]+'/')
            if os.path.exists('d:/picture/'+i[0:len(i)-4]):
                CutPicture(root+i,'d:/picture/'+i[0:len(i)-4]+'/')
            else:
                os.makedirs('d:/picture/'+i[0:len(i)-4])
                CutPicture(root+i,'d:/picture/'+i[0:len(i)-4]+'/')




