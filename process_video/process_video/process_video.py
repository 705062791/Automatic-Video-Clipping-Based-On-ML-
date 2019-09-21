import cv2
import os

def ProcessVideo(video_file,save_file):
    print('start')
    video = cv2.VideoCapture(video_file)

    #获得视频参数
    if video.isOpened():
        frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
        heigh = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = video.get(cv2.CAP_PROP_FPS)

        #保存视频
        write = cv2.VideoWriter(save_file,cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),fps,(int(width),int(heigh)))

        for i in range(int(frame_num)):
            if i%200 == 0:
                print('process {}th frame over'.format(i))

            flag,frame = video.read()

            cv2.putText(frame,str(i),(20,40),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
            
            write.write(frame)

        write.release()
        video.release()
        print('over')
        return 0
    else:
        print('路径错误！无法成功读取视频')
        write.release()
        video.release()
        return 0


if __name__ == '__main__':

    for root,dir,files in os.walk('D:/data/无人机拍摄/mp4/'):
        if root and dir and files:
            print(dir)
            print(files)
            print(root)
            for i in files:
                ProcessVideo(root+i,root+dir[0]+'/'+'Marked_'+i)

        
