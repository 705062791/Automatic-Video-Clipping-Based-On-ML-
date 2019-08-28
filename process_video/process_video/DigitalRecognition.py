from aip import AipOcr
import cv2
import copy
import base64
import numpy as np
import os  
def img_change_to_BASE64(img):
    image = cv2.imencode('.jpg',img)[1]
    image_code =str(base64.b64encode(image))[2:-1]
    return image_code

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def DigitalRecognition(video_file,save_file):
    app_id = '16991758'
    api_key = 'rASfoTIrrR1Zrw4X7OwGfDQ2'
    secret_key = 'W7ULWsmLQ5QCPGSfxudzKURTKmDYasAg'
    file = open(save_file,'w');

    client = AipOcr(app_id, api_key, secret_key)

    
    video_reader = cv2.VideoCapture(video_file)

    frame_num = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    heigh = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)

    for i in range(int(frame_num)):
        flag,frame = video_reader.read()

        frame_cropping = copy.deepcopy(frame[10:80,20:160])

        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

        output_1 = cv2.filter2D(frame_cropping,-1,kernel)

        for i in range(70):
            for j in range(140):
                pixal = [frame_cropping[i,j][0],frame_cropping[i,j][1],frame_cropping[i,j][2]]
                if pixal[2]>200 and pixal[1]<60 and pixal[0]<60:
                    frame_cropping[i,j] = [0,0,0]
                else:
                    frame_cropping[i,j] = [255,255,255]

        cv2.imwrite('img0.jpg', frame_cropping)
        cv2.imshow('crop window',frame_cropping)
        cv2.waitKey(10)



        img = get_file_content('img0.jpg')


        result = client.basicGeneral(img)
        print(len(result))
        if len(result) == 3:
         if result['words_result']:
             file.write(result['words_result'][0]['words']+'\r\n');
             print(result['words_result'][0]['words'])
         else:
             file.write('None\r\n');
             print('None')

    

    

    file.close()

        
        
        

if __name__ == '__main__':

    for root,dir,files in os.walk('D:/无人机拍摄/第二次剪辑/唐福梅/'):
        print(dir)
        print(files)
        print(root)
        for i in files:
            print(root+i)
            DigitalRecognition(root+i,root+dir[0]+'/'+i[0:len(i)-4]+'.txt')

        

