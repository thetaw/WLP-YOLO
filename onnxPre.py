import timeit
import numpy as np
import onnxruntime as ort
import torch
import cv2
import os
import time
import tracemalloc

def get_all_file_paths(directory):
    all_paths = []

    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 拼接文件的完整路径
            all_paths.append(os.path.join(root, file))

    return all_paths

def preict_one_img(img_path,size):
    img = cv2.imread(img_path) #读取图片
    img = cv2.resize(img, size)#调整图片尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 把图片BGR变成RGB
 
    img = np.transpose(img,(2,0,1))#调整维度将HWC - CHW
    img = np.expand_dims(img, 0) #添加一个维度 就是batch维度
    img = img.astype(np.float32)#格式转成float32
    img /= 255

    # # 开始跟踪内存分配
    # tracemalloc.start()
   #调用onnxruntime run函数进行模型推理
    start_time = timeit.default_timer()

    outputs = ort_session.run(
        ['output0'],
        {"images": img},
    )
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    # 停止跟踪并获取快照
    current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # # 关闭跟踪
    # tracemalloc.stop()
    #print(f"代码执行时间: {elapsed_time/1000}秒")
    return elapsed_time

def preict_Cam_one_img(fram,size):
    img = cv2.resize(fram, size)#调整图片尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 把图片BGR变成RGB
    img = np.transpose(img,(2,0,1))#调整维度将HWC - CHW
    img = np.expand_dims(img, 0) #添加一个维度 就是batch维度
    img = img.astype(np.float32)#格式转成float32
    img /= 255
   #调用onnxruntime run函数进行模型推理
    outputs = ort_session.run(
        ['output0'],
        {"images": img},
    )

    

    # # 转 Tensor
    # preds = torch.Tensor(outputs[0]).transpose(-1, -2)
    # max = 0
    # # print(preds[0][1][4])
    # for pre in preds[0]:
    #     if pre[4] > max:
    #         max = pre[4]
    # print(max)

    # # from ultralytics.utils import ops
    # # preds = ops.non_max_suppression(preds, conf_thres=0.6, iou_thres=0.9, nc=1)
    # print(len(preds[0]))
    # #outputs的输出类型为list类型，所以要先将list转换成numpy再转换成torch
    # outputs1 = torch.from_numpy(np.array(outputs))
    # #通过softmax进行最后分数的计算
    # outputs_softmax = torch.softmax(outputs1[0], dim=1).numpy()[:, 0].tolist()[0]
   
if __name__ == '__main__':
    #cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device= "cpu"
    roviders = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
    print(roviders)
    #onnx路径

    model_path = 'yolov8n.quant.onnx'

    

    
    #加载onnx模型
    ort_session = ort.InferenceSession(model_path, providers=roviders)
    #图片路径
    i=r'G:\V8_Proj\datasets\VisDrone\VisDrone2019-DET-train/images'
    time_avg = 0
    all_image = get_all_file_paths(i)
    index = 0
    for image_path in all_image:
        index += 1
        time_avg = time_avg + preict_one_img(image_path,(640,640))
        print("{:.2f}%".format(index/len(all_image) * 100))
    rounded_num = time_avg / len(all_image)
    
    
    print("{:.7f}".format(1 / rounded_num))
    

    # # 读取视频文件
    # cap = cv2.VideoCapture(0)
    # # 逐帧进行预测
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     #print(frame.size)
    #     start_time = timeit.default_timer()
    #     preict_Cam_one_img(frame, (640,640))
    #     end_time = timeit.default_timer()
    #     elapsed_time = end_time - start_time
    #     #cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    #     cv2.putText(frame, str(1/elapsed_time), (50, 50) , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.imshow("Predictions", frame)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # # 释放资源并关闭窗口
    # cap.release()
    # cv2.destroyAllWindows()