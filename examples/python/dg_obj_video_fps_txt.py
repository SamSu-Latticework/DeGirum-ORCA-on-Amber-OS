#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:56:07 2024

@author: sam.latticework

Function:
    Test the speed of ORCA inference, expressed in FPS
"""

import cv2
import time
import degirum as dg


def main():
    # 打開視訊流
    # cap = cv2.VideoCapture(0) # CCD
    cap = cv2.VideoCapture(video_source)
    
    # 初始化計數器
    # fps_filtered = 30
    frame_count = 0
    start_time = time.time()

    while True:
        # 讀取每個影像
        ret, frame = cap.read()
        if ret:
            # ### You must use model.predict() to get results ###
            _ = model.predict(frame)
            # 增加計數器
            frame_count += 1
            # 計算時間差
            elapsed_time = time.time() - start_time
            # 計算FPS
            fps = int(frame_count / elapsed_time)
            # === 應用低通濾波器 ===
            # fps_filtered = int(fps_filtered * 0.9 + fps * 0.1)

            # Temperature
            Temperature = model.time_stats().get('DeviceTemperature_C').max
            Frequency = model.time_stats().get('DeviceFrequency_MHz').max

            # 顯示平滑過的FPS
            # print("\r", f'frame_count: {frame_count}, FPS: {fps_filtered}',end=" ",flush=True)
            print(f'frame_count: {frame_count}, FPS: {fps}, T: {Temperature}°C, F: {Frequency}_MHz')

        else:
            # Break the loop if the end of the video is reached
            break

    
    # 釋放視訊流
    cap.release()


if __name__ == "__main__":
    # Connect to AI inference engine
    hw_location = '172.17.0.1'

    # Connect to degirum server
    zoo = dg.connect(hw_location)
    # print(zoo.list_models())
    model_name = 'NVRVehicleV8n--512x512_quant_n2x_orca1_1'

    # load dg model
    model = zoo.load_model(model_name)
    
    # Measure time, temperature, etc.
    model.measure_time = True
    model.reset_time_stats()

    # file source
    video_source = "https://raw.githubusercontent.com/DeGirum/PySDKExamples/main/images/Traffic.mp4"

    # inference
    main()

