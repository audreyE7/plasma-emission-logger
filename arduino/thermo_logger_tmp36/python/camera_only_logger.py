# python/camera_only_logger.py
import cv2, time, csv, numpy as np
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
cap.set(cv2.CAP_PROP_EXPOSURE,-6)
roi=None; t0=time.time()
with open("results/runs/intensity_log.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["t_s","mean_intensity","snr"])
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if roi is None:
            h,w=gray.shape; roi=(w//4,h//4, w//2,h//2)
        x,y,w_,h_=roi
        sub=gray[y:y+h_,x:x+w_]
        mean=np.mean(sub); snr=mean/np.std(sub)
        wr.writerow([time.time()-t0,mean,snr])
