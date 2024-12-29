import numpy as np
import cv2 as cv
# import backgroudSubtraction
# import histogram
class Main:
    def __init__(self):
        self.message = "Hello, World!"

    def run(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
        # Capture frame-by-frame
            ret, frame = cap.read()
        
        # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
            mean = np.mean(yuv, axis=(0,1))
            std = np.std(yuv, axis=(0,1))
            mask = (np.abs(yuv - mean) / std >= 4.5).any(axis=2)
            mask_u8 = mask.astype(np.uint8) * 255
            
            # Display the resulting frame
            cv.imshow('frame', mask_u8)
            if cv.waitKey(1) == ord('q'):
                break
        
        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()
        # histogram.run()
        # print(self.message)
        

if __name__ == '__main__':
    Main().run()



 
