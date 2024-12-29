import cv2
import time

class FramesPerSecond:
	start_time = time.time()
	# FPS update time in seconds
	display_time = 2
	fc = 0
	FPS = 0
	
	def applyFrameRate(self, frame):
		self.fc+=1
		TIME = time.time() - self.start_time

		if (TIME) >= self.display_time:
			self.FPS = self.fc / (TIME)
			self.fc = 0
			self.start_time = time.time()

		fps_disp = "FPS: "+str(self.FPS)[:5]
		
		return cv2.putText(frame, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
