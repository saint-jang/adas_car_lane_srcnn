import cv2
import numpy as np

class MyLidar():
    def __init__(self):
        self.onoff = False
        self.display = False
        self.points = None
        self.accel = None
        self.mySpeed = None
        self.gyro = None
        self.newImage = []
        self.oldImage = []
        self.dim = (400, 400)

    def InputIMU(self, accel, gyro):
        self.accel = accel
        self.gyro = gyro

    def OnOff(self):
        if self.onoff == False:
            self.onoff = True
            dim = self.dim
            img_size = (dim[0], dim[1], 3)
            self.oldImage = np.zeros((img_size), dtype=np.uint8)
            self.mySpeed = [0.0, 0.0, 0.0]
        else:
            self.onoff = False
            cv2.destroyAllWindows()
            self.oldImage = []

    def CreateImage(self):
        lidar_range = 50
        dim = self.dim
        points = np.frombuffer(self.points.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(dim) / (2.0 * lidar_range)
        lidar_data += (0.5 * dim[0], 0.5 * dim[1])
        lidar_data = np.fabs(lidar_data)  
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (dim[0], dim[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        lidar_img = cv2.flip(lidar_img, 0)
        self.newImage = lidar_img
        
    def Display(self):
        if self.onoff == True:
            self.mySpeed[0] += self.accel[0]
            if -0.1 <= self.accel[0] <= 0.1:
                self.mySpeed[0] = 0.0
            self.mySpeed[1] += self.accel[1]
            if -0.1 <= self.accel[1] <= 0.1:
                self.mySpeed[1] = 0.0
            midH, midW = self.oldImage.shape[:2]
            mid = (midW // 2, midH // 2)
            t_M = np.float32([[1, 0, self.mySpeed[1] * 0.001], [0, 1, self.mySpeed[0] * 0.001]])
            a_M = cv2.getRotationMatrix2D(mid, self.gyro[2]*0.03, scale=1.0)
            self.oldImage = cv2.warpAffine(self.oldImage, t_M, (midW, midH))
            self.oldImage = cv2.warpAffine(self.oldImage, a_M, (midW, midH))
            self.CreateImage()
            self.oldImage = cv2.addWeighted(self.oldImage, 0.99, self.newImage, 0.5, 0)
            cv2.imshow("openCV", self.oldImage)
            cv2.waitKey(1)

    def InputPoints(self, points):
        self.points = points