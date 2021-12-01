import random
import torch
import cv2 

from models import *

# font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
torch.autograd.set_detect_anomaly(True)

# import matplotlib.pyplot as plt


class ViewBox(object):

    """ 
    Used to define the point of interest in an observation image
    (x,y): position of the point on the image
    (x_min, x_max, y_min, y_max): permissible coordinates for the point. if you try to set 
                                  posiiton outisde of the limits, the position values are clmaped.
    name: name of the point
    SELF.X = WIDTH; SELF.Y = HEIGHT
    """
    
    def __init__(self, image, radius = 64):
        self.image = image
        self.obs_shp = self.image.shape
        self.color, self.thickness = (255, 0, 0), 2 # Blue color in BGR
        self.radius = radius
        self.start_point = (self.obs_shp[1] - self.radius, self.obs_shp[0] + self.radius)
        self.end_point = (self.obs_shp[1] + self.radius, self.obs_shp[0] - self.radius)     
        self.x_min, self.x_max = int(self.radius), int(self.obs_shp[1] - self.radius)
        self.y_min, self.y_max = int(self.radius), int(self.obs_shp[0] - self.radius)
        self.x, self.y = random.randint(self.x_min, self.x_max), random.randint(self.y_min, self.y_max)

    def get_position(self):
        return (self.x, self.y)

    def move_box(self, a):

        """ moves the box in a given direction based on the action value """

        if a == 0: # MOVE BOX DOWN
            self.y -= self.radius
        elif a == 1: # MOVE BOX UP
            self.y += self.radius
        elif a == 2: # MOVE BOX LEFT
            self.x += self.radius
        elif a == 3: # MOVE BOX RIGHT
            self.x -= self.radius

        self.x = self.clamp(self.x, self.x_min, self.x_max)
        self.y = self.clamp(self.y, self.y_min, self.y_max)        

        self.start_point = (self.x - self.radius, self.y + self.radius)
        self.end_point = (self.x + self.radius, self.y - self.radius)   

    def clamp(self, n, minn, maxn):
        """ Clamp box to bounds of image """
        return max(min(maxn, n), minn)

    def clip_image(self, im):
        """ Function to send back just the view_box.radius*2 x view_box.radius*2 square around the current image coordinate """
        return im[self.y - self.radius:self.y + self.radius, self.x - self.radius:self.x + self.radius, :]


