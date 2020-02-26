import numpy as np
import cv2

class Targetipsc():
    '''
    init function
    '''
    def __init__(self, target_points, t_id):

        #assign target points
        self.t_points = target_points

        #create rectangle based on chosen points
        self.target_rect = cv2.boundingRect(target_points)

        #other variables
        self.t_trans_points = [[205,30], [385,30], [570,260], [570, 490], [390,722], [250,722], [22,490], [22,260]]
        self.t_mask = []
        self.transform_matrix = self.get_transform()

        #Loading image
        self.t_image_init = cv2.imread("ipsctarget1.jpg")
        self.t_image = self.t_image_init.copy

        #setting target id
        self.target_id = t_id

    #return id
    def get_id(self):
        return self.target_id

    '''
    Inside target function
    Function that checks if the shot is inside the target
    '''
    def inside_target(self, shot_coordinates):

        #checking distance
        dist = cv2.pointPolygonTest(self.t_points,(shot_coordinates[0], shot_coordinates[1]),True)

        #if distance is 0 or more, shot is inside the target
        if dist >= 0:
            self.update_target(shot_coordinates)
            return True
        else:
            return False

    '''
    Transform function
    Gets transformation between user set target and target image
    '''
    def get_transform(self):

        #convert points
        pts1 = np.float32(self.t_points)
        print(pts1)
        pts2 = np.float32(self.t_trans_points)
        print(pts2)

        #geting transform
        #self.transform_matrix = cv2.getPerspectiveTransform(pts1,pts2)

        return cv2.getPerspectiveTransform(pts1, pts2)

    '''
    Make target mask
    Makes mask based on target boundries
    '''
    def makeMask(self, frame):

        print("Making mask")

        print(frame.shape[:2])
        #mask_points = self.target_points-self.target_points.min(axis=0)
        target_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(target_mask, [self.t_points], -1, (255,255,255), -1, cv2.LINE_AA)

        self.target_mask = target_mask

    '''
    Update target function
    Outputs detected shot on the target image
    '''
    def update_target(self, shot_coordinates):
        print("Updating target")

        #dst = cv2.warpPerspective(shot_coordinates,self.transform_matrix,(300,300))
        relative_shot = cv2.transform(shot_coordinates, self.transform_matrix)

        cv2.circle(self.t_image, relative_shot, 3, (0,0,255), -1)


    '''
    Return image of the target
    '''
    def get_target_image(self):
        return self.t_image