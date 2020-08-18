import numpy as np
import cv2

class Targetipsc():

    def __init__(self, target_points, t_id):
        """Init function

        Args:
            target_points (list): list of points representing corners of the target
            t_id (int): target id
        """

        #assign target points
        self.t_points = target_points

        #create rectangle based on chosen points
        self.target_rect = self.get_t_rectangle(target_points)

        print("Target rectangle")
        print(self.target_rect)

        #other variables
        #self.t_trans_points = [[205,30], [385,30], [570,260], [570, 490], [390,722], [250,722], [22,490], [22,260]]
        self.t_trans_points = [[0, 0], [565, 0], [565, 713], [0, 713]]
        self.t_mask = []
        self.transform_matrix = self.get_transform()

        #Loading image
        self.t_image_init = cv2.imread("ipsctarget1.jpg")
        self.t_image = cv2.imread("ipsctarget1.jpg")

        #setting target id
        self.target_id = t_id

    def get_t_rectangle(self, t_points):
        """Function to create a target rectangle

        Args:
            t_points (list): list of points

        Returns:
            list: least bounding rectangle of the target
        """
        (x, y, w, h) = cv2.boundingRect(t_points)
        t1 = [x, y]
        t2 = [x+w, y]
        t3 = [x+w, y+h]
        t4 = [x, y+h]
        rect = [t1, t2, t3, t4]
        return rect

    def get_id(self):
        """Function for return of target ID

        Returns:
            int: target id
        """
        return self.target_id

    '''
    Inside target function
    Function that checks if the shot is inside the target
    '''
    def inside_target(self, shot_coordinates):
        """Function for checking if the shot is inside the given target

        Args:
            shot_coordinates (tuple): (xY)

        Returns:
            [type]: [description]
        """

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
        #pts1 = np.float32(self.t_points)
        pts1 = np.float32(self.target_rect)
        print(pts1)
        pts2 = np.float32(self.t_trans_points)
        print(pts2)

        #geting transform
        #self.transform_matrix = cv2.getPerspectiveTransform(pts1,pts2)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        #M = cv2.getAffineTransform(pts1, pts2)

        return M

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

        #shot_coordinates = np.float32(shot_coordinates)
        shot_coordinates = (shot_coordinates[0], shot_coordinates[1], 1)

        #calculate shot placement on target
        relative_shot = np.array(self.transform_matrix) @ np.array(shot_coordinates)

        #reassemble shot placement
        relative_shot = (int(relative_shot[0]), int(relative_shot[1]))
        
        #cv2.circle(self.t_image, relative_shot, 3, (0,0,255), -1)
        return relative_shot


    '''
    Return image of the target
    '''
    def get_target_image(self):
        return self.t_image