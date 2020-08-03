import cv2
import numpy as np
import glob
from tqdm import tqdm
import imutils

class Target():
    """Main class for a target
    """

    def __init__(self):
        """Init function
        """

        #class inits
        print("Initializing target")
        self.target_points = []
        self.re_scale_factor_x = 1.0
        self.re_scale_factor_y = 1.0
        self.target_rect = []
        self.red_threshold = 230
        self.target_mask = []
        self.target_image = cv2.imread("ipsctarget.jpg")
        self.target_trans_points = [[232, 33],
                                    [369, 33],
                                    [506, 206],
                                    [506, 381],
                                    [369, 556],
                                    [232, 556],
                                    [93, 383],
                                    [93, 207]]
        self.transform_matrix = []
        self.wait = False

    def callback(self, x):
        """Callback function for mouse clicks, not in use

        Args:
            x ([None]): empty variable
        """
        pass

    def click(self, event, x, y, flags, param):
        """Click function to get input from the mouse

        Args:
            event (int): numerically coded event, 4 means mouse click
            x (int): pixelwise x coordinate
            y (int): pixelwise y coordinate
        """

        #if click is detected
        if event == 4:
            #append coordinate to the list
            self.target_points.append((x,y))
            #write outs
            print("Point added")
            print("X : " + str(x))
            print("Y : " + str(y))


    def make_target(self, frame):
        """Function for creating new target. 
        User manually marks the "corners" of the IPSC style target

        Args:
            frame (ndarray): captured frame represented as a numpy array
        """

        print("Please, mark the target")

        #copy image from the frame
        image = frame.copy()

        #create window and set mouse callback
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click)

        #resize image for viewing
        #image = cv2.resize(image, None, fx = self.re_scale_factor_x, fy = self.re_scale_factor_y)

        #while user chooses points of the field
        while True:
            #draw chosen points
            if len(self.target_points)>0:
                for p in self.target_points:
                    cv2.circle(image, p, 3, (0,0,255), -1)

            #shows image
            cv2.imshow("image", image)

            #keypress detection
            key = cv2.waitKey(1) & 0xFF

            #reset points and image
            if key == ord('r'):
                self.target_points = []
                image = frame.copy()
                image = cv2.resize(image,
                                   None,
                                   fx=self.re_scale_factor_x,
                                   fy=self.re_scale_factor_y)
            #exit if done
            elif key == ord('c'):
                break

        #if there are more or equal 4 points
        if len(self.target_points) >= 8:

            #convert crop points
            self.target_points = np.asarray(np.divide(self.target_points, [self.re_scale_factor_x, self.re_scale_factor_y]), int)
            #reset all cv2 windows
            cv2.destroyAllWindows()

            #create rectangle based on chosen points
            self.target_rect = cv2.boundingRect(self.target_points)

    def make_mask(self, frame):
        """Function for creating of the target mask

        Args:
            frame (ndarray): captured frame in the form of numpy array
        """

        print("Making mask")

        print(frame.shape[:2])
        #mask_points = self.target_points-self.target_points.min(axis=0)
        target_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(target_mask, [self.target_points], -1, (255, 255, 255), -1, cv2.LINE_AA)

        #assignes the global target mask
        self.target_mask = target_mask

    def detect_shot(self, frame):
        """Function for detection of individual shots inside the target boundry

        Args:
            frame (ndarray): capured frame in the form of numpy array

        Returns:
            int: returns tuple containing coordiantes of the shot placement on the frame, 
            or none, in case shots are not detected
        """

        #if lazer light is detected
        if frame.any() > 0:

            #extract countours
            cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #if contours exists
            if len(cnts) > 0:
                for c in cnts:
                    if len(c) >= 4:
                        # compute the center of the contour
                        M = cv2.moments(c)
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]

                        #return coordinates if shot is detected
                        return (cX, cY)
            else:
                #return none if not
                return None

    def update_target(self, shot_coordinates):
        """Function for marking the detected shot on the target

        Args:
            shot_coordinates (int): shot coordinates in the form of (x,y)
        """

        print("Updating target")

        #transforming given coordinates
        dst = cv2.warpPerspective(shot_coordinates, self.transform_matrix, (300, 300))

        print("Result of target update is ")
        print(dst)

    def get_transform(self):
        """Simple function for transforming the target points between cam view and the display image
        """

        #convert points
        pts1 = np.float32(self.target_points)
        print(pts1)
        pts2 = np.float32(self.target_trans_points)
        print(pts2)

        #geting transform
        self.transform_matrix = cv2.getPerspectiveTransform(pts1,pts2)

    def inside_target(self, shot_coordinates):
        """Function that checks if the shot is inside the target

        Args:
            shot_coordinates (int): tuple containing shot coordinates

        Returns:
            bool: True if shot is inside, False if not
        """

        #checking distance
        dist = cv2.pointPolygonTest(self.target_points,
                                    (shot_coordinates[0], shot_coordinates[1]),
                                    True)

        #if distance is 0 or more, shot is inside the target
        #if dist >= 0:
        #    return True
        #else:
        #    return False

        return bool(dist >= 0)

    def run(self):
        """Main loop function
        """
        print("Running target")

        #open video capture
        cap = cv2.VideoCapture(0)

        if len(self.target_trans_points) == 0:

            self.make_target(self.target_image)

        #if no target made
        if len(self.target_points) == 0:

            #get frame
            ret, ini_frame = cap.read()

            #make target
            self.make_target(ini_frame)

            #generate transform
            #self.get_transform()       
            #make mask
            self.make_mask(ini_frame)

        running = True

        #main while loop
        while running:

            #read frame
            ret, frame = cap.read()

            #get red channel
            red_frame = np.array(frame[:,:,2])

            #threshold the frame
            red_frame[red_frame < self.red_threshold] = 0

            #apply the mask
            #red_frame[self.target_mask<255] = 0

            #get shot status
            shot_status = self.detect_shot(red_frame)

            #if shot detected
            if shot_status is not None:
                if self.wait == False:
                    #if shot is inside the target
                    if self.inside_target(shot_status):
                        print("Hit!")
                        #self.update_target(shot_status)
                    else:
                        print("Mike!")
                    self.wait = True
            else:
                self.wait = False

            #show frame
            #cv2.imshow('frame', red_frame)

            #Break is q pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = Target()
    t1.run()
    