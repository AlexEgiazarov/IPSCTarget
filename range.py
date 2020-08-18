import cv2
import numpy as np
import glob
from tqdm import tqdm
import imutils
from ipsctarget import Targetipsc

class Range():

    def __init__(self):
        """Init function
        """
        #class inits
        print("Initializing target")
        self.re_scale_factor_x = 1.0
        self.re_scale_factor_y = 1.0
        self.red_threshold = 230
        self.target_points = []
        self.targets = []
        self.target_image_set = []
        self.target_show = []
        self.wait = False

    def callback(self, x):
        """Callback function

        Args:
            x (None): dummy input
        """
        pass


    def click(self, event, x, y, flags, param):
        """Mouse click function

        Args:
            event (int): signifies which mouse button is clicked (4 is LMB)
            x (int): pixelwise x coordinate of the click
            y (int): pixelwise y coordinate of the click
            flags ([type]): todo
            param ([type]): todo
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
        """Make target function

        Args:
            frame (ndarray): given frame where user sets target boundries by clicking on the corners
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
            if len(self.target_points) > 0:
                for p in self.target_points:
                    cv2.circle(image, p, 3, (0, 0, 255), -1)
            cv2.imshow("image", image)
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
            self.target_points = np.asarray(np.divide(self.target_points,
                                                      [self.re_scale_factor_x,
                                                       self.re_scale_factor_y]), int)

            #create target
            target = Targetipsc(self.target_points, len(self.targets))
            #append target to the list
            self.targets.append(target)
            #TODO #NEED TO CHANGE THIS
            self.target_image_set.append(cv2.imread("ipsctarget1.jpg"))
            #reset target points
            self.target_points = []
            #reset all cv2 windows
            cv2.destroyAllWindows()

    def detect_shot(self, frame):
        """Function for detection of shots in the frame

        Args:
            frame (ndarray): a frame where the laser is detected

        Returns:
            tuple: coordinates of the shot, or None if no shot is detected, 
            should probably change that
        """
        #checks if anything exists in a frame
        if frame.any() > 0:
            #extracts contours
            cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            #if contours exists
            if len(cnts) > 0:
                for c in cnts:
                    if len(c) >= 4:
                        # compute the center of the contour
                        M = cv2.moments(c)
                        try:
                            cX = M["m10"] / M["m00"]
                            cY = M["m01"] / M["m00"]
                        except ZeroDivisionError:
                            return None
                        return (cX, cY)
            else:
                return None

    def nothing(self, x):
        """Callback function for the trackbar, does nothing
        """
        pass

    def calibrate_red(self, cap):
        """Function for calibration of the red levels in the video stream

        Args:
            cap (cv2 video capture): capture stream from OpenCV
        """
        #creates window and trackbar
        cv2.namedWindow("Calibration")
        cv2.createTrackbar('R', 'Calibration', self.red_threshold, 255, self.nothing)
        #status flag
        calibrate = True
        while calibrate:
            #read frame
            ret, frame = cap.read()
            #get red channel
            red_frame = np.array(frame[:, :, 2])
            #threshold the frame
            red_frame[red_frame < self.red_threshold] = 0
            #get trackbar position
            self.red_threshold = cv2.getTrackbarPos('R','Calibration')
            #show frame
            cv2.imshow('Calibration', red_frame)
            #detect keypress
            keypress = cv2.waitKey(1) & 0xFF
            #Break is q pressed
            if keypress == ord('q'):
                #stop calibration
                calibrate = False
                #destroy calibration window
                cv2.destroyWindow("Calibration")

    def run(self):
        """Main loop function
        """

        print("Running target")
        #open video capture
        cap = cv2.VideoCapture(0)
        #running flag
        running = True
        #main while loop
        while running:
            #read frame
            ret, frame = cap.read()
            #check if targets exist
            if len(self.target_image_set)>0:
                #show frame
                for idx, t_i in enumerate(self.target_image_set):
                    cv2.imshow("Target - {}".format(idx), t_i)
            else:
                cv2.imshow('frame', frame)

            #if no targets made
            if len(self.targets) == 0:
                print("Please, mark the targets")

            #if targets exists
            else:
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
                    print("Shots detected!")
                    #check if it is the first detected frame
                    if self.wait == False:
                        #set hit status
                        hit_status = False
                        #for each target in list
                        for idx, t in enumerate(self.targets):
                            #check if shot is inside the target
                            if t.inside_target(shot_status):
                                print("Target [{}] HIT".format(t.get_id()))
                                #self.target_image_set[idx] = t.get_target_image()
                                relative_shot = t.update_target(shot_status)
                                cv2.circle(self.target_image_set[idx],
                                           relative_shot,
                                           3,
                                           (0, 0, 255),
                                           -1)
                                hit_status = True

                        #check if hits were registered
                        if hit_status == False:
                            #if no hits was inside the target
                            print("Mike!")

                        #Check if this is correct
                        self.wait = True
                else:
                    self.wait = False

            keypress = cv2.waitKey(1) & 0xFF
            #Break is q pressed
            if keypress == ord('q'):
                break
            #make target if t is pressed
            elif keypress == ord('t'):
                self.make_target(frame)
            #calibrate red threshold
            elif keypress == ord('c'):
                self.calibrate_red(cap)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    t1 = Range()
    t1.run()