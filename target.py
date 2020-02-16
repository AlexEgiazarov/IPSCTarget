import cv2
import numpy as np
import glob
from tqdm import tqdm
import imutils

class Target():

    '''
    Init function
    '''
    def __init__(self):
        #class inits
        print("Initializing target")
        self.target_points = []
        self.re_scale_factor_x = 1.0
        self.re_scale_factor_y = 1.0
        self.target_rect = []
        self.red_threshold = 230
        self.target_mask = []
        self.target_image = cv2.imread("ipsctarget.jpg")
        self.target_trans_points = [[232,33],[369,33],[506,206],[506,381],[369,556],[232,556],[93,383],[93,207]]
        self.transform_matrix = []
        self.wait = False

    '''
    Callback function
    '''
    def callback(self, x):
        pass


    '''
    Click function
    '''
    def click(self, event, x, y, flags, param):

        #if click is detected
        if event == 4:
            
            #append coordinate to the list
            self.target_points.append((x,y))

            #write outs
            print("Point added")
            print("X : " + str(x))
            print("Y : " + str(y))

    '''
    Make target function
    Receives the frame, where user sets target boundries
    '''
    def makeTarget(self, frame):

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
            cv2.imshow("image", image)
            
            key = cv2.waitKey(1) & 0xFF

            #reset points and image
            if key == ord('r'):
                self.target_points = []
                image = frame.copy()
                image = cv2.resize(image, None, fx = self.re_scale_factor_x, fy = self.re_scale_factor_y)
            
            #exit if done
            elif key == ord('c'):
                break
        
        #if there are more or equal 4 points 
        if len(self.target_points) >= 8:
            
            #convert crop points
            self.target_points = np.asarray(np.divide(self.target_points,[self.re_scale_factor_x,self.re_scale_factor_y]),int)

            #reset all cv2 windows
            cv2.destroyAllWindows()

            #create rectangle based on chosen points
            self.target_rect = cv2.boundingRect(self.target_points)

    '''
    Make target mask
    Makes mask based on target boundries
    '''
    def makeMask(self, frame):

        print("Making mask")

        print(frame.shape[:2])
        #mask_points = self.target_points-self.target_points.min(axis=0)
        target_mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(target_mask, [self.target_points], -1, (255,255,255), -1, cv2.LINE_AA)

        self.target_mask = target_mask

    '''
    Detect shot function
    Detects if the the laser is inside of the target boundries or not
    '''
    def detectShot(self, frame):

        if frame.any() > 0:

            cnts = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            #if contours exists
            if len(cnts) > 0:
                for c in cnts:
                    if len(c)>=4:
                        # compute the center of the contour
                        M = cv2.moments(c)
                        cX = M["m10"] / M["m00"]
                        cY = M["m01"] / M["m00"]

                        return (cX, cY)
            else:
                return None

    '''
    Update target function
    Outputs detected shot on the target image
    '''
    def update_target(self, shot_coordinates):
        print("Updating target")

        dst = cv2.warpPerspective(shot_coordinates,self.transform_matrix,(300,300))

        print("Result of target update is ")
        print(dst)
        
    '''
    Transform function
    Gets transformation between user set target and target image
    '''
    def get_transform(self):

        #convert points
        pts1 = np.float32(self.target_points)
        print(pts1)
        pts2 = np.float32(self.target_trans_points)
        print(pts2)

        #geting transform
        self.transform_matrix = cv2.getPerspectiveTransform(pts1,pts2)

    '''
    Inside target function
    Function that checks if the shot is inside the target
    '''
    def inside_target(self, shot_coordinates):

        #checking distance
        dist = cv2.pointPolygonTest(self.target_points,(shot_coordinates[0], shot_coordinates[1]),True)

        #if distance is 0 or more, shot is inside the target
        if dist >= 0:
            return True
        else:
            return False

    '''
    Main run function
    '''
    def run(self):
        print("Running target")

        #open video capture
        cap = cv2.VideoCapture(0)

        if len(self.target_trans_points) == 0:

            self.makeTarget(self.target_image)

        #if no target made
        if len(self.target_points) == 0:

            #get frame
            ret, ini_frame = cap.read()

            #make target
            self.makeTarget(ini_frame)

            #generate transform
            #self.get_transform()
            
            #make mask
            self.makeMask(ini_frame)

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
            shot_status = self.detectShot(red_frame)

            
            #if shot detected
            if shot_status is not None:
                
                if self.wait == False:
            
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