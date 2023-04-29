############################################################
# PREPROCESSING
############################################################

import numpy as np
import cv2
import cv2.aruco as aruco
import os 
import pickle
import argparse

class ShapeDetector:
    def __init__(self):
        pass

        #Compute curvelength
    def perim(self, c):
        return cv2.arcLength(c, True)

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        # approximates polygonal curve c with an error of 4 percent
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.90 and ar <= 1.10 else "rectangle"
        return shape

class Angles:
    def autorotate_borders(self, width, height, angle, margin=10):
        # Determine angle quadrant
        if (90-margin <= abs(angle) <= 90+margin):
            a_angle = angle - (angle/abs(angle))*90
            a_width = height
            a_height = width
        elif 180-margin <= abs(angle) <= 180:
            a_angle = angle - (angle/abs(angle))*180
            a_width = width
            a_height = height
        else:
            a_angle = angle
            a_width = width
            a_height = height
        return a_width, a_height, a_angle

    def rotate_to_quad(self, dest, angle):
        angles = np.zeros(4)
        # Create array of possible rotation
        for i in range(4):
            angles[i] = (angle+i*90)%360-180
        # Check whether close to 180
        if ((175 <= dest <= 180) or (-180 <= dest <= -175)):
            errors = np.min(np.concatenate((np.abs(dest-angles),np.abs(-dest-angles)), axis=0), axis=0)
        else:
            errors = np.abs(np.abs(dest-angles))
            # Rotate angle to proper quadrant
        return angles[np.argmin(errors)]
    
# Inherit from ShapeDetector class -> use methods from this class
class Frame(Angles):

    # Color detection bounds (H-value)
    lower_red_1 = 0
    upper_red_1 = 25
    lower_red_2 = 155
    upper_red_2 = 180
    lower_green = 35
    upper_green = 85
    lower_blue = 95
    upper_blue = 145

    #*Color detection bounds HSV_max=(180, 255, 255)
    clower_red = np.array([0,45,50])
    cupper_red = np.array([10,255,255])
    clower_red2 = np.array([170, 30, 50])
    cupper_red2 = np.array([180, 255, 255])
    clower_blue = np.array([60, 44, 50])
    cupper_blue = np.array([128, 220, 170])
    clower_green = np.array([33, 7, 80])
    cupper_green = np.array([56, 100, 170])

    def __init__(self, imagePath, fieldIndex = 0):
        # Path and image
        self.image = cv2.imread(imagePath, 1) # Load the image with colour (1 for colour, 0 for greyscale)
        
        # Features
        self.borders, self.blocks, self.arucos = self.__init_features()

        # Processed
        self.prcssd_img = self.processed_img()

        # Playing field data
        # self.playfielddata = self.__init_playfield(fieldIndex)

    def processed_img(self):
        # Determine dimension of input image
        height, width, channels = self.image.shape # shape gives number of pixel rows, columns, channels
        # Create new image (255 to make it white)
        prcssd_img = 255*np.ones((height,width,channels), np.uint8) #np.uint8: unsigned integers from 0-255; channels = number of components to represent each pixel
        # Draw borders
        for index,border in enumerate(self.borders):
            # Draw rotated rect with minimal area around bounds
            box = cv2.boxPoints(((border[0],border[1]),(border[2],border[3]),border[4])) #finds the four vertices of a rotated rectangle
            box = np.int0(box) #convert to type int0 which is equal to int8?
            # draws contours in 'box' on 'prcssd_img' with color 0 (white)
            cv2.drawContours(prcssd_img,[box],0,(255,255,255),-1) # Playing field backgrounds can be white
            # Draw additional score areas
            if index == 0:
                miny = np.min(box[:,1])
                maxy = np.max(box[:,1])
                minx = np.min(box[:,0])
                maxx = np.max(box[:,0])
                cv2.drawContours(prcssd_img,[box],0,(0,0,0),-1) # Obtain black border lines
            elif index == 1:
                blue_box = np.array([[np.min(box[:,0]),miny],[np.max(box[:,0]),miny],[np.max(box[:,0]),0],[np.min(box[:,0]),0]])
                blue_box_2 = np.array([[np.min(box[:,0]),maxy],[np.max(box[:,0]),maxy],[np.max(box[:,0]),height],[np.min(box[:,0]),height]])
                red_box = np.array([[0,np.min(box[:,1])],[minx,np.min(box[:,1])],[minx,np.max(box[:,1])],[0,np.max(box[:,1])]])
                cv2.drawContours(prcssd_img,[blue_box],0,(255,0,0),-1)
                cv2.drawContours(prcssd_img,[blue_box_2],0,(255,0,0),-1)
                cv2.drawContours(prcssd_img,[red_box],0,(0,0,255),-1)
            elif index == 2:
                cv2.drawContours(prcssd_img,[box],0,(0,255,0),-1)
            elif index == 3:
                blue_box = np.array([[np.min(box[:,0]),miny],[np.max(box[:,0]),miny],[np.max(box[:,0]),0],[np.min(box[:,0]),0]])
                blue_box_2 = np.array([[np.min(box[:,0]),maxy],[np.max(box[:,0]),maxy],[np.max(box[:,0]),height],[np.min(box[:,0]),height]])
                red_box = np.array([[maxx,np.min(box[:,1])],[width,np.min(box[:,1])],[width,np.max(box[:,1])],[maxx,np.max(box[:,1])]])
                cv2.drawContours(prcssd_img,[blue_box],0,(255,0,0),-1)
                cv2.drawContours(prcssd_img,[blue_box_2],0,(255,0,0),-1)
                cv2.drawContours(prcssd_img,[red_box],0,(0,0,255),-1)
        for block in self.blocks:
            # Draw rotated rect with minimal area around bounds
            box = cv2.boxPoints(((block[0],block[1]),(block[2],block[3]),block[4]))
            box = np.int0(box)
            if block[5] == "R":
                cv2.drawContours(prcssd_img,[box],0,(0,0,255),-1)
            elif block[5] == "G":
                cv2.drawContours(prcssd_img,[box],0,(0,255,0),-1)
            elif block[5] == "B":
                cv2.drawContours(prcssd_img,[box],0,(255,0,0),-1)
            elif block[5] == "A":
                cv2.drawContours(prcssd_img,[box],0,(255,0,247),-1)
            else:
                cv2.drawContours(prcssd_img,[box],0,(0,0,0),2)
        for mrkr in self.arucos:
            # Draw rotated rect with minimal area around bounds
            box = cv2.boxPoints(((mrkr[0],mrkr[1]),(mrkr[2],mrkr[3]),mrkr[4]))
            box = np.int0(box)
            cv2.drawContours(prcssd_img,[box],0,(255,0,247),2)
        # Draw additional score areas
        # Get aruco markers
        aruco_corners, aruco_centers = self.__init_aruco()
        return prcssd_img

    # Find features (borders, blocks and arucos)
    def __init_features(self):
        # Initialize return lists
        borders = []
        blocks = []
        # Take a copy of the imported image and convert to grayscale
        prcssd = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        clr_prcssd = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV) # converts image to HSV
        # Blur image to reduce noise
        prcssd = cv2.medianBlur(prcssd,5) #smoothens an image using median filter; preserves edges while reducing speckle and salt & pepper noise
        # Adaptive threshold
        prcssd2 = cv2.adaptiveThreshold(prcssd, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4) # 11,5 ideal, size of pixelneighbourhood
        # Find contours, only end points of segments are left, retr_tree constructs a hierarhy of nested contours
        contours, hierarchy = cv2.findContours(prcssd2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Local list for function
        parent_index = []
        # Loop over the contours -> pretty slow but not many contours so weird flex but ok
        for index,contour in enumerate(contours):
            # Draw matching rect
            rect = cv2.minAreaRect(contour)
            center, (width, height), angle = rect
            # First look for borders (exp: 850 - 1900)
            peri = 2*width+2*height
            if (700 < peri < 2000):
                width, height, angle = self.autorotate_borders(width, height, angle)
                new_el = [center[0], center[1], width, height, angle, index] # index for childs
                borders.append(new_el)
                parent_index.append(index)
                # print(width, height)
            elif ((12.5 <= width <= 25 or 12.5 <= height <= 25) and hierarchy[0,index,3] in parent_index and width < 1.3*height and width > 0.7*height):
                # Color detection + parent addition: hierarchy[0, index, 3]
                if (Frame.lower_red_1<=clr_prcssd[int(center[1]),int(center[0]),0]<=Frame.upper_red_1 or Frame.lower_red_2<=clr_prcssd[int(center[1]),int(center[0]),0]<=Frame.upper_red_2):
                    new_el = [center[0], center[1], width, height, angle, "R", hierarchy[0,index,3]] # color + parent added
                elif (Frame.lower_green<=clr_prcssd[int(center[1]),int(center[0]),0]<=Frame.upper_green):
                    new_el = [center[0], center[1], width, height, angle, "G", hierarchy[0,index,3]] # color + parent added
                elif (Frame.lower_blue<=clr_prcssd[int(center[1]),int(center[0]),0]<=Frame.upper_blue):
                    new_el = [center[0], center[1], width, height, angle, "B", hierarchy[0,index,3]] # color + parent added
                else:
                    new_el = [center[0], center[1], width, height, angle, "N", hierarchy[0,index,3]] # no color + parent added
                blocks.append(new_el)
        # Get aruco markers
        arucos = self.__init_aruco()
        # Filter out blocks recognized as aruco
        margin = 5
        for aruco in arucos:
            for index, block in enumerate(blocks):
                if (aruco[0]-margin<=block[0]<=aruco[0]+margin and aruco[1]-margin<=block[1]<=aruco[1]+margin):
                    blocks.pop(index)
        # Additional image recognition to get all blocks
        blocks_cf = self.__init_colorfeatures(borders)
        # Filter blocks with blocks_cf
        for block_cf in blocks_cf:
            # Bool
            present = False
            # Loop over blocks
            for block in blocks:
                if (block[0]-margin<=block_cf[0]<=block[0]+margin and block[1]-margin<=block_cf[1]<=block[1]+margin):
                    present = True
            # If block is not present -> add
            if not present:
                blocks.append(block_cf)
        # Returns list of borders and blocks
        return borders, blocks, arucos
    
    def __init_colorfeatures(self, borders):
        # Initialize return lists
        blocks = []
        # Take a copy of the imported image and convert to grayscale & HSV
        prcssd = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV) # converts image to HSV

        # Adaptive threshold
        r_mask = cv2.inRange(prcssd,Frame.clower_red,Frame.cupper_red) + cv2.inRange(prcssd,Frame.clower_red2,Frame.cupper_red2)
        g_mask = cv2.inRange(prcssd,Frame.clower_green,Frame.cupper_green)
        b_mask = cv2.inRange(prcssd, Frame.clower_blue, Frame.cupper_blue)

        # Find contours, only end points of segments are left, retr_tree constructs a hierarhy of nested contours
        r_contours, r_hierarchy = cv2.findContours(r_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        g_contours, g_hierarchy = cv2.findContours(g_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        b_contours, b_hierarchy = cv2.findContours(b_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Local list for function
        for index,contour in enumerate(r_contours):
            # Draw matching rect
            r_rect = cv2.minAreaRect(contour)
            center, (width, height), angle = r_rect
            if ((12.5 <= width <= 25 or 12.5 <= height <= 25) and width < 1.3*height and width > 0.7*height):
                block = [center[0], center[1], width, height, angle, "R", "out_of_bounce"]
                for b in borders:
                    if (b[0] - b[2]/2 < center[0] < b[0] + b[2]/2) and (b[1] - b[3]/2 < center[1] < b[1] + b[3]/2):
                        block[6] = b[5] # Last element in border list is parent index
                blocks.append(block)

        for index, contour in enumerate(g_contours):
            # Draw matching rect
            g_rect = cv2.minAreaRect(contour)
            center, (width, height), angle = g_rect
            if (12.5 <= width <= 25 or 12.5 <= height <= 25) and (width < 1.3*height and width > 0.7*height):
                block = [center[0], center[1], width, height, angle, "G", "out_of_bounce"]
                for b in borders:
                    if (b[0] - b[2]/2 < center[0] < b[0] + b[2]/2) and (b[1] - b[3]/2 < center[1] < b[1] + b[3]/2):
                        block[6] = b[5] # color added
                blocks.append(block)

        for index,contour in enumerate(b_contours):
            # Draw matching rect
            b_rect = cv2.minAreaRect(contour)
            center, (width, height), angle = b_rect
            if ((12.5 <= width <= 25 or 12.5 <= height <= 25) and width < 1.3*height and width > 0.7*height):
                block = [center[0], center[1], width, height, angle, "B", "out_of_bounce"]
                for b in borders:
                    if (b[0] - b[2]/2 < center[0] < b[0] + b[2]/2) and (b[1] - b[3]/2 < center[1] < b[1] + b[3]/2):
                        block[6] = b[5] # color added
                blocks.append(block)

        return blocks

    # Find aruco markers and return information in correct format
    def __init_aruco(self):
        # Return list
        arucos = []
        # Static for aruco detection
        ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        # Convert to grayscale
        prcssd = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Search for aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(prcssd, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        # Convert list of np arrays to np array (angle verification)
        npcrnrs = np.array(corners)
        # Run over found markers
        for corner in corners:
            # Fit rectangle to corners
            aruco_rect = cv2.minAreaRect(corner)
            # Get standard format properties
            center, (width, height), angle = aruco_rect
            # Check for correct angle
            c_angle = np.arctan2(corner[0,1,1]-corner[0,2,1],corner[0,1,0]-corner[0,2,0])*180/np.pi
            # Correct angle routine
            angle = self.rotate_to_quad(c_angle, angle)
            # Append to formatted list
            arucos.append([center[0], center[1], (width+height)/2, (width+height)/2, angle, "A"])
        return arucos
    
    def get_path_data(self, fieldIndex):
        # Keep all data
        borders = []
        zones = []
        blocks = []
        arucos = []
        # Left or right playfield
        if (fieldIndex == 1):
            # Rescale ratio
            rr = ((self.borders[3][2]/120)+(self.borders[3][3]/192))/2
            # Border
            thick_approx = (self.borders[0][3]-self.borders[3][3])/2
            # Offset = x - width/2 - greenfieldwidth
            x_off = self.borders[3][0] - (self.borders[3][2] + 2*thick_approx)/2 - self.borders[2][2]
            borders.append([self.borders[3][0] - x_off, self.borders[3][1], self.borders[3][2]+2*thick_approx, self.borders[3][3]+2*thick_approx, 0])
            borders.append([self.borders[3][0] - x_off, self.borders[3][1], self.borders[3][2], self.borders[3][3], 0])
            # Zones
            zones.append([borders[0][0], (borders[0][1]-borders[0][3]/2)/2, borders[1][2], borders[0][1]-borders[0][3]/2, 0, "ZBU"])
            zones.append([borders[0][0], (borders[0][1]+borders[0][3]/2) + zones[0][3]/2, borders[1][2], borders[0][1]-borders[0][3]/2, 0, "ZBL"])
            zones.append([(borders[0][0]+borders[0][2]/2)+self.borders[2][2]/2, borders[0][1], self.borders[2][2], borders[1][3], 0, "ZR"])
            zones.append([(borders[0][0]-borders[0][2]/2)-self.borders[2][2]/2, borders[0][1], self.borders[2][2], borders[1][3], 0, "ZG"])
            # blocks
            for b in self.blocks:
                if (b[6] == self.borders[3][5]):
                    n_b = b
                    n_b[0] = n_b[0] - x_off
                    blocks.append(n_b)
            # Arucos
            for aruco in self.arucos:
                if (borders[1][0] - borders[1][2]/2 < aruco[0]-x_off < borders[1][0] + borders[1][2]/2) and (borders[1][1] - borders[1][3]/2 < aruco[1] < borders[1][1] + borders[1][3]/2):
                    n_aruco = aruco
                    n_aruco[0] = n_aruco[0] - x_off
                    arucos.append(aruco)

        else:
            # Rescale ratio
            rr = ((self.borders[1][2]/120)+(self.borders[1][3]/192))/2
            # Border
            thick_approx = (self.borders[0][3]-self.borders[1][3])/2
            borders.append([self.borders[1][0], self.borders[1][1], self.borders[1][2]+2*thick_approx, self.borders[1][3]+2*thick_approx, 0])
            borders.append([self.borders[1][0], self.borders[1][1], self.borders[1][2], self.borders[1][3], 0])
            # Zones
            zones.append([borders[0][0], (borders[0][1]-borders[0][3]/2)/2, borders[1][2], borders[0][1]-borders[0][3]/2, 0, "ZBU"])
            zones.append([borders[0][0], (borders[0][1]+borders[0][3]/2) + zones[0][3]/2, borders[1][2], borders[0][1]-borders[0][3]/2, 0, "ZBL"])
            zones.append([(borders[0][0]-borders[0][2]/2)/2, borders[0][1], borders[0][0]-borders[0][2]/2, borders[1][3], 0, "ZR"])
            zones.append([(borders[0][0]+borders[0][2]/2)+self.borders[2][2]/2, borders[0][1], self.borders[2][2], borders[1][3], 0, "ZG"])
            # blocks
            for b in self.blocks:
                if (b[6] == self.borders[1][5]):
                    blocks.append(b)
            # Arucos
            for aruco in self.arucos:
                if (borders[1][0] - borders[1][2]/2 < aruco[0] < borders[1][0] + borders[1][2]/2) and (borders[1][1] - borders[1][3]/2 < aruco[1] < borders[1][1] + borders[1][3]/2):
                    arucos.append(aruco)

        return borders, zones, blocks, arucos, rr
     
    def __show(self, img):    
        cv2.imshow('Debug', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_image(self):    
        cv2.imshow('Image', np.hstack((self.image, self.prcssd_img)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_path_data(self, borders, zones, blocks, arucos = False):
        # Determine dimension of input image
        height = int(zones[0][3] + zones[1][3] + borders[0][3])
        width  = int(zones[2][2] + zones[3][2] + borders[0][2])
        channels = 3 

        # Create new image (255 to make it white)
        prcssd_img = 255*np.ones((height,width,channels), np.uint8) #np.uint8: unsigned integers from 0-255; channels = number of components to represent each pixel

        # Draw borders
        for index, border in enumerate(borders):
            box = cv2.boxPoints(((border[0],border[1]),(border[2],border[3]),border[4])) #finds the four vertices of a rotated rectangle
            box = np.int0(box)
            if index == 0:
                cv2.drawContours(prcssd_img,[box],0, (0,0,0),-1)
            else:
                cv2.drawContours(prcssd_img,[box],0,(255,255,255),-1)

        for zone in zones:
            box = cv2.boxPoints(((zone[0],zone[1]),(zone[2],zone[3]),zone[4])) #finds the four vertices of a rotated rectangle
            box = np.int0(box)
            if zone[5] == "ZR":
                cv2.drawContours(prcssd_img,[box],0,(0,0,255),-1)
            elif zone[5] == "ZG":
                cv2.drawContours(prcssd_img,[box],0,(0,255,0),-1)
            else:
                cv2.drawContours(prcssd_img,[box],0,(255,0,0),-1)

        for block in blocks:
            box = cv2.boxPoints(((block[0],block[1]),(block[2],block[3]),block[4])) #finds the four vertices of a rotated rectangle
            box = np.int0(box)
            if block[5] == "R":
                cv2.drawContours(prcssd_img,[box],0,(0,0,255),-1)
            elif block[5] == "G":
                cv2.drawContours(prcssd_img,[box],0,(0,255,0),-1)
            elif block[5] == "B":
                cv2.drawContours(prcssd_img,[box],0,(255,0,0),-1)
            else:
                cv2.drawContours(prcssd_img,[box],0,(0,0,0),2)
        if (arucos != False):
            for aruco in arucos:
                box = cv2.boxPoints(((aruco[0],aruco[1]),(aruco[2],aruco[3]),aruco[4])) #finds the four vertices of a rotated rectangle
                box = np.int0(box)
                cv2.drawContours(prcssd_img,[box],0,(255,0,247),-1)
        #self.__show(prcssd_img)

        return prcssd_img

def redraw_blocks(image, blocks):
    for block in blocks:
        box = cv2.boxPoints(((block[0],block[1]),(block[2],block[3]),block[4])) #finds the four vertices of a rotated rectangle
        box = np.int0(box)
        if block[5] == "R":
            cv2.drawContours(image,[box],0,(0,0,255),-1)
        elif block[5] == "G":
            cv2.drawContours(image,[box],0,(0,255,0),-1)
        elif block[5] == "B":
            cv2.drawContours(image,[box],0,(255,0,0),-1)
        else:
            cv2.drawContours(image,[box],0,(0,0,0),2)

if __name__ == '__main__':
    path = "images/frame1.jpg"
    testframe = Frame(path)
    testframe.show_image()

    for i in range(1,21):
        path = "images/frame"+str(i)+".jpg"
        testframe = Frame(path)
        #testframe.show_image()

        borders, zones, blocks, arucos = testframe.get_path_data(0)
        testframe.show_path_data(borders, zones, blocks, arucos)

        print("Borders: ",borders)
        print("Zones: ",zones)
        print("Blocks: ",blocks)
        print("Arucos: ",arucos)
