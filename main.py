############################################################
# MAIN FILE
############################################################

#import process_capture
import cv2
import numpy as np
import os
from Processing import Frame, redraw_blocks
import time
from Pathplanning import Pathplanning
from PathTracking import PathTracking
import TSP_graph as Graph
import TSP

############################################################
# INPUT                                                   ##
############################################################
imname = "frame6"                                         ##
fieldside = 1 # 0 is left, 1 is right                     ##
############################################################

if __name__ == '__main__':
    # Start timer
    print("HI")
    start = time.time()
    # Get image
    path = "images/" + imname + ".jpg"
    testframe = Frame(path)
    # Process image
    img = testframe.processed_img()
    # Save processed image
    cv2.imwrite("processing/" + imname + "_proc.png", img)
    # Get correct side data
    borders, zones, blocks, arucos, rescaleratio = testframe.get_path_data(fieldside)
    fieldimg = testframe.show_path_data(borders, zones, blocks, arucos)
    # TSP
    l = [zones[0][1],zones[1][1],zones[2][0],zones[3][0]]
    G = Graph.make_graph(blocks, l, arucos)
    tour, minCost = TSP.tsp(G)
    sortim = TSP.draw(fieldimg.copy(),blocks,arucos,tour)
    # Save sorting image
    cv2.imwrite("sorting/" + imname + "_sort.png", sortim)
    # Sorts the blocks and orders them
    blocks = TSP.TourSeq(blocks, tour)
    #Create pathplanning object
    p = Pathplanning(blocks, arucos, fieldimg.copy(), zones, borders)
    # Path tracking init
    robotyaw = arucos[0][4]
    blocks_new = blocks
    step = 1
    totallength = 0
    # Tracking loop
    for i in range(len(blocks)):
        if (i==0):
            # Init graph for A*
            graph = p.create_graph()
            processed_graph, startindex, endindex, xy_path = p.Dijkstra_A(graph, arucos[0], blocks[0])
            # X/Y array for pathtracking to block
            x = [i[0]/rescaleratio for i in xy_path]
            y = [i[1]/rescaleratio for i in xy_path]
            # Pathtracking object
            robotTrack = PathTracking(x, y, robotyaw, False)
            # Draw path and end position
            pathimage, robotyaw, pathlength = robotTrack.drawPath(fieldimg.copy(), rescaleratio)
            # Redraw block for better visibility
            redraw_blocks(pathimage, blocks_new[0:2])
            # Update pathlength
            totallength += pathlength
            # Save first step
            cv2.imwrite("pathplanning/" + imname + "_step" + str(step) + "outof10.png", pathimage)
            # Increase step
            step += 1
            # Update fieldimage
            fieldimg = testframe.show_path_data(borders, zones, blocks_new)
        else:
            # A* 
            border_start = [xy_path[-1][0], xy_path[-1][1], 0, 0, 0]
            graph = p.create_graph(arucos = False)
            processed_graph, startindex, endindex, xy_path = p.Dijkstra_A(graph, border_start, blocks[i])
            # X/Y array for pathtracking to block
            x = [i[0]/rescaleratio for i in xy_path]
            y = [i[1]/rescaleratio for i in xy_path]
            # Pathtracking object
            robotTrack = PathTracking(x, y, robotyaw, False)
            # Draw path and end position
            fullpathimage, robotyaw, pathlength = robotTrack.drawPath(pathimage, rescaleratio)
            # Redraw block for better visibility
            redraw_blocks(fullpathimage, blocks_new[0:2])
            # Update pathlength
            totallength += pathlength
            # Save step
            cv2.imwrite("pathplanning/" + imname + "_step" + str(step) + "outof10.png", fullpathimage)
            # Increase step
            step += 1
            # Update blocks for image
            blocks_new = blocks_new[1:]
            # Update fieldimage
            fieldimg = testframe.show_path_data(borders, zones, blocks_new)
        # Update blocks array: remove sorted block
        p.blocks = p.blocks[1:]
        # A*
        graph = p.create_graph(arucos = False)
        processed_graph, startindex, endindex, xy_path = p.Dijkstra_A(graph, blocks[i])
        # X/Y array for pathtracking to border
        x = [i[0]/rescaleratio for i in xy_path]
        y = [i[1]/rescaleratio for i in xy_path]
        # Pathtracking object
        robotTrack = PathTracking(x, y, robotyaw, False)
        # img = testframe.show_path_data(borders, zones, blocks_new)
        pathimage, robotyaw, pathlength = robotTrack.drawPath(fieldimg.copy(), rescaleratio)
        # Update pathlength
        totallength += pathlength
        # One last ride
        if (i == len(blocks)-1):
            # Redraw block for better visibility
            redraw_blocks(pathimage, blocks_new)
            # Save step
            cv2.imwrite("pathplanning/" + imname + "_step" + str(step) + "outof10.png", pathimage)

    # Stop timer
    end = time.time()
    processtime = end - start

    # Print process time
    print("Total path length: " + str(totallength))
    print("Total process time: " + str(processtime))
