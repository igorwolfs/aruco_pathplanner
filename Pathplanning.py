############################################################
# IMPLEMENTATION OF Pathplanning - INTERMEDIATE LAYER
############################################################

from Processing import Frame
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time

import cv2
import numpy as np
import os
from Processing import Frame, redraw_blocks
import time
import TSP_graph as Graph
import TSP


class Pathplanning:

# field sampling & safety parameters
    x_sample = 15 # 15
    y_sample = 15 # 15
    d_secure = 30 # 25

    def __init__(self, blocks, arucos, image, zones, borders):
        self.blocks = blocks
        self.image = np.array(image)
        self.arucos = np.append(np.array(arucos[0]), 0)
        self.zones = zones
        self.right_side = False
        for i in range(len(self.zones)): # determine wether the graph is drawn on the left or the right side
            if (((self.zones[i][5] == 'ZR') * (self.zones[i][0] > 100)) or ((self.zones[i][5] =='ZG') * (self.zones[i][0] < 100))):
                self.right_side = True
        self.borders = borders

# Creates a graph within the provided borders

    def create_graph(self, blocks_n_i = True, arucos = True):
        rows, columns, dim = self.image.shape
        if (blocks_n_i == True):
            blocks = self.blocks

        else:
            blocks_n_i = np.array(blocks_n_i)
            blocks = self.blocks
            for block_n in blocks_n_i:
                blocks = [i for i in blocks if not (i == block_n).all()]
        if (arucos == True):
            arucos = self.arucos
        elif (arucos == False):
            arucos = [0, 0, 0, 0, 0, 0, 0]

        # creates the nodes
        sampled_rows = np.arange(self.borders[0][1]-self.borders[0][3]/2,self.borders[0][1]+self.borders[0][3]/2, Pathplanning.x_sample, dtype = int)
        sampled_columns = np.arange(self.borders[0][0]-self.borders[0][2]/2,self.borders[0][0]+self.borders[0][2]/2, Pathplanning.y_sample, dtype = int)
        
        if (blocks == []):
            squares = [[0, 0, 0, 0, 0, 0]]
        else:
            squares = np.append(blocks, [arucos], axis = 0) 

        # safety perimeter creation
        amount_of_squares = len(squares)
        arr_d_secure = Pathplanning.d_secure*np.ones(amount_of_squares)

        # makes the graph 
        columns = ['free', 'X', 'Y', 'distance_to_start', 'sourceX', 'sourceY', 'distance_to_end', 'distance_total']
        vertices_df = pd.DataFrame(index = [], columns=columns)

        squares = np.array(squares)
        x_blocks = squares[:, 0].astype(float)
        y_blocks = squares[:, 1].astype(float)

        for row in sampled_rows:
            for column in sampled_columns:
                
                # Hier check je of je array niet te dicht ligt bij de blokjes of de aruco's
                # Je zorgt dat de afstand tussen de blokjes groter is dan arr_d_secure
                
                x_array = column*np.ones(amount_of_squares)
                y_array = row*np.ones(amount_of_squares)
                d_blocks = ((x_blocks-x_array) ** 2 + (y_blocks-y_array) ** 2) ** (1/2)
                print(d_blocks)
                if ( np.all(d_blocks > arr_d_secure)):
                    vertex = pd.DataFrame({'free': [True], 'X': [column], 'Y': [row], 'distance_to_start': [9999], 'sourceX': [-1], 'sourceY': [-1], 'distance_to_end': [9999], 'distance_total': [2*9999]})
                    vertices_df = vertices_df.append(vertex, ignore_index=True)
        return vertices_df

#* Past het A* algoritme toe
    def Dijkstra_A(self, vertices_df, startlocation, endlocation = False):
        start = time.time()
        
        # Definieer het centrum van de startlocatie
        
        start_centre = np.array([startlocation[0], startlocation[1]])

        nodes = np.array([])
        
        # x_sample en y_sample zijn de sampling distances voor de graph
        
        z = min(Pathplanning.x_sample, Pathplanning.y_sample) 


# set destination if destination is a zone
# Dit deel code wordt gebruikt in het geval dat je bij een blokje bent en een doel een zone is.

        if (endlocation == False):
            zones = []
            zone_nodes = []
            for zone in self.zones:
                
                # Check welke kleur je hebt om te weten naar welke zone je moet gaan
                # Blauw heeft 2 zones, dus in geval van blauw moet je onderscheidt maken tussen de 
                # upper of lower blue zone. Die wordt bepaald op basis van een algoritme
                # daarvoor die het kortste pad aangeeft (TSP)
                
                if (startlocation[5] == 'G'):
                    color = 'G'
                    if (zone[5] == 'ZG'):
                        y_coord = np.arange(zone[1]-zone[3]/2, zone[1]+zone[3]/2, Pathplanning.y_sample, dtype = int)
                        if (self.right_side == False):
                            x_coord = (zone[0]-zone[2]/2)*np.ones(y_coord.size)
                        else:
                            x_coord = (zone[0]+zone[2]/2)*np.ones(y_coord.size)
                        for i in range(len(x_coord)):
                            zone_nodes = np.append(zone_nodes, [np.around(x_coord[i]), y_coord[i]])
                        zones = zone

                elif (startlocation[5] == 'R'):
                    color = 'R'
                    if (zone[5] == 'ZR'):
                        zones.append(zone)
                        y_coord = np.arange(zone[1]-zone[3]/2, zone[1]+zone[3]/2, Pathplanning.y_sample, dtype = int)
                        if (self.right_side == False):
                            x_coord = (zone[0]+zone[2]/2)*np.ones(y_coord.size)
                        else:
                            x_coord = (zone[0]-zone[2]/2)*np.ones(y_coord.size)
                        for i in range(len(x_coord)):
                            zone_nodes = np.append(zone_nodes, [np.around(x_coord[i]), y_coord[i]])
                        zones = zone

                else:
                    color = 'B'
                    if (zone[5] == 'ZBU'):
                        zones.append(zone)
                        x_coord = np.arange(zone[0]-zone[2]/2, zone[0]+zone[2]/2, Pathplanning.x_sample, dtype = int)
                        y_coord = (zone[1]+zone[3]/2)*np.ones(x_coord.size)
                        for i in range(len(x_coord)):
                            zone_nodes = np.append(zone_nodes, [np.around(x_coord[i]), y_coord[i]])
                    if (zone[5] == 'ZBL'):
                        zones.append(zone)
                        x_coord = np.arange(zone[0]-zone[2]/2, zone[0]+zone[2]/2, Pathplanning.x_sample, dtype = int)
                        y_coord = (zone[1]-zone[3]/2)*np.ones(x_coord.size)
                        for i in range(len(x_coord)):
                            zone_nodes = np.append(zone_nodes, [np.around(x_coord[i]), y_coord[i]])

            # In dit deel van de code ga je nodes bijcreëeren langs de kanten van de zone waar je heen moet zodat je
            # recht op een blokje afrijdt.
            for i in range(len(zone_nodes)//2):
                vertex = pd.DataFrame({'free': [True], 'X': [zone_nodes[2 * i]], 'Y': [zone_nodes[2 * i + 1]], 'distance_to_start': [9999], 'sourceX': [-1], 'sourceY': [-1], 'distance_to_end': [0], 'distance_total': [9999]})
                vertices_df = vertices_df.append(vertex, ignore_index=True)
                
            self.__checkgraph(vertices_df)

        # set destination if destination is a point
        else:

        # TARGET
            delta_end = endlocation[4]
            cos_end = np.cos(np.pi*delta_end/180)
            sin_end = np.sin(np.pi*delta_end/180)

            directions = np.array([[sin_end*z, -cos_end*z], [cos_end*z, sin_end*z]])
            end_centre = np.array([endlocation[0], endlocation[1]])

            for d in directions:
                for i in np.linspace(1,3,3):
                    nodes = np.append(nodes, [d * i + end_centre])
                    nodes = np.append(nodes, [-d * i + end_centre])
            
            endvertex = pd.DataFrame({'free': [True], 'X': [endlocation[0]], 'Y': [endlocation[1]], 'distance_to_start': [9999], 'sourceX': [-1], 'sourceY': [-1], 'distance_to_end': [0], 'distance_total': [9999]})
            vertices_df = vertices_df.append(endvertex, ignore_index=True)

        # START node
            delta_start = startlocation[4]
            cos_start = np.cos(np.pi * delta_start / 180)
            sin_start = np.sin(np.pi * delta_start / 180)

            d = np.array([z * cos_start, z * sin_start])

            for i in np.linspace(1, 3, 3):
                nodes = np.append(nodes, [d * i + start_centre])
        
            for i in range(len(nodes)//2):
                current_dte = self.euclidian_dist(nodes[2*i:2*i+2], endlocation)
                current_dtotal = current_dte + 9999
                vertex = pd.DataFrame({'free': [True], 'X': [nodes[2 * i]], 'Y': [nodes[2 * i + 1]], 'distance_to_start': [9999], 'sourceX': [-1], 'sourceY': [-1], 'distance_to_end': [current_dte], 'distance_total': [current_dtotal]})
                vertices_df = vertices_df.append(vertex, ignore_index=True) #ignore_index = true: does not use index labels
            

#* add source & destination nodes
        if endlocation == False:
            if (color == 'G'):
                if (self.right_side == False):
                    end_dist = self.euclidian_dist(startlocation, [zones[0]-zones[2]/2, startlocation[1]])
                else:
                    end_dist = self.euclidian_dist(startlocation, [zones[0]+zones[2]/2, startlocation[1]])
            elif (color == 'R'):
                if (self.right_side == False):
                    end_dist = self.euclidian_dist(startlocation, [zones[0]+zones[2]/2, startlocation[1]])
                else:
                    end_dist = self.euclidian_dist(startlocation, [zones[0]-zones[2]/2, startlocation[1]])
            else:
                end_dist = min(self.euclidian_dist(start_centre, [startlocation[0], (zones[0][1]+zones[0][3]/2)]), self.euclidian_dist(start_centre, [start_centre[0], (zones[1][1]-zones[1][3]/2)]))
        else:
            end_dist = self.euclidian_dist(startlocation, endlocation)

        startvertex = pd.DataFrame({'free': [True], 'X': [startlocation[0]], 'Y': [startlocation[1]], 'distance_to_start': [0], 'sourceX': [startlocation[0]], 'sourceY': [startlocation[1]], 'distance_to_end': [end_dist], 'distance_total': [end_dist]})
       
        vertices_df = vertices_df.append(startvertex, ignore_index=True)

        startindex = vertices_df[(vertices_df['X'] == startlocation[0]) & (vertices_df['Y'] == startlocation[1])].index[0]
        if (endlocation == False):
            endindex = np.array([])
            for i in range(len(zone_nodes)//2):
                endindex = np.append(endindex, vertices_df[(vertices_df['X'] == zone_nodes[2*i]) & (vertices_df['Y'] == zone_nodes[2*i+1])].index[0])

        else:
            endindex = np.array(vertices_df[(vertices_df['X'] == endlocation[0]) & (vertices_df['Y'] == endlocation[1])].index[0]) 

#* Dijkstra & A*
        
        while (vertices_df[vertices_df['free'] == True].shape[0] != 0): 
            Q = vertices_df[vertices_df['free'] == True]
            
            # de index is de index van de node met de kleinste afstand tot de eindbestemming
            # Dit is de volgende node die we moeten nemen
            # Hiervan zoek je de x en y coordinaat
            
            currentindex = Q.astype(float).idxmin()['distance_total'] 
            vertices_df.loc[currentindex, 'free'] = False 
            currentvertexX = vertices_df.loc[currentindex, 'X']
            currentvertexY = vertices_df.loc[currentindex, 'Y']
            currentvertexdist = vertices_df.loc[currentindex, 'distance_to_start']
            
            # als die index de eindindex is dan stop je met itereren
            
            if currentindex in endindex:
                endindex = currentindex
                break

            
            # Hier itereer je over alle vrije nodes
            
            Q = vertices_df[vertices_df['free'] == True] 
            for vertexindex in Q.index:
                neighborvertexX = Q.loc[vertexindex, 'X']
                neighborvertexY = Q.loc[vertexindex, 'Y']
                neighborvertexdist = Q.loc[vertexindex, 'distance_to_start']

                # Afstand tussen de vrije node waarover je itereert en de laatst vastgelegde node
                sourcevertexdist = self.euclidian_dist([neighborvertexX,neighborvertexY], [currentvertexX,currentvertexY])
                
                # Check of de nodes buurnodes zijn door de euclidische afstand te nemen
                if sourcevertexdist < max(1.7 * Pathplanning.x_sample, 1.7 * Pathplanning.y_sample):
                    
                    # afstand tot de start + afstand tot de buurnode < afstand van de buurnode tot de start (9999)
                    # Dit is de A* stap
                    if currentvertexdist + sourcevertexdist < neighborvertexdist:
                        
                        # Zet de vaste node als vertrekpunt voor de volgende node
                        # Zet de nieuwe afstand tot de start als de afstand van de sourcenode + afstand van source tot buurnode
                        
                        vertices_df.loc[vertexindex, 'sourceX'] = currentvertexX
                        vertices_df.loc[vertexindex, 'sourceY'] = currentvertexY
                        vertices_df.loc[vertexindex, 'distance_to_start'] = currentvertexdist + sourcevertexdist
                        
                        # Als de eindlocatie een zone is, bereken dan de afstand tot de zone en zet deze als end_dist
                        
                        if endlocation == False:
                            # determine distance through projection
                            if (color == 'G'):
                                if (self.right_side == False):
                                    end_dist = self.euclidian_dist([currentvertexX, currentvertexY], [zones[0]-zones[2]/2, currentvertexY])
                                else:
                                    end_dist = self.euclidian_dist([currentvertexX, currentvertexY], [zones[0]+zones[2]/2, currentvertexY])
                            elif (color == 'R'):
                                if (self.right_side == False):
                                    end_dist = self.euclidian_dist([currentvertexX, currentvertexY], [zones[0]+zones[2]/2, currentvertexY])
                                else:
                                    end_dist = self.euclidian_dist([currentvertexX, currentvertexY], [zones[0]-zones[2]/2, currentvertexY])
                            else:
                                end_dist = min(self.euclidian_dist([currentvertexX, currentvertexY], [currentvertexX, (zones[0][1]+zones[0][3]/2)]), self.euclidian_dist([currentvertexX, currentvertexY], [currentvertexX, (zones[1][1]-zones[1][3]/2)]))
                        
                        # Als de eindlocatie een node is (bvb een blokje) bereken de afstand tot eindlocatie
                        # dan met de locatie van dat blokje
                        
                        else:
                            end_dist = self.euclidian_dist([currentvertexX, currentvertexY], endlocation)
                        
                        # Pas de einddistance aan bij de node waarover je itereert
                        vertices_df.loc[vertexindex, 'distance_to_end'] = end_dist
                        
                        # Pas de totale distance aan bij de node waarover je itereert
                        vertices_df.loc[vertexindex, 'distance_total'] = currentvertexdist + sourcevertexdist + end_dist
        
        end = time.time()
        # look up the path
        xy_path = []
        
        # P is de verzameling van alle nodes waarover geitereerd is.
        P = vertices_df[vertices_df['free'] == False]
        vertexindex = endindex
        
        # Maak het pad van begin tot het eind. Begin vanaf het einde en werk zo terug
        while True:
            x1 = vertices_df.loc[vertexindex, 'X']
            y1 = vertices_df.loc[vertexindex, 'Y']
            xy_path.append([x1, y1])
            sourceindex = P[(P['X'] == P.loc[vertexindex, 'sourceX']) & (P['Y'] == P.loc[vertexindex, 'sourceY'])].index[0]
            vertexindex = sourceindex
            # break als je bij de start bent
            if vertexindex == startindex:
                xy_path.append([P.loc[vertexindex, 'sourceX'],P.loc[vertexindex, 'sourceY']])
                break
            
        # Geef de nodes terug en het pad in de juiste volgorde
        return vertices_df, startindex, endindex, [x for x in reversed(xy_path)]

#* Draws the path explored by the algorithm
    def draw_tree(self, vertices_df, startindex, endindex, linethickness=3):

        image = self.image.copy()
        P = vertices_df[vertices_df['free'] == False]
        for vertexindex in P.index:
            x1 = int(vertices_df.loc[vertexindex, 'X'])
            y1 = int(vertices_df.loc[vertexindex, 'Y'])
            x2 = int(vertices_df.loc[vertexindex, 'sourceX'])
            y2 = int(vertices_df.loc[vertexindex, 'sourceY'])
            image = cv2.line(image, (x1, y1), (x2, y2),(200, 200, 200), linethickness)

#* Draws the shortest path 
    def draw_shortest_path(self, vertices_df, startindex, endindex, linethickness = 3):
        vertexindex = endindex
        image = self.image.copy()
        P = vertices_df[vertices_df['free'] == False]
        while True:
            x1 = int(vertices_df.loc[vertexindex, 'X'])
            y1 = int(vertices_df.loc[vertexindex, 'Y'])
            x2 = int(vertices_df.loc[vertexindex, 'sourceX'])
            y2 = int(vertices_df.loc[vertexindex, 'sourceY'])
            sourceindex = P[(P['X'] == P.loc[vertexindex, 'sourceX']) & (P['Y'] == P.loc[vertexindex, 'sourceY'])].index[0]
            image = cv2.line(image, (x1, y1), (x2, y2), (0, 200, 0), linethickness)
            vertexindex = sourceindex
            if vertexindex == startindex:
                break
        self.__show(image)

#* Returns the euclidian distance
    def euclidian_dist(self, begin, end):
        return np.sqrt((begin[0]-end[0])**2 + (begin[1]-end[1])**2)

#* DEBUG: shows image
    def __show(self, img):    
        cv2.imshow('Debug', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#* DEBUG: shows graph
    def __checkgraph(self, graph):
        img = self.image.copy()
        for index, row in graph.iterrows():
            cv2.circle(img, (int(round(row['X'])), int(round(row['Y']))), 5, (255, 0, 0))
            
        cv2.startWindowThread()
        cv2.namedWindow('Graph nodes')
        cv2.imshow('Graph nodes',img) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    cv2.startWindowThread()
    imname = "frame6"                                         ##
    fieldside = 1 # 0 is left, 1 is right
    path = "images/" + imname + ".jpg"
    testframe = Frame(path)
    # Process image
    img = testframe.processed_img()
    # Save processed image
    cv2.imwrite("processing/" + imname + "_proc.png", img)
    
    
    # Get get path data past edge detection toe om de blokjes en borders te vinden
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
    
    p.__show(fieldimg)
    # Path tracking init
    robotyaw = arucos[0][4]
    blocks_new = blocks
    step = 1
    totallength = 0
    # Tracking loop
    graph = p.create_graph()
    processed_graph, startindex, endindex, xy_path = p.Dijkstra_A(graph, arucos[0], blocks[0])
    # X/Y array for pathtracking to block
    x = [i[0]/rescaleratio for i in xy_path]
    y = [i[1]/rescaleratio for i in xy_path]
    
    #*Gebruiksaanwijzing
    #Bij de graph kan je 'blocks' & 'arucos ingeven,
    #vul je niets in neemt hij de waarden die je bij init pathplanning hebt meegegeven
    #Vul je wel iets in bij blocks, vul dan de blocks in waar je wil dat hij geen rekening mee houdt
    #geef wel, zelfs als je maar 1 blok meegeeft, deze mee als een geneste list
    #Bij arucos vul je False in als je wil dat hij niet met de Aruco rekening houdt op het veld
    #xy_path geeft je de coordinaten van de weg terug als [[xbegin, ybegin], ..., [xeind, yeind]]