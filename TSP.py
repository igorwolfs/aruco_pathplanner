############################################################
# UPPER LAYER
############################################################

import numpy as np
import TSP_graph as Graph
import cv2

# Traveling Salesman problem
def tsp(graph):
    N = len(graph) #aantal knopen in graaf
    tab = np.ones((N,2**N))*(100000) #memo is een matrix (Nx2**N) waarin alle data zit. De rijen stellen de laatst bezochte knoop voor terwijl de kolommen getallen in bit-voorstelling voorstellen. (01 = initiële knoop bezocht; 11 = knoop met index 0 en 1 bezocht; 10 = knoop met index 1 bezocht maar index 0 niet, ..., 11111111111111111111 = alle knopen bezocht)
    tab = initiate(graph,tab)
    tab = main(graph,tab,N)
    minCost = MinCost(graph,tab,N)
    tour = Tour(graph,tab,N)
    return tour, minCost

#Start bij initiële knoop en voeg er één toe (~ kortste pad tussen twee knopen)
def initiate(graph, tab):
    tab[1][1 | 1 << 1] = graph[0][1]
    return tab

#Nu stap per stap de lus vergroten tot elke knoop erin zit
def main(graph, tab, N):
    for r in range(3,N+1): #vanaf 3 knopen tot alle knopen in rekening worden gebracht
        for state in combinations(r,N): #genereren van alle mogelijke combinaties van r knopen bezocht
            if check(0,state): #de beginknoop moet zeker al bezocht zijn, combinaties met een 0 als laatste bit mogen dus al weg
                continue
            for newnode in range(1,N): #itereren over alle knopen die we kunnen toevoegen aan het pad (t.o.v. vorige iteratie)
                if check(newnode,state): #volgende knoop moet een 1 in de 'subset' hebben want anders is deze niet bezocht
                    continue
                prevstate=state ^ (1 << newnode) #state is de huidige 'subset' met de knoop 'newnode' op 0 (dus zogezegd niet bezocht), zo kan je de afstand van start --> 'endnode' hergebruiken uit vorige iteratie en dan elke keer een knoop toevoegen.
                minDistance = 100000 #initiële afstand tot volgende knoop op oneindig zetten
                for lastnode in range(1,N): #Itereren over de overgebleven bezochte knopen ("endnode' is de laatste knoop van de het pad uit de vorige iteratie, dus toen er een knoop minder was")
                    if lastnode==newnode or check(lastnode,state) or graph[lastnode][newnode] == 100000: #Nieuwe knoop mag niet de vorige knoop zijn en moet deel zijn van de 'subset' die we bekijken
                        continue
                    newDistance = tab[lastnode][prevstate] + graph[lastnode][newnode] #nieuwe afstand = afstand van begin tot 'endnode' + afstand tussen 'lastnode' en 'newnode'
                    if newDistance < minDistance:
                        minDistance = newDistance
                        tab[newnode][state] = minDistance
    return tab

def check(i,state): #returns true if the ith bit in the subset is not 1
    return (1 << i & state) == 0

def combinations(r,n):
    states=[]
    combinations2(0,0,r,n,states)
    return states

def combinations2(s,x,r,n,states):
    if r==0:
        states.append(s)
    else:
        for i in range(x,n):
            s = s | (1 << i)
            combinations2(s,i+1,r-1,n,states)
            s = s & ~(1 << i)
    return states

def MinCost(graph,tab,N): #Berekent de lengte van kortste tour
    finalstate = (1 << N) - 1
    minCost = 100000 #infinity
    for lastnode in range(1,N):
        tourCost = tab[lastnode][finalstate] + graph[lastnode][0]
        if tourCost < minCost:
            minCost = tourCost
    return minCost

def Tour(graph,tab,N): #Bepaalt de volgorde in de kortste tour (zie Whiteboard)
    lastindex = 0
    state = (1 << N) - 1
    tour = np.zeros(N+1,dtype=int)
    for i in range(N-1,0,-1):
        index = 1
        for j in range(1,N):
            if check(j,state):
                continue
            prevDist = tab[index][state] + graph[index][lastindex]
            newDist = tab[j][state] + graph[j][lastindex]
            if newDist < prevDist:
                index = j
        tour[i] = index
        state = state ^ (1 << index)
        lastindex = index
    return tour

def TourSeq(matrix, tour):
    N = []
    for i in tour[2:len(tour)-1:2]:
        N.append(matrix[i-2])
    return N


def draw(image, M, Aruco, tour):
    n = int(len(M)/2)
    x = tour[2]-2
    cv2.line(image,(int(Aruco[0][0]),int(Aruco[0][1])),(int(M[x][0]),int(M[x][1])),(0,135,241),2)
    for i in range(n-1):
        j = tour[2+2*i]-2
        k = tour[4+2*i]-2
        cv2.line(image,(int(M[j][0]),int(M[j][1])),(int(M[k][0]),int(M[k][1])),(0,135,241),2)
    # show(image)
    return image

def show(image):
    cv2.imshow('Path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
