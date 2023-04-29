
import math
import numpy as np
# blokjes matrix p1[x, y, breedte, hoogte, hoek, kleur]

def distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)

def make_graph(M, l, Aruco):        #M is matrix met gegevens van blokje, l is lijst met gegevens zones [xR,xG,yBO,yBB], Aruco is lijst met gegevens van marker                                      
    M, n = expand_blokjesmatrix(M,l)
    graph = np.ones((2*n+2,2*n+2))*100000       #lijst met als rijen 'van' knopen en als kolommen 'naar' knopen, alle afstanden voorlopig oneindig
    for i in range(2*n+2):      #n blokjes + zwevende knoop (knoop 0)
        graph[i][i] = 0         #diagonaalelementen nul (blijven kost niks)
    graph[0][1] = 0         #pad van zwevende knoop naar arucomarker gratis
    for i in range(n+2,2*n+2):
        graph[i][0] = 0         #pad van zones naar zwevende knoop heeft kost 0
    for i in range(2,n+2):
        graph[1][i] = distance(M[i-2], Aruco[0]) #afstand van Aruco naar blokje
        graph[i][i+n] = distance(M[i-2], M[(i-2)+n]) #afstand van blokje naar juiste zone
    for i in range(n+2,2*n+2):
        for j in range(2,n+2):
            graph[i][j] = distance(M[i-2], M[j-2]) #pad van zone naar elke knoop
    return graph

def expand_blokjesmatrix(M,l):      #blokjesmatrix aanpassen zodat de zoneknopen gebaseerd op loodrechte afstand blok <> zone ook gedefiniëerd zijn
    n = len(M)
    N = []
    for p in M:
        if p[5] == 'R':
            zoneknoop = [l[2],p[1],p[2],p[3],p[4],p[5],p[6]]
        if p[5] == 'G':
            zoneknoop = [l[3],p[1],p[2],p[3],p[4],p[5],p[6]]
        if p[5] == 'B': #beslissing naar welke blauwe zone de blok moet (boven <> onder) gebaseerd op kortste afstand
            if (p[1]-l[0]) > (l[1]-p[1]):
                zoneknoop = [p[0],l[1],p[2],p[3],p[4],p[5],p[6]]
            else:
                zoneknoop = [p[0],l[0],p[2],p[3],p[4],p[5],p[6]]
        N.append(zoneknoop)
    for i in N:
        M.append(i)
    return M, n