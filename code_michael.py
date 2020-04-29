import numpy as np
import osrm
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely import geometry
from shapely.geometry.polygon import LinearRing, Polygon, LineString

import datetime
date = np.datetime64(datetime.datetime.now())


#Polygone bestimmmen


def projection(A, mitte,p): #MerkatorProjection
    if p == "Mercator":
        A = A*2*np.pi/360
        mitte = mitte[0]*2*np.pi/360 
        x = A[0]-mitte #geogr. Länge
        y = np.log((1+np.sin(A[1]))/(1-np.sin(A[1])))/2
        return np.array([x,y])
    if p == "transvMercator":
        A = A*2*np.pi/360
        mitte = mitte*2*np.pi/360
        A[0]=A[0]-mitte[0]
        a = 1
        k0=1
        x = k0*a/2*np.log((1+np.sin(A[0])*np.cos(A[1]))/(1-np.sin(A[0])*np.cos(A[1])))
        y = k0*a*np.arctan(np.tan(A[1])/np.cos(A[0]))
        return np.array([x,y])
    if p == "p1":
        r = 1
        A = A*2*np.pi/360
        mitte = mitte*2*np.pi/360 
        return np.array([r*(A[0]-mitte[0])*np.cos(mitte[1]),r*(A[1]-mitte[1])])
    if p == "p2":
        r = 1
        A = A*2*np.pi/360
        mitte = mitte*2*np.pi/360 
        return np.array([r*(A[0]-mitte[0])*np.cos(A[1]),r*(A[1]-mitte[1])])
    if p == "p0":
        return A-mitte
    else:
        print("FEHLER projection: Für p wurde keine gültige Projektion angegeben!")

def projectionInv(X, mitte,p): #MerkatorProjection
    if p == "Mercator":
        mitte = mitte[0]*2*np.pi/360 
        l = X[0]+mitte
        if abs(l) > np.pi:
            print("FEHLER: projectionInv gibt unmöglichen Längengrad aus ")
        p = np.arctan(np.sinh(X[1]))
        if abs(p) > np.pi/2:
            print("FEHLER: projectionInv gibt unmöglichen Breitengrad aus ")
        return np.array([l,p])/(2*np.pi)*360
    if p == "transvMercator":
        mitte = mitte*2*np.pi/360
        a = 1
        k0=1
        l = np.arctan(np.sinh(X[0]/k0/a)/np.cos(X[1]/a/k0))+mitte[0]
        b = np.arcsin(np.sin(X[1]/k0/a)/np.cosh(X[0]/k0/a))
        return np.array([l,b])*360/(2*np.pi)
    if p == "p1":
        r =1
        mitte = mitte*2*np.pi/360
        #print(mitte[1]/np.pi)
        b = X[1]/r+mitte[1]
        l = (X[0]/(r*np.cos(mitte[1])))+mitte[0]
        return np.array([l,b])*360/(2*np.pi)
    if p == "p2":
        r =1
        mitte = mitte*2*np.pi/360
        b = X[1]/r+mitte[1]
        l = X[0]/(r*np.cos(b))+mitte[0]
        return np.array([l,b])*360/(2*np.pi)
    if p == "p0":
        return X+mitte
    else:
        print("FEHLER projectionInv: Für p wurde keine gültige Projektion angegeben!")
def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return(x, y)

def angle(a,b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0.0:
        print("FEHLER angle: a ist der Nullvektor!")
    if np.linalg.norm(b) == 0.0:
        print("FEHLER angle: b ist der Nullvektor!")
    a = np.dot(a,b)/abs(np.linalg.norm(a))/abs(np.linalg.norm(b))
    if a > 1 and abs(a-1.0) < 10**(-15):
        return 0.0
    else:
        return np.arccos(a)
    
def angleX(a):
    return np.angle(a[0]+a[1]*1j)

def distance(a,b,E): #double Entfernung Route in einheit E 
    if E == "Einheitskugel":
        #a = np.array(a)*2.0*np.pi/360.0      
        #b = np.array(b)*2.0*np.pi/360.0  
        a = np.radians(a)
        b = np.radians(b)
        R =1.0
        d = R*np.arccos(np.sin(a[1])*np.sin(b[1])+np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0]))  #KM
        return d
    if E == "Orthodrome" or E== "Erde":
        #a = np.array(a)*2.0*np.pi/360.0      
        #b = np.array(b)*2.0*np.pi/360.0 
        a = np.radians(a)
        b = np.radians(b)
        R = 6378.0*1000.0 #m
        d = R*np.arccos(np.sin(a[1])*np.sin(b[1])+np.cos(a[1])*np.cos(b[1])*np.cos(a[0]-b[0])) #KM
        return d
    if E == "OSRM_duration" or E == "OSRM_distance":
        client = osrm.Client(host='http://134.76.24.136/osrm') #https://github.com/gojuno/osrm-py
        coordinates = [a,b]
        response = client.route(coordinates=coordinates)
        if E == "OSRM_duration":
            d = response['routes'][0]['legs'][0]['duration'] #Einheit:s?
        if E == "OSRM_distance":
            d = response['routes'][0]['distance'] #Einheit:m?
        if d == 0.0:
            print("FEHLER distance: Distance ist Null!")
        return d
    else:
        print("FEHLER distance: Es wurde keine gültige Maßeinheit für E angegeben")
        
    
def distance_pos(a,b,E): # list [distance, psition a, position b] entfernung und tatsächlicher routenen start und ende
    if E == "Einheitskugel":
        #a_ = np.array(a)*2.0*np.pi/360.0
        #b_ = np.array(b)*2.0*np.pi/360.0
        a_ = np.radians(a)
        b_ = np.radians(b)
        R = 1.0
        d = R*np.arccos(np.sin(a_[1])*np.sin(b_[1])+np.cos(a_[1])*np.cos(b_[1])*np.cos(a_[0]-b_[0]))
        #a = np.array(a)/2/np.pi*360      
        #b = np.array(b)/2/np.pi*360  
        return [d,a,b]
    if E == "Orthodrome"or E== "Erde":
        #a_ = np.array(a)*2.0*np.pi/360.0      
        #b_ = np.array(b)*2.0*np.pi/360.0
        a_ = np.radians(a)
        b_ = np.radians(b)
        R = 6378.0*1000.0 #m
        d = R*np.arccos(np.sin(a_[1])*np.sin(b_[1])+np.cos(a_[1])*np.cos(b_[1])*np.cos(a_[0]-b_[0]))
        #a = np.array(a)/2/np.pi*360      
        #b = np.array(b)/2/np.pi*360         
        return [d,a,b]
    if E == "OSRM_duration" or E == "OSRM_distance":
        client = osrm.Client(host='http://134.76.24.136/osrm')#https://github.com/gojuno/osrm-py
        coordinates = [a,b]
        response = client.route(coordinates=coordinates)
        if E == "OSRM_duration":
            d = response['routes'][0]['legs'][0]['duration'] #Einheit:s?
            a = np.array(response['waypoints'][0]['location'])
            b = np.array(response['waypoints'][1]['location'])
        if E == "OSRM_distance":
            d = response['routes'][0]['distance'] #Einheit: m?
            a = np.array(response['waypoints'][0]['location'])
            b = np.array(response['waypoints'][1]['location'])
        if d == 0.0:
            print("FEHLER distance_pos: Distance ist Null!")
        if np.linalg.norm(a) == 0.0:
            print("FEHLER distance_pos: a nicht gefunden!")
        if np.linalg.norm(b) == 0.0:
            print("FEHLER distance_pos: b nicht gefunden!")
        return [d, a, b]
    else:
        print("FEHLER distance_pos: Es wurde keine gültige Ma?einheit für E angegeben")
        
        
def plt_polygone(R,a):
    n = len(R)
    i=1
    while i < n:
        X = [R[i-1][0],R[i][0]]
        Y = [R[i-1][1],R[i][1]]
        plt.plot(X,Y,'ro-')
        i+=1
    X = [R[i-1][0],R[0][0]]
    Y = [R[i-1][1],R[0][1]]
    plt.plot(X,Y,'ro-')
    plt.plot(a[0],a[1],'bo')
    #print("\n"+"Position der Ecken: "+str(R) + " und des Mittelpunktes: " + str(a) )
    #plt.axis('equal')
    plt.show()
  
def polygon(a,b1,n,E,p): #NICHT FERTIG !!!! Implementierung  der projection und anderes fehlt! A = mITTELPUNKT, B1 = ERSTER ECKPUNKT, n = ANzahl der Ecken, E = Einheit der Abstände np.array
    a_geo = a    
    mitte = a
    a = projection(a,mitte,p)
    b1 = projection(b1,mitte,p)    
    r = abs(np.linalg.norm(b1-a))#Außenradius
    alpha = angleX(b1-a)
    gamma = 2*np.pi/n
    B=[b1]
    D = np.array([[0.0]*n]*n)
    
    x = distance_pos(a,b1,E)
    D[0,0]= x[0]#distance(a,b1,E)
    x[1]=projection(x[1],mitte,p)
    x[2]=projection(x[2],mitte,p)
    plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'bo-')    
    
    i=1
    while i< n-1:
        
        B = B + [np.array([np.cos(alpha+i*gamma),np.sin(alpha+i*gamma)])*r+a]
        
        x = distance_pos(a_geo,projectionInv(B[i], mitte,p),E)
        D[i,i]= x[0]#distance(a_geo,projectionInv(B[i], mitte),E)
        x[1]=projection(x[1],mitte,p)
        x[2]=projection(x[2],mitte,p)        
        plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'bo-')
        
        x = distance_pos(projectionInv(B[i], mitte,p),projectionInv(B[i-1], mitte,p),E)
        D[i,i-1] = x[0]#distance(projectionInv(B[i], mitte),projectionInv(B[i-1], mitte),E)
        D[i-1,i] = D[i,i-1]
        x[1]=projection(x[1],mitte,p)
        x[2]=projection(x[2],mitte,p) 
        plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'ro-')
        
        i +=1
    
    B = B + [np.array([np.cos(alpha+i*gamma),np.sin(alpha+i*gamma)])*r+a]
    
    x = distance_pos(a_geo,projectionInv(B[i], mitte,p),E)    
    D[i,i]=x[0]
    x[1]=projection(x[1],mitte,p)
    x[2]=projection(x[2],mitte,p)     
    plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'bo-')

    x = distance_pos(projectionInv(B[i], mitte,p),projectionInv(B[i-1], mitte,p),E)       
    D[i,i-1] = x[0]#distance(projectionInv(B[i], mitte),projectionInv(B[i-1], mitte),E)
    D[i-1,i] = D[i,i-1]
    x[1]=projection(x[1],mitte,p)
    x[2]=projection(x[2],mitte,p) 
    plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'ro-')
    
    x = distance_pos(projectionInv(B[i], mitte,p),projectionInv(B[0], mitte,p),E)
    D[i,0] = x[0]#distance(projectionInv(B[i], mitte),projectionInv(B[0], mitte),E)
    D[0,i] = D[i,0]
    x[1]=projection(x[1],mitte,p)
    x[2]=projection(x[2],mitte,p) 
    plt.plot([x[1][0],x[2][0]],[x[1][1],x[2][1]],'ro-')
    
    #plt_polygone(B,a)        
    
    return D
    

def constr(b1,b2,w,s): #gibt neue ecke aus
    alpha = angleX(b1-b2)
    #print("alpha= "+str(alpha/np.pi))    
    if alpha >= w:
        g = alpha -w
        
    if alpha < w:
        g = 2*np.pi - w + alpha
        
    return np.array([np.cos(g),np.sin(g)])*s+b2 #geometrie überprüfen!! Eindeutigkeit ?!
    
def Seite(s,S,i):
    return s
    #return np.mean(S)
    #return s*i-np.sum(S)


def winkel(w,W,i): 
    #return w*i-np.sum(W)
    return w 
    #return np.mean(W)
    

def polygon2(a,b1,n,E,p): #A = mITTELPUNKT, B1 = ERSTER ECKPUNKT, n = ANzahl der Ecken, E = Einheit der Abstände np.array
    a_geo = a    
    mitte = a
    a = projection(a,mitte,p)
    b1 = projection(b1,mitte,p)
    r = abs(np.linalg.norm(b1-a))#Außenradius
    #print("r= "+str(r))
    alpha = angleX(b1-a)
    gamma = 2*np.pi/n
    w = np.pi-2*np.pi/n
    s = 2*r*np.sin(np.pi/n)  
    #print("s= "+str(s))
    #print(w)
    #rint(s)
    #Variablen speichern
    W = np.array([])
    S = np.array([])    
    D = np.array([[0.0]*n]*n)
    
    #Erste Ecke
    B=[b1]
    R = [b1]
    D[0,0]= distance(a_geo,projectionInv(b1, mitte,p),E)


    
    
    #Erste Seite
    B = B + [np.array([np.cos(alpha+gamma),np.sin(alpha+gamma)])*r+a]
    x = distance_pos(projectionInv(B[0], mitte,p),projectionInv(B[1], mitte,p),E)
    R= R + [projection(x[2], mitte,p)]
    #R = R + [B[1]]   
    S = np.append(S,abs(np.linalg.norm(R[1]-R[0])))
    
    D[0,1]=x[0]
    D[1,0]=x[0]
    D[1,1]=distance(a_geo,projectionInv(R[1], mitte,p),E)
    
    #Zweite Seite
    #print(angleX([-1,0])/np.pi)
    #print(w/np.pi)
    B = B + [constr(R[0],R[1],winkel(w,W,2),Seite(s,S,2))]
    x = distance_pos(projectionInv(R[1], mitte,p),projectionInv(B[2], mitte,p),E)
    R= R + [projection(x[2], mitte,p)]
    #R = R + [B[2]]    
    #print(np.linalg.norm(R[2]-a))
    S = np.append(S,abs(np.linalg.norm(R[2]-R[1])))
    W = np.append(W,angle(R[2]-R[1],R[0]-R[1]))
    
    D[2,1]=x[0]
    D[1,2]=x[0]
    D[2,2]=distance(a_geo,projectionInv(R[2], mitte,p),E)
    
    #Seiten dazwischen
    i=3
    if n>3:
        while i< n :
            #print("R= "+str(R[i-1]))            
            B = B + [constr(R[i-2],R[i-1],winkel(w,W,i),Seite(s,S,i))]
            #print("B = "+str(B[i]))
            x = distance_pos(projectionInv(R[i-1], mitte,p),projectionInv(B[i], mitte,p),E)
            R= R + [projection(x[2], mitte,p)]
            #R = R + [B[i]]  
            #print("r_i= "+str(np.linalg.norm(R[i]-a))             
            S = np.append(S,abs(np.linalg.norm(R[i]-R[i-1])))
            W = np.append(W,angle(R[i]-R[i-1],R[i-2]-R[i]))
            
            D[i,i-1]=x[0]
            D[i-1,i]=x[0]
            D[i,i]=distance(a_geo,projectionInv(R[i], mitte,p),E)
            
            i +=1

    #letzte Seite
    W = np.append(W,angle(R[0]-R[i-1],R[i-2]-R[i-1]))
    x = distance(projectionInv(R[i-1], mitte,p),projectionInv(b1, mitte,p),E)
    S = np.append(S,abs(np.linalg.norm(R[0]-R[i-1])))
    W = np.append(W,angle(R[1]-R[0],R[i-1]-R[0]))
    D[0,i-1]=x
    D[i-1,0]=x
    
    #plt_polygone(R,a)    
    #for r in R:
        #print(projectionInv(r, mitte,p))
        #print(distance(a_geo,projectionInv(r, mitte,p),E))
    return D, R, B, a

def D_orth(Ex,a):
    E = "Einheitskugel"
    n = len(Ex)
    D = np.array([[0.0]*n]*n)
    #print(Ex)
    for i in np.arange(0,n):
        D[i,i] = distance(Ex[i],a,E)
    
    for i in np.arange(0,n):
        D[i,(i+1)%n]= distance(Ex[i],Ex[(i+1)%n],E)
    
    return D
        
    

#Polygone auswerten -> Krümmung berechnen

def heron_fm(a,b,c):
    s = (a+b+c)/2
    return np.sqrt(abs(s*(s-a)*(s-b)*(s-c)))

def area(D): #S
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error area: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error area: D is not a square Matrix!")
    n=d[0]
    A=0.0
    i=0
    while i< (n-1):
        A+= heron_fm(D[i,i],D[i,i+1],D[i+1,i+1])/3
        i+=1
    A+= heron_fm(D[i,i],D[i,0],D[0,0])/3
    return A

def Omega(D):
    A = []
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error Omega: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error Omega: D is not a square Matrix!")
    n=d[0]
    #print("\n"+"Abstände:"+"\n"+str(D))    
    i=0
    while i<(n-1):
        #print((D[i,i]**2+D[i+1,i+1]**2-D[i,i+1]**2)/(2*D[i,i]*D[i+1,i+1]))
        a = ((D[i,i]**2+D[i+1,i+1]**2-D[i,i+1]**2)/(2*D[i,i]*D[i+1,i+1]))
        if abs(a) > 1:
            print(str(i+1)+" ACHTUNG Omega: arccos Wertebereich um "+str(abs(a)-1)+" überschritten!")            
            print(D[i,i])
            print(D[i+1,i+1])  
            print(D[i+1,i+1]+D[i,i])
            print(D[i,i+1])
            if abs(abs(a)-1)< 10**(-9):
                if a < 0:
                    a = -1
                else:
                    a=1
        A = A +[np.arccos(a)]
        i+=1
    
    a = ((D[i,i]**2+D[0,0]**2-D[i,0]**2)/(2*D[i,i]*D[0,0]))
    if abs(a) > 1:
        print(str(i+1)+" ACHTUNG Omega: arccos Wertebereich um "+str(abs(a)-1)+" überschritten!")
        print(D[0,0])
        print(D[i,i])
        print(D[i,0])
        if abs(abs(a)-1)< 10**(-9):
                if a < 0:
                    a = -1
                else:
                    a=1
    A = A +[np.arccos(a)]
    
    #print("\n"+"Winkel: "+str(A))    
    return np.array(A)

def angel_defect(O,n): #R
    #O = Omega(D)
    #print(O)    
    Sum = np.sum(O)
    R =((2)*np.pi-Sum)#/(n-2) #Richtig?!
    return R


def Delta(O):
    #O = Omega(D)
    E = [1.0]*np.shape(O)[0]
    return np.sin(angle(O,E))

##Tests:

def average_radius(D):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error average_radius: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error average_radius: D is not a square Matrix!")    
    x = np.diag(D)
    return np.mean(x)
    
    
def norm_std_radius(D):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error average_radius: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error average_radius: D is not a square Matrix!")    
    x = np.diag(D)
    return np.std(x)/np.mean(x)
    
def mean_edge(D):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error radius_sphere: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error radius_sphere: D is not a square Matrix!")
    n=d[0]
    #print(n)
    U = 0.0    
    i=0
    while i <n:
        U = U + D[i,(i+1)%n]
        i = i + 1
    return U/n
    
    
def norm_std_edge(D):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error radius_sphere: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error radius_sphere: D is not a square Matrix!")
    n=d[0]
    #print(n)
    U = []    
    i=0
    while i <n:
        U.append(D[i,(i+1)%n])
        i = i + 1
    U = np.array(U)
    return np.std(U)/np.mean(U)
    
def symmetry_Edges(D,eF):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error symmetry_Edges: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error symmetry_Edges: D is not a square Matrix!")
    n=d[0]
    d=np.array(np.shape(D))
    m = mean_edge(D)
    S = True
    i=0
    while i <n:
        if abs(D[i,(i+1)%n]-m)/m > eF:
            S = False
        i = i+1
    return S

def symmetry_Radius(D,rF):
    d=np.array(np.shape(D))
    if np.shape(d)[0] != 2:
        print("Error symmetry_Radius: D is not a Matrix!")
    if d[0]!= d[1]:
        print("Error symmetry_Radius: D is not a square Matrix!")
    n=d[0]
    d=np.array(np.shape(D))
    m = average_radius(D)
    S = True
    i=0
    while i <n:
        if abs(D[i,i]-m)/m > rF:
            S = False
        i = i+1
    return S

def test_inside(Ex, a):
    P = np.array(Ex)
    S = True
    if np.amax(P[:,0])<a[0] or np.amax(P[:,1])<a[1] or np.amin(P[:,0])>a[0] or np.amin(P[:,1])>a[1]:
        S = False
    return S
    
def test_inside2(Ex, a):
    P = np.array(Ex)
    S = True
    coordinates = P 
    #print(coordinates)
    poly =Polygon(coordinates)
    point = Point(a)
     
    if poly.contains(point):
        S = True
    else:
        S = False
         
    return S
    
def test_intersection(Ex):
    P = np.array(Ex)
    n = len(P)
    N = np.arange(0,n)
    S = False
    for i in N:
        for j in N:
            if i != j and (i+1)%n != j and (j+1)%n != i:
                v1 = Polygon([P[i],P[(i+1)%n],P[i],P[(i+1)%n]])
                v2 = Polygon([P[j],P[(j+1)%n],P[j],P[(j+1)%n]])
                if v1.intersects(v2) == True:
                    S = True
#                plt.clf()
#                x1,y1 = v1.exterior.xy
#                x2,y2 = v2.exterior.xy 
#                plt.plot(x1, y1)
#                plt.plot(x2, y2)
#                plt.show()
#                print(v1.intersects(v2))
    return S
    
def test_aTouching(Ex,a,mB):
    P = np.array(Ex)
    n = len(P)
    N = np.arange(0,n)
    S = False
    for i in N:
        point = Point(a)
        line = LineString([P[i],P[(i+1)%n]])
        buffer = line.buffer(mB)    
        if buffer.contains(point) == True:
            S = True
    return S
    
def test_PsTouching(Ex,eB):
    P = np.array(Ex)
    n = len(P)
    N = np.arange(0,n)
    S = False
    for j in N:
        for i in N:
            if j != i and j != (i+1)%n:
                point = Point(P[j])
                line = LineString([P[i],P[(i+1)%n]])
                buffer = line.buffer(eB)    
                if buffer.contains(point) == True:
                    S = True
        
    return S
    
def test_symm_angle(Ex,aF):
    P = np.array(Ex)
    n = len(P)
    N = np.arange(0,n)
    S = True
    aL = []
    for i in N:
        a= P[i]-P[(i+1)%n]
        b= P[(i+2)%n]-P[(i+1)%n]
        aL.append(angle(a,b))
    m = np.mean(np.array(aL))
    for a in aL:
        if abs(a-m)/m>aF:
            S = False
    return S
        
    
    

def erste_ecke(a,r,E):
    b = np.array([0,r])+a
    x = distance_pos(a,b,E)
    return x[1], x[2]


def first_vertex_offset(a,r,E,p,offset):
    b = np.array([0,r])+a
    a_xy = projection(a,a,p)
    b_xy = projection(b,a,p)
   
    r_length = abs(np.linalg.norm(a_xy-b_xy))
    r_xy = np.array(pol2cart(r_length,np.radians(offset))) #transform r_length and offset to cart. coord.
   
    b_xy = a_xy + r_xy
    x = distance_pos(a,projectionInv(b_xy,a,p),E)
    return x[1], x[2]

#Ergebnisse in Datei schreiben

def writeFile(fileName_, Liste, Legende):
  f = open(fileName_,"w")
  f.write("#" + Legende + "\n")
  i = 0
  while i < len(Liste):
    Zeile = ""
    k = 0
    while k < len(Liste[i]):
      Zeile = Zeile + str(Liste[i][k]) + " "
      k = k + 1   
    f.write(Zeile+ "\n")
    i = i + 1

#Versuchsparameter: 
#a= [länge,breite]
A =[
    #np.array([10.006839,53.498684]), #Hamburg
    #np.array([9.834144,49.304735]),#Buchenbach
    #np.array([9.959794,51.549821]),#Göttingen
    #np.array([9.925473,51.525748]),#Göttingen Südstadt
    #np.array([ 9.951367,51.560618]),#Göttingen Nordcampus
    #np.array([ -73.988896,40.750324]),#New York
    #np.array([-46.640039,-23.577318]),#S
    np.array([10.163439,52.520176]),#Hannover
    #np.array([ 10.687959,52.414715]),#Wolfsburg
    #np.array([10.112690,51.887916]),#Seesen
    #np.array([13.405713,52.520165])#Berlin
    np.array([10.795960,48.759415]),#Kaisheim
    np.array([9.854321,50.561438])#Fulda
    #np.array([,]),
    ]

#R =[0.01
#R=[0.03,0.01,0.02,0.05 # in Breitengrad
#    ]
 
R = np.arange(0.3, 2.0,0.01)
Name = "time"
#R = np.arange(0.25,1.0,0.05)

#E = "Orthodrome"#"OSRM_distance"#"Orthodrome"
ListE = ["OSRM_duration"]#,"Einheitskugel"]#["Orthodrome","OSRM_distance"]#,"OSRM_duration"]

N = [3,4,5,6,7,8,9]

Offset = np.arange(0,0.5,0.1)#range(90,210,24)

#p = "Mercator"
p = "transvMercator"
#p= "p2"


fileName_ ="Riemannkrümmung_all" #'Riemann_Krümmung_n='+str(n)+"_Messung="+str(E)+'.txt'
Legende = 'center_longitude, '\
          'center_latitude, '\
          'number_edges, planed_euklidin_radius_[], '\
          'measuring_unit, '\
          'offset_angle_in_deg, '\
          'curvature_unit_spehere, average_radius_in_lengt_units, '\
          'angle_of_defect_R, '\
          'area_S, '\
          'Delta, '\
          'curvature_K=R/S, '\
          'corrected_curvature_K_corr=K/(1+4*Delta), '\
          'normalization_curvature, '\
          'normalized_curvature, '\
          'normalized_corrected_curvature'#'measuring_unit, projection, number_of_corners, center_longitude, center_latitude, planed_euklidin_radius_[], average_radius_in_lengt_units, angle_of_defect_R, area_S, Delta, curvature_K=R/S, corrected_curvature_K_corr=K/(1+4*Delta), normalization_curvature, normalized_curvature, normalized_corrected_curvature'

#==============================================================================
# 
# #Rechnung starten
# print("start")
# ZählerGesamt = 0
# ZählerGut = 0
# ZählerSchlecht = 0
# ZählerSchlechtGut=0
# Liste =[]
# for E in ListE:
#     for a in A:
#         X = []
#         Y = []
#         stdY = []
#         Rad = []
#         stdRad =[]
#         Ad = []
#         stdAd =[]
#         for r in R:
#             AD = []      
#             KR = []
#             RAD = []        
#             for n in N:
#                 for offset in Offset: ### NEU ; auch Ausgabe in "Liste" geändert ###
#                     ZählerGesamt +=1                    
#                     a1, b1 = first_vertex_offset(a,r,E,p,2*np.pi/n*offset)
#                     #print("\n"+str(n)+"-Ecke mit Mitte bei a ="+str(a1)+", theo. Radius= "+str(r)+"Messsystem " + E)
#                     D, Ex, B, a_proj = polygon2(a1,b1,n,E,p)
#                    
#                     #eF = 0.7
#                     #rF = 0.8
#                     #aF = 0.4
#                     #mB = 1/10000
#                     #eB = 1/10000
#                     Ex_ =[]                    
#                     for pr in Ex:
#                         Ex_.append(projectionInv(pr, a1,p))
#                     Ex_=np.array(Ex_)
#                     #print(Ex_)                    
#                     
#                     D_E = D_orth(Ex_,a1)
#                     #print(D_E)
#                     O_E = Omega(D_E)
#                     ad_E = angel_defect(O_E,n)
#                     ar_E = area(D_E)
#                     
#                     O = Omega(D)
#                     ad = angel_defect(O,n)
#                     ar = area(D)
#                     #print(ad)
#                     #print(ar)                    
#                     
#                     if abs(ad_E/ar_E-1)<0.1 and  np.isinf(ad/ar) == False and np.isnan(ad/ar) == False:
#                         ZählerGut +=1
#                         #print("Sufficiently symmetric!")
#                         #if abs(ad/ar) > 10**(-5) or ad/ar == float('Inf'):
#                             #plt_polygone(Ex,projection(a1,a1,p))
#                             #print("curvature [1/m²] = "+str(ad/ar))
#                             #print("curvature_unit_spehere = "+str(ad_E/ar_E))
#                         norm_curv = 1/(mean_edge(D))**2
#                         delta = Delta(O)
#                         Liste = Liste +[[a1[0],a1[1],n,r,E,offset,ad_E/ar_E,average_radius(D),ad,ar,delta,ad/ar,ad/ar/(1+4*delta),norm_curv,ad/ar/norm_curv,ad/ar/(1+4*delta)/norm_curv]]
#                         AD.append(ad)
#                         KR.append(ad/ar)
#                         RAD.append(average_radius(D))
#                     else:
#                         ZählerSchlecht +=1
#                         if abs(ad/ar) < 10**(-5):
#                             ZählerSchlechtGut +=1
#                             #plt_polygone(Ex,projection(a1,a1,p))
#                             #print("curvature [1/m²] = "+str(ad/ar))
#                             #print("curvature_unit_spehere = "+str(ad_E/ar_E))
#             if len(KR) > 0:
#                 X.append(r)
#                 Y.append(np.mean(np.array(KR)))
#                 stdY.append(np.std(np.array(KR)))
#                 
#                 if np.isnan(Y[-1]):
#                     print(KR)
#                 
#                 Rad.append(np.mean(np.array(RAD)))
#                 stdRad.append(np.std(np.array(RAD)))
#                 Ad.append(np.mean(np.array(AD)))
#                 stdAd.append(np.std(np.array(AD)))
#         
#         
#         #plt.plot(X,np.log(np.array(Y)),label = "mean" )
#         #plt.plot(X,np.log(np.array(stdY)),label = "std" )
#         plt.clf()
#         plt.plot(np.array(Rad)/1000,Y,label = "mean" )  
#         plt.xlabel("km")
#         plt.ylabel("Krümmung [1/m²]")    
#         plt.title(E+"_"+str(a))
#         plt.legend( bbox_to_anchor=(0.99, 0.63))    
#         plt.savefig(E+"_"+str(a)+"_KR_mean-R"+".png")
#         
#         plt.clf()
#         plt.plot(np.array(Rad)/1000,stdY,label = "std" )  
#         plt.xlabel("km")
#         plt.ylabel("Krümmung [1/m²]")    
#         plt.title(E+"_"+str(a))
#         plt.legend( bbox_to_anchor=(0.99, 0.63))    
#         plt.savefig(E+"_"+str(a)+"_KR_std-R"+".png")
#         
#         np.save(E+"_"+str(a)+"_KR-R_"+Name,np.array([X,Y,stdY,Rad,stdRad,Ad,stdAd]))
#         
# #Ergebnisse in Datei ausgeben
# writeFile(fileName_+"_"+str(date)+".txt", Liste, Legende)
# np.save(fileName_+"_"+str(date), np.array(Liste))
# print(ZählerGesamt)
# print(ZählerGut)
# print(ZählerSchlecht)
# print(ZählerSchlechtGut)
# 
# 
#==============================================================================


#==============================================================================
# #OSRM KR Seesen
# D= np.load("OSRM_distance"+"_[ 10.11269   51.887916]"+"_KR-R"+".npy")
# print(D[1])
# #mean
# 
# X = D[3]/1000
# Y= D[1]
# stdY = D[2]
# 
# plt.plot(X,Y,label = "mean" )
# #plt.plot(X,np.log(np.array(stdY)),label = "std" )
# #plt.xlabel("Breitengrad")
# plt.xlabel("Radius Polygon [km]")
# plt.ylabel("Krümmung [1/m²]")    
# plt.title("OSRM_distance")
# #plt.xlim(xmin = 200, xmax = 400)
# plt.legend( bbox_to_anchor=(0.99, 0.63))    
# plt.show()
# 
# #std
# 
# plt.plot(X,np.log(np.array(stdY)),label = "std" )
# #plt.xlabel("Breitengrad")
# plt.xlabel("Radius Polygon [km]")
# plt.ylabel("Krümmung log [1/m²]")    
# plt.title("OSRM_distance")
# plt.legend( bbox_to_anchor=(0.99, 0.63))    
# plt.show()
# 
# #std norm
# Y= np.divide(D[2],D[1])
# 
# plt.plot(X,Y,label = "std norm" )
# #plt.xlabel("Breitengrad")
# plt.xlabel("Radius Polygon [km]")
# plt.ylabel("Krümmung ")    
# plt.title("OSRM_distance")
# plt.legend( bbox_to_anchor=(0.99, 0.63))    
# plt.show()
# 
# 
# 
#==============================================================================
#==============================================================================
# #EInheitskugel
# D= np.load("Einheitskugel"+"_KR-R"+".npy")
# #print(D[1])
# 
# X = D[0]
# Y= D[1]
# stdY = D[2]
# 
# #plt.plot(X,np.log(Y),label = "mean" )
# plt.plot(X,np.log(np.array(stdY)),label = "std" )
# plt.xlabel("Breitengrad")
# plt.ylabel("Krümmung log ")    
# plt.title("Einheitskugel")
# plt.legend( bbox_to_anchor=(0.99, 0.63))    
# plt.show()
# 
#==============================================================================



#Daten auswerten
import pandas as pd

def X_meanY(datafile,x_name,y_name):
    X = np.sort(np.array(list(set(datafile[x_name].as_matrix()))))
    Y = []
    stdY = []    
    for x in X:
        L = np.float_(datafile.loc[datafile[x_name]==x][y_name].as_matrix())
        Y.append(np.mean(L))
        stdY.append(np.std(L))
    return np.float_(X), np.array(Y), np.array(stdY)
    

#==============================================================================
# #Krümmungen
# distanceU = "OSRM_duration"
# timestamp = "2018-10-23T13:24:29.572806"
# #==============================================================================
# # nameY = "normalized_curvature"#'curvature_K=R/S'
# # UX = "Polygon Radius [h]"
# # sx = 1/60/60
# # UY = ""
# # long = "10.164781"
# # lat =     "52.521398"
# #==============================================================================
# nameY = 'curvature_K=R/S'
# UX = "Polygon Radius [h]"
# sx = 1/60/60
# UY = "1/s²"
# long = "10.164781"
# lat =     "52.521398"
# 
# 
# header = Legende.split(", ")
# print (header)
# 
# D = np.load(fileName_+"_"+timestamp+".npy")
# 
# df = pd.DataFrame(D, columns = header)
# 
# df = df.loc[df['measuring_unit']==distanceU]
#     
# df = df.loc[df['center_longitude']==long]
# df = df.loc[df['center_latitude']==lat]
#     
# r,Y, stdY = X_meanY(df,"planed_euklidin_radius_[]",nameY)
# r,X,stdX = X_meanY(df,"planed_euklidin_radius_[]",'average_radius_in_lengt_units')
# 
# 
# #print(X)
# #print(Y)
# 
# plt.plot(X*sx,Y, label = "mean "+ nameY)
# plt.xlabel(UX)
# plt.ylabel(UY)
# plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
# plt.legend() 
# # plt.ylabel("Krümmung log ")   
# plt.show()
# 
# plt.plot(X*sx,stdY, label = "std "+nameY)
# plt.xlabel(UX)
# plt.ylabel(UY)
# plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
# plt.legend( )
# # plt.ylabel("Krümmung log ")   
# plt.show()
# 
# 
# plt.plot(X*sx,stdY/Y, label = "normed std "+nameY)
# plt.xlabel(UX)
# plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
# plt.legend( )
# # plt.ylabel("Krümmung log ")   
# plt.show()
#==============================================================================

##Eta

distanceU = "OSRM_distance"
timestamp = "2018-10-22T11:40:30.355350"#"2018-10-23T13:24:29.572806"
nameY ='curvature_K=R/S'
UX = "Polygon Radius [km]"
sx = 1/1000
UY = ""#"1/s²"
long = "10.164781"
lat =     "52.521398"


header = Legende.split(", ")
print (header)

D = np.load(fileName_+"_"+timestamp+".npy")

df = pd.DataFrame(D, columns = header)

df = df.loc[df['measuring_unit']==distanceU]
    
df = df.loc[df['center_longitude']==long]
df = df.loc[df['center_latitude']==lat]
    
r,Y, stdY = X_meanY(df,"planed_euklidin_radius_[]",nameY)
r,X,stdX = X_meanY(df,"planed_euklidin_radius_[]",'average_radius_in_lengt_units')


print(X)
print(Y)

Y = X*np.sqrt(np.absolute(Y))

print(Y)

plt.plot(X*sx,Y, label = "mean "+ nameY)
plt.xlabel(UX)
plt.ylabel(UY)
plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
plt.legend() 
# plt.ylabel("Krümmung log ")   
plt.show()

plt.plot(X*sx,stdY, label = "std "+nameY)
plt.xlabel(UX)
plt.ylabel(UY)
plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
plt.legend( )
# plt.ylabel("Krümmung log ")   
plt.show()


plt.plot(X*sx,stdY/Y, label = "normed std "+nameY)
plt.xlabel(UX)
plt.title(distanceU+",Mittelpunkt=["+lat+","+long+"]")
plt.legend( )
# plt.ylabel("Krümmung log ")   
plt.show()

#print(df.where(df["number_edges"]=3))






