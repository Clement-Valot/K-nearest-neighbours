# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:11:48 2020

@author: valcr
"""
import random
import time
import numpy as np

#On crée notre liste des différents types de fleur
types_fleur=["Iris-setosa","Iris-virginica","Iris-versicolor"]
dim=len(types_fleur)
    
#Méthode pour extraire les données du fichier et les mettre dans une liste de liste
def Extraire_Dataset():
    data = []
    myFile = open("iris.data","r")
    for line in myFile:
        tab_UneLigne = line.split(',')
        for i in range(4):
            if(tab_UneLigne[i].isnumeric == False or line ==" " or line=="\n"):   
                break
            tab_UneLigne[i]=float(tab_UneLigne[i])   
            tab_UneLigne[len(tab_UneLigne)-1] = tab_UneLigne[len(tab_UneLigne)-1].rstrip()#Enleve le \n
        data.append(tab_UneLigne)  
    myFile.close()
    return data

#Méthode qui calcule la distance euclidienne entre les 4 variables de 2 fleurs
#fd= une fleur du dataset
#fi= la fleur inconnue dont on cherche le type
def Distance_Euclidienne(fd,fi):
    return ((fd[0]-fi[0])**2+(fd[1]-fi[1])**2+(fd[2]-fi[2])**2+(fd[3]-fi[3])**2)**0.5

def Distance_Manhattan(fd,fi):
    return abs(fd[0]-fi[0])+abs(fd[1]-fi[1])+abs(fd[2]-fi[2])+abs(fd[3]-fi[3])

#Pour trier la liste dans l'ordre croissant.
#Il est nécessaire de trier la liste dans la méthode apprentissage pour lorsque
#l'on fait intervenir la variable k qui prend les fleurs du dataset qui ont les 
#données les plus proches de notre fleur inconnue (distance plus proche de 0).
def Tri(liste):
    n = len(liste)
    for i in range (n):
        for j in range (0,n-i-1):
            if (liste[j][0] > liste[j+1][0]):
                liste[j], liste[j+1] = liste[j+1], liste[j]
    return liste

#Méthode qui retourne l'index de la liste où se trouve la plus grande valeur.
#On a besoin de cette fonction à la fin de la méthode apprentissage où, après 
#avoir compté les occurences de chaque type de fleur dans les k sélectionnées,
#on veut savoir quel type a le plus d'occurences. Mais comme on a deux listes
#différentes, une liste qui répertorie les types et une autre les occurences,
#on doit retourner l'index du max de la liste d'occurence pour la liste des types.
def Max(liste):
    index=0
    Max=liste[0]
    for i in range (len(liste)):
        if(liste[i]>Max):
            Max=liste[i]
            index=i
    return index

#! Méthode répartissant en 3 listes de données les données d'un fichier
#list_Data : Liste de listes contenant toute les données
#dimGp1 : taille de la liste destinée à l'apprentissage (plus grand nombre de valeurs)
#dimGp2 : taille de la liste destinée aux tests
#dimGp3 : taille de la liste destinée à l'évaluation finale
def Repartir_3Groupes(list_Data, dimGp1, dimGp2, dimGp3):
    
    size_App = [i for i in range(len(list_Data))] #Liste des index de 0 à 149
    #On sélectionne alléatoirement dimGp1 % d'index dans la liste des index
    index_Gp1 = random.sample(size_App, int((dimGp1/100)*len(list_Data)))
    index_Gp1.sort() #On tri la liste des index sélectionnés dans l'ordre croissant
    size_App = list(set(size_App)-set(index_Gp1)) #On supprime les index utilisés dans index_Gp1
    #Selection aleatoire d'index dans ceux restant
    index_Gp2 = random.sample(size_App, int((dimGp2/100)*len(list_Data)))
    index_Gp2.sort()
    size_App = list(set(size_App)-set(index_Gp2)) #On supprime les index utilisés dans index_Gp2
    
    index_Gp3 = random.sample(size_App, int((dimGp3/100)*len(list_Data)))
    index_Gp3.sort()
    size_App = list(set(size_App)-set(index_Gp3))  
    
    #On affecte les donnees suivant leur index
    list_Apprentissage = [list_Data[i] for i in index_Gp1]
    list_Test = [list_Data[i] for i in index_Gp2]
    list_Eval = [list_Data[i] for i in index_Gp3]  
    
    #Retourne 3 tableau de 80%/10%/10% des données si on rentre 80/10/10 en parametres
    return list_Apprentissage, list_Test, list_Eval

#Cette méthode renvoie la matrice de confusion. la matrice de confusion est 
#une matrice qui mesure la qualité d'un système de classification. Chaque ligne 
#correspond à une classe réelle, chaque colonne correspond à une classe estimée.
#Dans notre cas, on a 3 types d'iris différents donc notre matrice est de 
#dimension 3*3.
def Matrice_Confusion(dataset,k):
    # Les lignes représentent les valeurs réelles des classes des fleurs et les
    # colonnes représentent les valeurs estimées par notre méthode Apprentissage
    confusion=np.zeros((dim,dim))
    #On parcourt chaque fleur de notre dataset
    for i in range (len(dataset)):
        classe=dataset[i][4]
        #On parcourt les lignes (classes réelles) de notre matrice
        for j in range (dim):
            #On choisit quelle ligne on va parcourir
            if(classe==types_fleur[j]):
                #On parcourt les lignes (classes réelles) de notre matrice
                for l in range (dim):
                    #On choisit quelle colonne (donc case) on va incrémenter
                    if(types_fleur[l]==Apprentissage(dataset[i], dataset, k)):
                        confusion[j][l]+=1
    return confusion

#Méthode qui va faire la matrice de confusion pour chaque k de 1 au nombre de 
#données dans le dataset et renvoyer le meilleur k, cad le k pour lequel le
#nombre d'erreurs (somme des éléments qui ne sont pas sur la diagonale) est
#minimum.
#Cette méthode est très longue, c'est pour cela qu'il ne faut pas l'utiliser
#a chaque fois que l'on lance le programme.
#Dans notre cas, hormis 1, le meilleur k est 15.
def Choisir_k():
    t0=time.time()
    Min=10
    dataset=Extraire_Dataset()
    for k in range (2,int(len(dataset)/2)):
        matrice=Matrice_Confusion(dataset,k)
        nb_erreurs=Nbr_Erreurs(matrice)
        if(nb_erreurs<=Min):
            Min=nb_erreurs
            index=k
    t1=time.time()
    duree=t1-t0
    minutes=int(duree//60)
    secondes=round(duree%60)
    print("temps d'exécution pour trouver le meilleur k:",minutes,"min",secondes,"s.")
    return index

#Compte le nombre d'erreurs dans une matrice de confusion ie fait la somme des
#termes de la matrice qui ne sont pas sur sa diagonale.
def Nbr_Erreurs(matrice):
    nb_erreurs=0
    for i in range(dim):
       for j in range (dim):
            if(i!=j):
                nb_erreurs+=matrice[i,j]   
    return nb_erreurs
    
#Cette méthode renvoie deux types de précision:
#   - Une qui calcule la précision de la méthode apprentissage en se basant sur
#     un échantillon du dataset de taille choisie.
#   - Une autre qui va se baser sur la matrice de confusion du dataset complet 
def Precision(k):
    vrai=0
    PlA=70 #Part liste apprentissage
    PlT=15 #Part liste test
    PlE=100-PlA-PlT #Part liste évaluation
    dataset=Extraire_Dataset()
    list_Apprentissage, list_Test, list_Eval = Repartir_3Groupes(dataset, PlA, PlT, PlE)
    #Pour faire varier la précision on utilise moins de données dans list_Apprentissage (70% au lieu de 80%)
    #Et plus de données dans list_Test et list_Eval (15% && 15%)
    for i in list_Test: 
        #On test notre algo pour les valeurs de list_Test.
        #Valeurs inconnues par Apprentissage normalement car la méthode utilise uniquement list_Apprentissage
        fleur=i
        if(fleur[4]==Apprentissage(fleur[0:4], list_Apprentissage, k)):
            vrai+=1  
    precision=vrai/len(list_Test)*100
    print("On a eu",precision,"% de précision avec notre algorithme lorsqu'on a testé",PlT,"% du dataset.")
    print("Pour le dataset complet, on a la matrice de confusion suivante:")
    matrice=Matrice_Confusion(dataset,k)
    print(matrice)
    nb_erreurs=Nbr_Erreurs(matrice)
    print("Avec une précision de",(len(dataset)-nb_erreurs)/len(dataset)*100,"%.")
    
     
def Apprentissage(fleur, dataset,k):
    #Liste qui contiendra les valeurs des distances euclidiennes entre chaque
    #fleur du dataset et la fleur inconnue, associée avec le type de la fleur
    #du dataset analysée.
    #On a donc une liste de liste de deux éléments (distance, type)
    liste=[]
    for i in range (len(dataset)):
        distance=Distance_Euclidienne(dataset[i], fleur)
        #distance=Distance_Manhattan(dataset[i], fleur)
        #On associe le type à la distance
        donnees=[distance,dataset[i][4]]
        liste.append(donnees)
    #On trie la liste dans l'ordre croissant
    liste=Tri(liste)
    #On crée notre liste dans laquelle on va mettre les données (distance et type)
    #des k meilleures fleurs du dataset (celles dont la distance euclidienne
    #est la plus proche de 0).
    Analyse_Donnees=[]
    for i in range (k):
        Analyse_Donnees.append(liste[i])
    #On crée la liste des occurences de chaque type de fleur.
    #La case à l'index 0 correspond à l'occurence du type à l'index 0 dans types_fleur
    nbr_type=np.zeros(len(types_fleur))
    for i in range (k):
        for j in range (len(types_fleur)):
            if(Analyse_Donnees[i][1]==types_fleur[j]):
                nbr_type[j]+=1
    #On prend l'index de la liste des occurences pour lequel l'occurence est max
    index=Max(nbr_type)
    #on associe à fleur le type de la liste des types à l'index précédemment obtenu
    fleur=types_fleur[index]
    return fleur     
        
if __name__=='__main__' :
    k=15
    Precision(k)
    #print("Pour une matrice de confusion optimisé, il faudrait k =", Choisir_k())