#import liberaries
import pandas as pd
import numpy as np
import random
import math
import scipy.optimize as spo

class K_MEAN():
    random.seed(101)

    def __init__(self,df):
        self.df = df
        self.centers = []

    def __euclidian_distance(self,a,b):
        dif = a-b
        return math.sqrt(np.dot(dif,dif))

    def __distance_center_cluster(self,center,df):
        sum = 0
        for key,line in df.iterrows():
            sum += self.__euclidian_distance(center,np.array(line))
        return sum

    def train(self,n):

        information_lines = self.df.copy()
        df_with_clusters = self.df.copy()
        df_with_clusters.loc[:,['cluster']] = 0
        information_lines = information_lines.iloc[:,1:]

        #chose the inicial points
        while(1):
            index = random.choices([i for i in range(len(self.df))],k = n)
            if len(index) == len(set(index)):
                break

        old_centers = np.array(self.df.iloc[index,1:])

        while(1):
            cluster = []
            new_centers = []

            #Clustering the lines
            for key,line in information_lines.iterrows():
                menor = np.inf
                index_menor = 0
                array_line = np.array(line)

                #Calculate the distances
                for i in range(n):
                    distance = self.__euclidian_distance(array_line,old_centers[i])
                    if distance < menor:
                        index_menor = i+1
                        menor = distance

                cluster.append(index_menor)

            df_with_clusters.loc[:,'cluster'] = cluster

            #Recalculate centers
            for i in range(n):
                result = spo.minimize(self.__distance_center_cluster,old_centers[i],args = (df_with_clusters.loc[df_with_clusters.cluster == i+1].iloc[:,1:11]))
                new_centers.append(result.x)

            #Verifify if centers has change
            change = False
            for i in range(n):
                new_centers[i] = list(map(lambda x: round(x,1),new_centers[i]))
                if abs(new_centers[i]-old_centers[i]).sum() != 0:
                    change = True
                    old_centers[i] = new_centers[i]
            
            if change == False:
                self.centers = new_centers
                self.df = df_with_clusters
                break

    def sum_squared_erros(self):
        n = len(self.centers)
        distance = 0
        for i in range(1,n+1):
            distance += self.__distance_center_cluster(self.centers[i-1],self.df.loc[self.df.cluster == i].iloc[:,1:11])
        return distance/len(self.df)
