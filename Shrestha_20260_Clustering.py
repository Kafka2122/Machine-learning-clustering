import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import OPTICS
import sys

class data_clustering():

    def __init__(self,clf_opt='DBSCAN'):
        self.clf_opt=clf_opt

        # Selection of classifiers

    def get_data(self):

        # importing the training data file into a dataframe using pandas

        df =  pd.read_csv(r'data.csv', header=None) # change path if required

        return df

    def plot(self):

        # for plotting the data

        df = self.get_data()

        plt.scatter(df[0],df[1],s= 4,color='red')
        plt.legend()
        plt.show()

    def clustering_pipeline(self,df):

        # creating the different classifiers

        # K-Means Clustering

        if self.clf_opt=='kmeans':
            print('\n\t### Using Kmeans clustering  ### \n')
            cluster = Kmeans(n_clusters = 2, random_state = 42)
            label_predict = cluster.fit_predict(df)
        # Agglomerative clustering

        elif self.clf_opt == 'Agsingle':
            print('\n\t### Using Agglomerative clustering  ### \n')
            cluster = AgglomerativeClustering(linkage='single')
            label_predict = cluster.fit_predict(df)

        # Agglomerative clustering average linkage

        elif self.clf_opt == 'Agaverge':
            print('\n\t### Using Agglomerative clustering  ### \n')
            cluster = AgglomerativeClustering(linkage='average')
            label_predict = cluster.fit_predict(df)

        # Agglomerative clusterin complete

        elif self.clf_opt == 'Agcomplete':
            print('\n\t### Using Agglomerative clustering  ### \n')
            cluster = AgglomerativeClustering(linkage='complete')
            label_predict = cluster.fit_predict(df)

        # dbscan. This is the best clustering technique
        elif self.clf_opt == 'DBSCAN':
            print('\n\t### Using DBSCAN clustering  ### \n')
            cluster = DBSCAN(eps = 0.06)
            print(df)
            label_predict = cluster.fit_predict(df)
            df['label'] = label_predict

            x = df.query('label != 1 & label != 0') # spearating datapoints that are not labelled as 0 or 1
            df_new = pd.DataFrame(x)# creating a new dataframe with labels that are not 0 or 1
            df_new['label'] = np.repeat(0,229) # giving all the labels 0 for easy computation
            array2 = np.array(df_new) # changing the dataframe into array

            y = df.query('label == 1 or label == 0') # filter out labels with 0 or 1 
            df_with_0_1 = pd.DataFrame(y) # creating a new dataframe with labels that are 0 or 1
            df_with_0_1 = df_with_0_1.reset_index(drop = True)
            array1 = np.array(df_with_0_1) # changing the dataframe into array

            sum = []
            lbl = []
            actual_lbl = []
            for i in range(len(array2)):
                for j in range(len(array1)):
                    x = array1[j] - array2[i]
                    sum.append(x[0]**2 + x[1]**2)
                    lbl.append(x[2])
                min_pos = sum.index(min(sum))
                actual_lbl.append(lbl[min_pos])
                sum = []
                lbl = []

            df_new['label'] = actual_lbl

            for i in df_new.index:
                df['label'][i] = df_new['label'][i] 
            label_predict = df['label'].to_numpy()
            # spectral clustering

        elif self.clf_opt == 'spectral':
            print('\n\t### Using Spectral clustering  ### \n')
            cluster= SpectralClustering(n_clusters=2)
            label_predict = cluster.fit_predict(df)

            # optic clustering

        elif self.clf_opt == 'optic':
            print('\n\t### Using Optic clustering  ### \n')
            cluster= OPTICS(min_samples=7, xi=0.1, min_cluster_size=0.2)
            label_predict = cluster.fit_predict(df)

        return label_predict,df

    #Write clusters to .txt file   
     
    def output_file(self,label_predict):
        with open("output.txt", "w") as f:
            for i in label_predict:
                f.write(str(i)+'\n')   