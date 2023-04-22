from Shrestha_20260_Clustering import data_clustering

clf_opt = 'DBSCAN'

cluster =  data_clustering(clf_opt)

cluster.plot()
x = cluster.get_data()
label_pred, df2  = cluster.clustering_pipeline(x)
cluster.output_file(label_pred)
print(df2)
