from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.2,min_samples = 4 )
y_predicted_dbscan = dbscan.fit_predict(df_k)
df_k['clusters_dbscan'] = y_predicted_dbscan
df_k

from sklearn.metrics import silhouette_samples, silhouette_score
score = silhouette_score(df_k, dbscan.labels_, metric='euclidean')
print(score)