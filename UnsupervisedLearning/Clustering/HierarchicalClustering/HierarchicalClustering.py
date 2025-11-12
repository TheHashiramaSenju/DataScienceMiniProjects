from sklearn.cluster import AgglomerativeClustering
hierarchical = AgglomerativeClustering(n_clusters=3)
y_predicted_hierarchical = hierarchical.fit_predict(df_k)
df_k['clusters_hierarchical']= y_predicted_hierarchical
df_k.head(20)

