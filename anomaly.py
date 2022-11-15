from models import *
from arguments import *
#https://towardsdatascience.com/anomaly-detection-for-time-series-with-monte-carlo-simulations-e43c77ba53c
from sklearn.cluster import KMeans
#https://app.neptune.ai/theaayushbajaj/Anomaly-Detection/n/Anomaly-Detection-49ba1752-fc3a-4abb-b35f-0e2ea4fd4afa/48dc19d8-3c75-4989-a2c0-67839393a093
args = get_args()

#https://github.com/udohsolomon/anomaly-detection/blob/master/anomaly_detection.py
np.random.seed(4)
path=args.path

"""# **Data loading**"""
train_data, test_data = data_loading(mat_file=path, train_test_ratio=4)

mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, len(train_data)*5).reshape(-1,5)

train_noisy= train_data + noise



# Generating  train and real_test , and noisy train dataframes
train_ori, test_ori, train_n_ori = create_dataframe(train_data), create_dataframe(test_data), create_dataframe(train_noisy)
print(train_ori.describe())
print(test_ori.describe())
print(train_n_ori.describe())

#normalizing the data

train, test, train_n = normalize(train_ori, test_ori, train_n_ori)
# print(train.describe())
# print(test.describe())
# print(train_n.describe())

# Generating  faulty_test dataframe
test_faulty= fault_generation(test.copy(), type=args.failure, sensor=args.fsensor, magnitude=args.fmagnitude, start=args.fstart, stop=args.fstop)

from sklearn.decomposition import PCA
pca = PCA(2)
x_pca = pca.fit_transform(train)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']
cdict = {0: 'red', 1: 'blue'}
# Plot
import matplotlib.pyplot as plt
plt.scatter(x_pca['PC1'], x_pca['PC2'])
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# data = test_faulty
# n_cluster = range(1, 20)
# kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
# scores = [kmeans[i].score(data) for i in range(len(kmeans))]
# fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(n_cluster, scores)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Score')
# plt.title('Elbow Curve')
# plt.show();
#
