#by Vahid

from models import *
from arguments import *

args = get_args()


np.random.seed(4)
path=args.path

"""# **Data loading**"""
train_data, test_data = data_loading(mat_file=path, train_test_ratio=4)

mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, len(train_data)*5).reshape(-1,5)

train_noisy= train_data + noise



# Generating  train and real_test , and noisy train dataframes
train, test, train_n = create_dataframe(train_data), create_dataframe(test_data), create_dataframe(train_noisy)
# print(train.describe())
# print(test.describe())
# print(train_n.describe())

#normalizing the data
train, test, train_n = normalize(train, test , train_n)
print(train.describe())
print(test.describe())
print(train_n.describe())
# train.plot(kind= 'bar')
# test.plot(kind= 'bar')
# train_n.plot(kind= 'bar')

# Generating  faulty_test dataframe
test_faulty= fault_generation(test.copy(), type='Complete_failure')




args.model="MAE"
MAE= model(args, train, train_n, test, test_faulty, norm=1)
MAE.train_model()
MAE.reconstruct(train, description="train")
MAE.reconstruct(test, description="test")

args.model="AE"
AE= model(args, train, train_n, test, test_faulty, norm=1)
AE.train_model()
AE.reconstruct(train, description="train")
AE.reconstruct(test, description="test")

args.model="DAE" 
DAE= model(args, train, train_n, test, test_faulty, norm=1)
DAE.train_model()
DAE.reconstruct(train, description="train")
DAE.reconstruct(test, description="test")


args.model="VAE"
VAE= model(args, train, train_n, test, test_faulty, norm=1)
VAE.train_model()
VAE.reconstruct(train, description="train")
VAE.reconstruct(test, description="test")

args.model="MVAE"
MVAE= model(args, train, train_n, test, test_faulty, norm=1)
MVAE.train_model()                             
MVAE.reconstruct(train, description="train")
MVAE.reconstruct(test, description="test")



