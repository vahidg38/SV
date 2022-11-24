#by Vahid
#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_autoencoder/
import numpy as np

from models import *
from arguments import *
from sklearn.decomposition import PCA

TO_TRAIN=False
TO_TEST=True


args = get_args()


np.random.seed(4)
path=args.path

"""# **Data loading**"""
train_data, test_data = data_loading(mat_file=path, train_test_ratio=4)
# train_data, test_data = data_loading_csv()

mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, len(train_data)*5).reshape(-1,5)
# noise = np.random.normal(mu, sigma, len(train_data)*4).reshape(-1,4)
# noise=np.insert(noise, 0, [0], axis=1)
train_noisy= train_data + noise



# Generating  train and real_test , and noisy train dataframes
sensors_=['PM25', 'PM10', 'CO2', 'Temp', 'Humidity']
train_ori, test_ori, train_n_ori = create_dataframe(train_data, sensors=sensors_), create_dataframe(test_data,sensors=sensors_), create_dataframe(train_noisy,sensors=sensors_)
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

#test_faulty=None

#

args.model="integrated"
integrated_model= model(args, train, train_n, test, test_faulty)

if TO_TRAIN:
    integrated_model.train_model()

if TO_TEST:

    z, x,e=integrated_model.reconstruct(train,train_ori, description="train")
    z, x,ee=integrated_model.reconstruct(test,test_ori ,description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z, x, ee = integrated_model.reconstruct(test_faulty, test_ori, description=args.failure)
    # e = np.array(e)
    #
    # print(np.shape(e))
    #
    # spe_trn=[]
    # for i in range(len(e)):
    #  spe_trn.append(e[i]*e.T[:,i])
    #
    #
    # #spe_trn= np.matmul(e,e.T)
    #
    #
    # print(np.shape(spe_trn))


#raise Exception("Error!, age should be greater than 18!")
args.model="MAE"
MeAE= model(args, train, train_n, test, test_faulty)
#MAE.optimization()
if TO_TRAIN:
    MeAE.train_model()
if TO_TEST:
    MeAE.reconstruct(train,train_ori, description="train")
    z, x,e= MeAE.reconstruct(test,test_ori ,description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z, x, ee=MeAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    # print(f"MSE for {args.model}:  {MSE(z,x)}")
    # print(f"RR for {args.model}:  {RR(z,x)}")

    # for i in range(len(e["error"])//10):
    # for i in range(args.pstart,args.pstop):
    # for i in range(args.fstart-10, args.fstart+10):
    #     print(i)
    #     print(f"error sum: {np.sum(np.absolute(ee['error'][i][0]))}")
    #     print(f"error: {ee['error'][i][0]}")
    #     print(f"recons: {z.values[i]}")
    #     print(f" Input: {x.values[i]}")
    #     print("###############################")
    #




args.model="AE"
AE= model(args, train, train_n, test, test_faulty)
#AE.optimization()
if TO_TRAIN:
    AE.train_model()
if TO_TEST:
    AE.reconstruct(train,train_ori, description="train")
    z, x,e= AE.reconstruct(test, test_ori, description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z,x,e=AE.reconstruct(test_faulty,test_ori ,description=args.failure)
    # print(f"MSE for {args.model}:  {MSE(z,x)}")
    # print(f"RR for {args.model}:  {RR(z,x)}")


args.model="DAE"
DAE= model(args, train, train_n, test, test_faulty)
#DAE.optimization()
if TO_TRAIN:
    DAE.train_model()
if TO_TEST:
    DAE.reconstruct(train,train_ori, description="train")
    z, x,e= DAE.reconstruct(test,test_ori, description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z,x,e=DAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    # print(f"MSE for {args.model}:  {MSE(z,x)}")
    # print(f"RR for {args.model}:  {RR(z,x)}")


args.model="VAE"
VAE= model(args, train, train_n, test, test_faulty)
#VAE.optimization()
if TO_TRAIN:
    VAE.train_model()
if TO_TEST:
    VAE.reconstruct(train,train_ori ,description="train")
    z, x,e= VAE.reconstruct(test,test_ori, description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z,x,e=VAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    # print(f"MSE for {args.model}:  {MSE(z,x)}")
    # print(f"RR for {args.model}:  {RR(z,x)}")

args.model="MVAE"
MVAE= model(args, train, train_n, test, test_faulty)
#MVAE.optimization()
if TO_TRAIN:
    MVAE.train_model()
if TO_TEST:
    MVAE.reconstruct(train,train_ori, description="train")
    z, x,e= MVAE.reconstruct(test, test_ori, description="test")
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")
    z,x,e=MVAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    # print(f"MSE for {args.model}:  {MSE(z,x)}")
    # print(f"RR for {args.model}:  {RR(z,x)}")



