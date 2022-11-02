# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
from utils import *
from models import *
from statsmodels.graphics.tsaplots import plot_acf
from arguments import *

args = get_args()


np.random.seed(4)
path=args.path

"""# **Data loading**"""
train_data, test_data, norm = data_loading(mat_file=path)

# Generating  train and real_test dataframes
train, test = create_dataframe(train_data, test_data)
# Generating  faulty_test dataframe
test_faulty= fault_generation(test.copy(), type='Complete_failure')
# Generating  noisy train dataframe
train_n= fault_generation(train.copy(), type='Degradation', magnitude=1, start=0, stop=len(train))  # noisy train data

model= model(args, train, train_n, test, test_faulty)
#model train
# model reconstruct


model.train_model()

raise Exception("Inappropriate failure type.")



model_para = torch.load('./best_model_snap.pt') #memae
model.load_state_dict(model_para)

model_para = torch.load('./best_model_2_snap.pt') #ae
model_2.load_state_dict(model_para)

model_para = torch.load('./best_model_3_snap.pt') #dae
model_3.load_state_dict(model_para)

model_para = torch.load('./best_model_4_snap.pt') #dae
model_4.load_state_dict(model_para)

model_para = torch.load('./best_model_5_snap.pt') #dae
model_5.load_state_dict(model_para)

"""def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    plt.figure(figsize = (5,4), dpi = 450)
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    # plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=15, c=dbscan.labels_[core_mask])
    # plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    # plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("t-SNE parameter 1", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("t-SNE parameter 2", fontsize=14)
    else:
        plt.tick_params(labelleft=False)

latent_mae=np.array(latent_mae)
d = sio.loadmat('latent_mae.mat') 
print(d['latent_mae'])
"""

bars=df.columns

rec_train=[]
rec_train_2=[]
rec_train_3=[]
rec_train_4=[]
rec_train_5=[]
lat_mae=[]
lat_ae=[]
lat_dae=[]
lat_vae=[]
lat_mvae=[]

c=len(df)

for i in range(len(df)) :
    obs= torch.from_numpy(df.iloc[i+0:i+1].to_numpy())
    reconstructed=model(obs.float())
    reconstructed_2=model_2(obs.float())
    reconstructed_3=model_3(obs.float())
    reconstructed_4=model_4(obs.float())
    reconstructed_5=model_5(obs.float())
    
    rec_train.append(reconstructed['output'].detach().numpy()[0] )
    lat_mae.append(reconstructed['latent'].detach().numpy()[0])
    rec_train_2.append(reconstructed_2['output'].detach().numpy()[0] )
    lat_ae.append(reconstructed_2['latent'].detach().numpy()[0])
    rec_train_3.append(reconstructed_3['output'].detach().numpy()[0] )
    lat_dae.append(reconstructed_3['latent'].detach().numpy()[0])
    rec_train_4.append(reconstructed_4['output'].detach().numpy() [0]  )
    lat_vae.append(reconstructed_4['latent'].detach().numpy()[0])
    rec_train_5.append(reconstructed_5['output'].detach().numpy()[0] )
    lat_mvae.append(reconstructed_5['latent'].detach().numpy()[0])
  
df_rec_memae=pd.DataFrame(rec_train, columns=df.columns)
df_rec_ae=pd.DataFrame(rec_train_2, columns=df.columns)
df_rec_dae=pd.DataFrame(rec_train_3, columns=df.columns)
df_rec_vae=pd.DataFrame(rec_train_4, columns=df.columns)
df_rec_mvae=pd.DataFrame(rec_train_5, columns=df.columns)

models=['memae','ae','dae','vae','mvae']
 #print(reconstructed['output'].detach().numpy()[0])
if not os.path.exists('./train'):
  os.mkdir('./train') 
for b in df.columns:
  for m in models:

    
    if m=='memae':
        plt.plot(df_rec_memae[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_MemAE',marker='.')

        plt.plot(df[b].iloc[0:700], label='Actual',color='blue', marker='.')
        plt.legend()
        plt.xlabel(f"{b}_train_{m}")
        plt.title("train")
        plt.savefig(f'./train/{b}_train_{m}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()  
    elif m=='ae':
        plt.plot(df_rec_ae[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_AE',marker='.') 

        plt.plot(df[b].iloc[0:700], label='Actual',color='blue', marker='.')
        plt.legend()
        plt.xlabel(f"{b}_train_{m}")
        plt.title("train")
        plt.savefig(f'./train/{b}_train_{m}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()  
    elif m=='dae':
        plt.plot(df_rec_dae[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_DAE',marker='.')

        plt.plot(df[b].iloc[0:700], label='Actual',color='blue', marker='.')
        plt.legend()
        plt.xlabel(f"{b}_train_{m}")
        plt.title("train")
        plt.savefig(f'./train/{b}_train_{m}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()   
    elif m=='vae':
        plt.plot(df_rec_vae[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_VAE',marker='.')   

        plt.plot(df[b].iloc[0:700], label='Actual',color='blue', marker='.')
        plt.legend()
        plt.xlabel(f"{b}_train_{m}")
        plt.title("train")
        plt.savefig(f'./train/{b}_train_{m}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()  
    elif m=='mvae':
        plt.plot(df_rec_mvae[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_MVAE',marker='.')

        plt.plot(df[b].iloc[0:700], label='Actual',color='blue', marker='.')
        plt.legend()
        plt.xlabel(f"{b}_train_{m}")
        plt.title("train")
        plt.savefig(f'./train/{b}_train_{m}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()    


   # plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='red',label='Reconstructed_MemAE',marker='.')
   # plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='pink',label='Reconstructed_AE',marker='.')
   # plt.plot(df_rec_3[b].iloc[0:700],linestyle = 'dotted', color='green',label='Reconstructed_DAE',marker='.')
   # plt.plot(df_rec_4[b].iloc[0:700],linestyle = 'dotted', color='orange',label='Reconstructed_VAE',marker='.')
   # plt.plot(df_rec_5[b].iloc[0:700],linestyle = 'dotted', color='orange',label='Reconstructed_MVAE',marker='.')

   # plt.legend()
    #plt.xlabel(b+"_train")
    #plt.show()
   # plt.title("train")
   # plt.savefig(f'./train_models_{b}.jpg', bbox_inches="tight", pad_inches=0.0) 
    #plt.show()
    #plt.clf()    

rec=np.array(rec_train[:7000])
sio.savemat("./reconst_train_memae.mat", {"z_train":  rec})
rec=np.array(rec_train_2[:7000])
sio.savemat("./reconst_train_ae.mat", {"z_train":  rec})
rec=np.array(rec_train_3[:7000])
sio.savemat("./reconst_train_dae.mat", {"z_train":  rec})
rec=np.array(rec_train_4[:7000])
sio.savemat("./reconst_train_vae.mat", {"z_train":  rec})
rec=np.array(rec_train_5[:7000])
sio.savemat("./reconst_train_mvae.mat", {"z_train":  rec})


rec=np.array(lat_mae[:7000])
sio.savemat("./reconst_latent_memae.mat", {"latent":  rec})
rec=np.array(lat_ae[:7000])
sio.savemat("./reconst_latent_ae.mat", {"latent":  rec})
rec=np.array(lat_dae[:7000])
sio.savemat("./reconst_latent_dae.mat", {"latent":  rec})
rec=np.array(lat_vae[:7000])
sio.savemat("./reconst_latent_vae.mat", {"latent":  rec})
rec=np.array(lat_mvae[:7000])
sio.savemat("./reconst_latent_mvae.mat", {"latent":  rec})
#data22 = sio.loadmat('reconst_train.mat')
#print(rec[:,2])

rec_test=[]
rec_test_2=[]
rec_test_3=[]
rec_test_4=[]
rec_test_5=[]

lat_mae=[]
lat_ae=[]
lat_dae=[]
lat_vae=[]
lat_mvae=[]


for i in range(len(df_faulty)) :
      obs= torch.from_numpy(df_faulty.iloc[i+0:i+1].to_numpy())
      reconstructed=model(obs.float())
      reconstructed_2=model_2(obs.float())
      reconstructed_3=model_3(obs.float())
      reconstructed_4=model_4(obs.float())
      reconstructed_5=model_5(obs.float())

      rec_test.append(reconstructed['output'].detach().numpy()[0] )
      lat_mae.append(reconstructed['latent'].detach().numpy()[0])
      rec_test_2.append(reconstructed_2['output'].detach().numpy()[0] )
      lat_ae.append(reconstructed_2['latent'].detach().numpy()[0])
      rec_test_3.append(reconstructed_3['output'].detach().numpy()[0] )
      lat_dae.append(reconstructed_3['latent'].detach().numpy()[0])
      rec_test_4.append(reconstructed_4['output'].detach().numpy()[0] )
      lat_vae.append(reconstructed_4['latent'].detach().numpy()[0])
      rec_test_5.append(reconstructed_5['output'].detach().numpy()[0] )
      lat_mvae.append(reconstructed_5['latent'].detach().numpy()[0])
    # print(reconstructed['att'])
#print(rec_test_4)    
df_rec=pd.DataFrame(rec_test, columns=df.columns)
df_rec_2=pd.DataFrame(rec_test_2, columns=df.columns)
df_rec_3=pd.DataFrame(rec_test_3, columns=df.columns)
df_rec_4=pd.DataFrame(rec_test_4, columns=df.columns)
df_rec_5=pd.DataFrame(rec_test_5, columns=df.columns)
  #print(reconstructed['output'].detach().numpy()[0])


rec=np.array(rec_test)
sio.savemat("./reconst_test_memae.mat", {"z_test": rec})  

rec=np.array(rec_test_2)
sio.savemat("./reconst_test_ae.mat", {"z_test": rec}) 

rec=np.array(rec_test_3)
sio.savemat("./reconst_test_dae.mat", {"z_test": rec}) 

rec=np.array(rec_test_4)
sio.savemat("./reconst_test_vae.mat", {"z_test": rec}) 

rec=np.array(rec_test_5)
sio.savemat("./reconst_test_mvae.mat", {"z_test": rec}) 




rec=np.array(lat_mvae)
sio.savemat("./reconst_test_latent_memae.mat", {"latent_t":  rec})
rec=np.array(lat_ae)
sio.savemat("./reconst_test_latent_ae.mat", {"latent_t":  rec})
rec=np.array(lat_dae)
sio.savemat("./reconst_test_latent_dae.mat", {"latent_t":  rec})
rec=np.array(lat_vae)
sio.savemat("./reconst_test_latent_vae.mat", {"latent_t":  rec})
rec=np.array(lat_mvae)
sio.savemat("./reconst_test_latent_mvae.mat", {"latent_t":  rec})

"""just run for each model everytime to get better fig"""

if not os.path.exists('./test'):
  os.mkdir('./test') 

for b in df.columns:
      plt.figure(figsize=(10, 6))
      plt.plot(df_real[b].iloc[0:700], label='Actual',color='blue', alpha=0.5,marker='.')
      plt.plot(df_faulty[b].iloc[0:700], label='Faulty', color='black',alpha=0.6,marker='.')  

      ##plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MemAE',alpha=1,marker="8")   # just run for each model everytime to get better fig
      #plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_AE',alpha=0.8,marker="8")
      plt.plot(df_rec_5[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MVAE',alpha=0.9,marker="8")
    
      plt.legend()
      plt.xlabel(b+"_test")
      
      plt.title("test")
      plt.savefig(f'./test/test_mvae_{b}.jpg', bbox_inches="tight", pad_inches=0.0)
     #plt.show() 
      plt.clf()

for b in df.columns:
      plt.figure(figsize=(10, 6))
      plt.plot(df_real[b].iloc[0:700], label='Actual',color='blue', alpha=0.5,marker='.')
      plt.plot(df_faulty[b].iloc[0:700], label='Faulty', color='black',alpha=0.6,marker='.')  

      ##plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MemAE',alpha=1,marker="8")   # just run for each model everytime to get better fig
      #plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_AE',alpha=0.8,marker="8")
      plt.plot(df_rec_4[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_VAE',alpha=0.9,marker="8")
    
      plt.legend()
      plt.xlabel(b+"_test")
      
      plt.title("test")
      plt.savefig(f'./test/test_vae_{b}.jpg', bbox_inches="tight", pad_inches=0.0)
     # plt.show() 
      plt.clf()

for b in df.columns:
      plt.figure(figsize=(10, 6))
      plt.plot(df_real[b].iloc[0:700], label='Actual',color='blue', alpha=0.5,marker='.')
      plt.plot(df_faulty[b].iloc[0:700], label='Faulty', color='black',alpha=0.6,marker='.')  

      plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MemAE',alpha=1,marker="8")   # just run for each model everytime to get better fig
      #plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_AE',alpha=0.8,marker="8")
      #plt.plot(df_rec_4[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_VAE',alpha=0.9,marker="8")
    
      plt.legend()
      plt.xlabel(b+"_test")
      
      plt.title("test")
      plt.savefig(f'./test/test_memae_{b}.jpg', bbox_inches="tight", pad_inches=0.0)
      #plt.show() 
      plt.clf()

for b in df.columns:
      plt.figure(figsize=(10, 6))
      plt.plot(df_real[b].iloc[0:700], label='Actual',color='blue', alpha=0.5,marker='.')
      plt.plot(df_faulty[b].iloc[0:700], label='Faulty', color='black',alpha=0.6,marker='.')  

      #plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MemAE',alpha=1,marker="8")   # just run for each model everytime to get better fig
      plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_AE',alpha=0.8,marker="8")
      #plt.plot(df_rec_3[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_DAE',alpha=0.9,marker="8")
    
      plt.legend()
      plt.xlabel(b+"_test")
      
      plt.title("test")
      plt.savefig(f'./test/test_ae_{b}.jpg', bbox_inches="tight", pad_inches=0.0)
     # plt.show() 
      plt.clf()

for b in df.columns:
      plt.figure(figsize=(10, 6))
      plt.plot(df_real[b].iloc[0:700], label='Actual',color='blue', alpha=0.5,marker='.')
      plt.plot(df_faulty[b].iloc[0:700], label='Faulty', color='black',alpha=0.6,marker='.')  

      #plt.plot(df_rec[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_MemAE',alpha=1,marker="8")   # just run for each model everytime to get better fig
      #plt.plot(df_rec_2[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_AE',alpha=0.8,marker="8")
      plt.plot(df_rec_3[b].iloc[0:700],linestyle = 'dotted', color='darkred',label='Reconstructed_DAE',alpha=0.9,marker="8")
    
      plt.legend()
      plt.xlabel(b+"_test")
      
      plt.title("test")
      plt.savefig(f'./test/test_dae_{b}.jpg', bbox_inches="tight", pad_inches=0.0)
     # plt.show() 
      plt.clf()

#np.array(rec_train).shape
#np.array(Xtrn).shape
"""
if not os.path.exists('./tsne2d'):
  os.mkdir('./tsne2d') 

tsne = TSNE(n_components=2)
ori_tsne = tsne.fit_transform(Xtrn)
rec_tsne = tsne.fit_transform(rec_train)
rec_tsne_2 = tsne.fit_transform(rec_train_2)
rec_tsne_3= tsne.fit_transform(rec_train_3)
rec_tsne_4 = tsne.fit_transform(rec_train_4)
rec_tsne_5 = tsne.fit_transform(rec_train_5)


ori = pd.DataFrame(ori_tsne , columns=["Dim1", "Dim2"])
rec = pd.DataFrame(rec_tsne , columns=["Dim1", "Dim2"])
rec_2 = pd.DataFrame(rec_tsne_2 , columns=["Dim1", "Dim2"])
rec_3 = pd.DataFrame(rec_tsne_3 , columns=["Dim1", "Dim2"])
rec_4 = pd.DataFrame(rec_tsne_4 , columns=["Dim1", "Dim2"])
rec_5 = pd.DataFrame(rec_tsne_5 , columns=["Dim1", "Dim2"])


sns.scatterplot(data=ori, x="Dim1", y="Dim2", label="original")
sns.scatterplot(data=rec_5, x="Dim1", y="Dim2" ,label="reconstructed_MVAE")
plt.legend()
plt.savefig(f'./tsne2d/mvae2d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()

sns.scatterplot(data=ori, x="Dim1", y="Dim2", label="original")
sns.scatterplot(data=rec, x="Dim1", y="Dim2" ,label="reconstructed_MemAE")
plt.legend()
plt.savefig(f'./tsne2d/Memae2d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

sns.scatterplot(data=ori, x="Dim1", y="Dim2", label="original")
sns.scatterplot(data=rec_2, x="Dim1", y="Dim2" ,label="reconstructed_AE")
plt.legend()
plt.savefig(f'./tsne2d/ae2d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

sns.scatterplot(data=ori, x="Dim1", y="Dim2", label="original")
sns.scatterplot(data=rec_3, x="Dim1", y="Dim2" ,label="reconstructed_DAE")
plt.legend()
plt.savefig(f'./tsne2d/dae2d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

sns.scatterplot(data=ori, x="Dim1", y="Dim2", label="original")
sns.scatterplot(data=rec_4, x="Dim1", y="Dim2" ,label="reconstructed_VAE")
plt.legend()
plt.savefig(f'./tsne2d/vae2d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()
"""
"""
if not os.path.exists('./tsne3d'):
  os.mkdir('./tsne3d') 

tsne = TSNE(n_components=3)
ori_tsne = tsne.fit_transform(Xtrn)
rec_tsne = tsne.fit_transform(rec_train)
rec_tsne_2 = tsne.fit_transform(rec_train_2)
rec_tsne_3= tsne.fit_transform(rec_train_3)
rec_tsne_4 = tsne.fit_transform(rec_train_4)
rec_tsne_5 = tsne.fit_transform(rec_train_5)


ori_3d = pd.DataFrame(ori_tsne , columns=["Dim1", "Dim2","Dim3"])
rec_3d = pd.DataFrame(rec_tsne , columns=["Dim1", "Dim2","Dim3"])
rec_2_3d = pd.DataFrame(rec_tsne_2 , columns=["Dim1", "Dim2","Dim3"])
rec_3_3d = pd.DataFrame(rec_tsne_3 , columns=["Dim1", "Dim2","Dim3"])
rec_4_3d = pd.DataFrame(rec_tsne_4 , columns=["Dim1", "Dim2","Dim3"])
rec_5_3d = pd.DataFrame(rec_tsne_5 , columns=["Dim1", "Dim2","Dim3"])
    

axes = plt.axes(projection='3d')

axes.scatter3D(ori_3d["Dim1"], ori_3d["Dim2"], ori_3d["Dim3"], label='original')
axes.scatter3D(rec_5_3d["Dim1"], rec_5_3d["Dim2"], rec_5_3d["Dim3"], label='reconstructed_MVAE"')

axes.set_xlabel('Dim1')
axes.set_ylabel('Dim2')
axes.set_zlabel('Dim3')

plt.legend()
plt.savefig(f'./tsne3d/Mvae3d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()

axes = plt.axes(projection='3d')

axes.scatter3D(ori_3d["Dim1"], ori_3d["Dim2"], ori_3d["Dim3"], label='original')
axes.scatter3D(rec_3d["Dim1"], rec_3d["Dim2"], rec_3d["Dim3"], label='reconstructed_MemAE"')

axes.set_xlabel('Dim1')
axes.set_ylabel('Dim2')
axes.set_zlabel('Dim3')

plt.legend()
plt.savefig(f'./tsne3d/Memae3d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

axes = plt.axes(projection='3d')
axes.scatter3D(ori_3d["Dim1"], ori_3d["Dim2"], ori_3d["Dim3"], label='original')
axes.scatter3D(rec_2_3d["Dim1"], rec_2_3d["Dim2"], rec_2_3d["Dim3"], label='reconstructed_AE"')

axes.set_xlabel('Dim1')
axes.set_ylabel('Dim2')
axes.set_zlabel('Dim3')

plt.legend()
plt.savefig(f'./tsne3d/ae3d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

axes = plt.axes(projection='3d')
axes.scatter3D(ori_3d["Dim1"], ori_3d["Dim2"], ori_3d["Dim3"], label='original')
axes.scatter3D(rec_3_3d["Dim1"], rec_3_3d["Dim2"], rec_3_3d["Dim3"], label='reconstructed_DAE"')

axes.set_xlabel('Dim1')
axes.set_ylabel('Dim2')
axes.set_zlabel('Dim3')

plt.legend()
plt.savefig(f'./tsne3d/dae3d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

axes = plt.axes(projection='3d')
axes.scatter3D(ori_3d["Dim1"], ori_3d["Dim2"], ori_3d["Dim3"], label='original')
axes.scatter3D(rec_4_3d["Dim1"], rec_4_3d["Dim2"], rec_4_3d["Dim3"], label='reconstructed_VAE"')

axes.set_xlabel('Dim1')
axes.set_ylabel('Dim2')
axes.set_zlabel('Dim3')

plt.legend()
plt.savefig(f'./tsne3d/vae3d.jpg', bbox_inches="tight", pad_inches=0.0)
plt.clf()
#plt.show()

"""