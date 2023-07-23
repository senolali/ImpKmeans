import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from scipy import stats
import scipy.io
import time
from fcmeans import FCM
from sklearn.metrics.cluster import adjusted_rand_score
from IPython import get_ipython
import warnings
from sklearn_extra.cluster import KMedoids
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')

def findInitialPoints(X,k,r):  
    points=np.empty((0,X.shape[1]),float)
    for j in range(k):
        krnl='gaussian'
        b_width=0.05
        kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
        x = kde.score_samples(X)
        i=np.argmax(x)
        points=np.vstack((points,X[i,:]))
        kdtree = KDTree(X)
        ind = kdtree.query_radius([X[i,:]], r)
        X = np.delete(X, ind[0][:],axis=0)
        if(X.size==0):
            break
    return points

def plotKDE(XX):
    xmin=XX[:,0].min()
    xmax=XX[:,0].max()
    ymin=XX[:,1].min()
    ymax=XX[:,1].max()
    X, Y = np.mgrid[xmin:xmax:600j, ymin:ymax:600j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([XX[:,0], XX[:,1]])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    iso = kernel((XX[:,0], XX[:,1]))
    c=iso.reshape(len(iso),1)
    A=np.column_stack((XX,c))
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    im=ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax],aspect='auto')
    ax.plot(XX[:,0], XX[:,1], 'k.', markersize=2)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    fig.colorbar(im)      
    
    
def plotGraph(X,labels,algo,index,index_value,dpi=80):
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams["figure.figsize"] = (16,16)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    # plt.subplots(figsize=(10, 10))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=9)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

    s=str(str(algo+" Clustering => {%s=%0.4f}"%(index,index_value))) 
    plt.title(s)
    plt.rcParams.update({'font.size': 7})
    plt.grid()

datasets={1,2,3,4,5}
for selected_dataset in datasets:
    if selected_dataset==1:
        data = scipy.io.loadmat("Datasets/outlier.mat")
        data=data["outlier"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="Outliers"
        k=4
        r=0.4
    elif selected_dataset==2:
        data = np.loadtxt("Datasets/Aggregation.txt",delimiter=',', dtype=float)
        X=data[:,0:2]
        labels_true = data[:,2]
        labels_true=np.ravel(labels_true)
        dataset_name="Aggregation"
        k=7
        r=0.4
    elif selected_dataset==3:
        data = np.loadtxt("Datasets/2d-20c.txt", delimiter=',', dtype=float)
        X=data[:,0:2]
        labels_true=data[:,2]
        dataset_name="2d-20c"
        k=20
        r=0.1
    elif selected_dataset==4:
        data = np.loadtxt("Datasets/fourty.txt", delimiter=',', dtype=float)
        X=data[:,0:2]
        labels_true=data[:,2]
        dataset_name="Fourty"
        k=40
        r=0.025
    elif selected_dataset==5:
        data = np.loadtxt("Datasets/s-set1.txt", delimiter=',', dtype=float)
        X=data[:,0:2]
        labels_true=data[:,2]
        dataset_name="S-set1"
        k=15
        r=0.2
    
    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X)

    print("\n\n######-Results for %s dataset-##############"%dataset_name)   
    #############################ImpKmeans (Proposed Algorithm) ##################################################    
    start = time.time()
    initial=findInitialPoints(X,k,r)
    kmeans = KMeans(n_clusters=initial.shape[0], init=initial,max_iter=1).fit(X)
    end = time.time() 
    centers=kmeans.cluster_centers_
    Imp_labels=kmeans.labels_
    Imp_ARI=adjusted_rand_score(labels_true.reshape(-1), Imp_labels)
    print("ImpKmeans ARI=",Imp_ARI)
    time_ImpKmeans=end-start
    print("Elapsed time=%.8f\n"%time_ImpKmeans)
    #plotKDE##
    if(X.shape[1]==2):
        plotKDE(X)
        plt_name=str("results/"+dataset_name+"_KDE.png")
        plt.savefig(plt_name,dpi=500,bbox_inches='tight')
        plt.show()
        
    
    #############################LLoyd's kmeans ##################################################
    start = time.time()
    kmeans = KMeans(n_clusters=k, init ='random',max_iter=100,n_init=10,random_state=0).fit(X) #init{‘k-means++’, ‘random’}, 
    end = time.time() 
    centers=kmeans.cluster_centers_
    kmeans_labels=kmeans.labels_
    kmeans_ARI=adjusted_rand_score(labels_true.reshape(-1), kmeans_labels)
    print("kmeans ARI=",kmeans_ARI)
    time_Kmeans=end-start
    print("Elapsed time=%.8f\n"%time_Kmeans)
    #############################################################################
    
    #############################kmeans++ ##################################################
    start = time.time()
    kmeans = KMeans(n_clusters=k, init ='k-means++',max_iter=100,n_init=10,random_state=0).fit(X) #init{‘k-means++’, ‘random’}, 
    end = time.time()
    centers=kmeans.cluster_centers_
    pp_labels=kmeans.labels_
    pp_ARI=adjusted_rand_score(labels_true.reshape(-1), pp_labels)
    print("kmeans++ ARI=",pp_ARI)
    time_Kmeanspp=end-start
    print("Elapsed time=%.8f\n"%time_Kmeanspp)
    #############################################################################
    
    ############################# kmedoids ##################################################
    start = time.time()
    cobj = KMedoids(n_clusters=k).fit(X)
    end = time.time()
    medoids_labels = cobj.labels_
    centers=cobj.cluster_centers_
    medoids_ARI=adjusted_rand_score(labels_true.reshape(-1), medoids_labels)
    print("k-medoids ARI=",medoids_ARI)
    time_Kmedoids=end-start
    print("Elapsed time=%.8f\n"%time_Kmedoids)
    
    ############################# FCM ##################################################
    start = time.time()
    fcm = FCM(n_clusters=k)
    end=time.time()
    fcm.fit(X)
    centers = fcm.centers
    FCM_labels = fcm.predict(X)
    time_FCM=end-start
    FCM_ARI=adjusted_rand_score(labels_true.reshape(-1), FCM_labels)
    print("FCM ARI=",FCM_ARI)
    print("Elapsed time=%.8f\n"%time_FCM)
    ##############################################################################
    
    f, axarr = plt.subplots(nrows=1,ncols=5,sharex=True, sharey=True)
    f.set_figheight(4)
    f.set_figheight(4)
 
    plt.sca(axarr[0]); 
    plotGraph(X,kmeans_labels,"k-means","ARI",kmeans_ARI)
        
    plt.sca(axarr[1]); 
    plotGraph(X,medoids_labels,"k-medoids","ARI",medoids_ARI)
    
    plt.sca(axarr[2]);
    plotGraph(X,pp_labels,"k-means++","ARI",pp_ARI)

    plt.sca(axarr[3]);
    plotGraph(X,FCM_labels,"FCM","ARI",FCM_ARI)

    plt.sca(axarr[4]);
    plotGraph(X,Imp_labels,"ImpKmeans","ARI",Imp_ARI)

    plt_name=str("results/"+dataset_name+".png")
    plt.savefig(plt_name)
    plt.subplots_adjust(hspace=0.01, wspace=.01)
    a=str(dataset_name+' dataset')
    plt.setp(axarr[0], ylabel=a)
    axarr[0].yaxis.label.set_size(16)
    
    plt.savefig(plt_name,dpi=500,bbox_inches='tight') 
    plt.show()

    
    
    

        
