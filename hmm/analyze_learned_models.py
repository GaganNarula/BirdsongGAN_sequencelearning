from hmm_utils import *
from glob import glob
import pdb
import matplotlib.pyplot as plt

BASEPATH = '/home/songbird/data/hmm_dump/batchnorm_net_nz32/'

# load some bird's models
#BIRDS  = ['b7r16', 'b14r16', 'k3r16', 'k6r16', 'b6r17', 'b8r17']
#tutorday = [9, 11, 4, 5, 15, 15]
BIRDS = ['b14r16']
tutorday = 11
match_str = '_K120_unchained'

def get_models():
    birdmodels = [None for _ in range(len(BIRDS))]
    birdscores = [None for _ in range(len(BIRDS))]
    for j,b in enumerate(BIRDS):
        # how many days of models are there for this bird?
        birdpath = glob(BASEPATH + b + match_str)[0]
        days = sorted(glob(birdpath+'/day*'))
        models = [None for _ in range(len(days))]
        scores = [None for _ in range(len(days))]
        for i,d in enumerate(days):
            fls = glob(d+'/model*')[0]
            m = joblib.load(fls)
            models[i] = m['model']
            scores[i] = m['val_score']
        birdmodels[j] = models
        birdscores[j] = scores
    return birdmodels, birdscores


def get_entropies(birdmodels):
    birdentropies = []
    for models in birdmodels:
        birdentropies.append([average_entropy(m.transmat_) for m in models])
    return birdentropies

def get_distance_bw_means(birdmodels):
    per_bird_dists = []
    for j,models in enumerate(birdmodels):
        # in each model there are K = 100? multivariate gaussians
        # assume variances are equal, then only need to find euclidean distance
        # between means
        meanss = []
        dists = []
        for i,m in enumerate(models):
            meanss.append(m.means_)
            if i > 0:
                # now make a nearest neighbour search between current means and
                # previous means
                m1 = meanss[i-1]
                m2 = meanss[i]
                dist = np.zeros((len(m1), len(m2)))
                for k in range(len(m1)):
                    dist[k] = np.array([np.sqrt(np.sum((m1[k] - m)**2)) for m in m2])
                dists.append(dist)
        per_bird_dists.append(dists)
        data = {'bird': BIRDS[j], 'means': meanss, 'distances': per_bird_dists}
    return data

def 
if __name__=='__main__':
    birdmodels, birdscores = get_models()
    birdentropies = get_entropies(birdmodels)
    distance_data = get_distance_bw_means(birdmodels)
    data_dict = {'models': birdmodels, 'scores': birdscores, 'entropies': birdentropies,
                'distances': distance_data}
    joblib.dump(data_dict,'/home/songbird/data/hmm_dump/analysis/models_entropies_distance.pkl')
    
    # display
    # For entropy find minimum length
    minlen = [len(e) for e in birdentropies]
    minlen = min(minlen)
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize = (22,10))
    for k,e in enumerate(birdentropies):
        plt.plot(e, '-s', lw=2)
        plt.text(tutorday[k], 1, '*', color ='k', fontsize = 11)
        plt.text(tutorday[k], 0.8, BIRDS[k], color = 'k', fontsize = 11)
    plt.xlabel('days')
    plt.ylabel('Entropy (bits)')
    plt.legend(BIRDS)
    plt.show()
    
    ent = np.stack([e[:minlen] for e in birdentropies], axis=0)
    avg_entropy = np.mean(ent, axis = 0)
    std_entropy = np.std(ent, axis = 0)
    plt.figure(figsize=(20,10))
    plt.plot(avg_entropy, '-r', lw=2)
    plt.fill_between(np.arange(minlen), y1 = avg_entropy - std_entropy,
                    y2= avg_entropy + std_entropy, color='r', alpha = 0.2)
    plt.xlabel('days')
    plt.ylabel('Entropy (bits)')
    plt.show()
    
    # distances
    dist = distance_data['distances']
    dists = [None for _ in range(len(BIRDS))]
    for b in range(len(BIRDS)):
        # find max of min distance for each day
        dd = dist[b]
        dists[b] = np.array([np.max(np.min(d, axis=1),axis=0) for d in dd])
        
    plt.figure(figsize=(22,10))
    for k,d in enumerate(dists):
        plt.plot(d, '-s', lw=2)
        plt.text(tutorday[k], 2, '*', color ='k', fontsize = 11)
        plt.text(tutorday[k], 1.9, BIRDS[k], color = 'k', fontsize = 11)
    plt.xlabel('days')
    plt.ylabel('Max(Min Distance)')
    plt.legend(BIRDS)
    plt.show()
    
    minlen = [len(d) for d in dists]
    minlen = min(minlen)
    dists = np.stack([d[:minlen] for d in dists], axis=0)
    avg_dist = np.mean(dists, axis = 0)
    std_dist = np.std(dists, axis = 0)
    plt.figure(figsize=(20,10))
    plt.plot(avg_dist, '-r', lw=2)
    plt.fill_between(np.arange(minlen), y1 = avg_dist - std_dist,
                    y2= avg_dist + std_dist, color='r', alpha = 0.2)
    plt.xlabel('days')
    plt.ylabel('Euc. Norm distance')
    plt.show()
        
        
                
        

    
        
    
    

