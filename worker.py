from scipy.spatial.distance import cosine
from multiprocessing import Pool
import numpy as np

def calculate_anomaly_score(parameters):# for a data point
    #parameters = [[x,model] for x in logs]
    log = parameters[0]
    model = parameters[1]
    #print(log)
    
    """Compute a distance of a log entry to elements of SOM."""
    dist_smallest = np.inf
    #print(model_1.shape[1])
    for x in range(model.shape[0]):
        for y in range(model.shape[1]):
            dist = cosine(model[x][y],log)
            #print(dist)
            #dist = np.linalg.norm(model[x][y] - log)
            if (dist <= dist_smallest):
                dist_smallest = dist
    return dist_smallest

def get_anomaly_score(logs, parallelism, model): # for whole dataset 
    parameters = [[x,model] for x in logs]
    pool = Pool(parallelism)
    dist = pool.map(calculate_anomaly_score, parameters) 
    #print(parameters)
    #dist = calculate_anomaly_score(parameters)
    pool.close()
    pool.join()
    #dist = [] 
    #for log in logs:
    #    dist.append(calculate_anomaly_score(log,model))
    return dist

if __name__ ==  '__main__': 
    get_anomaly_score(logs, parallelism, model)