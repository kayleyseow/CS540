from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)

def get_covariance(dataset):
    data = np.transpose(dataset)
    return (1/2413)*np.dot(data, dataset)

def get_eig(S, m):
    values, vectors = eigh(S, subset_by_index=(len(S)-m,len(S)-1))
    
    vectordes = np.fliplr(vectors)
    valuesdes = np.flip(values)
    revvec = vectordes[::-1]
    
    diagvals = np.diag(valuesdes)
    
    return diagvals, revvec

def get_eig_prop(S, prop):
    a = eigh(S)[0][::-1]
    avg = np.trace(np.diag(a))
    a = eigh(S, subset_by_value=[prop*avg, np.inf])
    return np.diag(a[0][::-1]), np.flip(a[1])[::-1]

def project_image(image, U):
    res = 0
    
    for i in range(U.shape[1]):
        alpha = np.dot(np.transpose(U[:, i]), image)
        res += np.dot(alpha, U[:, i])    
        
    return res

def display_image(orig, proj):
    
    oImage = np.reshape(orig, (32, 32))
    pImage = np.reshape(proj, (32, 32))
    
    figure, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    
    org = ax1.imshow(np.transpose(oImage), aspect = 'equal')
    pro = ax2.imshow(np.transpose(pImage), aspect = 'equal')
    figure.colorbar(org, ax = ax1)
    figure.colorbar(pro, ax = ax2)
    
    plt.show()