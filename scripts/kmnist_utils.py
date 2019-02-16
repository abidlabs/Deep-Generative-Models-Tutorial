import requests
import os
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total, unit: x  # If tqdm doesn't exist, replace it with a function that does nothing
    print('**** Could not import tqdm. Please install tqdm for download progressbars! (pip install tqdm) ****')

# Python2 compatibility
try:
    input = raw_input
except NameError:
    pass


def load_data():
    
    url_list = ['http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz',
                'http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz']
    
    download_folder = 'datasets/kmnist/'
    try:
        os.mkdir(download_folder)
    except FileExistsError:
        pass
    
    for url in url_list:
        path = download_folder + url.split('/')[-1]
        if not(os.path.isfile(path)):
            r = requests.get(url, stream=True)

            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                print('Downloading {} - {:.1f} MB'.format(path, (total_length / 1024000)))

                for chunk in tqdm(r.iter_content(chunk_size=1024), total=int(total_length / 1024) + 1, unit="KB"):
                    if chunk:
                        f.write(chunk) 
                        
    def load(f):
        return np.load(f)['arr_0']                        
    
    x_train = load(download_folder + 'kmnist-train-imgs.npz')
    x_test = load(download_folder + 'kmnist-test-imgs.npz')
    y_train = load(download_folder + 'kmnist-train-labels.npz')
    y_test = load(download_folder + 'kmnist-test-labels.npz')
    return (x_train, y_train), (x_test, y_test)