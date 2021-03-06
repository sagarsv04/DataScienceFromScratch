import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import cv2
import os, sys
import zipfile


# os.getcwd()
# os.chdir(r'D:\CodeRepo\DataSciencePractice\DeepDream')

url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
data_dir = './data/'
supported_video = ['mp4', 'avi']
# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0
model_fn = 'tensorflow_inception_graph.pb'
imagenet_mean = 117.0
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return 0


def check_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False


def check_if_image(file_path):
    # file_path = "pilatus800.jpg"
    if check_file_exist(file_path):
        try:
            PIL.Image.open(file_path)
        except IOError:
            return False
        return True
    else:
        print("Input file doesn't exists")
        return False


def check_if_video(file_path):
    # file_path = "pilatus800.mp4"
    if check_file_exist(file_path):
        file_ext = file_path.split('.')[-1]
        if file_ext in supported_video:
            return True
        else:
            print("Unsupported input file.")
            return False
    else:
        print("Input file doesn't exists")
        return False


def download_inception(data_dir):
    make_dir(data_dir)
    model_name = os.path.split(url)[-1]

    local_zip_file = os.path.join(data_dir, model_name)

    if not check_file_exist(local_zip_file):
        # Download
        print("Downloading and Extracting Inception !")
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    return 0


def run_deepdream(input_filename):
    # input_filename = 'input.jpg'
    # Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input')
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    # Helper functions for TF Graph visualization

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def

    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def

    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()

    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
            # normalizing the gradient, so the same step size should work
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
        showarray(visstd(img))

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]

    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    # End of Helper functions

    def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.8, octave_n=6, octave_scale=1.2):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
        # split the image into a number of octaves
        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))

        # return frame
        output_frame = img / 255.0
        return output_frame


    if check_if_image(input_filename):
        # open image
        img0 = PIL.Image.open(input_filename)
        img0 = np.float32(img0)
        output_frame = render_deepdream(tf.square(T('mixed4c')), img0)
        # output deep dream image via matplotlib
        showarray(output_frame)

    elif check_if_video(input_filename):
        # open video file
        cap = cv2.VideoCapture(input_filename)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frame length {0}.'.format(length))
        writer = None
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            # Apply gradient ascent to that layer
            output_frame = render_deepdream(tf.square(T('mixed3a')), frame)
            if writer is None:
                frame_size = (output_frame.shape[1], output_frame.shape[0])
                writer = cv2.VideoWriter('./output.avi', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, frame_size)

            writer.write(np.uint8(np.clip(output_frame, 0, 1)*255))
            i += 1
            print('Frame {0} of {1} complete.'.format(i, length))

        cap.release()

    else:
        print("Input file doesn't fit the required input format.")

    return 0


def main():

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        print("Input file given:", input_filename)

        download_inception(data_dir)
        # run deepdream on video or image file
        run_deepdream(input_filename)
    else:
        print("Please provide an input file !")

    return 0



if __name__ == '__main__':
    main()
