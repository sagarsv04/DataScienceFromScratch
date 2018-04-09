import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os, sys
import zipfile


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DataSciencePractice\DeepDream')

url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
data_dir = './data/'
supported_video = ['mp4', 'avi']
# start with a gray image with a little noise
img_noise = np.random.uniform(size=(224,224,3)) + 100.0


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


def download_inception():
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

def showarray(a):
    a = np.uint8(np.clip(a, 0, 1)*255)
    plt.imshow(a)
    plt.show()


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]


def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)


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
    output_frame = np.uint8(np.clip(output_frame, 0, 1)*255)
    return output_frame


def run_deepdream(input_filename):

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
                writer = cv2.VideoWriter('./output.avi', cv2.cv.FOURCC(*'XVID'), 30, frame_size)

            writer.write(output_frame)
            i += 1
            print('Frame %i complete.' % i)

        cap.release()

    else:
        print("Input file doesn't fit the required input format.")

    return 0


def main():

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        print("Input file given:", input_filename)

        download_inception()
        # run deepdream on video or image file
        run_deepdream(input_filename)
    else:
        print("Please provide an input file !")

    return 0




if __name__ == '__main__':
    main()
