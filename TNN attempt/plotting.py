'''
A place to keep useful TNN plotting functions.
'''

from tqdm.notebook import tqdm
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from network import Network
from PIL import Image as PImage 
from PIL import ImageDraw, ImageFont

def create_gif(array_3d, filename):
    # Define the color map to convert +1/-1 to white/black
    colormap = {+1: [255, 255, 255], -1: [0, 0, 0]}
    
    # Transpose to iterate through 2D frames along the 3rd dimension
    array_series = np.transpose(array_3d, (2, 0, 1))

    # List to store all the frames
    frames = []

    for i, array in enumerate(tqdm(array_series)):
        # Convert array elements to white or black based on the colormap
        image = np.array([colormap[i] for i in np.ravel(array)], dtype=np.uint8)

        # Use zoom to increase pixel size
        zoom_factor = 10  # Change this to adjust your pixel size
        image = zoom(image.reshape(*array.shape, -1), (zoom_factor, zoom_factor, 1))

        pil_image = PImage.fromarray(image)
        
        draw = ImageDraw.Draw(pil_image)
        
         # You may need a specific font file (.ttf) for this line. If not available use a default font.
         # Change 'Arial.ttf' with your own font file path.
         # Change '20' with desired font size.
        fnt = ImageFont.truetype('Arial.ttf',36) 
        
         # Draw text on top right corner of each frame. You may need to adjust position and color as per your requirement.
        
        draw.text((pil_image.size[0]-50 ,10), str(i), fill=(255 ,0 ,0),font=fnt) 
        
         ## Append modified image (with text) into frames list
        
        frames.append(np.array(pil_image))
    


    # Save to GIF
    imageio.mimsave(filename, frames, 'GIF', duration=0.2,loop=0)



def unchanged_pixels(array_series):
    # Use np.equal() and np.all() to find pixels that haven't changed.
    unchanged = np.all(np.equal(array_series, array_series[:,:,0,None]), axis=2)

    return unchanged

def plot_unchanged_pixels(unchanged):
    # Convert boolean values to int: True -> 1 (white), False -> 0 (black)
    img = unchanged.astype(int)
    plt.imshow(img)
    plt.show()