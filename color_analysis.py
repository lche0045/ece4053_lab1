import colorsys
from PIL import Image
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import picamera
from time import sleep
import time 
import cv2
from statistics import mean

# create the variable holder to plot 
hue_avg = list()
saturation_avg = list()
lightness_avg = list()
grey_avg = list()
redness_avg = list()
greenness_avg = list()
blueness_avg = list()

# determine the maximum iteration 
max_iter = 19

def main():

    start_time = time.time()
    path = r'C:\Users\nicho\Desktop\school\Year 4 Sem 1\wet etching\mask_try\sample_1.jpg'
    frame = cv2.imread(path)
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
    # Threshold of blue in HSV space
    lower_brown = numpy.array([5, 20, 10])
    upper_brown = numpy.array([30, 255, 250])
     
    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # perform "closing" morphology on the mask 
    kernel = numpy.ones((10,10),numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # perform filling on the mask 
    mask = fillhole(mask)
    
    
    
    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-brown regions to extract the chips region
    result = cv2.bitwise_and(frame, frame, mask = mask)
    
    for i in range(max_iter):
        try:
            # camera = picamera.PiCamera()
            # camera.resolution = (800, 600)
            # camera.start_preview()
            # sleep(5)
            # camera.capture("snapshot.jpg", resize=(640, 480))
            # camera.stop_preview()
            # camera.close()
            
            # this is to open precaptured image 
            # image = Image.open(f"sample_{i}.jpg") # this is masked image
            
            # this is to open precaptured image 
            image = cv2.imread(f"sample_{i}.jpg")
            
            # mask the image to extract chips region and save
            image = cv2.bitwise_and(image, image, mask = mask)
            cv2.imwrite(f"sample_masked_{i}.jpg", image)
            
            # grey scale func will generate the mean value of pixels and an grey scaled image
            # grey_image, mean_of_grey_array = grey_scale(image)
            # grey_image.save(f"grey_sample_{i}.jpg", quality=95)
            # grey_image = mpimg.imread(f"grey_sample_{i}.jpg")
            
            # # collect the grey scaled value to be plot out 
            # grey_avg.append(mean_of_grey_array)
            # plt.imshow(grey_image)
            # plt.show()
            

            # convert original RGB format into HLS format 
            image = Image.open(f"sample_masked_{i}.jpg") # this is masked image
            hls_array = create_hls_array(image)
            
            # redness calculator 
            redness, greenness, blueness = redness_calculator_np(image)
            redness_avg.append(redness)
            greenness_avg.append(greenness)
            blueness_avg.append(blueness)
            print(f"average redness = {redness}")
            
            # hue : --> generate the mean hue from the pixels of whole picture 
            hls_array_hue = numpy.squeeze(numpy.resize(hls_array[:, :, 0],[1,-1]))
            hls_array_hue_avg = numpy.mean(hls_array_hue)
            hue_avg.append(hls_array_hue_avg)
            print(f"average hue = {hls_array_hue_avg}")
            
            # lightness : --> generate the mean lightness from the pixels of whole picture
            hls_array_lightness = numpy.squeeze(numpy.resize(hls_array[:, :, 1],[1,-1]))
            hls_array_lightness_avg = numpy.mean(hls_array_lightness)
            lightness_avg.append(hls_array_lightness_avg)
            print(f"average lightness = {hls_array_lightness_avg}")
            
            # saturation : --> generate the mean saturation from the pixels of whole picture
            hls_array_saturation = numpy.squeeze(numpy.resize(hls_array[:, :, 2],[1,-1]))
            hls_array_saturation_avg = numpy.mean(hls_array_saturation)
            saturation_avg.append(hls_array_saturation_avg)
            print(f"average saturation = {hls_array_saturation_avg}")
            # time.sleep(20)

            # # create image from HLS array
            # new_image = image_from_hls_array(hls_array)

            # # save the created image 
            # new_image.save(f"sample_{i}.jpg", quality=95)
            

        except IOError as e:

            print(e)
            
    end_time = time.time()
    print("%d iterations took %.2f seconds, which corresponds to %.2fs/iteration" % (i, end_time - start_time, (end_time - start_time)/i))
    
    redness_avg_numpy = numpy.gradient(numpy.array(redness_avg),2)
    print(redness_avg_numpy)
    x = numpy.arange(0,int(len(lightness_avg)),1)


    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.plot(x, hue_avg, marker='o')
    ax1.plot(x, saturation_avg, marker='o')
    ax1.plot(x, lightness_avg, marker='o')
    
    ax2.plot(x, redness_avg, marker='o')
    ax2.plot(x, greenness_avg, marker='o')
    ax2.plot(x, blueness_avg, marker='o')
    
    ax3.plot(x, redness_avg_numpy, marker = 'o' )

    ax1.set_title("HSV parameters of images (9 s interval)", fontsize=10)
    # ax1.set_xticks(x, x, rotation ='vertical')
    ax1.legend(['hue','saturation','lightness'])
    
    ax2.set_title("Avg RGB of images (9 s interval)", fontsize=10)
    # ax1.set_xticks(x, x, rotation ='vertical')
    ax2.legend(['R','G','B'])
    

def create_hls_array(image):

    """
    Creates a numpy array holding the hue, lightness
    and saturation values for the Pillow image.
    """

    pixels = image.load()

    hls_array = numpy.empty(shape=(image.height, image.width, 3), dtype=float)

    for row in range(0, image.height):

        for column in range(0, image.width):

            rgb = pixels[column, row]

            hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)

            hls_array[row, column, 0] = hls[0]
            hls_array[row, column, 1] = hls[1]
            hls_array[row, column, 2] = hls[2]

    return hls_array


def redness_calculator_np(image):
    
    # converting into numpy version
    pix = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    
    # check for redness (R)
    redness = pix[:,:,0] - (pix[:,:,1]+pix[:,:,2])/2
    redness[redness<0] = 0
    avg_redness = numpy.mean(redness)
    
    # check for greenness (G)
    greenness = pix[:,:,1] - (pix[:,:,0]+pix[:,:,2])/2
    greenness[greenness<0] = 0
    avg_greenness = numpy.mean(greenness)
    
    # check for blueness (B)
    blueness = pix[:,:,2] - (pix[:,:,0]+pix[:,:,1])/2
    blueness[blueness<0] = 0
    avg_blueness = numpy.mean(blueness)
    
    return avg_redness,avg_greenness,avg_blueness
    



def image_from_hls_array(hls_array):

    """
    Creates a Pillow image from the HSL array
    generated by the create_hls_array function.
    """

    new_image = Image.new("RGB", (hls_array.shape[1], hls_array.shape[0]))

    for row in range(0, new_image.height):

        for column in range(0, new_image.width):

            rgb = colorsys.hls_to_rgb(hls_array[row, column, 0],
                                      hls_array[row, column, 1],
                                      hls_array[row, column, 2])

            rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

            new_image.putpixel((column, row), rgb)

    return new_image

# def grey_scale(image):
#     pixels = image.load()
#     grey_array = numpy.empty(shape=(image.height, image.width))
#     new_image = Image.new("RGB", (image.width, image.height))
    
#     for row in range(0, image.height):

#         for column in range(0, image.width):
#             rgb = pixels[column, row]
#             greyed = (rgb[0] + rgb[1] + rgb[2])/3
#             grey_array[row, column] = greyed
#             grey_rgb = (int(greyed),int(greyed),int(greyed))
#             new_image.putpixel((column, row), grey_rgb)
            
#     mean_of_grey_array = numpy.mean(numpy.resize(grey_array,(1,-1)))
    
#     return new_image, mean_of_grey_array
    
def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = numpy.zeros((h + 2, w + 2), numpy.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out 
    


main()