from cv2 import cv2 
import matplotlib.pylab as plt
import numpy as np

def mask_roi(img, verticies):
    '''
    A function used to mask out all the portions
    of a colour image that are irrlevant to lane detection

    Varialbes:
        img - the orginal image you would like to mask
        vertices - the verticies of the shape you would
                    like to mask out. Must be a numpy array

    Returns: 
        The masked image 
    '''
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_colour = (255,) * channel_count
    cv2.fillPoly(mask, verticies, match_mask_colour)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_roi_grayscale(img, verticies):
    '''
    A function used to mask out all the portions
    of a grayscale image that are irrlevant to lane detection

    Varialbes:
        img - the orginal image you would like to mask
        vertices - the verticies of the shape you would
                    like to mask out. Must be a numpy array

    Returns: 
        The masked image 
    '''
    mask = np.zeros_like(img)
    match_mask_colour = 255
    cv2.fillPoly(mask, verticies, match_mask_colour)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    """
    Draws lines onto the image given based one the array of lines

    Varibles
        img - the image you would like the lines displayed on
        lines - array of vectors used to display the lines

    Returns
        img with lines drawn on
    """
    copy_img = np.copy(img)
    blank_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_img, (x1,y1), (x2,y2), (0,255,0), thickness=15)

    line_img = cv2.addWeighted(copy_img, 0.8, blank_img, 1, 0.0)

    return line_img

def average_slop_intercept(img, lines):
    """
    Finds the average slope and intercept of the lines found,
    since the functions prior are used to find lanes lines, they
    should be similar on either side

    Varibles
        img - img that the lines are intended to be drawn on
        lines - array of lines determined by desired algorithum,
                planned to be used with hough

    Retruns
        an array of arrays with the average line coordinates 
    """
    left_fit = []
    right_fit = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            parameters = np.polyfit((x1,x2),(y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))

    # find the average of the values
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # create coordinates to be plotted
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)

    return np.array([[left_line],[right_line]])


def make_coordinates(img, line_parameters): 
    """
    Helper function to assist in determining the coordinates
    from the average line data
    """ 
    slope, intercept = line_parameters
    y1 = int(img.shape[0])
    y2 = int(y1*(2/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])


image = cv2.imread("road.jpg")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#print(image.shape)
height = rgb_image.shape[0]
width = rgb_image.shape[1]

# roi - region of interest 
roi = [(0,height), (width/2,height/3), (width, 900)]

#convert image to greyscale
grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

# apply guassian blur
blur_image = cv2.GaussianBlur(grey_image,(7,7),0)

# run through canny edge detection
canny_image = cv2.Canny(blur_image, 100, 250)

# call the masking function 
masked_image = mask_roi_grayscale(canny_image, np.array([roi], np.int32))

# lines determined by the hough lines algorithum
lines = cv2.HoughLinesP(masked_image, rho=2, theta=np.pi/180, 
threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=15)

# determin just one line from all the lines found
avg_lines = average_slop_intercept(rgb_image, lines)

# draw the lines on the image
line_img = draw_lines(rgb_image, avg_lines)


## PLOT CODE SECTION ##

# # show orginal image
# plt.imshow(rgb_image)
# plt.show()

# # show grey image
# plt.imshow(grey_image, cmap="gray")
# plt.show()

# # show blurred image
# plt.imshow(blur_image,cmap="gray")
# plt.show()

# # show canny image
# plt.imshow(canny_image,cmap="gray")
# plt.show()

# # show masked image
# plt.imshow(masked_image)
# plt.show()

# show lined image
plt.imshow(line_img)
plt.show()