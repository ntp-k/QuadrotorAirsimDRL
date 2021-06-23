import airsim
from airsim.AirSimClient import *
import numpy as np
from PIL import Image

drone = MultirotorClient()
drone.confirmConnection()
drone.enableApiControl(True)
drone.armDisarm(True)

drone.takeoffAsync().join()
drone.moveToPositionAsync(0, 0, -40, 5).join()
drone.moveToPositionAsync(-10, 0, -40, 5).join()
drone.moveToPositionAsync(-10, 0, -45, 5).join()
image_request = airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)
responses = drone.simGetImages([image_request])

img1d = np.array(responses[0].image_data_float, dtype=np.float)
# print(img1d.size)
print(img1d.min() , ',', img1d.max())

img1d = np.where(img1d > 255, 255, img1d)
print(img1d.min() , ',', img1d.max())

img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

image = Image.fromarray(img2d)
im_final = np.array(image.resize((84, 84)).convert("L"))
im = im_final.reshape([84, 84, 1])

image.show()
'''
img1d = np.array(responses[0].image_data_float, dtype=np.float)
# img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
img1d = np.full(img1d.size, 255, dtype=np.float ) - (255/img1d)
# img1d = 255 / img1d
img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

image = Image.fromarray(img2d)
im_final = np.array(image.resize((84, 84)).convert("L"))
im = im_final.reshape([84, 84, 1])

image.show()
'''

# # get numpy array
# img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 

# # reshape array to 4 channel image array H X W X 4
# img_rgb = img1d.reshape(response.height, response.width, 3)

# # original image is fliped vertically
# img_rgb = np.flipud(img_rgb)



# # write to png 
# airsim.write_png('/Users/natthaphat/Work/CPE/Senior/QuadrotorAirsimDRL/images/test_onGroung_depthPer.png', img2d) 