import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite, get_writer
from tqdm import tqdm

np.set_printoptions(linewidth=1000)

#import imageio
#
#n = 1
#with imageio.get_writer('movie.gif', mode='I') as writer:
#    for i in tqdm(np.linspace(0, 360, 27)[:-1]):
#        image = imageio.imread(f'output/frame{n}.png')
#        writer.append_data(image)
#        n += 1
#
#quit()


import shutil, os
try:
    shutil.rmtree('motion_Ishita_average')
except FileNotFoundError:
    pass
os.makedirs('motion_Ishita_average')


try:
    shutil.rmtree('Ishita_frames')
except FileNotFoundError:
    pass
os.makedirs('Ishita_frames')

# import imageio
# vid = imageio.get_reader('Ishita.mp4',  'ffmpeg')
# for num in range(0, 275):
#     print(num)
#     image = vid.get_data(num)
#     imwrite(f'Ishita_frames/frame{num}.png', image)
# quit()


writer = get_writer('final.gif', fps=20)
writer2 = get_writer('final_both.gif', fps=20)
fig, axes = plt.subplots(2)
for ax in axes.flatten():
    ax.axis('off')

starting_frame, ending_frame = 142, 270

for n in tqdm(range(starting_frame,ending_frame)):
    motion_right = imread(f'motion_frames_Ishita/frame{n}.png')[:,:,0]
    motion_left = imread(f'motion_frames_Ishita/frame{n}.png')[:,:,2]

    for x in range(n+1, n + 5):
        if x == n + 2:
            image = imread(f'Ishita_frames/frame{x}.png')
            axes[0].imshow(image)
        motion_right = np.add(motion_right, imread(f'motion_frames_Ishita/frame{x}.png')[:,:,0])
        motion_left = np.add(motion_right, imread(f'motion_frames_Ishita/frame{x}.png')[:,:,2])

    image_right = np.copy(image)

    for i, j in [(i, j) for i in range(image.shape[0]) for j in range(image.shape[1])]:
        if motion_right[i,j] < 100:
            image_right[i,j] = 0.25 * image_right[i,j]

    axes[1].imshow(image_right)
    plt.pause(1)

    plt.savefig(f'motion_Ishita_average/frame{n}.png')

    #writer.append_data(fig)
    writer2.append_data(image_right)
writer.close()
writer2.close()
