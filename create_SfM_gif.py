from imageio import imread, get_writer, imwrite
import matplotlib.pyplot as plt
from tqdm import tqdm

fig, axes = plt.subplots(4, 3)
with get_writer('output.gif', mode='I', fps=20) as output_gif:
    for i in tqdm(range(160, 260)):
        for ax in axes.flatten():
            ax.axis('off')

        axes[0, 1].imshow(imread(f'OutputFinal/20/2/20_2_frame{i}.png')[:240, 160:480, :])

        axes[1, 0].imshow(imread(f'OutputFinal/20/2/20_2_frame{i}.png')[240:, 160:480, :])
        axes[1, 1].imshow(imread(f'OutputFinal/20/5/20_5_frame{i}.png')[240:, 160:480, :])
        axes[1, 2].imshow(imread(f'OutputFinal/20/10/20_10_frame{i}.png')[240:, 160:480, :])
        axes[2, 0].imshow(imread(f'OutputFinal/50/2/50_2_frame{i}.png')[240:, 160:480, :])
        axes[2, 1].imshow(imread(f'OutputFinal/50/5/50_5_frame{i}.png')[240:, 160:480, :])
        axes[2, 2].imshow(imread(f'OutputFinal/50/10/50_10_frame{i}.png')[240:, 160:480, :])
        axes[3, 0].imshow(imread(f'OutputFinal/100/2/100_2_frame{i}.png')[240:, 160:480, :])
        axes[3, 1].imshow(imread(f'OutputFinal/100/5/100_5_frame{i}.png')[240:, 160:480, :])
        axes[3, 2].imshow(imread(f'OutputFinal/100/10/100_10_frame{i}.png')[240:, 160:480, :])

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        plt.savefig(f'FinalOutput/frame{i-160}.png')
        output_gif.append_data(imread(f'FinalOutput/frame{i-160}.png'))

        plt.pause(1)
        for ax in axes.flatten():
            ax.cla()
