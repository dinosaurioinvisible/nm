
# from glob import glob
import os
import tonic 
import tonic.transforms as transforms
import matplotlib.pyplot as plt

# data: x, y, t (microseconds), p
dirpath = os.path.abspath(os.path.join(os.getcwd(),'..','nmnist_data'))
# loadpath = os.path.join(dirpath,'NMNIST','Test')

dataset = tonic.datasets.NMNIST(save_to=dirpath, train=False)

dx = 1000
events, target = dataset[dx]

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)

frames = frame_transform(events)

def plot_frames(frames):
    fig, axes = plt.subplots(1, len(frames))
    for axis,frame in zip(axes,frames):
        axis.imshow(frame[1] - frame[0])
        axis.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()
plot_frames(frames)

ft = 10000
denoise_transform = transforms.Denoise(filter_time=ft)

events_denoised = denoise_transform(events)
frames_denoised = frame_transform(events_denoised)

plot_frames(frames_denoised)

# transform = transforms.compose([denoise_transform, frame_transform])
# dataset = tonic.datasets.NMNIST(save_to=dirpath, transform=transform)

rev_p = 1
window = 10000
random_time_reversal_transform = transforms.Compose([transforms.RandomTimeReversal(p=rev_p),
                                   transforms.ToFrame(sensor_size, time_window=window)])

frames_reversed = random_time_reversal_transform(events)

frame_transform_tw = transforms.ToFrame(sensor_size=sensor_size, time_window=window)
frames_base = frame_transform_tw(events)

tj_std = 100
time_jitter_transform = transforms.Compose([transforms.TimeJitter(std=tj_std, 
                                                                  clip_negative=True,
                                                                  sort_timestamps=False),
                                            transforms.ToFrame(sensor_size=sensor_size, time_window=window)])

frames_time_jittered = time_jitter_transform(events)

from mkdata import SaccadeScramble

scramble_transform = transforms.Compose([transforms.Denoise(filter_time=ft),
                                         SaccadeScramble(),
                                        transforms.ToFrame(sensor_size,
                                                time_window=int(window))])

frames_scrambled = scramble_transform(events)

ani = tonic.utils.plot_animation(frames_base)
ani = tonic.utils.plot_animation(frames_reversed)
ani = tonic.utils.plot_animation(frames_time_jittered)
ani = tonic.utils.plot_animation(frames_scrambled)











# 


