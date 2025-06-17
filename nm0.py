
# does the order of numbers matter? can it be made to matter?
# instantaneous representation of causality/time-progression requires this

import tonic

dataset = tonic.datasets.NMNIST(save_to='../nm_data', train=True)
events, target = dataset[0] 
# events : x, y, t, p

tonic.utils.plot_event_grid(events)

