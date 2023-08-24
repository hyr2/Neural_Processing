import numpy as np
import matplotlib.pyplot as plt
import os
from probeinterface import Probe
from probeinterface.plotting import plot_probe

# Geometry of 1x32 device (Chong/Luan Lab)
n = 32
positions = np.zeros((n, 2))
for i in range(n):
    x = 2
    y = i % n
    positions[i] = x, y
positions *= 25

# Generating Pribe
probe = Probe(ndim=2, si_units='um')
probe.set_contacts(positions=positions, shapes='square', shape_params={'width': 20})
print(probe)
# contour
polygon = [(65, -10), (50,-30) ,(35, -10), (-35, 820), (-35, 1120), (65, 1120)]
probe.set_planar_contour(polygon)
# Plotting
plot_probe(probe)
plt.axis('off')
plt.savefig(os.path.join('/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/subfigures/','1x32.svg'),format = 'svg',transparent=True)
# 3D
probe_3d = probe.to_3d(axes='xz')
probe_3d.rotate(theta=15, center=[0, 0, 0], axis=[1, 0, 0])
probe_3d.rotate(theta=-5, center=[0, 0, 0], axis=[0, 1, 0])
plot_probe(probe_3d)
ax = plt.gca()
ax.view_init(elev=9, azim=104)
ax.dist = 6
plt.axis('off')
plt.savefig(os.path.join('/home/hyr2-office/Documents/Paper/SIngle-Figures-SVG-LuanVersion/Fig2/subfigures/','1x32_3D.svg'),format = 'svg',transparent=True)
plt.show()