import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

steps = 1000
t = np.linspace(0, 10, steps)
x = np.sin(t)
phi = 5 + np.sin(2*t)
theta = np.sin(4*t)

figr = plt.figure()
gr = figr.add_subplot(1,1,1)
# gr.set_xlim([-7.5, 7.5])
# gr.set_ylim([-7.5, 7.5])
gr.set_xlim([-3.5, 2.5])
gr.set_ylim([-2.5, 3.5])

gr.plot([-0.5, 0.5, 0, -0.5], [0.5, 0.5, 1, 0.5], linewidth=1)

lenOA = 2.25
lenAB = 2

xO = 0
yO = 1

xA = xO + lenOA * np.sin(phi)
yA = yO + lenOA * np.cos(phi)

xB = xA - lenAB * np.sin(theta)
yB = yA - lenAB * np.cos(theta)

pO = gr.plot(xO, yO)[0]
pA = gr.plot(xA[0], yA[0], marker='.')[0]
pB = gr.plot(xB[0], yB[0], marker='.')[0]
lineAB = gr.plot([xA[0], xB[0]], [yA[0], yB[0]], color='black')[0]
lineOA = gr.plot([xO, xA[0]], [yO, yA[0]], color='black')[0]

Ns = 3
r1 = 0.1
r2 = 0.4
numpnts = np.linspace(0, 1, 50*Ns+1)
Betas = numpnts * (Ns * 2 * np.pi - phi[0]+1.5)
Xs = (r1 + (r2 - r1) * numpnts * np.cos(Betas)) - 0.1
Ys = (r1 + (r2 - r1) * numpnts * np.sin(Betas)) - 0.1

SpPruzh = gr.plot(Xs + xO, Ys + yO)[0]

def run(i):
    pA.set_data(xA[i], yA[i])
    pB.set_data(xB[i], yB[i])
    lineAB.set_data([xA[i], xB[i]], [yA[i], yB[i]])
    lineOA.set_data([xO, xA[i]], [yO, yA[i]])
    

    Betas = numpnts * (Ns * 2 * np.pi - phi[i]+1.5)
    Xs = (r1 + (r2 - r1) * numpnts * np.cos(Betas)) - 0.1
    Ys = (r1 + (r2 - r1) * numpnts * np.sin(Betas)) - 0.1

    SpPruzh.set_data(Xs + xO, Ys + yO)
    
    return

anim = FuncAnimation(figr, run, frames=steps, interval=0.5)

plt.show()