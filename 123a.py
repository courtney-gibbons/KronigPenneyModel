import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

V0 = 10
a = 2
b = 0.2
m = 1
hbar = 1

#3
def determinant(E, k):
    k1 = np.sqrt(2*m*E) / hbar
    k2prime = np.sqrt(2*m*(V0-E)) / hbar
    matrix = np.array([
        [1, 1, -1, -1],
        [1j*k1, -1j*k1, -k2prime, k2prime],
        [np.exp(1j*(k1-k)*a), np.exp(-1j*(k1+k)*a), -np.exp(-(k2prime-1j*k)*b), -np.exp((k2prime+1j*k)*b)],
        [1j*(k1-k)*np.exp(1j*(k1-k)*a), -1j*(k1+k)*np.exp(-1j*(k1+k)*a), -(k2prime-1j*k)*np.exp(-(k2prime-1j*k)*b), (k2prime+1j*k)*np.exp((k2prime+1j*k)*b)]
    ])
    return np.linalg.det(matrix).real

#4
k = 0.41
Es = np.arange(0.1, V0, 0.01)
dets = np.zeros(len(Es))
for i in range(0, len(Es)):
    dets[i] = determinant(Es[i], k)
plt.figure()
plt.plot(Es, dets)
plt.xlabel('Energy')
plt.ylabel('Determinants')
plt.title('Determinants vs Energy for k = 0.41')
plt.show()

#5
def findAllowedEs(k):
    Es = np.arange(0.1, V0, 0.01)
    dets = np.zeros(len(Es))
    for i in range(0, len(Es)):
        dets[i] = determinant(Es[i], k)
    allowedEs = []
    for i in range(0, len(dets)-1):
        if (dets[i] * dets[i+1] < 0):
            allowedEs.append(Es[i])
    return allowedEs

#6
ks = np.arange(-np.pi/a, np.pi/a, 0.1)
allowedEs = {}
for i in range(len(ks)):
    allowedEs[ks[i]] = findAllowedEs(ks[i])
for k in ks:
    print('Allowed energy for k = ' + str(k) + ': ' + str(allowedEs[k]))

#7
firstBand = np.zeros(len(ks))
secondBand = np.zeros(len(ks))
thirdBand = np.zeros(len(ks))
for i in range(len(ks)):
    kAllowedEs = allowedEs[ks[i]]
    firstBand[i] = kAllowedEs[0]
    secondBand[i] = kAllowedEs[1]
    thirdBand[i] = kAllowedEs[2]
plt.figure()
plt.plot(ks, firstBand)
plt.plot(ks, secondBand)
plt.plot(ks, thirdBand)
plt.xlabel('Momentum (k)')
plt.ylabel('Energy (E)')
plt.title('Band Structure')
plt.show()

#8a
Eg1 = np.mean(secondBand-firstBand)
Eg2 = np.mean(thirdBand-secondBand)
print('Approximate band gap between bands 1 and 2: ' + str(Eg1))
print('Approximate band gap between bands 2 and 3: ' + str(Eg2))

#8b
index = np.argmin(abs(secondBand-firstBand-3))
print('k = ' + str(ks[index]))

#9
firstBandDOS = np.zeros(len(ks))
for i in range(0, len(ks)-1):
    firstBandDOS[i] = 2 * (ks[i+1] - ks[i]) / np.pi / (firstBand[i+1] - firstBand[i])
secondBandDOS = np.zeros(len(ks))
for i in range(0, len(ks)-1):
    secondBandDOS[i] = 2 * (ks[i+1] - ks[i]) / np.pi / (secondBand[i+1] - secondBand[i])
thirdBandDOS = np.zeros(len(ks))
for i in range(0, len(ks)-1):
    thirdBandDOS[i] = 2 * (ks[i+1] - ks[i]) / np.pi / (thirdBand[i+1] - thirdBand[i])
plt.figure()
plt.fill_between(firstBand, np.zeros(len(firstBandDOS)), firstBandDOS)
plt.fill_between(secondBand, np.zeros(len(secondBandDOS)), secondBandDOS)
plt.fill_between(thirdBand, np.zeros(len(thirdBandDOS)), thirdBandDOS)
plt.xlabel('Energy (E)')
plt.ylabel('DOS D(E)')
plt.title('Density of States')
plt.ylim([0, None])
plt.show()