import numpy as np
import neurolab as nl

target = [
    [1, 0, 0, 0, 1,
     1, 1, 0, 1, 1,
     1, 1, 0, 1, 1,
     1, 0, 1, 0, 1,
     1, 0, 1, 0, 1],

    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],

    [1, 1, 1, 1, 1,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     1, 0, 1, 0, 0,
     1, 1, 1, 0, 0]
]

chars = ['К', 'В', 'О']
target = np.asfarray(target)
target[target == 0] = -1

net = nl.net.newhop(target)
output = net.sim(target)

print("Test on train samples:")
for i in range(len(output)):
    print(chars[i], (output[i] == target[i]).all())

print("Test of defaced O:")
test = np.asfarray([0, 1, 1, 1, 0,
                    0, 0, 0, 0, 1,
                    0, 0, 0, 0, 1,
                    1, 0, 0, 0, 1,
                    0, 1, 1, 1, 0])
test[test == 0] = -1
output = net.sim([test])
print((output[0] == target[0]).all(), 'Sim. steps', len(net.layers[0].outs))

