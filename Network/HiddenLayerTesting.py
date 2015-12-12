from pybrain.tools.shortcuts import buildNetwork

list = 10 * (10,) + (10,)
print list

network = buildNetwork(28 * 28, *list)

