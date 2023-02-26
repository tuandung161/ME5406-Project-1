import pickle

map_4_4 = [[0,1,1,1],
           [1,2,1,2],
           [1,1,1,2],
           [2,1,1,3]]

pickle.dump(map_4_4, open("map_4_4.p","wb"))

result = pickle.load(open("map_4_4.p", "rb"))

print(result)