
majority = [
    [[0,0,0],0],
    [[0,0,1],0],
    [[0,1,0],0],
    [[1,0,0],0],
    [[0,1,1],1],
    [[1,0,1],1],
    [[1,1,0],1],
    [[1,1,1],1]
]

xor = [[[0,0,0],0],
       [[0,0,1],1],
       [[0,1,0],1],
       [[1,0,0],1],
       [[0,1,1],0],
       [[1,0,1],0],
       [[1,1,0],0],
       [[1,1,1],0]]

one_wire_not = [
       [[0,0,0],1],
       [[0,0,1],1],
       [[0,1,0],1],
       [[1,0,0],0],
       [[0,1,1],1],
       [[1,0,1],0],
       [[1,1,0],0],
       [[1,1,1],0]]

xor_hand_built = [
       [[0,0],0],
       [[0,1],1],
       [[1,0],1],
       [[1,1],0]]