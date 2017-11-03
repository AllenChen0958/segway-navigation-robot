import scipy.io as sio
import sys
mat = sio.loadmat(sys.argv[1])
mat = mat['colors']
# for i in range (len(mat)):
#    print("{}: {}".format(i, mat[i]))
# index (ref: https://github.com/hellochick/Indoor-segmentation)
# mat[index] = [r, g, b]
# #0 Wall 255 0 0
# index = 1
# mat[index] = [255, 0, 0]

# #1 Floor 0 253 12 
# index = 4
# mat[index] = [0, 253, 12]

# #2 Tree 141 141 39
# index = 5
# mat[index] = [141, 141, 39]

# #3 Furniture 18 53 219
# index = 8
# mat[index] = [18, 53, 219]

# #4 Others 5 stairs 191 242 11
# index = 7
# mat[index] = [191, 242, 11]
# index = 26
# mat[index] = [191, 242, 11]
# #6 Ceil 77 77 77
# index = 6
# mat[index] = [77, 77, 77]
# sio.savemat('color150_2.mat', {'colors':mat})
####################################
#0 Wall 255 0 0
# index = 1
# mat[index] = [255, 0, 0]

# #1 Floor 0 253 12 
# index = 4
# mat[index] = [0, 253, 12]

# #2 Tree 141 141 39
# index = 5
# mat[index] = [255, 0, 0]

# #3 Furniture 18 53 219
# index = 8
# mat[index] = [255, 0, 0]

# #4 Others 5 stairs 191 242 11
# index = 7
# mat[index] = [255, 0, 0]
# index = 26
# mat[index] = [255, 0, 0]
# #6 Ceil 77 77 77
# index = 6
# mat[index] = [255, 0, 0]

# #6 person 77 77 77
# index = 13
# mat[index] = [255, 0, 0]

#https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
#=================================================
# for i in range(len(mat)):
#     if i==4:
#         mat[i] = [0, 253, 12]
#     else:
#         mat[i] = [255, 0, 0]

for i in range(len(mat)):
    if i==4:
        mat[i] = [0, 255, 0]
    elif i==1:
        mat[i] = [255, 0, 0]
    elif i==6:
        #mat[i] = [255, 0, 255]
        mat[i] = [255, 0, 0]
    else:
        mat[i] = [0, 0, 255]
sio.savemat('color150_2.mat', {'colors':mat})
