import h5py
import os
import numpy as np
from multiprocessing import Pool
from torch.utils.data import Dataset
import torch


try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data/'

class CGNSdata(Dataset):
    # mean=[96.02471],std=[8.589679],min=85,max=95
    def __init__(self,path,train=True,solution="Temperature"):
        self.hdf_path = os.sep.join([path,"hdf"])
        if train:
            path = os.sep.join([path,"train"])
        else:
            path = os.sep.join([path,"test"])
        file_list = os.listdir(path)
        self.file_list = list(map(lambda x:os.sep.join([path,x]),file_list))
        self.solution = '/Base/Zone/FlowSolution.N:1/%s/ data' % solution

    def __getitem__(self, index):
        path = self.file_list[index]
        index_map = np.load(path)
        fname = path.split(os.sep)[-1]
        fname = fname.split('.')[0]
        angle = fname.split('-')[-1]
        angle = angle.replace('_','.')
        angle = angle.replace('n','-')
        angle = np.float32(angle)
        src_name = os.sep.join([self.hdf_path,fname+".cgns_hdf"])
        src_f = h5py.File(src_name,"r")
        output = src_f[self.solution][:][index_map]
        output = torch.from_numpy(output[None,:,:])
        output = (output - 85)/(95-85)
        return np.array([angle]),output
    def __len__(self):
        return len(self.file_list)

def structure(l:list):
    k_list = ["root"]
    while(True):
        head = l.pop(0)
        if head == "|":
            print("|",end=" ")
        elif head == "\n":
            print()
            if len(l)>0:
                l += "\n"
            else:
                break
        elif head =="None":
            print("None",end=" ")
        else:
            k_name = k_list.pop(0)
            print(k_name,end=" ")
            try:
                keys = head.keys()
                for k in keys:
                    l.append(head[k])
                    k_list.append(k)
            except AttributeError:
                l.append("None")
            l.append("|")

def list_directory(node):
    try:
        keys = node.keys()
        for k in keys:
            list_directory(node[k])
    except AttributeError:
        print(node.name)

def row(a_b_sq:list):
    # a,b = point index
    a = a_b_sq[0]
    b = a_b_sq[1]
    squares = a_b_sq[2]
    list_a = []
    list_b = []
    mask = np.full((squares.shape[0],),True)
    while(True):
        list_a.append(a)
        list_b.append(b)
        sq_unmasked = np.logical_and(np.any(squares == a,axis=-1),np.any(squares == b,axis=-1))
        sq = np.argwhere(np.logical_and(sq_unmasked,mask))
        if len(sq) == 0:
            break
        sq = sq[0,0]
        square = squares[sq]
        ai = np.argwhere(square==a)[0,0]
        bi = np.argwhere(square==b)[0,0]
        mask[sq] = False
        a = square[(ai+3)%4]
        b = square[(bi+1)%4]
    return list_a,list_b


def mesh_to_grid(coordinates,cells):
    yy = 228
    xx = 989
    min_x = np.min(coordinates[:,0])
    min_x_arg = np.argwhere(coordinates[:,0]==min_x).reshape(-1)
    # top_left_arg = min_x_arg[np.argmax(coordinates[min_x_arg][:,1])]  #top-left: 500, bottom-left: 501
    sort_index = np.argsort(coordinates[min_x_arg][:,1])
    start_index_list = min_x_arg[sort_index][::-1]
    squares = np.copy(cells)
    a_b_sq_input = []
    output_array = np.full((yy,xx),-1)
    for i in range(0,len(start_index_list),2):
        a_b_sq_input.append([start_index_list[i],start_index_list[i+1],squares])
    with Pool(16) as p:
        outs = p.map(row,a_b_sq_input)
    for o in range(len(outs)):
        output_array[2*o] = outs[o][0]
        output_array[2*o+1] = outs[o][1]
    print(output_array)
    return output_array
        
        



if __name__ == "__main__":
    # for (dirpath, dirnames, filenames) in os.walk("Case_20_hdf"):
    #     for fname in filenames:
    #         inp = os.sep.join([dirpath, fname])
    #         print(inp)
    #         f = h5py.File(inp,'r')
    #         x = f['/Base/Zone/GridCoordinates/CoordinateX/ data'][:]
    #         y = f['/Base/Zone/GridCoordinates/CoordinateY/ data'][:]
    #         coordinates = np.stack([x,y],axis=1).astype(np.float32)
    #         cells_unspecified = f['/Base/Zone/unspecified/ElementConnectivity/ data'][:]
    #         cells_unspecified = cells_unspecified.reshape(-1,5)[:,1:].astype(np.uint32)
    #         cells_unspecified -= 1
    #         grid=mesh_to_grid(coordinates,cells_unspecified)
    #         opt_name = inp.split(os.sep)
    #         opt_name[0] = opt_name[0][:-3] + "npy"
    #         opt_name[1] = opt_name[1][:-9]
    #         opt_name = os.sep.join(opt_name)
    #         print(opt_name)
    #         np.save(opt_name,grid)

    # l = [f['Base']['Zone'], "\n"]
    # structure(l)
    f = h5py.File('test.cgns','r')
    list_directory(f)
    # print(f['/Base/Zone/FlowSolution.N:1/Pressure/ data'][:100])
    temperature = f['/Base/Zone/FlowSolution.N:1/Temperature/ data'][:]
    grid = np.load("test.npy")
    temperature_grid = temperature[grid]
    # print(temperature_grid)
    import matplotlib.pyplot as plt 
    plt.imshow(temperature_grid)
    plt.show()
    # data_loader = CGNSdata('.\\Case_30_npy')
    # t_list = []
    # for i in range(len(data_loader)):
    #     t = data_loader[i][1]
    #     t_list.append(t)
    # t_array = np.stack(t_list)
    # print(t_array.shape)
    # print(np.max(t_array))
    # print(np.mean(t_array))

