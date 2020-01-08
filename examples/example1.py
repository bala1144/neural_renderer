"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse
import torch
import numpy as np
import tqdm
import torch.nn as nn
import cv2
from skimage import io
from HumanML.util.SMPLHelper import SMPLPytorchHelper, simplevisulizer

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class IntriniscModel(nn.Module):
    # model is used to sanity check the loss over the depth map by optimising over the intrinisc parameters
    def __init__(self, device = 'cuda', mode='depth', gtInput=None):
        super(IntriniscModel, self).__init__()

        # load .obj
        # load the  vertices from the pre optimized model
        mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/HumanML/Overfitting_10_12_2019_18_12_00/PC_75_ofset0.5_wofset0.9_40000.obj'
        # mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/DepthFBC/new/frame000000_40000.obj'
        vertices, faces = readObj(mesh_file_path, device)
        vertices = vertices.view(-1, 6890, 3)
        faces = faces.view(1, -1, 3)
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        #load the reference depth_map
        if gtInput is None:
            file_path = '/home/bala/Documents/Link to GuidedResearch/Datasets/Yawar/depth/frame000000.pgm'
            img = io.imread(file_path)
            gtDepthMap = torch.from_numpy(img).type(torch.float32)
            self.register_buffer('gtDepthMap', gtDepthMap)
        else:
            gtDepthMap = gtInput.type(torch.float32)
            self.register_buffer('gtDepthMap', gtDepthMap)

        Intrinsic = torch.tensor([[573.353 * 1.2, 0, 319.85],
                          [0, 576.057, 240.632],
                          [0, 0, 1]], dtype=torch.float32, device=device)
        Intrinsic = Intrinsic.view(-1,3,3)

        K = nn.Parameter(Intrinsic)

        R = torch.eye(3, dtype=torch.float32, device=device)
        R = R.view(-1, 3, 3)

        # t = torch.tensor([-0.0313091, -0.141842, 2.13459], dtype=torch.float32, device=device)
        t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        t = t.view(-1, 1, 3)

        # setup renderer
        self.mode = mode
        renderer = nr.Renderer(camera_mode='projection', image_size=(640, 480), anti_aliasing=True, K=K, R=R, t=t, far=100)
        self.renderer = renderer

    def forward(self):

        image = self.renderer(self.vertices, self.faces, self.textures, mode=self.mode)
        if self.mode== 'depth':
            loss = depthMapLoss(image[0], self.gtDepthMap)
        else:
            loss = L2Loss(image[0], self.gtDepthMap)
        return loss

def trainIntriniscmodel(renderedSilhoutte=None):
    print('trainIntriniscmodel')
    model = IntriniscModel(gtInput=renderedSilhoutte, mode='depth')
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=10)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        mode = 'rgb'
        images = model.renderer(model.vertices, model.faces, model.textures, mode=mode)
        if mode == 'rgb':
            images = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
            cv2.imshow('rendered', images)
            cv2.waitKey(1)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.item() < 70:
            break

def readObj(file, device='cpu'):
    lines = open(file, 'r')
    vertices = []
    faces = []
    for line in lines:
        if 'v' in line:
            vertices.append(list(map(float, line.split()[1:])))
        if 'f' in line:
            faces.append( list(map(int, line.split()[1:])))
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    if faces:
        faces = torch.tensor( faces, dtype=torch.int32, device=device)
        faces = faces - 1

    return vertices, faces

def guyRender():
    print('Render SMPL')
    # other settings
    camera_distance = 2.732
    elevation = 0
    azimuth = 0
    texture_size = 2

    depth_width = 640
    depth_height = 480

    colour_width = 1296
    colour_height = 968

    batch_size = 1
    # load the SMPL model  pose_params = torch.zeros(batch_size, 72) * 0.2
    pose_params = torch.zeros(batch_size, 72) * 0.2
    shape_params = torch.zeros(batch_size, 10) * 0.03

    device = 'cuda'

    # load the  vertices from the pre optimized model
    mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/HumanML/Overfitting_10_12_2019_18_12_00/PC_75_ofset0.5_wofset0.9_40000.obj'
    # mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/DepthFBC/new/frame000000_40000.obj'
    vertices, faces = readObj(mesh_file_path, device)
    vertices = vertices.view(-1, 6890, 3)
    faces = faces.view(1, -1, 3)


    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    print('textures.shape', textures.shape)

    K = torch.tensor([[573.353,	0,	319.85],
                      [0,	576.057, 240.632],
                      [0, 0, 1]], dtype=torch.float32, device=device)


    # K = torch.tensor([[1161.04, 0, 648.21],
    #                  [0, 1161.72, 485.785],
    #                  [0, 0, 1]], dtype=torch.float32, device=device)

    K = K.view(-1, 3, 3)

    R = torch.eye(3, dtype=torch.float32, device=device)
    R = R.view(-1, 3, 3)

    # t = torch.tensor([-0.0313091, -0.141842, 2.13459], dtype=torch.float32, device=device)
    t = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    t = t.view(-1, 1, 3)
    mode = 'silhouettes'
    renderer = nr.Renderer(camera_mode='projection', image_size=(depth_width, depth_height), anti_aliasing=True, K=K, R=R, t=t, far=100)
    # renderer = nr.Renderer(camera_mode='projection', image_size=(colour_width, colour_height), anti_aliasing=True, K=K, R=R, t=t, far=100)

    # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    images = renderer(vertices, faces, textures, mode)
    if mode == 'rgb':
        images = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        # images = images.detach().cpu().numpy()[0] # [image_size, image_size, RGB]
    else:
        torch_images = images.cpu()[0] # [image_size, image_size, RGB]
        images = images.detach().cpu().numpy()[0]  # [image_size, image_size, RGB]

    # print('images shape', images.shape)
    cv2.imwrite('rendered.jpg', images * 255)
    cv2.imshow('image', images)
    cv2.waitKey(0)
    return torch_images

def depthMapLoss(renderedDepth, gtDepthMap):
    # get the rendered image and create a mask
    mask = torch.zeros(renderedDepth.shape, dtype=torch.float32, requires_grad=False, device='cuda')
    U, V = torch.where(renderedDepth > 0)
    mask[U[:], V[:]] = 1.0

    loss = torch.sum((mask * (renderedDepth - 0.001 * gtDepthMap)) ** 2)  # Multiplying with 0.001 to convert cm to m

    return loss

def L2Loss(renderInput,gt):

    loss = torch.sum((renderInput - gt)** 2)  # Multiplying with 0.001 to convert cm to m
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    batch_size = 1
    # load the SMPL model  pose_params = torch.zeros(batch_size, 72) * 0.2
    pose_params = torch.zeros(batch_size, 72) * 0.2
    shape_params = torch.zeros(batch_size, 10) * 0.03

    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]
    device = 'cuda'
    mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/HumanML/Overfitting_10_12_2019_18_12_00/PC_75_ofset0.5_wofset0.9_40000.obj'
    # mesh_file_path = '/home/bala/Documents/Link to GuidedResearch/Results/DepthFBC/new/frame000000_40000.obj'
    helper = SMPLPytorchHelper(device)
    # vertices, openPosejoints = helper.SMPLPytorch_generator(pose_params, shape_params, False)
    vertices, faces = readObj(mesh_file_path, device)
    vertices = vertices.view(-1, 6890, 3)
    # vertices = vertices.detach().cpu().numpy().squeeze()
    # simplevisulizer(vertices)
    # print('vertices.shape', vertices.shape)

    help_faces = helper.faces
    # faces = faces.astype('int32')
    # faces = torch.from_numpy(faces)
    faces = faces.view(1,-1,3)

    print('vertices.shape', vertices.shape)
    print('faces.shape', faces.shape)

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    print('textures.shape', textures.shape)

    # to gpu

    # create renderer
    mode = 'rgb'
    renderer = nr.Renderer(camera_mode='look_at', image_size=(640, 480))

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    # writer = imageio.get_writer(args.filename_output, mode='I')
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images = renderer(vertices, faces, textures, mode)  # [batch_size, RGB, image_size, image_size]
        if mode == 'rgb':
            images = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        else:
            images = images.detach().cpu().numpy()[0] # [image_size, image_size, RGB]


        cv2.imshow('images', images)
        cv2.waitKey(1)
    #     writer.append_data((255*image).astype(np.uint8))
    # writer.close()


def debugMaskFunction():
    renderedDepth =  torch.tensor([[0,1.0,0,0],
                                   [0,100000,0,1],
                                   [0,0,10,0],
                                   [1,0,0,0]])
    mask =  torch.zeros(renderedDepth.shape, dtype=torch.float32, requires_grad=False, device='cpu')
    U, V = torch.where(renderedDepth > 0)
    mask[U[:], V[:]] = 1.0
    print(mask)
    print(mask * renderedDepth)

if __name__ == '__main__':
    # main()
    # load the depth image from the yawar sequence and compare
    # file_path = '/home/bala/Documents/Link to GuidedResearch/Datasets/Yawar/depth/frame000000.pgm'
    # file_path = '/media/bala/OSDisk/Users/bala/Documents/myprojects/GuidedResearch/NR/examples/rendered.jpg'
    # img = io.imread(file_path)
    # gtDepthMap = torch.from_numpy(img).type(torch.float32)

    renderedDepth = guyRender()
    # print('renderedDepth', renderedDepth.shape)
    # print('renderedDepth', renderedDepth.shape)
    # gtDepthMap = torch.from_numpy(img).type(torch.float32)
    # depthMapLoss(renderedDepth, gtDepthMap)
    # trainIntriniscmodel()
    # debugMaskFunction()


