import dill as pickle
import torch
from scene import Scene, GaussianModel, DeformModel, Revolute
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import sys
import math
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim, chamfer_distance_loss
from utils.general_utils import farthest_point_sampling


def visualize(xyz, factor):
    import open3d as o3d
    import numpy as np

    pcd = o3d.geometry.PointCloud()
    xyz = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())

    factor = factor.detach().cpu().numpy()
    colors = (factor - factor.min()) / (factor.max() - factor.min()) * 255
    colors.astype(np.uint8)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


def get_images(
    viewpoint_cams,
    gaussians,
    deformModel,
    revoluteParams,
    pipe,
    background,
):
    # images = []
    for id, cam in enumerate(viewpoint_cams):
        new_xyz, new_rotations = deformModel.step(
            gaussians,
            revoluteParams.axis,
            revoluteParams.pivot,
            revoluteParams.theta,
            is_render=True,
        )
        render_pkg_re = render(
            cam, gaussians, pipe, background, new_xyz, new_rotations, 0.0, False
        )
        image = render_pkg_re["render"]
        image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.uint8(image_np * 255), "RGB")
        img.save(f"./rendered_img/{id}.png", format="PNG")


if __name__ == "__main__":
    with open("./end_frame_params.pkl", "rb") as f:
        data = pickle.load(f)

    visualize(data["xyz"], data["factors"])
    exit()

    parser = ArgumentParser(description="Training script parameters")
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    lp = ModelParams(parser)
    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)
    deform = DeformModel()
    deform.train_setting(opt)
    revolute = Revolute()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree)
    viewpoint_stack_full = scene.getTrainCameras().copy()
    viewpoint_stack_start_frame = []
    viewpoint_stack_end_frame = []
    for viewpoint_cam in viewpoint_stack_full:
        if viewpoint_cam.fid == 0:
            viewpoint_stack_start_frame.append(viewpoint_cam)
        else:
            viewpoint_stack_end_frame.append(viewpoint_cam)

    with open("./load_data/factor_xyz.pkl", "rb") as f:
        data = pickle.load(f)

    factor = data["factors"].detach()
    gaussians = data["gaussians"]
    revolute = data["revolute"]

    # gt_axis = torch.tensor([0, 1.0, 0], dtype=torch.float32, device="cuda")
    # gt_pivot = torch.tensor([0.013, 0.138, 0.73], dtype=torch.float32, device="cuda")
    # gt_theta = math.pi / 3
    axis = revolute.axis
    pivot = revolute.pivot
    pivot = torch.tensor([0.7294, 0.1751, -0.1518], device=pivot.device)
    theta = -1 * revolute.theta
    # theta = torch.tensor([-1 * math.pi * 160 / 180], device=theta.device)

    print(axis, pivot, theta)
    print(factor)
    xyz = gaussians.get_xyz

    # factor = xyz[:, 2] > (gt_pivot[2] + 0.1)
    # factor = torch.zeros(xyz.shape[0])
    # factor[0 : factor.shape[0] // 2] = 1
    factor[factor >= 0.2] = 1
    factor[factor < 0.2] = 0

    # factor = factor.float()
    # num = factor.sum().item()
    # print(f"{factor}, {num}/{factor.shape[0]}")

    with open("./load_data/end_frame_gaussian.pkl", "rb") as f:
        end_frame_gaussians = pickle.load(f)

    gt_xyz = farthest_point_sampling(
        end_frame_gaussians["gaussians"].get_xyz.detach(), 10000
    )
    for id, cam in enumerate(viewpoint_stack_end_frame):
        new_xyz, new_rotations = deform.deform(
            gaussians,
            axis,
            pivot,
            theta,
            is_render=False,
            factor=factor,
        )
        render_pkg_re = render(
            cam,
            gaussians,
            pipe,
            background,
            new_xyz,
            new_rotations,
            d_scaling=0.0,
        )
        image = render_pkg_re["render"]

        # Loss
        gt_image = cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        cd_loss = chamfer_distance_loss(new_xyz, gt_xyz)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)  # + chamfer_distance_loss(new_xyz, gt_xyz)
        )
        loss = loss + cd_loss
        print(f"#{id} camera with cd: {cd_loss} and total loss: {loss}")

        image_np = image.detach().cpu().numpy().transpose((1, 2, 0))

        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.uint8(image_np * 255), "RGB")
        img.save(f"./rendered_img/visualization/{id}.png", format="PNG")
