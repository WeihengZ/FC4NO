import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import os
import meshio
import pyvista as pv

def compute_nrmse_range(gt, pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE)
    normalized by the range of the ground truth.

    Parameters:
        gt (np.ndarray): Ground truth values
        pred (np.ndarray): Predicted values

    Returns:
        float: NRMSE normalized by the range of gt
    """
    gt = gt.flatten()
    pred = pred.flatten()

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    range_gt = np.max(gt) - np.min(gt)

    if range_gt < 1e-8:  # avoid division by zero
        return np.nan

    nrmse_range = rmse / range_gt
    return nrmse_range

def plot_model(args, model, device, test_loader, scale_factor):
    model.eval()
    best_pair = []
    worst_pair = []
    best_acc = np.inf
    worst_acc = - np.inf
    num_samples = 0
    with torch.no_grad():
        for data in test_loader:
            # extract data
            graph_data = data.to(device)
            if hasattr(model, 'predict'):
                output = model.predict(graph_data)
            else:
                output = model(graph_data)

            coor = graph_data.x.detach().cpu().numpy()
            GT = graph_data.y.detach().cpu().numpy()
            pred = output.detach().cpu().numpy()

            max_ = scale_factor[1]
            min_ = scale_factor[0]
            GT = GT * (max_ - min_) + min_
            pred = pred * (max_ - min_) + min_

            if np.linalg.norm(GT, ord=2) > 0:
                stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2) + 1e-8)
                if stress_relative_err < best_acc:
                    best_acc = stress_relative_err
                    best_pair = [coor, GT, pred]
                if stress_relative_err  > worst_acc:
                    worst_acc = stress_relative_err
                    worst_pair = [coor, GT, pred]
    
    
    # make the plots
    def make_a_plot(cases, c, name):
        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111, projection='3d')
        if name == 'gt':
            sc = ax.scatter(cases[0][:, 0], cases[0][:, 2] + 0.25, cases[0][:, 1], 
                            c=cases[1][:, 0], marker='o', vmin=min_, vmax=max_)
        if name == 'pred':
            sc = ax.scatter(cases[0][:, 0], cases[0][:, 2] + 0.25, cases[0][:, 1], 
                            c=cases[2][:, 0], marker='o', vmin=min_, vmax=max_)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_color((0,0,0,0))  
        ax.yaxis.line.set_color((0,0,0,0))  
        ax.zaxis.line.set_color((0,0,0,0))  
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlim([-0.2,1])
        ax.set_ylim([-0,0.5])
        ax.set_zlim([0,0.5])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        # fig.subplots_adjust(left=-0.2, right=1.0, top=1, bottom=0)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar.set_label('Normalized stress')  # Optional: Add label
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(r'res/samples/{}/{}_{}_{}.png'.format(args.data, name, c, args.model))

    make_a_plot(best_pair, 'best', 'gt')
    make_a_plot(best_pair, 'best', 'pred')
    make_a_plot(worst_pair, 'worst', 'gt')
    make_a_plot(worst_pair, 'worst', 'pred')

def plot_model_jet(args, model, device, test_loader):
    model.eval()
    best_pair = []
    worst_pair = []
    best_acc = np.inf
    worst_acc = - np.inf
    num_samples = 0
    with torch.no_grad():
        for data in test_loader:
            # extract data
            graph_data = data.to(device)
            output = model(graph_data)

            coor = graph_data.x.detach().cpu().numpy()
            GT = graph_data.y.detach().cpu().numpy()

            pred = output.detach().cpu().numpy()
            stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))

            # map the features back to original space and compute the relative error
            max_ = 2104.0876
            min_ = -1551.0989
            pred = pred * (max_ - min_) + min_
            GT = GT * (max_ - min_) + min_
            
            plt.figure()
            plt.subplot(121)
            plt.hist(GT, bins=100, alpha=0.2, label='GT')
            plt.subplot(122)
            plt.hist(pred, bins=100, alpha=0.2, label='Pred')
            plt.legend()
            plt.savefig(r'./test.png')
            assert 1==2

            stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))

            if np.linalg.norm(GT, ord=2) > 0:
                stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))
                plot_min = np.minimum(np.amin(GT), np.amin(pred))
                plot_max = np.maximum(np.amax(GT), np.amax(pred))
                
                BC_points = graph_data.control_points.detach().cpu().numpy()
                if stress_relative_err < best_acc:
                    best_acc = stress_relative_err
                    best_pair = [coor, GT, pred, plot_min, plot_max, BC_points]
                if stress_relative_err  > worst_acc:
                    worst_acc = stress_relative_err
                    worst_pair = [coor, GT, pred, plot_min, plot_max, BC_points]
    
    
    # make the plots
    def make_a_plot(cases, c, name):
        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(111, projection='3d')
        if name == 'gt':
            sc = ax.scatter(cases[0][:, 0], cases[0][:, 1], cases[0][:, 2], 
                            c=cases[1][:, 0], marker='o', s=0.1, vmin=cases[3], vmax=cases[4])
            edges = detect_edge_segments(cases[5])
            for edge in edges:
                ax.plot([edge[0], edge[3]],
                        [edge[1], edge[4]],
                        [edge[2], edge[5]],
                        c='red', alpha=0.6)
            # sc = ax.scatter(cases[6][:, 0], cases[6][:, 1], cases[6][:, 2], 
            #                 c='k', marker='o', s=1)
            # for edge in cases[5]:
            #     x = [edge[0], edge[3]]
            #     y = [edge[1], edge[4]]
            #     z = [edge[2], edge[5]]
            #     ax.plot(x, y, z, c='k')

        if name == 'pred':
            sc = ax.scatter(cases[0][:, 0], cases[0][:, 1], cases[0][:, 2], 
                            c=cases[2][:, 0], marker='o', vmin=cases[3], vmax=cases[4])
            for edge in cases[5]:
                x = [edge[0], edge[3]]
                y = [edge[1], edge[4]]
                z = [edge[2], edge[5]]
                ax.plot(x, y, z, c='k')
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        ax.xaxis.line.set_color((0,0,0,0))  
        ax.yaxis.line.set_color((0,0,0,0))  
        ax.zaxis.line.set_color((0,0,0,0))  
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_xlim([-0.2,1])
        # ax.set_ylim([-0,0.5])
        # ax.set_zlim([0,0.5])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.grid(False)
        # fig.subplots_adjust(left=-0.2, right=1.0, top=1, bottom=0)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar.set_label('Normalized stress')  # Optional: Add label
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(r'res/samples/{}/{}_{}_{}.png'.format(args.data, name, c, args.model))

    make_a_plot(best_pair, 'best', 'gt')
    # make_a_plot(best_pair, 'best', 'pred')
    # make_a_plot(worst_pair, 'worst', 'gt')
    # make_a_plot(worst_pair, 'worst', 'pred')
    print('accuracy of best case:', best_acc)
    print('accuracy of worst case:', worst_acc)

def plot_model_driver(args, model, device, test_loader, scale_factor):
    model.eval()
    best_pair = []
    worst_pair = []
    best_acc = np.inf
    worst_acc = - np.inf
    num_samples = 0
    with torch.no_grad():
        for data, graph_path in test_loader:
            # count
            num_samples += 1
            print('num_samples:', num_samples)
            if num_samples > 3:
                break

            # extract data
            graph_data = data.to(device)
            if hasattr(model, 'predict'):
                output = model.predict(graph_data)
            else:
                output = model(graph_data)

            coor = graph_data.x.detach().cpu().numpy()
            GT = graph_data.y.detach().cpu().numpy()
            pred = output.detach().cpu().numpy()

            # map the features back to original space and compute the relative error
            mean_ = scale_factor[0]
            std_ = scale_factor[1]
            GT = GT * (std_ ) + mean_
            pred = pred * (std_ ) + mean_

            if np.linalg.norm(GT, ord=2) > 0:
                # stress_relative_err = np.linalg.norm(GT - pred, ord=2) / (np.linalg.norm(GT, ord=2))

                # 95th percentile
                stress_relative_err = compute_nrmse_range(GT, pred)
                
                if stress_relative_err < best_acc:
                    best_acc = stress_relative_err
                    best_pair = [coor, GT, pred, graph_path]
                if stress_relative_err  > worst_acc:
                    worst_acc = stress_relative_err
                    worst_pair = [coor, GT, pred, graph_path]
                print(stress_relative_err)

    def extract_cells_from_faces(faces_array):
        """
        Converts pyvista.faces (a flat array of face sizes and indices)
        into a dict of face_type -> list of face connectivity arrays.
        """
        from collections import defaultdict

        i = 0
        cells = defaultdict(list)

        while i < len(faces_array):
            n = faces_array[i]  # number of points in this face
            pts = faces_array[i+1:i+1+n]

            if n == 3:
                cells["triangle"].append(pts)
            elif n == 4:
                cells["quad"].append(pts)
            elif n == 2:
                cells["line"].append(pts)
            else:
                print(f"Unsupported face with {n} vertices: skipping.")
            
            i += n + 1

        # convert to proper format
        return [(cell_type, np.array(faces)) for cell_type, faces in cells.items()]

    def save_vtu(pair, case):
        graph_path = pair[-1]
        sim_id = os.path.splitext(os.path.basename(graph_path[0]))[0].replace('graph_', '')
        group_name = "_".join(sim_id.split("_")[:-1])
        vtk_filename = '/taiga/illinois/eng/cee/meidani/Vincent/{}/PressureVTK/{}/{}.vtk'.format(args.data, group_name, sim_id)
        mesh = pv.read(vtk_filename)

        # Check shape consistency
        num_nodes = mesh.n_points
        assert len(pair[1]) == num_nodes, f"Ground truth length {len(pair[1])} doesn't match number of mesh points {num_nodes}"
        assert len(pair[2]) == num_nodes, f"Prediction length {len(pair[2])} doesn't match number of mesh points {num_nodes}"

        point_data = {
            "ML_pred": pair[2],
            "gt": pair[1],
            "err": np.abs(pair[1] - pair[2]),
        }

        # Convert quad faces for meshio
        mesh = mesh.triangulate()
        cells = extract_cells_from_faces(mesh.faces)

        # Create meshio mesh and write VTU file
        meshio_mesh = meshio.Mesh(points=mesh.points, cells=cells, point_data=point_data)
        out_path = f"./res/samples/{args.data}/{case}_{args.model}.vtu"
        meshio.write(out_path, meshio_mesh)
    
    save_vtu(best_pair, 'best')
    save_vtu(worst_pair, 'worst')
    print('accuracy of best case:', best_acc)
    print('accuracy of worst case:', worst_acc)