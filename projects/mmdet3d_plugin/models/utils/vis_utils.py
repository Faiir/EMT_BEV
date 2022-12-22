
import numpy as np
import torch 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import torch.nn.functional as F 
INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])


def generate_instance_colours(instance_map):
    # Most distinct 22 colors (kelly colors from https://stackoverflow.com/questions/470690/how-to-automatically-generate
    # -n-distinct-colors)
    # plus some colours from AD40k

    return {instance_id: INSTANCE_COLOURS[global_instance_id % len(INSTANCE_COLOURS)] for
            instance_id, global_instance_id in instance_map.items()
            }


def plot_instance_map(instance_image, instance_map, max_instance_num=None, instance_colours=None, bg_image=None):
    if isinstance(instance_image, torch.Tensor):
        instance_image = instance_image.cpu().numpy()
    assert isinstance(instance_image, np.ndarray)
    if instance_colours is None:
        instance_colours = generate_instance_colours(instance_map)
    if len(instance_image.shape) > 2:
        instance_image = instance_image.reshape(
            (instance_image.shape[-2], instance_image.shape[-1]))

    if max_instance_num is not None:
        for key, value in instance_colours.items():
            if key > max_instance_num:
                instance_colours[key] = [255, 0, 0]
    
    # white background + colorful objects
    if bg_image is None:
        plot_image = 255 * \
            np.ones(
                (instance_image.shape[0], instance_image.shape[1], 3), dtype=np.uint8)
    else:
        plot_image = bg_image

    for key, value in instance_colours.items():
        plot_image[instance_image == key] = value

    return plot_image, instance_colours


def plot_all(pred_mask, gt_mask, pred_labels, gt_labels,save_name, save_path=r"/home/niklas/ETM_BEV/BEVerse/viz"):
    
    num_classes = 100
    _idx = num_classes + 1
    #batch_size = pred_mask.shape[0]
    num_frame = pred_mask.shape[1]
    
    #for b in range(batch_size):
    temporal_instances = pred_mask
    mask_cls = F.softmax(pred_labels[0], dim=-1)[:, :_idx]
    _, labels = mask_cls.max(-1)  # 100 100
    valid = (labels < num_classes)
    labels = labels [valid]
    temporal_instances = temporal_instances[valid]
    for i in labels:
        temporal_instances[i] = temporal_instances * labels
        
    temporal_instances = temporal_instances.detach().cpu().numpy()
    
    instance_ids = np.unique(labels.detach().cpu().numpy())[1:]
    
    instance_ids = instance_ids[instance_ids != 255]
    print(instance_ids)
    instance_map = dict(zip(instance_ids, instance_ids))
    instance_colours_dict = {}
    plt.figure(0, figsize=(20, 8))
    plt.title("Prediction Mask")
    for i in range(num_frame):
        color_instance_i, instance_colours = plot_instance_map(
            temporal_instances[i], instance_map)
        instance_colours_dict.update(instance_colours)
        plt.subplot(1, num_frame, i + 1)
        plt.imshow(color_instance_i)
        plt.axis('off')
        #plt.show()
    values = instance_ids
    colors = [instance_colours_dict[k] for k in instance_colours_dict]
    color_name = "Instance_Colors"

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Instance {l}".format(
        l=values[i])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches)

    plt.grid(True)
    plt.savefig(
        f'{save_path}/{save_name}_prediction_batch_{b}.png')
    
    plt.close()
    
    temporal_instances = gt_mask[0]
    instance_ids = np.unique(temporal_instances)[1:]
    instance_ids = instance_ids[instance_ids != 255]
    print(instance_ids)
    instance_map = dict(zip(instance_ids, instance_ids))
    plt.figure(0, figsize=(20, 8))
    plt.title("Prediction Mask")
    for i in range(num_frame):
        color_instance_i, instance_colours = plot_instance_map(
            temporal_instances[i], instance_map)
        instance_colours_dict.update(instance_colours)
        plt.subplot(1, num_frame, i + 1)
        plt.imshow(color_instance_i)
        plt.axis('off')
        #plt.show()
    values = instance_ids
    colors = [instance_colours_dict[k] for k in instance_colours_dict]
    color_name = "Instance_Colors"

    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="Instance {l}".format(
        l=values[i])) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches)

    plt.grid(True)
    plt.savefig(
        f'{save_path}/{save_name}_gt_batch_{b}.png')
    plt.close()

    print(f"saved img to: {save_path}/{save_name}")
