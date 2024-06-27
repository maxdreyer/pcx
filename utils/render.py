import numpy as np
import scipy.ndimage
import torch
from PIL import ImageFilter, Image, ImageDraw
from crp.image import get_crop_range, imgify
from torchvision.transforms.functional import gaussian_blur
from zennit.core import stabilize


@torch.no_grad()
def vis_opaque_img_border(data_batch, heatmaps, rf=False, alpha=0.4, vis_th=0.05, crop_th=0.05,
                          kernel_size=39) -> Image.Image:
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th.
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.
    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0]
        filtered_heat = filtered_heat / (filtered_heat.abs().max())
        vis_mask = filtered_heat > vis_th
        # imgs.append(imgify(img.detach().cpu()).convert('RGB'))
        # continue
        if True:
            row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)

            dr = row2 - row1
            dc = col2 - col1
            if dr > dc:
                col1 -= (dr - dc) // 2
                col2 += (dr - dc) // 2
                if col1 < 0:
                    col2 -= col1
                    col1 = 0
            elif dc > dr:
                row1 -= (dc - dr) // 2
                row2 += (dc - dr) // 2
                if row1 < 0:
                    row2 -= row1
                    row1 = 0

            img_t = img[..., row1:row2, col1:col2]
            vis_mask_t = vis_mask[row1:row2, col1:col2]

            if img_t.sum() != 0 and vis_mask_t.sum() != 0:
                # check whether img_t or vis_mask_t is not empty
                img = img_t
                vis_mask = vis_mask_t

        # vis_mask = scipy.ndimage.gaussian_filter(vis_mask.float().cpu().numpy() * 1.0, 4)
        # vis_mask = torch.from_numpy(vis_mask / vis_mask.max()).to(img.device)
        # inv_mask = 1 - vis_mask
        inv_mask = ~vis_mask
        outside = (img * vis_mask).sum((1, 2)).mean(0) / stabilize(vis_mask.sum()) > 0.5

        img = img * vis_mask + img * inv_mask * alpha + outside * 0 * inv_mask * (1 - alpha)

        img = imgify(img.detach().cpu()).convert('RGBA')

        img_ = np.array(img).copy()
        img_[..., 3] = (vis_mask * 255).detach().cpu().numpy().astype(np.uint8)
        img_ = mystroke(Image.fromarray(img_), 0, color='black' if outside else 'black')

        img.paste(img_, (0, 0), img_)

        imgs.append(img.convert('RGB'))

    return imgs


def mystroke(img, size: int, color: str = 'black'):

    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke)
    fill = (0, 0, 0, 180) if color == 'black' else (255, 255, 255, 180)
    for x in range(X):
        for y in range(Y):
            if edge[x, y][3] > 0:
                draw.ellipse((x - size, y - size, x + size, y + size), fill=fill)
    stroke.paste(img, (0, 0), img)

    return stroke