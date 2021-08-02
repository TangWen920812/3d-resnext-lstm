# # _*_coding:utf-8_*_
# import numpy as np
# import imgaug as ia
# import imgaug.augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapOnImage
# from numba import jit
# import matplotlib.pyplot as plt
#
# ia.seed(1)
#
# @jit
# def aug_func(img, lbl, num_class=2):
#     # img: h w n
#     # lbl: h w
#
#     # Define our augmentation pipeline.
#     seq = iaa.Sequential([
#         # horizontal flips
#         iaa.Fliplr(0.5),
#         # iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
#         # iaa.Sharpen((0.0, 1.0)),  # sharpen the image
#         iaa.Sometimes(0.5,
#             iaa.Affine(
#             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#             rotate=(-30, 30),
#             shear=(-8, 8),
#             mode=["constant"],
#             cval=0),
#                       )
#         # iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
#         # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
#     ], random_order=True)
#     segmap = SegmentationMapOnImage(lbl, shape=img.shape, nb_classes=num_class)
#     img_aug, lbl_aug = seq(image=img, segmentation_maps=segmap)
#     lbl_aug = lbl_aug.get_arr_int()
#
#
#     return img_aug, lbl_aug
#
#
# # @jit
# def aug_process_seg(imgs, lbls):
#     # imgs: batch_size n h w
#     # lbls: batch_size h w
#     shape = imgs.shape
#     new_imgs = np.zeros(shape)
#     new_imgs[:,3:, :, :] = imgs[:,3:, :, :]
#     new_lbls = np.zeros((shape[0], shape[2], shape[3]), dtype=np.uint8)
#
#     for i in xrange(shape[0]):
#         img = imgs[i, :3, :, :]
#         img = np.transpose(img, (1,2,0))
#         lbl = lbls[i]
#         img_aug, lbl_aug = aug_func(img.astype(np.int16), lbl.astype(np.int16))
#         img_aug = np.transpose(img_aug, (2,0,1))
#         new_imgs[i, :3, :, :] = img_aug
#         new_lbls[i] = lbl_aug
#
#     return new_imgs, new_lbls
#
#
#
#
# _*_coding:utf-8_*_
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from numba import jit
import matplotlib.pyplot as plt

ia.seed(1)

# @jit
def aug_func(img):
    # img: h w n
    # lbl: h w

    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        # horizontal flips
        iaa.Fliplr(0.5),  # horizontal flips
        # iaa.Crop(percent=(0, 0.1)),  # random crops
        # # Small gaussian blur with random sigma between 0 and 0.5.
        # # But we only blur about 50% of all images.
        # iaa.Sometimes(0.5,
        #               iaa.GaussianBlur(sigma=(0, 0.5))
        #               ),
        # # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.75, 1.5)),
        # # Add gaussian noise.
        # # For 50% of all images, we sample the noise once per pixel.
        # # For the other 50% of all images, we sample the noise per pixel AND
        # # channel. This can change the color (not only brightness) of the
        # # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # # Make some images brighter and some darker.
        # # In 20% of all cases, we sample the multiplier once per channel,
        # # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # # Apply affine transformations to each image.
        # # Scale/zoom them, translate/move them, rotate them and shear them.

        iaa.Sometimes(0.5,
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-30, 30),
            shear=(-8, 8),
            mode=["constant"],
            cval=0),
                      )
        # iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
    ], random_order=True)

    img_aug = seq.augment_images(img)



    return img_aug


# @jit
def aug_process(imgs):
    # imgs: (N,cls,h,w)

    imgs = np.transpose(imgs, (0, 2, 3, 1))  # (N,h,w,cls)
    images_aug = aug_func(imgs)
    images_aug = np.transpose(images_aug, (0, 3, 1, 2))  # (N,h,w,cls)
    return images_aug




