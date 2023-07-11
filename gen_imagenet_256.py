'''codes used to rescale original ImageNet dataset to 256 x 256 format, which is compatible with our toolkit for ImageNet
'''
import os


from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

transform_resize = transforms.Compose([
            transforms.Resize(size=[256, 256]),
            transforms.ToTensor(),
])

def create_256_scaled_version(src_directory, dst_directory, is_train_set=True):

    import time

    st = time.time()

    if is_train_set:
        classes = sorted(entry.name for entry in os.scandir(src_directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {src_directory}.")

        cnt = 0
        tot = len(classes)

        for cls_name in classes:

            print('start :', cls_name)

            cnt += 1

            dst_cls_dir_path = os.path.join(dst_directory, cls_name)
            if not os.path.exists(dst_cls_dir_path):
                os.mkdir(dst_cls_dir_path)
            src_cls_dir_path = os.path.join(src_directory, cls_name)
            img_entries = sorted(entry.name for entry in os.scandir(src_cls_dir_path))

            #with Pool(8) as p:
            #    p.map(sub_process, pars_set)


            for img_entry in img_entries:
                src_img_path = os.path.join(src_cls_dir_path, img_entry)
                dst_img_path = os.path.join(dst_cls_dir_path, img_entry)
                scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
                save_image(scaled_img, dst_img_path)

            print('[time: %f minutes] progress by classes [%d/%d], done : %s' % ( (time.time() - st)/60, cnt, tot, cls_name) )


    else:

        img_entries = sorted(entry.name for entry in os.scandir(src_directory))
        tot = len(img_entries)
        for i, img_entry in enumerate(img_entries):
            src_img_path = os.path.join(src_directory, img_entry)
            dst_img_path = os.path.join(dst_directory, img_entry)
            scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
            save_image(scaled_img, dst_img_path)
            print('[time: %f minutes] progress : [%d/%d]' % ((time.time() - st)/60, i+1, tot))





create_256_scaled_version('./data/imagenet/train', './data/imagenet_256/train', is_train_set=True)

create_256_scaled_version('./data/imagenet/val', './data/imagenet_256/val',
                          is_train_set=False)