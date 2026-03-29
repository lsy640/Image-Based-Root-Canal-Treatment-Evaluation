
from PIL import Image
import torchvision.transforms as transforms


def main():
    img_file = "./inputs/root_dataset/images/1.jpg"
    tooth_file = "./outputs/root_dataset_NestedUNet_woDS/0/1.jpg"
    root_file = "./outputs/root_dataset_NestedUNet_woDS/1/1.jpg"
    out_file = "./rgb/rgb.jpg"
    image_transforms = transforms.Compose([
        transforms.Resize([192, 192]),
        transforms.Grayscale(1),
    ])

    img = Image.open(img_file)
    # 将图像采样为192*192大小灰度图
    img = image_transforms(img)

    root = Image.open(root_file)
    root = image_transforms(root)

    tooth = Image.open(tooth_file)
    tooth = image_transforms(tooth)
    new = Image.merge('RGB', [img, tooth, root])

    new.save(out_file)


if __name__ == '__main__':
    main()
