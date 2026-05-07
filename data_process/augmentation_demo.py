from PIL import Image, ImageDraw
import torchvision.transforms as transforms

# 生成一张简单示意图片（模拟杯子）
def make_sample_cup():
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle((80, 100, 176, 220), outline='black', fill='lightgray')
    draw.ellipse((90, 90, 166, 110), outline='black', fill='lightgray')
    draw.text((110, 230), "Cup", fill='black')
    return img

img = make_sample_cup()
img.save("original.png")

crop = transforms.RandomCrop(200)(img)
crop.save("random_crop.png")

blur = transforms.GaussianBlur(5, sigma=1.0)(img)
blur.save("gaussian_blur.png")

bright = transforms.ColorJitter(brightness=0.5)(img)
bright.save("brightness_adjust.png")

print("增强对比图已生成：original.png, random_crop.png, gaussian_blur.png, brightness_adjust.png")