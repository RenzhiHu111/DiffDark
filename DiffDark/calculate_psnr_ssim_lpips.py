import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 路径设置
gt_path = './data/val/gt/'
results_path = './results/images_lol/lol/data/val/input'

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))
assert len(imgsName) == len(gtsName)

# 计算 PSNR 和 SSIM
cumulative_psnr, cumulative_ssim = 0, 0
for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=False)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=False)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim

# 计算 LPIPS
import argparse
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default=gt_path)
parser.add_argument('-d1', '--dir1', type=str, default=results_path)
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if opt.use_gpu:
    loss_fn.cuda()

files = os.listdir(opt.dir0)
cumulative_lpips = 0

for file in files:
    if os.path.exists(os.path.join(opt.dir1, file)):
        img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))
        if opt.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = loss_fn.forward(img0, img1)
        print('%s: %.4f' % (file, dist01))
        cumulative_lpips += dist01

# 输出最终结果
avg_psnr = cumulative_psnr / len(imgsName)
avg_ssim = cumulative_ssim / len(imgsName)
avg_lpips = cumulative_lpips / len(imgsName)

print('Testing set, PSNR is %.4f and SSIM is %.4f and LPIPS is %.4f' % (avg_psnr, avg_ssim, avg_lpips))
print(results_path)