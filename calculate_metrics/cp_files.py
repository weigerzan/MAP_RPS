import os
import shutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import skimage
import numpy as np
from skimage.color import rgb2ycbcr
import torch_fidelity
import tqdm
import lpips
import torch
import json
import math
# os.sep.join([''])

def cal_metrics(source_path, output_path='exp/learned_results/temp'):
    metrics_file = os.sep.join([source_path, 'metrics.json'])
    print(metrics_file)
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    else:
        if os.path.exists(output_path):
            # 删除文件夹及其内容
            shutil.rmtree(output_path)
        os.makedirs(os.sep.join([output_path, 'orig']), exist_ok=True)
        os.makedirs(os.sep.join([output_path, 'generated']), exist_ok=True)
        n1 = n2 = 0
        for filename in os.listdir(source_path):
            # if '_0' in filename and 'orig' not in filename and 'y0' not in filename:
            if '_-1' in filename and 'orig' not in filename and 'y0' not in filename:
                shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'generated', filename]))
            elif 'orig' in filename:
                shutil.copyfile(os.sep.join([source_path, filename]), os.sep.join([output_path, 'orig', filename]))
            else:
                pass

        orig_path = os.sep.join([output_path, 'orig'])
        generated_path = os.sep.join([output_path, 'generated'])
        N = len(os.listdir(generated_path))
        assert (N == 1000 or N == 100 or N == 10)
        # Calculated SSIM
        SSIM_sum = 0
        PSNR_sum = 0
        LPIPS_sum = 0
        MSE_sum = 0
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        # print('calculating PSNR, SSIM & LPIPS')
        with torch.no_grad():
            for k in tqdm.tqdm(range(N)):
                source_path = os.sep.join([orig_path, 'orig_{}.png'.format(k)])
                source_image = skimage.io.imread(source_path)/255.0
                # source_image = rgb2ycbcr(source_image/255.0)[:, :, 0]

                # denoise_path = os.sep.join([generated_path, '{}_0.png'.format(k)])
                denoise_path = os.sep.join([generated_path, '{}_-1.png'.format(k)])
                generated_image = skimage.io.imread(denoise_path)/255.0
                # print(source_image)
                # generated_image = rgb2ycbcr(generated_image/255.0)[:, :, 0]
                # print(source_image.shape)
                # print(generated_image)
                # SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
                SSIM = ssim(source_image, generated_image, data_range=generated_image.max() - generated_image.min(), channel_axis=-1)
                if math.isnan(SSIM):
                    print(k)
                    N -= 1
                    continue
                SSIM_sum += SSIM
                PSNR = psnr(source_image, generated_image)
                PSNR_sum += PSNR
                MSE_sum += np.mean((source_image - generated_image)**2)**0.5
                source_image = source_image * 2 - 1
                generated_image = generated_image * 2 - 1
                LPIPS = loss_fn_vgg(torch.tensor(source_image).permute(2,0,1).to(torch.float32).cuda(), torch.tensor(generated_image).permute(2,0,1).to(torch.float32).cuda())
                LPIPS_sum += LPIPS[0,0,0,0]
                # print('Image {}: PSNR {:.2f}, SSIM {:.4f}, LPIPS {:.4f}, RMSE {:.5f}'.format(k, PSNR_sum/(k+1), SSIM_sum/(k+1), LPIPS_sum/(k+1), np.mean((source_image - generated_image)**2)**0.5))
                # print(SSIM_sum/(k+1))
                # print(PSNR_sum/(k+1))
            # print('Average SSIM: {}'.format(SSIM_sum/N))
            # print('Average LPIPS: {}'.format(LPIPS_sum/N))
            # print('calculating KID & FID')
            Results = torch_fidelity.calculate_metrics(input1=orig_path, input2=generated_path, fid=True)
            # print('PSNR: {:.2f}, SSIM: {:.4f}, LPIPS: {:.4f}, FID: {:.2f}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['frechet_inception_distance']))
            print('{:.2f} / {:.4f} / {:.4f} / {:.2f} / {:.5f}'.format(PSNR_sum/N, SSIM_sum/N, LPIPS_sum/N, Results['frechet_inception_distance'], MSE_sum/N))
        metrics = {'PSNR': PSNR_sum/N, 'SSIM': SSIM_sum/N, 'LPIPS': LPIPS_sum.item()/N, 'FID': Results['frechet_inception_distance'], 'RMSE': MSE_sum/N}
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        return {'PSNR': PSNR_sum/N, 'SSIM': SSIM_sum/N, 'LPIPS': LPIPS_sum.item()/N, 'FID': Results['frechet_inception_distance'], 'RMSE': MSE_sum/N}


if __name__ == '__main__':
    # source_path = '/home/zhangjiawei/scripts/inverse_learned_coeff_ablation/exp/image_samples/ablation/diff_train_steps/cs2/ddnm/5steps_200epochs'
    source_path = '/home/zhangjiawei/scripts/DP_pixel/exp/image_samples/rebuttal/ffhq/denoise_0.3/mmse/50_0.95'
    # source_path = '/home/zhangjiawei/scripts/DP_pixel/exp/image_samples/rebuttal/ffhq/inp_box_0.05/map_ps/0_1000_2.0'
    # source_path = '/home/zhangjiawei/scripts/inverse_learned_coeff/exp/image_samples/rebuttal_icml/ffhq/deblur_aniso_noisy/diffpir'
    output_path = 'exp/learned_results/temp_1'
    cal_metrics(source_path=source_path, output_path=output_path)