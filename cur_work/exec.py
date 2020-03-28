from lib import *

work_dir = "../pict/"
s = str(datetime.now())
save_dir = '../res_' + s[8:10] + '_' + s[5:7] + '/'
print(save_dir)
try:
    # Create target Directory
    mkdir(save_dir)
except FileExistsError:
    print("Directory " , save_dir ,  " already exists")

filename = "low_pig_rotated.jpg"
filename = "q.jpg"

img = rgb2gray(plt.imread(work_dir + filename))
c = Cepstrum(img, batch_size=256, step=0.5)

q = c.make_big_image()
plt.imsave('big_img.png', q, cmap='gray')

if (DEBUG):
    plt.imsave('lines_img.png', c.lines_img, cmap='gray')

    d = blend_images('orig_img.png', 'big_img.png')
    d.save('temp_vis.png')
    
    d = blend_images('temp_vis.png', 'lines_img.png', alpha=0.2, colH = 120)
    d.save(save_dir + filename[:-4] +'_vis.png')

    d = blend_images('orig_img.png', 'lines_img.png', alpha=0.6, colH = 120)
    d.save(save_dir + filename[:-4] +'_lines.png')
    
    plt.imsave(save_dir + filename[:-4] + '_restored_img.png', c.restored_image_full, cmap='gray')
    gc.collect()