import os

folder = "./image_dataset_complete/Le/Le_negative/"

cnt = 0
for file in os.listdir(folder):
    os.rename(os.path.join(folder,file), os.path.join(folder,"image_{:03d}.pgm".format(cnt)))
    print("image_{:02d}.pgm".format(cnt))
    cnt += 1






