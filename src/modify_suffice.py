import sys,os
from glob import glob
from shutil import copyfile

def modify4train(input, output):
    if not os.path.exists(output):
        os.makedirs(output)
    subdirectories = glob(os.path.join(input, '*'))
    for i in range(len(subdirectories)):
        save_path = os.path.join(output, 
                        os.path.basename(subdirectories[i]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imgs = glob(os.path.join(subdirectories[i], '*.JPEG'))
        for j in range(len(imgs)):
            new_name = os.path.basename(imgs[j])
            new_name = '{}{}'.format(new_name[:new_name.rfind('.')],
                            new_name[new_name.rfind('.'):].lower())
            print(os.path.join(save_path, new_name))
            copyfile(imgs[j], os.path.join(save_path, new_name))
    
def modify4val(input, output):
    save_path = output
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    imgs = glob(os.path.join(input, '*.JPEG'))
    for j in range(len(imgs)):
        new_name = os.path.basename(imgs[j])
        new_name = '{}{}'.format(new_name[:new_name.rfind('.')],
                        new_name[new_name.rfind('.'):].lower())
        print(os.path.join(save_path, new_name))
        copyfile(imgs[j], os.path.join(save_path, new_name))

def main(args):
    folder_name = os.path.basename(args[1])
    if folder_name == 'train':
        modify4train(args[1],args[2])
    elif folder_name == 'val':
        modify4val(args[1],args[2])

if __name__ == '__main__':
    main(sys.argv)