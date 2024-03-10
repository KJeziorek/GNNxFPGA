import os
import shutil

directory = 'dataset/Caltech101'
ann_directory = 'dataset/Caltech101_annotations'

dst_train = 'dataset/ncaltech101/train'
dst_test = 'dataset/ncaltech101/test'
dst_val = 'dataset/ncaltech101/val'

train = 0.8
test = 0.1

for subdir, dirs, files in os.walk(directory):
    for dire in dirs:
        print(dire)
        if dire == 'BACKGROUND_Google':
            continue
        os.makedirs(os.path.join(dst_train, dire), exist_ok=True)
        os.makedirs(os.path.join(dst_test, dire), exist_ok=True)
        os.makedirs(os.path.join(dst_val, dire), exist_ok=True)
		
        os.makedirs(os.path.join('dataset/ncaltech101/annotations', dire), exist_ok=True)
        size = len(os.listdir(os.path.join(directory, dire)))
        train_id = int(size*train)
        test_id = int(size*(train+test))
        for id, filename in enumerate(os.listdir(os.path.join(directory, dire))):
            src = os.path.join(directory, dire, filename)
            annot = os.path.join(ann_directory, dire, filename.replace("image", "annotation"))
            shutil.copyfile(annot, os.path.join('dataset/ncaltech101/annotations', dire, filename.replace("image", "annotation")))
            
            if id < train_id:
                shutil.copyfile(src, os.path.join(dst_train, dire, filename))
            elif id < test_id:
                shutil.copyfile(src, os.path.join(dst_test, dire, filename))
            else:
                shutil.copyfile(src, os.path.join(dst_val, dire, filename))
    
#         os.removedirs(os.path.join(directory, dire))

# os.removedirs(directory)
# os.removedirs(ann_directory)