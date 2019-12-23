# reading 

    pymic/train_infer/train_infer.py
    TrainInferAgent.__create_dataset.NiftyDataset(root_dir=root_dir,
                                    csv_file  = train_csv,
                                    modal_num = modal_num,
                                    with_label= True,
                                    transform = transforms.Compose(self.transform_list))

NiftyDataset.__getitem__(self,idx):
1. 模态的数量 self.modal_num =1 
2. 读取.nii.gz格式的文件到字典中
    - 字典的格式为
    output['data_array'] = data_array
    output['origin']     = origin
    output['spacing']    = (spacing[2], spacing[1], spacing[0])
    output['direction']  = direction
3. for i in range (self.modal_num):

        得到 names_list，image_list
        len(names_list) = self.modal_num
        len(image_list) = self.modal_num

4. image =np.concatenate(image_list, axis = 0)

        sample = {'image': image, 'names' : names_list[0], 
                'origin':image_dict['origin'],
                'spacing': image_dict['spacing'],
                'direction':image_dict['direction']}

5. sample['label'] = label
- 学习： pandas.iloc方法基于下标索引读取


6. 对输入进行处理

- 处理的方法有： [ChannelWiseNormalize, RandomFlip,  RandomCrop, LabelToProbability]
- import torchvision.transforms as transforms
- transform = transforms.Compose(self.transform_list))
    - ChannelWiseNormalize
        - 用于归一化输入
    - RandomFlip
        - 随机翻转
    - RandomCrop
        - 随机裁剪，输出的大小为 [48, 48, 48]
    - LabelToProbability
        - 前景和背景分离，

                for i in range(self.class_num):
                temp_prob = label == i*np.ones_like(label)
                label_prob.append(temp_prob)
                label_prob = np.asarray(label_prob, np.float32)
        