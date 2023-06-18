import os
import numpy as np
import pandas as pd
import cv2
from scipy import io
import datetime as date
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

'''
- Ref
https://github.com/yiminglin-ai/imdb-clean/tree/master
https://github.com/imdeepmind/processed-imdb-wiki-dataset
https://github.com/ivangrov/Datasets-ASAP/blob/master/%5BPart%201%5D%20IMDB-WIKI/Read_IMDB_WIKI_Dataset.ipynb
'''

class IMDBDataset(Dataset):

    def __init__(self, mode, data_dir, output_size,
                 random_scale, random_mirror, random_crop, max_age=100, test_size=0.1):

        self.mode = mode
        self.test_size = test_size
        self.max_age = max_age

        print('-'*50)
        print(f'{self.mode} IMDBDataset Init !')
        print('self.test_size:', self.test_size)
        print('self.max_age:', self.max_age)
        print('-'*50)

        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'imdb_crop')
        self.mat_path = os.path.join(self.data_dir, 'imdb_meta', 'imdb.mat')
        print('-'*30)
        print('self.data_dir:', self.data_dir)
        print('self.img_dir:', self.img_dir)
        print('self.mat_path:', self.mat_path)
        print('-'*30)

        self.output_size = output_size
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        print('-'*30)
        print('self.output_size:', self.output_size)
        print('self.random_scale:', self.random_scale)
        print('self.random_mirror:', self.random_mirror)
        print('self.random_crop:', self.random_crop)
        print('-'*30)

        # self.meta_csv_path = os.path.join(self.data_dir, 'imdb_meta', 'meta.csv')
        # print('self.meta_csv_path:', self.meta_csv_path)
        # if not os.path.exists(self.meta_csv_path):
        #     self.__build_parsing_mat()
        #     self.__build_gender_dataset()
        #     self.__build_age_dataset()
        # else:
        #     print(f' ==> {self.meta_csv_path} exists!')
        self.train_final_gender_imdb = {}
        self.test_final_gender_imdb = {}
        self.train_final_age_imdb = {}
        self.test_final_age_imdb = {}
        self.train_meta_csv_path = os.path.join(self.data_dir, 'csvs', 'imdb_train_new_1024.csv')
        self.test_meta_csv_path = os.path.join(self.data_dir, 'csvs', 'imdb_test_new_1024.csv')
        self.gender_data_build_flag = os.path.exists(os.path.join(self.data_dir, 'data', 'gender'))
        self.age_data_build_flag = os.path.exists(os.path.join(self.data_dir, 'data', 'age'))
        print('-'*30)
        print('self.train_meta_csv_path:', self.train_meta_csv_path)
        print('self.test_meta_csv_path:', self.test_meta_csv_path)
        print('self.gender_data_build_flag:', self.gender_data_build_flag)
        print('self.age_data_build_flag:', self.age_data_build_flag)
        print('-'*30)
        self.__build_gender_dataset()
        self.__build_age_dataset()
        print('-'*30)
        print('len(self.train_final_gender_imdb.keys()):', len(self.train_final_gender_imdb.keys()))
        print('len(self.test_final_gender_imdb.keys()):', len(self.test_final_gender_imdb.keys()))
        print('len(self.train_final_age_imdb.keys()):', len(self.train_final_age_imdb.keys()))
        print('len(self.test_final_age_imdb.keys()):', len(self.test_final_age_imdb.keys()))
        print('-'*30)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((122.67891434, 116.66876762, 104.00698793), (1., 1., 1.))

    def __build_age_dataset(self):

        # meta = pd.read_csv(self.meta_csv_path)
        train_meta = pd.read_csv(self.train_meta_csv_path)
        test_meta = pd.read_csv(self.test_meta_csv_path)

        # meta = meta.drop(['gender'], axis=1)
        D_train = train_meta.drop(['gender', 'x_min', 'y_min', 'x_max', 'y_max', 'head_roll', 'head_yaw', 'head_pitch'], axis=1)
        D_test = test_meta.drop(['gender', 'x_min', 'y_min', 'x_max', 'y_max', 'head_roll', 'head_yaw', 'head_pitch'], axis=1)

        # meta = meta[meta['age'] >= 0]
        # meta = meta[meta['age'] <= self.max_age]
        D_train = D_train[D_train['age'] >= 1]
        D_train = D_train[D_train['age'] <= self.max_age]
        D_test = D_test[D_test['age'] >= 1]
        D_test = D_test[D_test['age'] <= self.max_age]

        # D_train, D_test = train_test_split(meta, test_size=self.test_size, random_state=42)

        # Shuffling the dataset
        D_train = D_train.sample(frac=1, random_state=42)
        D_test = D_test.sample(frac=1, random_state=42)

        # Making the directory structure
        output_dir_train = os.path.join(self.data_dir, 'data', 'age', 'train')
        output_dir_test = os.path.join(self.data_dir, 'data', 'age', 'test')
        print('output_dir_train:', output_dir_train)
        print('output_dir_test:', output_dir_test) 

        for i in range(1, self.max_age + 1):

            if not os.path.exists(os.path.join(output_dir_train, str(i))):
                os.makedirs(os.path.join(output_dir_train, str(i)))

            if not os.path.exists(os.path.join(output_dir_test, str(i))):
                os.makedirs(os.path.join(output_dir_test, str(i)))

        # Finally making the training and testing set
        print('-'*30)
        for counter, image in enumerate(D_train.values):

            if counter % 10000 == 0:
                print(f'build age train dataset - idx: {counter} ...')

            if not self.age_data_build_flag:
                # img = cv2.imread(image[1], 1)
                img_path = os.path.join(self.data_dir, 'data', 'imdb-clean-1024', image[0])
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (self.output_size[0], self.output_size[1]))
            out_path = os.path.join(output_dir_train, str(image[1]), str(counter) + '.jpg')
            if not self.age_data_build_flag:
                cv2.imwrite(out_path, img)
            # print('--('+str(counter)+')Processing--')
            
            self.train_final_age_imdb[counter] = [out_path, image[1]]
        print('-'*30)

        print('-'*30)
        for counter, image in enumerate(D_test.values):

            if counter % 10000 == 0:
                print(f'build age test dataset - idx: {counter} ...')

            if not self.age_data_build_flag:
                # img = cv2.imread(image[1], 1)
                img_path = os.path.join(self.data_dir, 'data', 'imdb-clean-1024', image[0])
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (self.output_size[0], self.output_size[1]))
            out_path = os.path.join(output_dir_test, str(image[1]), str(counter) + '.jpg')
            if not self.age_data_build_flag:
                cv2.imwrite(out_path, img)
            # print('--('+str(counter)+')Processing--')

            self.test_final_age_imdb[counter] = [out_path, image[1]]
        print('-'*30)

        print('-'*50)
        print(' ====> Complete to build age dataset !')
        print('-'*50)
            
    def __build_gender_dataset(self):

        # meta = pd.read_csv(self.meta_csv_path)
        train_meta = pd.read_csv(self.train_meta_csv_path)
        test_meta = pd.read_csv(self.test_meta_csv_path)

        # meta = meta.drop(['age'], axis=1)
        D_train = train_meta.drop(['age', 'x_min', 'y_min', 'x_max', 'y_max', 'head_roll', 'head_yaw', 'head_pitch'], axis=1)
        D_test = test_meta.drop(['age', 'x_min', 'y_min', 'x_max', 'y_max', 'head_roll', 'head_yaw', 'head_pitch'], axis=1)
        
        # D_train, D_test = train_test_split(meta, test_size=self.test_size, random_state=42)

        D_train_male = D_train[D_train['gender'] == 'male']
        D_train_female = D_train[D_train['gender'] == 'female']

        no_male = len(D_train_male)
        no_female = len(D_train_female)
        print('-'*30)
        print('len(D_train_male):', len(D_train_male))
        print('len(D_train_female):', len(D_train_female))
        print('-'*30)

        D_test_male = D_test[D_test['gender'] == 'male']
        D_test_female = D_test[D_test['gender'] == 'female']

        no_male = len(D_test_male)
        no_female = len(D_test_female)
        print('-'*30)
        print('len(D_test_male):', len(D_test_male))
        print('len(D_test_female):', len(D_test_female))
        print('-'*30)

        # # The dataset contains more male faces that female faces. This can couse some problems.
        # # One feature can start dominating on other feature. To solve this I am selecting equal number of male and female faces in the training set
        # if no_male > no_female:
        #     extra = D_train_male[no_female:]
        #     D_train_male = D_train_male[0:no_female]

        #     D_test = pd.concat((D_test, extra))
        # else:
        #     extra = D_train_male[no_male:]
        #     D_train_male = D_train_male[0:no_male]

        #     D_test = pd.concat((D_test, extra))
        # D_train = pd.concat((D_train_male, D_train_female))
        print('-'*30)
        print('D_train.shape:', D_train.shape)
        print('D_test.shape:', D_test.shape)
        print('-'*30)
        
        # Shuffling the dataset
        D_train = D_train.sample(frac=1, random_state=42)
        D_test = D_test.sample(frac=1, random_state=42)

        # Generating folder struture for the data
        output_dir_train_male = os.path.join(self.data_dir, 'data', 'gender', 'train', 'male')
        output_dir_train_female = os.path.join(self.data_dir, 'data', 'gender', 'train', 'female')
        print('-'*30)
        print('output_dir_train_male:', output_dir_train_male)
        print('output_dir_train_female:', output_dir_train_female) 

        if not os.path.exists(output_dir_train_male):
            os.makedirs(output_dir_train_male)

        if not os.path.exists(output_dir_train_female):
            os.makedirs(output_dir_train_female)

        output_dir_test_male = os.path.join(self.data_dir, 'data', 'gender', 'test', 'male')
        output_dir_test_female = os.path.join(self.data_dir, 'data', 'gender', 'test', 'female')
        print('output_dir_test_male:', output_dir_test_male)
        print('output_dir_test_female:', output_dir_test_female) 
        print('-'*30)

        if not os.path.exists(output_dir_test_male):
            os.makedirs(output_dir_test_male)

        if not os.path.exists(output_dir_test_female):
            os.makedirs(output_dir_test_female)

        # Finally processing the image train and test set
        print('-'*30)
        for counter, image in enumerate(D_train.values):

            if counter % 10000 == 0:
                print(f'build gender train dataset - counter: {counter} ...')

            if not self.gender_data_build_flag:
                # img = cv2.imread(image[1], 1)
                img_path = os.path.join(self.data_dir, 'data', 'imdb-clean-1024', image[0])
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (self.output_size[0], self.output_size[1]))
            # if image[0] == 'male':
            if image[1] == 'M':
                out_path = os.path.join(output_dir_train_male, str(counter) + '.jpg')
                if not self.gender_data_build_flag:
                    cv2.imwrite(out_path, img)
            elif image[1] == 'F':
                out_path = os.path.join(output_dir_train_female, str(counter) + '.jpg')
                if not self.gender_data_build_flag:
                    cv2.imwrite(out_path, img)
            # print('--('+str(counter)+')Processing--')

            self.train_final_gender_imdb[counter] = [out_path, image[1]]
        print('-'*30)

        print('-'*30)
        for counter, image in enumerate(D_test.values):

            if counter % 10000 == 0:
                print(f'build gender test dataset - counter: {counter} ...')

            if not self.gender_data_build_flag:
                # img = cv2.imread(image[1], 1)
                img_path = os.path.join(self.data_dir, 'data', 'imdb-clean-1024', image[0])
                img = cv2.imread(img_path, 1)
                img = cv2.resize(img, (self.output_size[0], self.output_size[1]))
            # if image[0] == 'male':
            if image[1] == 'M':
                out_path = os.path.join(output_dir_test_male, str(counter) + '.jpg')
                if not self.gender_data_build_flag:
                    cv2.imwrite(out_path, img)
            elif image[1] == 'F':
                out_path = os.path.join(output_dir_test_female, str(counter) + '.jpg')
                if not self.gender_data_build_flag:
                    cv2.imwrite(out_path, img)
            # print('--('+str(counter)+')Processing--')

            self.test_final_gender_imdb[counter] = [out_path, image[1]]
        print('-'*30)

        print('-'*50)
        print(' ====> Complete to build gender dataset !')
        print('-'*50)



    # def __build_parsing_mat(self):
    #     '''
    #     --- Format of IMDB Mat file ---
    #     dob: date of birth (Matlab serial date number)
    #     photo_taken: year when the photo was taken
    #     full_path: path to file
    #     gender: 0 for female and 1 for male, NaN if unknown
    #     name: name of the celebrity
    #     face_location: location of the face. To crop the face in Matlab run
    #     img(face_location(2):face_location(4),face_location(1):face_location(3),:))
    #     face_score: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    #     second_face_score: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
    #     celeb_names (IMDB only): list of all celebrity names
    #     celeb_id (IMDB only): index of celebrity name
    #     '''
    #     imdb_data = io.loadmat(self.mat_path)

    #     imdb = imdb_data['imdb']

    #     imdb_photo_taken = imdb[0][0][1][0]
    #     imdb_full_path = imdb[0][0][2][0]
    #     imdb_gender = imdb[0][0][3][0]
    #     imdb_face_score1 = imdb[0][0][6][0]
    #     imdb_face_score2 = imdb[0][0][7][0]
    #     assert len(imdb_photo_taken) == len(imdb_full_path) == len(imdb_gender) == len(imdb_face_score1) == len(imdb_face_score2), "Length of both lists should be same!"

    #     # Path
    #     imdb_path = []
    #     for path in imdb_full_path:
    #         imdb_path.append(os.path.join(self.img_dir, path[0]))

    #     # Gender
    #     imdb_genders = []
    #     for n in range(len(imdb_gender)):
    #         # imdb_genders.append(imdb_gender[n])
    #         if imdb_gender[n] == 1:
    #             imdb_genders.append('male')
    #         else:
    #             imdb_genders.append('female')
        
    #     # Age 
    #     imdb_dob = []
    #     for file in imdb_path:
    #         temp = file.split('_')[3]
    #         temp = temp.split('-')
            
    #         if len(temp[1]) == 1:
    #             temp[1] = '0' + temp[1]
    #         if len(temp[2]) == 1:
    #             temp[2] = '0' + temp[2]

    #         if temp[1] == '00':
    #             temp[1] = '01'
    #         if temp[2] == '00':
    #             temp[2] = '01'
            
    #         imdb_dob.append('-'.join(temp))

    #     imdb_age = []
    #     for i in range(len(imdb_dob)):
    #         try:
    #             d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
    #             d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
    #             rdelta = relativedelta(d2, d1)
    #             diff = rdelta.years
    #         except Exception as ex:
    #             # print(ex)
    #             diff = -1
    #         imdb_age.append(diff)

    #     # Final imdb (age/gender/path/face_score1/face_score2)
    #     assert len(imdb_age) == len(imdb_genders) == len(imdb_path) == len(imdb_face_score1) == len(imdb_face_score2), "Length of both lists should be same!"
    #     self.final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
    #     # if self.mode == 'TRAIN':
    #     #     self.final_imdb = self.final_imdb[:int(len(self.final_imdb) * 0.9)]
    #     # elif self.mode == 'EVAL':
    #     #     self.final_imdb = self.final_imdb[int(len(self.final_imdb) * 0.9):]

    #     print('-'*30)
    #     print('len(self.final_imdb):', len(self.final_imdb))
    #     print('self.final_imdb[:3]:', self.final_imdb[:3])
    #     print('-'*30)

    #     meta = pd.DataFrame(self.final_imdb)
    #     cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']
    #     meta.columns = cols

    #     meta = meta[meta['face_score1'] != '-inf']
    #     meta = meta[meta['face_score2'] == 'nan']

    #     meta = meta.drop(['face_score1', 'face_score2'], axis=1)

    #     meta = meta.sample(frac=1)

    #     meta.to_csv(self.meta_csv_path, index=False)

    #     print('-'*50)
    #     print(' ====> Complete to build meta csv !')
    #     print('-'*50)



    # def __build_parsing_mat(self):
    #     '''
    #     --- Format of IMDB Mat file ---
    #     dob: date of birth (Matlab serial date number)
    #     photo_taken: year when the photo was taken
    #     full_path: path to file
    #     gender: 0 for female and 1 for male, NaN if unknown
    #     name: name of the celebrity
    #     face_location: location of the face. To crop the face in Matlab run
    #     img(face_location(2):face_location(4),face_location(1):face_location(3),:))
    #     face_score: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    #     second_face_score: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
    #     celeb_names (IMDB only): list of all celebrity names
    #     celeb_id (IMDB only): index of celebrity name
    #     '''

    #     imdb_data = io.loadmat(self.mat_path)
    #     imdb = imdb_data['imdb'][0][0]

    #     imdb_photo_taken = imdb[1][0]
    #     imdb_path = imdb[2][0]
    #     imdb_gender = imdb[3][0]
    #     imdb_face_score1 = imdb[6][0]
    #     imdb_face_score2 = imdb[7][0]
    #     assert len(imdb_photo_taken) == len(imdb_path) == len(imdb_gender) == len(imdb_face_score1) == len(imdb_face_score2), "Length of both lists should be same!"

    #     imdb_path = []
    #     imdb_gender = []
    #     imdb_age = []
    #     imdb_face_score1 = []
    #     imdb_face_score2 = []

    #     print('-'*30)
    #     for i in range(len(imdb_photo_taken)):

    #         if i % 50000 ==0:
    #             print(f'build dataset - idx: {i} ...')
            
    #         bYear = int(imdb[0][0][i] / 365) # Birth year
    #         taken = imdb[1][0][i] # Photo taken
    #         path = imdb[2][0][i][0] # Img path
    #         gender = imdb[3][0][i] # Female/Male
    #         name = imdb[4][0][i] # Name
    #         faceBox= imdb[5][0][i] # Face coords
    #         faceScore = imdb[6][0][i] # Face score
    #         secFaceScore = imdb[7][0][i] # Sec face score

    #         #Calculating shit
    #         age = taken - bYear
            
    #         faceScore = str(faceScore)
    #         secFaceScore = str(secFaceScore)

    #         if 'n' not in faceScore: # Implies that there isn't a face in the image
                
    #             if 'a' in secFaceScore: # Implies that no second face was found
                
    #                 if age >= 0: 

    #                     try:
    #                         gender = int(gender)
                            
    #                         # Path
    #                         path = os.path.join(self.img_dir, path)
    #                         imdb_path.append(path)
                            
    #                         # Gender
    #                         if gender == 1:
    #                             imdb_gender.append('male')
    #                         elif gender == 0:
    #                             imdb_gender.append('female')

    #                         # Age
    #                         imdb_age.append(age)

    #                         # Score
    #                         imdb_face_score1.append(faceScore)
    #                         imdb_face_score2.append(secFaceScore)
                            
    #                     except Exception as ex:
    #                         # print(gender, ex)
    #                         continue

    #     print('-'*30)

    #     # Final imdb (age/gender/path/face_score1/face_score2)
    #     assert len(imdb_age) == len(imdb_gender) == len(imdb_path), "Length of both lists should be same!"
    #     self.final_imdb = np.vstack((imdb_age, imdb_gender, imdb_path)).T
    #     # if self.mode == 'TRAIN':
    #     #     self.final_imdb = self.final_imdb[:int(len(self.final_imdb) * 0.9)]
    #     # elif self.mode == 'EVAL':
    #     #     self.final_imdb = self.final_imdb[int(len(self.final_imdb) * 0.9):]

    #     print('-'*30)
    #     print('len(self.final_imdb):', len(self.final_imdb))
    #     print('self.final_imdb[:3]:', self.final_imdb[:3])
    #     print('-'*30)

    #     meta = pd.DataFrame(self.final_imdb)
    #     cols = ['age', 'gender', 'path']
    #     meta.columns = cols

    #     # meta = meta.sample(frac=1)

    #     meta.to_csv(self.meta_csv_path, index=False)

    #     print('-'*50)
    #     print(' ====> Complete to build meta csv !')
    #     print('-'*50)



    def __len__(self):
        if self.mode == 'TRAIN':
            return len(self.train_final_gender_imdb.keys())
        elif self.mode == 'EVAL':
            return len(self.test_final_gender_imdb.keys())
    
    def __gender_one_hot_encode(self, labels, num_classes=2):
        '''
        Convert class indices into one-hot encoded vectors.

        Args:
        labels (torch.Tensor): tensor of class indices
        num_classes (int): total number of classes

        Returns:
        torch.Tensor: one-hot encoded representation of the input
        '''

        if num_classes == 2:
            if labels == 'M': # male
                one_hot = torch.tensor([0, 1]).long()
            elif labels == 'F': # female
                one_hot = torch.tensor([1, 0]).long()
            else:
                one_hot = torch.tensor([0, 0]).long()
        elif num_classes == 100:
            one_hot = np.zeros((100))
            one_hot[int(labels)] = 1
            one_hot = torch.tensor(one_hot).long()

        return one_hot

    def __getitem__(self, idx):
        
        if self.mode == 'TRAIN':
            final_gender_imdb = self.train_final_gender_imdb
            final_age_imdb = self.train_final_age_imdb
        elif self.mode == 'EVAL':
            final_gender_imdb = self.test_final_gender_imdb
            final_age_imdb = self.test_final_age_imdb

        imdb_path, imdb_genders = final_gender_imdb[idx]
        _, imdb_age = final_age_imdb[idx]
        
        # Read img
        image = Image.open(imdb_path)
        if len(np.array(image).shape) == 2:
            image = image.convert("RGB")
        w, h = image.size

        # imdb_genders_tensor = torch.tensor(int(imdb_genders))
        # if  imdb_genders == 'F':
        #     imdb_genders_tensor = torch.tensor(0.0).float()
        # elif  imdb_genders == 'M':
        #     imdb_genders_tensor = torch.tensor(1.0).float()
        
        ## One-hot genders
        imdb_genders_tensor = self.__gender_one_hot_encode(imdb_genders)

        ## Normalized ages (Regressor)
        # imdb_age = float(imdb_age)
        # imdb_age_tensor = torch.tensor(imdb_age).float()
        # # imdb_age_tensor = torch.tensor(imdb_age / self.max_age).float()

        # imdb_age = float(imdb_age)
        # imdb_age_tensor = torch.tensor(imdb_age).float()
        
        ## One-hot age
        imdb_age_tensor = self.__gender_one_hot_encode(imdb_age, num_classes=100)

        if self.random_scale:
            scale = int(min(w, h) * (np.random.uniform() + 0.5))
            resize_bl = transforms.Resize(size=scale, interpolation=PIL.Image.BILINEAR)
            resize_nn = transforms.Resize(size=scale, interpolation=PIL.Image.NEAREST)
            image = resize_bl(image)
        else:
            resize_bl = transforms.Resize(self.output_size, interpolation=PIL.Image.BILINEAR)
            image = resize_bl(image)

        if self.random_mirror:
            if np.random.uniform() < 0.5:
                image = TF.hflip(image)

        if self.random_crop:
            # pad the width if needed
            if image.size[0] < self.output_size[1]:
                image = TF.pad(image, (self.output_size[1] - image.size[0], 0))

            # pad the height if needed
            if image.size[1] < self.output_size[0]:
                image = TF.pad(image, (0, self.output_size[0] - image.size[1]))

            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.output_size)
            image = TF.crop(image, i, j, h, w)

        image = self.normalize(self.to_tensor(np.array(image) - 255.).float() + 255.)

        return image, imdb_genders_tensor.float(), imdb_age_tensor.float()