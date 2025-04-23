"""--------------------------------------------------------------
三、CNN模型训练
训练模型：共八层神经网络，卷积层特征提取，池化层降维,全连接层进行分类。
训练数据：22784，测试数据：727，训练集：测试集=20:1
共两类：我的人脸（yes),不是我的人脸（no）。
共八层： 第一、二层（卷积层1、池化层1），输入图片64*64*3，输出图片32*32*32
        第三、四层（卷积层2、池化层2），输入图片32*32*32，输出图片16*16*64
        第五、六层（卷积层3、池化层3），输入图片16*16*64，输出图片8*8*64
        第七层（全连接层），输入图片8*8*64，reshape到1*4096，输出1*512
        第八层（输出层），输入1*512，输出1*2
学习率：0.01
损失函数：交叉熵
优化器：Adam
------------------------------------------------------------------"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import shutil
import random
import os
import time
import matplotlib.pyplot as plt
import sys
import units


"""定义读取人脸数据函数，根据不同的人名，分配不同的onehot值"""
def get_images_labels(in_path, height, width):
    global imgs, labs
    print(f"\n开始处理文件夹: {in_path}")
    total_files = 0
    processed_files = 0
    
    # 首先统计总文件数
    for root, dirs, files in os.walk(in_path):
        total_files += len([f for f in files if f.endswith('.jpg')])
    
    print(f"发现 {total_files} 个jpg文件")
    
    # 处理文件
    for root, dirs, files in os.walk(in_path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        print(f"警告: 无法读取图片 {file_path}")
                        continue
                        
                    t, b, l, r = units.img_padding(img)
                    img_big = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    img_big = cv2.resize(img_big, (height, width))
                    imgs.append(img_big)
                    labs.append(in_path)
                    processed_files += 1
                    
                    if processed_files % 1000 == 0:
                        print(f"已处理 {processed_files}/{total_files} 个文件")
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
    
    print(f"完成处理文件夹 {in_path}: 成功处理 {processed_files}/{total_files} 个文件")


"""定义CNN模型"""
def create_model(input_shape, num_classes):
    model = models.Sequential([
        # 第一层：卷积层
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第二层：卷积层
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第三层：卷积层
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 第四层：卷积层
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


if __name__ == '__main__':
    try:
        """定义参数"""
        faces_my_path = './faces_my'         # [0,0,0,1]
        faces_sxx_path = './faces_sxx'       # [0,0,1,0]
        faces_wtt_path = './faces_wtt'       # [0,1,0,0]
        faces_other_path = './faces_other'   # [1,0,0,0]
        num_class = 4
        batch_size = 64
        initial_learning_rate = 0.0005
        size = 64
        
        print("开始初始化...")
        
        # 检查文件夹是否存在
        for path in [faces_my_path, faces_sxx_path, faces_wtt_path, faces_other_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件夹不存在: {path}")
            print(f"检查文件夹 {path}: 存在")
        
        # 数据增强
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        
        imgs = []  # 存放人脸图片
        labs = []  # 存放人脸图片对应的标签

        """1、读取人脸数据、分配标签"""
        print("\n开始加载所有图片数据...")
        get_images_labels(faces_my_path, size, size)
        get_images_labels(faces_sxx_path, size, size)
        get_images_labels(faces_wtt_path, size, size)
        get_images_labels(faces_other_path, size, size)

        imgs = np.array(imgs)
        print(f"\n总共加载了 {len(imgs)} 张图片")
        print(f"标签数量: {len(labs)}")

        # 转换标签为one-hot编码
        print("\n开始转换标签...")
        for id, lab in enumerate(labs):
            if lab == faces_my_path:
                labs[id] = [0, 0, 0, 1]
            elif lab == faces_sxx_path:
                labs[id] = [0, 0, 1, 0]
            elif lab == faces_wtt_path:
                labs[id] = [0, 1, 0, 0]
            else:
                labs[id] = [1, 0, 0, 0]
        labs = np.array(labs)
        print("标签转换完成")

        """2、随机划分测试集与训练集"""
        print("\n开始划分训练集和测试集...")
        train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.2,
                                                            random_state=42, stratify=labs)
        
        """3、归一化"""
        print("开始数据归一化...")
        train_x = train_x.astype('float32') / 255.0
        test_x = test_x.astype('float32') / 255.0
        
        print('训练集大小: ', len(train_x))
        print('测试集大小: ', len(test_x))

        """4、创建并编译模型"""
        print("\n开始创建和编译模型...")
        model = create_model((size, size, 3), num_class)
        
        # 使用学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        """5、训练模型"""
        print("\n开始准备训练数据...")
        # 创建数据增强生成器
        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_data = train_data.shuffle(10000).batch(batch_size)
        train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                   num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.prefetch(tf.data.AUTOTUNE)
        
        # 创建验证数据集
        val_data = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
        val_data = val_data.prefetch(tf.data.AUTOTUNE)
        
        # 添加早停和模型检查点回调
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                './model_multi/train_faces.keras',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        print("\n开始训练模型...")
        history = model.fit(
            train_data,
            epochs=200,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n训练完成！")
        
        """6、评估模型"""
        test_loss, test_acc = model.evaluate(test_x, test_y)
        print(f'Test accuracy: {test_acc}')
        
        """7、保存训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig('./model_multi/training_history.png')
        plt.close()

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
