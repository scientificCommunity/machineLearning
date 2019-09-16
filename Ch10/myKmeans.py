import numpy as np
import tensorflow as tf

import glob
# images_dir 下存放着需要预处理的图像
images_dir = '/data/captcha/backup/a/'

# 查找图片文件, 根据具体数据集自由添加各种图片格式(jpg, jpeg, png, bmp等等)
images_paths = glob.glob(images_dir+'*.jpg')
# images_paths += glob.glob(images_dir+'*.jpeg')
# images_paths += glob.glob(images_dir+'*.png')
print('Find {} images, the first 10 image paths are:'.format(len(images_paths)))
for path in images_paths[:10]:
    print(path)

# split training set and test data
test_split_factor = 0.2
n_test_path = int(len(images_paths)*test_split_factor)
# 转出numpy数据，方便使用
train_image_paths = np.asarray(images_paths[:-n_test_path])
test_image_paths = np.asarray(images_paths[-n_test_path:])
print('Number of train set is {}'.format(train_image_paths.shape[0]))
print('Number of test set is {}'.format(test_image_paths.shape[0]))




def parse_data(filename):
    '''
    导入数据，进行预处理，输出两张图像,
    分别是输入图像和目标图像（例如，在图像去噪中，输入的是一张带噪声图像，目标图像是无噪声图像）
    Args:
        filaneme, 图片的路径
    Returns:
        输入图像，目标图像
    '''
    # 读取图像
    image = tf.read_file(filename)
    # 解码图片
    image = tf.image.decode_image(image)

    # 数据预处理，或者数据增强，这一步根据需要自由发挥

    # 随机提取patch
    image = tf.random_crop(image, size=(100,100, 3))
    # 数据增强，随机水平翻转图像
    image = tf.image.random_flip_left_right(image)
    # 图像归一化
    image = tf.cast(image, tf.float32) / 255.0
    # 加噪声
    n_image =gaussian_noise_layer(image, 0.5)

    return n_image, image

def gaussian_noise_layer(input_image, std):
    noise = tf.random_normal(shape=tf.shape(input_image), mean=0.0, stddev=std, dtype=tf.float32)
    noise_image = tf.cast(input_image, tf.float32) + noise
    noise_image = tf.clip_by_value(noise_image, 0, 1.0)
    return noise_image

import matplotlib.pyplot as plt
#%config InlineBackend.figure_format='retina'

# 显示图像
def view_samples(samples, nrows, ncols, figsize=(5,5)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return fig, axes

def train_generator(batchsize, shuffle=True):
    '''
    生成器，用于生产训练数据
    Args:
        batchsize,训练的batch size
        shuffle, 是否随机打乱batch

    Returns:
        训练需要的数据
    '''

    with tf.Session() as sess:
        # 创建数据库
        train_dataset = tf.data.Dataset().from_tensor_slices(train_image_paths)
        # 预处理数据
        train_dataset = train_dataset.map(parse_data)
        # 设置 batch size
        train_dataset = train_dataset.batch(batchsize)
        # 无限重复数据
        train_dataset = train_dataset.repeat()
        # 洗牌，打乱
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)

        # 创建迭代器
        train_iterator = train_dataset.make_initializable_iterator()
        sess.run(train_iterator.initializer)
        train_batch = train_iterator.get_next()

        # 开始生成数据
        while True:
            try:
                x_batch, y_batch = sess.run(train_batch)
                yield (x_batch, y_batch)
            except:
                # 如果没有  train_dataset = train_dataset.repeat()
                # 数据遍历完就到end了，就会抛出异常
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()
                x_batch, y_batch = sess.run(train_batch)
                yield (x_batch, y_batch)

# 测试一下我们的代码
train_gen = train_generator(16)

iteration = 5
for i in range(iteration):
    noise_x, x = next(train_gen)
    _ = view_samples(noise_x, 4,4)
    _ = view_samples(x, 4, 4)