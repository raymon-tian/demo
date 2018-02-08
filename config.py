#coding=utf-8

config = dict()
# weights of teacher
config['weight_path'] = 'weight/imagenet12-vgg16/vgg16-org.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/stage1/stage1_epoch14.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/conv3_3/stage2_epoch3.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single5x/conv4_2/stage2_epoch94.pth'
# config['stu_weight_path'] = './weight/mnist-demo/mnist_org.ckpt'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single2x/conv4_2/stage2_epoch7.pth'
config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single4x/conv4_2/stage2_epoch30.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single5x/single5x_final.pth'
config['gamma'] = 1e0
config['beta'] = 1e0
config['epoch'] = 100
config['lr'] = 1e-4
config['batch_size'] = 32
config['cuda'] = True
config['seed'] = 1
config['test_batch_size'] = 64
config['save_freq'] = 2
config['phase'] = 2
config['conv_pruned_names'] = [
    ('conv1_1','conv1_2',0.25),
    # ('conv1_2','conv2_1',0.25),
    ('conv2_1','conv2_2',0.25),
    # ('conv2_2','conv3_1',0.25),
    ('conv3_1','conv3_2',0.25),
    # ('conv3_2','conv3_3',0.25),
    ('conv3_3','conv4_1',0.25),
    # ('conv4_1','conv4_2',0.25),
    ('conv4_2','conv4_3',0.25),
    # ('conv4_3','conv5_1',0.25),
    # ('conv5_1','conv5_2',0.25),
    # ('conv5_2','conv5_3',0.25),
]
config['channel_select_algo'] = 'subspace_cluster'
# config['channel_select_algo'] = 'sparse_vec'
config['model_name'] = 'vgg16'
config['dataset_name'] = 'imagenet12'
config['topC'] = 3
config['randomN'] = 10
config['exp_name'] = 'single4x/conv4_2'
config['explain'] = 'compress conv4_2'
config['start_epoch'] = 31