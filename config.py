#coding=utf-8

config = dict()
# weights of teacher
config['weight_path'] = 'weight/imagenet12-vgg16/vgg16-org.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/stage1/stage1_epoch14.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/conv3_3/stage2_epoch3.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single5x/conv4_2/stage2_epoch94.pth'
# config['stu_weight_path'] = './weight/mnist-demo/mnist_org.ckpt'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single2x/conv4_2/stage2_epoch7.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single4x/stage3/stage3_epoch170.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single4x/stage3-temp/stage3_epoch7.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single_layer/conv1_1/x1.5/stage2_epoch2_iter356.pth'
config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/stage1/stage1_epoch14.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single_layer/conv4_2/x3.5/stage2_epoch12.pth'
config['gamma'] = 1e0
config['beta'] = 1e0
config['epoch'] = 20
config['lr'] = 1e-3
config['batch_size'] = 28
config['cuda'] = True
config['seed'] = 1
config['test_batch_size'] = 28
config['test_freq'] = 10
config['save_freq'] = 4
config['phase'] = 2
config['conv_pruned_names'] = [
    # ('conv1_1','conv1_2', 2./3.),
    # ('conv1_2','conv2_1',0.25),
    # ('conv2_1','conv2_2',0.25),
    # ('conv2_2','conv3_1',0.25),
    # ('conv3_1','conv3_2',0.25),
    # ('conv3_2','conv3_3',0.25),
    # ('conv3_3','conv4_1',0.25),
    # ('conv4_1','conv4_2',0.25),
    ('conv4_2','conv4_3',2./5.),
    # ('conv4_3','conv5_1',0.25),
    # ('conv5_1','conv5_2',1),
    # ('conv5_2','conv5_3',1),
]
config['channel_select_algo'] = 'subspace_cluster'
# config['channel_select_algo'] = 'random'
config['model_name'] = 'vgg16'
config['dataset_name'] = 'imagenet12'
config['topC'] = 0
config['randomN'] = 10
config['exp_name'] = 'single_layer/conv4_2/x2.5'
config['explain'] = 'single_layer conv4_2 x2.5 cluster'
config['start_epoch'] = 1
config['resample_data_freq'] = 4
config['save_iter'] = 10000
