#coding=utf-8

config = dict()
# weights of teacher
config['weight_path'] = 'weight/imagenet12-vgg16/vgg16-org.pth'
# config['stu_weight_path'] = 'weight/mnist-demo/sparse_vec/stage1_epoch50.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/conv3_3/stage2_epoch3.pth'
config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single5x/conv1_1/stage2_epoch33.pth'
# config['stu_weight_path'] = './weight/mnist-demo/mnist_org.ckpt'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single2x/conv4_2/stage2_epoch7.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/vgg16-org.pth'
# config['stu_weight_path'] = 'weight/imagenet12-vgg16/subspace_cluster/single5x/single5x_final.pth'
config['gamma'] = 2*1e0
config['beta'] = 1e0
config['epoch'] = 100
config['lr'] = 1e-3
config['batch_size'] = 32
config['cuda'] = True
config['seed'] = 1
config['test_batch_size'] = 64
config['save_freq'] = 1
config['phase'] = 2
config['conv_pruned_names'] = [
    # ('conv1_1','conv1_2',0.2),
    # ('conv1_2','conv2_1',0.2),
    ('conv2_1','conv2_2',0.2),
    # ('conv2_2','conv3_1',0.2),
    # ('conv3_1','conv3_2',0.2),
    # ('conv3_2','conv3_3',0.2),
    # ('conv3_3','conv4_1',0.2),
    # ('conv4_1','conv4_2',0.2),
    # ('conv4_2','conv4_3',0.2),
    # ('conv4_3','conv5_1',0.2),
    # ('conv5_1','conv5_2',0.2),
    # ('conv5_2','conv5_3',0.2),
]
config['channel_select_algo'] = 'subspace_cluster'
# config['channel_select_algo'] = 'sparse_vec'
config['model_name'] = 'vgg16'
config['dataset_name'] = 'imagenet12'
config['topC'] = 3
config['randomN'] = 10
config['exp_name'] = 'single5x/conv2_1'
config['explain'] = 'compress conv2_1'
config['start_epoch'] = 1
