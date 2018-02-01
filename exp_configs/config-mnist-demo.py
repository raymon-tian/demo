#coding=utf-8

config = dict()
# weights of teacher
config['weight_path'] = './weight/mnist-demo/mnist_org.ckpt'
# config['stu_weight_path'] = 'weight/mnist-demo/sparse_vec/stage1_epoch50.pth'
config['stu_weight_path'] = 'weight/mnist-demo/subspace_cluster/stage1_epoch50.pth'
# config['stu_weight_path'] = './weight/mnist-demonet/stage1_epoch50.pth'
# config['stu_weight_path'] = './weight/mnist-demo/mnist_org.ckpt'
config['gamma'] = 1e0
config['beta'] = 1e0
config['epoch'] = 50
config['lr'] = 1e-3
config['batch_size'] = 2048
config['cuda'] = True
config['seed'] = 1
config['test_batch_size'] = 1000
config['save_freq'] = 5
config['phase'] = 2
config['conv_pruned_names'] = [
    ('conv1','conv2',0.4)

]
config['channel_select_algo'] = 'subspace_cluster'
# config['channel_select_algo'] = 'sparse_vec'
config['model_name'] = 'demo'
config['dataset_name'] = 'mnist'
config['fine_tune'] = False