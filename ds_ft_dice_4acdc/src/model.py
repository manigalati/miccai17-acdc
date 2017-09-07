from __future__ import division
import os
import time
from glob import glob
import cv2
import scipy.ndimage
from ops import *
from utils import *
from seg_eval import *
class unet_3D_xy(object):
    """ Implementation of 3D U-net"""
    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_chn     = param_set['inputI_chn']
        self.output_chn     = param_set['output_chn']
        self.resize_r       = param_set['resize_r']
        self.traindata_dir  = param_set['traindata_dir']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.beta1          = param_set['beta1']
        self.epoch          = param_set['epoch']
        self.model_name     = param_set['model_name']
        self.save_intval    = param_set['save_intval']
        self.testdata_dir   = param_set['testdata_dir']
        self.labeling_dir   = param_set['labeling_dir']
        # input size for each dim
        self.inputI_size = param_set['inputI_size']
        self.inputI_size = [int(s) for s in self.inputI_size.split(',')]
        # output size for each dim
        self.outputI_size = param_set['outputI_size']
        self.outputI_size = [int(s) for s in self.outputI_size.split(',')]
        # overlapping ita for each dim
        self.ovlp_ita = param_set['ovlp_ita']
        self.ovlp_ita = [int(s) for s in self.ovlp_ita.split(',')]

        self.rename_map = param_set['rename_map']
        self.rename_map = [int(s) for s in self.rename_map.split(',')]

        # build model graph
        self.build_model()

    # dice loss function
    def dice_loss_fun(self, pred, input_gt):
        input_gt = tf.one_hot(input_gt, self.output_chn)
        # print(input_gt.shape)
        dice = 0
        for i in range(self.output_chn):
            inse = tf.reduce_mean(pred[:, :, :, :, i]*input_gt[:, :, :, :, i])
            l = tf.reduce_sum(pred[:, :, :, :, i]*pred[:, :, :, :, i])
            r = tf.reduce_sum(input_gt[:, :, :, :, i] * input_gt[:, :, :, :, i])
            dice = dice + 2*inse/(l+r+0.0001)
        return -dice

    # build model graph
    def build_model(self):
        # input
        self.input_I = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.inputI_size[0], self.inputI_size[1], self.inputI_size[2], self.inputI_chn], name='inputI')
        self.input_gt = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.inputI_size[0], self.inputI_size[1], self.inputI_size[2]], name='target')
        # probability for classes
        self.pred_prob, self.pred_label, self.aux0_prob = self.unet_3D_model(self.input_I)
        # ========= dice loss
        self.main_dice_loss = self.dice_loss_fun(self.pred_prob, self.input_gt)
        self.aux0_dice_loss = self.dice_loss_fun(self.aux0_prob, self.input_gt)
        #
        self.total_dice_loss = self.main_dice_loss + 0.8*self.aux0_dice_loss

        self.total_loss = self.total_dice_loss
        # self.total_loss = self.total_wght_loss

        # trainable variables
        self.u_vars = tf.trainable_variables()

        # extract the layers for fine tuning
        ft_layer = ['conv1/kernel:0',
                    'conv2/kernel:0',
                    'conv3a/kernel:0',
                    'conv3b/kernel:0',
                    'conv4a/kernel:0',
                    'conv4b/kernel:0']

        self.ft_vars = []
        for var in self.u_vars:
            for k in range(len(ft_layer)):
                if ft_layer[k] in var.name:
                    self.ft_vars.append(var)
                    break

        # create model saver
        self.saver = tf.train.Saver()
        # saver to load pre-trained C3D model
        self.saver_ft = tf.train.Saver(self.ft_vars)

    # 3D unet graph
    def unet_3D_model(self, inputI):
        """3D U-net"""
        phase_flag = (self.phase=='train')
        concat_dim = 4
        # with tf.variable_scope("unet3D_model") as scope:
        # down-sampling path
        # compute down-sample path in gpu0
        with tf.device("/gpu:0"):
            # conv1_1 = conv_bn_relu(input=inputI, output_chn=64, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv1')
            conv1_1 = conv3d(input=inputI, output_chn=64, kernel_size=3, stride=1, use_bias=False, name='conv1')
            conv1_bn = tf.contrib.layers.batch_norm(conv1_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv1_batch_norm")
            conv1_relu = tf.nn.relu(conv1_bn, name='conv1_relu')
            #
            # conv2_1 = conv_bn_relu(input=pool1, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv2')
            conv2_1 = conv3d(input=conv1_relu, output_chn=128, kernel_size=3, stride=1, use_bias=False, name='conv2')
            conv2_bn = tf.contrib.layers.batch_norm(conv2_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv2_batch_norm")
            conv2_relu = tf.nn.relu(conv2_bn, name='conv2_relu')
            pool2 = tf.layers.max_pooling3d(inputs=conv2_relu, pool_size=2, strides=2, name='pool2')
            #
            # conv3_1 = conv_bn_relu(input=pool2, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv3a')
            conv3_1 = conv3d(input=pool2, output_chn=256, kernel_size=3, stride=1, use_bias=False, name='conv3a')
            conv3_1_bn = tf.contrib.layers.batch_norm(conv3_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_1_batch_norm")
            conv3_1_relu = tf.nn.relu(conv3_1_bn, name='conv3_1_relu')
            # conv3_2 = conv_bn_relu(input=conv3_1, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv3b')
            conv3_2 = conv3d(input=conv3_1_relu, output_chn=256, kernel_size=3, stride=1, use_bias=False, name='conv3b')
            conv3_2_bn = tf.contrib.layers.batch_norm(conv3_2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv3_2_batch_norm")
            conv3_2_relu = tf.nn.relu(conv3_2_bn, name='conv3_2_relu')
            pool3 = tf.layers.max_pooling3d(inputs=conv3_2_relu, pool_size=2, strides=2, name='pool3')
            #
            # conv4_1 = conv_bn_relu(input=pool3, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv4a')
            conv4_1 = conv3d(input=pool3, output_chn=512, kernel_size=3, stride=1, use_bias=False, name='conv4a')
            conv4_1_bn = tf.contrib.layers.batch_norm(conv4_1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_1_batch_norm")
            conv4_1_relu = tf.nn.relu(conv4_1_bn, name='conv4_1_relu')
            # conv4_2 = conv_bn_relu(input=conv4_1, output_chn=512, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='Conv4b')
            conv4_2 = conv3d(input=conv4_1_relu, output_chn=512, kernel_size=3, stride=1, use_bias=False, name='conv4b')
            conv4_2_bn = tf.contrib.layers.batch_norm(conv4_2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=phase_flag, scope="conv4_2_batch_norm")
            conv4_2_relu = tf.nn.relu(conv4_2_bn, name='conv4_2_relu')

        # up-sampling path
        # compute up-sample path in gpu1
        with tf.device("/gpu:1"):
            deconv1_1 = deconv_bn_relu(input=conv4_2_relu, output_chn=256, is_training=phase_flag, name='deconv1_1')
            #
            concat_1 = tf.concat([deconv1_1, conv3_2_relu], axis=concat_dim, name='concat_1')
            deconv1_2 = conv_bn_relu(input=concat_1, output_chn=256, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv1_2')
            deconv1_3 = conv_bn_relu(input=deconv1_2, output_chn=128, kernel_size=3, stride=1, use_bias=False, is_training=phase_flag, name='deconv1_3')
            deconv2_1 = deconv_bn_relu(input=deconv1_3, output_chn=128, is_training=phase_flag, name='deconv2_1')
            #
            concat_2 = tf.concat([deconv2_1, conv2_relu], axis=concat_dim, name='concat_2')
            deconv2_2 = conv_bn_relu(input=concat_2, output_chn=64, kernel_size=3,stride=1, use_bias=False, is_training=phase_flag, name='deconv2_2')
            deconv2_3 = conv_bn_relu(input=deconv2_2, output_chn=64, kernel_size=3,stride=1, use_bias=False, is_training=phase_flag, name='deconv2_3')
            #
            # predicted probability
            pred_prob = conv3d(input=deconv2_3, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='pred_prob')
            # ======================
            # auxiliary prediction 0
            aux0_conv = conv3d(input=deconv1_3, output_chn=self.output_chn, kernel_size=1, stride=1, use_bias=True, name='aux0_conv')
            aux0_prob = Deconv3d(input=aux0_conv, output_chn=self.output_chn, name='aux0_prob')

        with tf.device("/cpu:0"):
            # predicted labels
            soft_prob = tf.nn.softmax(pred_prob, name='pred_soft')
            pred_label = tf.argmax(soft_prob, axis=4, name='argmax')

        return pred_prob, pred_label, aux0_prob

    def set5layersLR_Adam(self, LR1, LR2, LR3a, LR3b, LR4a, LR4b, restLR):
        opt1 = tf.train.AdamOptimizer(LR1, beta1=self.beta1)
        opt2 = tf.train.AdamOptimizer(LR2, beta1=self.beta1)
        opt3a = tf.train.AdamOptimizer(LR3a, beta1=self.beta1)
        opt3b = tf.train.AdamOptimizer(LR3b, beta1=self.beta1)
        opt4a = tf.train.AdamOptimizer(LR4a, beta1=self.beta1)
        opt4b = tf.train.AdamOptimizer(LR4b, beta1=self.beta1)
        optrest = tf.train.AdamOptimizer(restLR, beta1=self.beta1)
        grads = tf.gradients(self.total_loss, tf.trainable_variables())
        tmp = tf.trainable_variables()
        top1 = opt1.apply_gradients(zip(grads[0:3], tmp[0:3]))
        top2 = opt2.apply_gradients(zip(grads[3:6], tmp[3:6]))
        top3a = opt3a.apply_gradients(zip(grads[6:9], tmp[6:9]))
        top3b = opt3b.apply_gradients(zip(grads[9:12], tmp[9:12]))
        top4a = opt4a.apply_gradients(zip(grads[12:15], tmp[12:15]))
        top4b = opt4b.apply_gradients(zip(grads[15:18], tmp[15:18]))
        toprest = optrest.apply_gradients(zip(grads[18:], tmp[18:]))
        train_op = tf.group(top1, top2, top3a, top3b, top4a, top4b, toprest)
        print "setting layerwise learining rate......"
        return train_op

    # train function
    def train(self):
        """Train 3D U-net"""
        # ====== initialization
        global_steps = tf.Variable(0, trainable=False)
        # moving average
        variable_average = tf.train.ExponentialMovingAverage(0.99, global_steps)
        variable_average_op = variable_average.apply(tf.trainable_variables())
        # ======

        # save .log
        self.log_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # ====== Adam
        learning_rate = self.lr
        train_op = self.set5layersLR_Adam(learning_rate, learning_rate, learning_rate,
                                     learning_rate, learning_rate, learning_rate, learning_rate)
        # control train_op to run first
        with tf.control_dependencies([train_op, variable_average_op]):
            train_op_ = tf.no_op(name='train')
        # ======

        pair_list = []
        for p in range(100):
            img_path = os.path.join(self.traindata_dir, ('pat_' + str(p) + '.nii.gz'))
            gt_path = os.path.join(self.traindata_dir, ('pat_' + str(p) + '_gt.nii.gz'))
            pair_list.append(img_path)
            pair_list.append(gt_path)
        img_clec, label_clec = load_data_pairs(pair_list, self.resize_r, self.rename_map)
        # temporary file to save loss
        loss_log = open("loss.txt", "w")

        # ====== initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # load C3D model
        self.initialize_finetune()
        # ======

        for epoch in np.arange(self.epoch):
            start_time = time.time()
            # train batch
            batch_img, batch_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, rot_flag=True)

            # for sl in range(batch_img.shape[3]):
            #     cv2.imshow("slice", np.concatenate(((batch_img[0, :, :, sl, 0]*100).astype('uint8'), 50*batch_label[0, :, :, sl].astype('uint8'))))
            #     cv2.waitKey(0)

            # validation batch
            batch_val_img, batch_val_label = get_batch_patches(img_clec, label_clec, self.inputI_size, self.batch_size, chn=1, rot_flag=True)

            # Update 3D U-net
            _, cur_train_loss = self.sess.run([train_op_, self.total_loss], feed_dict={self.input_I: batch_img, self.input_gt: batch_label})
            # self.log_writer.add_summary(summary_str, counter)

            # current and validation loss
            # cur_valid_loss = self.total_loss.eval({self.input_I: batch_val_img, self.input_gt: batch_val_label})
            cur_valid_loss = 0
            # cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: batch_val_img})
            # dice_loss = self.sess.run(self.dice_loss, feed_dict={self.input_I: batch_val_img, self.input_gt: batch_val_label})
            print np.unique(batch_label)
            # print np.unique(cube_label)
            # dice value
            # dice_c = []
            # for c in range(self.output_chn):
            #     ints = np.sum(((batch_val_label[0,:,:,:]==c)*1)*((cube_label[0,:,:,:]==c)*1))
            #     union = np.sum(((batch_val_label[0,:,:,:]==c)*1) + ((cube_label[0,:,:,:]==c)*1)) + 0.0001
            #     dice_c.append((2.0*ints)/union)
            # print dice_c


            loss_log.write("%s    %s\n" % (cur_train_loss, cur_valid_loss))

            counter += 1
            print("Epoch: [%2d] time: %4.4f, train_loss: %.8f, valid_loss: %.8f" % (epoch, time.time() - start_time, cur_train_loss, cur_valid_loss))

            if np.mod(counter, self.save_intval) == 0:
                self.save_chkpoint(self.chkpoint_dir, self.model_name, counter)

        loss_log.close()

    # test function for cross validation
    def test4crsv(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # all dice
        all_dice = np.zeros([100, self.output_chn])

        # test
        test_cnt = 0
        for k in range(101, 151):
            print "========== processing No. %d volume..." % k
            # load the volume
            test_img_path = os.path.join(self.testdata_dir, ('patient' + str(k) + '_ED.nii.gz'))
            vol_file = nib.load(test_img_path)
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = np.array([vol_data.shape[0], vol_data.shape[1], 32]).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp

                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_orig = compose_label_cube2vol(cube_label_list, resize_dim, self.inputI_size, self.ovlp_ita, self.output_chn)
            composed_label = np.zeros(composed_orig.shape, dtype='int16')
            # rename label
            for i in range(len(self.rename_map)):
                composed_label[composed_orig == i] = self.rename_map[i]
            composed_label = composed_label.astype('int16')
            print np.unique(composed_label)

            # for s in range(composed_label.shape[2]):
            #     cv2.imshow('volume_seg', np.concatenate(((vol_data_resz[:, :, s]*255.0).astype('uint8'), (composed_label[:, :, s]/4).astype('uint8')), axis=1))
            #     cv2.waitKey(30)

            # save predicted label
            composed_label_resz = resize(composed_label, vol_data.shape, order=0, preserve_range=True)
            composed_label_resz = composed_label_resz.astype('int16')

            # remove minor connected components
            # composed_label_resz = remove_complex_cc(composed_label_resz, rej_ratio=0.3, rename_map=self.rename_map)
            # composed_label_resz = remove_minor_cc(composed_label_resz, rej_ratio=0.3, rename_map=self.rename_map)

            labeling_path = os.path.join(self.labeling_dir, ('patient' + str(k) + '_ED.nii.gz'))
            labeling_vol = nib.Nifti1Image(composed_label_resz, ref_affine)
            nib.save(labeling_vol, labeling_path)

            # # evaluation
            # gt_path = os.path.join(self.testdata_dir, ('pat_' + str(k) + '_gt.nii.gz'))
            # gt_file = nib.load(gt_path)
            # gt_label = gt_file.get_data().copy()
            # k_dice_c = seg_eval_metric(composed_label_resz, gt_label)
            # print k_dice_c
            # all_dice[test_cnt, :] = np.asarray(k_dice_c)

            test_cnt = test_cnt + 1

        # mean_dice = np.mean(all_dice, axis=0)
        # print "average dice: "
        # print mean_dice

    # test the model
    def test_generate_map(self):
        """Test 3D U-net"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        start_time = time.time()
        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # test
        test_cnt = 0
        for k in range(100, 200):
            print "========== processing No. %d volume..." % k
            # load the volume
            test_img_path = os.path.join(self.testdata_dir, ('pat_' + str(k) + '.nii.gz'))
            vol_file = nib.load(test_img_path)
            ref_affine = vol_file.affine
            # get volume data
            vol_data = vol_file.get_data().copy()
            resize_dim = np.array([vol_data.shape[0], vol_data.shape[1], 32]).astype('int')
            vol_data_resz = resize(vol_data, resize_dim, order=1, preserve_range=True)
            # normalization
            vol_data_resz = vol_data_resz.astype('float32')
            vol_data_resz = vol_data_resz / 255.0

            # decompose volume into list of cubes
            cube_list = decompose_vol2cube(vol_data_resz, self.batch_size, self.inputI_size, self.inputI_chn, self.ovlp_ita)
            # predict on each cube
            cube_prob_list = []
            cube_label_list = []
            for c in range(len(cube_list)):
                cube2test = cube_list[c]
                mean_temp = np.mean(cube2test)
                dev_temp = np.std(cube2test)
                cube2test_norm = (cube2test - mean_temp) / dev_temp
                # probability cube
                cube_probs = self.sess.run(self.pred_prob, feed_dict={self.input_I: cube2test_norm})
                cube_prob_list.append(cube_probs)
                # label cube
                cube_label = self.sess.run(self.pred_label, feed_dict={self.input_I: cube2test_norm})
                cube_label_list.append(cube_label)
                # print np.unique(cube_label)
            # compose cubes into a volume
            composed_prob_orig = compose_prob_cube2vol(cube_prob_list, resize_dim, self.inputI_size, self.ovlp_ita, self.output_chn)

            # resize probability map and save
            # composed_prob_resz = np.array([vol_data.shape[0], vol_data.shape[1], vol_data.shape[2], self.output_chn])
            min_prob = np.min(composed_prob_orig)
            max_prob = np.max(composed_prob_orig)
            for p in range(self.output_chn):
                composed_prob_resz = resize(composed_prob_orig[:,:,:,p], vol_data.shape, order=1, preserve_range=True)

                # composed_prob_resz = (composed_prob_resz - np.min(composed_prob_resz)) / (np.max(composed_prob_resz) - np.min(composed_prob_resz))
                composed_prob_resz = (composed_prob_resz - min_prob) / (max_prob - min_prob)
                composed_prob_resz = composed_prob_resz * 255
                composed_prob_resz = composed_prob_resz.astype('int16')

                c_map_path = os.path.join(self.labeling_dir, ('auxi_' + str(k) + '_c' + str(p) + '.nii.gz'))
                c_map_vol = nib.Nifti1Image(composed_prob_resz, ref_affine)
                nib.save(c_map_vol, c_map_path)

            test_cnt = test_cnt + 1


    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        model_dir = "%s_%s" % (self.batch_size, self.outputI_size[0])
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.batch_size, self.outputI_size[0])
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    # load C3D model
    def initialize_finetune(self):
        checkpoint_dir = '../outcome/model/C3D_unet_1chn'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver_ft.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))