from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import errno
import logging
import sys
from darch.datasets import onehot_to_idx
from sklearn.metrics import f1_score

class ClassifierEvaluator:
    """Trains and evaluates a classifier on some datasets passed as argument.

    Uses a number of training tricks, namely, early stopping, keeps the model 
    that achieves the best validation performance, reduces the step size 
    after the validation performance fails to increases for some number of 
    epochs.

    """

    def __init__(self, train_dataset, val_dataset, in_d, nclasses, model_path,
                 training_epochs_max=200, time_minutes_max=180.0,
                 stop_patience=20, rate_patience=7, batch_patience=np.inf,
                 save_patience=2, rate_mult=0.5, batch_mult=2,
                 optimizer_type='adam', sgd_momentum=0.99,
                 learning_rate_init=1e-3, learning_rate_min=1e-6, batch_size_init=32,
                 display_step=1, test_dataset=None, in_dtype=tf.float32, log_file='deep_arch.log', tensorboard_dir='tb_logs',
                 evaluation='accuracy'):

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.in_d = list(in_d)
        self.nclasses = nclasses

        self.training_epochs = training_epochs_max
        self.time_minutes_max = time_minutes_max
        self.display_step = display_step
        self.stop_patience = stop_patience
        self.rate_patience = rate_patience
        self.batch_patience = batch_patience
        self.save_patience = save_patience
        self.rate_mult = rate_mult
        self.batch_mult = batch_mult
        self.learning_rate_init = learning_rate_init
        self.learning_rate_min = learning_rate_min
        self.batch_size_init = batch_size_init
        self.optimizer_type = optimizer_type
        self.sgd_momentum = sgd_momentum
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.in_dtype = in_dtype
        self.tensorboard_dir = tensorboard_dir
        self.evaluation = evaluation
        
        if not os.path.exists(os.path.dirname(model_path)):
            try:
                os.makedirs(os.path.dirname(model_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        logging.basicConfig(filename=log_file, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    def eval_model(self, b, model_nb):

        dir_checkpoint_for_model = os.path.join(self.model_path, 'model_'+str(model_nb))
        if not os.path.exists(os.path.dirname(dir_checkpoint_for_model)):
            try:
                print('Creating dir for ', 'model_'+str(model_nb))
                os.makedirs(dir_checkpoint_for_model)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise



        tf.reset_default_graph()

        x = tf.placeholder(shape=[None] + self.in_d, dtype=self.in_dtype)
        y = tf.placeholder("float", [None, self.nclasses])

        learning_rate = tf.placeholder("float")
        learning_rate_val = self.learning_rate_init
        batch_size = self.batch_size_init
        sgd_momentum = self.sgd_momentum

        # compilation 
        train_feed = {}
        eval_feed = {}
        pred = b.compile(x, train_feed, eval_feed)
        saver = tf.train.Saver()

        # Define loss and optimizer
        with tf.name_scope('softmax_cross_entropy'):
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        # chooses the optimizer. (this can be put in a function).
        if self.optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif self.optimizer_type == 'sgd_mom':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=sgd_momentum)
        else:
            raise ValueError("Unknown optimizer.")
        optimizer = optimizer.minimize(cost)

        # For computing the accuracy of the model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        num_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))

        def compute_accuracy(dataset, ev_feed, ev_batch_size):
            nc = 0
            n_left = dataset.get_num_examples()
            while n_left > 0:
                images, labels = dataset.next_batch(ev_batch_size)
                ev_feed.update({x: images, 
                                y: labels})
                nc += num_correct.eval(ev_feed)
                # update the number of examples left.
                eff_batch_size = labels.shape[0]
                n_left -= eff_batch_size
            
            acc = float(nc) / dataset.get_num_examples()
            return acc

        def compute_fscore(dataset, ev_feed, ev_batch_size):
            n_left = dataset.get_num_examples()

            true_label = np.zeros(n_left)
            predicted_label = np.zeros(n_left)
            idx = 0

            while n_left > 0:
                images, labels = dataset.next_batch(ev_batch_size)
                eff_batch_size = labels.shape[0]
                true_label[idx:idx+eff_batch_size] = onehot_to_idx(labels)
                ev_feed.update({x: images,
                                y: labels})
                predictions = tf.argmax(pred, 1).eval(ev_feed)
                predicted_label[idx:idx + eff_batch_size] = predictions
                # update the number of examples left.
                idx += eff_batch_size
                n_left -= eff_batch_size

            return f1_score(true_label, predicted_label, average=self.evaluation)


        if self.evaluation == 'accuracy':
            evaluation_fn = compute_accuracy
        else:
            evaluation_fn = compute_fscore

        average_cost_placeholder = tf.placeholder(tf.float32, shape=())
        valeval_placeholder = tf.placeholder(tf.float32, shape=())
        tr_cost_summary_average = tf.summary.scalar('cost_average'.format(eval=self.evaluation), average_cost_placeholder)
        val_eval_summary_full = tf.summary.scalar('val_{eval}_full'.format(eval=self.evaluation), valeval_placeholder)

        tb_folder_for_run = os.path.join(self.tensorboard_dir, 'model_'+str(model_nb))
        train_writer = tf.summary.FileWriter(tb_folder_for_run)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session(
                #config=tf.ConfigProto(
                #    allow_soft_placement=True
                #)
            ) as sess:
            train_writer.add_graph(sess.graph)

            sess.run(init)
            # for early stopping
            best_vacc = - np.inf
            best_vacc_saved = - np.inf
            stop_counter = self.stop_patience
            rate_counter = self.rate_patience
            batch_counter = self.batch_patience
            save_counter = self.save_patience
            time_start = time.time()

            train_num_examples = self.train_dataset.get_num_examples()

            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(train_num_examples / batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = self.train_dataset.next_batch(batch_size)
                    #print((batch_x.shape, batch_y.shape))
                    #import ipdb; ipdb.set_trace()
                    # Run optimization op (backprop) and cost op (to get loss value)
                    train_feed.update({x: batch_x, 
                                       y: batch_y, 
                                       learning_rate: learning_rate_val})

                    _, c = sess.run([optimizer, cost], feed_dict=train_feed)
                    # Compute average loss
                    avg_cost += c / total_batch

                # early stopping
                vacc = evaluation_fn(self.val_dataset, eval_feed, 1024)
                summary = sess.run(val_eval_summary_full, feed_dict={valeval_placeholder: vacc})
                train_writer.add_summary(summary, epoch)

                summary = sess.run(tr_cost_summary_average, feed_dict={average_cost_placeholder: avg_cost})
                train_writer.add_summary(summary, epoch)


                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    log_str = ' '.join(str(x) for x in [
                        "Time:", "%7.1f" % (time.time() - time_start),
                        "Epoch:", '%04d' % (epoch + 1),
                        "cost=", "{:.9f}".format(avg_cost),
                        "val_acc=", "%.5f" % vacc,
                        "learn_rate=", '%.3e' % learning_rate_val
                    ])
                    logging.info(log_str)

                if best_vacc < vacc:
                    best_vacc = vacc
                    # reinitialize all the counters.
                    stop_counter = self.stop_patience
                    rate_counter = self.rate_patience
                    batch_counter = self.batch_patience
                    save_counter = self.save_patience
                else:
                    stop_counter -= 1
                    rate_counter -= 1
                    batch_counter -= 1
                    if stop_counter == 0:
                        break   

                    if rate_counter == 0:
                        learning_rate_val *= self.rate_mult
                        rate_counter = self.rate_patience

                        if learning_rate_val < self.learning_rate_min:
                            learning_rate_val = self.learning_rate_min

                    if batch_counter == 0:
                        batch_size *= self.batch_mult
                        batch_counter = self.batch_patience

                    if best_vacc_saved < vacc:
                        save_counter -= 1

                        if save_counter == 0:
                            save_path = saver.save(sess, os.path.join(dir_checkpoint_for_model, 'model.ckpt'))
                            logging.info("Model saved in file: %s" % save_path)

                            save_counter = self.save_patience
                            best_vacc_saved = vacc

                # at the end of the epoch, if spent more time than budget, exit.
                time_now = time.time()
                if (time_now - time_start) / 60.0 > self.time_minutes_max:
                    break

            train_writer.close()
            # if the model saved has better performance than the current model,
            # load it.
            if best_vacc_saved > vacc:
                saver.restore(sess, save_path)
                logging.info("Model restored from file: %s" % save_path)

            logging.info("Optimization Finished!")

            vacc = evaluation_fn(self.val_dataset, eval_feed, 1024)
            logging.info("Validation {eval}: {measure}".format(eval=self.evaluation, measure=vacc))

            if self.test_dataset is not None:
                tacc = evaluation_fn(self.test_dataset, eval_feed, 1024)
                logging.info("Test {eval}: {measure}".format(eval=self.evaluation, measure=tacc))

        return vacc
