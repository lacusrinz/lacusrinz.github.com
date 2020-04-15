---
title: 使用TensorFlow 实现物体识别 -- 多GPU运行
date: 2020-01-17 21:40:59
tags: [tensorflow]
categories: 机器学习应用
---

## 目标
使用多GPU，在SSD_Mobilenet_v1模型上训练文1中的猫狗数据集

<!--more-->

## 环境
Ubuntu 18.04.2 下使用docker镜像：
tensorflow/tensorflow   1.15.0-gpu-py3  
CUDA Version               10.2

## 问题
直接使用`model_main.py`文件默认使用第一块GPU， 需要将`train_and_evaluate`方法中的的`estimator`设置为支持多GPU，这里我们使用最简单的单机双卡，配置如下
```python
mirrored_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=1))
config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, 
                                  train_distribute=mirrored_strategy, 
                                  )
```

直接运行会有各种各样的报错，报错与tf版本相关

## 解决
通过以下参考 [issue#5421](https://github.com/tensorflow/models/issues/5421)，进行调整

**关于版本**
tf 1.13.1 版本以前的，无法进行多GPU训练

**报错处理**

问题1
```shell
ValueError: Variable FeatureExtractor/MobilenetV2/Conv/weights/replica_1/ExponentialMovingAverage/ does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
```
具体内容可以参考 [issue#27392](https://github.com/tensorflow/tensorflow/issues/27392)
可以通过在训练中关闭`use_moving_average`来暂时解决
```shell
train_config: {
  batch_size: 2
  optimizer {
    rms_prop_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.004
          decay_steps: 800720
          decay_factor: 0.95
        }
      }
      momentum_optimizer_value: 0.9
      decay: 0.9
      epsilon: 1.0
    }
    use_moving_average: false  # add this line
  }
```

问题2
```shell
...
  File "/data/fanzong/miniconda3/envs/tf_cuda10/lib/python3.7/site-packages/tensorflow/python/distribute/distribute_lib.py", line 126, in _require_cross_replica_or_default_context_extended
    raise RuntimeError("Method requires being in cross-replica context, use "
RuntimeError: Method requires being in cross-replica context, use get_replica_context().merge_call()

```
可以通过注释掉`model_lib.py`里面的`model_fn`方法下面scaffold相关代码
```python
    # EVAL executes on CPU, so use regular non-TPU EstimatorSpec.
    if use_tpu and mode != tf.estimator.ModeKeys.EVAL:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          scaffold_fn=scaffold_fn,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metric_ops,
          export_outputs=export_outputs)
    else:
      ## for multi gpu training
      # if scaffold is None:
      #   keep_checkpoint_every_n_hours = (
      #       train_config.keep_checkpoint_every_n_hours)
      #   saver = tf.train.Saver(
      #       sharded=True,
      #       keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
      #       save_relative_paths=True)
      #   tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
      #   scaffold = tf.train.Scaffold(saver=saver)
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=detections,
          loss=total_loss,
          train_op=train_op,
          eval_metric_ops=eval_metric_ops,
          export_outputs=export_outputs,
          scaffold=scaffold)

  return model_fn
```

## 测试
修改后的代码正常运行，使用2块1070ti显卡，10w步耗时8小时，相较文1中12+小时有一定提升。