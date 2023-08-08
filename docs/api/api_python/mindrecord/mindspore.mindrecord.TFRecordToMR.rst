
.. py:class:: mindspore.mindrecord.TFRecordToMR(source, destination, feature_dict, bytes_fields=None)

    将TFRecord格式数据集转换为MindRecord格式数据集。

    参数：
        - **source** (str) - 待转换的TFRecord文件路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **feature_dict** (dict[str, FixedLenFeature]) - TFRecord的feature类别的字典，当前支持
          `FixedLenFeature <https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature>`_ 类型。
        - **bytes_fields** (list[str]，可选) - `feature_dict` 中的字节字段，可以为字节类型的图像字段。默认值：None，表示没有诸如图像的二进制字段。

    异常：
        - **ValueError** - 无效参数。
        - **Exception** - 找不到TensorFlow模块或其版本不正确。

    .. py:method:: run()

        执行从TFRecord格式数据集到MindRecord格式数据集的转换。

        返回：
            MSRStatus，SUCCESS或FAILED。

    .. py:method:: tfrecord_iterator()

        生成一个字典，其key是schema中的字段，value是数据。

        返回：
            dict，key与schema中字段名相同的数据字典。

    .. py:method:: tfrecord_iterator_oldversion()

        生成一个字典，其中key是schema中的字段，value是数据。该函数适用于早于2.1.0版本的TensorFlow。

        返回：
            dict，key与schema中字段名相同的数据字典。

    .. py:method:: transform()

        封装 :func:`mindspore.mindrecord.TFRecordToMR.run` 函数来保证异常时正常退出。

        返回：
            MSRStatus，SUCCESS或FAILED。
