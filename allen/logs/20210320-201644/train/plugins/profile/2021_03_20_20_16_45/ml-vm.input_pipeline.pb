	�H�1@�H�1@!�H�1@	�Sx����?�Sx����?!�Sx����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�H�1@��y7�?A��V|C�0@Y Sh�?*H+��Z@)      =2F
Iterator::Model#I��B�?!�m�ݽJ@)hz��L��?1��r�4�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat:�8��?!NH�Eo8@)�GS=��?1�=���2@:Preprocessing2U
Iterator::Model::ParallelMapV2w��g�?!�S�Q�/@)w��g�?1�S�Q�/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�c�� w�?!yPk��/@)ӄ�'c|�?1 ���`&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorR~R���x?!�ʈnB�@)R~R���x?1�ʈnB�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��ZӼ�t?!�U	�@)��ZӼ�t?1�U	�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���|�r�?!@�{"BG@)��b�Ds?1Y����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Ɵ�lX�?!\2!87�1@)�0&��^?1����~�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�Sx����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��y7�?��y7�?!��y7�?      ��!       "      ��!       *      ��!       2	��V|C�0@��V|C�0@!��V|C�0@:      ��!       B      ��!       J	 Sh�? Sh�?! Sh�?R      ��!       Z	 Sh�? Sh�?! Sh�?JCPU_ONLYY�Sx����?b 