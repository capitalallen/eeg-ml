�	��?�03@��?�03@!��?�03@	��9��?��9��?!��9��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��?�03@�辜��?Ar���	�2@Yc)��R�?*	����`@2F
Iterator::Modelf/�N[�?!�����D@)��#�G�?1#�.�1=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatÃf׽�?!�C=�l�9@)/5B?S��?1�q+��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ�v0b��?!�
��6+9@)��4�R�?1DxH��+3@:Preprocessing2U
Iterator::Model::ParallelMapV2<-?p�'�?!�=�k�u(@)<-?p�'�?1�=�k�u(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(��ȯ?!�J�u(�@)(��ȯ?1�J�u(�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���X�?!zf�JM@)��]M~?1î����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZI+���y?!�H��e�@)ZI+���y?1�H��e�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�� v�С?!~�k�m�:@)��ZDc?1�'{m��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��9��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�辜��?�辜��?!�辜��?      ��!       "      ��!       *      ��!       2	r���	�2@r���	�2@!r���	�2@:      ��!       B      ��!       J	c)��R�?c)��R�?!c)��R�?R      ��!       Z	c)��R�?c)��R�?!c)��R�?JCPU_ONLYY��9��?b Y      Y@q��ð�#@"�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 