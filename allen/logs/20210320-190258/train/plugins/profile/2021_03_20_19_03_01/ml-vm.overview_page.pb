�$	��V�I��?C�����?�F ^�/�?!;�� �~�?	aC����@Za ���@!aC����@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$;�� �~�?�s`9B�?A�J=B��?YLb�G�?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails�F ^�/�?;�?l�ф?Ab�o�Z?*	R���Q`@2F
Iterator::ModelڨN��?!������F@)c� ��^�?1������@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat������?!�����/:@)B�L���?122222F4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�0�q�	�?!AAAAA94@)l���?1



�.@:Preprocessing2U
Iterator::Model::ParallelMapV29
3��?!�(@)9
3��?1�(@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor}�r��?!UUUUU�@)}�r��?1UUUUU�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ֪]�?!ZZZZZ
K@)�RB��~?1(((((�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���|�ry?!�����@)���|�ry?1�����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��Ü��?!#####)6@)��T���d?1��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s5.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�X���@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	����vܤ?+q'�%$�?;�?l�ф?!�s`9B�?	!       "	!       *	!       2$	�!����?�j�4��?b�o�Z?!�J=B��?:	!       B	!       J	Lb�G�?����s�?!Lb�G�?R	!       Z	Lb�G�?����s�?!Lb�G�?JCPU_ONLYY�X���@b Y      Y@q�t�Ǡ�?"�
both�Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nomoderate"s5.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQ2"CPU: B 