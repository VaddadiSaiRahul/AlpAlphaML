��%
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements#
handle��element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.6.22v2.6.1-9-gc2363d6d0258�$
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

: *
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
lstm_18/lstm_cell_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_18/lstm_cell_36/kernel
�
/lstm_18/lstm_cell_36/kernel/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_36/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_18/lstm_cell_36/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*6
shared_name'%lstm_18/lstm_cell_36/recurrent_kernel
�
9lstm_18/lstm_cell_36/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_18/lstm_cell_36/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_18/lstm_cell_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_18/lstm_cell_36/bias
�
-lstm_18/lstm_cell_36/bias/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_36/bias*
_output_shapes	
:�*
dtype0
�
lstm_19/lstm_cell_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*,
shared_namelstm_19/lstm_cell_37/kernel
�
/lstm_19/lstm_cell_37/kernel/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_37/kernel*
_output_shapes
:	@�*
dtype0
�
%lstm_19/lstm_cell_37/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%lstm_19/lstm_cell_37/recurrent_kernel
�
9lstm_19/lstm_cell_37/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_19/lstm_cell_37/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
lstm_19/lstm_cell_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_19/lstm_cell_37/bias
�
-lstm_19/lstm_cell_37/bias/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_37/bias*
_output_shapes	
:�*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_18/lstm_cell_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_18/lstm_cell_36/kernel/m
�
6Adam/lstm_18/lstm_cell_36/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_18/lstm_cell_36/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_18/lstm_cell_36/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_18/lstm_cell_36/recurrent_kernel/m
�
@Adam/lstm_18/lstm_cell_36/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_18/lstm_cell_36/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_18/lstm_cell_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_18/lstm_cell_36/bias/m
�
4Adam/lstm_18/lstm_cell_36/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_18/lstm_cell_36/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_19/lstm_cell_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_19/lstm_cell_37/kernel/m
�
6Adam/lstm_19/lstm_cell_37/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_19/lstm_cell_37/kernel/m*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_19/lstm_cell_37/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_19/lstm_cell_37/recurrent_kernel/m
�
@Adam/lstm_19/lstm_cell_37/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_19/lstm_cell_37/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_19/lstm_cell_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_19/lstm_cell_37/bias/m
�
4Adam/lstm_19/lstm_cell_37/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_19/lstm_cell_37/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_18/lstm_cell_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_18/lstm_cell_36/kernel/v
�
6Adam/lstm_18/lstm_cell_36/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_18/lstm_cell_36/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_18/lstm_cell_36/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_18/lstm_cell_36/recurrent_kernel/v
�
@Adam/lstm_18/lstm_cell_36/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_18/lstm_cell_36/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_18/lstm_cell_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_18/lstm_cell_36/bias/v
�
4Adam/lstm_18/lstm_cell_36/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_18/lstm_cell_36/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_19/lstm_cell_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_19/lstm_cell_37/kernel/v
�
6Adam/lstm_19/lstm_cell_37/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_19/lstm_cell_37/kernel/v*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_19/lstm_cell_37/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_19/lstm_cell_37/recurrent_kernel/v
�
@Adam/lstm_19/lstm_cell_37/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_19/lstm_cell_37/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_19/lstm_cell_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_19/lstm_cell_37/bias/v
�
4Adam/lstm_19/lstm_cell_37/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_19/lstm_cell_37/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
�3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�2
value�2B�2 B�2
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem`ma&mb'mc(md)me*mf+mgvhvi&vj'vk(vl)vm*vn+vo
8
&0
'1
(2
)3
*4
+5
6
7
 
8
&0
'1
(2
)3
*4
+5
6
7
�
trainable_variables
regularization_losses
,layer_metrics

-layers
	variables
.metrics
/non_trainable_variables
0layer_regularization_losses
 
�
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
 

&0
'1
(2
 

&0
'1
(2
�
trainable_variables
regularization_losses
6layer_metrics

7layers
	variables
8layer_regularization_losses
9metrics
:non_trainable_variables

;states
�
<
state_size

)kernel
*recurrent_kernel
+bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
 

)0
*1
+2
 

)0
*1
+2
�
trainable_variables
regularization_losses
Alayer_metrics

Blayers
	variables
Clayer_regularization_losses
Dmetrics
Enon_trainable_variables

Fstates
 
 
 
�
trainable_variables
regularization_losses
Glayer_metrics

Hlayers
	variables
Imetrics
Jnon_trainable_variables
Klayer_regularization_losses
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
regularization_losses
Llayer_metrics

Mlayers
	variables
Nmetrics
Onon_trainable_variables
Player_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_18/lstm_cell_36/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_18/lstm_cell_36/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_18/lstm_cell_36/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_19/lstm_cell_37/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_19/lstm_cell_37/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_19/lstm_cell_37/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

Q0
 
 
 

&0
'1
(2
 

&0
'1
(2
�
2trainable_variables
3regularization_losses
Rlayer_metrics

Slayers
4	variables
Tmetrics
Unon_trainable_variables
Vlayer_regularization_losses
 

0
 
 
 
 
 

)0
*1
+2
 

)0
*1
+2
�
=trainable_variables
>regularization_losses
Wlayer_metrics

Xlayers
?	variables
Ymetrics
Znon_trainable_variables
[layer_regularization_losses
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	\total
	]count
^	variables
_	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

\0
]1

^	variables
}{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_18/lstm_cell_36/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_18/lstm_cell_36/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_18/lstm_cell_36/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_19/lstm_cell_37/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_19/lstm_cell_37/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_19/lstm_cell_37/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_18/lstm_cell_36/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_18/lstm_cell_36/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_18/lstm_cell_36/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_19/lstm_cell_37/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_19/lstm_cell_37/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_19/lstm_cell_37/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_18_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_18_inputlstm_18/lstm_cell_36/kernel%lstm_18/lstm_cell_36/recurrent_kernellstm_18/lstm_cell_36/biaslstm_19/lstm_cell_37/kernel%lstm_19/lstm_cell_37/recurrent_kernellstm_19/lstm_cell_37/biasdense_9/kerneldense_9/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_324171
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_18/lstm_cell_36/kernel/Read/ReadVariableOp9lstm_18/lstm_cell_36/recurrent_kernel/Read/ReadVariableOp-lstm_18/lstm_cell_36/bias/Read/ReadVariableOp/lstm_19/lstm_cell_37/kernel/Read/ReadVariableOp9lstm_19/lstm_cell_37/recurrent_kernel/Read/ReadVariableOp-lstm_19/lstm_cell_37/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp6Adam/lstm_18/lstm_cell_36/kernel/m/Read/ReadVariableOp@Adam/lstm_18/lstm_cell_36/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_18/lstm_cell_36/bias/m/Read/ReadVariableOp6Adam/lstm_19/lstm_cell_37/kernel/m/Read/ReadVariableOp@Adam/lstm_19/lstm_cell_37/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_19/lstm_cell_37/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp6Adam/lstm_18/lstm_cell_36/kernel/v/Read/ReadVariableOp@Adam/lstm_18/lstm_cell_36/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_18/lstm_cell_36/bias/v/Read/ReadVariableOp6Adam/lstm_19/lstm_cell_37/kernel/v/Read/ReadVariableOp@Adam/lstm_19/lstm_cell_37/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_19/lstm_cell_37/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_326484
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_18/lstm_cell_36/kernel%lstm_18/lstm_cell_36/recurrent_kernellstm_18/lstm_cell_36/biaslstm_19/lstm_cell_37/kernel%lstm_19/lstm_cell_37/recurrent_kernellstm_19/lstm_cell_37/biastotalcountAdam/dense_9/kernel/mAdam/dense_9/bias/m"Adam/lstm_18/lstm_cell_36/kernel/m,Adam/lstm_18/lstm_cell_36/recurrent_kernel/m Adam/lstm_18/lstm_cell_36/bias/m"Adam/lstm_19/lstm_cell_37/kernel/m,Adam/lstm_19/lstm_cell_37/recurrent_kernel/m Adam/lstm_19/lstm_cell_37/bias/mAdam/dense_9/kernel/vAdam/dense_9/bias/v"Adam/lstm_18/lstm_cell_36/kernel/v,Adam/lstm_18/lstm_cell_36/recurrent_kernel/v Adam/lstm_18/lstm_cell_36/bias/v"Adam/lstm_19/lstm_cell_37/kernel/v,Adam/lstm_19/lstm_cell_37/recurrent_kernel/v Adam/lstm_19/lstm_cell_37/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_326587��#
�

�
-__inference_sequential_9_layer_call_fn_324192

inputs
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3236092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�J
�

lstm_18_while_body_324585,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0:	�P
=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�K
<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_36_matmul_readvariableop_resource:	�N
;lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource:	@�I
:lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource:	���1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItem�
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�
!lstm_18/while/lstm_cell_36/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_18/while/lstm_cell_36/MatMul�
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
#lstm_18/while/lstm_cell_36/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_18/while/lstm_cell_36/MatMul_1�
lstm_18/while/lstm_cell_36/addAddV2+lstm_18/while/lstm_cell_36/MatMul:product:0-lstm_18/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_18/while/lstm_cell_36/add�
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�
"lstm_18/while/lstm_cell_36/BiasAddBiasAdd"lstm_18/while/lstm_cell_36/add:z:09lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_18/while/lstm_cell_36/BiasAdd�
*lstm_18/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_36/split/split_dim�
 lstm_18/while/lstm_cell_36/splitSplit3lstm_18/while/lstm_cell_36/split/split_dim:output:0+lstm_18/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_18/while/lstm_cell_36/split�
"lstm_18/while/lstm_cell_36/SigmoidSigmoid)lstm_18/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_18/while/lstm_cell_36/Sigmoid�
$lstm_18/while/lstm_cell_36/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_18/while/lstm_cell_36/Sigmoid_1�
lstm_18/while/lstm_cell_36/mulMul(lstm_18/while/lstm_cell_36/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_18/while/lstm_cell_36/mul�
lstm_18/while/lstm_cell_36/ReluRelu)lstm_18/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_18/while/lstm_cell_36/Relu�
 lstm_18/while/lstm_cell_36/mul_1Mul&lstm_18/while/lstm_cell_36/Sigmoid:y:0-lstm_18/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/mul_1�
 lstm_18/while/lstm_cell_36/add_1AddV2"lstm_18/while/lstm_cell_36/mul:z:0$lstm_18/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/add_1�
$lstm_18/while/lstm_cell_36/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_18/while/lstm_cell_36/Sigmoid_2�
!lstm_18/while/lstm_cell_36/Relu_1Relu$lstm_18/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_18/while/lstm_cell_36/Relu_1�
 lstm_18/while/lstm_cell_36/mul_2Mul(lstm_18/while/lstm_cell_36/Sigmoid_2:y:0/lstm_18/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/mul_2�
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/y�
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/y�
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1�
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity�
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1�
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2�
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3�
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_36/mul_2:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_18/while/Identity_4�
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_36/add_1:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_18/while/Identity_5�
lstm_18/while/NoOpNoOp2^lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_18/while/NoOp"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_36_matmul_readvariableop_resource;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0"�
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_37_layer_call_fn_326304

inputs
states_0
states_1
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3228532
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
while_cond_324940
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_324940___redundant_placeholder04
0while_while_cond_324940___redundant_placeholder14
0while_while_cond_324940___redundant_placeholder24
0while_while_cond_324940___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_325394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�[
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_325478

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325394*
condR
while_cond_325393*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
while_body_322091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_36_322115_0:	�.
while_lstm_cell_36_322117_0:	@�*
while_lstm_cell_36_322119_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_36_322115:	�,
while_lstm_cell_36_322117:	@�(
while_lstm_cell_36_322119:	���*while/lstm_cell_36/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_36_322115_0while_lstm_cell_36_322117_0while_lstm_cell_36_322119_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3220772,
*while/lstm_cell_36/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_36/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_36_322115while_lstm_cell_36_322115_0"8
while_lstm_cell_36_322117while_lstm_cell_36_322117_0"8
while_lstm_cell_36_322119while_lstm_cell_36_322119_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_36/StatefulPartitionedCall*while/lstm_cell_36/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_322300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322300___redundant_placeholder04
0while_while_cond_322300___redundant_placeholder14
0while_while_cond_322300___redundant_placeholder24
0while_while_cond_322300___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_322720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322720___redundant_placeholder04
0while_while_cond_322720___redundant_placeholder14
0while_while_cond_322720___redundant_placeholder24
0while_while_cond_322720___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324142
lstm_18_input!
lstm_18_324121:	�!
lstm_18_324123:	@�
lstm_18_324125:	�!
lstm_19_324128:	@�!
lstm_19_324130:	 �
lstm_19_324132:	� 
dense_9_324136: 
dense_9_324138:
identity��dense_9/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�lstm_18/StatefulPartitionedCall�lstm_19/StatefulPartitionedCall�
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_324121lstm_18_324123lstm_18_324125*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3239982!
lstm_18/StatefulPartitionedCall�
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0lstm_19_324128lstm_19_324130lstm_19_324132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3238252!
lstm_19/StatefulPartitionedCall�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3236582#
!dropout_9/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_9_324136dense_9_324138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3236022!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_9/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�?
�
while_body_325243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�]
�
&sequential_9_lstm_19_while_body_321911F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3E
Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0�
}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_9_lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0:	@�]
Jsequential_9_lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �X
Isequential_9_lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0:	�'
#sequential_9_lstm_19_while_identity)
%sequential_9_lstm_19_while_identity_1)
%sequential_9_lstm_19_while_identity_2)
%sequential_9_lstm_19_while_identity_3)
%sequential_9_lstm_19_while_identity_4)
%sequential_9_lstm_19_while_identity_5C
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensorY
Fsequential_9_lstm_19_while_lstm_cell_37_matmul_readvariableop_resource:	@�[
Hsequential_9_lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource:	 �V
Gsequential_9_lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource:	���>sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�=sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�?sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2N
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape�
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_19_while_placeholderUsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02@
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem�
=sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02?
=sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�
.sequential_9/lstm_19/while/lstm_cell_37/MatMulMatMulEsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������20
.sequential_9/lstm_19/while/lstm_cell_37/MatMul�
?sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02A
?sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
0sequential_9/lstm_19/while/lstm_cell_37/MatMul_1MatMul(sequential_9_lstm_19_while_placeholder_2Gsequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_9/lstm_19/while/lstm_cell_37/MatMul_1�
+sequential_9/lstm_19/while/lstm_cell_37/addAddV28sequential_9/lstm_19/while/lstm_cell_37/MatMul:product:0:sequential_9/lstm_19/while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2-
+sequential_9/lstm_19/while/lstm_cell_37/add�
>sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02@
>sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�
/sequential_9/lstm_19/while/lstm_cell_37/BiasAddBiasAdd/sequential_9/lstm_19/while/lstm_cell_37/add:z:0Fsequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_9/lstm_19/while/lstm_cell_37/BiasAdd�
7sequential_9/lstm_19/while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_19/while/lstm_cell_37/split/split_dim�
-sequential_9/lstm_19/while/lstm_cell_37/splitSplit@sequential_9/lstm_19/while/lstm_cell_37/split/split_dim:output:08sequential_9/lstm_19/while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2/
-sequential_9/lstm_19/while/lstm_cell_37/split�
/sequential_9/lstm_19/while/lstm_cell_37/SigmoidSigmoid6sequential_9/lstm_19/while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 21
/sequential_9/lstm_19/while/lstm_cell_37/Sigmoid�
1sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_1Sigmoid6sequential_9/lstm_19/while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 23
1sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_1�
+sequential_9/lstm_19/while/lstm_cell_37/mulMul5sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_1:y:0(sequential_9_lstm_19_while_placeholder_3*
T0*'
_output_shapes
:��������� 2-
+sequential_9/lstm_19/while/lstm_cell_37/mul�
,sequential_9/lstm_19/while/lstm_cell_37/ReluRelu6sequential_9/lstm_19/while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2.
,sequential_9/lstm_19/while/lstm_cell_37/Relu�
-sequential_9/lstm_19/while/lstm_cell_37/mul_1Mul3sequential_9/lstm_19/while/lstm_cell_37/Sigmoid:y:0:sequential_9/lstm_19/while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2/
-sequential_9/lstm_19/while/lstm_cell_37/mul_1�
-sequential_9/lstm_19/while/lstm_cell_37/add_1AddV2/sequential_9/lstm_19/while/lstm_cell_37/mul:z:01sequential_9/lstm_19/while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2/
-sequential_9/lstm_19/while/lstm_cell_37/add_1�
1sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_2Sigmoid6sequential_9/lstm_19/while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 23
1sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_2�
.sequential_9/lstm_19/while/lstm_cell_37/Relu_1Relu1sequential_9/lstm_19/while/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 20
.sequential_9/lstm_19/while/lstm_cell_37/Relu_1�
-sequential_9/lstm_19/while/lstm_cell_37/mul_2Mul5sequential_9/lstm_19/while/lstm_cell_37/Sigmoid_2:y:0<sequential_9/lstm_19/while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2/
-sequential_9/lstm_19/while/lstm_cell_37/mul_2�
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_19_while_placeholder_1&sequential_9_lstm_19_while_placeholder1sequential_9/lstm_19/while/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem�
 sequential_9/lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_19/while/add/y�
sequential_9/lstm_19/while/addAddV2&sequential_9_lstm_19_while_placeholder)sequential_9/lstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/while/add�
"sequential_9/lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_19/while/add_1/y�
 sequential_9/lstm_19/while/add_1AddV2Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counter+sequential_9/lstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/while/add_1�
#sequential_9/lstm_19/while/IdentityIdentity$sequential_9/lstm_19/while/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identity�
%sequential_9/lstm_19/while/Identity_1IdentityHsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_1�
%sequential_9/lstm_19/while/Identity_2Identity"sequential_9/lstm_19/while/add:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_2�
%sequential_9/lstm_19/while/Identity_3IdentityOsequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_3�
%sequential_9/lstm_19/while/Identity_4Identity1sequential_9/lstm_19/while/lstm_cell_37/mul_2:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2'
%sequential_9/lstm_19/while/Identity_4�
%sequential_9/lstm_19/while/Identity_5Identity1sequential_9/lstm_19/while/lstm_cell_37/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2'
%sequential_9/lstm_19/while/Identity_5�
sequential_9/lstm_19/while/NoOpNoOp?^sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp>^sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp@^sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_9/lstm_19/while/NoOp"S
#sequential_9_lstm_19_while_identity,sequential_9/lstm_19/while/Identity:output:0"W
%sequential_9_lstm_19_while_identity_1.sequential_9/lstm_19/while/Identity_1:output:0"W
%sequential_9_lstm_19_while_identity_2.sequential_9/lstm_19/while/Identity_2:output:0"W
%sequential_9_lstm_19_while_identity_3.sequential_9/lstm_19/while/Identity_3:output:0"W
%sequential_9_lstm_19_while_identity_4.sequential_9/lstm_19/while/Identity_4:output:0"W
%sequential_9_lstm_19_while_identity_5.sequential_9/lstm_19/while/Identity_5:output:0"�
Gsequential_9_lstm_19_while_lstm_cell_37_biasadd_readvariableop_resourceIsequential_9_lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0"�
Hsequential_9_lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resourceJsequential_9_lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0"�
Fsequential_9_lstm_19_while_lstm_cell_37_matmul_readvariableop_resourceHsequential_9_lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0"�
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0"�
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2�
>sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp>sequential_9/lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp=sequential_9/lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp2�
?sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp?sequential_9/lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
��
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324518

inputsF
3lstm_18_lstm_cell_36_matmul_readvariableop_resource:	�H
5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource:	@�C
4lstm_18_lstm_cell_36_biasadd_readvariableop_resource:	�F
3lstm_19_lstm_cell_37_matmul_readvariableop_resource:	@�H
5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource:	 �C
4lstm_19_lstm_cell_37_biasadd_readvariableop_resource:	�8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity��dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�*lstm_18/lstm_cell_36/MatMul/ReadVariableOp�,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�lstm_18/while�+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�*lstm_19/lstm_cell_37/MatMul/ReadVariableOp�,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�lstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape�
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack�
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1�
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2�
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros/mul/y�
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_18/zeros/Less/y�
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros/packed/1�
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/Const�
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros_1/mul/y�
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_18/zeros_1/Less/y�
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros_1/packed/1�
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/Const�
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/zeros_1�
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm�
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1�
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack�
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1�
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2�
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1�
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_18/TensorArrayV2/element_shape�
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2�
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor�
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack�
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1�
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2�
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_18/strided_slice_2�
*lstm_18/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_18/lstm_cell_36/MatMul/ReadVariableOp�
lstm_18/lstm_cell_36/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/MatMul�
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_18/lstm_cell_36/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/MatMul_1�
lstm_18/lstm_cell_36/addAddV2%lstm_18/lstm_cell_36/MatMul:product:0'lstm_18/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/add�
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_18/lstm_cell_36/BiasAddBiasAddlstm_18/lstm_cell_36/add:z:03lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/BiasAdd�
$lstm_18/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_36/split/split_dim�
lstm_18/lstm_cell_36/splitSplit-lstm_18/lstm_cell_36/split/split_dim:output:0%lstm_18/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_18/lstm_cell_36/split�
lstm_18/lstm_cell_36/SigmoidSigmoid#lstm_18/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Sigmoid�
lstm_18/lstm_cell_36/Sigmoid_1Sigmoid#lstm_18/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_18/lstm_cell_36/Sigmoid_1�
lstm_18/lstm_cell_36/mulMul"lstm_18/lstm_cell_36/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul�
lstm_18/lstm_cell_36/ReluRelu#lstm_18/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Relu�
lstm_18/lstm_cell_36/mul_1Mul lstm_18/lstm_cell_36/Sigmoid:y:0'lstm_18/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul_1�
lstm_18/lstm_cell_36/add_1AddV2lstm_18/lstm_cell_36/mul:z:0lstm_18/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/add_1�
lstm_18/lstm_cell_36/Sigmoid_2Sigmoid#lstm_18/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_18/lstm_cell_36/Sigmoid_2�
lstm_18/lstm_cell_36/Relu_1Relulstm_18/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Relu_1�
lstm_18/lstm_cell_36/mul_2Mul"lstm_18/lstm_cell_36/Sigmoid_2:y:0)lstm_18/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul_2�
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_18/TensorArrayV2_1/element_shape�
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/time�
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counter�
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_36_matmul_readvariableop_resource5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource4lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_18_while_body_324280*%
condR
lstm_18_while_cond_324279*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_18/while�
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack�
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_18/strided_slice_3/stack�
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1�
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2�
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_18/strided_slice_3�
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/perm�
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimee
lstm_19/ShapeShapelstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_19/Shape�
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack�
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1�
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2�
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicel
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros/mul/y�
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_19/zeros/Less/y�
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lessr
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros/packed/1�
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/Const�
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/zerosp
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros_1/mul/y�
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_19/zeros_1/Less/y�
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessv
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros_1/packed/1�
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/Const�
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/zeros_1�
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/perm�
lstm_19/transpose	Transposelstm_18/transpose_1:y:0lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1�
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack�
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1�
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2�
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1�
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_19/TensorArrayV2/element_shape�
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2�
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor�
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack�
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1�
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2�
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_19/strided_slice_2�
*lstm_19/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_19/lstm_cell_37/MatMul/ReadVariableOp�
lstm_19/lstm_cell_37/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/MatMul�
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_19/lstm_cell_37/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/MatMul_1�
lstm_19/lstm_cell_37/addAddV2%lstm_19/lstm_cell_37/MatMul:product:0'lstm_19/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/add�
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_19/lstm_cell_37/BiasAddBiasAddlstm_19/lstm_cell_37/add:z:03lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/BiasAdd�
$lstm_19/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_37/split/split_dim�
lstm_19/lstm_cell_37/splitSplit-lstm_19/lstm_cell_37/split/split_dim:output:0%lstm_19/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_19/lstm_cell_37/split�
lstm_19/lstm_cell_37/SigmoidSigmoid#lstm_19/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Sigmoid�
lstm_19/lstm_cell_37/Sigmoid_1Sigmoid#lstm_19/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_19/lstm_cell_37/Sigmoid_1�
lstm_19/lstm_cell_37/mulMul"lstm_19/lstm_cell_37/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul�
lstm_19/lstm_cell_37/ReluRelu#lstm_19/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Relu�
lstm_19/lstm_cell_37/mul_1Mul lstm_19/lstm_cell_37/Sigmoid:y:0'lstm_19/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul_1�
lstm_19/lstm_cell_37/add_1AddV2lstm_19/lstm_cell_37/mul:z:0lstm_19/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/add_1�
lstm_19/lstm_cell_37/Sigmoid_2Sigmoid#lstm_19/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_19/lstm_cell_37/Sigmoid_2�
lstm_19/lstm_cell_37/Relu_1Relulstm_19/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Relu_1�
lstm_19/lstm_cell_37/mul_2Mul"lstm_19/lstm_cell_37/Sigmoid_2:y:0)lstm_19/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul_2�
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_19/TensorArrayV2_1/element_shape�
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/time�
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counter�
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_37_matmul_readvariableop_resource5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource4lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_19_while_body_324427*%
condR
lstm_19_while_cond_324426*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_19/while�
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack�
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_19/strided_slice_3/stack�
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1�
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2�
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_19/strided_slice_3�
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/perm�
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtime�
dropout_9/IdentityIdentity lstm_19/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2
dropout_9/Identity�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMuldropout_9/Identity:output:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp,^lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_36/MatMul/ReadVariableOp-^lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_37/MatMul/ReadVariableOp-^lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2Z
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_36/MatMul/ReadVariableOp*lstm_18/lstm_cell_36/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_37/MatMul/ReadVariableOp*lstm_19/lstm_cell_37/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_323602

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�\
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_325673
inputs_0>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325589*
condR
while_cond_325588*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_325393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325393___redundant_placeholder04
0while_while_cond_325393___redundant_placeholder14
0while_while_cond_325393___redundant_placeholder24
0while_while_cond_325393___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_323335
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�

�
-__inference_sequential_9_layer_call_fn_324094
lstm_18_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3240542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�
F
*__inference_dropout_9_layer_call_fn_326131

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3235902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�[
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_326126

inputs>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_326042*
condR
while_cond_326041*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_323492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323492___redundant_placeholder04
0while_while_cond_323492___redundant_placeholder14
0while_while_cond_323492___redundant_placeholder24
0while_while_cond_323492___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�

�
lstm_19_while_cond_324731,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1D
@lstm_19_while_lstm_19_while_cond_324731___redundant_placeholder0D
@lstm_19_while_lstm_19_while_cond_324731___redundant_placeholder1D
@lstm_19_while_lstm_19_while_cond_324731___redundant_placeholder2D
@lstm_19_while_lstm_19_while_cond_324731___redundant_placeholder3
lstm_19_while_identity
�
lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�J
�

lstm_19_while_body_324732,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0:	@�P
=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �K
<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorL
9lstm_19_while_lstm_cell_37_matmul_readvariableop_resource:	@�N
;lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource:	 �I
:lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource:	���1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItem�
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�
!lstm_19/while/lstm_cell_37/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_19/while/lstm_cell_37/MatMul�
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
#lstm_19/while/lstm_cell_37/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_19/while/lstm_cell_37/MatMul_1�
lstm_19/while/lstm_cell_37/addAddV2+lstm_19/while/lstm_cell_37/MatMul:product:0-lstm_19/while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_19/while/lstm_cell_37/add�
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�
"lstm_19/while/lstm_cell_37/BiasAddBiasAdd"lstm_19/while/lstm_cell_37/add:z:09lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_19/while/lstm_cell_37/BiasAdd�
*lstm_19/while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_37/split/split_dim�
 lstm_19/while/lstm_cell_37/splitSplit3lstm_19/while/lstm_cell_37/split/split_dim:output:0+lstm_19/while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_19/while/lstm_cell_37/split�
"lstm_19/while/lstm_cell_37/SigmoidSigmoid)lstm_19/while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_19/while/lstm_cell_37/Sigmoid�
$lstm_19/while/lstm_cell_37/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_19/while/lstm_cell_37/Sigmoid_1�
lstm_19/while/lstm_cell_37/mulMul(lstm_19/while/lstm_cell_37/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_19/while/lstm_cell_37/mul�
lstm_19/while/lstm_cell_37/ReluRelu)lstm_19/while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_19/while/lstm_cell_37/Relu�
 lstm_19/while/lstm_cell_37/mul_1Mul&lstm_19/while/lstm_cell_37/Sigmoid:y:0-lstm_19/while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/mul_1�
 lstm_19/while/lstm_cell_37/add_1AddV2"lstm_19/while/lstm_cell_37/mul:z:0$lstm_19/while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/add_1�
$lstm_19/while/lstm_cell_37/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_19/while/lstm_cell_37/Sigmoid_2�
!lstm_19/while/lstm_cell_37/Relu_1Relu$lstm_19/while/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_19/while/lstm_cell_37/Relu_1�
 lstm_19/while/lstm_cell_37/mul_2Mul(lstm_19/while/lstm_cell_37/Sigmoid_2:y:0/lstm_19/while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/mul_2�
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/y�
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/y�
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1�
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity�
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1�
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2�
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3�
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_37/mul_2:z:0^lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_19/while/Identity_4�
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_37/add_1:z:0^lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_19/while/Identity_5�
lstm_19/while/NoOpNoOp2^lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_19/while/NoOp"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_37_matmul_readvariableop_resource;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0"�
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
��
�	
!__inference__wrapped_model_322002
lstm_18_inputS
@sequential_9_lstm_18_lstm_cell_36_matmul_readvariableop_resource:	�U
Bsequential_9_lstm_18_lstm_cell_36_matmul_1_readvariableop_resource:	@�P
Asequential_9_lstm_18_lstm_cell_36_biasadd_readvariableop_resource:	�S
@sequential_9_lstm_19_lstm_cell_37_matmul_readvariableop_resource:	@�U
Bsequential_9_lstm_19_lstm_cell_37_matmul_1_readvariableop_resource:	 �P
Asequential_9_lstm_19_lstm_cell_37_biasadd_readvariableop_resource:	�E
3sequential_9_dense_9_matmul_readvariableop_resource: B
4sequential_9_dense_9_biasadd_readvariableop_resource:
identity��+sequential_9/dense_9/BiasAdd/ReadVariableOp�*sequential_9/dense_9/MatMul/ReadVariableOp�8sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�7sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp�9sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�sequential_9/lstm_18/while�8sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�7sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp�9sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�sequential_9/lstm_19/whileu
sequential_9/lstm_18/ShapeShapelstm_18_input*
T0*
_output_shapes
:2
sequential_9/lstm_18/Shape�
(sequential_9/lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_18/strided_slice/stack�
*sequential_9/lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_1�
*sequential_9/lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_2�
"sequential_9/lstm_18/strided_sliceStridedSlice#sequential_9/lstm_18/Shape:output:01sequential_9/lstm_18/strided_slice/stack:output:03sequential_9/lstm_18/strided_slice/stack_1:output:03sequential_9/lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_18/strided_slice�
 sequential_9/lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_9/lstm_18/zeros/mul/y�
sequential_9/lstm_18/zeros/mulMul+sequential_9/lstm_18/strided_slice:output:0)sequential_9/lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/zeros/mul�
!sequential_9/lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_9/lstm_18/zeros/Less/y�
sequential_9/lstm_18/zeros/LessLess"sequential_9/lstm_18/zeros/mul:z:0*sequential_9/lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/zeros/Less�
#sequential_9/lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_9/lstm_18/zeros/packed/1�
!sequential_9/lstm_18/zeros/packedPack+sequential_9/lstm_18/strided_slice:output:0,sequential_9/lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_18/zeros/packed�
 sequential_9/lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_18/zeros/Const�
sequential_9/lstm_18/zerosFill*sequential_9/lstm_18/zeros/packed:output:0)sequential_9/lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_9/lstm_18/zeros�
"sequential_9/lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_9/lstm_18/zeros_1/mul/y�
 sequential_9/lstm_18/zeros_1/mulMul+sequential_9/lstm_18/strided_slice:output:0+sequential_9/lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/zeros_1/mul�
#sequential_9/lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2%
#sequential_9/lstm_18/zeros_1/Less/y�
!sequential_9/lstm_18/zeros_1/LessLess$sequential_9/lstm_18/zeros_1/mul:z:0,sequential_9/lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_18/zeros_1/Less�
%sequential_9/lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_9/lstm_18/zeros_1/packed/1�
#sequential_9/lstm_18/zeros_1/packedPack+sequential_9/lstm_18/strided_slice:output:0.sequential_9/lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_18/zeros_1/packed�
"sequential_9/lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_18/zeros_1/Const�
sequential_9/lstm_18/zeros_1Fill,sequential_9/lstm_18/zeros_1/packed:output:0+sequential_9/lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_9/lstm_18/zeros_1�
#sequential_9/lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_18/transpose/perm�
sequential_9/lstm_18/transpose	Transposelstm_18_input,sequential_9/lstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:���������2 
sequential_9/lstm_18/transpose�
sequential_9/lstm_18/Shape_1Shape"sequential_9/lstm_18/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_18/Shape_1�
*sequential_9/lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_1/stack�
,sequential_9/lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_1�
,sequential_9/lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_2�
$sequential_9/lstm_18/strided_slice_1StridedSlice%sequential_9/lstm_18/Shape_1:output:03sequential_9/lstm_18/strided_slice_1/stack:output:05sequential_9/lstm_18/strided_slice_1/stack_1:output:05sequential_9/lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_1�
0sequential_9/lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_9/lstm_18/TensorArrayV2/element_shape�
"sequential_9/lstm_18/TensorArrayV2TensorListReserve9sequential_9/lstm_18/TensorArrayV2/element_shape:output:0-sequential_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_18/TensorArrayV2�
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_18/transpose:y:0Ssequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor�
*sequential_9/lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_2/stack�
,sequential_9/lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_1�
,sequential_9/lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_2�
$sequential_9/lstm_18/strided_slice_2StridedSlice"sequential_9/lstm_18/transpose:y:03sequential_9/lstm_18/strided_slice_2/stack:output:05sequential_9/lstm_18/strided_slice_2/stack_1:output:05sequential_9/lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_2�
7sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_18_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp�
(sequential_9/lstm_18/lstm_cell_36/MatMulMatMul-sequential_9/lstm_18/strided_slice_2:output:0?sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(sequential_9/lstm_18/lstm_cell_36/MatMul�
9sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_18_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02;
9sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�
*sequential_9/lstm_18/lstm_cell_36/MatMul_1MatMul#sequential_9/lstm_18/zeros:output:0Asequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_9/lstm_18/lstm_cell_36/MatMul_1�
%sequential_9/lstm_18/lstm_cell_36/addAddV22sequential_9/lstm_18/lstm_cell_36/MatMul:product:04sequential_9/lstm_18/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%sequential_9/lstm_18/lstm_cell_36/add�
8sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�
)sequential_9/lstm_18/lstm_cell_36/BiasAddBiasAdd)sequential_9/lstm_18/lstm_cell_36/add:z:0@sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_9/lstm_18/lstm_cell_36/BiasAdd�
1sequential_9/lstm_18/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_18/lstm_cell_36/split/split_dim�
'sequential_9/lstm_18/lstm_cell_36/splitSplit:sequential_9/lstm_18/lstm_cell_36/split/split_dim:output:02sequential_9/lstm_18/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2)
'sequential_9/lstm_18/lstm_cell_36/split�
)sequential_9/lstm_18/lstm_cell_36/SigmoidSigmoid0sequential_9/lstm_18/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2+
)sequential_9/lstm_18/lstm_cell_36/Sigmoid�
+sequential_9/lstm_18/lstm_cell_36/Sigmoid_1Sigmoid0sequential_9/lstm_18/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2-
+sequential_9/lstm_18/lstm_cell_36/Sigmoid_1�
%sequential_9/lstm_18/lstm_cell_36/mulMul/sequential_9/lstm_18/lstm_cell_36/Sigmoid_1:y:0%sequential_9/lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:���������@2'
%sequential_9/lstm_18/lstm_cell_36/mul�
&sequential_9/lstm_18/lstm_cell_36/ReluRelu0sequential_9/lstm_18/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2(
&sequential_9/lstm_18/lstm_cell_36/Relu�
'sequential_9/lstm_18/lstm_cell_36/mul_1Mul-sequential_9/lstm_18/lstm_cell_36/Sigmoid:y:04sequential_9/lstm_18/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2)
'sequential_9/lstm_18/lstm_cell_36/mul_1�
'sequential_9/lstm_18/lstm_cell_36/add_1AddV2)sequential_9/lstm_18/lstm_cell_36/mul:z:0+sequential_9/lstm_18/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2)
'sequential_9/lstm_18/lstm_cell_36/add_1�
+sequential_9/lstm_18/lstm_cell_36/Sigmoid_2Sigmoid0sequential_9/lstm_18/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2-
+sequential_9/lstm_18/lstm_cell_36/Sigmoid_2�
(sequential_9/lstm_18/lstm_cell_36/Relu_1Relu+sequential_9/lstm_18/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2*
(sequential_9/lstm_18/lstm_cell_36/Relu_1�
'sequential_9/lstm_18/lstm_cell_36/mul_2Mul/sequential_9/lstm_18/lstm_cell_36/Sigmoid_2:y:06sequential_9/lstm_18/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2)
'sequential_9/lstm_18/lstm_cell_36/mul_2�
2sequential_9/lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   24
2sequential_9/lstm_18/TensorArrayV2_1/element_shape�
$sequential_9/lstm_18/TensorArrayV2_1TensorListReserve;sequential_9/lstm_18/TensorArrayV2_1/element_shape:output:0-sequential_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_9/lstm_18/TensorArrayV2_1x
sequential_9/lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_9/lstm_18/time�
-sequential_9/lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_9/lstm_18/while/maximum_iterations�
'sequential_9/lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_18/while/loop_counter�
sequential_9/lstm_18/whileWhile0sequential_9/lstm_18/while/loop_counter:output:06sequential_9/lstm_18/while/maximum_iterations:output:0"sequential_9/lstm_18/time:output:0-sequential_9/lstm_18/TensorArrayV2_1:handle:0#sequential_9/lstm_18/zeros:output:0%sequential_9/lstm_18/zeros_1:output:0-sequential_9/lstm_18/strided_slice_1:output:0Lsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_18_lstm_cell_36_matmul_readvariableop_resourceBsequential_9_lstm_18_lstm_cell_36_matmul_1_readvariableop_resourceAsequential_9_lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_9_lstm_18_while_body_321764*2
cond*R(
&sequential_9_lstm_18_while_cond_321763*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
sequential_9/lstm_18/while�
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2G
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_18/while:output:3Nsequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype029
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack�
*sequential_9/lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_9/lstm_18/strided_slice_3/stack�
,sequential_9/lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_18/strided_slice_3/stack_1�
,sequential_9/lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_3/stack_2�
$sequential_9/lstm_18/strided_slice_3StridedSlice@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_18/strided_slice_3/stack:output:05sequential_9/lstm_18/strided_slice_3/stack_1:output:05sequential_9/lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_3�
%sequential_9/lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_18/transpose_1/perm�
 sequential_9/lstm_18/transpose_1	Transpose@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2"
 sequential_9/lstm_18/transpose_1�
sequential_9/lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_18/runtime�
sequential_9/lstm_19/ShapeShape$sequential_9/lstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/Shape�
(sequential_9/lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_19/strided_slice/stack�
*sequential_9/lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_1�
*sequential_9/lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_2�
"sequential_9/lstm_19/strided_sliceStridedSlice#sequential_9/lstm_19/Shape:output:01sequential_9/lstm_19/strided_slice/stack:output:03sequential_9/lstm_19/strided_slice/stack_1:output:03sequential_9/lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_19/strided_slice�
 sequential_9/lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_9/lstm_19/zeros/mul/y�
sequential_9/lstm_19/zeros/mulMul+sequential_9/lstm_19/strided_slice:output:0)sequential_9/lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/zeros/mul�
!sequential_9/lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_9/lstm_19/zeros/Less/y�
sequential_9/lstm_19/zeros/LessLess"sequential_9/lstm_19/zeros/mul:z:0*sequential_9/lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/zeros/Less�
#sequential_9/lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_9/lstm_19/zeros/packed/1�
!sequential_9/lstm_19/zeros/packedPack+sequential_9/lstm_19/strided_slice:output:0,sequential_9/lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_19/zeros/packed�
 sequential_9/lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_19/zeros/Const�
sequential_9/lstm_19/zerosFill*sequential_9/lstm_19/zeros/packed:output:0)sequential_9/lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_9/lstm_19/zeros�
"sequential_9/lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_9/lstm_19/zeros_1/mul/y�
 sequential_9/lstm_19/zeros_1/mulMul+sequential_9/lstm_19/strided_slice:output:0+sequential_9/lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/zeros_1/mul�
#sequential_9/lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2%
#sequential_9/lstm_19/zeros_1/Less/y�
!sequential_9/lstm_19/zeros_1/LessLess$sequential_9/lstm_19/zeros_1/mul:z:0,sequential_9/lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_19/zeros_1/Less�
%sequential_9/lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_9/lstm_19/zeros_1/packed/1�
#sequential_9/lstm_19/zeros_1/packedPack+sequential_9/lstm_19/strided_slice:output:0.sequential_9/lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_19/zeros_1/packed�
"sequential_9/lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_19/zeros_1/Const�
sequential_9/lstm_19/zeros_1Fill,sequential_9/lstm_19/zeros_1/packed:output:0+sequential_9/lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_9/lstm_19/zeros_1�
#sequential_9/lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_19/transpose/perm�
sequential_9/lstm_19/transpose	Transpose$sequential_9/lstm_18/transpose_1:y:0,sequential_9/lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2 
sequential_9/lstm_19/transpose�
sequential_9/lstm_19/Shape_1Shape"sequential_9/lstm_19/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/Shape_1�
*sequential_9/lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_1/stack�
,sequential_9/lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_1�
,sequential_9/lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_2�
$sequential_9/lstm_19/strided_slice_1StridedSlice%sequential_9/lstm_19/Shape_1:output:03sequential_9/lstm_19/strided_slice_1/stack:output:05sequential_9/lstm_19/strided_slice_1/stack_1:output:05sequential_9/lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_1�
0sequential_9/lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_9/lstm_19/TensorArrayV2/element_shape�
"sequential_9/lstm_19/TensorArrayV2TensorListReserve9sequential_9/lstm_19/TensorArrayV2/element_shape:output:0-sequential_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_19/TensorArrayV2�
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2L
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_19/transpose:y:0Ssequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor�
*sequential_9/lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_2/stack�
,sequential_9/lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_1�
,sequential_9/lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_2�
$sequential_9/lstm_19/strided_slice_2StridedSlice"sequential_9/lstm_19/transpose:y:03sequential_9/lstm_19/strided_slice_2/stack:output:05sequential_9/lstm_19/strided_slice_2/stack_1:output:05sequential_9/lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_2�
7sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_19_lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype029
7sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp�
(sequential_9/lstm_19/lstm_cell_37/MatMulMatMul-sequential_9/lstm_19/strided_slice_2:output:0?sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(sequential_9/lstm_19/lstm_cell_37/MatMul�
9sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_19_lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02;
9sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�
*sequential_9/lstm_19/lstm_cell_37/MatMul_1MatMul#sequential_9/lstm_19/zeros:output:0Asequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_9/lstm_19/lstm_cell_37/MatMul_1�
%sequential_9/lstm_19/lstm_cell_37/addAddV22sequential_9/lstm_19/lstm_cell_37/MatMul:product:04sequential_9/lstm_19/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%sequential_9/lstm_19/lstm_cell_37/add�
8sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�
)sequential_9/lstm_19/lstm_cell_37/BiasAddBiasAdd)sequential_9/lstm_19/lstm_cell_37/add:z:0@sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_9/lstm_19/lstm_cell_37/BiasAdd�
1sequential_9/lstm_19/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_19/lstm_cell_37/split/split_dim�
'sequential_9/lstm_19/lstm_cell_37/splitSplit:sequential_9/lstm_19/lstm_cell_37/split/split_dim:output:02sequential_9/lstm_19/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2)
'sequential_9/lstm_19/lstm_cell_37/split�
)sequential_9/lstm_19/lstm_cell_37/SigmoidSigmoid0sequential_9/lstm_19/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2+
)sequential_9/lstm_19/lstm_cell_37/Sigmoid�
+sequential_9/lstm_19/lstm_cell_37/Sigmoid_1Sigmoid0sequential_9/lstm_19/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2-
+sequential_9/lstm_19/lstm_cell_37/Sigmoid_1�
%sequential_9/lstm_19/lstm_cell_37/mulMul/sequential_9/lstm_19/lstm_cell_37/Sigmoid_1:y:0%sequential_9/lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2'
%sequential_9/lstm_19/lstm_cell_37/mul�
&sequential_9/lstm_19/lstm_cell_37/ReluRelu0sequential_9/lstm_19/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2(
&sequential_9/lstm_19/lstm_cell_37/Relu�
'sequential_9/lstm_19/lstm_cell_37/mul_1Mul-sequential_9/lstm_19/lstm_cell_37/Sigmoid:y:04sequential_9/lstm_19/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2)
'sequential_9/lstm_19/lstm_cell_37/mul_1�
'sequential_9/lstm_19/lstm_cell_37/add_1AddV2)sequential_9/lstm_19/lstm_cell_37/mul:z:0+sequential_9/lstm_19/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2)
'sequential_9/lstm_19/lstm_cell_37/add_1�
+sequential_9/lstm_19/lstm_cell_37/Sigmoid_2Sigmoid0sequential_9/lstm_19/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2-
+sequential_9/lstm_19/lstm_cell_37/Sigmoid_2�
(sequential_9/lstm_19/lstm_cell_37/Relu_1Relu+sequential_9/lstm_19/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2*
(sequential_9/lstm_19/lstm_cell_37/Relu_1�
'sequential_9/lstm_19/lstm_cell_37/mul_2Mul/sequential_9/lstm_19/lstm_cell_37/Sigmoid_2:y:06sequential_9/lstm_19/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2)
'sequential_9/lstm_19/lstm_cell_37/mul_2�
2sequential_9/lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    24
2sequential_9/lstm_19/TensorArrayV2_1/element_shape�
$sequential_9/lstm_19/TensorArrayV2_1TensorListReserve;sequential_9/lstm_19/TensorArrayV2_1/element_shape:output:0-sequential_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_9/lstm_19/TensorArrayV2_1x
sequential_9/lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_9/lstm_19/time�
-sequential_9/lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_9/lstm_19/while/maximum_iterations�
'sequential_9/lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_19/while/loop_counter�
sequential_9/lstm_19/whileWhile0sequential_9/lstm_19/while/loop_counter:output:06sequential_9/lstm_19/while/maximum_iterations:output:0"sequential_9/lstm_19/time:output:0-sequential_9/lstm_19/TensorArrayV2_1:handle:0#sequential_9/lstm_19/zeros:output:0%sequential_9/lstm_19/zeros_1:output:0-sequential_9/lstm_19/strided_slice_1:output:0Lsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_19_lstm_cell_37_matmul_readvariableop_resourceBsequential_9_lstm_19_lstm_cell_37_matmul_1_readvariableop_resourceAsequential_9_lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *2
body*R(
&sequential_9_lstm_19_while_body_321911*2
cond*R(
&sequential_9_lstm_19_while_cond_321910*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
sequential_9/lstm_19/while�
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2G
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_19/while:output:3Nsequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype029
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack�
*sequential_9/lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_9/lstm_19/strided_slice_3/stack�
,sequential_9/lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_19/strided_slice_3/stack_1�
,sequential_9/lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_3/stack_2�
$sequential_9/lstm_19/strided_slice_3StridedSlice@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_19/strided_slice_3/stack:output:05sequential_9/lstm_19/strided_slice_3/stack_1:output:05sequential_9/lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_3�
%sequential_9/lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_19/transpose_1/perm�
 sequential_9/lstm_19/transpose_1	Transpose@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2"
 sequential_9/lstm_19/transpose_1�
sequential_9/lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_19/runtime�
sequential_9/dropout_9/IdentityIdentity-sequential_9/lstm_19/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2!
sequential_9/dropout_9/Identity�
*sequential_9/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_9_dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_9/dense_9/MatMul/ReadVariableOp�
sequential_9/dense_9/MatMulMatMul(sequential_9/dropout_9/Identity:output:02sequential_9/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_9/dense_9/MatMul�
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOp�
sequential_9/dense_9/BiasAddBiasAdd%sequential_9/dense_9/MatMul:product:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_9/dense_9/BiasAdd�
IdentityIdentity%sequential_9/dense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp+^sequential_9/dense_9/MatMul/ReadVariableOp9^sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp8^sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp:^sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp^sequential_9/lstm_18/while9^sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp8^sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp:^sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp^sequential_9/lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2X
*sequential_9/dense_9/MatMul/ReadVariableOp*sequential_9/dense_9/MatMul/ReadVariableOp2t
8sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp8sequential_9/lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp2r
7sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp7sequential_9/lstm_18/lstm_cell_36/MatMul/ReadVariableOp2v
9sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp9sequential_9/lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp28
sequential_9/lstm_18/whilesequential_9/lstm_18/while2t
8sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp8sequential_9/lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp2r
7sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp7sequential_9/lstm_19/lstm_cell_37/MatMul/ReadVariableOp2v
9sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp9sequential_9/lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp28
sequential_9/lstm_19/whilesequential_9/lstm_19/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�J
�

lstm_19_while_body_324427,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0:	@�P
=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �K
<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorL
9lstm_19_while_lstm_cell_37_matmul_readvariableop_resource:	@�N
;lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource:	 �I
:lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource:	���1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItem�
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp�
!lstm_19/while/lstm_cell_37/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_19/while/lstm_cell_37/MatMul�
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp�
#lstm_19/while/lstm_cell_37/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_19/while/lstm_cell_37/MatMul_1�
lstm_19/while/lstm_cell_37/addAddV2+lstm_19/while/lstm_cell_37/MatMul:product:0-lstm_19/while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_19/while/lstm_cell_37/add�
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp�
"lstm_19/while/lstm_cell_37/BiasAddBiasAdd"lstm_19/while/lstm_cell_37/add:z:09lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_19/while/lstm_cell_37/BiasAdd�
*lstm_19/while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_37/split/split_dim�
 lstm_19/while/lstm_cell_37/splitSplit3lstm_19/while/lstm_cell_37/split/split_dim:output:0+lstm_19/while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_19/while/lstm_cell_37/split�
"lstm_19/while/lstm_cell_37/SigmoidSigmoid)lstm_19/while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_19/while/lstm_cell_37/Sigmoid�
$lstm_19/while/lstm_cell_37/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_19/while/lstm_cell_37/Sigmoid_1�
lstm_19/while/lstm_cell_37/mulMul(lstm_19/while/lstm_cell_37/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_19/while/lstm_cell_37/mul�
lstm_19/while/lstm_cell_37/ReluRelu)lstm_19/while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_19/while/lstm_cell_37/Relu�
 lstm_19/while/lstm_cell_37/mul_1Mul&lstm_19/while/lstm_cell_37/Sigmoid:y:0-lstm_19/while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/mul_1�
 lstm_19/while/lstm_cell_37/add_1AddV2"lstm_19/while/lstm_cell_37/mul:z:0$lstm_19/while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/add_1�
$lstm_19/while/lstm_cell_37/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_19/while/lstm_cell_37/Sigmoid_2�
!lstm_19/while/lstm_cell_37/Relu_1Relu$lstm_19/while/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_19/while/lstm_cell_37/Relu_1�
 lstm_19/while/lstm_cell_37/mul_2Mul(lstm_19/while/lstm_cell_37/Sigmoid_2:y:0/lstm_19/while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_19/while/lstm_cell_37/mul_2�
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/y�
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/y�
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1�
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity�
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1�
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2�
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3�
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_37/mul_2:z:0^lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_19/while/Identity_4�
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_37/add_1:z:0^lstm_19/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_19/while/Identity_5�
lstm_19/while/NoOpNoOp2^lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_19/while/NoOp"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_37_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_37_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_37_matmul_readvariableop_resource;lstm_19_while_lstm_cell_37_matmul_readvariableop_resource_0"�
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_37/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_37/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324054

inputs!
lstm_18_324033:	�!
lstm_18_324035:	@�
lstm_18_324037:	�!
lstm_19_324040:	@�!
lstm_19_324042:	 �
lstm_19_324044:	� 
dense_9_324048: 
dense_9_324050:
identity��dense_9/StatefulPartitionedCall�!dropout_9/StatefulPartitionedCall�lstm_18/StatefulPartitionedCall�lstm_19/StatefulPartitionedCall�
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_324033lstm_18_324035lstm_18_324037*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3239982!
lstm_18/StatefulPartitionedCall�
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0lstm_19_324040lstm_19_324042lstm_19_324044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3238252!
lstm_19/StatefulPartitionedCall�
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3236582#
!dropout_9/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_9_324048dense_9_324050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3236022!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_9/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_325092
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
&sequential_9_lstm_18_while_cond_321763F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3H
Dsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1^
Zsequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_321763___redundant_placeholder0^
Zsequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_321763___redundant_placeholder1^
Zsequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_321763___redundant_placeholder2^
Zsequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_321763___redundant_placeholder3'
#sequential_9_lstm_18_while_identity
�
sequential_9/lstm_18/while/LessLess&sequential_9_lstm_18_while_placeholderDsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/while/Less�
#sequential_9/lstm_18/while/IdentityIdentity#sequential_9/lstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identity"S
#sequential_9_lstm_18_while_identity,sequential_9/lstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�F
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_322370

inputs&
lstm_cell_36_322288:	�&
lstm_cell_36_322290:	@�"
lstm_cell_36_322292:	�
identity��$lstm_cell_36/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_36_322288lstm_cell_36_322290lstm_cell_36_322292*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3222232&
$lstm_cell_36/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_36_322288lstm_cell_36_322290lstm_cell_36_322292*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322301*
condR
while_cond_322300*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity}
NoOpNoOp%^lstm_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_36/StatefulPartitionedCall$lstm_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�\
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_325025
inputs_0>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_324941*
condR
while_cond_324940*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
��
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324830

inputsF
3lstm_18_lstm_cell_36_matmul_readvariableop_resource:	�H
5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource:	@�C
4lstm_18_lstm_cell_36_biasadd_readvariableop_resource:	�F
3lstm_19_lstm_cell_37_matmul_readvariableop_resource:	@�H
5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource:	 �C
4lstm_19_lstm_cell_37_biasadd_readvariableop_resource:	�8
&dense_9_matmul_readvariableop_resource: 5
'dense_9_biasadd_readvariableop_resource:
identity��dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�*lstm_18/lstm_cell_36/MatMul/ReadVariableOp�,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�lstm_18/while�+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�*lstm_19/lstm_cell_37/MatMul/ReadVariableOp�,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�lstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape�
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack�
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1�
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2�
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros/mul/y�
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_18/zeros/Less/y�
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros/packed/1�
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/Const�
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros_1/mul/y�
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_18/zeros_1/Less/y�
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_18/zeros_1/packed/1�
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/Const�
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/zeros_1�
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm�
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1�
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack�
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1�
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2�
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1�
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_18/TensorArrayV2/element_shape�
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2�
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor�
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack�
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1�
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2�
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_18/strided_slice_2�
*lstm_18/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_18/lstm_cell_36/MatMul/ReadVariableOp�
lstm_18/lstm_cell_36/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/MatMul�
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_18/lstm_cell_36/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/MatMul_1�
lstm_18/lstm_cell_36/addAddV2%lstm_18/lstm_cell_36/MatMul:product:0'lstm_18/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/add�
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_18/lstm_cell_36/BiasAddBiasAddlstm_18/lstm_cell_36/add:z:03lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_18/lstm_cell_36/BiasAdd�
$lstm_18/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_36/split/split_dim�
lstm_18/lstm_cell_36/splitSplit-lstm_18/lstm_cell_36/split/split_dim:output:0%lstm_18/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_18/lstm_cell_36/split�
lstm_18/lstm_cell_36/SigmoidSigmoid#lstm_18/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Sigmoid�
lstm_18/lstm_cell_36/Sigmoid_1Sigmoid#lstm_18/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_18/lstm_cell_36/Sigmoid_1�
lstm_18/lstm_cell_36/mulMul"lstm_18/lstm_cell_36/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul�
lstm_18/lstm_cell_36/ReluRelu#lstm_18/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Relu�
lstm_18/lstm_cell_36/mul_1Mul lstm_18/lstm_cell_36/Sigmoid:y:0'lstm_18/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul_1�
lstm_18/lstm_cell_36/add_1AddV2lstm_18/lstm_cell_36/mul:z:0lstm_18/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/add_1�
lstm_18/lstm_cell_36/Sigmoid_2Sigmoid#lstm_18/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_18/lstm_cell_36/Sigmoid_2�
lstm_18/lstm_cell_36/Relu_1Relulstm_18/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/Relu_1�
lstm_18/lstm_cell_36/mul_2Mul"lstm_18/lstm_cell_36/Sigmoid_2:y:0)lstm_18/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_18/lstm_cell_36/mul_2�
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_18/TensorArrayV2_1/element_shape�
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/time�
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counter�
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_36_matmul_readvariableop_resource5lstm_18_lstm_cell_36_matmul_1_readvariableop_resource4lstm_18_lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_18_while_body_324585*%
condR
lstm_18_while_cond_324584*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_18/while�
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack�
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_18/strided_slice_3/stack�
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1�
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2�
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_18/strided_slice_3�
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/perm�
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimee
lstm_19/ShapeShapelstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_19/Shape�
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack�
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1�
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2�
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicel
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros/mul/y�
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_19/zeros/Less/y�
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lessr
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros/packed/1�
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/Const�
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/zerosp
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros_1/mul/y�
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_19/zeros_1/Less/y�
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessv
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/zeros_1/packed/1�
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/Const�
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/zeros_1�
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/perm�
lstm_19/transpose	Transposelstm_18/transpose_1:y:0lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1�
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack�
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1�
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2�
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1�
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_19/TensorArrayV2/element_shape�
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2�
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor�
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack�
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1�
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2�
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_19/strided_slice_2�
*lstm_19/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_19/lstm_cell_37/MatMul/ReadVariableOp�
lstm_19/lstm_cell_37/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/MatMul�
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_19/lstm_cell_37/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/MatMul_1�
lstm_19/lstm_cell_37/addAddV2%lstm_19/lstm_cell_37/MatMul:product:0'lstm_19/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/add�
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_19/lstm_cell_37/BiasAddBiasAddlstm_19/lstm_cell_37/add:z:03lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_19/lstm_cell_37/BiasAdd�
$lstm_19/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_37/split/split_dim�
lstm_19/lstm_cell_37/splitSplit-lstm_19/lstm_cell_37/split/split_dim:output:0%lstm_19/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_19/lstm_cell_37/split�
lstm_19/lstm_cell_37/SigmoidSigmoid#lstm_19/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Sigmoid�
lstm_19/lstm_cell_37/Sigmoid_1Sigmoid#lstm_19/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_19/lstm_cell_37/Sigmoid_1�
lstm_19/lstm_cell_37/mulMul"lstm_19/lstm_cell_37/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul�
lstm_19/lstm_cell_37/ReluRelu#lstm_19/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Relu�
lstm_19/lstm_cell_37/mul_1Mul lstm_19/lstm_cell_37/Sigmoid:y:0'lstm_19/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul_1�
lstm_19/lstm_cell_37/add_1AddV2lstm_19/lstm_cell_37/mul:z:0lstm_19/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/add_1�
lstm_19/lstm_cell_37/Sigmoid_2Sigmoid#lstm_19/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_19/lstm_cell_37/Sigmoid_2�
lstm_19/lstm_cell_37/Relu_1Relulstm_19/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/Relu_1�
lstm_19/lstm_cell_37/mul_2Mul"lstm_19/lstm_cell_37/Sigmoid_2:y:0)lstm_19/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_19/lstm_cell_37/mul_2�
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_19/TensorArrayV2_1/element_shape�
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/time�
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counter�
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_37_matmul_readvariableop_resource5lstm_19_lstm_cell_37_matmul_1_readvariableop_resource4lstm_19_lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_19_while_body_324732*%
condR
lstm_19_while_cond_324731*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_19/while�
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack�
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_19/strided_slice_3/stack�
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1�
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2�
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_19/strided_slice_3�
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/perm�
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtimew
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_9/dropout/Const�
dropout_9/dropout/MulMul lstm_19/strided_slice_3:output:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout_9/dropout/Mul�
dropout_9/dropout/ShapeShape lstm_19/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape�
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype020
.dropout_9/dropout/random_uniform/RandomUniform�
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_9/dropout/GreaterEqual/y�
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2 
dropout_9/dropout/GreaterEqual�
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout_9/dropout/Cast�
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout_9/dropout/Mul_1�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_9/MatMul/ReadVariableOp�
dense_9/MatMulMatMuldropout_9/dropout/Mul_1:z:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9/MatMul�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_9/BiasAdds
IdentityIdentitydense_9/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp,^lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_36/MatMul/ReadVariableOp-^lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_37/MatMul/ReadVariableOp-^lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2Z
+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_36/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_36/MatMul/ReadVariableOp*lstm_18/lstm_cell_36/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_36/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_37/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_37/MatMul/ReadVariableOp*lstm_19/lstm_cell_37/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_37/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_323334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323334___redundant_placeholder04
0while_while_cond_323334___redundant_placeholder14
0while_while_cond_323334___redundant_placeholder24
0while_while_cond_323334___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_19_layer_call_fn_325500
inputs_0
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3230002
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
(__inference_lstm_18_layer_call_fn_324863

inputs
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3234192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_37_layer_call_fn_326287

inputs
states_0
states_1
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3227072
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�

�
lstm_18_while_cond_324584,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1D
@lstm_18_while_lstm_18_while_cond_324584___redundant_placeholder0D
@lstm_18_while_lstm_18_while_cond_324584___redundant_placeholder1D
@lstm_18_while_lstm_18_while_cond_324584___redundant_placeholder2D
@lstm_18_while_lstm_18_while_cond_324584___redundant_placeholder3
lstm_18_while_identity
�
lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_323913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323913___redundant_placeholder04
0while_while_cond_323913___redundant_placeholder14
0while_while_cond_323913___redundant_placeholder24
0while_while_cond_323913___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_19_layer_call_fn_325522

inputs
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3238252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�I
�
__inference__traced_save_326484
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_18_lstm_cell_36_kernel_read_readvariableopD
@savev2_lstm_18_lstm_cell_36_recurrent_kernel_read_readvariableop8
4savev2_lstm_18_lstm_cell_36_bias_read_readvariableop:
6savev2_lstm_19_lstm_cell_37_kernel_read_readvariableopD
@savev2_lstm_19_lstm_cell_37_recurrent_kernel_read_readvariableop8
4savev2_lstm_19_lstm_cell_37_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_9_kernel_m_read_readvariableop2
.savev2_adam_dense_9_bias_m_read_readvariableopA
=savev2_adam_lstm_18_lstm_cell_36_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_18_lstm_cell_36_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_18_lstm_cell_36_bias_m_read_readvariableopA
=savev2_adam_lstm_19_lstm_cell_37_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_19_lstm_cell_37_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_19_lstm_cell_37_bias_m_read_readvariableop4
0savev2_adam_dense_9_kernel_v_read_readvariableop2
.savev2_adam_dense_9_bias_v_read_readvariableopA
=savev2_adam_lstm_18_lstm_cell_36_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_18_lstm_cell_36_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_18_lstm_cell_36_bias_v_read_readvariableopA
=savev2_adam_lstm_19_lstm_cell_37_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_19_lstm_cell_37_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_19_lstm_cell_37_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_18_lstm_cell_36_kernel_read_readvariableop@savev2_lstm_18_lstm_cell_36_recurrent_kernel_read_readvariableop4savev2_lstm_18_lstm_cell_36_bias_read_readvariableop6savev2_lstm_19_lstm_cell_37_kernel_read_readvariableop@savev2_lstm_19_lstm_cell_37_recurrent_kernel_read_readvariableop4savev2_lstm_19_lstm_cell_37_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_9_kernel_m_read_readvariableop.savev2_adam_dense_9_bias_m_read_readvariableop=savev2_adam_lstm_18_lstm_cell_36_kernel_m_read_readvariableopGsavev2_adam_lstm_18_lstm_cell_36_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_18_lstm_cell_36_bias_m_read_readvariableop=savev2_adam_lstm_19_lstm_cell_37_kernel_m_read_readvariableopGsavev2_adam_lstm_19_lstm_cell_37_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_19_lstm_cell_37_bias_m_read_readvariableop0savev2_adam_dense_9_kernel_v_read_readvariableop.savev2_adam_dense_9_bias_v_read_readvariableop=savev2_adam_lstm_18_lstm_cell_36_kernel_v_read_readvariableopGsavev2_adam_lstm_18_lstm_cell_36_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_18_lstm_cell_36_bias_v_read_readvariableop=savev2_adam_lstm_19_lstm_cell_37_kernel_v_read_readvariableopGsavev2_adam_lstm_19_lstm_cell_37_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_19_lstm_cell_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : :: : : : : :	�:	@�:�:	@�:	 �:�: : : ::	�:	@�:�:	@�:	 �:�: ::	�:	@�:�:	@�:	 �:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%	!

_output_shapes
:	@�:!


_output_shapes	
:�:%!

_output_shapes
:	@�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	@�:%!

_output_shapes
:	 �:!

_output_shapes	
:�:$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	�:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	@�:%!

_output_shapes
:	 �:!

_output_shapes	
:�: 

_output_shapes
: 
�[
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_323419

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323335*
condR
while_cond_323334*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�[
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_323577

inputs>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323493*
condR
while_cond_323492*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_323609

inputs!
lstm_18_323420:	�!
lstm_18_323422:	@�
lstm_18_323424:	�!
lstm_19_323578:	@�!
lstm_19_323580:	 �
lstm_19_323582:	� 
dense_9_323603: 
dense_9_323605:
identity��dense_9/StatefulPartitionedCall�lstm_18/StatefulPartitionedCall�lstm_19/StatefulPartitionedCall�
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_323420lstm_18_323422lstm_18_323424*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3234192!
lstm_18/StatefulPartitionedCall�
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0lstm_19_323578lstm_19_323580lstm_19_323582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3235772!
lstm_19/StatefulPartitionedCall�
dropout_9/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3235902
dropout_9/PartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_9_323603dense_9_323605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3236022!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_323590

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_322707

inputs

states
states_11
matmul_readvariableop_resource:	@�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�F
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_323000

inputs&
lstm_cell_37_322918:	@�&
lstm_cell_37_322920:	 �"
lstm_cell_37_322922:	�
identity��$lstm_cell_37/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_37_322918lstm_cell_37_322920lstm_cell_37_322922*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3228532&
$lstm_cell_37/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_37_322918lstm_cell_37_322920lstm_cell_37_322922*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322931*
condR
while_cond_322930*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity}
NoOpNoOp%^lstm_cell_37/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_37/StatefulPartitionedCall$lstm_cell_37/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�\
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_325824
inputs_0>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325740*
condR
while_cond_325739*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
(__inference_lstm_18_layer_call_fn_324852
inputs_0
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3223702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
c
*__inference_dropout_9_layer_call_fn_326136

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3236582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
&sequential_9_lstm_19_while_cond_321910F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3H
Dsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1^
Zsequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_321910___redundant_placeholder0^
Zsequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_321910___redundant_placeholder1^
Zsequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_321910___redundant_placeholder2^
Zsequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_321910___redundant_placeholder3'
#sequential_9_lstm_19_while_identity
�
sequential_9/lstm_19/while/LessLess&sequential_9_lstm_19_while_placeholderDsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/while/Less�
#sequential_9/lstm_19/while/IdentityIdentity#sequential_9/lstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identity"S
#sequential_9_lstm_19_while_identity,sequential_9/lstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_322223

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_326141

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:��������� 2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:��������� 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�?
�
while_body_326042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�	
�
$__inference_signature_wrapper_324171
lstm_18_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_3220022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�F
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_322790

inputs&
lstm_cell_37_322708:	@�&
lstm_cell_37_322710:	 �"
lstm_cell_37_322712:	�
identity��$lstm_cell_37/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_37/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_37_322708lstm_cell_37_322710lstm_cell_37_322712*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3227072&
$lstm_cell_37/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_37_322708lstm_cell_37_322710lstm_cell_37_322712*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322721*
condR
while_cond_322720*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������ *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������ 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity}
NoOpNoOp%^lstm_cell_37/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_37/StatefulPartitionedCall$lstm_cell_37/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_322853

inputs

states
states_11
matmul_readvariableop_resource:	@�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates:OK
'
_output_shapes
:��������� 
 
_user_specified_namestates
�
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_326153

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�\
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_325176
inputs_0>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325092*
condR
while_cond_325091*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�%
�
while_body_322931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_37_322955_0:	@�.
while_lstm_cell_37_322957_0:	 �*
while_lstm_cell_37_322959_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_37_322955:	@�,
while_lstm_cell_37_322957:	 �(
while_lstm_cell_37_322959:	���*while/lstm_cell_37/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_37/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_37_322955_0while_lstm_cell_37_322957_0while_lstm_cell_37_322959_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3228532,
*while/lstm_cell_37/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_37/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_37/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_37/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_37_322955while_lstm_cell_37_322955_0"8
while_lstm_cell_37_322957while_lstm_cell_37_322957_0"8
while_lstm_cell_37_322959while_lstm_cell_37_322959_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_37/StatefulPartitionedCall*while/lstm_cell_37/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�

�
lstm_19_while_cond_324426,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1D
@lstm_19_while_lstm_19_while_cond_324426___redundant_placeholder0D
@lstm_19_while_lstm_19_while_cond_324426___redundant_placeholder1D
@lstm_19_while_lstm_19_while_cond_324426___redundant_placeholder2D
@lstm_19_while_lstm_19_while_cond_324426___redundant_placeholder3
lstm_19_while_identity
�
lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_18_layer_call_fn_324874

inputs
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3239982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_325091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325091___redundant_placeholder04
0while_while_cond_325091___redundant_placeholder14
0while_while_cond_325091___redundant_placeholder24
0while_while_cond_325091___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
�
(__inference_lstm_19_layer_call_fn_325511

inputs
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3235772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�?
�
while_body_323741
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�?
�
while_body_324941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_36_layer_call_fn_326189

inputs
states_0
states_1
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3220772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�[
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_323825

inputs>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323741*
condR
while_cond_323740*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�?
�
while_body_325891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�[
�
C__inference_lstm_19_layer_call_and_return_conditional_losses_325975

inputs>
+lstm_cell_37_matmul_readvariableop_resource:	@�@
-lstm_cell_37_matmul_1_readvariableop_resource:	 �;
,lstm_cell_37_biasadd_readvariableop_resource:	�
identity��#lstm_cell_37/BiasAdd/ReadVariableOp�"lstm_cell_37/MatMul/ReadVariableOp�$lstm_cell_37/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������@2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_37/MatMul/ReadVariableOpReadVariableOp+lstm_cell_37_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_37/MatMul/ReadVariableOp�
lstm_cell_37/MatMulMatMulstrided_slice_2:output:0*lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul�
$lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_37_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_37/MatMul_1/ReadVariableOp�
lstm_cell_37/MatMul_1MatMulzeros:output:0,lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/MatMul_1�
lstm_cell_37/addAddV2lstm_cell_37/MatMul:product:0lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/add�
#lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_37/BiasAdd/ReadVariableOp�
lstm_cell_37/BiasAddBiasAddlstm_cell_37/add:z:0+lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_37/BiasAdd~
lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_37/split/split_dim�
lstm_cell_37/splitSplit%lstm_cell_37/split/split_dim:output:0lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_37/split�
lstm_cell_37/SigmoidSigmoidlstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid�
lstm_cell_37/Sigmoid_1Sigmoidlstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_1�
lstm_cell_37/mulMullstm_cell_37/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul}
lstm_cell_37/ReluRelulstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu�
lstm_cell_37/mul_1Mullstm_cell_37/Sigmoid:y:0lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_1�
lstm_cell_37/add_1AddV2lstm_cell_37/mul:z:0lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/add_1�
lstm_cell_37/Sigmoid_2Sigmoidlstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Sigmoid_2|
lstm_cell_37/Relu_1Relulstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/Relu_1�
lstm_cell_37/mul_2Mullstm_cell_37/Sigmoid_2:y:0!lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_37/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_37_matmul_readvariableop_resource-lstm_cell_37_matmul_1_readvariableop_resource,lstm_cell_37_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325891*
condR
while_cond_325890*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimes
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity�
NoOpNoOp$^lstm_cell_37/BiasAdd/ReadVariableOp#^lstm_cell_37/MatMul/ReadVariableOp%^lstm_cell_37/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_37/BiasAdd/ReadVariableOp#lstm_cell_37/BiasAdd/ReadVariableOp2H
"lstm_cell_37/MatMul/ReadVariableOp"lstm_cell_37/MatMul/ReadVariableOp2L
$lstm_cell_37/MatMul_1/ReadVariableOp$lstm_cell_37/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_9_layer_call_fn_326162

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3236022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�?
�
while_body_323493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_325890
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325890___redundant_placeholder04
0while_while_cond_325890___redundant_placeholder14
0while_while_cond_325890___redundant_placeholder24
0while_while_cond_325890___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�

�
-__inference_sequential_9_layer_call_fn_323628
lstm_18_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3236092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�
�
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326238

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�J
�

lstm_18_while_body_324280,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0:	�P
=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�K
<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_36_matmul_readvariableop_resource:	�N
;lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource:	@�I
:lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource:	���1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItem�
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�
!lstm_18/while/lstm_cell_36/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_18/while/lstm_cell_36/MatMul�
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
#lstm_18/while/lstm_cell_36/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_18/while/lstm_cell_36/MatMul_1�
lstm_18/while/lstm_cell_36/addAddV2+lstm_18/while/lstm_cell_36/MatMul:product:0-lstm_18/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_18/while/lstm_cell_36/add�
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�
"lstm_18/while/lstm_cell_36/BiasAddBiasAdd"lstm_18/while/lstm_cell_36/add:z:09lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_18/while/lstm_cell_36/BiasAdd�
*lstm_18/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_36/split/split_dim�
 lstm_18/while/lstm_cell_36/splitSplit3lstm_18/while/lstm_cell_36/split/split_dim:output:0+lstm_18/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_18/while/lstm_cell_36/split�
"lstm_18/while/lstm_cell_36/SigmoidSigmoid)lstm_18/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_18/while/lstm_cell_36/Sigmoid�
$lstm_18/while/lstm_cell_36/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_18/while/lstm_cell_36/Sigmoid_1�
lstm_18/while/lstm_cell_36/mulMul(lstm_18/while/lstm_cell_36/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_18/while/lstm_cell_36/mul�
lstm_18/while/lstm_cell_36/ReluRelu)lstm_18/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_18/while/lstm_cell_36/Relu�
 lstm_18/while/lstm_cell_36/mul_1Mul&lstm_18/while/lstm_cell_36/Sigmoid:y:0-lstm_18/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/mul_1�
 lstm_18/while/lstm_cell_36/add_1AddV2"lstm_18/while/lstm_cell_36/mul:z:0$lstm_18/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/add_1�
$lstm_18/while/lstm_cell_36/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_18/while/lstm_cell_36/Sigmoid_2�
!lstm_18/while/lstm_cell_36/Relu_1Relu$lstm_18/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_18/while/lstm_cell_36/Relu_1�
 lstm_18/while/lstm_cell_36/mul_2Mul(lstm_18/while/lstm_cell_36/Sigmoid_2:y:0/lstm_18/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_18/while/lstm_cell_36/mul_2�
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/y�
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/y�
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1�
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity�
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1�
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2�
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3�
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_36/mul_2:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_18/while/Identity_4�
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_36/add_1:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_18/while/Identity_5�
lstm_18/while/NoOpNoOp2^lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_18/while/NoOp"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_36_matmul_readvariableop_resource;lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0"�
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�?
�
while_body_325589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
-__inference_lstm_cell_36_layer_call_fn_326206

inputs
states_0
states_1
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3222232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�%
�
while_body_322721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_37_322745_0:	@�.
while_lstm_cell_37_322747_0:	 �*
while_lstm_cell_37_322749_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_37_322745:	@�,
while_lstm_cell_37_322747:	 �(
while_lstm_cell_37_322749:	���*while/lstm_cell_37/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_37/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_37_322745_0while_lstm_cell_37_322747_0while_lstm_cell_37_322749_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:��������� :��������� :��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_3227072,
*while/lstm_cell_37/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_37/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_37/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_37/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_37_322745while_lstm_cell_37_322745_0"8
while_lstm_cell_37_322747while_lstm_cell_37_322747_0"8
while_lstm_cell_37_322749while_lstm_cell_37_322749_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_37/StatefulPartitionedCall*while/lstm_cell_37/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�
�
(__inference_lstm_18_layer_call_fn_324841
inputs_0
unknown:	�
	unknown_0:	@�
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3221602
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
lstm_18_while_cond_324279,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1D
@lstm_18_while_lstm_18_while_cond_324279___redundant_placeholder0D
@lstm_18_while_lstm_18_while_cond_324279___redundant_placeholder1D
@lstm_18_while_lstm_18_while_cond_324279___redundant_placeholder2D
@lstm_18_while_lstm_18_while_cond_324279___redundant_placeholder3
lstm_18_while_identity
�
lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_323658

inputs
identity�c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2
dropout/GreaterEqual/y�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:��������� :O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326336

inputs
states_0
states_11
matmul_readvariableop_resource:	@�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
"__inference__traced_restore_326587
file_prefix1
assignvariableop_dense_9_kernel: -
assignvariableop_1_dense_9_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_18_lstm_cell_36_kernel:	�K
8assignvariableop_8_lstm_18_lstm_cell_36_recurrent_kernel:	@�;
,assignvariableop_9_lstm_18_lstm_cell_36_bias:	�B
/assignvariableop_10_lstm_19_lstm_cell_37_kernel:	@�L
9assignvariableop_11_lstm_19_lstm_cell_37_recurrent_kernel:	 �<
-assignvariableop_12_lstm_19_lstm_cell_37_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_9_kernel_m: 5
'assignvariableop_16_adam_dense_9_bias_m:I
6assignvariableop_17_adam_lstm_18_lstm_cell_36_kernel_m:	�S
@assignvariableop_18_adam_lstm_18_lstm_cell_36_recurrent_kernel_m:	@�C
4assignvariableop_19_adam_lstm_18_lstm_cell_36_bias_m:	�I
6assignvariableop_20_adam_lstm_19_lstm_cell_37_kernel_m:	@�S
@assignvariableop_21_adam_lstm_19_lstm_cell_37_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_lstm_19_lstm_cell_37_bias_m:	�;
)assignvariableop_23_adam_dense_9_kernel_v: 5
'assignvariableop_24_adam_dense_9_bias_v:I
6assignvariableop_25_adam_lstm_18_lstm_cell_36_kernel_v:	�S
@assignvariableop_26_adam_lstm_18_lstm_cell_36_recurrent_kernel_v:	@�C
4assignvariableop_27_adam_lstm_18_lstm_cell_36_bias_v:	�I
6assignvariableop_28_adam_lstm_19_lstm_cell_37_kernel_v:	@�S
@assignvariableop_29_adam_lstm_19_lstm_cell_37_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_lstm_19_lstm_cell_37_bias_v:	�
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_18_lstm_cell_36_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_18_lstm_cell_36_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_18_lstm_cell_36_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_19_lstm_cell_37_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_19_lstm_cell_37_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_19_lstm_cell_37_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_9_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_9_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_18_lstm_cell_36_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_18_lstm_cell_36_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_18_lstm_cell_36_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_19_lstm_cell_37_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_19_lstm_cell_37_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_19_lstm_cell_37_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_9_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_9_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_18_lstm_cell_36_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_18_lstm_cell_36_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_18_lstm_cell_36_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_19_lstm_cell_37_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_19_lstm_cell_37_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_19_lstm_cell_37_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31f
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_32�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
while_cond_325242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325242___redundant_placeholder04
0while_while_cond_325242___redundant_placeholder14
0while_while_cond_325242___redundant_placeholder24
0while_while_cond_325242___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�?
�
while_body_323914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_36_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_36_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_36_matmul_readvariableop_resource:	�F
3while_lstm_cell_36_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_36_biasadd_readvariableop_resource:	���)while/lstm_cell_36/BiasAdd/ReadVariableOp�(while/lstm_cell_36/MatMul/ReadVariableOp�*while/lstm_cell_36/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_36/MatMul/ReadVariableOp�
while/lstm_cell_36/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul�
*while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_36/MatMul_1/ReadVariableOp�
while/lstm_cell_36/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/MatMul_1�
while/lstm_cell_36/addAddV2#while/lstm_cell_36/MatMul:product:0%while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/add�
)while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_36/BiasAdd/ReadVariableOp�
while/lstm_cell_36/BiasAddBiasAddwhile/lstm_cell_36/add:z:01while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_36/BiasAdd�
"while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_36/split/split_dim�
while/lstm_cell_36/splitSplit+while/lstm_cell_36/split/split_dim:output:0#while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_36/split�
while/lstm_cell_36/SigmoidSigmoid!while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid�
while/lstm_cell_36/Sigmoid_1Sigmoid!while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_1�
while/lstm_cell_36/mulMul while/lstm_cell_36/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul�
while/lstm_cell_36/ReluRelu!while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu�
while/lstm_cell_36/mul_1Mulwhile/lstm_cell_36/Sigmoid:y:0%while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_1�
while/lstm_cell_36/add_1AddV2while/lstm_cell_36/mul:z:0while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/add_1�
while/lstm_cell_36/Sigmoid_2Sigmoid!while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Sigmoid_2�
while/lstm_cell_36/Relu_1Reluwhile/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/Relu_1�
while/lstm_cell_36/mul_2Mul while/lstm_cell_36/Sigmoid_2:y:0'while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_36/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_36/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_36/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_36/BiasAdd/ReadVariableOp)^while/lstm_cell_36/MatMul/ReadVariableOp+^while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_36_biasadd_readvariableop_resource4while_lstm_cell_36_biasadd_readvariableop_resource_0"l
3while_lstm_cell_36_matmul_1_readvariableop_resource5while_lstm_cell_36_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_36_matmul_readvariableop_resource3while_lstm_cell_36_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_36/BiasAdd/ReadVariableOp)while/lstm_cell_36/BiasAdd/ReadVariableOp2T
(while/lstm_cell_36/MatMul/ReadVariableOp(while/lstm_cell_36/MatMul/ReadVariableOp2X
*while/lstm_cell_36/MatMul_1/ReadVariableOp*while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326368

inputs
states_0
states_11
matmul_readvariableop_resource:	@�3
 matmul_1_readvariableop_resource:	 �.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:��������� 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:��������� 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:��������� 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:��������� 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:��������� 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:��������� 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:��������� 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:��������� 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������@:��������� :��������� : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/0:QM
'
_output_shapes
:��������� 
"
_user_specified_name
states/1
�
�
while_cond_326041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_326041___redundant_placeholder04
0while_while_cond_326041___redundant_placeholder14
0while_while_cond_326041___redundant_placeholder24
0while_while_cond_326041___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�?
�
while_body_325740
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_37_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_37_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_37_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_37_matmul_readvariableop_resource:	@�F
3while_lstm_cell_37_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_37_biasadd_readvariableop_resource:	���)while/lstm_cell_37/BiasAdd/ReadVariableOp�(while/lstm_cell_37/MatMul/ReadVariableOp�*while/lstm_cell_37/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
(while/lstm_cell_37/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_37_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_37/MatMul/ReadVariableOp�
while/lstm_cell_37/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul�
*while/lstm_cell_37/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_37_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_37/MatMul_1/ReadVariableOp�
while/lstm_cell_37/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_37/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/MatMul_1�
while/lstm_cell_37/addAddV2#while/lstm_cell_37/MatMul:product:0%while/lstm_cell_37/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/add�
)while/lstm_cell_37/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_37_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_37/BiasAdd/ReadVariableOp�
while/lstm_cell_37/BiasAddBiasAddwhile/lstm_cell_37/add:z:01while/lstm_cell_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_37/BiasAdd�
"while/lstm_cell_37/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_37/split/split_dim�
while/lstm_cell_37/splitSplit+while/lstm_cell_37/split/split_dim:output:0#while/lstm_cell_37/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_37/split�
while/lstm_cell_37/SigmoidSigmoid!while/lstm_cell_37/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid�
while/lstm_cell_37/Sigmoid_1Sigmoid!while/lstm_cell_37/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_1�
while/lstm_cell_37/mulMul while/lstm_cell_37/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul�
while/lstm_cell_37/ReluRelu!while/lstm_cell_37/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu�
while/lstm_cell_37/mul_1Mulwhile/lstm_cell_37/Sigmoid:y:0%while/lstm_cell_37/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_1�
while/lstm_cell_37/add_1AddV2while/lstm_cell_37/mul:z:0while/lstm_cell_37/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/add_1�
while/lstm_cell_37/Sigmoid_2Sigmoid!while/lstm_cell_37/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Sigmoid_2�
while/lstm_cell_37/Relu_1Reluwhile/lstm_cell_37/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/Relu_1�
while/lstm_cell_37/mul_2Mul while/lstm_cell_37/Sigmoid_2:y:0'while/lstm_cell_37/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_37/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_37/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identitywhile/lstm_cell_37/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_37/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_37/BiasAdd/ReadVariableOp)^while/lstm_cell_37/MatMul/ReadVariableOp+^while/lstm_cell_37/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_37_biasadd_readvariableop_resource4while_lstm_cell_37_biasadd_readvariableop_resource_0"l
3while_lstm_cell_37_matmul_1_readvariableop_resource5while_lstm_cell_37_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_37_matmul_readvariableop_resource3while_lstm_cell_37_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_37/BiasAdd/ReadVariableOp)while/lstm_cell_37/BiasAdd/ReadVariableOp2T
(while/lstm_cell_37/MatMul/ReadVariableOp(while/lstm_cell_37/MatMul/ReadVariableOp2X
*while/lstm_cell_37/MatMul_1/ReadVariableOp*while/lstm_cell_37/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
: 
�%
�
while_body_322301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_36_322325_0:	�.
while_lstm_cell_36_322327_0:	@�*
while_lstm_cell_36_322329_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_36_322325:	�,
while_lstm_cell_36_322327:	@�(
while_lstm_cell_36_322329:	���*while/lstm_cell_36/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape�
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem�
*while/lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_36_322325_0while_lstm_cell_36_322327_0while_lstm_cell_36_322329_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3222232,
*while/lstm_cell_36/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_36/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2�
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3�
while/Identity_4Identity3while/lstm_cell_36/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_36/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_36/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_36_322325while_lstm_cell_36_322325_0"8
while_lstm_cell_36_322327while_lstm_cell_36_322327_0"8
while_lstm_cell_36_322329while_lstm_cell_36_322329_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_36/StatefulPartitionedCall*while/lstm_cell_36/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�
�
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326270

inputs
states_0
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/0:QM
'
_output_shapes
:���������@
"
_user_specified_name
states/1
�
�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324118
lstm_18_input!
lstm_18_324097:	�!
lstm_18_324099:	@�
lstm_18_324101:	�!
lstm_19_324104:	@�!
lstm_19_324106:	 �
lstm_19_324108:	� 
dense_9_324112: 
dense_9_324114:
identity��dense_9/StatefulPartitionedCall�lstm_18/StatefulPartitionedCall�lstm_19/StatefulPartitionedCall�
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_324097lstm_18_324099lstm_18_324101*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_18_layer_call_and_return_conditional_losses_3234192!
lstm_18/StatefulPartitionedCall�
lstm_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0lstm_19_324104lstm_19_324106lstm_19_324108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3235772!
lstm_19/StatefulPartitionedCall�
dropout_9/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_3235902
dropout_9/PartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_9_324112dense_9_324114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_9_layer_call_and_return_conditional_losses_3236022!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_18_input
�
�
while_cond_323740
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_323740___redundant_placeholder04
0while_while_cond_323740___redundant_placeholder14
0while_while_cond_323740___redundant_placeholder24
0while_while_cond_323740___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�F
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_322160

inputs&
lstm_cell_36_322078:	�&
lstm_cell_36_322080:	@�"
lstm_cell_36_322082:	�
identity��$lstm_cell_36/StatefulPartitionedCall�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
$lstm_cell_36/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_36_322078lstm_cell_36_322080lstm_cell_36_322082*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������@:���������@:���������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_3220772&
$lstm_cell_36/StatefulPartitionedCall�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_36_322078lstm_cell_36_322080lstm_cell_36_322082*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_322091*
condR
while_cond_322090*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@2

Identity}
NoOpNoOp%^lstm_cell_36/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_36/StatefulPartitionedCall$lstm_cell_36/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_325739
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325739___redundant_placeholder04
0while_while_cond_325739___redundant_placeholder14
0while_while_cond_325739___redundant_placeholder24
0while_while_cond_325739___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_322077

inputs

states
states_11
matmul_readvariableop_resource:	�3
 matmul_1_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:����������2
add�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������@2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������@2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������@2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������@2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������@2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������@2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������@2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity_2�
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������@:���������@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������@
 
_user_specified_namestates:OK
'
_output_shapes
:���������@
 
_user_specified_namestates
�

�
-__inference_sequential_9_layer_call_fn_324213

inputs
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_9_layer_call_and_return_conditional_losses_3240542
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_lstm_19_layer_call_fn_325489
inputs_0
unknown:	@�
	unknown_0:	 �
	unknown_1:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_lstm_19_layer_call_and_return_conditional_losses_3227902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_325588
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_325588___redundant_placeholder04
0while_while_cond_325588___redundant_placeholder14
0while_while_cond_325588___redundant_placeholder24
0while_while_cond_325588___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�
�
while_cond_322930
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322930___redundant_placeholder04
0while_while_cond_322930___redundant_placeholder14
0while_while_cond_322930___redundant_placeholder24
0while_while_cond_322930___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :��������� :��������� : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :

_output_shapes
:
�

�
C__inference_dense_9_layer_call_and_return_conditional_losses_326172

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
while_cond_322090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_322090___redundant_placeholder04
0while_while_cond_322090___redundant_placeholder14
0while_while_cond_322090___redundant_placeholder24
0while_while_cond_322090___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������@:���������@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
:
�[
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_323998

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_323914*
condR
while_cond_323913*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�]
�
&sequential_9_lstm_18_while_body_321764F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3E
Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0�
}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_9_lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0:	�]
Jsequential_9_lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0:	@�X
Isequential_9_lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0:	�'
#sequential_9_lstm_18_while_identity)
%sequential_9_lstm_18_while_identity_1)
%sequential_9_lstm_18_while_identity_2)
%sequential_9_lstm_18_while_identity_3)
%sequential_9_lstm_18_while_identity_4)
%sequential_9_lstm_18_while_identity_5C
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensorY
Fsequential_9_lstm_18_while_lstm_cell_36_matmul_readvariableop_resource:	�[
Hsequential_9_lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource:	@�V
Gsequential_9_lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource:	���>sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�=sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�?sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2N
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape�
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_18_while_placeholderUsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02@
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem�
=sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02?
=sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp�
.sequential_9/lstm_18/while/lstm_cell_36/MatMulMatMulEsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������20
.sequential_9/lstm_18/while/lstm_cell_36/MatMul�
?sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02A
?sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp�
0sequential_9/lstm_18/while/lstm_cell_36/MatMul_1MatMul(sequential_9_lstm_18_while_placeholder_2Gsequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_9/lstm_18/while/lstm_cell_36/MatMul_1�
+sequential_9/lstm_18/while/lstm_cell_36/addAddV28sequential_9/lstm_18/while/lstm_cell_36/MatMul:product:0:sequential_9/lstm_18/while/lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2-
+sequential_9/lstm_18/while/lstm_cell_36/add�
>sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02@
>sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp�
/sequential_9/lstm_18/while/lstm_cell_36/BiasAddBiasAdd/sequential_9/lstm_18/while/lstm_cell_36/add:z:0Fsequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_9/lstm_18/while/lstm_cell_36/BiasAdd�
7sequential_9/lstm_18/while/lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_18/while/lstm_cell_36/split/split_dim�
-sequential_9/lstm_18/while/lstm_cell_36/splitSplit@sequential_9/lstm_18/while/lstm_cell_36/split/split_dim:output:08sequential_9/lstm_18/while/lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2/
-sequential_9/lstm_18/while/lstm_cell_36/split�
/sequential_9/lstm_18/while/lstm_cell_36/SigmoidSigmoid6sequential_9/lstm_18/while/lstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@21
/sequential_9/lstm_18/while/lstm_cell_36/Sigmoid�
1sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_1Sigmoid6sequential_9/lstm_18/while/lstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@23
1sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_1�
+sequential_9/lstm_18/while/lstm_cell_36/mulMul5sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_1:y:0(sequential_9_lstm_18_while_placeholder_3*
T0*'
_output_shapes
:���������@2-
+sequential_9/lstm_18/while/lstm_cell_36/mul�
,sequential_9/lstm_18/while/lstm_cell_36/ReluRelu6sequential_9/lstm_18/while/lstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2.
,sequential_9/lstm_18/while/lstm_cell_36/Relu�
-sequential_9/lstm_18/while/lstm_cell_36/mul_1Mul3sequential_9/lstm_18/while/lstm_cell_36/Sigmoid:y:0:sequential_9/lstm_18/while/lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2/
-sequential_9/lstm_18/while/lstm_cell_36/mul_1�
-sequential_9/lstm_18/while/lstm_cell_36/add_1AddV2/sequential_9/lstm_18/while/lstm_cell_36/mul:z:01sequential_9/lstm_18/while/lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2/
-sequential_9/lstm_18/while/lstm_cell_36/add_1�
1sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_2Sigmoid6sequential_9/lstm_18/while/lstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@23
1sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_2�
.sequential_9/lstm_18/while/lstm_cell_36/Relu_1Relu1sequential_9/lstm_18/while/lstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@20
.sequential_9/lstm_18/while/lstm_cell_36/Relu_1�
-sequential_9/lstm_18/while/lstm_cell_36/mul_2Mul5sequential_9/lstm_18/while/lstm_cell_36/Sigmoid_2:y:0<sequential_9/lstm_18/while/lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2/
-sequential_9/lstm_18/while/lstm_cell_36/mul_2�
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_18_while_placeholder_1&sequential_9_lstm_18_while_placeholder1sequential_9/lstm_18/while/lstm_cell_36/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem�
 sequential_9/lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_18/while/add/y�
sequential_9/lstm_18/while/addAddV2&sequential_9_lstm_18_while_placeholder)sequential_9/lstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/while/add�
"sequential_9/lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_18/while/add_1/y�
 sequential_9/lstm_18/while/add_1AddV2Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counter+sequential_9/lstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/while/add_1�
#sequential_9/lstm_18/while/IdentityIdentity$sequential_9/lstm_18/while/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identity�
%sequential_9/lstm_18/while/Identity_1IdentityHsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_1�
%sequential_9/lstm_18/while/Identity_2Identity"sequential_9/lstm_18/while/add:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_2�
%sequential_9/lstm_18/while/Identity_3IdentityOsequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_3�
%sequential_9/lstm_18/while/Identity_4Identity1sequential_9/lstm_18/while/lstm_cell_36/mul_2:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2'
%sequential_9/lstm_18/while/Identity_4�
%sequential_9/lstm_18/while/Identity_5Identity1sequential_9/lstm_18/while/lstm_cell_36/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*'
_output_shapes
:���������@2'
%sequential_9/lstm_18/while/Identity_5�
sequential_9/lstm_18/while/NoOpNoOp?^sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp>^sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp@^sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_9/lstm_18/while/NoOp"S
#sequential_9_lstm_18_while_identity,sequential_9/lstm_18/while/Identity:output:0"W
%sequential_9_lstm_18_while_identity_1.sequential_9/lstm_18/while/Identity_1:output:0"W
%sequential_9_lstm_18_while_identity_2.sequential_9/lstm_18/while/Identity_2:output:0"W
%sequential_9_lstm_18_while_identity_3.sequential_9/lstm_18/while/Identity_3:output:0"W
%sequential_9_lstm_18_while_identity_4.sequential_9/lstm_18/while/Identity_4:output:0"W
%sequential_9_lstm_18_while_identity_5.sequential_9/lstm_18/while/Identity_5:output:0"�
Gsequential_9_lstm_18_while_lstm_cell_36_biasadd_readvariableop_resourceIsequential_9_lstm_18_while_lstm_cell_36_biasadd_readvariableop_resource_0"�
Hsequential_9_lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resourceJsequential_9_lstm_18_while_lstm_cell_36_matmul_1_readvariableop_resource_0"�
Fsequential_9_lstm_18_while_lstm_cell_36_matmul_readvariableop_resourceHsequential_9_lstm_18_while_lstm_cell_36_matmul_readvariableop_resource_0"�
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0"�
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2�
>sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp>sequential_9/lstm_18/while/lstm_cell_36/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp=sequential_9/lstm_18/while/lstm_cell_36/MatMul/ReadVariableOp2�
?sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp?sequential_9/lstm_18/while/lstm_cell_36/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������@:-)
'
_output_shapes
:���������@:

_output_shapes
: :

_output_shapes
: 
�[
�
C__inference_lstm_18_layer_call_and_return_conditional_losses_325327

inputs>
+lstm_cell_36_matmul_readvariableop_resource:	�@
-lstm_cell_36_matmul_1_readvariableop_resource:	@�;
,lstm_cell_36_biasadd_readvariableop_resource:	�
identity��#lstm_cell_36/BiasAdd/ReadVariableOp�"lstm_cell_36/MatMul/ReadVariableOp�$lstm_cell_36/MatMul_1/ReadVariableOp�whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1�
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros_1/packed/1�
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:���������2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2
TensorArrayV2/element_shape�
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2�
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   27
5TensorArrayUnstack/TensorListFromTensor/element_shape�
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
strided_slice_2�
"lstm_cell_36/MatMul/ReadVariableOpReadVariableOp+lstm_cell_36_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_36/MatMul/ReadVariableOp�
lstm_cell_36/MatMulMatMulstrided_slice_2:output:0*lstm_cell_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul�
$lstm_cell_36/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_36_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_36/MatMul_1/ReadVariableOp�
lstm_cell_36/MatMul_1MatMulzeros:output:0,lstm_cell_36/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/MatMul_1�
lstm_cell_36/addAddV2lstm_cell_36/MatMul:product:0lstm_cell_36/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/add�
#lstm_cell_36/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_36/BiasAdd/ReadVariableOp�
lstm_cell_36/BiasAddBiasAddlstm_cell_36/add:z:0+lstm_cell_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_36/BiasAdd~
lstm_cell_36/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_36/split/split_dim�
lstm_cell_36/splitSplit%lstm_cell_36/split/split_dim:output:0lstm_cell_36/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_36/split�
lstm_cell_36/SigmoidSigmoidlstm_cell_36/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid�
lstm_cell_36/Sigmoid_1Sigmoidlstm_cell_36/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_1�
lstm_cell_36/mulMullstm_cell_36/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul}
lstm_cell_36/ReluRelulstm_cell_36/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu�
lstm_cell_36/mul_1Mullstm_cell_36/Sigmoid:y:0lstm_cell_36/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_1�
lstm_cell_36/add_1AddV2lstm_cell_36/mul:z:0lstm_cell_36/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/add_1�
lstm_cell_36/Sigmoid_2Sigmoidlstm_cell_36/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Sigmoid_2|
lstm_cell_36/Relu_1Relulstm_cell_36/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/Relu_1�
lstm_cell_36/mul_2Mullstm_cell_36/Sigmoid_2:y:0!lstm_cell_36/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_36/mul_2�
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2
TensorArrayV2_1/element_shape�
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter�
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_36_matmul_readvariableop_resource-lstm_cell_36_matmul_1_readvariableop_resource,lstm_cell_36_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_325243*
condR
while_cond_325242*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
while�
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   22
0TensorArrayV2Stack/TensorListStack/element_shape�
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02$
"TensorArrayV2Stack/TensorListStack�
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm�
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:���������@2

Identity�
NoOpNoOp$^lstm_cell_36/BiasAdd/ReadVariableOp#^lstm_cell_36/MatMul/ReadVariableOp%^lstm_cell_36/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_36/BiasAdd/ReadVariableOp#lstm_cell_36/BiasAdd/ReadVariableOp2H
"lstm_cell_36/MatMul/ReadVariableOp"lstm_cell_36/MatMul/ReadVariableOp2L
$lstm_cell_36/MatMul_1/ReadVariableOp$lstm_cell_36/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_18_input:
serving_default_lstm_18_input:0���������;
dense_90
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
p_default_save_signature
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�
trainable_variables
regularization_losses
	variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
!iter

"beta_1

#beta_2
	$decay
%learning_ratem`ma&mb'mc(md)me*mf+mgvhvi&vj'vk(vl)vm*vn+vo"
	optimizer
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
�
trainable_variables
regularization_losses
,layer_metrics

-layers
	variables
.metrics
/non_trainable_variables
0layer_regularization_losses
q__call__
p_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
�
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
�
trainable_variables
regularization_losses
6layer_metrics

7layers
	variables
8layer_regularization_losses
9metrics
:non_trainable_variables

;states
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
�
<
state_size

)kernel
*recurrent_kernel
+bias
=trainable_variables
>regularization_losses
?	variables
@	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
�
trainable_variables
regularization_losses
Alayer_metrics

Blayers
	variables
Clayer_regularization_losses
Dmetrics
Enon_trainable_variables

Fstates
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
regularization_losses
Glayer_metrics

Hlayers
	variables
Imetrics
Jnon_trainable_variables
Klayer_regularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 : 2dense_9/kernel
:2dense_9/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
regularization_losses
Llayer_metrics

Mlayers
	variables
Nmetrics
Onon_trainable_variables
Player_regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	�2lstm_18/lstm_cell_36/kernel
8:6	@�2%lstm_18/lstm_cell_36/recurrent_kernel
(:&�2lstm_18/lstm_cell_36/bias
.:,	@�2lstm_19/lstm_cell_37/kernel
8:6	 �2%lstm_19/lstm_cell_37/recurrent_kernel
(:&�2lstm_19/lstm_cell_37/bias
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
�
2trainable_variables
3regularization_losses
Rlayer_metrics

Slayers
4	variables
Tmetrics
Unon_trainable_variables
Vlayer_regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
�
=trainable_variables
>regularization_losses
Wlayer_metrics

Xlayers
?	variables
Ymetrics
Znon_trainable_variables
[layer_regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	\total
	]count
^	variables
_	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
\0
]1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
%:# 2Adam/dense_9/kernel/m
:2Adam/dense_9/bias/m
3:1	�2"Adam/lstm_18/lstm_cell_36/kernel/m
=:;	@�2,Adam/lstm_18/lstm_cell_36/recurrent_kernel/m
-:+�2 Adam/lstm_18/lstm_cell_36/bias/m
3:1	@�2"Adam/lstm_19/lstm_cell_37/kernel/m
=:;	 �2,Adam/lstm_19/lstm_cell_37/recurrent_kernel/m
-:+�2 Adam/lstm_19/lstm_cell_37/bias/m
%:# 2Adam/dense_9/kernel/v
:2Adam/dense_9/bias/v
3:1	�2"Adam/lstm_18/lstm_cell_36/kernel/v
=:;	@�2,Adam/lstm_18/lstm_cell_36/recurrent_kernel/v
-:+�2 Adam/lstm_18/lstm_cell_36/bias/v
3:1	@�2"Adam/lstm_19/lstm_cell_37/kernel/v
=:;	 �2,Adam/lstm_19/lstm_cell_37/recurrent_kernel/v
-:+�2 Adam/lstm_19/lstm_cell_37/bias/v
�B�
!__inference__wrapped_model_322002lstm_18_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_sequential_9_layer_call_fn_323628
-__inference_sequential_9_layer_call_fn_324192
-__inference_sequential_9_layer_call_fn_324213
-__inference_sequential_9_layer_call_fn_324094�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_9_layer_call_and_return_conditional_losses_324518
H__inference_sequential_9_layer_call_and_return_conditional_losses_324830
H__inference_sequential_9_layer_call_and_return_conditional_losses_324118
H__inference_sequential_9_layer_call_and_return_conditional_losses_324142�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_18_layer_call_fn_324841
(__inference_lstm_18_layer_call_fn_324852
(__inference_lstm_18_layer_call_fn_324863
(__inference_lstm_18_layer_call_fn_324874�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_18_layer_call_and_return_conditional_losses_325025
C__inference_lstm_18_layer_call_and_return_conditional_losses_325176
C__inference_lstm_18_layer_call_and_return_conditional_losses_325327
C__inference_lstm_18_layer_call_and_return_conditional_losses_325478�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_lstm_19_layer_call_fn_325489
(__inference_lstm_19_layer_call_fn_325500
(__inference_lstm_19_layer_call_fn_325511
(__inference_lstm_19_layer_call_fn_325522�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_lstm_19_layer_call_and_return_conditional_losses_325673
C__inference_lstm_19_layer_call_and_return_conditional_losses_325824
C__inference_lstm_19_layer_call_and_return_conditional_losses_325975
C__inference_lstm_19_layer_call_and_return_conditional_losses_326126�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
*__inference_dropout_9_layer_call_fn_326131
*__inference_dropout_9_layer_call_fn_326136�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dropout_9_layer_call_and_return_conditional_losses_326141
E__inference_dropout_9_layer_call_and_return_conditional_losses_326153�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dense_9_layer_call_fn_326162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_9_layer_call_and_return_conditional_losses_326172�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_324171lstm_18_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_lstm_cell_36_layer_call_fn_326189
-__inference_lstm_cell_36_layer_call_fn_326206�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326238
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326270�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_lstm_cell_37_layer_call_fn_326287
-__inference_lstm_cell_37_layer_call_fn_326304�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326336
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326368�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 �
!__inference__wrapped_model_322002y&'()*+:�7
0�-
+�(
lstm_18_input���������
� "1�.
,
dense_9!�
dense_9����������
C__inference_dense_9_layer_call_and_return_conditional_losses_326172\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� {
(__inference_dense_9_layer_call_fn_326162O/�,
%�"
 �
inputs��������� 
� "�����������
E__inference_dropout_9_layer_call_and_return_conditional_losses_326141\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
E__inference_dropout_9_layer_call_and_return_conditional_losses_326153\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� }
*__inference_dropout_9_layer_call_fn_326131O3�0
)�&
 �
inputs��������� 
p 
� "���������� }
*__inference_dropout_9_layer_call_fn_326136O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_lstm_18_layer_call_and_return_conditional_losses_325025�&'(O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "2�/
(�%
0������������������@
� �
C__inference_lstm_18_layer_call_and_return_conditional_losses_325176�&'(O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "2�/
(�%
0������������������@
� �
C__inference_lstm_18_layer_call_and_return_conditional_losses_325327q&'(?�<
5�2
$�!
inputs���������

 
p 

 
� ")�&
�
0���������@
� �
C__inference_lstm_18_layer_call_and_return_conditional_losses_325478q&'(?�<
5�2
$�!
inputs���������

 
p

 
� ")�&
�
0���������@
� �
(__inference_lstm_18_layer_call_fn_324841}&'(O�L
E�B
4�1
/�,
inputs/0������������������

 
p 

 
� "%�"������������������@�
(__inference_lstm_18_layer_call_fn_324852}&'(O�L
E�B
4�1
/�,
inputs/0������������������

 
p

 
� "%�"������������������@�
(__inference_lstm_18_layer_call_fn_324863d&'(?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
(__inference_lstm_18_layer_call_fn_324874d&'(?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
C__inference_lstm_19_layer_call_and_return_conditional_losses_325673})*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "%�"
�
0��������� 
� �
C__inference_lstm_19_layer_call_and_return_conditional_losses_325824})*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "%�"
�
0��������� 
� �
C__inference_lstm_19_layer_call_and_return_conditional_losses_325975m)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "%�"
�
0��������� 
� �
C__inference_lstm_19_layer_call_and_return_conditional_losses_326126m)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "%�"
�
0��������� 
� �
(__inference_lstm_19_layer_call_fn_325489p)*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p 

 
� "���������� �
(__inference_lstm_19_layer_call_fn_325500p)*+O�L
E�B
4�1
/�,
inputs/0������������������@

 
p

 
� "���������� �
(__inference_lstm_19_layer_call_fn_325511`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
(__inference_lstm_19_layer_call_fn_325522`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326238�&'(��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
H__inference_lstm_cell_36_layer_call_and_return_conditional_losses_326270�&'(��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "s�p
i�f
�
0/0���������@
E�B
�
0/1/0���������@
�
0/1/1���������@
� �
-__inference_lstm_cell_36_layer_call_fn_326189�&'(��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p 
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
-__inference_lstm_cell_36_layer_call_fn_326206�&'(��}
v�s
 �
inputs���������
K�H
"�
states/0���������@
"�
states/1���������@
p
� "c�`
�
0���������@
A�>
�
1/0���������@
�
1/1���������@�
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326336�)*+��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
H__inference_lstm_cell_37_layer_call_and_return_conditional_losses_326368�)*+��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "s�p
i�f
�
0/0��������� 
E�B
�
0/1/0��������� 
�
0/1/1��������� 
� �
-__inference_lstm_cell_37_layer_call_fn_326287�)*+��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p 
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
-__inference_lstm_cell_37_layer_call_fn_326304�)*+��}
v�s
 �
inputs���������@
K�H
"�
states/0��������� 
"�
states/1��������� 
p
� "c�`
�
0��������� 
A�>
�
1/0��������� 
�
1/1��������� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_324118u&'()*+B�?
8�5
+�(
lstm_18_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_324142u&'()*+B�?
8�5
+�(
lstm_18_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_324518n&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_9_layer_call_and_return_conditional_losses_324830n&'()*+;�8
1�.
$�!
inputs���������
p

 
� "%�"
�
0���������
� �
-__inference_sequential_9_layer_call_fn_323628h&'()*+B�?
8�5
+�(
lstm_18_input���������
p 

 
� "�����������
-__inference_sequential_9_layer_call_fn_324094h&'()*+B�?
8�5
+�(
lstm_18_input���������
p

 
� "�����������
-__inference_sequential_9_layer_call_fn_324192a&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "�����������
-__inference_sequential_9_layer_call_fn_324213a&'()*+;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_324171�&'()*+K�H
� 
A�>
<
lstm_18_input+�(
lstm_18_input���������"1�.
,
dense_9!�
dense_9���������