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
�"serve*2.6.22v2.6.1-9-gc2363d6d0258��$
z
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_22/kernel
s
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel*
_output_shapes

: *
dtype0
r
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_22/bias
k
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
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
lstm_44/lstm_cell_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_44/lstm_cell_88/kernel
�
/lstm_44/lstm_cell_88/kernel/Read/ReadVariableOpReadVariableOplstm_44/lstm_cell_88/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_44/lstm_cell_88/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*6
shared_name'%lstm_44/lstm_cell_88/recurrent_kernel
�
9lstm_44/lstm_cell_88/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_44/lstm_cell_88/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_44/lstm_cell_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_44/lstm_cell_88/bias
�
-lstm_44/lstm_cell_88/bias/Read/ReadVariableOpReadVariableOplstm_44/lstm_cell_88/bias*
_output_shapes	
:�*
dtype0
�
lstm_45/lstm_cell_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*,
shared_namelstm_45/lstm_cell_89/kernel
�
/lstm_45/lstm_cell_89/kernel/Read/ReadVariableOpReadVariableOplstm_45/lstm_cell_89/kernel*
_output_shapes
:	@�*
dtype0
�
%lstm_45/lstm_cell_89/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%lstm_45/lstm_cell_89/recurrent_kernel
�
9lstm_45/lstm_cell_89/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_45/lstm_cell_89/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
lstm_45/lstm_cell_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_45/lstm_cell_89/bias
�
-lstm_45/lstm_cell_89/bias/Read/ReadVariableOpReadVariableOplstm_45/lstm_cell_89/bias*
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
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_22/kernel/m
�
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/m
y
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_44/lstm_cell_88/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_44/lstm_cell_88/kernel/m
�
6Adam/lstm_44/lstm_cell_88/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_44/lstm_cell_88/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_44/lstm_cell_88/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_44/lstm_cell_88/recurrent_kernel/m
�
@Adam/lstm_44/lstm_cell_88/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_44/lstm_cell_88/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_44/lstm_cell_88/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_44/lstm_cell_88/bias/m
�
4Adam/lstm_44/lstm_cell_88/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_44/lstm_cell_88/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_45/lstm_cell_89/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_45/lstm_cell_89/kernel/m
�
6Adam/lstm_45/lstm_cell_89/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_45/lstm_cell_89/kernel/m*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_45/lstm_cell_89/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_45/lstm_cell_89/recurrent_kernel/m
�
@Adam/lstm_45/lstm_cell_89/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_45/lstm_cell_89/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_45/lstm_cell_89/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_45/lstm_cell_89/bias/m
�
4Adam/lstm_45/lstm_cell_89/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_45/lstm_cell_89/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_22/kernel/v
�
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_22/bias/v
y
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_44/lstm_cell_88/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_44/lstm_cell_88/kernel/v
�
6Adam/lstm_44/lstm_cell_88/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_44/lstm_cell_88/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_44/lstm_cell_88/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_44/lstm_cell_88/recurrent_kernel/v
�
@Adam/lstm_44/lstm_cell_88/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_44/lstm_cell_88/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_44/lstm_cell_88/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_44/lstm_cell_88/bias/v
�
4Adam/lstm_44/lstm_cell_88/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_44/lstm_cell_88/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_45/lstm_cell_89/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_45/lstm_cell_89/kernel/v
�
6Adam/lstm_45/lstm_cell_89/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_45/lstm_cell_89/kernel/v*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_45/lstm_cell_89/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_45/lstm_cell_89/recurrent_kernel/v
�
@Adam/lstm_45/lstm_cell_89/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_45/lstm_cell_89/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_45/lstm_cell_89/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_45/lstm_cell_89/bias/v
�
4Adam/lstm_45/lstm_cell_89/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_45/lstm_cell_89/bias/v*
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
[Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_44/lstm_cell_88/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_44/lstm_cell_88/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_44/lstm_cell_88/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_45/lstm_cell_89/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_45/lstm_cell_89/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_45/lstm_cell_89/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
~|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_44/lstm_cell_88/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_44/lstm_cell_88/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_44/lstm_cell_88/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_45/lstm_cell_89/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_45/lstm_cell_89/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_45/lstm_cell_89/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_44/lstm_cell_88/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_44/lstm_cell_88/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_44/lstm_cell_88/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_45/lstm_cell_89/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_45/lstm_cell_89/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_45/lstm_cell_89/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_44_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_44_inputlstm_44/lstm_cell_88/kernel%lstm_44/lstm_cell_88/recurrent_kernellstm_44/lstm_cell_88/biaslstm_45/lstm_cell_89/kernel%lstm_45/lstm_cell_89/recurrent_kernellstm_45/lstm_cell_89/biasdense_22/kerneldense_22/bias*
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
$__inference_signature_wrapper_685558
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_44/lstm_cell_88/kernel/Read/ReadVariableOp9lstm_44/lstm_cell_88/recurrent_kernel/Read/ReadVariableOp-lstm_44/lstm_cell_88/bias/Read/ReadVariableOp/lstm_45/lstm_cell_89/kernel/Read/ReadVariableOp9lstm_45/lstm_cell_89/recurrent_kernel/Read/ReadVariableOp-lstm_45/lstm_cell_89/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp6Adam/lstm_44/lstm_cell_88/kernel/m/Read/ReadVariableOp@Adam/lstm_44/lstm_cell_88/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_44/lstm_cell_88/bias/m/Read/ReadVariableOp6Adam/lstm_45/lstm_cell_89/kernel/m/Read/ReadVariableOp@Adam/lstm_45/lstm_cell_89/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_45/lstm_cell_89/bias/m/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp6Adam/lstm_44/lstm_cell_88/kernel/v/Read/ReadVariableOp@Adam/lstm_44/lstm_cell_88/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_44/lstm_cell_88/bias/v/Read/ReadVariableOp6Adam/lstm_45/lstm_cell_89/kernel/v/Read/ReadVariableOp@Adam/lstm_45/lstm_cell_89/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_45/lstm_cell_89/bias/v/Read/ReadVariableOpConst*,
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
__inference__traced_save_687871
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_22/kerneldense_22/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_44/lstm_cell_88/kernel%lstm_44/lstm_cell_88/recurrent_kernellstm_44/lstm_cell_88/biaslstm_45/lstm_cell_89/kernel%lstm_45/lstm_cell_89/recurrent_kernellstm_45/lstm_cell_89/biastotalcountAdam/dense_22/kernel/mAdam/dense_22/bias/m"Adam/lstm_44/lstm_cell_88/kernel/m,Adam/lstm_44/lstm_cell_88/recurrent_kernel/m Adam/lstm_44/lstm_cell_88/bias/m"Adam/lstm_45/lstm_cell_89/kernel/m,Adam/lstm_45/lstm_cell_89/recurrent_kernel/m Adam/lstm_45/lstm_cell_89/bias/mAdam/dense_22/kernel/vAdam/dense_22/bias/v"Adam/lstm_44/lstm_cell_88/kernel/v,Adam/lstm_44/lstm_cell_88/recurrent_kernel/v Adam/lstm_44/lstm_cell_88/bias/v"Adam/lstm_45/lstm_cell_89/kernel/v,Adam/lstm_45/lstm_cell_89/recurrent_kernel/v Adam/lstm_45/lstm_cell_89/bias/v*+
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
"__inference__traced_restore_687974��#
�[
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_687362

inputs>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_687278*
condR
while_cond_687277*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_687428
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_687428___redundant_placeholder04
0while_while_cond_687428___redundant_placeholder14
0while_while_cond_687428___redundant_placeholder24
0while_while_cond_687428___redundant_placeholder3
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
�[
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_687513

inputs>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_687429*
condR
while_cond_687428*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_684107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684107___redundant_placeholder04
0while_while_cond_684107___redundant_placeholder14
0while_while_cond_684107___redundant_placeholder24
0while_while_cond_684107___redundant_placeholder3
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
��
�
I__inference_sequential_22_layer_call_and_return_conditional_losses_685905

inputsF
3lstm_44_lstm_cell_88_matmul_readvariableop_resource:	�H
5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource:	@�C
4lstm_44_lstm_cell_88_biasadd_readvariableop_resource:	�F
3lstm_45_lstm_cell_89_matmul_readvariableop_resource:	@�H
5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource:	 �C
4lstm_45_lstm_cell_89_biasadd_readvariableop_resource:	�9
'dense_22_matmul_readvariableop_resource: 6
(dense_22_biasadd_readvariableop_resource:
identity��dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�*lstm_44/lstm_cell_88/MatMul/ReadVariableOp�,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�lstm_44/while�+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�*lstm_45/lstm_cell_89/MatMul/ReadVariableOp�,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�lstm_45/whileT
lstm_44/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_44/Shape�
lstm_44/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice/stack�
lstm_44/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_44/strided_slice/stack_1�
lstm_44/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_44/strided_slice/stack_2�
lstm_44/strided_sliceStridedSlicelstm_44/Shape:output:0$lstm_44/strided_slice/stack:output:0&lstm_44/strided_slice/stack_1:output:0&lstm_44/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_44/strided_slicel
lstm_44/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros/mul/y�
lstm_44/zeros/mulMullstm_44/strided_slice:output:0lstm_44/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros/mulo
lstm_44/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_44/zeros/Less/y�
lstm_44/zeros/LessLesslstm_44/zeros/mul:z:0lstm_44/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros/Lessr
lstm_44/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros/packed/1�
lstm_44/zeros/packedPacklstm_44/strided_slice:output:0lstm_44/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_44/zeros/packedo
lstm_44/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/zeros/Const�
lstm_44/zerosFilllstm_44/zeros/packed:output:0lstm_44/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/zerosp
lstm_44/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros_1/mul/y�
lstm_44/zeros_1/mulMullstm_44/strided_slice:output:0lstm_44/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros_1/muls
lstm_44/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_44/zeros_1/Less/y�
lstm_44/zeros_1/LessLesslstm_44/zeros_1/mul:z:0lstm_44/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros_1/Lessv
lstm_44/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros_1/packed/1�
lstm_44/zeros_1/packedPacklstm_44/strided_slice:output:0!lstm_44/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_44/zeros_1/packeds
lstm_44/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/zeros_1/Const�
lstm_44/zeros_1Filllstm_44/zeros_1/packed:output:0lstm_44/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/zeros_1�
lstm_44/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_44/transpose/perm�
lstm_44/transpose	Transposeinputslstm_44/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_44/transposeg
lstm_44/Shape_1Shapelstm_44/transpose:y:0*
T0*
_output_shapes
:2
lstm_44/Shape_1�
lstm_44/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice_1/stack�
lstm_44/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_1/stack_1�
lstm_44/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_1/stack_2�
lstm_44/strided_slice_1StridedSlicelstm_44/Shape_1:output:0&lstm_44/strided_slice_1/stack:output:0(lstm_44/strided_slice_1/stack_1:output:0(lstm_44/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_44/strided_slice_1�
#lstm_44/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_44/TensorArrayV2/element_shape�
lstm_44/TensorArrayV2TensorListReserve,lstm_44/TensorArrayV2/element_shape:output:0 lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_44/TensorArrayV2�
=lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_44/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_44/transpose:y:0Flstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_44/TensorArrayUnstack/TensorListFromTensor�
lstm_44/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice_2/stack�
lstm_44/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_2/stack_1�
lstm_44/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_2/stack_2�
lstm_44/strided_slice_2StridedSlicelstm_44/transpose:y:0&lstm_44/strided_slice_2/stack:output:0(lstm_44/strided_slice_2/stack_1:output:0(lstm_44/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_44/strided_slice_2�
*lstm_44/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3lstm_44_lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_44/lstm_cell_88/MatMul/ReadVariableOp�
lstm_44/lstm_cell_88/MatMulMatMul lstm_44/strided_slice_2:output:02lstm_44/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/MatMul�
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_44/lstm_cell_88/MatMul_1MatMullstm_44/zeros:output:04lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/MatMul_1�
lstm_44/lstm_cell_88/addAddV2%lstm_44/lstm_cell_88/MatMul:product:0'lstm_44/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/add�
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_44/lstm_cell_88/BiasAddBiasAddlstm_44/lstm_cell_88/add:z:03lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/BiasAdd�
$lstm_44/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_44/lstm_cell_88/split/split_dim�
lstm_44/lstm_cell_88/splitSplit-lstm_44/lstm_cell_88/split/split_dim:output:0%lstm_44/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_44/lstm_cell_88/split�
lstm_44/lstm_cell_88/SigmoidSigmoid#lstm_44/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Sigmoid�
lstm_44/lstm_cell_88/Sigmoid_1Sigmoid#lstm_44/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_44/lstm_cell_88/Sigmoid_1�
lstm_44/lstm_cell_88/mulMul"lstm_44/lstm_cell_88/Sigmoid_1:y:0lstm_44/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul�
lstm_44/lstm_cell_88/ReluRelu#lstm_44/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Relu�
lstm_44/lstm_cell_88/mul_1Mul lstm_44/lstm_cell_88/Sigmoid:y:0'lstm_44/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul_1�
lstm_44/lstm_cell_88/add_1AddV2lstm_44/lstm_cell_88/mul:z:0lstm_44/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/add_1�
lstm_44/lstm_cell_88/Sigmoid_2Sigmoid#lstm_44/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_44/lstm_cell_88/Sigmoid_2�
lstm_44/lstm_cell_88/Relu_1Relulstm_44/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Relu_1�
lstm_44/lstm_cell_88/mul_2Mul"lstm_44/lstm_cell_88/Sigmoid_2:y:0)lstm_44/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul_2�
%lstm_44/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_44/TensorArrayV2_1/element_shape�
lstm_44/TensorArrayV2_1TensorListReserve.lstm_44/TensorArrayV2_1/element_shape:output:0 lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_44/TensorArrayV2_1^
lstm_44/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_44/time�
 lstm_44/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_44/while/maximum_iterationsz
lstm_44/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_44/while/loop_counter�
lstm_44/whileWhile#lstm_44/while/loop_counter:output:0)lstm_44/while/maximum_iterations:output:0lstm_44/time:output:0 lstm_44/TensorArrayV2_1:handle:0lstm_44/zeros:output:0lstm_44/zeros_1:output:0 lstm_44/strided_slice_1:output:0?lstm_44/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_44_lstm_cell_88_matmul_readvariableop_resource5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource4lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
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
lstm_44_while_body_685667*%
condR
lstm_44_while_cond_685666*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_44/while�
8lstm_44/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_44/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_44/TensorArrayV2Stack/TensorListStackTensorListStacklstm_44/while:output:3Alstm_44/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_44/TensorArrayV2Stack/TensorListStack�
lstm_44/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_44/strided_slice_3/stack�
lstm_44/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_44/strided_slice_3/stack_1�
lstm_44/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_3/stack_2�
lstm_44/strided_slice_3StridedSlice3lstm_44/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_44/strided_slice_3/stack:output:0(lstm_44/strided_slice_3/stack_1:output:0(lstm_44/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_44/strided_slice_3�
lstm_44/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_44/transpose_1/perm�
lstm_44/transpose_1	Transpose3lstm_44/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_44/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_44/transpose_1v
lstm_44/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/runtimee
lstm_45/ShapeShapelstm_44/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_45/Shape�
lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice/stack�
lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_45/strided_slice/stack_1�
lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_45/strided_slice/stack_2�
lstm_45/strided_sliceStridedSlicelstm_45/Shape:output:0$lstm_45/strided_slice/stack:output:0&lstm_45/strided_slice/stack_1:output:0&lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_45/strided_slicel
lstm_45/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros/mul/y�
lstm_45/zeros/mulMullstm_45/strided_slice:output:0lstm_45/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros/mulo
lstm_45/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_45/zeros/Less/y�
lstm_45/zeros/LessLesslstm_45/zeros/mul:z:0lstm_45/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros/Lessr
lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros/packed/1�
lstm_45/zeros/packedPacklstm_45/strided_slice:output:0lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_45/zeros/packedo
lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/zeros/Const�
lstm_45/zerosFilllstm_45/zeros/packed:output:0lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/zerosp
lstm_45/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros_1/mul/y�
lstm_45/zeros_1/mulMullstm_45/strided_slice:output:0lstm_45/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros_1/muls
lstm_45/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_45/zeros_1/Less/y�
lstm_45/zeros_1/LessLesslstm_45/zeros_1/mul:z:0lstm_45/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros_1/Lessv
lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros_1/packed/1�
lstm_45/zeros_1/packedPacklstm_45/strided_slice:output:0!lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_45/zeros_1/packeds
lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/zeros_1/Const�
lstm_45/zeros_1Filllstm_45/zeros_1/packed:output:0lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/zeros_1�
lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_45/transpose/perm�
lstm_45/transpose	Transposelstm_44/transpose_1:y:0lstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_45/transposeg
lstm_45/Shape_1Shapelstm_45/transpose:y:0*
T0*
_output_shapes
:2
lstm_45/Shape_1�
lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice_1/stack�
lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_1/stack_1�
lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_1/stack_2�
lstm_45/strided_slice_1StridedSlicelstm_45/Shape_1:output:0&lstm_45/strided_slice_1/stack:output:0(lstm_45/strided_slice_1/stack_1:output:0(lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_45/strided_slice_1�
#lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_45/TensorArrayV2/element_shape�
lstm_45/TensorArrayV2TensorListReserve,lstm_45/TensorArrayV2/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_45/TensorArrayV2�
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_45/transpose:y:0Flstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_45/TensorArrayUnstack/TensorListFromTensor�
lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice_2/stack�
lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_2/stack_1�
lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_2/stack_2�
lstm_45/strided_slice_2StridedSlicelstm_45/transpose:y:0&lstm_45/strided_slice_2/stack:output:0(lstm_45/strided_slice_2/stack_1:output:0(lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_45/strided_slice_2�
*lstm_45/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3lstm_45_lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_45/lstm_cell_89/MatMul/ReadVariableOp�
lstm_45/lstm_cell_89/MatMulMatMul lstm_45/strided_slice_2:output:02lstm_45/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/MatMul�
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_45/lstm_cell_89/MatMul_1MatMullstm_45/zeros:output:04lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/MatMul_1�
lstm_45/lstm_cell_89/addAddV2%lstm_45/lstm_cell_89/MatMul:product:0'lstm_45/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/add�
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_45/lstm_cell_89/BiasAddBiasAddlstm_45/lstm_cell_89/add:z:03lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/BiasAdd�
$lstm_45/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_45/lstm_cell_89/split/split_dim�
lstm_45/lstm_cell_89/splitSplit-lstm_45/lstm_cell_89/split/split_dim:output:0%lstm_45/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_45/lstm_cell_89/split�
lstm_45/lstm_cell_89/SigmoidSigmoid#lstm_45/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Sigmoid�
lstm_45/lstm_cell_89/Sigmoid_1Sigmoid#lstm_45/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_45/lstm_cell_89/Sigmoid_1�
lstm_45/lstm_cell_89/mulMul"lstm_45/lstm_cell_89/Sigmoid_1:y:0lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul�
lstm_45/lstm_cell_89/ReluRelu#lstm_45/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Relu�
lstm_45/lstm_cell_89/mul_1Mul lstm_45/lstm_cell_89/Sigmoid:y:0'lstm_45/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul_1�
lstm_45/lstm_cell_89/add_1AddV2lstm_45/lstm_cell_89/mul:z:0lstm_45/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/add_1�
lstm_45/lstm_cell_89/Sigmoid_2Sigmoid#lstm_45/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_45/lstm_cell_89/Sigmoid_2�
lstm_45/lstm_cell_89/Relu_1Relulstm_45/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Relu_1�
lstm_45/lstm_cell_89/mul_2Mul"lstm_45/lstm_cell_89/Sigmoid_2:y:0)lstm_45/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul_2�
%lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_45/TensorArrayV2_1/element_shape�
lstm_45/TensorArrayV2_1TensorListReserve.lstm_45/TensorArrayV2_1/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_45/TensorArrayV2_1^
lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/time�
 lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_45/while/maximum_iterationsz
lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/while/loop_counter�
lstm_45/whileWhile#lstm_45/while/loop_counter:output:0)lstm_45/while/maximum_iterations:output:0lstm_45/time:output:0 lstm_45/TensorArrayV2_1:handle:0lstm_45/zeros:output:0lstm_45/zeros_1:output:0 lstm_45/strided_slice_1:output:0?lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_45_lstm_cell_89_matmul_readvariableop_resource5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource4lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
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
lstm_45_while_body_685814*%
condR
lstm_45_while_cond_685813*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_45/while�
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_45/TensorArrayV2Stack/TensorListStackTensorListStacklstm_45/while:output:3Alstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_45/TensorArrayV2Stack/TensorListStack�
lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_45/strided_slice_3/stack�
lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_45/strided_slice_3/stack_1�
lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_3/stack_2�
lstm_45/strided_slice_3StridedSlice3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_45/strided_slice_3/stack:output:0(lstm_45/strided_slice_3/stack_1:output:0(lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_45/strided_slice_3�
lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_45/transpose_1/perm�
lstm_45/transpose_1	Transpose3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_45/transpose_1v
lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/runtime�
dropout_22/IdentityIdentity lstm_45/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2
dropout_22/Identity�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMuldropout_22/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/BiasAddt
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp,^lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp+^lstm_44/lstm_cell_88/MatMul/ReadVariableOp-^lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp^lstm_44/while,^lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp+^lstm_45/lstm_cell_89/MatMul/ReadVariableOp-^lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp^lstm_45/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2Z
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp2X
*lstm_44/lstm_cell_88/MatMul/ReadVariableOp*lstm_44/lstm_cell_88/MatMul/ReadVariableOp2\
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp2
lstm_44/whilelstm_44/while2Z
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp2X
*lstm_45/lstm_cell_89/MatMul/ReadVariableOp*lstm_45/lstm_cell_89/MatMul/ReadVariableOp2\
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp2
lstm_45/whilelstm_45/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_687127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
while_cond_686629
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_686629___redundant_placeholder04
0while_while_cond_686629___redundant_placeholder14
0while_while_cond_686629___redundant_placeholder24
0while_while_cond_686629___redundant_placeholder3
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
while_body_686781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
�%
�
while_body_683478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_88_683502_0:	�.
while_lstm_cell_88_683504_0:	@�*
while_lstm_cell_88_683506_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_88_683502:	�,
while_lstm_cell_88_683504:	@�(
while_lstm_cell_88_683506:	���*while/lstm_cell_88/StatefulPartitionedCall�
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
*while/lstm_cell_88/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_88_683502_0while_lstm_cell_88_683504_0while_lstm_cell_88_683506_0*
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6834642,
*while/lstm_cell_88/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_88/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_88/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_88/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_88/StatefulPartitionedCall*"
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
while_lstm_cell_88_683502while_lstm_cell_88_683502_0"8
while_lstm_cell_88_683504while_lstm_cell_88_683504_0"8
while_lstm_cell_88_683506while_lstm_cell_88_683506_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_88/StatefulPartitionedCall*while/lstm_cell_88/StatefulPartitionedCall: 
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
�F
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_684177

inputs&
lstm_cell_89_684095:	@�&
lstm_cell_89_684097:	 �"
lstm_cell_89_684099:	�
identity��$lstm_cell_89/StatefulPartitionedCall�whileD
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
$lstm_cell_89/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_89_684095lstm_cell_89_684097lstm_cell_89_684099*
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6840942&
$lstm_cell_89/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_89_684095lstm_cell_89_684097lstm_cell_89_684099*
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
while_body_684108*
condR
while_cond_684107*K
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
NoOpNoOp%^lstm_cell_89/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_89/StatefulPartitionedCall$lstm_cell_89/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�[
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_686714

inputs>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_686630*
condR
while_cond_686629*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�I
�
__inference__traced_save_687871
file_prefix.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_44_lstm_cell_88_kernel_read_readvariableopD
@savev2_lstm_44_lstm_cell_88_recurrent_kernel_read_readvariableop8
4savev2_lstm_44_lstm_cell_88_bias_read_readvariableop:
6savev2_lstm_45_lstm_cell_89_kernel_read_readvariableopD
@savev2_lstm_45_lstm_cell_89_recurrent_kernel_read_readvariableop8
4savev2_lstm_45_lstm_cell_89_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableopA
=savev2_adam_lstm_44_lstm_cell_88_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_44_lstm_cell_88_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_44_lstm_cell_88_bias_m_read_readvariableopA
=savev2_adam_lstm_45_lstm_cell_89_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_45_lstm_cell_89_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_45_lstm_cell_89_bias_m_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableopA
=savev2_adam_lstm_44_lstm_cell_88_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_44_lstm_cell_88_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_44_lstm_cell_88_bias_v_read_readvariableopA
=savev2_adam_lstm_45_lstm_cell_89_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_45_lstm_cell_89_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_45_lstm_cell_89_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_44_lstm_cell_88_kernel_read_readvariableop@savev2_lstm_44_lstm_cell_88_recurrent_kernel_read_readvariableop4savev2_lstm_44_lstm_cell_88_bias_read_readvariableop6savev2_lstm_45_lstm_cell_89_kernel_read_readvariableop@savev2_lstm_45_lstm_cell_89_recurrent_kernel_read_readvariableop4savev2_lstm_45_lstm_cell_89_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop=savev2_adam_lstm_44_lstm_cell_88_kernel_m_read_readvariableopGsavev2_adam_lstm_44_lstm_cell_88_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_44_lstm_cell_88_bias_m_read_readvariableop=savev2_adam_lstm_45_lstm_cell_89_kernel_m_read_readvariableopGsavev2_adam_lstm_45_lstm_cell_89_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_45_lstm_cell_89_bias_m_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop=savev2_adam_lstm_44_lstm_cell_88_kernel_v_read_readvariableopGsavev2_adam_lstm_44_lstm_cell_88_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_44_lstm_cell_88_bias_v_read_readvariableop=savev2_adam_lstm_45_lstm_cell_89_kernel_v_read_readvariableopGsavev2_adam_lstm_45_lstm_cell_89_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_45_lstm_cell_89_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
(__inference_lstm_45_layer_call_fn_686887
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6843872
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
�	
�
$__inference_signature_wrapper_685558
lstm_44_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_6833892
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
_user_specified_namelstm_44_input
�J
�

lstm_44_while_body_685972,
(lstm_44_while_lstm_44_while_loop_counter2
.lstm_44_while_lstm_44_while_maximum_iterations
lstm_44_while_placeholder
lstm_44_while_placeholder_1
lstm_44_while_placeholder_2
lstm_44_while_placeholder_3+
'lstm_44_while_lstm_44_strided_slice_1_0g
clstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0:	�P
=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�K
<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
lstm_44_while_identity
lstm_44_while_identity_1
lstm_44_while_identity_2
lstm_44_while_identity_3
lstm_44_while_identity_4
lstm_44_while_identity_5)
%lstm_44_while_lstm_44_strided_slice_1e
alstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensorL
9lstm_44_while_lstm_cell_88_matmul_readvariableop_resource:	�N
;lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource:	@�I
:lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource:	���1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
?lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_44/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0lstm_44_while_placeholderHlstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_44/while/TensorArrayV2Read/TensorListGetItem�
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�
!lstm_44/while/lstm_cell_88/MatMulMatMul8lstm_44/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_44/while/lstm_cell_88/MatMul�
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
#lstm_44/while/lstm_cell_88/MatMul_1MatMullstm_44_while_placeholder_2:lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_44/while/lstm_cell_88/MatMul_1�
lstm_44/while/lstm_cell_88/addAddV2+lstm_44/while/lstm_cell_88/MatMul:product:0-lstm_44/while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_44/while/lstm_cell_88/add�
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�
"lstm_44/while/lstm_cell_88/BiasAddBiasAdd"lstm_44/while/lstm_cell_88/add:z:09lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_44/while/lstm_cell_88/BiasAdd�
*lstm_44/while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_44/while/lstm_cell_88/split/split_dim�
 lstm_44/while/lstm_cell_88/splitSplit3lstm_44/while/lstm_cell_88/split/split_dim:output:0+lstm_44/while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_44/while/lstm_cell_88/split�
"lstm_44/while/lstm_cell_88/SigmoidSigmoid)lstm_44/while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_44/while/lstm_cell_88/Sigmoid�
$lstm_44/while/lstm_cell_88/Sigmoid_1Sigmoid)lstm_44/while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_44/while/lstm_cell_88/Sigmoid_1�
lstm_44/while/lstm_cell_88/mulMul(lstm_44/while/lstm_cell_88/Sigmoid_1:y:0lstm_44_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_44/while/lstm_cell_88/mul�
lstm_44/while/lstm_cell_88/ReluRelu)lstm_44/while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_44/while/lstm_cell_88/Relu�
 lstm_44/while/lstm_cell_88/mul_1Mul&lstm_44/while/lstm_cell_88/Sigmoid:y:0-lstm_44/while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/mul_1�
 lstm_44/while/lstm_cell_88/add_1AddV2"lstm_44/while/lstm_cell_88/mul:z:0$lstm_44/while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/add_1�
$lstm_44/while/lstm_cell_88/Sigmoid_2Sigmoid)lstm_44/while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_44/while/lstm_cell_88/Sigmoid_2�
!lstm_44/while/lstm_cell_88/Relu_1Relu$lstm_44/while/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_44/while/lstm_cell_88/Relu_1�
 lstm_44/while/lstm_cell_88/mul_2Mul(lstm_44/while/lstm_cell_88/Sigmoid_2:y:0/lstm_44/while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/mul_2�
2lstm_44/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_44_while_placeholder_1lstm_44_while_placeholder$lstm_44/while/lstm_cell_88/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_44/while/TensorArrayV2Write/TensorListSetIteml
lstm_44/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_44/while/add/y�
lstm_44/while/addAddV2lstm_44_while_placeholderlstm_44/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_44/while/addp
lstm_44/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_44/while/add_1/y�
lstm_44/while/add_1AddV2(lstm_44_while_lstm_44_while_loop_counterlstm_44/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_44/while/add_1�
lstm_44/while/IdentityIdentitylstm_44/while/add_1:z:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity�
lstm_44/while/Identity_1Identity.lstm_44_while_lstm_44_while_maximum_iterations^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_1�
lstm_44/while/Identity_2Identitylstm_44/while/add:z:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_2�
lstm_44/while/Identity_3IdentityBlstm_44/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_3�
lstm_44/while/Identity_4Identity$lstm_44/while/lstm_cell_88/mul_2:z:0^lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_44/while/Identity_4�
lstm_44/while/Identity_5Identity$lstm_44/while/lstm_cell_88/add_1:z:0^lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_44/while/Identity_5�
lstm_44/while/NoOpNoOp2^lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp1^lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp3^lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_44/while/NoOp"9
lstm_44_while_identitylstm_44/while/Identity:output:0"=
lstm_44_while_identity_1!lstm_44/while/Identity_1:output:0"=
lstm_44_while_identity_2!lstm_44/while/Identity_2:output:0"=
lstm_44_while_identity_3!lstm_44/while/Identity_3:output:0"=
lstm_44_while_identity_4!lstm_44/while/Identity_4:output:0"=
lstm_44_while_identity_5!lstm_44/while/Identity_5:output:0"P
%lstm_44_while_lstm_44_strided_slice_1'lstm_44_while_lstm_44_strided_slice_1_0"z
:lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0"|
;lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0"x
9lstm_44_while_lstm_cell_88_matmul_readvariableop_resource;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0"�
alstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensorclstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp2d
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp2h
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
while_cond_684879
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684879___redundant_placeholder04
0while_while_cond_684879___redundant_placeholder14
0while_while_cond_684879___redundant_placeholder24
0while_while_cond_684879___redundant_placeholder3
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
while_cond_686327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_686327___redundant_placeholder04
0while_while_cond_686327___redundant_placeholder14
0while_while_cond_686327___redundant_placeholder24
0while_while_cond_686327___redundant_placeholder3
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
�
�
)__inference_dense_22_layer_call_fn_687549

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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_6849892
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
�
�
'sequential_22_lstm_45_while_cond_683297H
Dsequential_22_lstm_45_while_sequential_22_lstm_45_while_loop_counterN
Jsequential_22_lstm_45_while_sequential_22_lstm_45_while_maximum_iterations+
'sequential_22_lstm_45_while_placeholder-
)sequential_22_lstm_45_while_placeholder_1-
)sequential_22_lstm_45_while_placeholder_2-
)sequential_22_lstm_45_while_placeholder_3J
Fsequential_22_lstm_45_while_less_sequential_22_lstm_45_strided_slice_1`
\sequential_22_lstm_45_while_sequential_22_lstm_45_while_cond_683297___redundant_placeholder0`
\sequential_22_lstm_45_while_sequential_22_lstm_45_while_cond_683297___redundant_placeholder1`
\sequential_22_lstm_45_while_sequential_22_lstm_45_while_cond_683297___redundant_placeholder2`
\sequential_22_lstm_45_while_sequential_22_lstm_45_while_cond_683297___redundant_placeholder3(
$sequential_22_lstm_45_while_identity
�
 sequential_22/lstm_45/while/LessLess'sequential_22_lstm_45_while_placeholderFsequential_22_lstm_45_while_less_sequential_22_lstm_45_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_22/lstm_45/while/Less�
$sequential_22/lstm_45/while/IdentityIdentity$sequential_22/lstm_45/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_22/lstm_45/while/Identity"U
$sequential_22_lstm_45_while_identity-sequential_22/lstm_45/while/Identity:output:0*(
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
while_body_684880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_44_layer_call_fn_686228
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6835472
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
�
�
while_cond_685300
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_685300___redundant_placeholder04
0while_while_cond_685300___redundant_placeholder14
0while_while_cond_685300___redundant_placeholder24
0while_while_cond_685300___redundant_placeholder3
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
�
d
+__inference_dropout_22_layer_call_fn_687523

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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6850452
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
�
�
while_cond_685127
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_685127___redundant_placeholder04
0while_while_cond_685127___redundant_placeholder14
0while_while_cond_685127___redundant_placeholder24
0while_while_cond_685127___redundant_placeholder3
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
��
�
I__inference_sequential_22_layer_call_and_return_conditional_losses_686217

inputsF
3lstm_44_lstm_cell_88_matmul_readvariableop_resource:	�H
5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource:	@�C
4lstm_44_lstm_cell_88_biasadd_readvariableop_resource:	�F
3lstm_45_lstm_cell_89_matmul_readvariableop_resource:	@�H
5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource:	 �C
4lstm_45_lstm_cell_89_biasadd_readvariableop_resource:	�9
'dense_22_matmul_readvariableop_resource: 6
(dense_22_biasadd_readvariableop_resource:
identity��dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�*lstm_44/lstm_cell_88/MatMul/ReadVariableOp�,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�lstm_44/while�+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�*lstm_45/lstm_cell_89/MatMul/ReadVariableOp�,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�lstm_45/whileT
lstm_44/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_44/Shape�
lstm_44/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice/stack�
lstm_44/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_44/strided_slice/stack_1�
lstm_44/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_44/strided_slice/stack_2�
lstm_44/strided_sliceStridedSlicelstm_44/Shape:output:0$lstm_44/strided_slice/stack:output:0&lstm_44/strided_slice/stack_1:output:0&lstm_44/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_44/strided_slicel
lstm_44/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros/mul/y�
lstm_44/zeros/mulMullstm_44/strided_slice:output:0lstm_44/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros/mulo
lstm_44/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_44/zeros/Less/y�
lstm_44/zeros/LessLesslstm_44/zeros/mul:z:0lstm_44/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros/Lessr
lstm_44/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros/packed/1�
lstm_44/zeros/packedPacklstm_44/strided_slice:output:0lstm_44/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_44/zeros/packedo
lstm_44/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/zeros/Const�
lstm_44/zerosFilllstm_44/zeros/packed:output:0lstm_44/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/zerosp
lstm_44/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros_1/mul/y�
lstm_44/zeros_1/mulMullstm_44/strided_slice:output:0lstm_44/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros_1/muls
lstm_44/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_44/zeros_1/Less/y�
lstm_44/zeros_1/LessLesslstm_44/zeros_1/mul:z:0lstm_44/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_44/zeros_1/Lessv
lstm_44/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_44/zeros_1/packed/1�
lstm_44/zeros_1/packedPacklstm_44/strided_slice:output:0!lstm_44/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_44/zeros_1/packeds
lstm_44/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/zeros_1/Const�
lstm_44/zeros_1Filllstm_44/zeros_1/packed:output:0lstm_44/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/zeros_1�
lstm_44/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_44/transpose/perm�
lstm_44/transpose	Transposeinputslstm_44/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_44/transposeg
lstm_44/Shape_1Shapelstm_44/transpose:y:0*
T0*
_output_shapes
:2
lstm_44/Shape_1�
lstm_44/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice_1/stack�
lstm_44/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_1/stack_1�
lstm_44/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_1/stack_2�
lstm_44/strided_slice_1StridedSlicelstm_44/Shape_1:output:0&lstm_44/strided_slice_1/stack:output:0(lstm_44/strided_slice_1/stack_1:output:0(lstm_44/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_44/strided_slice_1�
#lstm_44/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_44/TensorArrayV2/element_shape�
lstm_44/TensorArrayV2TensorListReserve,lstm_44/TensorArrayV2/element_shape:output:0 lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_44/TensorArrayV2�
=lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_44/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_44/transpose:y:0Flstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_44/TensorArrayUnstack/TensorListFromTensor�
lstm_44/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_44/strided_slice_2/stack�
lstm_44/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_2/stack_1�
lstm_44/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_2/stack_2�
lstm_44/strided_slice_2StridedSlicelstm_44/transpose:y:0&lstm_44/strided_slice_2/stack:output:0(lstm_44/strided_slice_2/stack_1:output:0(lstm_44/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_44/strided_slice_2�
*lstm_44/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3lstm_44_lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_44/lstm_cell_88/MatMul/ReadVariableOp�
lstm_44/lstm_cell_88/MatMulMatMul lstm_44/strided_slice_2:output:02lstm_44/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/MatMul�
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_44/lstm_cell_88/MatMul_1MatMullstm_44/zeros:output:04lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/MatMul_1�
lstm_44/lstm_cell_88/addAddV2%lstm_44/lstm_cell_88/MatMul:product:0'lstm_44/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/add�
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_44/lstm_cell_88/BiasAddBiasAddlstm_44/lstm_cell_88/add:z:03lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_44/lstm_cell_88/BiasAdd�
$lstm_44/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_44/lstm_cell_88/split/split_dim�
lstm_44/lstm_cell_88/splitSplit-lstm_44/lstm_cell_88/split/split_dim:output:0%lstm_44/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_44/lstm_cell_88/split�
lstm_44/lstm_cell_88/SigmoidSigmoid#lstm_44/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Sigmoid�
lstm_44/lstm_cell_88/Sigmoid_1Sigmoid#lstm_44/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_44/lstm_cell_88/Sigmoid_1�
lstm_44/lstm_cell_88/mulMul"lstm_44/lstm_cell_88/Sigmoid_1:y:0lstm_44/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul�
lstm_44/lstm_cell_88/ReluRelu#lstm_44/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Relu�
lstm_44/lstm_cell_88/mul_1Mul lstm_44/lstm_cell_88/Sigmoid:y:0'lstm_44/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul_1�
lstm_44/lstm_cell_88/add_1AddV2lstm_44/lstm_cell_88/mul:z:0lstm_44/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/add_1�
lstm_44/lstm_cell_88/Sigmoid_2Sigmoid#lstm_44/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_44/lstm_cell_88/Sigmoid_2�
lstm_44/lstm_cell_88/Relu_1Relulstm_44/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/Relu_1�
lstm_44/lstm_cell_88/mul_2Mul"lstm_44/lstm_cell_88/Sigmoid_2:y:0)lstm_44/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_44/lstm_cell_88/mul_2�
%lstm_44/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_44/TensorArrayV2_1/element_shape�
lstm_44/TensorArrayV2_1TensorListReserve.lstm_44/TensorArrayV2_1/element_shape:output:0 lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_44/TensorArrayV2_1^
lstm_44/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_44/time�
 lstm_44/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_44/while/maximum_iterationsz
lstm_44/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_44/while/loop_counter�
lstm_44/whileWhile#lstm_44/while/loop_counter:output:0)lstm_44/while/maximum_iterations:output:0lstm_44/time:output:0 lstm_44/TensorArrayV2_1:handle:0lstm_44/zeros:output:0lstm_44/zeros_1:output:0 lstm_44/strided_slice_1:output:0?lstm_44/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_44_lstm_cell_88_matmul_readvariableop_resource5lstm_44_lstm_cell_88_matmul_1_readvariableop_resource4lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
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
lstm_44_while_body_685972*%
condR
lstm_44_while_cond_685971*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_44/while�
8lstm_44/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_44/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_44/TensorArrayV2Stack/TensorListStackTensorListStacklstm_44/while:output:3Alstm_44/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_44/TensorArrayV2Stack/TensorListStack�
lstm_44/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_44/strided_slice_3/stack�
lstm_44/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_44/strided_slice_3/stack_1�
lstm_44/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_44/strided_slice_3/stack_2�
lstm_44/strided_slice_3StridedSlice3lstm_44/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_44/strided_slice_3/stack:output:0(lstm_44/strided_slice_3/stack_1:output:0(lstm_44/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_44/strided_slice_3�
lstm_44/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_44/transpose_1/perm�
lstm_44/transpose_1	Transpose3lstm_44/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_44/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_44/transpose_1v
lstm_44/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_44/runtimee
lstm_45/ShapeShapelstm_44/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_45/Shape�
lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice/stack�
lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_45/strided_slice/stack_1�
lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_45/strided_slice/stack_2�
lstm_45/strided_sliceStridedSlicelstm_45/Shape:output:0$lstm_45/strided_slice/stack:output:0&lstm_45/strided_slice/stack_1:output:0&lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_45/strided_slicel
lstm_45/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros/mul/y�
lstm_45/zeros/mulMullstm_45/strided_slice:output:0lstm_45/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros/mulo
lstm_45/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_45/zeros/Less/y�
lstm_45/zeros/LessLesslstm_45/zeros/mul:z:0lstm_45/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros/Lessr
lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros/packed/1�
lstm_45/zeros/packedPacklstm_45/strided_slice:output:0lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_45/zeros/packedo
lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/zeros/Const�
lstm_45/zerosFilllstm_45/zeros/packed:output:0lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/zerosp
lstm_45/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros_1/mul/y�
lstm_45/zeros_1/mulMullstm_45/strided_slice:output:0lstm_45/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros_1/muls
lstm_45/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_45/zeros_1/Less/y�
lstm_45/zeros_1/LessLesslstm_45/zeros_1/mul:z:0lstm_45/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_45/zeros_1/Lessv
lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/zeros_1/packed/1�
lstm_45/zeros_1/packedPacklstm_45/strided_slice:output:0!lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_45/zeros_1/packeds
lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/zeros_1/Const�
lstm_45/zeros_1Filllstm_45/zeros_1/packed:output:0lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/zeros_1�
lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_45/transpose/perm�
lstm_45/transpose	Transposelstm_44/transpose_1:y:0lstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_45/transposeg
lstm_45/Shape_1Shapelstm_45/transpose:y:0*
T0*
_output_shapes
:2
lstm_45/Shape_1�
lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice_1/stack�
lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_1/stack_1�
lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_1/stack_2�
lstm_45/strided_slice_1StridedSlicelstm_45/Shape_1:output:0&lstm_45/strided_slice_1/stack:output:0(lstm_45/strided_slice_1/stack_1:output:0(lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_45/strided_slice_1�
#lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_45/TensorArrayV2/element_shape�
lstm_45/TensorArrayV2TensorListReserve,lstm_45/TensorArrayV2/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_45/TensorArrayV2�
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_45/transpose:y:0Flstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_45/TensorArrayUnstack/TensorListFromTensor�
lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_45/strided_slice_2/stack�
lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_2/stack_1�
lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_2/stack_2�
lstm_45/strided_slice_2StridedSlicelstm_45/transpose:y:0&lstm_45/strided_slice_2/stack:output:0(lstm_45/strided_slice_2/stack_1:output:0(lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_45/strided_slice_2�
*lstm_45/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3lstm_45_lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_45/lstm_cell_89/MatMul/ReadVariableOp�
lstm_45/lstm_cell_89/MatMulMatMul lstm_45/strided_slice_2:output:02lstm_45/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/MatMul�
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_45/lstm_cell_89/MatMul_1MatMullstm_45/zeros:output:04lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/MatMul_1�
lstm_45/lstm_cell_89/addAddV2%lstm_45/lstm_cell_89/MatMul:product:0'lstm_45/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/add�
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_45/lstm_cell_89/BiasAddBiasAddlstm_45/lstm_cell_89/add:z:03lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_45/lstm_cell_89/BiasAdd�
$lstm_45/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_45/lstm_cell_89/split/split_dim�
lstm_45/lstm_cell_89/splitSplit-lstm_45/lstm_cell_89/split/split_dim:output:0%lstm_45/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_45/lstm_cell_89/split�
lstm_45/lstm_cell_89/SigmoidSigmoid#lstm_45/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Sigmoid�
lstm_45/lstm_cell_89/Sigmoid_1Sigmoid#lstm_45/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_45/lstm_cell_89/Sigmoid_1�
lstm_45/lstm_cell_89/mulMul"lstm_45/lstm_cell_89/Sigmoid_1:y:0lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul�
lstm_45/lstm_cell_89/ReluRelu#lstm_45/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Relu�
lstm_45/lstm_cell_89/mul_1Mul lstm_45/lstm_cell_89/Sigmoid:y:0'lstm_45/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul_1�
lstm_45/lstm_cell_89/add_1AddV2lstm_45/lstm_cell_89/mul:z:0lstm_45/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/add_1�
lstm_45/lstm_cell_89/Sigmoid_2Sigmoid#lstm_45/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_45/lstm_cell_89/Sigmoid_2�
lstm_45/lstm_cell_89/Relu_1Relulstm_45/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/Relu_1�
lstm_45/lstm_cell_89/mul_2Mul"lstm_45/lstm_cell_89/Sigmoid_2:y:0)lstm_45/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_45/lstm_cell_89/mul_2�
%lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_45/TensorArrayV2_1/element_shape�
lstm_45/TensorArrayV2_1TensorListReserve.lstm_45/TensorArrayV2_1/element_shape:output:0 lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_45/TensorArrayV2_1^
lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/time�
 lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_45/while/maximum_iterationsz
lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_45/while/loop_counter�
lstm_45/whileWhile#lstm_45/while/loop_counter:output:0)lstm_45/while/maximum_iterations:output:0lstm_45/time:output:0 lstm_45/TensorArrayV2_1:handle:0lstm_45/zeros:output:0lstm_45/zeros_1:output:0 lstm_45/strided_slice_1:output:0?lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_45_lstm_cell_89_matmul_readvariableop_resource5lstm_45_lstm_cell_89_matmul_1_readvariableop_resource4lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
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
lstm_45_while_body_686119*%
condR
lstm_45_while_cond_686118*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_45/while�
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_45/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_45/TensorArrayV2Stack/TensorListStackTensorListStacklstm_45/while:output:3Alstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_45/TensorArrayV2Stack/TensorListStack�
lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_45/strided_slice_3/stack�
lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_45/strided_slice_3/stack_1�
lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_45/strided_slice_3/stack_2�
lstm_45/strided_slice_3StridedSlice3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_45/strided_slice_3/stack:output:0(lstm_45/strided_slice_3/stack_1:output:0(lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_45/strided_slice_3�
lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_45/transpose_1/perm�
lstm_45/transpose_1	Transpose3lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_45/transpose_1v
lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_45/runtimey
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_22/dropout/Const�
dropout_22/dropout/MulMul lstm_45/strided_slice_3:output:0!dropout_22/dropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout_22/dropout/Mul�
dropout_22/dropout/ShapeShape lstm_45/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape�
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype021
/dropout_22/dropout/random_uniform/RandomUniform�
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2#
!dropout_22/dropout/GreaterEqual/y�
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2!
dropout_22/dropout/GreaterEqual�
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout_22/dropout/Cast�
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout_22/dropout/Mul_1�
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_22/MatMul/ReadVariableOp�
dense_22/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/MatMul�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_22/BiasAdd/ReadVariableOp�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_22/BiasAddt
IdentityIdentitydense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp,^lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp+^lstm_44/lstm_cell_88/MatMul/ReadVariableOp-^lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp^lstm_44/while,^lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp+^lstm_45/lstm_cell_89/MatMul/ReadVariableOp-^lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp^lstm_45/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2Z
+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp+lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp2X
*lstm_44/lstm_cell_88/MatMul/ReadVariableOp*lstm_44/lstm_cell_88/MatMul/ReadVariableOp2\
,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp,lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp2
lstm_44/whilelstm_44/while2Z
+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp+lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp2X
*lstm_45/lstm_cell_89/MatMul/ReadVariableOp*lstm_45/lstm_cell_89/MatMul/ReadVariableOp2\
,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp,lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp2
lstm_45/whilelstm_45/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�[
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_684806

inputs>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_684722*
condR
while_cond_684721*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687723

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
�
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_685045

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
�%
�
while_body_684318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_89_684342_0:	@�.
while_lstm_cell_89_684344_0:	 �*
while_lstm_cell_89_684346_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_89_684342:	@�,
while_lstm_cell_89_684344:	 �(
while_lstm_cell_89_684346:	���*while/lstm_cell_89/StatefulPartitionedCall�
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
*while/lstm_cell_89/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_89_684342_0while_lstm_cell_89_684344_0while_lstm_cell_89_684346_0*
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6842402,
*while/lstm_cell_89/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_89/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_89/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_89/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_89/StatefulPartitionedCall*"
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
while_lstm_cell_89_684342while_lstm_cell_89_684342_0"8
while_lstm_cell_89_684344while_lstm_cell_89_684344_0"8
while_lstm_cell_89_684346while_lstm_cell_89_684346_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_89/StatefulPartitionedCall*while/lstm_cell_89/StatefulPartitionedCall: 
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
ԓ
�	
!__inference__wrapped_model_683389
lstm_44_inputT
Asequential_22_lstm_44_lstm_cell_88_matmul_readvariableop_resource:	�V
Csequential_22_lstm_44_lstm_cell_88_matmul_1_readvariableop_resource:	@�Q
Bsequential_22_lstm_44_lstm_cell_88_biasadd_readvariableop_resource:	�T
Asequential_22_lstm_45_lstm_cell_89_matmul_readvariableop_resource:	@�V
Csequential_22_lstm_45_lstm_cell_89_matmul_1_readvariableop_resource:	 �Q
Bsequential_22_lstm_45_lstm_cell_89_biasadd_readvariableop_resource:	�G
5sequential_22_dense_22_matmul_readvariableop_resource: D
6sequential_22_dense_22_biasadd_readvariableop_resource:
identity��-sequential_22/dense_22/BiasAdd/ReadVariableOp�,sequential_22/dense_22/MatMul/ReadVariableOp�9sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�8sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp�:sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�sequential_22/lstm_44/while�9sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�8sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp�:sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�sequential_22/lstm_45/whilew
sequential_22/lstm_44/ShapeShapelstm_44_input*
T0*
_output_shapes
:2
sequential_22/lstm_44/Shape�
)sequential_22/lstm_44/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_22/lstm_44/strided_slice/stack�
+sequential_22/lstm_44/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_22/lstm_44/strided_slice/stack_1�
+sequential_22/lstm_44/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_22/lstm_44/strided_slice/stack_2�
#sequential_22/lstm_44/strided_sliceStridedSlice$sequential_22/lstm_44/Shape:output:02sequential_22/lstm_44/strided_slice/stack:output:04sequential_22/lstm_44/strided_slice/stack_1:output:04sequential_22/lstm_44/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_22/lstm_44/strided_slice�
!sequential_22/lstm_44/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_22/lstm_44/zeros/mul/y�
sequential_22/lstm_44/zeros/mulMul,sequential_22/lstm_44/strided_slice:output:0*sequential_22/lstm_44/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_22/lstm_44/zeros/mul�
"sequential_22/lstm_44/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_22/lstm_44/zeros/Less/y�
 sequential_22/lstm_44/zeros/LessLess#sequential_22/lstm_44/zeros/mul:z:0+sequential_22/lstm_44/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_22/lstm_44/zeros/Less�
$sequential_22/lstm_44/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_22/lstm_44/zeros/packed/1�
"sequential_22/lstm_44/zeros/packedPack,sequential_22/lstm_44/strided_slice:output:0-sequential_22/lstm_44/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_22/lstm_44/zeros/packed�
!sequential_22/lstm_44/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_22/lstm_44/zeros/Const�
sequential_22/lstm_44/zerosFill+sequential_22/lstm_44/zeros/packed:output:0*sequential_22/lstm_44/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_22/lstm_44/zeros�
#sequential_22/lstm_44/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_22/lstm_44/zeros_1/mul/y�
!sequential_22/lstm_44/zeros_1/mulMul,sequential_22/lstm_44/strided_slice:output:0,sequential_22/lstm_44/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_22/lstm_44/zeros_1/mul�
$sequential_22/lstm_44/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_22/lstm_44/zeros_1/Less/y�
"sequential_22/lstm_44/zeros_1/LessLess%sequential_22/lstm_44/zeros_1/mul:z:0-sequential_22/lstm_44/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_22/lstm_44/zeros_1/Less�
&sequential_22/lstm_44/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2(
&sequential_22/lstm_44/zeros_1/packed/1�
$sequential_22/lstm_44/zeros_1/packedPack,sequential_22/lstm_44/strided_slice:output:0/sequential_22/lstm_44/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_22/lstm_44/zeros_1/packed�
#sequential_22/lstm_44/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_22/lstm_44/zeros_1/Const�
sequential_22/lstm_44/zeros_1Fill-sequential_22/lstm_44/zeros_1/packed:output:0,sequential_22/lstm_44/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_22/lstm_44/zeros_1�
$sequential_22/lstm_44/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_22/lstm_44/transpose/perm�
sequential_22/lstm_44/transpose	Transposelstm_44_input-sequential_22/lstm_44/transpose/perm:output:0*
T0*+
_output_shapes
:���������2!
sequential_22/lstm_44/transpose�
sequential_22/lstm_44/Shape_1Shape#sequential_22/lstm_44/transpose:y:0*
T0*
_output_shapes
:2
sequential_22/lstm_44/Shape_1�
+sequential_22/lstm_44/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_22/lstm_44/strided_slice_1/stack�
-sequential_22/lstm_44/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_44/strided_slice_1/stack_1�
-sequential_22/lstm_44/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_44/strided_slice_1/stack_2�
%sequential_22/lstm_44/strided_slice_1StridedSlice&sequential_22/lstm_44/Shape_1:output:04sequential_22/lstm_44/strided_slice_1/stack:output:06sequential_22/lstm_44/strided_slice_1/stack_1:output:06sequential_22/lstm_44/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_22/lstm_44/strided_slice_1�
1sequential_22/lstm_44/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_22/lstm_44/TensorArrayV2/element_shape�
#sequential_22/lstm_44/TensorArrayV2TensorListReserve:sequential_22/lstm_44/TensorArrayV2/element_shape:output:0.sequential_22/lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_22/lstm_44/TensorArrayV2�
Ksequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2M
Ksequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_22/lstm_44/transpose:y:0Tsequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensor�
+sequential_22/lstm_44/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_22/lstm_44/strided_slice_2/stack�
-sequential_22/lstm_44/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_44/strided_slice_2/stack_1�
-sequential_22/lstm_44/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_44/strided_slice_2/stack_2�
%sequential_22/lstm_44/strided_slice_2StridedSlice#sequential_22/lstm_44/transpose:y:04sequential_22/lstm_44/strided_slice_2/stack:output:06sequential_22/lstm_44/strided_slice_2/stack_1:output:06sequential_22/lstm_44/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2'
%sequential_22/lstm_44/strided_slice_2�
8sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOpReadVariableOpAsequential_22_lstm_44_lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02:
8sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp�
)sequential_22/lstm_44/lstm_cell_88/MatMulMatMul.sequential_22/lstm_44/strided_slice_2:output:0@sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_22/lstm_44/lstm_cell_88/MatMul�
:sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOpCsequential_22_lstm_44_lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02<
:sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp�
+sequential_22/lstm_44/lstm_cell_88/MatMul_1MatMul$sequential_22/lstm_44/zeros:output:0Bsequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_22/lstm_44/lstm_cell_88/MatMul_1�
&sequential_22/lstm_44/lstm_cell_88/addAddV23sequential_22/lstm_44/lstm_cell_88/MatMul:product:05sequential_22/lstm_44/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_22/lstm_44/lstm_cell_88/add�
9sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOpBsequential_22_lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp�
*sequential_22/lstm_44/lstm_cell_88/BiasAddBiasAdd*sequential_22/lstm_44/lstm_cell_88/add:z:0Asequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_22/lstm_44/lstm_cell_88/BiasAdd�
2sequential_22/lstm_44/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_22/lstm_44/lstm_cell_88/split/split_dim�
(sequential_22/lstm_44/lstm_cell_88/splitSplit;sequential_22/lstm_44/lstm_cell_88/split/split_dim:output:03sequential_22/lstm_44/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2*
(sequential_22/lstm_44/lstm_cell_88/split�
*sequential_22/lstm_44/lstm_cell_88/SigmoidSigmoid1sequential_22/lstm_44/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2,
*sequential_22/lstm_44/lstm_cell_88/Sigmoid�
,sequential_22/lstm_44/lstm_cell_88/Sigmoid_1Sigmoid1sequential_22/lstm_44/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2.
,sequential_22/lstm_44/lstm_cell_88/Sigmoid_1�
&sequential_22/lstm_44/lstm_cell_88/mulMul0sequential_22/lstm_44/lstm_cell_88/Sigmoid_1:y:0&sequential_22/lstm_44/zeros_1:output:0*
T0*'
_output_shapes
:���������@2(
&sequential_22/lstm_44/lstm_cell_88/mul�
'sequential_22/lstm_44/lstm_cell_88/ReluRelu1sequential_22/lstm_44/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2)
'sequential_22/lstm_44/lstm_cell_88/Relu�
(sequential_22/lstm_44/lstm_cell_88/mul_1Mul.sequential_22/lstm_44/lstm_cell_88/Sigmoid:y:05sequential_22/lstm_44/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_22/lstm_44/lstm_cell_88/mul_1�
(sequential_22/lstm_44/lstm_cell_88/add_1AddV2*sequential_22/lstm_44/lstm_cell_88/mul:z:0,sequential_22/lstm_44/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2*
(sequential_22/lstm_44/lstm_cell_88/add_1�
,sequential_22/lstm_44/lstm_cell_88/Sigmoid_2Sigmoid1sequential_22/lstm_44/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2.
,sequential_22/lstm_44/lstm_cell_88/Sigmoid_2�
)sequential_22/lstm_44/lstm_cell_88/Relu_1Relu,sequential_22/lstm_44/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2+
)sequential_22/lstm_44/lstm_cell_88/Relu_1�
(sequential_22/lstm_44/lstm_cell_88/mul_2Mul0sequential_22/lstm_44/lstm_cell_88/Sigmoid_2:y:07sequential_22/lstm_44/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_22/lstm_44/lstm_cell_88/mul_2�
3sequential_22/lstm_44/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   25
3sequential_22/lstm_44/TensorArrayV2_1/element_shape�
%sequential_22/lstm_44/TensorArrayV2_1TensorListReserve<sequential_22/lstm_44/TensorArrayV2_1/element_shape:output:0.sequential_22/lstm_44/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_22/lstm_44/TensorArrayV2_1z
sequential_22/lstm_44/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_22/lstm_44/time�
.sequential_22/lstm_44/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_22/lstm_44/while/maximum_iterations�
(sequential_22/lstm_44/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_22/lstm_44/while/loop_counter�
sequential_22/lstm_44/whileWhile1sequential_22/lstm_44/while/loop_counter:output:07sequential_22/lstm_44/while/maximum_iterations:output:0#sequential_22/lstm_44/time:output:0.sequential_22/lstm_44/TensorArrayV2_1:handle:0$sequential_22/lstm_44/zeros:output:0&sequential_22/lstm_44/zeros_1:output:0.sequential_22/lstm_44/strided_slice_1:output:0Msequential_22/lstm_44/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_22_lstm_44_lstm_cell_88_matmul_readvariableop_resourceCsequential_22_lstm_44_lstm_cell_88_matmul_1_readvariableop_resourceBsequential_22_lstm_44_lstm_cell_88_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������@:���������@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_22_lstm_44_while_body_683151*3
cond+R)
'sequential_22_lstm_44_while_cond_683150*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
sequential_22/lstm_44/while�
Fsequential_22/lstm_44/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2H
Fsequential_22/lstm_44/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_22/lstm_44/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_22/lstm_44/while:output:3Osequential_22/lstm_44/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02:
8sequential_22/lstm_44/TensorArrayV2Stack/TensorListStack�
+sequential_22/lstm_44/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_22/lstm_44/strided_slice_3/stack�
-sequential_22/lstm_44/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_22/lstm_44/strided_slice_3/stack_1�
-sequential_22/lstm_44/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_44/strided_slice_3/stack_2�
%sequential_22/lstm_44/strided_slice_3StridedSliceAsequential_22/lstm_44/TensorArrayV2Stack/TensorListStack:tensor:04sequential_22/lstm_44/strided_slice_3/stack:output:06sequential_22/lstm_44/strided_slice_3/stack_1:output:06sequential_22/lstm_44/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_22/lstm_44/strided_slice_3�
&sequential_22/lstm_44/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_22/lstm_44/transpose_1/perm�
!sequential_22/lstm_44/transpose_1	TransposeAsequential_22/lstm_44/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_22/lstm_44/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2#
!sequential_22/lstm_44/transpose_1�
sequential_22/lstm_44/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_22/lstm_44/runtime�
sequential_22/lstm_45/ShapeShape%sequential_22/lstm_44/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_22/lstm_45/Shape�
)sequential_22/lstm_45/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_22/lstm_45/strided_slice/stack�
+sequential_22/lstm_45/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_22/lstm_45/strided_slice/stack_1�
+sequential_22/lstm_45/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_22/lstm_45/strided_slice/stack_2�
#sequential_22/lstm_45/strided_sliceStridedSlice$sequential_22/lstm_45/Shape:output:02sequential_22/lstm_45/strided_slice/stack:output:04sequential_22/lstm_45/strided_slice/stack_1:output:04sequential_22/lstm_45/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_22/lstm_45/strided_slice�
!sequential_22/lstm_45/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_22/lstm_45/zeros/mul/y�
sequential_22/lstm_45/zeros/mulMul,sequential_22/lstm_45/strided_slice:output:0*sequential_22/lstm_45/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_22/lstm_45/zeros/mul�
"sequential_22/lstm_45/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_22/lstm_45/zeros/Less/y�
 sequential_22/lstm_45/zeros/LessLess#sequential_22/lstm_45/zeros/mul:z:0+sequential_22/lstm_45/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_22/lstm_45/zeros/Less�
$sequential_22/lstm_45/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_22/lstm_45/zeros/packed/1�
"sequential_22/lstm_45/zeros/packedPack,sequential_22/lstm_45/strided_slice:output:0-sequential_22/lstm_45/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_22/lstm_45/zeros/packed�
!sequential_22/lstm_45/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_22/lstm_45/zeros/Const�
sequential_22/lstm_45/zerosFill+sequential_22/lstm_45/zeros/packed:output:0*sequential_22/lstm_45/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_22/lstm_45/zeros�
#sequential_22/lstm_45/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_22/lstm_45/zeros_1/mul/y�
!sequential_22/lstm_45/zeros_1/mulMul,sequential_22/lstm_45/strided_slice:output:0,sequential_22/lstm_45/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_22/lstm_45/zeros_1/mul�
$sequential_22/lstm_45/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_22/lstm_45/zeros_1/Less/y�
"sequential_22/lstm_45/zeros_1/LessLess%sequential_22/lstm_45/zeros_1/mul:z:0-sequential_22/lstm_45/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_22/lstm_45/zeros_1/Less�
&sequential_22/lstm_45/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_22/lstm_45/zeros_1/packed/1�
$sequential_22/lstm_45/zeros_1/packedPack,sequential_22/lstm_45/strided_slice:output:0/sequential_22/lstm_45/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_22/lstm_45/zeros_1/packed�
#sequential_22/lstm_45/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_22/lstm_45/zeros_1/Const�
sequential_22/lstm_45/zeros_1Fill-sequential_22/lstm_45/zeros_1/packed:output:0,sequential_22/lstm_45/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_22/lstm_45/zeros_1�
$sequential_22/lstm_45/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_22/lstm_45/transpose/perm�
sequential_22/lstm_45/transpose	Transpose%sequential_22/lstm_44/transpose_1:y:0-sequential_22/lstm_45/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2!
sequential_22/lstm_45/transpose�
sequential_22/lstm_45/Shape_1Shape#sequential_22/lstm_45/transpose:y:0*
T0*
_output_shapes
:2
sequential_22/lstm_45/Shape_1�
+sequential_22/lstm_45/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_22/lstm_45/strided_slice_1/stack�
-sequential_22/lstm_45/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_45/strided_slice_1/stack_1�
-sequential_22/lstm_45/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_45/strided_slice_1/stack_2�
%sequential_22/lstm_45/strided_slice_1StridedSlice&sequential_22/lstm_45/Shape_1:output:04sequential_22/lstm_45/strided_slice_1/stack:output:06sequential_22/lstm_45/strided_slice_1/stack_1:output:06sequential_22/lstm_45/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_22/lstm_45/strided_slice_1�
1sequential_22/lstm_45/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_22/lstm_45/TensorArrayV2/element_shape�
#sequential_22/lstm_45/TensorArrayV2TensorListReserve:sequential_22/lstm_45/TensorArrayV2/element_shape:output:0.sequential_22/lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_22/lstm_45/TensorArrayV2�
Ksequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2M
Ksequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_22/lstm_45/transpose:y:0Tsequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensor�
+sequential_22/lstm_45/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_22/lstm_45/strided_slice_2/stack�
-sequential_22/lstm_45/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_45/strided_slice_2/stack_1�
-sequential_22/lstm_45/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_45/strided_slice_2/stack_2�
%sequential_22/lstm_45/strided_slice_2StridedSlice#sequential_22/lstm_45/transpose:y:04sequential_22/lstm_45/strided_slice_2/stack:output:06sequential_22/lstm_45/strided_slice_2/stack_1:output:06sequential_22/lstm_45/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_22/lstm_45/strided_slice_2�
8sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOpReadVariableOpAsequential_22_lstm_45_lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02:
8sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp�
)sequential_22/lstm_45/lstm_cell_89/MatMulMatMul.sequential_22/lstm_45/strided_slice_2:output:0@sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_22/lstm_45/lstm_cell_89/MatMul�
:sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOpCsequential_22_lstm_45_lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02<
:sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp�
+sequential_22/lstm_45/lstm_cell_89/MatMul_1MatMul$sequential_22/lstm_45/zeros:output:0Bsequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_22/lstm_45/lstm_cell_89/MatMul_1�
&sequential_22/lstm_45/lstm_cell_89/addAddV23sequential_22/lstm_45/lstm_cell_89/MatMul:product:05sequential_22/lstm_45/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_22/lstm_45/lstm_cell_89/add�
9sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOpBsequential_22_lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp�
*sequential_22/lstm_45/lstm_cell_89/BiasAddBiasAdd*sequential_22/lstm_45/lstm_cell_89/add:z:0Asequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_22/lstm_45/lstm_cell_89/BiasAdd�
2sequential_22/lstm_45/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_22/lstm_45/lstm_cell_89/split/split_dim�
(sequential_22/lstm_45/lstm_cell_89/splitSplit;sequential_22/lstm_45/lstm_cell_89/split/split_dim:output:03sequential_22/lstm_45/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2*
(sequential_22/lstm_45/lstm_cell_89/split�
*sequential_22/lstm_45/lstm_cell_89/SigmoidSigmoid1sequential_22/lstm_45/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2,
*sequential_22/lstm_45/lstm_cell_89/Sigmoid�
,sequential_22/lstm_45/lstm_cell_89/Sigmoid_1Sigmoid1sequential_22/lstm_45/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2.
,sequential_22/lstm_45/lstm_cell_89/Sigmoid_1�
&sequential_22/lstm_45/lstm_cell_89/mulMul0sequential_22/lstm_45/lstm_cell_89/Sigmoid_1:y:0&sequential_22/lstm_45/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2(
&sequential_22/lstm_45/lstm_cell_89/mul�
'sequential_22/lstm_45/lstm_cell_89/ReluRelu1sequential_22/lstm_45/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2)
'sequential_22/lstm_45/lstm_cell_89/Relu�
(sequential_22/lstm_45/lstm_cell_89/mul_1Mul.sequential_22/lstm_45/lstm_cell_89/Sigmoid:y:05sequential_22/lstm_45/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_22/lstm_45/lstm_cell_89/mul_1�
(sequential_22/lstm_45/lstm_cell_89/add_1AddV2*sequential_22/lstm_45/lstm_cell_89/mul:z:0,sequential_22/lstm_45/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2*
(sequential_22/lstm_45/lstm_cell_89/add_1�
,sequential_22/lstm_45/lstm_cell_89/Sigmoid_2Sigmoid1sequential_22/lstm_45/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2.
,sequential_22/lstm_45/lstm_cell_89/Sigmoid_2�
)sequential_22/lstm_45/lstm_cell_89/Relu_1Relu,sequential_22/lstm_45/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2+
)sequential_22/lstm_45/lstm_cell_89/Relu_1�
(sequential_22/lstm_45/lstm_cell_89/mul_2Mul0sequential_22/lstm_45/lstm_cell_89/Sigmoid_2:y:07sequential_22/lstm_45/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_22/lstm_45/lstm_cell_89/mul_2�
3sequential_22/lstm_45/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    25
3sequential_22/lstm_45/TensorArrayV2_1/element_shape�
%sequential_22/lstm_45/TensorArrayV2_1TensorListReserve<sequential_22/lstm_45/TensorArrayV2_1/element_shape:output:0.sequential_22/lstm_45/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_22/lstm_45/TensorArrayV2_1z
sequential_22/lstm_45/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_22/lstm_45/time�
.sequential_22/lstm_45/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_22/lstm_45/while/maximum_iterations�
(sequential_22/lstm_45/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_22/lstm_45/while/loop_counter�
sequential_22/lstm_45/whileWhile1sequential_22/lstm_45/while/loop_counter:output:07sequential_22/lstm_45/while/maximum_iterations:output:0#sequential_22/lstm_45/time:output:0.sequential_22/lstm_45/TensorArrayV2_1:handle:0$sequential_22/lstm_45/zeros:output:0&sequential_22/lstm_45/zeros_1:output:0.sequential_22/lstm_45/strided_slice_1:output:0Msequential_22/lstm_45/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_22_lstm_45_lstm_cell_89_matmul_readvariableop_resourceCsequential_22_lstm_45_lstm_cell_89_matmul_1_readvariableop_resourceBsequential_22_lstm_45_lstm_cell_89_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :��������� :��������� : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_22_lstm_45_while_body_683298*3
cond+R)
'sequential_22_lstm_45_while_cond_683297*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
sequential_22/lstm_45/while�
Fsequential_22/lstm_45/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2H
Fsequential_22/lstm_45/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_22/lstm_45/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_22/lstm_45/while:output:3Osequential_22/lstm_45/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02:
8sequential_22/lstm_45/TensorArrayV2Stack/TensorListStack�
+sequential_22/lstm_45/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_22/lstm_45/strided_slice_3/stack�
-sequential_22/lstm_45/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_22/lstm_45/strided_slice_3/stack_1�
-sequential_22/lstm_45/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_22/lstm_45/strided_slice_3/stack_2�
%sequential_22/lstm_45/strided_slice_3StridedSliceAsequential_22/lstm_45/TensorArrayV2Stack/TensorListStack:tensor:04sequential_22/lstm_45/strided_slice_3/stack:output:06sequential_22/lstm_45/strided_slice_3/stack_1:output:06sequential_22/lstm_45/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2'
%sequential_22/lstm_45/strided_slice_3�
&sequential_22/lstm_45/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_22/lstm_45/transpose_1/perm�
!sequential_22/lstm_45/transpose_1	TransposeAsequential_22/lstm_45/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_22/lstm_45/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2#
!sequential_22/lstm_45/transpose_1�
sequential_22/lstm_45/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_22/lstm_45/runtime�
!sequential_22/dropout_22/IdentityIdentity.sequential_22/lstm_45/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2#
!sequential_22/dropout_22/Identity�
,sequential_22/dense_22/MatMul/ReadVariableOpReadVariableOp5sequential_22_dense_22_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_22/dense_22/MatMul/ReadVariableOp�
sequential_22/dense_22/MatMulMatMul*sequential_22/dropout_22/Identity:output:04sequential_22/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_22/dense_22/MatMul�
-sequential_22/dense_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_22/dense_22/BiasAdd/ReadVariableOp�
sequential_22/dense_22/BiasAddBiasAdd'sequential_22/dense_22/MatMul:product:05sequential_22/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_22/dense_22/BiasAdd�
IdentityIdentity'sequential_22/dense_22/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^sequential_22/dense_22/BiasAdd/ReadVariableOp-^sequential_22/dense_22/MatMul/ReadVariableOp:^sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp9^sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp;^sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp^sequential_22/lstm_44/while:^sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp9^sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp;^sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp^sequential_22/lstm_45/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2^
-sequential_22/dense_22/BiasAdd/ReadVariableOp-sequential_22/dense_22/BiasAdd/ReadVariableOp2\
,sequential_22/dense_22/MatMul/ReadVariableOp,sequential_22/dense_22/MatMul/ReadVariableOp2v
9sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp9sequential_22/lstm_44/lstm_cell_88/BiasAdd/ReadVariableOp2t
8sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp8sequential_22/lstm_44/lstm_cell_88/MatMul/ReadVariableOp2x
:sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp:sequential_22/lstm_44/lstm_cell_88/MatMul_1/ReadVariableOp2:
sequential_22/lstm_44/whilesequential_22/lstm_44/while2v
9sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp9sequential_22/lstm_45/lstm_cell_89/BiasAdd/ReadVariableOp2t
8sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp8sequential_22/lstm_45/lstm_cell_89/MatMul/ReadVariableOp2x
:sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp:sequential_22/lstm_45/lstm_cell_89/MatMul_1/ReadVariableOp2:
sequential_22/lstm_45/whilesequential_22/lstm_45/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_44_input
�F
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_684387

inputs&
lstm_cell_89_684305:	@�&
lstm_cell_89_684307:	 �"
lstm_cell_89_684309:	�
identity��$lstm_cell_89/StatefulPartitionedCall�whileD
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
$lstm_cell_89/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_89_684305lstm_cell_89_684307lstm_cell_89_684309*
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6842402&
$lstm_cell_89/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_89_684305lstm_cell_89_684307lstm_cell_89_684309*
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
while_body_684318*
condR
while_cond_684317*K
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
NoOpNoOp%^lstm_cell_89/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_89/StatefulPartitionedCall$lstm_cell_89/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
(__inference_lstm_44_layer_call_fn_686261

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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6853852
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
�
�
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_683610

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
�%
�
while_body_683688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_88_683712_0:	�.
while_lstm_cell_88_683714_0:	@�*
while_lstm_cell_88_683716_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_88_683712:	�,
while_lstm_cell_88_683714:	@�(
while_lstm_cell_88_683716:	���*while/lstm_cell_88/StatefulPartitionedCall�
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
*while/lstm_cell_88/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_88_683712_0while_lstm_cell_88_683714_0while_lstm_cell_88_683716_0*
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6836102,
*while/lstm_cell_88/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_88/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_88/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_88/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_88/StatefulPartitionedCall*"
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
while_lstm_cell_88_683712while_lstm_cell_88_683712_0"8
while_lstm_cell_88_683714while_lstm_cell_88_683714_0"8
while_lstm_cell_88_683716while_lstm_cell_88_683716_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_88/StatefulPartitionedCall*while/lstm_cell_88/StatefulPartitionedCall: 
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687755

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
�
(__inference_lstm_45_layer_call_fn_686898

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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6849642
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
�
�
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_684094

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
�?
�
while_body_686479
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
while_body_685301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
while_body_687429
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
�
"__inference__traced_restore_687974
file_prefix2
 assignvariableop_dense_22_kernel: .
 assignvariableop_1_dense_22_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_44_lstm_cell_88_kernel:	�K
8assignvariableop_8_lstm_44_lstm_cell_88_recurrent_kernel:	@�;
,assignvariableop_9_lstm_44_lstm_cell_88_bias:	�B
/assignvariableop_10_lstm_45_lstm_cell_89_kernel:	@�L
9assignvariableop_11_lstm_45_lstm_cell_89_recurrent_kernel:	 �<
-assignvariableop_12_lstm_45_lstm_cell_89_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_22_kernel_m: 6
(assignvariableop_16_adam_dense_22_bias_m:I
6assignvariableop_17_adam_lstm_44_lstm_cell_88_kernel_m:	�S
@assignvariableop_18_adam_lstm_44_lstm_cell_88_recurrent_kernel_m:	@�C
4assignvariableop_19_adam_lstm_44_lstm_cell_88_bias_m:	�I
6assignvariableop_20_adam_lstm_45_lstm_cell_89_kernel_m:	@�S
@assignvariableop_21_adam_lstm_45_lstm_cell_89_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_lstm_45_lstm_cell_89_bias_m:	�<
*assignvariableop_23_adam_dense_22_kernel_v: 6
(assignvariableop_24_adam_dense_22_bias_v:I
6assignvariableop_25_adam_lstm_44_lstm_cell_88_kernel_v:	�S
@assignvariableop_26_adam_lstm_44_lstm_cell_88_recurrent_kernel_v:	@�C
4assignvariableop_27_adam_lstm_44_lstm_cell_88_bias_v:	�I
6assignvariableop_28_adam_lstm_45_lstm_cell_89_kernel_v:	@�S
@assignvariableop_29_adam_lstm_45_lstm_cell_89_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_lstm_45_lstm_cell_89_bias_v:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_22_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_22_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_44_lstm_cell_88_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_44_lstm_cell_88_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_44_lstm_cell_88_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_45_lstm_cell_89_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_45_lstm_cell_89_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_45_lstm_cell_89_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_22_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_22_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_44_lstm_cell_88_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_44_lstm_cell_88_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_44_lstm_cell_88_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_45_lstm_cell_89_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_45_lstm_cell_89_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_45_lstm_cell_89_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_22_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_22_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_44_lstm_cell_88_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_44_lstm_cell_88_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_44_lstm_cell_88_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_45_lstm_cell_89_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_45_lstm_cell_89_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_45_lstm_cell_89_bias_vIdentity_30:output:0"/device:CPU:0*
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
�
�
-__inference_lstm_cell_88_layer_call_fn_687576

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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6834642
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
�
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_684977

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
while_body_686630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
.__inference_sequential_22_layer_call_fn_685015
lstm_44_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8� *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_6849962
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
_user_specified_namelstm_44_input
�
�
(__inference_lstm_44_layer_call_fn_686250

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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6848062
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
�^
�
'sequential_22_lstm_45_while_body_683298H
Dsequential_22_lstm_45_while_sequential_22_lstm_45_while_loop_counterN
Jsequential_22_lstm_45_while_sequential_22_lstm_45_while_maximum_iterations+
'sequential_22_lstm_45_while_placeholder-
)sequential_22_lstm_45_while_placeholder_1-
)sequential_22_lstm_45_while_placeholder_2-
)sequential_22_lstm_45_while_placeholder_3G
Csequential_22_lstm_45_while_sequential_22_lstm_45_strided_slice_1_0�
sequential_22_lstm_45_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_45_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_22_lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0:	@�^
Ksequential_22_lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �Y
Jsequential_22_lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0:	�(
$sequential_22_lstm_45_while_identity*
&sequential_22_lstm_45_while_identity_1*
&sequential_22_lstm_45_while_identity_2*
&sequential_22_lstm_45_while_identity_3*
&sequential_22_lstm_45_while_identity_4*
&sequential_22_lstm_45_while_identity_5E
Asequential_22_lstm_45_while_sequential_22_lstm_45_strided_slice_1�
}sequential_22_lstm_45_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_45_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_22_lstm_45_while_lstm_cell_89_matmul_readvariableop_resource:	@�\
Isequential_22_lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource:	 �W
Hsequential_22_lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource:	���?sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�>sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�@sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
Msequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2O
Msequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_22_lstm_45_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_45_tensorarrayunstack_tensorlistfromtensor_0'sequential_22_lstm_45_while_placeholderVsequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02A
?sequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItem�
>sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOpIsequential_22_lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02@
>sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�
/sequential_22/lstm_45/while/lstm_cell_89/MatMulMatMulFsequential_22/lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_22/lstm_45/while/lstm_cell_89/MatMul�
@sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOpKsequential_22_lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02B
@sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
1sequential_22/lstm_45/while/lstm_cell_89/MatMul_1MatMul)sequential_22_lstm_45_while_placeholder_2Hsequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_22/lstm_45/while/lstm_cell_89/MatMul_1�
,sequential_22/lstm_45/while/lstm_cell_89/addAddV29sequential_22/lstm_45/while/lstm_cell_89/MatMul:product:0;sequential_22/lstm_45/while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_22/lstm_45/while/lstm_cell_89/add�
?sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOpJsequential_22_lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�
0sequential_22/lstm_45/while/lstm_cell_89/BiasAddBiasAdd0sequential_22/lstm_45/while/lstm_cell_89/add:z:0Gsequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_22/lstm_45/while/lstm_cell_89/BiasAdd�
8sequential_22/lstm_45/while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_22/lstm_45/while/lstm_cell_89/split/split_dim�
.sequential_22/lstm_45/while/lstm_cell_89/splitSplitAsequential_22/lstm_45/while/lstm_cell_89/split/split_dim:output:09sequential_22/lstm_45/while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split20
.sequential_22/lstm_45/while/lstm_cell_89/split�
0sequential_22/lstm_45/while/lstm_cell_89/SigmoidSigmoid7sequential_22/lstm_45/while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 22
0sequential_22/lstm_45/while/lstm_cell_89/Sigmoid�
2sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_1Sigmoid7sequential_22/lstm_45/while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 24
2sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_1�
,sequential_22/lstm_45/while/lstm_cell_89/mulMul6sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_1:y:0)sequential_22_lstm_45_while_placeholder_3*
T0*'
_output_shapes
:��������� 2.
,sequential_22/lstm_45/while/lstm_cell_89/mul�
-sequential_22/lstm_45/while/lstm_cell_89/ReluRelu7sequential_22/lstm_45/while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2/
-sequential_22/lstm_45/while/lstm_cell_89/Relu�
.sequential_22/lstm_45/while/lstm_cell_89/mul_1Mul4sequential_22/lstm_45/while/lstm_cell_89/Sigmoid:y:0;sequential_22/lstm_45/while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_22/lstm_45/while/lstm_cell_89/mul_1�
.sequential_22/lstm_45/while/lstm_cell_89/add_1AddV20sequential_22/lstm_45/while/lstm_cell_89/mul:z:02sequential_22/lstm_45/while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 20
.sequential_22/lstm_45/while/lstm_cell_89/add_1�
2sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_2Sigmoid7sequential_22/lstm_45/while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 24
2sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_2�
/sequential_22/lstm_45/while/lstm_cell_89/Relu_1Relu2sequential_22/lstm_45/while/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 21
/sequential_22/lstm_45/while/lstm_cell_89/Relu_1�
.sequential_22/lstm_45/while/lstm_cell_89/mul_2Mul6sequential_22/lstm_45/while/lstm_cell_89/Sigmoid_2:y:0=sequential_22/lstm_45/while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_22/lstm_45/while/lstm_cell_89/mul_2�
@sequential_22/lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_22_lstm_45_while_placeholder_1'sequential_22_lstm_45_while_placeholder2sequential_22/lstm_45/while/lstm_cell_89/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_22/lstm_45/while/TensorArrayV2Write/TensorListSetItem�
!sequential_22/lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_22/lstm_45/while/add/y�
sequential_22/lstm_45/while/addAddV2'sequential_22_lstm_45_while_placeholder*sequential_22/lstm_45/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_22/lstm_45/while/add�
#sequential_22/lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_22/lstm_45/while/add_1/y�
!sequential_22/lstm_45/while/add_1AddV2Dsequential_22_lstm_45_while_sequential_22_lstm_45_while_loop_counter,sequential_22/lstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_22/lstm_45/while/add_1�
$sequential_22/lstm_45/while/IdentityIdentity%sequential_22/lstm_45/while/add_1:z:0!^sequential_22/lstm_45/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_22/lstm_45/while/Identity�
&sequential_22/lstm_45/while/Identity_1IdentityJsequential_22_lstm_45_while_sequential_22_lstm_45_while_maximum_iterations!^sequential_22/lstm_45/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_45/while/Identity_1�
&sequential_22/lstm_45/while/Identity_2Identity#sequential_22/lstm_45/while/add:z:0!^sequential_22/lstm_45/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_45/while/Identity_2�
&sequential_22/lstm_45/while/Identity_3IdentityPsequential_22/lstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_22/lstm_45/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_45/while/Identity_3�
&sequential_22/lstm_45/while/Identity_4Identity2sequential_22/lstm_45/while/lstm_cell_89/mul_2:z:0!^sequential_22/lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_22/lstm_45/while/Identity_4�
&sequential_22/lstm_45/while/Identity_5Identity2sequential_22/lstm_45/while/lstm_cell_89/add_1:z:0!^sequential_22/lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_22/lstm_45/while/Identity_5�
 sequential_22/lstm_45/while/NoOpNoOp@^sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp?^sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOpA^sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_22/lstm_45/while/NoOp"U
$sequential_22_lstm_45_while_identity-sequential_22/lstm_45/while/Identity:output:0"Y
&sequential_22_lstm_45_while_identity_1/sequential_22/lstm_45/while/Identity_1:output:0"Y
&sequential_22_lstm_45_while_identity_2/sequential_22/lstm_45/while/Identity_2:output:0"Y
&sequential_22_lstm_45_while_identity_3/sequential_22/lstm_45/while/Identity_3:output:0"Y
&sequential_22_lstm_45_while_identity_4/sequential_22/lstm_45/while/Identity_4:output:0"Y
&sequential_22_lstm_45_while_identity_5/sequential_22/lstm_45/while/Identity_5:output:0"�
Hsequential_22_lstm_45_while_lstm_cell_89_biasadd_readvariableop_resourceJsequential_22_lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0"�
Isequential_22_lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resourceKsequential_22_lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0"�
Gsequential_22_lstm_45_while_lstm_cell_89_matmul_readvariableop_resourceIsequential_22_lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0"�
Asequential_22_lstm_45_while_sequential_22_lstm_45_strided_slice_1Csequential_22_lstm_45_while_sequential_22_lstm_45_strided_slice_1_0"�
}sequential_22_lstm_45_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_45_tensorarrayunstack_tensorlistfromtensorsequential_22_lstm_45_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2�
?sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp?sequential_22/lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp2�
>sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp>sequential_22/lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp2�
@sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp@sequential_22/lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
while_cond_683477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_683477___redundant_placeholder04
0while_while_cond_683477___redundant_placeholder14
0while_while_cond_683477___redundant_placeholder24
0while_while_cond_683477___redundant_placeholder3
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
while_cond_684721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684721___redundant_placeholder04
0while_while_cond_684721___redundant_placeholder14
0while_while_cond_684721___redundant_placeholder24
0while_while_cond_684721___redundant_placeholder3
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
while_cond_686780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_686780___redundant_placeholder04
0while_while_cond_686780___redundant_placeholder14
0while_while_cond_686780___redundant_placeholder24
0while_while_cond_686780___redundant_placeholder3
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
while_cond_683687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_683687___redundant_placeholder04
0while_while_cond_683687___redundant_placeholder14
0while_while_cond_683687___redundant_placeholder24
0while_while_cond_683687___redundant_placeholder3
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
�
�
I__inference_sequential_22_layer_call_and_return_conditional_losses_685529
lstm_44_input!
lstm_44_685508:	�!
lstm_44_685510:	@�
lstm_44_685512:	�!
lstm_45_685515:	@�!
lstm_45_685517:	 �
lstm_45_685519:	�!
dense_22_685523: 
dense_22_685525:
identity�� dense_22/StatefulPartitionedCall�"dropout_22/StatefulPartitionedCall�lstm_44/StatefulPartitionedCall�lstm_45/StatefulPartitionedCall�
lstm_44/StatefulPartitionedCallStatefulPartitionedCalllstm_44_inputlstm_44_685508lstm_44_685510lstm_44_685512*
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6853852!
lstm_44/StatefulPartitionedCall�
lstm_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_44/StatefulPartitionedCall:output:0lstm_45_685515lstm_45_685517lstm_45_685519*
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6852122!
lstm_45/StatefulPartitionedCall�
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6850452$
"dropout_22/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_685523dense_22_685525*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_6849892"
 dense_22/StatefulPartitionedCall�
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_22/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall ^lstm_44/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2B
lstm_44/StatefulPartitionedCalllstm_44/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_44_input
�F
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_683547

inputs&
lstm_cell_88_683465:	�&
lstm_cell_88_683467:	@�"
lstm_cell_88_683469:	�
identity��$lstm_cell_88/StatefulPartitionedCall�whileD
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
$lstm_cell_88/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_88_683465lstm_cell_88_683467lstm_cell_88_683469*
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6834642&
$lstm_cell_88/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_88_683465lstm_cell_88_683467lstm_cell_88_683469*
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
while_body_683478*
condR
while_cond_683477*K
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
NoOpNoOp%^lstm_cell_88/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_88/StatefulPartitionedCall$lstm_cell_88/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�\
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_686412
inputs_0>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_686328*
condR
while_cond_686327*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�?
�
while_body_685128
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
D__inference_dense_22_layer_call_and_return_conditional_losses_687559

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
�?
�
while_body_686976
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_685212

inputs>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_685128*
condR
while_cond_685127*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�\
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_686563
inputs_0>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_686479*
condR
while_cond_686478*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�
�
'sequential_22_lstm_44_while_cond_683150H
Dsequential_22_lstm_44_while_sequential_22_lstm_44_while_loop_counterN
Jsequential_22_lstm_44_while_sequential_22_lstm_44_while_maximum_iterations+
'sequential_22_lstm_44_while_placeholder-
)sequential_22_lstm_44_while_placeholder_1-
)sequential_22_lstm_44_while_placeholder_2-
)sequential_22_lstm_44_while_placeholder_3J
Fsequential_22_lstm_44_while_less_sequential_22_lstm_44_strided_slice_1`
\sequential_22_lstm_44_while_sequential_22_lstm_44_while_cond_683150___redundant_placeholder0`
\sequential_22_lstm_44_while_sequential_22_lstm_44_while_cond_683150___redundant_placeholder1`
\sequential_22_lstm_44_while_sequential_22_lstm_44_while_cond_683150___redundant_placeholder2`
\sequential_22_lstm_44_while_sequential_22_lstm_44_while_cond_683150___redundant_placeholder3(
$sequential_22_lstm_44_while_identity
�
 sequential_22/lstm_44/while/LessLess'sequential_22_lstm_44_while_placeholderFsequential_22_lstm_44_while_less_sequential_22_lstm_44_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_22/lstm_44/while/Less�
$sequential_22/lstm_44/while/IdentityIdentity$sequential_22/lstm_44/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_22/lstm_44/while/Identity"U
$sequential_22_lstm_44_while_identity-sequential_22/lstm_44/while/Identity:output:0*(
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

�
D__inference_dense_22_layer_call_and_return_conditional_losses_684989

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
�[
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_684964

inputs>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_684880*
condR
while_cond_684879*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_687277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_687277___redundant_placeholder04
0while_while_cond_687277___redundant_placeholder14
0while_while_cond_687277___redundant_placeholder24
0while_while_cond_687277___redundant_placeholder3
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
while_cond_687126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_687126___redundant_placeholder04
0while_while_cond_687126___redundant_placeholder14
0while_while_cond_687126___redundant_placeholder24
0while_while_cond_687126___redundant_placeholder3
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
I__inference_sequential_22_layer_call_and_return_conditional_losses_685441

inputs!
lstm_44_685420:	�!
lstm_44_685422:	@�
lstm_44_685424:	�!
lstm_45_685427:	@�!
lstm_45_685429:	 �
lstm_45_685431:	�!
dense_22_685435: 
dense_22_685437:
identity�� dense_22/StatefulPartitionedCall�"dropout_22/StatefulPartitionedCall�lstm_44/StatefulPartitionedCall�lstm_45/StatefulPartitionedCall�
lstm_44/StatefulPartitionedCallStatefulPartitionedCallinputslstm_44_685420lstm_44_685422lstm_44_685424*
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6853852!
lstm_44/StatefulPartitionedCall�
lstm_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_44/StatefulPartitionedCall:output:0lstm_45_685427lstm_45_685429lstm_45_685431*
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6852122!
lstm_45/StatefulPartitionedCall�
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_45/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6850452$
"dropout_22/StatefulPartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_22_685435dense_22_685437*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_6849892"
 dense_22/StatefulPartitionedCall�
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_22/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall ^lstm_44/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2B
lstm_44/StatefulPartitionedCalllstm_44/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�\
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_687060
inputs_0>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_686976*
condR
while_cond_686975*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
-__inference_lstm_cell_89_layer_call_fn_687674

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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6840942
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
�
(__inference_lstm_44_layer_call_fn_686239
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6837572
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
�
�
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_683464

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
�\
�
C__inference_lstm_45_layer_call_and_return_conditional_losses_687211
inputs_0>
+lstm_cell_89_matmul_readvariableop_resource:	@�@
-lstm_cell_89_matmul_1_readvariableop_resource:	 �;
,lstm_cell_89_biasadd_readvariableop_resource:	�
identity��#lstm_cell_89/BiasAdd/ReadVariableOp�"lstm_cell_89/MatMul/ReadVariableOp�$lstm_cell_89/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_89/MatMul/ReadVariableOpReadVariableOp+lstm_cell_89_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_89/MatMul/ReadVariableOp�
lstm_cell_89/MatMulMatMulstrided_slice_2:output:0*lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul�
$lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_89_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_89/MatMul_1/ReadVariableOp�
lstm_cell_89/MatMul_1MatMulzeros:output:0,lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/MatMul_1�
lstm_cell_89/addAddV2lstm_cell_89/MatMul:product:0lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/add�
#lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_89_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_89/BiasAdd/ReadVariableOp�
lstm_cell_89/BiasAddBiasAddlstm_cell_89/add:z:0+lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_89/BiasAdd~
lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_89/split/split_dim�
lstm_cell_89/splitSplit%lstm_cell_89/split/split_dim:output:0lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_89/split�
lstm_cell_89/SigmoidSigmoidlstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid�
lstm_cell_89/Sigmoid_1Sigmoidlstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_1�
lstm_cell_89/mulMullstm_cell_89/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul}
lstm_cell_89/ReluRelulstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu�
lstm_cell_89/mul_1Mullstm_cell_89/Sigmoid:y:0lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_1�
lstm_cell_89/add_1AddV2lstm_cell_89/mul:z:0lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/add_1�
lstm_cell_89/Sigmoid_2Sigmoidlstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Sigmoid_2|
lstm_cell_89/Relu_1Relulstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/Relu_1�
lstm_cell_89/mul_2Mullstm_cell_89/Sigmoid_2:y:0!lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_89/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_89_matmul_readvariableop_resource-lstm_cell_89_matmul_1_readvariableop_resource,lstm_cell_89_biasadd_readvariableop_resource*
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
while_body_687127*
condR
while_cond_687126*K
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
NoOpNoOp$^lstm_cell_89/BiasAdd/ReadVariableOp#^lstm_cell_89/MatMul/ReadVariableOp%^lstm_cell_89/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_89/BiasAdd/ReadVariableOp#lstm_cell_89/BiasAdd/ReadVariableOp2H
"lstm_cell_89/MatMul/ReadVariableOp"lstm_cell_89/MatMul/ReadVariableOp2L
$lstm_cell_89/MatMul_1/ReadVariableOp$lstm_cell_89/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�?
�
while_body_684722
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
�
G
+__inference_dropout_22_layer_call_fn_687518

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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6849772
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_685385

inputs>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_685301*
condR
while_cond_685300*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�[
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_686865

inputs>
+lstm_cell_88_matmul_readvariableop_resource:	�@
-lstm_cell_88_matmul_1_readvariableop_resource:	@�;
,lstm_cell_88_biasadd_readvariableop_resource:	�
identity��#lstm_cell_88/BiasAdd/ReadVariableOp�"lstm_cell_88/MatMul/ReadVariableOp�$lstm_cell_88/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_88/MatMul/ReadVariableOpReadVariableOp+lstm_cell_88_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_88/MatMul/ReadVariableOp�
lstm_cell_88/MatMulMatMulstrided_slice_2:output:0*lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul�
$lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_88_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_88/MatMul_1/ReadVariableOp�
lstm_cell_88/MatMul_1MatMulzeros:output:0,lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/MatMul_1�
lstm_cell_88/addAddV2lstm_cell_88/MatMul:product:0lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/add�
#lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_88_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_88/BiasAdd/ReadVariableOp�
lstm_cell_88/BiasAddBiasAddlstm_cell_88/add:z:0+lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_88/BiasAdd~
lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_88/split/split_dim�
lstm_cell_88/splitSplit%lstm_cell_88/split/split_dim:output:0lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_88/split�
lstm_cell_88/SigmoidSigmoidlstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid�
lstm_cell_88/Sigmoid_1Sigmoidlstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_1�
lstm_cell_88/mulMullstm_cell_88/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul}
lstm_cell_88/ReluRelulstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu�
lstm_cell_88/mul_1Mullstm_cell_88/Sigmoid:y:0lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_1�
lstm_cell_88/add_1AddV2lstm_cell_88/mul:z:0lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/add_1�
lstm_cell_88/Sigmoid_2Sigmoidlstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Sigmoid_2|
lstm_cell_88/Relu_1Relulstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/Relu_1�
lstm_cell_88/mul_2Mullstm_cell_88/Sigmoid_2:y:0!lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_88/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_88_matmul_readvariableop_resource-lstm_cell_88_matmul_1_readvariableop_resource,lstm_cell_88_biasadd_readvariableop_resource*
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
while_body_686781*
condR
while_cond_686780*K
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
NoOpNoOp$^lstm_cell_88/BiasAdd/ReadVariableOp#^lstm_cell_88/MatMul/ReadVariableOp%^lstm_cell_88/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_88/BiasAdd/ReadVariableOp#lstm_cell_88/BiasAdd/ReadVariableOp2H
"lstm_cell_88/MatMul/ReadVariableOp"lstm_cell_88/MatMul/ReadVariableOp2L
$lstm_cell_88/MatMul_1/ReadVariableOp$lstm_cell_88/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
while_body_684108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_89_684132_0:	@�.
while_lstm_cell_89_684134_0:	 �*
while_lstm_cell_89_684136_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_89_684132:	@�,
while_lstm_cell_89_684134:	 �(
while_lstm_cell_89_684136:	���*while/lstm_cell_89/StatefulPartitionedCall�
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
*while/lstm_cell_89/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_89_684132_0while_lstm_cell_89_684134_0while_lstm_cell_89_684136_0*
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6840942,
*while/lstm_cell_89/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_89/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_89/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_89/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_89/StatefulPartitionedCall*"
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
while_lstm_cell_89_684132while_lstm_cell_89_684132_0"8
while_lstm_cell_89_684134while_lstm_cell_89_684134_0"8
while_lstm_cell_89_684136while_lstm_cell_89_684136_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_89/StatefulPartitionedCall*while/lstm_cell_89/StatefulPartitionedCall: 
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
�J
�

lstm_45_while_body_686119,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3+
'lstm_45_while_lstm_45_strided_slice_1_0g
clstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0:	@�P
=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �K
<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
lstm_45_while_identity
lstm_45_while_identity_1
lstm_45_while_identity_2
lstm_45_while_identity_3
lstm_45_while_identity_4
lstm_45_while_identity_5)
%lstm_45_while_lstm_45_strided_slice_1e
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorL
9lstm_45_while_lstm_cell_89_matmul_readvariableop_resource:	@�N
;lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource:	 �I
:lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource:	���1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0lstm_45_while_placeholderHlstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_45/while/TensorArrayV2Read/TensorListGetItem�
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�
!lstm_45/while/lstm_cell_89/MatMulMatMul8lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_45/while/lstm_cell_89/MatMul�
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
#lstm_45/while/lstm_cell_89/MatMul_1MatMullstm_45_while_placeholder_2:lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_45/while/lstm_cell_89/MatMul_1�
lstm_45/while/lstm_cell_89/addAddV2+lstm_45/while/lstm_cell_89/MatMul:product:0-lstm_45/while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_45/while/lstm_cell_89/add�
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�
"lstm_45/while/lstm_cell_89/BiasAddBiasAdd"lstm_45/while/lstm_cell_89/add:z:09lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_45/while/lstm_cell_89/BiasAdd�
*lstm_45/while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_45/while/lstm_cell_89/split/split_dim�
 lstm_45/while/lstm_cell_89/splitSplit3lstm_45/while/lstm_cell_89/split/split_dim:output:0+lstm_45/while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_45/while/lstm_cell_89/split�
"lstm_45/while/lstm_cell_89/SigmoidSigmoid)lstm_45/while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_45/while/lstm_cell_89/Sigmoid�
$lstm_45/while/lstm_cell_89/Sigmoid_1Sigmoid)lstm_45/while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_45/while/lstm_cell_89/Sigmoid_1�
lstm_45/while/lstm_cell_89/mulMul(lstm_45/while/lstm_cell_89/Sigmoid_1:y:0lstm_45_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_45/while/lstm_cell_89/mul�
lstm_45/while/lstm_cell_89/ReluRelu)lstm_45/while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_45/while/lstm_cell_89/Relu�
 lstm_45/while/lstm_cell_89/mul_1Mul&lstm_45/while/lstm_cell_89/Sigmoid:y:0-lstm_45/while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/mul_1�
 lstm_45/while/lstm_cell_89/add_1AddV2"lstm_45/while/lstm_cell_89/mul:z:0$lstm_45/while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/add_1�
$lstm_45/while/lstm_cell_89/Sigmoid_2Sigmoid)lstm_45/while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_45/while/lstm_cell_89/Sigmoid_2�
!lstm_45/while/lstm_cell_89/Relu_1Relu$lstm_45/while/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_45/while/lstm_cell_89/Relu_1�
 lstm_45/while/lstm_cell_89/mul_2Mul(lstm_45/while/lstm_cell_89/Sigmoid_2:y:0/lstm_45/while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/mul_2�
2lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_45_while_placeholder_1lstm_45_while_placeholder$lstm_45/while/lstm_cell_89/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_45/while/TensorArrayV2Write/TensorListSetIteml
lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_45/while/add/y�
lstm_45/while/addAddV2lstm_45_while_placeholderlstm_45/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_45/while/addp
lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_45/while/add_1/y�
lstm_45/while/add_1AddV2(lstm_45_while_lstm_45_while_loop_counterlstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_45/while/add_1�
lstm_45/while/IdentityIdentitylstm_45/while/add_1:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity�
lstm_45/while/Identity_1Identity.lstm_45_while_lstm_45_while_maximum_iterations^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_1�
lstm_45/while/Identity_2Identitylstm_45/while/add:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_2�
lstm_45/while/Identity_3IdentityBlstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_3�
lstm_45/while/Identity_4Identity$lstm_45/while/lstm_cell_89/mul_2:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_45/while/Identity_4�
lstm_45/while/Identity_5Identity$lstm_45/while/lstm_cell_89/add_1:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_45/while/Identity_5�
lstm_45/while/NoOpNoOp2^lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp1^lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp3^lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_45/while/NoOp"9
lstm_45_while_identitylstm_45/while/Identity:output:0"=
lstm_45_while_identity_1!lstm_45/while/Identity_1:output:0"=
lstm_45_while_identity_2!lstm_45/while/Identity_2:output:0"=
lstm_45_while_identity_3!lstm_45/while/Identity_3:output:0"=
lstm_45_while_identity_4!lstm_45/while/Identity_4:output:0"=
lstm_45_while_identity_5!lstm_45/while/Identity_5:output:0"P
%lstm_45_while_lstm_45_strided_slice_1'lstm_45_while_lstm_45_strided_slice_1_0"z
:lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0"|
;lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0"x
9lstm_45_while_lstm_cell_89_matmul_readvariableop_resource;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0"�
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp2d
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp2h
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_45_layer_call_fn_686876
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6841772
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
�

�
lstm_44_while_cond_685971,
(lstm_44_while_lstm_44_while_loop_counter2
.lstm_44_while_lstm_44_while_maximum_iterations
lstm_44_while_placeholder
lstm_44_while_placeholder_1
lstm_44_while_placeholder_2
lstm_44_while_placeholder_3.
*lstm_44_while_less_lstm_44_strided_slice_1D
@lstm_44_while_lstm_44_while_cond_685971___redundant_placeholder0D
@lstm_44_while_lstm_44_while_cond_685971___redundant_placeholder1D
@lstm_44_while_lstm_44_while_cond_685971___redundant_placeholder2D
@lstm_44_while_lstm_44_while_cond_685971___redundant_placeholder3
lstm_44_while_identity
�
lstm_44/while/LessLesslstm_44_while_placeholder*lstm_44_while_less_lstm_44_strided_slice_1*
T0*
_output_shapes
: 2
lstm_44/while/Lessu
lstm_44/while/IdentityIdentitylstm_44/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_44/while/Identity"9
lstm_44_while_identitylstm_44/while/Identity:output:0*(
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
�
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_687528

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
�^
�
'sequential_22_lstm_44_while_body_683151H
Dsequential_22_lstm_44_while_sequential_22_lstm_44_while_loop_counterN
Jsequential_22_lstm_44_while_sequential_22_lstm_44_while_maximum_iterations+
'sequential_22_lstm_44_while_placeholder-
)sequential_22_lstm_44_while_placeholder_1-
)sequential_22_lstm_44_while_placeholder_2-
)sequential_22_lstm_44_while_placeholder_3G
Csequential_22_lstm_44_while_sequential_22_lstm_44_strided_slice_1_0�
sequential_22_lstm_44_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_44_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_22_lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0:	�^
Ksequential_22_lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�Y
Jsequential_22_lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0:	�(
$sequential_22_lstm_44_while_identity*
&sequential_22_lstm_44_while_identity_1*
&sequential_22_lstm_44_while_identity_2*
&sequential_22_lstm_44_while_identity_3*
&sequential_22_lstm_44_while_identity_4*
&sequential_22_lstm_44_while_identity_5E
Asequential_22_lstm_44_while_sequential_22_lstm_44_strided_slice_1�
}sequential_22_lstm_44_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_44_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_22_lstm_44_while_lstm_cell_88_matmul_readvariableop_resource:	�\
Isequential_22_lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource:	@�W
Hsequential_22_lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource:	���?sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�>sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�@sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
Msequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2O
Msequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_22_lstm_44_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_44_tensorarrayunstack_tensorlistfromtensor_0'sequential_22_lstm_44_while_placeholderVsequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02A
?sequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItem�
>sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOpIsequential_22_lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02@
>sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�
/sequential_22/lstm_44/while/lstm_cell_88/MatMulMatMulFsequential_22/lstm_44/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_22/lstm_44/while/lstm_cell_88/MatMul�
@sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOpKsequential_22_lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02B
@sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
1sequential_22/lstm_44/while/lstm_cell_88/MatMul_1MatMul)sequential_22_lstm_44_while_placeholder_2Hsequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_22/lstm_44/while/lstm_cell_88/MatMul_1�
,sequential_22/lstm_44/while/lstm_cell_88/addAddV29sequential_22/lstm_44/while/lstm_cell_88/MatMul:product:0;sequential_22/lstm_44/while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_22/lstm_44/while/lstm_cell_88/add�
?sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOpJsequential_22_lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�
0sequential_22/lstm_44/while/lstm_cell_88/BiasAddBiasAdd0sequential_22/lstm_44/while/lstm_cell_88/add:z:0Gsequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_22/lstm_44/while/lstm_cell_88/BiasAdd�
8sequential_22/lstm_44/while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_22/lstm_44/while/lstm_cell_88/split/split_dim�
.sequential_22/lstm_44/while/lstm_cell_88/splitSplitAsequential_22/lstm_44/while/lstm_cell_88/split/split_dim:output:09sequential_22/lstm_44/while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split20
.sequential_22/lstm_44/while/lstm_cell_88/split�
0sequential_22/lstm_44/while/lstm_cell_88/SigmoidSigmoid7sequential_22/lstm_44/while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@22
0sequential_22/lstm_44/while/lstm_cell_88/Sigmoid�
2sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_1Sigmoid7sequential_22/lstm_44/while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@24
2sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_1�
,sequential_22/lstm_44/while/lstm_cell_88/mulMul6sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_1:y:0)sequential_22_lstm_44_while_placeholder_3*
T0*'
_output_shapes
:���������@2.
,sequential_22/lstm_44/while/lstm_cell_88/mul�
-sequential_22/lstm_44/while/lstm_cell_88/ReluRelu7sequential_22/lstm_44/while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2/
-sequential_22/lstm_44/while/lstm_cell_88/Relu�
.sequential_22/lstm_44/while/lstm_cell_88/mul_1Mul4sequential_22/lstm_44/while/lstm_cell_88/Sigmoid:y:0;sequential_22/lstm_44/while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_22/lstm_44/while/lstm_cell_88/mul_1�
.sequential_22/lstm_44/while/lstm_cell_88/add_1AddV20sequential_22/lstm_44/while/lstm_cell_88/mul:z:02sequential_22/lstm_44/while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@20
.sequential_22/lstm_44/while/lstm_cell_88/add_1�
2sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_2Sigmoid7sequential_22/lstm_44/while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@24
2sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_2�
/sequential_22/lstm_44/while/lstm_cell_88/Relu_1Relu2sequential_22/lstm_44/while/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@21
/sequential_22/lstm_44/while/lstm_cell_88/Relu_1�
.sequential_22/lstm_44/while/lstm_cell_88/mul_2Mul6sequential_22/lstm_44/while/lstm_cell_88/Sigmoid_2:y:0=sequential_22/lstm_44/while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_22/lstm_44/while/lstm_cell_88/mul_2�
@sequential_22/lstm_44/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_22_lstm_44_while_placeholder_1'sequential_22_lstm_44_while_placeholder2sequential_22/lstm_44/while/lstm_cell_88/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_22/lstm_44/while/TensorArrayV2Write/TensorListSetItem�
!sequential_22/lstm_44/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_22/lstm_44/while/add/y�
sequential_22/lstm_44/while/addAddV2'sequential_22_lstm_44_while_placeholder*sequential_22/lstm_44/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_22/lstm_44/while/add�
#sequential_22/lstm_44/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_22/lstm_44/while/add_1/y�
!sequential_22/lstm_44/while/add_1AddV2Dsequential_22_lstm_44_while_sequential_22_lstm_44_while_loop_counter,sequential_22/lstm_44/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_22/lstm_44/while/add_1�
$sequential_22/lstm_44/while/IdentityIdentity%sequential_22/lstm_44/while/add_1:z:0!^sequential_22/lstm_44/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_22/lstm_44/while/Identity�
&sequential_22/lstm_44/while/Identity_1IdentityJsequential_22_lstm_44_while_sequential_22_lstm_44_while_maximum_iterations!^sequential_22/lstm_44/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_44/while/Identity_1�
&sequential_22/lstm_44/while/Identity_2Identity#sequential_22/lstm_44/while/add:z:0!^sequential_22/lstm_44/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_44/while/Identity_2�
&sequential_22/lstm_44/while/Identity_3IdentityPsequential_22/lstm_44/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_22/lstm_44/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_22/lstm_44/while/Identity_3�
&sequential_22/lstm_44/while/Identity_4Identity2sequential_22/lstm_44/while/lstm_cell_88/mul_2:z:0!^sequential_22/lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_22/lstm_44/while/Identity_4�
&sequential_22/lstm_44/while/Identity_5Identity2sequential_22/lstm_44/while/lstm_cell_88/add_1:z:0!^sequential_22/lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_22/lstm_44/while/Identity_5�
 sequential_22/lstm_44/while/NoOpNoOp@^sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp?^sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOpA^sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_22/lstm_44/while/NoOp"U
$sequential_22_lstm_44_while_identity-sequential_22/lstm_44/while/Identity:output:0"Y
&sequential_22_lstm_44_while_identity_1/sequential_22/lstm_44/while/Identity_1:output:0"Y
&sequential_22_lstm_44_while_identity_2/sequential_22/lstm_44/while/Identity_2:output:0"Y
&sequential_22_lstm_44_while_identity_3/sequential_22/lstm_44/while/Identity_3:output:0"Y
&sequential_22_lstm_44_while_identity_4/sequential_22/lstm_44/while/Identity_4:output:0"Y
&sequential_22_lstm_44_while_identity_5/sequential_22/lstm_44/while/Identity_5:output:0"�
Hsequential_22_lstm_44_while_lstm_cell_88_biasadd_readvariableop_resourceJsequential_22_lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0"�
Isequential_22_lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resourceKsequential_22_lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0"�
Gsequential_22_lstm_44_while_lstm_cell_88_matmul_readvariableop_resourceIsequential_22_lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0"�
Asequential_22_lstm_44_while_sequential_22_lstm_44_strided_slice_1Csequential_22_lstm_44_while_sequential_22_lstm_44_strided_slice_1_0"�
}sequential_22_lstm_44_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_44_tensorarrayunstack_tensorlistfromtensorsequential_22_lstm_44_while_tensorarrayv2read_tensorlistgetitem_sequential_22_lstm_44_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2�
?sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp?sequential_22/lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp2�
>sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp>sequential_22/lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp2�
@sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp@sequential_22/lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
�F
�
C__inference_lstm_44_layer_call_and_return_conditional_losses_683757

inputs&
lstm_cell_88_683675:	�&
lstm_cell_88_683677:	@�"
lstm_cell_88_683679:	�
identity��$lstm_cell_88/StatefulPartitionedCall�whileD
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
$lstm_cell_88/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_88_683675lstm_cell_88_683677lstm_cell_88_683679*
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6836102&
$lstm_cell_88/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_88_683675lstm_cell_88_683677lstm_cell_88_683679*
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
while_body_683688*
condR
while_cond_683687*K
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
NoOpNoOp%^lstm_cell_88/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_88/StatefulPartitionedCall$lstm_cell_88/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
I__inference_sequential_22_layer_call_and_return_conditional_losses_685505
lstm_44_input!
lstm_44_685484:	�!
lstm_44_685486:	@�
lstm_44_685488:	�!
lstm_45_685491:	@�!
lstm_45_685493:	 �
lstm_45_685495:	�!
dense_22_685499: 
dense_22_685501:
identity�� dense_22/StatefulPartitionedCall�lstm_44/StatefulPartitionedCall�lstm_45/StatefulPartitionedCall�
lstm_44/StatefulPartitionedCallStatefulPartitionedCalllstm_44_inputlstm_44_685484lstm_44_685486lstm_44_685488*
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6848062!
lstm_44/StatefulPartitionedCall�
lstm_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_44/StatefulPartitionedCall:output:0lstm_45_685491lstm_45_685493lstm_45_685495*
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6849642!
lstm_45/StatefulPartitionedCall�
dropout_22/PartitionedCallPartitionedCall(lstm_45/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6849772
dropout_22/PartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_685499dense_22_685501*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_6849892"
 dense_22/StatefulPartitionedCall�
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_22/StatefulPartitionedCall ^lstm_44/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
lstm_44/StatefulPartitionedCalllstm_44/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_44_input
�?
�
while_body_687278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_89_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_89_matmul_readvariableop_resource:	@�F
3while_lstm_cell_89_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_89_biasadd_readvariableop_resource:	���)while/lstm_cell_89/BiasAdd/ReadVariableOp�(while/lstm_cell_89/MatMul/ReadVariableOp�*while/lstm_cell_89/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_89/MatMul/ReadVariableOp�
while/lstm_cell_89/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul�
*while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_89/MatMul_1/ReadVariableOp�
while/lstm_cell_89/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/MatMul_1�
while/lstm_cell_89/addAddV2#while/lstm_cell_89/MatMul:product:0%while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/add�
)while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_89/BiasAdd/ReadVariableOp�
while/lstm_cell_89/BiasAddBiasAddwhile/lstm_cell_89/add:z:01while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_89/BiasAdd�
"while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_89/split/split_dim�
while/lstm_cell_89/splitSplit+while/lstm_cell_89/split/split_dim:output:0#while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_89/split�
while/lstm_cell_89/SigmoidSigmoid!while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid�
while/lstm_cell_89/Sigmoid_1Sigmoid!while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_1�
while/lstm_cell_89/mulMul while/lstm_cell_89/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul�
while/lstm_cell_89/ReluRelu!while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu�
while/lstm_cell_89/mul_1Mulwhile/lstm_cell_89/Sigmoid:y:0%while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_1�
while/lstm_cell_89/add_1AddV2while/lstm_cell_89/mul:z:0while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/add_1�
while/lstm_cell_89/Sigmoid_2Sigmoid!while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Sigmoid_2�
while/lstm_cell_89/Relu_1Reluwhile/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/Relu_1�
while/lstm_cell_89/mul_2Mul while/lstm_cell_89/Sigmoid_2:y:0'while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_89/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_89/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_89/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_89/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_89/BiasAdd/ReadVariableOp)^while/lstm_cell_89/MatMul/ReadVariableOp+^while/lstm_cell_89/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_89_biasadd_readvariableop_resource4while_lstm_cell_89_biasadd_readvariableop_resource_0"l
3while_lstm_cell_89_matmul_1_readvariableop_resource5while_lstm_cell_89_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_89_matmul_readvariableop_resource3while_lstm_cell_89_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_89/BiasAdd/ReadVariableOp)while/lstm_cell_89/BiasAdd/ReadVariableOp2T
(while/lstm_cell_89/MatMul/ReadVariableOp(while/lstm_cell_89/MatMul/ReadVariableOp2X
*while/lstm_cell_89/MatMul_1/ReadVariableOp*while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_687540

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
�
�
while_cond_686478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_686478___redundant_placeholder04
0while_while_cond_686478___redundant_placeholder14
0while_while_cond_686478___redundant_placeholder24
0while_while_cond_686478___redundant_placeholder3
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
�
�
-__inference_lstm_cell_88_layer_call_fn_687593

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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_6836102
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
�
�
(__inference_lstm_45_layer_call_fn_686909

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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6852122
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
�
�
-__inference_lstm_cell_89_layer_call_fn_687691

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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_6842402
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
lstm_45_while_cond_685813,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3.
*lstm_45_while_less_lstm_45_strided_slice_1D
@lstm_45_while_lstm_45_while_cond_685813___redundant_placeholder0D
@lstm_45_while_lstm_45_while_cond_685813___redundant_placeholder1D
@lstm_45_while_lstm_45_while_cond_685813___redundant_placeholder2D
@lstm_45_while_lstm_45_while_cond_685813___redundant_placeholder3
lstm_45_while_identity
�
lstm_45/while/LessLesslstm_45_while_placeholder*lstm_45_while_less_lstm_45_strided_slice_1*
T0*
_output_shapes
: 2
lstm_45/while/Lessu
lstm_45/while/IdentityIdentitylstm_45/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_45/while/Identity"9
lstm_45_while_identitylstm_45/while/Identity:output:0*(
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687657

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
�?
�
while_body_686328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_88_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_88_matmul_readvariableop_resource:	�F
3while_lstm_cell_88_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_88_biasadd_readvariableop_resource:	���)while/lstm_cell_88/BiasAdd/ReadVariableOp�(while/lstm_cell_88/MatMul/ReadVariableOp�*while/lstm_cell_88/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_88/MatMul/ReadVariableOp�
while/lstm_cell_88/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul�
*while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_88/MatMul_1/ReadVariableOp�
while/lstm_cell_88/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/MatMul_1�
while/lstm_cell_88/addAddV2#while/lstm_cell_88/MatMul:product:0%while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/add�
)while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_88/BiasAdd/ReadVariableOp�
while/lstm_cell_88/BiasAddBiasAddwhile/lstm_cell_88/add:z:01while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_88/BiasAdd�
"while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_88/split/split_dim�
while/lstm_cell_88/splitSplit+while/lstm_cell_88/split/split_dim:output:0#while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_88/split�
while/lstm_cell_88/SigmoidSigmoid!while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid�
while/lstm_cell_88/Sigmoid_1Sigmoid!while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_1�
while/lstm_cell_88/mulMul while/lstm_cell_88/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul�
while/lstm_cell_88/ReluRelu!while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu�
while/lstm_cell_88/mul_1Mulwhile/lstm_cell_88/Sigmoid:y:0%while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_1�
while/lstm_cell_88/add_1AddV2while/lstm_cell_88/mul:z:0while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/add_1�
while/lstm_cell_88/Sigmoid_2Sigmoid!while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Sigmoid_2�
while/lstm_cell_88/Relu_1Reluwhile/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/Relu_1�
while/lstm_cell_88/mul_2Mul while/lstm_cell_88/Sigmoid_2:y:0'while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_88/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_88/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_88/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_88/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_88/BiasAdd/ReadVariableOp)^while/lstm_cell_88/MatMul/ReadVariableOp+^while/lstm_cell_88/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_88_biasadd_readvariableop_resource4while_lstm_cell_88_biasadd_readvariableop_resource_0"l
3while_lstm_cell_88_matmul_1_readvariableop_resource5while_lstm_cell_88_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_88_matmul_readvariableop_resource3while_lstm_cell_88_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_88/BiasAdd/ReadVariableOp)while/lstm_cell_88/BiasAdd/ReadVariableOp2T
(while/lstm_cell_88/MatMul/ReadVariableOp(while/lstm_cell_88/MatMul/ReadVariableOp2X
*while/lstm_cell_88/MatMul_1/ReadVariableOp*while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
.__inference_sequential_22_layer_call_fn_685579

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
GPU 2J 8� *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_6849962
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
�
�
I__inference_sequential_22_layer_call_and_return_conditional_losses_684996

inputs!
lstm_44_684807:	�!
lstm_44_684809:	@�
lstm_44_684811:	�!
lstm_45_684965:	@�!
lstm_45_684967:	 �
lstm_45_684969:	�!
dense_22_684990: 
dense_22_684992:
identity�� dense_22/StatefulPartitionedCall�lstm_44/StatefulPartitionedCall�lstm_45/StatefulPartitionedCall�
lstm_44/StatefulPartitionedCallStatefulPartitionedCallinputslstm_44_684807lstm_44_684809lstm_44_684811*
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_6848062!
lstm_44/StatefulPartitionedCall�
lstm_45/StatefulPartitionedCallStatefulPartitionedCall(lstm_44/StatefulPartitionedCall:output:0lstm_45_684965lstm_45_684967lstm_45_684969*
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_6849642!
lstm_45/StatefulPartitionedCall�
dropout_22/PartitionedCallPartitionedCall(lstm_45/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_6849772
dropout_22/PartitionedCall�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_22_684990dense_22_684992*
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
GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_6849892"
 dense_22/StatefulPartitionedCall�
IdentityIdentity)dense_22/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_22/StatefulPartitionedCall ^lstm_44/StatefulPartitionedCall ^lstm_45/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2B
lstm_44/StatefulPartitionedCalllstm_44/StatefulPartitionedCall2B
lstm_45/StatefulPartitionedCalllstm_45/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_sequential_22_layer_call_fn_685481
lstm_44_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_44_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8� *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_6854412
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
_user_specified_namelstm_44_input
�J
�

lstm_44_while_body_685667,
(lstm_44_while_lstm_44_while_loop_counter2
.lstm_44_while_lstm_44_while_maximum_iterations
lstm_44_while_placeholder
lstm_44_while_placeholder_1
lstm_44_while_placeholder_2
lstm_44_while_placeholder_3+
'lstm_44_while_lstm_44_strided_slice_1_0g
clstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0:	�P
=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0:	@�K
<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0:	�
lstm_44_while_identity
lstm_44_while_identity_1
lstm_44_while_identity_2
lstm_44_while_identity_3
lstm_44_while_identity_4
lstm_44_while_identity_5)
%lstm_44_while_lstm_44_strided_slice_1e
alstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensorL
9lstm_44_while_lstm_cell_88_matmul_readvariableop_resource:	�N
;lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource:	@�I
:lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource:	���1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
?lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_44/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0lstm_44_while_placeholderHlstm_44/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_44/while/TensorArrayV2Read/TensorListGetItem�
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOpReadVariableOp;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp�
!lstm_44/while/lstm_cell_88/MatMulMatMul8lstm_44/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_44/while/lstm_cell_88/MatMul�
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOpReadVariableOp=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp�
#lstm_44/while/lstm_cell_88/MatMul_1MatMullstm_44_while_placeholder_2:lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_44/while/lstm_cell_88/MatMul_1�
lstm_44/while/lstm_cell_88/addAddV2+lstm_44/while/lstm_cell_88/MatMul:product:0-lstm_44/while/lstm_cell_88/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_44/while/lstm_cell_88/add�
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOpReadVariableOp<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp�
"lstm_44/while/lstm_cell_88/BiasAddBiasAdd"lstm_44/while/lstm_cell_88/add:z:09lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_44/while/lstm_cell_88/BiasAdd�
*lstm_44/while/lstm_cell_88/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_44/while/lstm_cell_88/split/split_dim�
 lstm_44/while/lstm_cell_88/splitSplit3lstm_44/while/lstm_cell_88/split/split_dim:output:0+lstm_44/while/lstm_cell_88/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_44/while/lstm_cell_88/split�
"lstm_44/while/lstm_cell_88/SigmoidSigmoid)lstm_44/while/lstm_cell_88/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_44/while/lstm_cell_88/Sigmoid�
$lstm_44/while/lstm_cell_88/Sigmoid_1Sigmoid)lstm_44/while/lstm_cell_88/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_44/while/lstm_cell_88/Sigmoid_1�
lstm_44/while/lstm_cell_88/mulMul(lstm_44/while/lstm_cell_88/Sigmoid_1:y:0lstm_44_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_44/while/lstm_cell_88/mul�
lstm_44/while/lstm_cell_88/ReluRelu)lstm_44/while/lstm_cell_88/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_44/while/lstm_cell_88/Relu�
 lstm_44/while/lstm_cell_88/mul_1Mul&lstm_44/while/lstm_cell_88/Sigmoid:y:0-lstm_44/while/lstm_cell_88/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/mul_1�
 lstm_44/while/lstm_cell_88/add_1AddV2"lstm_44/while/lstm_cell_88/mul:z:0$lstm_44/while/lstm_cell_88/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/add_1�
$lstm_44/while/lstm_cell_88/Sigmoid_2Sigmoid)lstm_44/while/lstm_cell_88/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_44/while/lstm_cell_88/Sigmoid_2�
!lstm_44/while/lstm_cell_88/Relu_1Relu$lstm_44/while/lstm_cell_88/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_44/while/lstm_cell_88/Relu_1�
 lstm_44/while/lstm_cell_88/mul_2Mul(lstm_44/while/lstm_cell_88/Sigmoid_2:y:0/lstm_44/while/lstm_cell_88/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_44/while/lstm_cell_88/mul_2�
2lstm_44/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_44_while_placeholder_1lstm_44_while_placeholder$lstm_44/while/lstm_cell_88/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_44/while/TensorArrayV2Write/TensorListSetIteml
lstm_44/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_44/while/add/y�
lstm_44/while/addAddV2lstm_44_while_placeholderlstm_44/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_44/while/addp
lstm_44/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_44/while/add_1/y�
lstm_44/while/add_1AddV2(lstm_44_while_lstm_44_while_loop_counterlstm_44/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_44/while/add_1�
lstm_44/while/IdentityIdentitylstm_44/while/add_1:z:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity�
lstm_44/while/Identity_1Identity.lstm_44_while_lstm_44_while_maximum_iterations^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_1�
lstm_44/while/Identity_2Identitylstm_44/while/add:z:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_2�
lstm_44/while/Identity_3IdentityBlstm_44/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_44/while/NoOp*
T0*
_output_shapes
: 2
lstm_44/while/Identity_3�
lstm_44/while/Identity_4Identity$lstm_44/while/lstm_cell_88/mul_2:z:0^lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_44/while/Identity_4�
lstm_44/while/Identity_5Identity$lstm_44/while/lstm_cell_88/add_1:z:0^lstm_44/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_44/while/Identity_5�
lstm_44/while/NoOpNoOp2^lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp1^lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp3^lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_44/while/NoOp"9
lstm_44_while_identitylstm_44/while/Identity:output:0"=
lstm_44_while_identity_1!lstm_44/while/Identity_1:output:0"=
lstm_44_while_identity_2!lstm_44/while/Identity_2:output:0"=
lstm_44_while_identity_3!lstm_44/while/Identity_3:output:0"=
lstm_44_while_identity_4!lstm_44/while/Identity_4:output:0"=
lstm_44_while_identity_5!lstm_44/while/Identity_5:output:0"P
%lstm_44_while_lstm_44_strided_slice_1'lstm_44_while_lstm_44_strided_slice_1_0"z
:lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource<lstm_44_while_lstm_cell_88_biasadd_readvariableop_resource_0"|
;lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource=lstm_44_while_lstm_cell_88_matmul_1_readvariableop_resource_0"x
9lstm_44_while_lstm_cell_88_matmul_readvariableop_resource;lstm_44_while_lstm_cell_88_matmul_readvariableop_resource_0"�
alstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensorclstm_44_while_tensorarrayv2read_tensorlistgetitem_lstm_44_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp1lstm_44/while/lstm_cell_88/BiasAdd/ReadVariableOp2d
0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp0lstm_44/while/lstm_cell_88/MatMul/ReadVariableOp2h
2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp2lstm_44/while/lstm_cell_88/MatMul_1/ReadVariableOp: 
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
.__inference_sequential_22_layer_call_fn_685600

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
GPU 2J 8� *R
fMRK
I__inference_sequential_22_layer_call_and_return_conditional_losses_6854412
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

lstm_45_while_body_685814,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3+
'lstm_45_while_lstm_45_strided_slice_1_0g
clstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0:	@�P
=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0:	 �K
<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0:	�
lstm_45_while_identity
lstm_45_while_identity_1
lstm_45_while_identity_2
lstm_45_while_identity_3
lstm_45_while_identity_4
lstm_45_while_identity_5)
%lstm_45_while_lstm_45_strided_slice_1e
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorL
9lstm_45_while_lstm_cell_89_matmul_readvariableop_resource:	@�N
;lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource:	 �I
:lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource:	���1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_45/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0lstm_45_while_placeholderHlstm_45/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_45/while/TensorArrayV2Read/TensorListGetItem�
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOpReadVariableOp;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp�
!lstm_45/while/lstm_cell_89/MatMulMatMul8lstm_45/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_45/while/lstm_cell_89/MatMul�
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOpReadVariableOp=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp�
#lstm_45/while/lstm_cell_89/MatMul_1MatMullstm_45_while_placeholder_2:lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_45/while/lstm_cell_89/MatMul_1�
lstm_45/while/lstm_cell_89/addAddV2+lstm_45/while/lstm_cell_89/MatMul:product:0-lstm_45/while/lstm_cell_89/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_45/while/lstm_cell_89/add�
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOpReadVariableOp<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp�
"lstm_45/while/lstm_cell_89/BiasAddBiasAdd"lstm_45/while/lstm_cell_89/add:z:09lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_45/while/lstm_cell_89/BiasAdd�
*lstm_45/while/lstm_cell_89/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_45/while/lstm_cell_89/split/split_dim�
 lstm_45/while/lstm_cell_89/splitSplit3lstm_45/while/lstm_cell_89/split/split_dim:output:0+lstm_45/while/lstm_cell_89/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_45/while/lstm_cell_89/split�
"lstm_45/while/lstm_cell_89/SigmoidSigmoid)lstm_45/while/lstm_cell_89/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_45/while/lstm_cell_89/Sigmoid�
$lstm_45/while/lstm_cell_89/Sigmoid_1Sigmoid)lstm_45/while/lstm_cell_89/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_45/while/lstm_cell_89/Sigmoid_1�
lstm_45/while/lstm_cell_89/mulMul(lstm_45/while/lstm_cell_89/Sigmoid_1:y:0lstm_45_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_45/while/lstm_cell_89/mul�
lstm_45/while/lstm_cell_89/ReluRelu)lstm_45/while/lstm_cell_89/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_45/while/lstm_cell_89/Relu�
 lstm_45/while/lstm_cell_89/mul_1Mul&lstm_45/while/lstm_cell_89/Sigmoid:y:0-lstm_45/while/lstm_cell_89/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/mul_1�
 lstm_45/while/lstm_cell_89/add_1AddV2"lstm_45/while/lstm_cell_89/mul:z:0$lstm_45/while/lstm_cell_89/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/add_1�
$lstm_45/while/lstm_cell_89/Sigmoid_2Sigmoid)lstm_45/while/lstm_cell_89/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_45/while/lstm_cell_89/Sigmoid_2�
!lstm_45/while/lstm_cell_89/Relu_1Relu$lstm_45/while/lstm_cell_89/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_45/while/lstm_cell_89/Relu_1�
 lstm_45/while/lstm_cell_89/mul_2Mul(lstm_45/while/lstm_cell_89/Sigmoid_2:y:0/lstm_45/while/lstm_cell_89/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_45/while/lstm_cell_89/mul_2�
2lstm_45/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_45_while_placeholder_1lstm_45_while_placeholder$lstm_45/while/lstm_cell_89/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_45/while/TensorArrayV2Write/TensorListSetIteml
lstm_45/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_45/while/add/y�
lstm_45/while/addAddV2lstm_45_while_placeholderlstm_45/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_45/while/addp
lstm_45/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_45/while/add_1/y�
lstm_45/while/add_1AddV2(lstm_45_while_lstm_45_while_loop_counterlstm_45/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_45/while/add_1�
lstm_45/while/IdentityIdentitylstm_45/while/add_1:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity�
lstm_45/while/Identity_1Identity.lstm_45_while_lstm_45_while_maximum_iterations^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_1�
lstm_45/while/Identity_2Identitylstm_45/while/add:z:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_2�
lstm_45/while/Identity_3IdentityBlstm_45/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_45/while/NoOp*
T0*
_output_shapes
: 2
lstm_45/while/Identity_3�
lstm_45/while/Identity_4Identity$lstm_45/while/lstm_cell_89/mul_2:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_45/while/Identity_4�
lstm_45/while/Identity_5Identity$lstm_45/while/lstm_cell_89/add_1:z:0^lstm_45/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_45/while/Identity_5�
lstm_45/while/NoOpNoOp2^lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp1^lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp3^lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_45/while/NoOp"9
lstm_45_while_identitylstm_45/while/Identity:output:0"=
lstm_45_while_identity_1!lstm_45/while/Identity_1:output:0"=
lstm_45_while_identity_2!lstm_45/while/Identity_2:output:0"=
lstm_45_while_identity_3!lstm_45/while/Identity_3:output:0"=
lstm_45_while_identity_4!lstm_45/while/Identity_4:output:0"=
lstm_45_while_identity_5!lstm_45/while/Identity_5:output:0"P
%lstm_45_while_lstm_45_strided_slice_1'lstm_45_while_lstm_45_strided_slice_1_0"z
:lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource<lstm_45_while_lstm_cell_89_biasadd_readvariableop_resource_0"|
;lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource=lstm_45_while_lstm_cell_89_matmul_1_readvariableop_resource_0"x
9lstm_45_while_lstm_cell_89_matmul_readvariableop_resource;lstm_45_while_lstm_cell_89_matmul_readvariableop_resource_0"�
alstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensorclstm_45_while_tensorarrayv2read_tensorlistgetitem_lstm_45_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp1lstm_45/while/lstm_cell_89/BiasAdd/ReadVariableOp2d
0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp0lstm_45/while/lstm_cell_89/MatMul/ReadVariableOp2h
2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp2lstm_45/while/lstm_cell_89/MatMul_1/ReadVariableOp: 
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
while_cond_686975
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_686975___redundant_placeholder04
0while_while_cond_686975___redundant_placeholder14
0while_while_cond_686975___redundant_placeholder24
0while_while_cond_686975___redundant_placeholder3
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
lstm_44_while_cond_685666,
(lstm_44_while_lstm_44_while_loop_counter2
.lstm_44_while_lstm_44_while_maximum_iterations
lstm_44_while_placeholder
lstm_44_while_placeholder_1
lstm_44_while_placeholder_2
lstm_44_while_placeholder_3.
*lstm_44_while_less_lstm_44_strided_slice_1D
@lstm_44_while_lstm_44_while_cond_685666___redundant_placeholder0D
@lstm_44_while_lstm_44_while_cond_685666___redundant_placeholder1D
@lstm_44_while_lstm_44_while_cond_685666___redundant_placeholder2D
@lstm_44_while_lstm_44_while_cond_685666___redundant_placeholder3
lstm_44_while_identity
�
lstm_44/while/LessLesslstm_44_while_placeholder*lstm_44_while_less_lstm_44_strided_slice_1*
T0*
_output_shapes
: 2
lstm_44/while/Lessu
lstm_44/while/IdentityIdentitylstm_44/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_44/while/Identity"9
lstm_44_while_identitylstm_44/while/Identity:output:0*(
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

�
lstm_45_while_cond_686118,
(lstm_45_while_lstm_45_while_loop_counter2
.lstm_45_while_lstm_45_while_maximum_iterations
lstm_45_while_placeholder
lstm_45_while_placeholder_1
lstm_45_while_placeholder_2
lstm_45_while_placeholder_3.
*lstm_45_while_less_lstm_45_strided_slice_1D
@lstm_45_while_lstm_45_while_cond_686118___redundant_placeholder0D
@lstm_45_while_lstm_45_while_cond_686118___redundant_placeholder1D
@lstm_45_while_lstm_45_while_cond_686118___redundant_placeholder2D
@lstm_45_while_lstm_45_while_cond_686118___redundant_placeholder3
lstm_45_while_identity
�
lstm_45/while/LessLesslstm_45_while_placeholder*lstm_45_while_less_lstm_45_strided_slice_1*
T0*
_output_shapes
: 2
lstm_45/while/Lessu
lstm_45/while/IdentityIdentitylstm_45/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_45/while/Identity"9
lstm_45_while_identitylstm_45/while/Identity:output:0*(
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_684240

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
�
�
while_cond_684317
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_684317___redundant_placeholder04
0while_while_cond_684317___redundant_placeholder14
0while_while_cond_684317___redundant_placeholder24
0while_while_cond_684317___redundant_placeholder3
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687625

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
states/1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_44_input:
serving_default_lstm_44_input:0���������<
dense_220
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
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
!: 2dense_22/kernel
:2dense_22/bias
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
.:,	�2lstm_44/lstm_cell_88/kernel
8:6	@�2%lstm_44/lstm_cell_88/recurrent_kernel
(:&�2lstm_44/lstm_cell_88/bias
.:,	@�2lstm_45/lstm_cell_89/kernel
8:6	 �2%lstm_45/lstm_cell_89/recurrent_kernel
(:&�2lstm_45/lstm_cell_89/bias
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
&:$ 2Adam/dense_22/kernel/m
 :2Adam/dense_22/bias/m
3:1	�2"Adam/lstm_44/lstm_cell_88/kernel/m
=:;	@�2,Adam/lstm_44/lstm_cell_88/recurrent_kernel/m
-:+�2 Adam/lstm_44/lstm_cell_88/bias/m
3:1	@�2"Adam/lstm_45/lstm_cell_89/kernel/m
=:;	 �2,Adam/lstm_45/lstm_cell_89/recurrent_kernel/m
-:+�2 Adam/lstm_45/lstm_cell_89/bias/m
&:$ 2Adam/dense_22/kernel/v
 :2Adam/dense_22/bias/v
3:1	�2"Adam/lstm_44/lstm_cell_88/kernel/v
=:;	@�2,Adam/lstm_44/lstm_cell_88/recurrent_kernel/v
-:+�2 Adam/lstm_44/lstm_cell_88/bias/v
3:1	@�2"Adam/lstm_45/lstm_cell_89/kernel/v
=:;	 �2,Adam/lstm_45/lstm_cell_89/recurrent_kernel/v
-:+�2 Adam/lstm_45/lstm_cell_89/bias/v
�B�
!__inference__wrapped_model_683389lstm_44_input"�
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
�2�
.__inference_sequential_22_layer_call_fn_685015
.__inference_sequential_22_layer_call_fn_685579
.__inference_sequential_22_layer_call_fn_685600
.__inference_sequential_22_layer_call_fn_685481�
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
I__inference_sequential_22_layer_call_and_return_conditional_losses_685905
I__inference_sequential_22_layer_call_and_return_conditional_losses_686217
I__inference_sequential_22_layer_call_and_return_conditional_losses_685505
I__inference_sequential_22_layer_call_and_return_conditional_losses_685529�
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
(__inference_lstm_44_layer_call_fn_686228
(__inference_lstm_44_layer_call_fn_686239
(__inference_lstm_44_layer_call_fn_686250
(__inference_lstm_44_layer_call_fn_686261�
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_686412
C__inference_lstm_44_layer_call_and_return_conditional_losses_686563
C__inference_lstm_44_layer_call_and_return_conditional_losses_686714
C__inference_lstm_44_layer_call_and_return_conditional_losses_686865�
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
(__inference_lstm_45_layer_call_fn_686876
(__inference_lstm_45_layer_call_fn_686887
(__inference_lstm_45_layer_call_fn_686898
(__inference_lstm_45_layer_call_fn_686909�
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_687060
C__inference_lstm_45_layer_call_and_return_conditional_losses_687211
C__inference_lstm_45_layer_call_and_return_conditional_losses_687362
C__inference_lstm_45_layer_call_and_return_conditional_losses_687513�
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
+__inference_dropout_22_layer_call_fn_687518
+__inference_dropout_22_layer_call_fn_687523�
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
F__inference_dropout_22_layer_call_and_return_conditional_losses_687528
F__inference_dropout_22_layer_call_and_return_conditional_losses_687540�
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
)__inference_dense_22_layer_call_fn_687549�
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
D__inference_dense_22_layer_call_and_return_conditional_losses_687559�
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
$__inference_signature_wrapper_685558lstm_44_input"�
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
-__inference_lstm_cell_88_layer_call_fn_687576
-__inference_lstm_cell_88_layer_call_fn_687593�
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687625
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687657�
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
-__inference_lstm_cell_89_layer_call_fn_687674
-__inference_lstm_cell_89_layer_call_fn_687691�
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687723
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687755�
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
!__inference__wrapped_model_683389{&'()*+:�7
0�-
+�(
lstm_44_input���������
� "3�0
.
dense_22"�
dense_22����������
D__inference_dense_22_layer_call_and_return_conditional_losses_687559\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� |
)__inference_dense_22_layer_call_fn_687549O/�,
%�"
 �
inputs��������� 
� "�����������
F__inference_dropout_22_layer_call_and_return_conditional_losses_687528\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
F__inference_dropout_22_layer_call_and_return_conditional_losses_687540\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� ~
+__inference_dropout_22_layer_call_fn_687518O3�0
)�&
 �
inputs��������� 
p 
� "���������� ~
+__inference_dropout_22_layer_call_fn_687523O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_lstm_44_layer_call_and_return_conditional_losses_686412�&'(O�L
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_686563�&'(O�L
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_686714q&'(?�<
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
C__inference_lstm_44_layer_call_and_return_conditional_losses_686865q&'(?�<
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
(__inference_lstm_44_layer_call_fn_686228}&'(O�L
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
(__inference_lstm_44_layer_call_fn_686239}&'(O�L
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
(__inference_lstm_44_layer_call_fn_686250d&'(?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
(__inference_lstm_44_layer_call_fn_686261d&'(?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
C__inference_lstm_45_layer_call_and_return_conditional_losses_687060})*+O�L
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_687211})*+O�L
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_687362m)*+?�<
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
C__inference_lstm_45_layer_call_and_return_conditional_losses_687513m)*+?�<
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
(__inference_lstm_45_layer_call_fn_686876p)*+O�L
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
(__inference_lstm_45_layer_call_fn_686887p)*+O�L
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
(__inference_lstm_45_layer_call_fn_686898`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
(__inference_lstm_45_layer_call_fn_686909`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687625�&'(��}
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
H__inference_lstm_cell_88_layer_call_and_return_conditional_losses_687657�&'(��}
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
-__inference_lstm_cell_88_layer_call_fn_687576�&'(��}
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
-__inference_lstm_cell_88_layer_call_fn_687593�&'(��}
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687723�)*+��}
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
H__inference_lstm_cell_89_layer_call_and_return_conditional_losses_687755�)*+��}
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
-__inference_lstm_cell_89_layer_call_fn_687674�)*+��}
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
-__inference_lstm_cell_89_layer_call_fn_687691�)*+��}
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
I__inference_sequential_22_layer_call_and_return_conditional_losses_685505u&'()*+B�?
8�5
+�(
lstm_44_input���������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_22_layer_call_and_return_conditional_losses_685529u&'()*+B�?
8�5
+�(
lstm_44_input���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_22_layer_call_and_return_conditional_losses_685905n&'()*+;�8
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
I__inference_sequential_22_layer_call_and_return_conditional_losses_686217n&'()*+;�8
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
.__inference_sequential_22_layer_call_fn_685015h&'()*+B�?
8�5
+�(
lstm_44_input���������
p 

 
� "�����������
.__inference_sequential_22_layer_call_fn_685481h&'()*+B�?
8�5
+�(
lstm_44_input���������
p

 
� "�����������
.__inference_sequential_22_layer_call_fn_685579a&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "�����������
.__inference_sequential_22_layer_call_fn_685600a&'()*+;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_685558�&'()*+K�H
� 
A�>
<
lstm_44_input+�(
lstm_44_input���������"3�0
.
dense_22"�
dense_22���������