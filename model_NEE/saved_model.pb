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
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

: *
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
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
lstm_24/lstm_cell_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_24/lstm_cell_48/kernel
�
/lstm_24/lstm_cell_48/kernel/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_48/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_24/lstm_cell_48/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*6
shared_name'%lstm_24/lstm_cell_48/recurrent_kernel
�
9lstm_24/lstm_cell_48/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_24/lstm_cell_48/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_24/lstm_cell_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_24/lstm_cell_48/bias
�
-lstm_24/lstm_cell_48/bias/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_48/bias*
_output_shapes	
:�*
dtype0
�
lstm_25/lstm_cell_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*,
shared_namelstm_25/lstm_cell_49/kernel
�
/lstm_25/lstm_cell_49/kernel/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_49/kernel*
_output_shapes
:	@�*
dtype0
�
%lstm_25/lstm_cell_49/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%lstm_25/lstm_cell_49/recurrent_kernel
�
9lstm_25/lstm_cell_49/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_25/lstm_cell_49/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
lstm_25/lstm_cell_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_25/lstm_cell_49/bias
�
-lstm_25/lstm_cell_49/bias/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_49/bias*
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_12/kernel/m
�
*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_24/lstm_cell_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_24/lstm_cell_48/kernel/m
�
6Adam/lstm_24/lstm_cell_48/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_48/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_24/lstm_cell_48/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_24/lstm_cell_48/recurrent_kernel/m
�
@Adam/lstm_24/lstm_cell_48/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_48/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_24/lstm_cell_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_24/lstm_cell_48/bias/m
�
4Adam/lstm_24/lstm_cell_48/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_48/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_25/lstm_cell_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_25/lstm_cell_49/kernel/m
�
6Adam/lstm_25/lstm_cell_49/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_49/kernel/m*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_25/lstm_cell_49/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_25/lstm_cell_49/recurrent_kernel/m
�
@Adam/lstm_25/lstm_cell_49/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_49/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_25/lstm_cell_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_25/lstm_cell_49/bias/m
�
4Adam/lstm_25/lstm_cell_49/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_49/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_12/kernel/v
�
*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_24/lstm_cell_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_24/lstm_cell_48/kernel/v
�
6Adam/lstm_24/lstm_cell_48/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_48/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_24/lstm_cell_48/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_24/lstm_cell_48/recurrent_kernel/v
�
@Adam/lstm_24/lstm_cell_48/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_48/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_24/lstm_cell_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_24/lstm_cell_48/bias/v
�
4Adam/lstm_24/lstm_cell_48/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_48/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_25/lstm_cell_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_25/lstm_cell_49/kernel/v
�
6Adam/lstm_25/lstm_cell_49/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_49/kernel/v*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_25/lstm_cell_49/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_25/lstm_cell_49/recurrent_kernel/v
�
@Adam/lstm_25/lstm_cell_49/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_49/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_25/lstm_cell_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_25/lstm_cell_49/bias/v
�
4Adam/lstm_25/lstm_cell_49/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_49/bias/v*
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
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_24/lstm_cell_48/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_24/lstm_cell_48/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_24/lstm_cell_48/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_25/lstm_cell_49/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_25/lstm_cell_49/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_25/lstm_cell_49/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_24/lstm_cell_48/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_24/lstm_cell_48/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_24/lstm_cell_48/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_25/lstm_cell_49/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_25/lstm_cell_49/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_25/lstm_cell_49/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_24/lstm_cell_48/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_24/lstm_cell_48/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_24/lstm_cell_48/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_25/lstm_cell_49/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_25/lstm_cell_49/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_25/lstm_cell_49/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_24_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_24_inputlstm_24/lstm_cell_48/kernel%lstm_24/lstm_cell_48/recurrent_kernellstm_24/lstm_cell_48/biaslstm_25/lstm_cell_49/kernel%lstm_25/lstm_cell_49/recurrent_kernellstm_25/lstm_cell_49/biasdense_12/kerneldense_12/bias*
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
$__inference_signature_wrapper_418496
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_24/lstm_cell_48/kernel/Read/ReadVariableOp9lstm_24/lstm_cell_48/recurrent_kernel/Read/ReadVariableOp-lstm_24/lstm_cell_48/bias/Read/ReadVariableOp/lstm_25/lstm_cell_49/kernel/Read/ReadVariableOp9lstm_25/lstm_cell_49/recurrent_kernel/Read/ReadVariableOp-lstm_25/lstm_cell_49/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_48/kernel/m/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_48/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_48/bias/m/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_49/kernel/m/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_49/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_49/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_48/kernel/v/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_48/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_48/bias/v/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_49/kernel/v/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_49/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_49/bias/v/Read/ReadVariableOpConst*,
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
__inference__traced_save_420809
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_24/lstm_cell_48/kernel%lstm_24/lstm_cell_48/recurrent_kernellstm_24/lstm_cell_48/biaslstm_25/lstm_cell_49/kernel%lstm_25/lstm_cell_49/recurrent_kernellstm_25/lstm_cell_49/biastotalcountAdam/dense_12/kernel/mAdam/dense_12/bias/m"Adam/lstm_24/lstm_cell_48/kernel/m,Adam/lstm_24/lstm_cell_48/recurrent_kernel/m Adam/lstm_24/lstm_cell_48/bias/m"Adam/lstm_25/lstm_cell_49/kernel/m,Adam/lstm_25/lstm_cell_49/recurrent_kernel/m Adam/lstm_25/lstm_cell_49/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/v"Adam/lstm_24/lstm_cell_48/kernel/v,Adam/lstm_24/lstm_cell_48/recurrent_kernel/v Adam/lstm_24/lstm_cell_48/bias/v"Adam/lstm_25/lstm_cell_49/kernel/v,Adam/lstm_25/lstm_cell_49/recurrent_kernel/v Adam/lstm_25/lstm_cell_49/bias/v*+
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
"__inference__traced_restore_420912��#
�
�
while_cond_417817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_417817___redundant_placeholder04
0while_while_cond_417817___redundant_placeholder14
0while_while_cond_417817___redundant_placeholder24
0while_while_cond_417817___redundant_placeholder3
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_420451

inputs>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_420367*
condR
while_cond_420366*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_lstm_24_layer_call_fn_419199

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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4183232
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
�?
�
while_body_418066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
while_cond_419718
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_419718___redundant_placeholder04
0while_while_cond_419718___redundant_placeholder14
0while_while_cond_419718___redundant_placeholder24
0while_while_cond_419718___redundant_placeholder3
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
�
'sequential_12_lstm_25_while_cond_416235H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3J
Fsequential_12_lstm_25_while_less_sequential_12_lstm_25_strided_slice_1`
\sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_416235___redundant_placeholder0`
\sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_416235___redundant_placeholder1`
\sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_416235___redundant_placeholder2`
\sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_416235___redundant_placeholder3(
$sequential_12_lstm_25_while_identity
�
 sequential_12/lstm_25/while/LessLess'sequential_12_lstm_25_while_placeholderFsequential_12_lstm_25_while_less_sequential_12_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_25/while/Less�
$sequential_12/lstm_25/while/IdentityIdentity$sequential_12/lstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_25/while/Identity"U
$sequential_12_lstm_25_while_identity-sequential_12/lstm_25/while/Identity:output:0*(
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
while_cond_418238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_418238___redundant_placeholder04
0while_while_cond_418238___redundant_placeholder14
0while_while_cond_418238___redundant_placeholder24
0while_while_cond_418238___redundant_placeholder3
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
�
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_418443
lstm_24_input!
lstm_24_418422:	�!
lstm_24_418424:	@�
lstm_24_418426:	�!
lstm_25_418429:	@�!
lstm_25_418431:	 �
lstm_25_418433:	�!
dense_12_418437: 
dense_12_418439:
identity�� dense_12/StatefulPartitionedCall�lstm_24/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_418422lstm_24_418424lstm_24_418426*
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4177442!
lstm_24/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0lstm_25_418429lstm_25_418431lstm_25_418433*
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4179022!
lstm_25/StatefulPartitionedCall�
dropout_12/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179152
dropout_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_12_418437dense_12_418439*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_4179272"
 dense_12/StatefulPartitionedCall�
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_24_input
�
�
(__inference_lstm_24_layer_call_fn_419188

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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4177442
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
�[
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_419803

inputs>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_419719*
condR
while_cond_419718*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_418239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420595

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
�
�
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_416548

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
�J
�

lstm_24_while_body_418910,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0:	�P
=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�K
<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_48_matmul_readvariableop_resource:	�N
;lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource:	@�I
:lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource:	���1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_24/while/TensorArrayV2Read/TensorListGetItem�
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�
!lstm_24/while/lstm_cell_48/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_24/while/lstm_cell_48/MatMul�
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
#lstm_24/while/lstm_cell_48/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_24/while/lstm_cell_48/MatMul_1�
lstm_24/while/lstm_cell_48/addAddV2+lstm_24/while/lstm_cell_48/MatMul:product:0-lstm_24/while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_24/while/lstm_cell_48/add�
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�
"lstm_24/while/lstm_cell_48/BiasAddBiasAdd"lstm_24/while/lstm_cell_48/add:z:09lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_24/while/lstm_cell_48/BiasAdd�
*lstm_24/while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_48/split/split_dim�
 lstm_24/while/lstm_cell_48/splitSplit3lstm_24/while/lstm_cell_48/split/split_dim:output:0+lstm_24/while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_24/while/lstm_cell_48/split�
"lstm_24/while/lstm_cell_48/SigmoidSigmoid)lstm_24/while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_24/while/lstm_cell_48/Sigmoid�
$lstm_24/while/lstm_cell_48/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_24/while/lstm_cell_48/Sigmoid_1�
lstm_24/while/lstm_cell_48/mulMul(lstm_24/while/lstm_cell_48/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_24/while/lstm_cell_48/mul�
lstm_24/while/lstm_cell_48/ReluRelu)lstm_24/while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_24/while/lstm_cell_48/Relu�
 lstm_24/while/lstm_cell_48/mul_1Mul&lstm_24/while/lstm_cell_48/Sigmoid:y:0-lstm_24/while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/mul_1�
 lstm_24/while/lstm_cell_48/add_1AddV2"lstm_24/while/lstm_cell_48/mul:z:0$lstm_24/while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/add_1�
$lstm_24/while/lstm_cell_48/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_24/while/lstm_cell_48/Sigmoid_2�
!lstm_24/while/lstm_cell_48/Relu_1Relu$lstm_24/while/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_24/while/lstm_cell_48/Relu_1�
 lstm_24/while/lstm_cell_48/mul_2Mul(lstm_24/while/lstm_cell_48/Sigmoid_2:y:0/lstm_24/while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/mul_2�
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder$lstm_24/while/lstm_cell_48/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_24/while/TensorArrayV2Write/TensorListSetIteml
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add/y�
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/addp
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add_1/y�
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/add_1�
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity�
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_1�
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_2�
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_3�
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_48/mul_2:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_24/while/Identity_4�
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_48/add_1:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_24/while/Identity_5�
lstm_24/while/NoOpNoOp2^lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp1^lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp3^lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_24/while/NoOp"9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"z
:lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0"|
;lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0"x
9lstm_24_while_lstm_cell_48_matmul_readvariableop_resource;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0"�
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp2d
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp2h
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
-__inference_lstm_cell_49_layer_call_fn_420629

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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4171782
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
�
�
)__inference_dense_12_layer_call_fn_420487

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
D__inference_dense_12_layer_call_and_return_conditional_losses_4179272
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
��
�
"__inference__traced_restore_420912
file_prefix2
 assignvariableop_dense_12_kernel: .
 assignvariableop_1_dense_12_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_24_lstm_cell_48_kernel:	�K
8assignvariableop_8_lstm_24_lstm_cell_48_recurrent_kernel:	@�;
,assignvariableop_9_lstm_24_lstm_cell_48_bias:	�B
/assignvariableop_10_lstm_25_lstm_cell_49_kernel:	@�L
9assignvariableop_11_lstm_25_lstm_cell_49_recurrent_kernel:	 �<
-assignvariableop_12_lstm_25_lstm_cell_49_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_12_kernel_m: 6
(assignvariableop_16_adam_dense_12_bias_m:I
6assignvariableop_17_adam_lstm_24_lstm_cell_48_kernel_m:	�S
@assignvariableop_18_adam_lstm_24_lstm_cell_48_recurrent_kernel_m:	@�C
4assignvariableop_19_adam_lstm_24_lstm_cell_48_bias_m:	�I
6assignvariableop_20_adam_lstm_25_lstm_cell_49_kernel_m:	@�S
@assignvariableop_21_adam_lstm_25_lstm_cell_49_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_lstm_25_lstm_cell_49_bias_m:	�<
*assignvariableop_23_adam_dense_12_kernel_v: 6
(assignvariableop_24_adam_dense_12_bias_v:I
6assignvariableop_25_adam_lstm_24_lstm_cell_48_kernel_v:	�S
@assignvariableop_26_adam_lstm_24_lstm_cell_48_recurrent_kernel_v:	@�C
4assignvariableop_27_adam_lstm_24_lstm_cell_48_bias_v:	�I
6assignvariableop_28_adam_lstm_25_lstm_cell_49_kernel_v:	@�S
@assignvariableop_29_adam_lstm_25_lstm_cell_49_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_lstm_25_lstm_cell_49_bias_v:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_24_lstm_cell_48_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_24_lstm_cell_48_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_24_lstm_cell_48_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_25_lstm_cell_49_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_25_lstm_cell_49_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_25_lstm_cell_49_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_12_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_12_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_24_lstm_cell_48_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_24_lstm_cell_48_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_24_lstm_cell_48_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_25_lstm_cell_49_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_25_lstm_cell_49_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_25_lstm_cell_49_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_12_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_12_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_24_lstm_cell_48_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_24_lstm_cell_48_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_24_lstm_cell_48_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_25_lstm_cell_49_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_25_lstm_cell_49_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_25_lstm_cell_49_bias_vIdentity_30:output:0"/device:CPU:0*
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

�
lstm_25_while_cond_418751,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1D
@lstm_25_while_lstm_25_while_cond_418751___redundant_placeholder0D
@lstm_25_while_lstm_25_while_cond_418751___redundant_placeholder1D
@lstm_25_while_lstm_25_while_cond_418751___redundant_placeholder2D
@lstm_25_while_lstm_25_while_cond_418751___redundant_placeholder3
lstm_25_while_identity
�
lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2
lstm_25/while/Lessu
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_25/while/Identity"9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
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

lstm_24_while_body_418605,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0:	�P
=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�K
<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_48_matmul_readvariableop_resource:	�N
;lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource:	@�I
:lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource:	���1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_24/while/TensorArrayV2Read/TensorListGetItem�
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�
!lstm_24/while/lstm_cell_48/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_24/while/lstm_cell_48/MatMul�
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
#lstm_24/while/lstm_cell_48/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_24/while/lstm_cell_48/MatMul_1�
lstm_24/while/lstm_cell_48/addAddV2+lstm_24/while/lstm_cell_48/MatMul:product:0-lstm_24/while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_24/while/lstm_cell_48/add�
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�
"lstm_24/while/lstm_cell_48/BiasAddBiasAdd"lstm_24/while/lstm_cell_48/add:z:09lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_24/while/lstm_cell_48/BiasAdd�
*lstm_24/while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_48/split/split_dim�
 lstm_24/while/lstm_cell_48/splitSplit3lstm_24/while/lstm_cell_48/split/split_dim:output:0+lstm_24/while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_24/while/lstm_cell_48/split�
"lstm_24/while/lstm_cell_48/SigmoidSigmoid)lstm_24/while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_24/while/lstm_cell_48/Sigmoid�
$lstm_24/while/lstm_cell_48/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_24/while/lstm_cell_48/Sigmoid_1�
lstm_24/while/lstm_cell_48/mulMul(lstm_24/while/lstm_cell_48/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_24/while/lstm_cell_48/mul�
lstm_24/while/lstm_cell_48/ReluRelu)lstm_24/while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_24/while/lstm_cell_48/Relu�
 lstm_24/while/lstm_cell_48/mul_1Mul&lstm_24/while/lstm_cell_48/Sigmoid:y:0-lstm_24/while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/mul_1�
 lstm_24/while/lstm_cell_48/add_1AddV2"lstm_24/while/lstm_cell_48/mul:z:0$lstm_24/while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/add_1�
$lstm_24/while/lstm_cell_48/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_24/while/lstm_cell_48/Sigmoid_2�
!lstm_24/while/lstm_cell_48/Relu_1Relu$lstm_24/while/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_24/while/lstm_cell_48/Relu_1�
 lstm_24/while/lstm_cell_48/mul_2Mul(lstm_24/while/lstm_cell_48/Sigmoid_2:y:0/lstm_24/while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_24/while/lstm_cell_48/mul_2�
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder$lstm_24/while/lstm_cell_48/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_24/while/TensorArrayV2Write/TensorListSetIteml
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add/y�
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/addp
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add_1/y�
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/add_1�
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity�
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_1�
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_2�
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_3�
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_48/mul_2:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_24/while/Identity_4�
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_48/add_1:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_24/while/Identity_5�
lstm_24/while/NoOpNoOp2^lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp1^lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp3^lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_24/while/NoOp"9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"z
:lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource<lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0"|
;lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource=lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0"x
9lstm_24_while_lstm_cell_48_matmul_readvariableop_resource;lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0"�
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp1lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp2d
0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp0lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp2h
2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp2lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
while_body_419568
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
�^
�
'sequential_12_lstm_25_while_body_416236H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3G
Csequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1_0�
sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0:	@�^
Ksequential_12_lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �Y
Jsequential_12_lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0:	�(
$sequential_12_lstm_25_while_identity*
&sequential_12_lstm_25_while_identity_1*
&sequential_12_lstm_25_while_identity_2*
&sequential_12_lstm_25_while_identity_3*
&sequential_12_lstm_25_while_identity_4*
&sequential_12_lstm_25_while_identity_5E
Asequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1�
}sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_25_while_lstm_cell_49_matmul_readvariableop_resource:	@�\
Isequential_12_lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource:	 �W
Hsequential_12_lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource:	���?sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�>sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�@sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2O
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_25_while_placeholderVsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02A
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem�
>sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02@
>sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�
/sequential_12/lstm_25/while/lstm_cell_49/MatMulMatMulFsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_12/lstm_25/while/lstm_cell_49/MatMul�
@sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02B
@sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
1sequential_12/lstm_25/while/lstm_cell_49/MatMul_1MatMul)sequential_12_lstm_25_while_placeholder_2Hsequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_12/lstm_25/while/lstm_cell_49/MatMul_1�
,sequential_12/lstm_25/while/lstm_cell_49/addAddV29sequential_12/lstm_25/while/lstm_cell_49/MatMul:product:0;sequential_12/lstm_25/while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_12/lstm_25/while/lstm_cell_49/add�
?sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�
0sequential_12/lstm_25/while/lstm_cell_49/BiasAddBiasAdd0sequential_12/lstm_25/while/lstm_cell_49/add:z:0Gsequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_12/lstm_25/while/lstm_cell_49/BiasAdd�
8sequential_12/lstm_25/while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_25/while/lstm_cell_49/split/split_dim�
.sequential_12/lstm_25/while/lstm_cell_49/splitSplitAsequential_12/lstm_25/while/lstm_cell_49/split/split_dim:output:09sequential_12/lstm_25/while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split20
.sequential_12/lstm_25/while/lstm_cell_49/split�
0sequential_12/lstm_25/while/lstm_cell_49/SigmoidSigmoid7sequential_12/lstm_25/while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 22
0sequential_12/lstm_25/while/lstm_cell_49/Sigmoid�
2sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_1Sigmoid7sequential_12/lstm_25/while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 24
2sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_1�
,sequential_12/lstm_25/while/lstm_cell_49/mulMul6sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_1:y:0)sequential_12_lstm_25_while_placeholder_3*
T0*'
_output_shapes
:��������� 2.
,sequential_12/lstm_25/while/lstm_cell_49/mul�
-sequential_12/lstm_25/while/lstm_cell_49/ReluRelu7sequential_12/lstm_25/while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2/
-sequential_12/lstm_25/while/lstm_cell_49/Relu�
.sequential_12/lstm_25/while/lstm_cell_49/mul_1Mul4sequential_12/lstm_25/while/lstm_cell_49/Sigmoid:y:0;sequential_12/lstm_25/while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_12/lstm_25/while/lstm_cell_49/mul_1�
.sequential_12/lstm_25/while/lstm_cell_49/add_1AddV20sequential_12/lstm_25/while/lstm_cell_49/mul:z:02sequential_12/lstm_25/while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 20
.sequential_12/lstm_25/while/lstm_cell_49/add_1�
2sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_2Sigmoid7sequential_12/lstm_25/while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 24
2sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_2�
/sequential_12/lstm_25/while/lstm_cell_49/Relu_1Relu2sequential_12/lstm_25/while/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 21
/sequential_12/lstm_25/while/lstm_cell_49/Relu_1�
.sequential_12/lstm_25/while/lstm_cell_49/mul_2Mul6sequential_12/lstm_25/while/lstm_cell_49/Sigmoid_2:y:0=sequential_12/lstm_25/while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_12/lstm_25/while/lstm_cell_49/mul_2�
@sequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_25_while_placeholder_1'sequential_12_lstm_25_while_placeholder2sequential_12/lstm_25/while/lstm_cell_49/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItem�
!sequential_12/lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_25/while/add/y�
sequential_12/lstm_25/while/addAddV2'sequential_12_lstm_25_while_placeholder*sequential_12/lstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_25/while/add�
#sequential_12/lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_25/while/add_1/y�
!sequential_12/lstm_25/while/add_1AddV2Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counter,sequential_12/lstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_25/while/add_1�
$sequential_12/lstm_25/while/IdentityIdentity%sequential_12/lstm_25/while/add_1:z:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_25/while/Identity�
&sequential_12/lstm_25/while/Identity_1IdentityJsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_1�
&sequential_12/lstm_25/while/Identity_2Identity#sequential_12/lstm_25/while/add:z:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_2�
&sequential_12/lstm_25/while/Identity_3IdentityPsequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_3�
&sequential_12/lstm_25/while/Identity_4Identity2sequential_12/lstm_25/while/lstm_cell_49/mul_2:z:0!^sequential_12/lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_12/lstm_25/while/Identity_4�
&sequential_12/lstm_25/while/Identity_5Identity2sequential_12/lstm_25/while/lstm_cell_49/add_1:z:0!^sequential_12/lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_12/lstm_25/while/Identity_5�
 sequential_12/lstm_25/while/NoOpNoOp@^sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp?^sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOpA^sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_25/while/NoOp"U
$sequential_12_lstm_25_while_identity-sequential_12/lstm_25/while/Identity:output:0"Y
&sequential_12_lstm_25_while_identity_1/sequential_12/lstm_25/while/Identity_1:output:0"Y
&sequential_12_lstm_25_while_identity_2/sequential_12/lstm_25/while/Identity_2:output:0"Y
&sequential_12_lstm_25_while_identity_3/sequential_12/lstm_25/while/Identity_3:output:0"Y
&sequential_12_lstm_25_while_identity_4/sequential_12/lstm_25/while/Identity_4:output:0"Y
&sequential_12_lstm_25_while_identity_5/sequential_12/lstm_25/while/Identity_5:output:0"�
Hsequential_12_lstm_25_while_lstm_cell_49_biasadd_readvariableop_resourceJsequential_12_lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0"�
Isequential_12_lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resourceKsequential_12_lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0"�
Gsequential_12_lstm_25_while_lstm_cell_49_matmul_readvariableop_resourceIsequential_12_lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0"�
Asequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1Csequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1_0"�
}sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2�
?sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp?sequential_12/lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp2�
>sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp>sequential_12/lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp2�
@sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp@sequential_12/lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
�F
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_416695

inputs&
lstm_cell_48_416613:	�&
lstm_cell_48_416615:	@�"
lstm_cell_48_416617:	�
identity��$lstm_cell_48/StatefulPartitionedCall�whileD
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
$lstm_cell_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_48_416613lstm_cell_48_416615lstm_cell_48_416617*
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4165482&
$lstm_cell_48/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_48_416613lstm_cell_48_416615lstm_cell_48_416617*
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
while_body_416626*
condR
while_cond_416625*K
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
NoOpNoOp%^lstm_cell_48/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_48/StatefulPartitionedCall$lstm_cell_48/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
while_cond_417045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_417045___redundant_placeholder04
0while_while_cond_417045___redundant_placeholder14
0while_while_cond_417045___redundant_placeholder24
0while_while_cond_417045___redundant_placeholder3
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_417902

inputs>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_417818*
condR
while_cond_417817*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_418379

inputs!
lstm_24_418358:	�!
lstm_24_418360:	@�
lstm_24_418362:	�!
lstm_25_418365:	@�!
lstm_25_418367:	 �
lstm_25_418369:	�!
dense_12_418373: 
dense_12_418375:
identity�� dense_12/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�lstm_24/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_418358lstm_24_418360lstm_24_418362*
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4183232!
lstm_24/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0lstm_25_418365lstm_25_418367lstm_25_418369*
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4181502!
lstm_25/StatefulPartitionedCall�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179832$
"dropout_12/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_12_418373dense_12_418375*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_4179272"
 dense_12/StatefulPartitionedCall�
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_12/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_417983

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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420661

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
while_cond_416625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_416625___redundant_placeholder04
0while_while_cond_416625___redundant_placeholder14
0while_while_cond_416625___redundant_placeholder24
0while_while_cond_416625___redundant_placeholder3
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
while_cond_416415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_416415___redundant_placeholder04
0while_while_cond_416415___redundant_placeholder14
0while_while_cond_416415___redundant_placeholder24
0while_while_cond_416415___redundant_placeholder3
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
�
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_417915

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
�

�
lstm_24_while_cond_418909,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1D
@lstm_24_while_lstm_24_while_cond_418909___redundant_placeholder0D
@lstm_24_while_lstm_24_while_cond_418909___redundant_placeholder1D
@lstm_24_while_lstm_24_while_cond_418909___redundant_placeholder2D
@lstm_24_while_lstm_24_while_cond_418909___redundant_placeholder3
lstm_24_while_identity
�
lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2
lstm_24/while/Lessu
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_24/while/Identity"9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
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
while_body_420065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
�
�
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_416402

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
�
d
+__inference_dropout_12_layer_call_fn_420461

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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179832
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
�?
�
while_body_419719
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
.__inference_sequential_12_layer_call_fn_418419
lstm_24_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_4183792
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
_user_specified_namelstm_24_input
�[
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_418150

inputs>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_418066*
condR
while_cond_418065*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�^
�
'sequential_12_lstm_24_while_body_416089H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3G
Csequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1_0�
sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0:	�^
Ksequential_12_lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�Y
Jsequential_12_lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0:	�(
$sequential_12_lstm_24_while_identity*
&sequential_12_lstm_24_while_identity_1*
&sequential_12_lstm_24_while_identity_2*
&sequential_12_lstm_24_while_identity_3*
&sequential_12_lstm_24_while_identity_4*
&sequential_12_lstm_24_while_identity_5E
Asequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1�
}sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_24_while_lstm_cell_48_matmul_readvariableop_resource:	�\
Isequential_12_lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource:	@�W
Hsequential_12_lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource:	���?sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�>sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�@sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
Msequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2O
Msequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_24_while_placeholderVsequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02A
?sequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem�
>sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02@
>sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp�
/sequential_12/lstm_24/while/lstm_cell_48/MatMulMatMulFsequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_12/lstm_24/while/lstm_cell_48/MatMul�
@sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02B
@sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp�
1sequential_12/lstm_24/while/lstm_cell_48/MatMul_1MatMul)sequential_12_lstm_24_while_placeholder_2Hsequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_12/lstm_24/while/lstm_cell_48/MatMul_1�
,sequential_12/lstm_24/while/lstm_cell_48/addAddV29sequential_12/lstm_24/while/lstm_cell_48/MatMul:product:0;sequential_12/lstm_24/while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_12/lstm_24/while/lstm_cell_48/add�
?sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp�
0sequential_12/lstm_24/while/lstm_cell_48/BiasAddBiasAdd0sequential_12/lstm_24/while/lstm_cell_48/add:z:0Gsequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_12/lstm_24/while/lstm_cell_48/BiasAdd�
8sequential_12/lstm_24/while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_24/while/lstm_cell_48/split/split_dim�
.sequential_12/lstm_24/while/lstm_cell_48/splitSplitAsequential_12/lstm_24/while/lstm_cell_48/split/split_dim:output:09sequential_12/lstm_24/while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split20
.sequential_12/lstm_24/while/lstm_cell_48/split�
0sequential_12/lstm_24/while/lstm_cell_48/SigmoidSigmoid7sequential_12/lstm_24/while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@22
0sequential_12/lstm_24/while/lstm_cell_48/Sigmoid�
2sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_1Sigmoid7sequential_12/lstm_24/while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@24
2sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_1�
,sequential_12/lstm_24/while/lstm_cell_48/mulMul6sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_1:y:0)sequential_12_lstm_24_while_placeholder_3*
T0*'
_output_shapes
:���������@2.
,sequential_12/lstm_24/while/lstm_cell_48/mul�
-sequential_12/lstm_24/while/lstm_cell_48/ReluRelu7sequential_12/lstm_24/while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2/
-sequential_12/lstm_24/while/lstm_cell_48/Relu�
.sequential_12/lstm_24/while/lstm_cell_48/mul_1Mul4sequential_12/lstm_24/while/lstm_cell_48/Sigmoid:y:0;sequential_12/lstm_24/while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_12/lstm_24/while/lstm_cell_48/mul_1�
.sequential_12/lstm_24/while/lstm_cell_48/add_1AddV20sequential_12/lstm_24/while/lstm_cell_48/mul:z:02sequential_12/lstm_24/while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@20
.sequential_12/lstm_24/while/lstm_cell_48/add_1�
2sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_2Sigmoid7sequential_12/lstm_24/while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@24
2sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_2�
/sequential_12/lstm_24/while/lstm_cell_48/Relu_1Relu2sequential_12/lstm_24/while/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@21
/sequential_12/lstm_24/while/lstm_cell_48/Relu_1�
.sequential_12/lstm_24/while/lstm_cell_48/mul_2Mul6sequential_12/lstm_24/while/lstm_cell_48/Sigmoid_2:y:0=sequential_12/lstm_24/while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_12/lstm_24/while/lstm_cell_48/mul_2�
@sequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_24_while_placeholder_1'sequential_12_lstm_24_while_placeholder2sequential_12/lstm_24/while/lstm_cell_48/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItem�
!sequential_12/lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_24/while/add/y�
sequential_12/lstm_24/while/addAddV2'sequential_12_lstm_24_while_placeholder*sequential_12/lstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_24/while/add�
#sequential_12/lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_24/while/add_1/y�
!sequential_12/lstm_24/while/add_1AddV2Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counter,sequential_12/lstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_24/while/add_1�
$sequential_12/lstm_24/while/IdentityIdentity%sequential_12/lstm_24/while/add_1:z:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_24/while/Identity�
&sequential_12/lstm_24/while/Identity_1IdentityJsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_1�
&sequential_12/lstm_24/while/Identity_2Identity#sequential_12/lstm_24/while/add:z:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_2�
&sequential_12/lstm_24/while/Identity_3IdentityPsequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_3�
&sequential_12/lstm_24/while/Identity_4Identity2sequential_12/lstm_24/while/lstm_cell_48/mul_2:z:0!^sequential_12/lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_12/lstm_24/while/Identity_4�
&sequential_12/lstm_24/while/Identity_5Identity2sequential_12/lstm_24/while/lstm_cell_48/add_1:z:0!^sequential_12/lstm_24/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_12/lstm_24/while/Identity_5�
 sequential_12/lstm_24/while/NoOpNoOp@^sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp?^sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOpA^sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_24/while/NoOp"U
$sequential_12_lstm_24_while_identity-sequential_12/lstm_24/while/Identity:output:0"Y
&sequential_12_lstm_24_while_identity_1/sequential_12/lstm_24/while/Identity_1:output:0"Y
&sequential_12_lstm_24_while_identity_2/sequential_12/lstm_24/while/Identity_2:output:0"Y
&sequential_12_lstm_24_while_identity_3/sequential_12/lstm_24/while/Identity_3:output:0"Y
&sequential_12_lstm_24_while_identity_4/sequential_12/lstm_24/while/Identity_4:output:0"Y
&sequential_12_lstm_24_while_identity_5/sequential_12/lstm_24/while/Identity_5:output:0"�
Hsequential_12_lstm_24_while_lstm_cell_48_biasadd_readvariableop_resourceJsequential_12_lstm_24_while_lstm_cell_48_biasadd_readvariableop_resource_0"�
Isequential_12_lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resourceKsequential_12_lstm_24_while_lstm_cell_48_matmul_1_readvariableop_resource_0"�
Gsequential_12_lstm_24_while_lstm_cell_48_matmul_readvariableop_resourceIsequential_12_lstm_24_while_lstm_cell_48_matmul_readvariableop_resource_0"�
Asequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1Csequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1_0"�
}sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2�
?sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp?sequential_12/lstm_24/while/lstm_cell_48/BiasAdd/ReadVariableOp2�
>sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp>sequential_12/lstm_24/while/lstm_cell_48/MatMul/ReadVariableOp2�
@sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp@sequential_12/lstm_24/while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
+__inference_dropout_12_layer_call_fn_420456

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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179152
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
�

�
D__inference_dense_12_layer_call_and_return_conditional_losses_417927

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
while_body_417818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
while_cond_419567
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_419567___redundant_placeholder04
0while_while_cond_419567___redundant_placeholder14
0while_while_cond_419567___redundant_placeholder24
0while_while_cond_419567___redundant_placeholder3
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

�
lstm_25_while_cond_419056,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1D
@lstm_25_while_lstm_25_while_cond_419056___redundant_placeholder0D
@lstm_25_while_lstm_25_while_cond_419056___redundant_placeholder1D
@lstm_25_while_lstm_25_while_cond_419056___redundant_placeholder2D
@lstm_25_while_lstm_25_while_cond_419056___redundant_placeholder3
lstm_25_while_identity
�
lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2
lstm_25/while/Lessu
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_25/while/Identity"9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
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
.__inference_sequential_12_layer_call_fn_418538

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
I__inference_sequential_12_layer_call_and_return_conditional_losses_4183792
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
�	
�
$__inference_signature_wrapper_418496
lstm_24_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_4163272
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
_user_specified_namelstm_24_input
�
�
while_cond_420064
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_420064___redundant_placeholder04
0while_while_cond_420064___redundant_placeholder14
0while_while_cond_420064___redundant_placeholder24
0while_while_cond_420064___redundant_placeholder3
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
�I
�
__inference__traced_save_420809
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_24_lstm_cell_48_kernel_read_readvariableopD
@savev2_lstm_24_lstm_cell_48_recurrent_kernel_read_readvariableop8
4savev2_lstm_24_lstm_cell_48_bias_read_readvariableop:
6savev2_lstm_25_lstm_cell_49_kernel_read_readvariableopD
@savev2_lstm_25_lstm_cell_49_recurrent_kernel_read_readvariableop8
4savev2_lstm_25_lstm_cell_49_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_48_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_48_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_48_bias_m_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_49_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_49_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_49_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_48_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_48_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_48_bias_v_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_49_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_49_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_49_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_24_lstm_cell_48_kernel_read_readvariableop@savev2_lstm_24_lstm_cell_48_recurrent_kernel_read_readvariableop4savev2_lstm_24_lstm_cell_48_bias_read_readvariableop6savev2_lstm_25_lstm_cell_49_kernel_read_readvariableop@savev2_lstm_25_lstm_cell_49_recurrent_kernel_read_readvariableop4savev2_lstm_25_lstm_cell_49_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop=savev2_adam_lstm_24_lstm_cell_48_kernel_m_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_48_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_24_lstm_cell_48_bias_m_read_readvariableop=savev2_adam_lstm_25_lstm_cell_49_kernel_m_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_49_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_25_lstm_cell_49_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop=savev2_adam_lstm_24_lstm_cell_48_kernel_v_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_48_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_24_lstm_cell_48_bias_v_read_readvariableop=savev2_adam_lstm_25_lstm_cell_49_kernel_v_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_49_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_25_lstm_cell_49_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
(__inference_lstm_25_layer_call_fn_419836

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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4179022
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
�
�
while_cond_420366
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_420366___redundant_placeholder04
0while_while_cond_420366___redundant_placeholder14
0while_while_cond_420366___redundant_placeholder24
0while_while_cond_420366___redundant_placeholder3
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_418323

inputs>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_418239*
condR
while_cond_418238*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
'sequential_12_lstm_24_while_cond_416088H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3J
Fsequential_12_lstm_24_while_less_sequential_12_lstm_24_strided_slice_1`
\sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_416088___redundant_placeholder0`
\sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_416088___redundant_placeholder1`
\sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_416088___redundant_placeholder2`
\sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_416088___redundant_placeholder3(
$sequential_12_lstm_24_while_identity
�
 sequential_12/lstm_24/while/LessLess'sequential_12_lstm_24_while_placeholderFsequential_12_lstm_24_while_less_sequential_12_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_24/while/Less�
$sequential_12/lstm_24/while/IdentityIdentity$sequential_12/lstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_24/while/Identity"U
$sequential_12_lstm_24_while_identity-sequential_12/lstm_24/while/Identity:output:0*(
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_420466

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
�
�
while_cond_419913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_419913___redundant_placeholder04
0while_while_cond_419913___redundant_placeholder14
0while_while_cond_419913___redundant_placeholder24
0while_while_cond_419913___redundant_placeholder3
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
while_cond_419416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_419416___redundant_placeholder04
0while_while_cond_419416___redundant_placeholder14
0while_while_cond_419416___redundant_placeholder24
0while_while_cond_419416___redundant_placeholder3
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
while_cond_417659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_417659___redundant_placeholder04
0while_while_cond_417659___redundant_placeholder14
0while_while_cond_417659___redundant_placeholder24
0while_while_cond_417659___redundant_placeholder3
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
�J
�

lstm_25_while_body_418752,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0:	@�P
=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �K
<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_49_matmul_readvariableop_resource:	@�N
;lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource:	 �I
:lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource:	���1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItem�
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�
!lstm_25/while/lstm_cell_49/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_25/while/lstm_cell_49/MatMul�
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
#lstm_25/while/lstm_cell_49/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_25/while/lstm_cell_49/MatMul_1�
lstm_25/while/lstm_cell_49/addAddV2+lstm_25/while/lstm_cell_49/MatMul:product:0-lstm_25/while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_25/while/lstm_cell_49/add�
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�
"lstm_25/while/lstm_cell_49/BiasAddBiasAdd"lstm_25/while/lstm_cell_49/add:z:09lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_25/while/lstm_cell_49/BiasAdd�
*lstm_25/while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_49/split/split_dim�
 lstm_25/while/lstm_cell_49/splitSplit3lstm_25/while/lstm_cell_49/split/split_dim:output:0+lstm_25/while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_25/while/lstm_cell_49/split�
"lstm_25/while/lstm_cell_49/SigmoidSigmoid)lstm_25/while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_25/while/lstm_cell_49/Sigmoid�
$lstm_25/while/lstm_cell_49/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_25/while/lstm_cell_49/Sigmoid_1�
lstm_25/while/lstm_cell_49/mulMul(lstm_25/while/lstm_cell_49/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_25/while/lstm_cell_49/mul�
lstm_25/while/lstm_cell_49/ReluRelu)lstm_25/while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_25/while/lstm_cell_49/Relu�
 lstm_25/while/lstm_cell_49/mul_1Mul&lstm_25/while/lstm_cell_49/Sigmoid:y:0-lstm_25/while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/mul_1�
 lstm_25/while/lstm_cell_49/add_1AddV2"lstm_25/while/lstm_cell_49/mul:z:0$lstm_25/while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/add_1�
$lstm_25/while/lstm_cell_49/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_25/while/lstm_cell_49/Sigmoid_2�
!lstm_25/while/lstm_cell_49/Relu_1Relu$lstm_25/while/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_25/while/lstm_cell_49/Relu_1�
 lstm_25/while/lstm_cell_49/mul_2Mul(lstm_25/while/lstm_cell_49/Sigmoid_2:y:0/lstm_25/while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/mul_2�
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_49/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_25/while/TensorArrayV2Write/TensorListSetIteml
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add/y�
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/addp
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add_1/y�
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/add_1�
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity�
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_1�
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_2�
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_3�
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_49/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_25/while/Identity_4�
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_49/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_25/while/Identity_5�
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_25/while/NoOp"9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_49_matmul_readvariableop_resource;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0"�
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
�F
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_417115

inputs&
lstm_cell_49_417033:	@�&
lstm_cell_49_417035:	 �"
lstm_cell_49_417037:	�
identity��$lstm_cell_49/StatefulPartitionedCall�whileD
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
$lstm_cell_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_49_417033lstm_cell_49_417035lstm_cell_49_417037*
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4170322&
$lstm_cell_49/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_49_417033lstm_cell_49_417035lstm_cell_49_417037*
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
while_body_417046*
condR
while_cond_417045*K
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
NoOpNoOp%^lstm_cell_49/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_49/StatefulPartitionedCall$lstm_cell_49/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
while_cond_417255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_417255___redundant_placeholder04
0while_while_cond_417255___redundant_placeholder14
0while_while_cond_417255___redundant_placeholder24
0while_while_cond_417255___redundant_placeholder3
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
�\
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_420149
inputs_0>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_420065*
condR
while_cond_420064*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�%
�
while_body_417046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_49_417070_0:	@�.
while_lstm_cell_49_417072_0:	 �*
while_lstm_cell_49_417074_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_49_417070:	@�,
while_lstm_cell_49_417072:	 �(
while_lstm_cell_49_417074:	���*while/lstm_cell_49/StatefulPartitionedCall�
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
*while/lstm_cell_49/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_49_417070_0while_lstm_cell_49_417072_0while_lstm_cell_49_417074_0*
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4170322,
*while/lstm_cell_49/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_49/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_49/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_49/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_49/StatefulPartitionedCall*"
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
while_lstm_cell_49_417070while_lstm_cell_49_417070_0"8
while_lstm_cell_49_417072while_lstm_cell_49_417072_0"8
while_lstm_cell_49_417074while_lstm_cell_49_417074_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_49/StatefulPartitionedCall*while/lstm_cell_49/StatefulPartitionedCall: 
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
while_cond_418065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_418065___redundant_placeholder04
0while_while_cond_418065___redundant_placeholder14
0while_while_cond_418065___redundant_placeholder24
0while_while_cond_418065___redundant_placeholder3
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
ԓ
�	
!__inference__wrapped_model_416327
lstm_24_inputT
Asequential_12_lstm_24_lstm_cell_48_matmul_readvariableop_resource:	�V
Csequential_12_lstm_24_lstm_cell_48_matmul_1_readvariableop_resource:	@�Q
Bsequential_12_lstm_24_lstm_cell_48_biasadd_readvariableop_resource:	�T
Asequential_12_lstm_25_lstm_cell_49_matmul_readvariableop_resource:	@�V
Csequential_12_lstm_25_lstm_cell_49_matmul_1_readvariableop_resource:	 �Q
Bsequential_12_lstm_25_lstm_cell_49_biasadd_readvariableop_resource:	�G
5sequential_12_dense_12_matmul_readvariableop_resource: D
6sequential_12_dense_12_biasadd_readvariableop_resource:
identity��-sequential_12/dense_12/BiasAdd/ReadVariableOp�,sequential_12/dense_12/MatMul/ReadVariableOp�9sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�8sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp�:sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�sequential_12/lstm_24/while�9sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�8sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp�:sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�sequential_12/lstm_25/whilew
sequential_12/lstm_24/ShapeShapelstm_24_input*
T0*
_output_shapes
:2
sequential_12/lstm_24/Shape�
)sequential_12/lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_24/strided_slice/stack�
+sequential_12/lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_24/strided_slice/stack_1�
+sequential_12/lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_24/strided_slice/stack_2�
#sequential_12/lstm_24/strided_sliceStridedSlice$sequential_12/lstm_24/Shape:output:02sequential_12/lstm_24/strided_slice/stack:output:04sequential_12/lstm_24/strided_slice/stack_1:output:04sequential_12/lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_24/strided_slice�
!sequential_12/lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_12/lstm_24/zeros/mul/y�
sequential_12/lstm_24/zeros/mulMul,sequential_12/lstm_24/strided_slice:output:0*sequential_12/lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_24/zeros/mul�
"sequential_12/lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_12/lstm_24/zeros/Less/y�
 sequential_12/lstm_24/zeros/LessLess#sequential_12/lstm_24/zeros/mul:z:0+sequential_12/lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_24/zeros/Less�
$sequential_12/lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_12/lstm_24/zeros/packed/1�
"sequential_12/lstm_24/zeros/packedPack,sequential_12/lstm_24/strided_slice:output:0-sequential_12/lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_24/zeros/packed�
!sequential_12/lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_24/zeros/Const�
sequential_12/lstm_24/zerosFill+sequential_12/lstm_24/zeros/packed:output:0*sequential_12/lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_12/lstm_24/zeros�
#sequential_12/lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_12/lstm_24/zeros_1/mul/y�
!sequential_12/lstm_24/zeros_1/mulMul,sequential_12/lstm_24/strided_slice:output:0,sequential_12/lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_24/zeros_1/mul�
$sequential_12/lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_12/lstm_24/zeros_1/Less/y�
"sequential_12/lstm_24/zeros_1/LessLess%sequential_12/lstm_24/zeros_1/mul:z:0-sequential_12/lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_24/zeros_1/Less�
&sequential_12/lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2(
&sequential_12/lstm_24/zeros_1/packed/1�
$sequential_12/lstm_24/zeros_1/packedPack,sequential_12/lstm_24/strided_slice:output:0/sequential_12/lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_24/zeros_1/packed�
#sequential_12/lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_24/zeros_1/Const�
sequential_12/lstm_24/zeros_1Fill-sequential_12/lstm_24/zeros_1/packed:output:0,sequential_12/lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_12/lstm_24/zeros_1�
$sequential_12/lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_24/transpose/perm�
sequential_12/lstm_24/transpose	Transposelstm_24_input-sequential_12/lstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:���������2!
sequential_12/lstm_24/transpose�
sequential_12/lstm_24/Shape_1Shape#sequential_12/lstm_24/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_24/Shape_1�
+sequential_12/lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_24/strided_slice_1/stack�
-sequential_12/lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_1/stack_1�
-sequential_12/lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_1/stack_2�
%sequential_12/lstm_24/strided_slice_1StridedSlice&sequential_12/lstm_24/Shape_1:output:04sequential_12/lstm_24/strided_slice_1/stack:output:06sequential_12/lstm_24/strided_slice_1/stack_1:output:06sequential_12/lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_1�
1sequential_12/lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_12/lstm_24/TensorArrayV2/element_shape�
#sequential_12/lstm_24/TensorArrayV2TensorListReserve:sequential_12/lstm_24/TensorArrayV2/element_shape:output:0.sequential_12/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_24/TensorArrayV2�
Ksequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2M
Ksequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_24/transpose:y:0Tsequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor�
+sequential_12/lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_24/strided_slice_2/stack�
-sequential_12/lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_2/stack_1�
-sequential_12/lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_2/stack_2�
%sequential_12/lstm_24/strided_slice_2StridedSlice#sequential_12/lstm_24/transpose:y:04sequential_12/lstm_24/strided_slice_2/stack:output:06sequential_12/lstm_24/strided_slice_2/stack_1:output:06sequential_12/lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_2�
8sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_24_lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02:
8sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp�
)sequential_12/lstm_24/lstm_cell_48/MatMulMatMul.sequential_12/lstm_24/strided_slice_2:output:0@sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_12/lstm_24/lstm_cell_48/MatMul�
:sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_24_lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02<
:sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�
+sequential_12/lstm_24/lstm_cell_48/MatMul_1MatMul$sequential_12/lstm_24/zeros:output:0Bsequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_12/lstm_24/lstm_cell_48/MatMul_1�
&sequential_12/lstm_24/lstm_cell_48/addAddV23sequential_12/lstm_24/lstm_cell_48/MatMul:product:05sequential_12/lstm_24/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_12/lstm_24/lstm_cell_48/add�
9sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�
*sequential_12/lstm_24/lstm_cell_48/BiasAddBiasAdd*sequential_12/lstm_24/lstm_cell_48/add:z:0Asequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_12/lstm_24/lstm_cell_48/BiasAdd�
2sequential_12/lstm_24/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_24/lstm_cell_48/split/split_dim�
(sequential_12/lstm_24/lstm_cell_48/splitSplit;sequential_12/lstm_24/lstm_cell_48/split/split_dim:output:03sequential_12/lstm_24/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2*
(sequential_12/lstm_24/lstm_cell_48/split�
*sequential_12/lstm_24/lstm_cell_48/SigmoidSigmoid1sequential_12/lstm_24/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2,
*sequential_12/lstm_24/lstm_cell_48/Sigmoid�
,sequential_12/lstm_24/lstm_cell_48/Sigmoid_1Sigmoid1sequential_12/lstm_24/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2.
,sequential_12/lstm_24/lstm_cell_48/Sigmoid_1�
&sequential_12/lstm_24/lstm_cell_48/mulMul0sequential_12/lstm_24/lstm_cell_48/Sigmoid_1:y:0&sequential_12/lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:���������@2(
&sequential_12/lstm_24/lstm_cell_48/mul�
'sequential_12/lstm_24/lstm_cell_48/ReluRelu1sequential_12/lstm_24/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2)
'sequential_12/lstm_24/lstm_cell_48/Relu�
(sequential_12/lstm_24/lstm_cell_48/mul_1Mul.sequential_12/lstm_24/lstm_cell_48/Sigmoid:y:05sequential_12/lstm_24/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_12/lstm_24/lstm_cell_48/mul_1�
(sequential_12/lstm_24/lstm_cell_48/add_1AddV2*sequential_12/lstm_24/lstm_cell_48/mul:z:0,sequential_12/lstm_24/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2*
(sequential_12/lstm_24/lstm_cell_48/add_1�
,sequential_12/lstm_24/lstm_cell_48/Sigmoid_2Sigmoid1sequential_12/lstm_24/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2.
,sequential_12/lstm_24/lstm_cell_48/Sigmoid_2�
)sequential_12/lstm_24/lstm_cell_48/Relu_1Relu,sequential_12/lstm_24/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2+
)sequential_12/lstm_24/lstm_cell_48/Relu_1�
(sequential_12/lstm_24/lstm_cell_48/mul_2Mul0sequential_12/lstm_24/lstm_cell_48/Sigmoid_2:y:07sequential_12/lstm_24/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_12/lstm_24/lstm_cell_48/mul_2�
3sequential_12/lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   25
3sequential_12/lstm_24/TensorArrayV2_1/element_shape�
%sequential_12/lstm_24/TensorArrayV2_1TensorListReserve<sequential_12/lstm_24/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_24/TensorArrayV2_1z
sequential_12/lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_24/time�
.sequential_12/lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_12/lstm_24/while/maximum_iterations�
(sequential_12/lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_24/while/loop_counter�
sequential_12/lstm_24/whileWhile1sequential_12/lstm_24/while/loop_counter:output:07sequential_12/lstm_24/while/maximum_iterations:output:0#sequential_12/lstm_24/time:output:0.sequential_12/lstm_24/TensorArrayV2_1:handle:0$sequential_12/lstm_24/zeros:output:0&sequential_12/lstm_24/zeros_1:output:0.sequential_12/lstm_24/strided_slice_1:output:0Msequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_24_lstm_cell_48_matmul_readvariableop_resourceCsequential_12_lstm_24_lstm_cell_48_matmul_1_readvariableop_resourceBsequential_12_lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
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
'sequential_12_lstm_24_while_body_416089*3
cond+R)
'sequential_12_lstm_24_while_cond_416088*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
sequential_12/lstm_24/while�
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2H
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_12/lstm_24/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_24/while:output:3Osequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02:
8sequential_12/lstm_24/TensorArrayV2Stack/TensorListStack�
+sequential_12/lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_12/lstm_24/strided_slice_3/stack�
-sequential_12/lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_24/strided_slice_3/stack_1�
-sequential_12/lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_3/stack_2�
%sequential_12/lstm_24/strided_slice_3StridedSliceAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_24/strided_slice_3/stack:output:06sequential_12/lstm_24/strided_slice_3/stack_1:output:06sequential_12/lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_3�
&sequential_12/lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_24/transpose_1/perm�
!sequential_12/lstm_24/transpose_1	TransposeAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2#
!sequential_12/lstm_24/transpose_1�
sequential_12/lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_24/runtime�
sequential_12/lstm_25/ShapeShape%sequential_12/lstm_24/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_25/Shape�
)sequential_12/lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_25/strided_slice/stack�
+sequential_12/lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_25/strided_slice/stack_1�
+sequential_12/lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_25/strided_slice/stack_2�
#sequential_12/lstm_25/strided_sliceStridedSlice$sequential_12/lstm_25/Shape:output:02sequential_12/lstm_25/strided_slice/stack:output:04sequential_12/lstm_25/strided_slice/stack_1:output:04sequential_12/lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_25/strided_slice�
!sequential_12/lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_12/lstm_25/zeros/mul/y�
sequential_12/lstm_25/zeros/mulMul,sequential_12/lstm_25/strided_slice:output:0*sequential_12/lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_25/zeros/mul�
"sequential_12/lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_12/lstm_25/zeros/Less/y�
 sequential_12/lstm_25/zeros/LessLess#sequential_12/lstm_25/zeros/mul:z:0+sequential_12/lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_25/zeros/Less�
$sequential_12/lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_12/lstm_25/zeros/packed/1�
"sequential_12/lstm_25/zeros/packedPack,sequential_12/lstm_25/strided_slice:output:0-sequential_12/lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_25/zeros/packed�
!sequential_12/lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_25/zeros/Const�
sequential_12/lstm_25/zerosFill+sequential_12/lstm_25/zeros/packed:output:0*sequential_12/lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_12/lstm_25/zeros�
#sequential_12/lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_12/lstm_25/zeros_1/mul/y�
!sequential_12/lstm_25/zeros_1/mulMul,sequential_12/lstm_25/strided_slice:output:0,sequential_12/lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_25/zeros_1/mul�
$sequential_12/lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_12/lstm_25/zeros_1/Less/y�
"sequential_12/lstm_25/zeros_1/LessLess%sequential_12/lstm_25/zeros_1/mul:z:0-sequential_12/lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_25/zeros_1/Less�
&sequential_12/lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_12/lstm_25/zeros_1/packed/1�
$sequential_12/lstm_25/zeros_1/packedPack,sequential_12/lstm_25/strided_slice:output:0/sequential_12/lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_25/zeros_1/packed�
#sequential_12/lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_25/zeros_1/Const�
sequential_12/lstm_25/zeros_1Fill-sequential_12/lstm_25/zeros_1/packed:output:0,sequential_12/lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_12/lstm_25/zeros_1�
$sequential_12/lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_25/transpose/perm�
sequential_12/lstm_25/transpose	Transpose%sequential_12/lstm_24/transpose_1:y:0-sequential_12/lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2!
sequential_12/lstm_25/transpose�
sequential_12/lstm_25/Shape_1Shape#sequential_12/lstm_25/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_25/Shape_1�
+sequential_12/lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_25/strided_slice_1/stack�
-sequential_12/lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_1/stack_1�
-sequential_12/lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_1/stack_2�
%sequential_12/lstm_25/strided_slice_1StridedSlice&sequential_12/lstm_25/Shape_1:output:04sequential_12/lstm_25/strided_slice_1/stack:output:06sequential_12/lstm_25/strided_slice_1/stack_1:output:06sequential_12/lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_1�
1sequential_12/lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_12/lstm_25/TensorArrayV2/element_shape�
#sequential_12/lstm_25/TensorArrayV2TensorListReserve:sequential_12/lstm_25/TensorArrayV2/element_shape:output:0.sequential_12/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_25/TensorArrayV2�
Ksequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2M
Ksequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_25/transpose:y:0Tsequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor�
+sequential_12/lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_25/strided_slice_2/stack�
-sequential_12/lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_2/stack_1�
-sequential_12/lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_2/stack_2�
%sequential_12/lstm_25/strided_slice_2StridedSlice#sequential_12/lstm_25/transpose:y:04sequential_12/lstm_25/strided_slice_2/stack:output:06sequential_12/lstm_25/strided_slice_2/stack_1:output:06sequential_12/lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_2�
8sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_25_lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02:
8sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp�
)sequential_12/lstm_25/lstm_cell_49/MatMulMatMul.sequential_12/lstm_25/strided_slice_2:output:0@sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_12/lstm_25/lstm_cell_49/MatMul�
:sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_25_lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02<
:sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�
+sequential_12/lstm_25/lstm_cell_49/MatMul_1MatMul$sequential_12/lstm_25/zeros:output:0Bsequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_12/lstm_25/lstm_cell_49/MatMul_1�
&sequential_12/lstm_25/lstm_cell_49/addAddV23sequential_12/lstm_25/lstm_cell_49/MatMul:product:05sequential_12/lstm_25/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_12/lstm_25/lstm_cell_49/add�
9sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�
*sequential_12/lstm_25/lstm_cell_49/BiasAddBiasAdd*sequential_12/lstm_25/lstm_cell_49/add:z:0Asequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_12/lstm_25/lstm_cell_49/BiasAdd�
2sequential_12/lstm_25/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_25/lstm_cell_49/split/split_dim�
(sequential_12/lstm_25/lstm_cell_49/splitSplit;sequential_12/lstm_25/lstm_cell_49/split/split_dim:output:03sequential_12/lstm_25/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2*
(sequential_12/lstm_25/lstm_cell_49/split�
*sequential_12/lstm_25/lstm_cell_49/SigmoidSigmoid1sequential_12/lstm_25/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2,
*sequential_12/lstm_25/lstm_cell_49/Sigmoid�
,sequential_12/lstm_25/lstm_cell_49/Sigmoid_1Sigmoid1sequential_12/lstm_25/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2.
,sequential_12/lstm_25/lstm_cell_49/Sigmoid_1�
&sequential_12/lstm_25/lstm_cell_49/mulMul0sequential_12/lstm_25/lstm_cell_49/Sigmoid_1:y:0&sequential_12/lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2(
&sequential_12/lstm_25/lstm_cell_49/mul�
'sequential_12/lstm_25/lstm_cell_49/ReluRelu1sequential_12/lstm_25/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2)
'sequential_12/lstm_25/lstm_cell_49/Relu�
(sequential_12/lstm_25/lstm_cell_49/mul_1Mul.sequential_12/lstm_25/lstm_cell_49/Sigmoid:y:05sequential_12/lstm_25/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_12/lstm_25/lstm_cell_49/mul_1�
(sequential_12/lstm_25/lstm_cell_49/add_1AddV2*sequential_12/lstm_25/lstm_cell_49/mul:z:0,sequential_12/lstm_25/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2*
(sequential_12/lstm_25/lstm_cell_49/add_1�
,sequential_12/lstm_25/lstm_cell_49/Sigmoid_2Sigmoid1sequential_12/lstm_25/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2.
,sequential_12/lstm_25/lstm_cell_49/Sigmoid_2�
)sequential_12/lstm_25/lstm_cell_49/Relu_1Relu,sequential_12/lstm_25/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2+
)sequential_12/lstm_25/lstm_cell_49/Relu_1�
(sequential_12/lstm_25/lstm_cell_49/mul_2Mul0sequential_12/lstm_25/lstm_cell_49/Sigmoid_2:y:07sequential_12/lstm_25/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_12/lstm_25/lstm_cell_49/mul_2�
3sequential_12/lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    25
3sequential_12/lstm_25/TensorArrayV2_1/element_shape�
%sequential_12/lstm_25/TensorArrayV2_1TensorListReserve<sequential_12/lstm_25/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_25/TensorArrayV2_1z
sequential_12/lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_25/time�
.sequential_12/lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_12/lstm_25/while/maximum_iterations�
(sequential_12/lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_25/while/loop_counter�
sequential_12/lstm_25/whileWhile1sequential_12/lstm_25/while/loop_counter:output:07sequential_12/lstm_25/while/maximum_iterations:output:0#sequential_12/lstm_25/time:output:0.sequential_12/lstm_25/TensorArrayV2_1:handle:0$sequential_12/lstm_25/zeros:output:0&sequential_12/lstm_25/zeros_1:output:0.sequential_12/lstm_25/strided_slice_1:output:0Msequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_25_lstm_cell_49_matmul_readvariableop_resourceCsequential_12_lstm_25_lstm_cell_49_matmul_1_readvariableop_resourceBsequential_12_lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
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
'sequential_12_lstm_25_while_body_416236*3
cond+R)
'sequential_12_lstm_25_while_cond_416235*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
sequential_12/lstm_25/while�
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2H
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_12/lstm_25/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_25/while:output:3Osequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02:
8sequential_12/lstm_25/TensorArrayV2Stack/TensorListStack�
+sequential_12/lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_12/lstm_25/strided_slice_3/stack�
-sequential_12/lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_25/strided_slice_3/stack_1�
-sequential_12/lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_3/stack_2�
%sequential_12/lstm_25/strided_slice_3StridedSliceAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_25/strided_slice_3/stack:output:06sequential_12/lstm_25/strided_slice_3/stack_1:output:06sequential_12/lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_3�
&sequential_12/lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_25/transpose_1/perm�
!sequential_12/lstm_25/transpose_1	TransposeAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2#
!sequential_12/lstm_25/transpose_1�
sequential_12/lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_25/runtime�
!sequential_12/dropout_12/IdentityIdentity.sequential_12/lstm_25/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2#
!sequential_12/dropout_12/Identity�
,sequential_12/dense_12/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_12/dense_12/MatMul/ReadVariableOp�
sequential_12/dense_12/MatMulMatMul*sequential_12/dropout_12/Identity:output:04sequential_12/dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_12/dense_12/MatMul�
-sequential_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_12/BiasAdd/ReadVariableOp�
sequential_12/dense_12/BiasAddBiasAdd'sequential_12/dense_12/MatMul:product:05sequential_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_12/dense_12/BiasAdd�
IdentityIdentity'sequential_12/dense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^sequential_12/dense_12/BiasAdd/ReadVariableOp-^sequential_12/dense_12/MatMul/ReadVariableOp:^sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp9^sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp;^sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp^sequential_12/lstm_24/while:^sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp9^sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp;^sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp^sequential_12/lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2^
-sequential_12/dense_12/BiasAdd/ReadVariableOp-sequential_12/dense_12/BiasAdd/ReadVariableOp2\
,sequential_12/dense_12/MatMul/ReadVariableOp,sequential_12/dense_12/MatMul/ReadVariableOp2v
9sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp9sequential_12/lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp8sequential_12/lstm_24/lstm_cell_48/MatMul/ReadVariableOp2x
:sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp:sequential_12/lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp2:
sequential_12/lstm_24/whilesequential_12/lstm_24/while2v
9sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp9sequential_12/lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp8sequential_12/lstm_25/lstm_cell_49/MatMul/ReadVariableOp2x
:sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp:sequential_12/lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp2:
sequential_12/lstm_25/whilesequential_12/lstm_25/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_24_input
�%
�
while_body_416626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_48_416650_0:	�.
while_lstm_cell_48_416652_0:	@�*
while_lstm_cell_48_416654_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_48_416650:	�,
while_lstm_cell_48_416652:	@�(
while_lstm_cell_48_416654:	���*while/lstm_cell_48/StatefulPartitionedCall�
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
*while/lstm_cell_48/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_48_416650_0while_lstm_cell_48_416652_0while_lstm_cell_48_416654_0*
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4165482,
*while/lstm_cell_48/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_48/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_48/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_48/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_48/StatefulPartitionedCall*"
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
while_lstm_cell_48_416650while_lstm_cell_48_416650_0"8
while_lstm_cell_48_416652while_lstm_cell_48_416652_0"8
while_lstm_cell_48_416654while_lstm_cell_48_416654_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_48/StatefulPartitionedCall*while/lstm_cell_48/StatefulPartitionedCall: 
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_417325

inputs&
lstm_cell_49_417243:	@�&
lstm_cell_49_417245:	 �"
lstm_cell_49_417247:	�
identity��$lstm_cell_49/StatefulPartitionedCall�whileD
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
$lstm_cell_49/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_49_417243lstm_cell_49_417245lstm_cell_49_417247*
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4171782&
$lstm_cell_49/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_49_417243lstm_cell_49_417245lstm_cell_49_417247*
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
while_body_417256*
condR
while_cond_417255*K
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
NoOpNoOp%^lstm_cell_49/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_49/StatefulPartitionedCall$lstm_cell_49/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�

�
lstm_24_while_cond_418604,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1D
@lstm_24_while_lstm_24_while_cond_418604___redundant_placeholder0D
@lstm_24_while_lstm_24_while_cond_418604___redundant_placeholder1D
@lstm_24_while_lstm_24_while_cond_418604___redundant_placeholder2D
@lstm_24_while_lstm_24_while_cond_418604___redundant_placeholder3
lstm_24_while_identity
�
lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2
lstm_24/while/Lessu
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_24/while/Identity"9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
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
�\
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_419350
inputs_0>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_419266*
condR
while_cond_419265*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�%
�
while_body_416416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_48_416440_0:	�.
while_lstm_cell_48_416442_0:	@�*
while_lstm_cell_48_416444_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_48_416440:	�,
while_lstm_cell_48_416442:	@�(
while_lstm_cell_48_416444:	���*while/lstm_cell_48/StatefulPartitionedCall�
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
*while/lstm_cell_48/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_48_416440_0while_lstm_cell_48_416442_0while_lstm_cell_48_416444_0*
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4164022,
*while/lstm_cell_48/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_48/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_48/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_48/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_48/StatefulPartitionedCall*"
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
while_lstm_cell_48_416440while_lstm_cell_48_416440_0"8
while_lstm_cell_48_416442while_lstm_cell_48_416442_0"8
while_lstm_cell_48_416444while_lstm_cell_48_416444_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_48/StatefulPartitionedCall*while/lstm_cell_48/StatefulPartitionedCall: 
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
while_body_417256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_49_417280_0:	@�.
while_lstm_cell_49_417282_0:	 �*
while_lstm_cell_49_417284_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_49_417280:	@�,
while_lstm_cell_49_417282:	 �(
while_lstm_cell_49_417284:	���*while/lstm_cell_49/StatefulPartitionedCall�
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
*while/lstm_cell_49/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_49_417280_0while_lstm_cell_49_417282_0while_lstm_cell_49_417284_0*
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4171782,
*while/lstm_cell_49/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_49/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_49/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_49/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_49/StatefulPartitionedCall*"
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
while_lstm_cell_49_417280while_lstm_cell_49_417280_0"8
while_lstm_cell_49_417282while_lstm_cell_49_417282_0"8
while_lstm_cell_49_417284while_lstm_cell_49_417284_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_49/StatefulPartitionedCall*while/lstm_cell_49/StatefulPartitionedCall: 
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
(__inference_lstm_25_layer_call_fn_419814
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4171152
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
�
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_417934

inputs!
lstm_24_417745:	�!
lstm_24_417747:	@�
lstm_24_417749:	�!
lstm_25_417903:	@�!
lstm_25_417905:	 �
lstm_25_417907:	�!
dense_12_417928: 
dense_12_417930:
identity�� dense_12/StatefulPartitionedCall�lstm_24/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_417745lstm_24_417747lstm_24_417749*
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4177442!
lstm_24/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0lstm_25_417903lstm_25_417905lstm_25_417907*
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4179022!
lstm_25/StatefulPartitionedCall�
dropout_12/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179152
dropout_12/PartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_12_417928dense_12_417930*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_4179272"
 dense_12/StatefulPartitionedCall�
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_417032

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
�
�
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_417178

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
while_body_419266
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
�\
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_419501
inputs_0>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_419417*
condR
while_cond_419416*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
��
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_419155

inputsF
3lstm_24_lstm_cell_48_matmul_readvariableop_resource:	�H
5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource:	@�C
4lstm_24_lstm_cell_48_biasadd_readvariableop_resource:	�F
3lstm_25_lstm_cell_49_matmul_readvariableop_resource:	@�H
5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource:	 �C
4lstm_25_lstm_cell_49_biasadd_readvariableop_resource:	�9
'dense_12_matmul_readvariableop_resource: 6
(dense_12_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�*lstm_24/lstm_cell_48/MatMul/ReadVariableOp�,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�lstm_24/while�+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�*lstm_25/lstm_cell_49/MatMul/ReadVariableOp�,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�lstm_25/whileT
lstm_24/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_24/Shape�
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice/stack�
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_1�
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_2�
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slicel
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros/mul/y�
lstm_24/zeros/mulMullstm_24/strided_slice:output:0lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/mulo
lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_24/zeros/Less/y�
lstm_24/zeros/LessLesslstm_24/zeros/mul:z:0lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/Lessr
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros/packed/1�
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros/packedo
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros/Const�
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/zerosp
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros_1/mul/y�
lstm_24/zeros_1/mulMullstm_24/strided_slice:output:0lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/muls
lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_24/zeros_1/Less/y�
lstm_24/zeros_1/LessLesslstm_24/zeros_1/mul:z:0lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/Lessv
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros_1/packed/1�
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros_1/packeds
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros_1/Const�
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/zeros_1�
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose/perm�
lstm_24/transpose	Transposeinputslstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_24/transposeg
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:2
lstm_24/Shape_1�
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_1/stack�
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_1�
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_2�
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slice_1�
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_24/TensorArrayV2/element_shape�
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2�
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_24/TensorArrayUnstack/TensorListFromTensor�
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_2/stack�
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_1�
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_2�
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_24/strided_slice_2�
*lstm_24/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3lstm_24_lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_24/lstm_cell_48/MatMul/ReadVariableOp�
lstm_24/lstm_cell_48/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/MatMul�
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_24/lstm_cell_48/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/MatMul_1�
lstm_24/lstm_cell_48/addAddV2%lstm_24/lstm_cell_48/MatMul:product:0'lstm_24/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/add�
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_24/lstm_cell_48/BiasAddBiasAddlstm_24/lstm_cell_48/add:z:03lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/BiasAdd�
$lstm_24/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_48/split/split_dim�
lstm_24/lstm_cell_48/splitSplit-lstm_24/lstm_cell_48/split/split_dim:output:0%lstm_24/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_24/lstm_cell_48/split�
lstm_24/lstm_cell_48/SigmoidSigmoid#lstm_24/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Sigmoid�
lstm_24/lstm_cell_48/Sigmoid_1Sigmoid#lstm_24/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_24/lstm_cell_48/Sigmoid_1�
lstm_24/lstm_cell_48/mulMul"lstm_24/lstm_cell_48/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul�
lstm_24/lstm_cell_48/ReluRelu#lstm_24/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Relu�
lstm_24/lstm_cell_48/mul_1Mul lstm_24/lstm_cell_48/Sigmoid:y:0'lstm_24/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul_1�
lstm_24/lstm_cell_48/add_1AddV2lstm_24/lstm_cell_48/mul:z:0lstm_24/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/add_1�
lstm_24/lstm_cell_48/Sigmoid_2Sigmoid#lstm_24/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_24/lstm_cell_48/Sigmoid_2�
lstm_24/lstm_cell_48/Relu_1Relulstm_24/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Relu_1�
lstm_24/lstm_cell_48/mul_2Mul"lstm_24/lstm_cell_48/Sigmoid_2:y:0)lstm_24/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul_2�
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_24/TensorArrayV2_1/element_shape�
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2_1^
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/time�
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_24/while/maximum_iterationsz
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/while/loop_counter�
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_48_matmul_readvariableop_resource5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource4lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
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
lstm_24_while_body_418910*%
condR
lstm_24_while_cond_418909*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_24/while�
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_24/TensorArrayV2Stack/TensorListStack�
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_24/strided_slice_3/stack�
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_24/strided_slice_3/stack_1�
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_3/stack_2�
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_24/strided_slice_3�
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/perm�
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_24/transpose_1v
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/runtimee
lstm_25/ShapeShapelstm_24/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_25/Shape�
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice/stack�
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_1�
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_2�
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slicel
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros/mul/y�
lstm_25/zeros/mulMullstm_25/strided_slice:output:0lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/mulo
lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_25/zeros/Less/y�
lstm_25/zeros/LessLesslstm_25/zeros/mul:z:0lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/Lessr
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros/packed/1�
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros/packedo
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros/Const�
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/zerosp
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros_1/mul/y�
lstm_25/zeros_1/mulMullstm_25/strided_slice:output:0lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/muls
lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_25/zeros_1/Less/y�
lstm_25/zeros_1/LessLesslstm_25/zeros_1/mul:z:0lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/Lessv
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros_1/packed/1�
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros_1/packeds
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros_1/Const�
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/zeros_1�
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm�
lstm_25/transpose	Transposelstm_24/transpose_1:y:0lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_25/transposeg
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:2
lstm_25/Shape_1�
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_1/stack�
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_1�
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_2�
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slice_1�
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_25/TensorArrayV2/element_shape�
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2�
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_25/TensorArrayUnstack/TensorListFromTensor�
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_2/stack�
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_1�
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_2�
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_25/strided_slice_2�
*lstm_25/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_25/lstm_cell_49/MatMul/ReadVariableOp�
lstm_25/lstm_cell_49/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/MatMul�
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_25/lstm_cell_49/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/MatMul_1�
lstm_25/lstm_cell_49/addAddV2%lstm_25/lstm_cell_49/MatMul:product:0'lstm_25/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/add�
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_25/lstm_cell_49/BiasAddBiasAddlstm_25/lstm_cell_49/add:z:03lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/BiasAdd�
$lstm_25/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_49/split/split_dim�
lstm_25/lstm_cell_49/splitSplit-lstm_25/lstm_cell_49/split/split_dim:output:0%lstm_25/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_25/lstm_cell_49/split�
lstm_25/lstm_cell_49/SigmoidSigmoid#lstm_25/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Sigmoid�
lstm_25/lstm_cell_49/Sigmoid_1Sigmoid#lstm_25/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_25/lstm_cell_49/Sigmoid_1�
lstm_25/lstm_cell_49/mulMul"lstm_25/lstm_cell_49/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul�
lstm_25/lstm_cell_49/ReluRelu#lstm_25/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Relu�
lstm_25/lstm_cell_49/mul_1Mul lstm_25/lstm_cell_49/Sigmoid:y:0'lstm_25/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul_1�
lstm_25/lstm_cell_49/add_1AddV2lstm_25/lstm_cell_49/mul:z:0lstm_25/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/add_1�
lstm_25/lstm_cell_49/Sigmoid_2Sigmoid#lstm_25/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_25/lstm_cell_49/Sigmoid_2�
lstm_25/lstm_cell_49/Relu_1Relulstm_25/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Relu_1�
lstm_25/lstm_cell_49/mul_2Mul"lstm_25/lstm_cell_49/Sigmoid_2:y:0)lstm_25/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul_2�
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_25/TensorArrayV2_1/element_shape�
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2_1^
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/time�
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_25/while/maximum_iterationsz
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/while/loop_counter�
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_49_matmul_readvariableop_resource5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource4lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
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
lstm_25_while_body_419057*%
condR
lstm_25_while_cond_419056*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_25/while�
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_25/TensorArrayV2Stack/TensorListStack�
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_25/strided_slice_3/stack�
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_25/strided_slice_3/stack_1�
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_3/stack_2�
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_25/strided_slice_3�
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/perm�
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_25/transpose_1v
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/runtimey
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_12/dropout/Const�
dropout_12/dropout/MulMul lstm_25/strided_slice_3:output:0!dropout_12/dropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout_12/dropout/Mul�
dropout_12/dropout/ShapeShape lstm_25/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape�
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype021
/dropout_12/dropout/random_uniform/RandomUniform�
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2#
!dropout_12/dropout/GreaterEqual/y�
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2!
dropout_12/dropout/GreaterEqual�
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout_12/dropout/Cast�
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout_12/dropout/Mul_1�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMuldropout_12/dropout/Mul_1:z:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/BiasAddt
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp,^lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp+^lstm_24/lstm_cell_48/MatMul/ReadVariableOp-^lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp^lstm_24/while,^lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_49/MatMul/ReadVariableOp-^lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp^lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2Z
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp2X
*lstm_24/lstm_cell_48/MatMul/ReadVariableOp*lstm_24/lstm_cell_48/MatMul/ReadVariableOp2\
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp2
lstm_24/whilelstm_24/while2Z
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_49/MatMul/ReadVariableOp*lstm_25/lstm_cell_49/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_420367
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_25_layer_call_fn_419825
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4173252
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
�\
�
C__inference_lstm_25_layer_call_and_return_conditional_losses_419998
inputs_0>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_419914*
condR
while_cond_419913*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�?
�
while_body_419914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_24_layer_call_fn_419177
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4166952
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
while_cond_419265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_419265___redundant_placeholder04
0while_while_cond_419265___redundant_placeholder14
0while_while_cond_419265___redundant_placeholder24
0while_while_cond_419265___redundant_placeholder3
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
(__inference_lstm_24_layer_call_fn_419166
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4164852
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
�?
�
while_body_419417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_416485

inputs&
lstm_cell_48_416403:	�&
lstm_cell_48_416405:	@�"
lstm_cell_48_416407:	�
identity��$lstm_cell_48/StatefulPartitionedCall�whileD
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
$lstm_cell_48/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_48_416403lstm_cell_48_416405lstm_cell_48_416407*
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4164022&
$lstm_cell_48/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_48_416403lstm_cell_48_416405lstm_cell_48_416407*
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
while_body_416416*
condR
while_cond_416415*K
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
NoOpNoOp%^lstm_cell_48/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_48/StatefulPartitionedCall$lstm_cell_48/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420563

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
�[
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_419652

inputs>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_419568*
condR
while_cond_419567*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_420215
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_420215___redundant_placeholder04
0while_while_cond_420215___redundant_placeholder14
0while_while_cond_420215___redundant_placeholder24
0while_while_cond_420215___redundant_placeholder3
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_420300

inputs>
+lstm_cell_49_matmul_readvariableop_resource:	@�@
-lstm_cell_49_matmul_1_readvariableop_resource:	 �;
,lstm_cell_49_biasadd_readvariableop_resource:	�
identity��#lstm_cell_49/BiasAdd/ReadVariableOp�"lstm_cell_49/MatMul/ReadVariableOp�$lstm_cell_49/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_49/MatMul/ReadVariableOpReadVariableOp+lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_49/MatMul/ReadVariableOp�
lstm_cell_49/MatMulMatMulstrided_slice_2:output:0*lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul�
$lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_cell_49/MatMul_1MatMulzeros:output:0,lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/MatMul_1�
lstm_cell_49/addAddV2lstm_cell_49/MatMul:product:0lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/add�
#lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_cell_49/BiasAddBiasAddlstm_cell_49/add:z:0+lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_49/BiasAdd~
lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_49/split/split_dim�
lstm_cell_49/splitSplit%lstm_cell_49/split/split_dim:output:0lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_49/split�
lstm_cell_49/SigmoidSigmoidlstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid�
lstm_cell_49/Sigmoid_1Sigmoidlstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_1�
lstm_cell_49/mulMullstm_cell_49/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul}
lstm_cell_49/ReluRelulstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu�
lstm_cell_49/mul_1Mullstm_cell_49/Sigmoid:y:0lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_1�
lstm_cell_49/add_1AddV2lstm_cell_49/mul:z:0lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/add_1�
lstm_cell_49/Sigmoid_2Sigmoidlstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Sigmoid_2|
lstm_cell_49/Relu_1Relulstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/Relu_1�
lstm_cell_49/mul_2Mullstm_cell_49/Sigmoid_2:y:0!lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_49/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_49_matmul_readvariableop_resource-lstm_cell_49_matmul_1_readvariableop_resource,lstm_cell_49_biasadd_readvariableop_resource*
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
while_body_420216*
condR
while_cond_420215*K
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
NoOpNoOp$^lstm_cell_49/BiasAdd/ReadVariableOp#^lstm_cell_49/MatMul/ReadVariableOp%^lstm_cell_49/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_49/BiasAdd/ReadVariableOp#lstm_cell_49/BiasAdd/ReadVariableOp2H
"lstm_cell_49/MatMul/ReadVariableOp"lstm_cell_49/MatMul/ReadVariableOp2L
$lstm_cell_49/MatMul_1/ReadVariableOp$lstm_cell_49/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
.__inference_sequential_12_layer_call_fn_418517

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
I__inference_sequential_12_layer_call_and_return_conditional_losses_4179342
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
�

�
.__inference_sequential_12_layer_call_fn_417953
lstm_24_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_4179342
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
_user_specified_namelstm_24_input
�?
�
while_body_417660
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_48_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_48_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_48_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_48_matmul_readvariableop_resource:	�F
3while_lstm_cell_48_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_48_biasadd_readvariableop_resource:	���)while/lstm_cell_48/BiasAdd/ReadVariableOp�(while/lstm_cell_48/MatMul/ReadVariableOp�*while/lstm_cell_48/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_48_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_48/MatMul/ReadVariableOp�
while/lstm_cell_48/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul�
*while/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_48_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_48/MatMul_1/ReadVariableOp�
while/lstm_cell_48/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/MatMul_1�
while/lstm_cell_48/addAddV2#while/lstm_cell_48/MatMul:product:0%while/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/add�
)while/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_48_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_48/BiasAdd/ReadVariableOp�
while/lstm_cell_48/BiasAddBiasAddwhile/lstm_cell_48/add:z:01while/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_48/BiasAdd�
"while/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_48/split/split_dim�
while/lstm_cell_48/splitSplit+while/lstm_cell_48/split/split_dim:output:0#while/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_48/split�
while/lstm_cell_48/SigmoidSigmoid!while/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid�
while/lstm_cell_48/Sigmoid_1Sigmoid!while/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_1�
while/lstm_cell_48/mulMul while/lstm_cell_48/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul�
while/lstm_cell_48/ReluRelu!while/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu�
while/lstm_cell_48/mul_1Mulwhile/lstm_cell_48/Sigmoid:y:0%while/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_1�
while/lstm_cell_48/add_1AddV2while/lstm_cell_48/mul:z:0while/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/add_1�
while/lstm_cell_48/Sigmoid_2Sigmoid!while/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Sigmoid_2�
while/lstm_cell_48/Relu_1Reluwhile/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/Relu_1�
while/lstm_cell_48/mul_2Mul while/lstm_cell_48/Sigmoid_2:y:0'while/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_48/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_48/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_48/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_48/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_48/BiasAdd/ReadVariableOp)^while/lstm_cell_48/MatMul/ReadVariableOp+^while/lstm_cell_48/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_48_biasadd_readvariableop_resource4while_lstm_cell_48_biasadd_readvariableop_resource_0"l
3while_lstm_cell_48_matmul_1_readvariableop_resource5while_lstm_cell_48_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_48_matmul_readvariableop_resource3while_lstm_cell_48_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_48/BiasAdd/ReadVariableOp)while/lstm_cell_48/BiasAdd/ReadVariableOp2T
(while/lstm_cell_48/MatMul/ReadVariableOp(while/lstm_cell_48/MatMul/ReadVariableOp2X
*while/lstm_cell_48/MatMul_1/ReadVariableOp*while/lstm_cell_48/MatMul_1/ReadVariableOp: 
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
e
F__inference_dropout_12_layer_call_and_return_conditional_losses_420478

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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420693

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
�J
�

lstm_25_while_body_419057,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0:	@�P
=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �K
<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_49_matmul_readvariableop_resource:	@�N
;lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource:	 �I
:lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource:	���1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItem�
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp�
!lstm_25/while/lstm_cell_49/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_25/while/lstm_cell_49/MatMul�
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp�
#lstm_25/while/lstm_cell_49/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_25/while/lstm_cell_49/MatMul_1�
lstm_25/while/lstm_cell_49/addAddV2+lstm_25/while/lstm_cell_49/MatMul:product:0-lstm_25/while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_25/while/lstm_cell_49/add�
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp�
"lstm_25/while/lstm_cell_49/BiasAddBiasAdd"lstm_25/while/lstm_cell_49/add:z:09lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_25/while/lstm_cell_49/BiasAdd�
*lstm_25/while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_49/split/split_dim�
 lstm_25/while/lstm_cell_49/splitSplit3lstm_25/while/lstm_cell_49/split/split_dim:output:0+lstm_25/while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_25/while/lstm_cell_49/split�
"lstm_25/while/lstm_cell_49/SigmoidSigmoid)lstm_25/while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_25/while/lstm_cell_49/Sigmoid�
$lstm_25/while/lstm_cell_49/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_25/while/lstm_cell_49/Sigmoid_1�
lstm_25/while/lstm_cell_49/mulMul(lstm_25/while/lstm_cell_49/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_25/while/lstm_cell_49/mul�
lstm_25/while/lstm_cell_49/ReluRelu)lstm_25/while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_25/while/lstm_cell_49/Relu�
 lstm_25/while/lstm_cell_49/mul_1Mul&lstm_25/while/lstm_cell_49/Sigmoid:y:0-lstm_25/while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/mul_1�
 lstm_25/while/lstm_cell_49/add_1AddV2"lstm_25/while/lstm_cell_49/mul:z:0$lstm_25/while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/add_1�
$lstm_25/while/lstm_cell_49/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_25/while/lstm_cell_49/Sigmoid_2�
!lstm_25/while/lstm_cell_49/Relu_1Relu$lstm_25/while/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_25/while/lstm_cell_49/Relu_1�
 lstm_25/while/lstm_cell_49/mul_2Mul(lstm_25/while/lstm_cell_49/Sigmoid_2:y:0/lstm_25/while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_25/while/lstm_cell_49/mul_2�
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_49/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_25/while/TensorArrayV2Write/TensorListSetIteml
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add/y�
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/addp
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add_1/y�
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/add_1�
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity�
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_1�
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_2�
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_3�
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_49/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_25/while/Identity_4�
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_49/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_25/while/Identity_5�
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_25/while/NoOp"9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_49_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_49_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_49_matmul_readvariableop_resource;lstm_25_while_lstm_cell_49_matmul_readvariableop_resource_0"�
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_49/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_49/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_418467
lstm_24_input!
lstm_24_418446:	�!
lstm_24_418448:	@�
lstm_24_418450:	�!
lstm_25_418453:	@�!
lstm_25_418455:	 �
lstm_25_418457:	�!
dense_12_418461: 
dense_12_418463:
identity�� dense_12/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�lstm_24/StatefulPartitionedCall�lstm_25/StatefulPartitionedCall�
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_418446lstm_24_418448lstm_24_418450*
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_4183232!
lstm_24/StatefulPartitionedCall�
lstm_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0lstm_25_418453lstm_25_418455lstm_25_418457*
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4181502!
lstm_25/StatefulPartitionedCall�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_4179832$
"dropout_12/StatefulPartitionedCall�
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_12_418461dense_12_418463*
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
D__inference_dense_12_layer_call_and_return_conditional_losses_4179272"
 dense_12/StatefulPartitionedCall�
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_12/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_24_input
�[
�
C__inference_lstm_24_layer_call_and_return_conditional_losses_417744

inputs>
+lstm_cell_48_matmul_readvariableop_resource:	�@
-lstm_cell_48_matmul_1_readvariableop_resource:	@�;
,lstm_cell_48_biasadd_readvariableop_resource:	�
identity��#lstm_cell_48/BiasAdd/ReadVariableOp�"lstm_cell_48/MatMul/ReadVariableOp�$lstm_cell_48/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_48/MatMul/ReadVariableOpReadVariableOp+lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_48/MatMul/ReadVariableOp�
lstm_cell_48/MatMulMatMulstrided_slice_2:output:0*lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul�
$lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_cell_48/MatMul_1MatMulzeros:output:0,lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/MatMul_1�
lstm_cell_48/addAddV2lstm_cell_48/MatMul:product:0lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/add�
#lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_cell_48/BiasAddBiasAddlstm_cell_48/add:z:0+lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_48/BiasAdd~
lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_48/split/split_dim�
lstm_cell_48/splitSplit%lstm_cell_48/split/split_dim:output:0lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_48/split�
lstm_cell_48/SigmoidSigmoidlstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid�
lstm_cell_48/Sigmoid_1Sigmoidlstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_1�
lstm_cell_48/mulMullstm_cell_48/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul}
lstm_cell_48/ReluRelulstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu�
lstm_cell_48/mul_1Mullstm_cell_48/Sigmoid:y:0lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_1�
lstm_cell_48/add_1AddV2lstm_cell_48/mul:z:0lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/add_1�
lstm_cell_48/Sigmoid_2Sigmoidlstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Sigmoid_2|
lstm_cell_48/Relu_1Relulstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/Relu_1�
lstm_cell_48/mul_2Mullstm_cell_48/Sigmoid_2:y:0!lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_48/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_48_matmul_readvariableop_resource-lstm_cell_48_matmul_1_readvariableop_resource,lstm_cell_48_biasadd_readvariableop_resource*
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
while_body_417660*
condR
while_cond_417659*K
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
NoOpNoOp$^lstm_cell_48/BiasAdd/ReadVariableOp#^lstm_cell_48/MatMul/ReadVariableOp%^lstm_cell_48/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_48/BiasAdd/ReadVariableOp#lstm_cell_48/BiasAdd/ReadVariableOp2H
"lstm_cell_48/MatMul/ReadVariableOp"lstm_cell_48/MatMul/ReadVariableOp2L
$lstm_cell_48/MatMul_1/ReadVariableOp$lstm_cell_48/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_420216
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_49_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_49_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_49_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_49_matmul_readvariableop_resource:	@�F
3while_lstm_cell_49_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_49_biasadd_readvariableop_resource:	���)while/lstm_cell_49/BiasAdd/ReadVariableOp�(while/lstm_cell_49/MatMul/ReadVariableOp�*while/lstm_cell_49/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_49_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_49/MatMul/ReadVariableOp�
while/lstm_cell_49/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul�
*while/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_49_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_49/MatMul_1/ReadVariableOp�
while/lstm_cell_49/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/MatMul_1�
while/lstm_cell_49/addAddV2#while/lstm_cell_49/MatMul:product:0%while/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/add�
)while/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_49_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_49/BiasAdd/ReadVariableOp�
while/lstm_cell_49/BiasAddBiasAddwhile/lstm_cell_49/add:z:01while/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_49/BiasAdd�
"while/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_49/split/split_dim�
while/lstm_cell_49/splitSplit+while/lstm_cell_49/split/split_dim:output:0#while/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_49/split�
while/lstm_cell_49/SigmoidSigmoid!while/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid�
while/lstm_cell_49/Sigmoid_1Sigmoid!while/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_1�
while/lstm_cell_49/mulMul while/lstm_cell_49/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul�
while/lstm_cell_49/ReluRelu!while/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu�
while/lstm_cell_49/mul_1Mulwhile/lstm_cell_49/Sigmoid:y:0%while/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_1�
while/lstm_cell_49/add_1AddV2while/lstm_cell_49/mul:z:0while/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/add_1�
while/lstm_cell_49/Sigmoid_2Sigmoid!while/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Sigmoid_2�
while/lstm_cell_49/Relu_1Reluwhile/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/Relu_1�
while/lstm_cell_49/mul_2Mul while/lstm_cell_49/Sigmoid_2:y:0'while/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_49/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_49/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_49/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_49/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_49/BiasAdd/ReadVariableOp)^while/lstm_cell_49/MatMul/ReadVariableOp+^while/lstm_cell_49/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_49_biasadd_readvariableop_resource4while_lstm_cell_49_biasadd_readvariableop_resource_0"l
3while_lstm_cell_49_matmul_1_readvariableop_resource5while_lstm_cell_49_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_49_matmul_readvariableop_resource3while_lstm_cell_49_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_49/BiasAdd/ReadVariableOp)while/lstm_cell_49/BiasAdd/ReadVariableOp2T
(while/lstm_cell_49/MatMul/ReadVariableOp(while/lstm_cell_49/MatMul/ReadVariableOp2X
*while/lstm_cell_49/MatMul_1/ReadVariableOp*while/lstm_cell_49/MatMul_1/ReadVariableOp: 
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
-__inference_lstm_cell_48_layer_call_fn_420514

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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4164022
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
�
�
-__inference_lstm_cell_48_layer_call_fn_420531

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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_4165482
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
(__inference_lstm_25_layer_call_fn_419847

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
C__inference_lstm_25_layer_call_and_return_conditional_losses_4181502
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
D__inference_dense_12_layer_call_and_return_conditional_losses_420497

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
�
�
-__inference_lstm_cell_49_layer_call_fn_420612

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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_4170322
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
��
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_418843

inputsF
3lstm_24_lstm_cell_48_matmul_readvariableop_resource:	�H
5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource:	@�C
4lstm_24_lstm_cell_48_biasadd_readvariableop_resource:	�F
3lstm_25_lstm_cell_49_matmul_readvariableop_resource:	@�H
5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource:	 �C
4lstm_25_lstm_cell_49_biasadd_readvariableop_resource:	�9
'dense_12_matmul_readvariableop_resource: 6
(dense_12_biasadd_readvariableop_resource:
identity��dense_12/BiasAdd/ReadVariableOp�dense_12/MatMul/ReadVariableOp�+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�*lstm_24/lstm_cell_48/MatMul/ReadVariableOp�,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�lstm_24/while�+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�*lstm_25/lstm_cell_49/MatMul/ReadVariableOp�,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�lstm_25/whileT
lstm_24/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_24/Shape�
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice/stack�
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_1�
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_2�
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slicel
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros/mul/y�
lstm_24/zeros/mulMullstm_24/strided_slice:output:0lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/mulo
lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_24/zeros/Less/y�
lstm_24/zeros/LessLesslstm_24/zeros/mul:z:0lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/Lessr
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros/packed/1�
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros/packedo
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros/Const�
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/zerosp
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros_1/mul/y�
lstm_24/zeros_1/mulMullstm_24/strided_slice:output:0lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/muls
lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_24/zeros_1/Less/y�
lstm_24/zeros_1/LessLesslstm_24/zeros_1/mul:z:0lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/Lessv
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_24/zeros_1/packed/1�
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros_1/packeds
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros_1/Const�
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/zeros_1�
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose/perm�
lstm_24/transpose	Transposeinputslstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_24/transposeg
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:2
lstm_24/Shape_1�
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_1/stack�
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_1�
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_2�
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slice_1�
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_24/TensorArrayV2/element_shape�
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2�
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_24/TensorArrayUnstack/TensorListFromTensor�
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_2/stack�
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_1�
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_2�
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_24/strided_slice_2�
*lstm_24/lstm_cell_48/MatMul/ReadVariableOpReadVariableOp3lstm_24_lstm_cell_48_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_24/lstm_cell_48/MatMul/ReadVariableOp�
lstm_24/lstm_cell_48/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_48/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/MatMul�
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp�
lstm_24/lstm_cell_48/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/MatMul_1�
lstm_24/lstm_cell_48/addAddV2%lstm_24/lstm_cell_48/MatMul:product:0'lstm_24/lstm_cell_48/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/add�
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp�
lstm_24/lstm_cell_48/BiasAddBiasAddlstm_24/lstm_cell_48/add:z:03lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_24/lstm_cell_48/BiasAdd�
$lstm_24/lstm_cell_48/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_48/split/split_dim�
lstm_24/lstm_cell_48/splitSplit-lstm_24/lstm_cell_48/split/split_dim:output:0%lstm_24/lstm_cell_48/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_24/lstm_cell_48/split�
lstm_24/lstm_cell_48/SigmoidSigmoid#lstm_24/lstm_cell_48/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Sigmoid�
lstm_24/lstm_cell_48/Sigmoid_1Sigmoid#lstm_24/lstm_cell_48/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_24/lstm_cell_48/Sigmoid_1�
lstm_24/lstm_cell_48/mulMul"lstm_24/lstm_cell_48/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul�
lstm_24/lstm_cell_48/ReluRelu#lstm_24/lstm_cell_48/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Relu�
lstm_24/lstm_cell_48/mul_1Mul lstm_24/lstm_cell_48/Sigmoid:y:0'lstm_24/lstm_cell_48/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul_1�
lstm_24/lstm_cell_48/add_1AddV2lstm_24/lstm_cell_48/mul:z:0lstm_24/lstm_cell_48/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/add_1�
lstm_24/lstm_cell_48/Sigmoid_2Sigmoid#lstm_24/lstm_cell_48/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_24/lstm_cell_48/Sigmoid_2�
lstm_24/lstm_cell_48/Relu_1Relulstm_24/lstm_cell_48/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/Relu_1�
lstm_24/lstm_cell_48/mul_2Mul"lstm_24/lstm_cell_48/Sigmoid_2:y:0)lstm_24/lstm_cell_48/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_24/lstm_cell_48/mul_2�
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_24/TensorArrayV2_1/element_shape�
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2_1^
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/time�
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_24/while/maximum_iterationsz
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/while/loop_counter�
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_48_matmul_readvariableop_resource5lstm_24_lstm_cell_48_matmul_1_readvariableop_resource4lstm_24_lstm_cell_48_biasadd_readvariableop_resource*
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
lstm_24_while_body_418605*%
condR
lstm_24_while_cond_418604*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_24/while�
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_24/TensorArrayV2Stack/TensorListStack�
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_24/strided_slice_3/stack�
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_24/strided_slice_3/stack_1�
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_3/stack_2�
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_24/strided_slice_3�
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/perm�
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_24/transpose_1v
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/runtimee
lstm_25/ShapeShapelstm_24/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_25/Shape�
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice/stack�
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_1�
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_2�
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slicel
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros/mul/y�
lstm_25/zeros/mulMullstm_25/strided_slice:output:0lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/mulo
lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_25/zeros/Less/y�
lstm_25/zeros/LessLesslstm_25/zeros/mul:z:0lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/Lessr
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros/packed/1�
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros/packedo
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros/Const�
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/zerosp
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros_1/mul/y�
lstm_25/zeros_1/mulMullstm_25/strided_slice:output:0lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/muls
lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_25/zeros_1/Less/y�
lstm_25/zeros_1/LessLesslstm_25/zeros_1/mul:z:0lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/Lessv
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/zeros_1/packed/1�
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros_1/packeds
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros_1/Const�
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/zeros_1�
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm�
lstm_25/transpose	Transposelstm_24/transpose_1:y:0lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_25/transposeg
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:2
lstm_25/Shape_1�
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_1/stack�
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_1�
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_2�
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slice_1�
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_25/TensorArrayV2/element_shape�
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2�
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_25/TensorArrayUnstack/TensorListFromTensor�
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_2/stack�
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_1�
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_2�
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_25/strided_slice_2�
*lstm_25/lstm_cell_49/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_49_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_25/lstm_cell_49/MatMul/ReadVariableOp�
lstm_25/lstm_cell_49/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_49/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/MatMul�
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp�
lstm_25/lstm_cell_49/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/MatMul_1�
lstm_25/lstm_cell_49/addAddV2%lstm_25/lstm_cell_49/MatMul:product:0'lstm_25/lstm_cell_49/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/add�
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp�
lstm_25/lstm_cell_49/BiasAddBiasAddlstm_25/lstm_cell_49/add:z:03lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_25/lstm_cell_49/BiasAdd�
$lstm_25/lstm_cell_49/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_49/split/split_dim�
lstm_25/lstm_cell_49/splitSplit-lstm_25/lstm_cell_49/split/split_dim:output:0%lstm_25/lstm_cell_49/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_25/lstm_cell_49/split�
lstm_25/lstm_cell_49/SigmoidSigmoid#lstm_25/lstm_cell_49/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Sigmoid�
lstm_25/lstm_cell_49/Sigmoid_1Sigmoid#lstm_25/lstm_cell_49/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_25/lstm_cell_49/Sigmoid_1�
lstm_25/lstm_cell_49/mulMul"lstm_25/lstm_cell_49/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul�
lstm_25/lstm_cell_49/ReluRelu#lstm_25/lstm_cell_49/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Relu�
lstm_25/lstm_cell_49/mul_1Mul lstm_25/lstm_cell_49/Sigmoid:y:0'lstm_25/lstm_cell_49/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul_1�
lstm_25/lstm_cell_49/add_1AddV2lstm_25/lstm_cell_49/mul:z:0lstm_25/lstm_cell_49/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/add_1�
lstm_25/lstm_cell_49/Sigmoid_2Sigmoid#lstm_25/lstm_cell_49/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_25/lstm_cell_49/Sigmoid_2�
lstm_25/lstm_cell_49/Relu_1Relulstm_25/lstm_cell_49/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/Relu_1�
lstm_25/lstm_cell_49/mul_2Mul"lstm_25/lstm_cell_49/Sigmoid_2:y:0)lstm_25/lstm_cell_49/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_25/lstm_cell_49/mul_2�
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_25/TensorArrayV2_1/element_shape�
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2_1^
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/time�
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_25/while/maximum_iterationsz
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/while/loop_counter�
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_49_matmul_readvariableop_resource5lstm_25_lstm_cell_49_matmul_1_readvariableop_resource4lstm_25_lstm_cell_49_biasadd_readvariableop_resource*
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
lstm_25_while_body_418752*%
condR
lstm_25_while_cond_418751*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_25/while�
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_25/TensorArrayV2Stack/TensorListStack�
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_25/strided_slice_3/stack�
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_25/strided_slice_3/stack_1�
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_3/stack_2�
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_25/strided_slice_3�
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/perm�
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_25/transpose_1v
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/runtime�
dropout_12/IdentityIdentity lstm_25/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2
dropout_12/Identity�
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_12/MatMul/ReadVariableOp�
dense_12/MatMulMatMuldropout_12/Identity:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/MatMul�
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp�
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_12/BiasAddt
IdentityIdentitydense_12/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp,^lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp+^lstm_24/lstm_cell_48/MatMul/ReadVariableOp-^lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp^lstm_24/while,^lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_49/MatMul/ReadVariableOp-^lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp^lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2Z
+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp+lstm_24/lstm_cell_48/BiasAdd/ReadVariableOp2X
*lstm_24/lstm_cell_48/MatMul/ReadVariableOp*lstm_24/lstm_cell_48/MatMul/ReadVariableOp2\
,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp,lstm_24/lstm_cell_48/MatMul_1/ReadVariableOp2
lstm_24/whilelstm_24/while2Z
+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_49/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_49/MatMul/ReadVariableOp*lstm_25/lstm_cell_49/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_49/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while:S O
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
lstm_24_input:
serving_default_lstm_24_input:0���������<
dense_120
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
!: 2dense_12/kernel
:2dense_12/bias
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
.:,	�2lstm_24/lstm_cell_48/kernel
8:6	@�2%lstm_24/lstm_cell_48/recurrent_kernel
(:&�2lstm_24/lstm_cell_48/bias
.:,	@�2lstm_25/lstm_cell_49/kernel
8:6	 �2%lstm_25/lstm_cell_49/recurrent_kernel
(:&�2lstm_25/lstm_cell_49/bias
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
&:$ 2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
3:1	�2"Adam/lstm_24/lstm_cell_48/kernel/m
=:;	@�2,Adam/lstm_24/lstm_cell_48/recurrent_kernel/m
-:+�2 Adam/lstm_24/lstm_cell_48/bias/m
3:1	@�2"Adam/lstm_25/lstm_cell_49/kernel/m
=:;	 �2,Adam/lstm_25/lstm_cell_49/recurrent_kernel/m
-:+�2 Adam/lstm_25/lstm_cell_49/bias/m
&:$ 2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
3:1	�2"Adam/lstm_24/lstm_cell_48/kernel/v
=:;	@�2,Adam/lstm_24/lstm_cell_48/recurrent_kernel/v
-:+�2 Adam/lstm_24/lstm_cell_48/bias/v
3:1	@�2"Adam/lstm_25/lstm_cell_49/kernel/v
=:;	 �2,Adam/lstm_25/lstm_cell_49/recurrent_kernel/v
-:+�2 Adam/lstm_25/lstm_cell_49/bias/v
�B�
!__inference__wrapped_model_416327lstm_24_input"�
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
.__inference_sequential_12_layer_call_fn_417953
.__inference_sequential_12_layer_call_fn_418517
.__inference_sequential_12_layer_call_fn_418538
.__inference_sequential_12_layer_call_fn_418419�
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_418843
I__inference_sequential_12_layer_call_and_return_conditional_losses_419155
I__inference_sequential_12_layer_call_and_return_conditional_losses_418443
I__inference_sequential_12_layer_call_and_return_conditional_losses_418467�
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
(__inference_lstm_24_layer_call_fn_419166
(__inference_lstm_24_layer_call_fn_419177
(__inference_lstm_24_layer_call_fn_419188
(__inference_lstm_24_layer_call_fn_419199�
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_419350
C__inference_lstm_24_layer_call_and_return_conditional_losses_419501
C__inference_lstm_24_layer_call_and_return_conditional_losses_419652
C__inference_lstm_24_layer_call_and_return_conditional_losses_419803�
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
(__inference_lstm_25_layer_call_fn_419814
(__inference_lstm_25_layer_call_fn_419825
(__inference_lstm_25_layer_call_fn_419836
(__inference_lstm_25_layer_call_fn_419847�
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_419998
C__inference_lstm_25_layer_call_and_return_conditional_losses_420149
C__inference_lstm_25_layer_call_and_return_conditional_losses_420300
C__inference_lstm_25_layer_call_and_return_conditional_losses_420451�
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
+__inference_dropout_12_layer_call_fn_420456
+__inference_dropout_12_layer_call_fn_420461�
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
F__inference_dropout_12_layer_call_and_return_conditional_losses_420466
F__inference_dropout_12_layer_call_and_return_conditional_losses_420478�
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
)__inference_dense_12_layer_call_fn_420487�
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
D__inference_dense_12_layer_call_and_return_conditional_losses_420497�
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
$__inference_signature_wrapper_418496lstm_24_input"�
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
-__inference_lstm_cell_48_layer_call_fn_420514
-__inference_lstm_cell_48_layer_call_fn_420531�
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420563
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420595�
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
-__inference_lstm_cell_49_layer_call_fn_420612
-__inference_lstm_cell_49_layer_call_fn_420629�
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420661
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420693�
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
!__inference__wrapped_model_416327{&'()*+:�7
0�-
+�(
lstm_24_input���������
� "3�0
.
dense_12"�
dense_12����������
D__inference_dense_12_layer_call_and_return_conditional_losses_420497\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� |
)__inference_dense_12_layer_call_fn_420487O/�,
%�"
 �
inputs��������� 
� "�����������
F__inference_dropout_12_layer_call_and_return_conditional_losses_420466\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
F__inference_dropout_12_layer_call_and_return_conditional_losses_420478\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� ~
+__inference_dropout_12_layer_call_fn_420456O3�0
)�&
 �
inputs��������� 
p 
� "���������� ~
+__inference_dropout_12_layer_call_fn_420461O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_lstm_24_layer_call_and_return_conditional_losses_419350�&'(O�L
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_419501�&'(O�L
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_419652q&'(?�<
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
C__inference_lstm_24_layer_call_and_return_conditional_losses_419803q&'(?�<
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
(__inference_lstm_24_layer_call_fn_419166}&'(O�L
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
(__inference_lstm_24_layer_call_fn_419177}&'(O�L
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
(__inference_lstm_24_layer_call_fn_419188d&'(?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
(__inference_lstm_24_layer_call_fn_419199d&'(?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
C__inference_lstm_25_layer_call_and_return_conditional_losses_419998})*+O�L
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_420149})*+O�L
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_420300m)*+?�<
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
C__inference_lstm_25_layer_call_and_return_conditional_losses_420451m)*+?�<
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
(__inference_lstm_25_layer_call_fn_419814p)*+O�L
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
(__inference_lstm_25_layer_call_fn_419825p)*+O�L
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
(__inference_lstm_25_layer_call_fn_419836`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
(__inference_lstm_25_layer_call_fn_419847`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420563�&'(��}
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
H__inference_lstm_cell_48_layer_call_and_return_conditional_losses_420595�&'(��}
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
-__inference_lstm_cell_48_layer_call_fn_420514�&'(��}
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
-__inference_lstm_cell_48_layer_call_fn_420531�&'(��}
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420661�)*+��}
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
H__inference_lstm_cell_49_layer_call_and_return_conditional_losses_420693�)*+��}
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
-__inference_lstm_cell_49_layer_call_fn_420612�)*+��}
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
-__inference_lstm_cell_49_layer_call_fn_420629�)*+��}
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_418443u&'()*+B�?
8�5
+�(
lstm_24_input���������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_12_layer_call_and_return_conditional_losses_418467u&'()*+B�?
8�5
+�(
lstm_24_input���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_12_layer_call_and_return_conditional_losses_418843n&'()*+;�8
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_419155n&'()*+;�8
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
.__inference_sequential_12_layer_call_fn_417953h&'()*+B�?
8�5
+�(
lstm_24_input���������
p 

 
� "�����������
.__inference_sequential_12_layer_call_fn_418419h&'()*+B�?
8�5
+�(
lstm_24_input���������
p

 
� "�����������
.__inference_sequential_12_layer_call_fn_418517a&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "�����������
.__inference_sequential_12_layer_call_fn_418538a&'()*+;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_418496�&'()*+K�H
� 
A�>
<
lstm_24_input+�(
lstm_24_input���������"3�0
.
dense_12"�
dense_12���������