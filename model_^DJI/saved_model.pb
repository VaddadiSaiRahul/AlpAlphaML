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
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

: *
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
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
lstm_36/lstm_cell_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_36/lstm_cell_72/kernel
�
/lstm_36/lstm_cell_72/kernel/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell_72/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_36/lstm_cell_72/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*6
shared_name'%lstm_36/lstm_cell_72/recurrent_kernel
�
9lstm_36/lstm_cell_72/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_36/lstm_cell_72/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_36/lstm_cell_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_36/lstm_cell_72/bias
�
-lstm_36/lstm_cell_72/bias/Read/ReadVariableOpReadVariableOplstm_36/lstm_cell_72/bias*
_output_shapes	
:�*
dtype0
�
lstm_37/lstm_cell_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*,
shared_namelstm_37/lstm_cell_73/kernel
�
/lstm_37/lstm_cell_73/kernel/Read/ReadVariableOpReadVariableOplstm_37/lstm_cell_73/kernel*
_output_shapes
:	@�*
dtype0
�
%lstm_37/lstm_cell_73/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%lstm_37/lstm_cell_73/recurrent_kernel
�
9lstm_37/lstm_cell_73/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_37/lstm_cell_73/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
lstm_37/lstm_cell_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_37/lstm_cell_73/bias
�
-lstm_37/lstm_cell_73/bias/Read/ReadVariableOpReadVariableOplstm_37/lstm_cell_73/bias*
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
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_18/kernel/m
�
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

: *
dtype0
�
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_36/lstm_cell_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_36/lstm_cell_72/kernel/m
�
6Adam/lstm_36/lstm_cell_72/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_36/lstm_cell_72/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_36/lstm_cell_72/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_36/lstm_cell_72/recurrent_kernel/m
�
@Adam/lstm_36/lstm_cell_72/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_36/lstm_cell_72/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_36/lstm_cell_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_36/lstm_cell_72/bias/m
�
4Adam/lstm_36/lstm_cell_72/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_36/lstm_cell_72/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_37/lstm_cell_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_37/lstm_cell_73/kernel/m
�
6Adam/lstm_37/lstm_cell_73/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_37/lstm_cell_73/kernel/m*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_37/lstm_cell_73/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_37/lstm_cell_73/recurrent_kernel/m
�
@Adam/lstm_37/lstm_cell_73/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_37/lstm_cell_73/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_37/lstm_cell_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_37/lstm_cell_73/bias/m
�
4Adam/lstm_37/lstm_cell_73/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_37/lstm_cell_73/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_18/kernel/v
�
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

: *
dtype0
�
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_36/lstm_cell_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_36/lstm_cell_72/kernel/v
�
6Adam/lstm_36/lstm_cell_72/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_36/lstm_cell_72/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_36/lstm_cell_72/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_36/lstm_cell_72/recurrent_kernel/v
�
@Adam/lstm_36/lstm_cell_72/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_36/lstm_cell_72/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_36/lstm_cell_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_36/lstm_cell_72/bias/v
�
4Adam/lstm_36/lstm_cell_72/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_36/lstm_cell_72/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_37/lstm_cell_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_37/lstm_cell_73/kernel/v
�
6Adam/lstm_37/lstm_cell_73/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_37/lstm_cell_73/kernel/v*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_37/lstm_cell_73/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_37/lstm_cell_73/recurrent_kernel/v
�
@Adam/lstm_37/lstm_cell_73/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_37/lstm_cell_73/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_37/lstm_cell_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_37/lstm_cell_73/bias/v
�
4Adam/lstm_37/lstm_cell_73/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_37/lstm_cell_73/bias/v*
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
VARIABLE_VALUEdense_18/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_18/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_36/lstm_cell_72/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_36/lstm_cell_72/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_36/lstm_cell_72/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_37/lstm_cell_73/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_37/lstm_cell_73/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_37/lstm_cell_73/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_36/lstm_cell_72/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_36/lstm_cell_72/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_36/lstm_cell_72/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_37/lstm_cell_73/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_37/lstm_cell_73/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_37/lstm_cell_73/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_36/lstm_cell_72/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_36/lstm_cell_72/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_36/lstm_cell_72/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_37/lstm_cell_73/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_37/lstm_cell_73/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_37/lstm_cell_73/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_36_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_36_inputlstm_36/lstm_cell_72/kernel%lstm_36/lstm_cell_72/recurrent_kernellstm_36/lstm_cell_72/biaslstm_37/lstm_cell_73/kernel%lstm_37/lstm_cell_73/recurrent_kernellstm_37/lstm_cell_73/biasdense_18/kerneldense_18/bias*
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
$__inference_signature_wrapper_585912
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_36/lstm_cell_72/kernel/Read/ReadVariableOp9lstm_36/lstm_cell_72/recurrent_kernel/Read/ReadVariableOp-lstm_36/lstm_cell_72/bias/Read/ReadVariableOp/lstm_37/lstm_cell_73/kernel/Read/ReadVariableOp9lstm_37/lstm_cell_73/recurrent_kernel/Read/ReadVariableOp-lstm_37/lstm_cell_73/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp6Adam/lstm_36/lstm_cell_72/kernel/m/Read/ReadVariableOp@Adam/lstm_36/lstm_cell_72/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_36/lstm_cell_72/bias/m/Read/ReadVariableOp6Adam/lstm_37/lstm_cell_73/kernel/m/Read/ReadVariableOp@Adam/lstm_37/lstm_cell_73/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_37/lstm_cell_73/bias/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp6Adam/lstm_36/lstm_cell_72/kernel/v/Read/ReadVariableOp@Adam/lstm_36/lstm_cell_72/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_36/lstm_cell_72/bias/v/Read/ReadVariableOp6Adam/lstm_37/lstm_cell_73/kernel/v/Read/ReadVariableOp@Adam/lstm_37/lstm_cell_73/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_37/lstm_cell_73/bias/v/Read/ReadVariableOpConst*,
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
__inference__traced_save_588225
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_18/kerneldense_18/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_36/lstm_cell_72/kernel%lstm_36/lstm_cell_72/recurrent_kernellstm_36/lstm_cell_72/biaslstm_37/lstm_cell_73/kernel%lstm_37/lstm_cell_73/recurrent_kernellstm_37/lstm_cell_73/biastotalcountAdam/dense_18/kernel/mAdam/dense_18/bias/m"Adam/lstm_36/lstm_cell_72/kernel/m,Adam/lstm_36/lstm_cell_72/recurrent_kernel/m Adam/lstm_36/lstm_cell_72/bias/m"Adam/lstm_37/lstm_cell_73/kernel/m,Adam/lstm_37/lstm_cell_73/recurrent_kernel/m Adam/lstm_37/lstm_cell_73/bias/mAdam/dense_18/kernel/vAdam/dense_18/bias/v"Adam/lstm_36/lstm_cell_72/kernel/v,Adam/lstm_36/lstm_cell_72/recurrent_kernel/v Adam/lstm_36/lstm_cell_72/bias/v"Adam/lstm_37/lstm_cell_73/kernel/v,Adam/lstm_37/lstm_cell_73/recurrent_kernel/v Adam/lstm_37/lstm_cell_73/bias/v*+
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
"__inference__traced_restore_588328��#
�J
�

lstm_36_while_body_586021,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3+
'lstm_36_while_lstm_36_strided_slice_1_0g
clstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0:	�P
=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�K
<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
lstm_36_while_identity
lstm_36_while_identity_1
lstm_36_while_identity_2
lstm_36_while_identity_3
lstm_36_while_identity_4
lstm_36_while_identity_5)
%lstm_36_while_lstm_36_strided_slice_1e
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorL
9lstm_36_while_lstm_cell_72_matmul_readvariableop_resource:	�N
;lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource:	@�I
:lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource:	���1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0lstm_36_while_placeholderHlstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_36/while/TensorArrayV2Read/TensorListGetItem�
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�
!lstm_36/while/lstm_cell_72/MatMulMatMul8lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_36/while/lstm_cell_72/MatMul�
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
#lstm_36/while/lstm_cell_72/MatMul_1MatMullstm_36_while_placeholder_2:lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_36/while/lstm_cell_72/MatMul_1�
lstm_36/while/lstm_cell_72/addAddV2+lstm_36/while/lstm_cell_72/MatMul:product:0-lstm_36/while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_36/while/lstm_cell_72/add�
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�
"lstm_36/while/lstm_cell_72/BiasAddBiasAdd"lstm_36/while/lstm_cell_72/add:z:09lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_36/while/lstm_cell_72/BiasAdd�
*lstm_36/while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_36/while/lstm_cell_72/split/split_dim�
 lstm_36/while/lstm_cell_72/splitSplit3lstm_36/while/lstm_cell_72/split/split_dim:output:0+lstm_36/while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_36/while/lstm_cell_72/split�
"lstm_36/while/lstm_cell_72/SigmoidSigmoid)lstm_36/while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_36/while/lstm_cell_72/Sigmoid�
$lstm_36/while/lstm_cell_72/Sigmoid_1Sigmoid)lstm_36/while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_36/while/lstm_cell_72/Sigmoid_1�
lstm_36/while/lstm_cell_72/mulMul(lstm_36/while/lstm_cell_72/Sigmoid_1:y:0lstm_36_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_36/while/lstm_cell_72/mul�
lstm_36/while/lstm_cell_72/ReluRelu)lstm_36/while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_36/while/lstm_cell_72/Relu�
 lstm_36/while/lstm_cell_72/mul_1Mul&lstm_36/while/lstm_cell_72/Sigmoid:y:0-lstm_36/while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/mul_1�
 lstm_36/while/lstm_cell_72/add_1AddV2"lstm_36/while/lstm_cell_72/mul:z:0$lstm_36/while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/add_1�
$lstm_36/while/lstm_cell_72/Sigmoid_2Sigmoid)lstm_36/while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_36/while/lstm_cell_72/Sigmoid_2�
!lstm_36/while/lstm_cell_72/Relu_1Relu$lstm_36/while/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_36/while/lstm_cell_72/Relu_1�
 lstm_36/while/lstm_cell_72/mul_2Mul(lstm_36/while/lstm_cell_72/Sigmoid_2:y:0/lstm_36/while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/mul_2�
2lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_36_while_placeholder_1lstm_36_while_placeholder$lstm_36/while/lstm_cell_72/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_36/while/TensorArrayV2Write/TensorListSetIteml
lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_36/while/add/y�
lstm_36/while/addAddV2lstm_36_while_placeholderlstm_36/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_36/while/addp
lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_36/while/add_1/y�
lstm_36/while/add_1AddV2(lstm_36_while_lstm_36_while_loop_counterlstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_36/while/add_1�
lstm_36/while/IdentityIdentitylstm_36/while/add_1:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity�
lstm_36/while/Identity_1Identity.lstm_36_while_lstm_36_while_maximum_iterations^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_1�
lstm_36/while/Identity_2Identitylstm_36/while/add:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_2�
lstm_36/while/Identity_3IdentityBlstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_3�
lstm_36/while/Identity_4Identity$lstm_36/while/lstm_cell_72/mul_2:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_36/while/Identity_4�
lstm_36/while/Identity_5Identity$lstm_36/while/lstm_cell_72/add_1:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_36/while/Identity_5�
lstm_36/while/NoOpNoOp2^lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp1^lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp3^lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_36/while/NoOp"9
lstm_36_while_identitylstm_36/while/Identity:output:0"=
lstm_36_while_identity_1!lstm_36/while/Identity_1:output:0"=
lstm_36_while_identity_2!lstm_36/while/Identity_2:output:0"=
lstm_36_while_identity_3!lstm_36/while/Identity_3:output:0"=
lstm_36_while_identity_4!lstm_36/while/Identity_4:output:0"=
lstm_36_while_identity_5!lstm_36/while/Identity_5:output:0"P
%lstm_36_while_lstm_36_strided_slice_1'lstm_36_while_lstm_36_strided_slice_1_0"z
:lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0"|
;lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0"x
9lstm_36_while_lstm_cell_72_matmul_readvariableop_resource;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0"�
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp2d
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp2h
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587716

inputs>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_587632*
condR
while_cond_587631*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_584041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_584041___redundant_placeholder04
0while_while_cond_584041___redundant_placeholder14
0while_while_cond_584041___redundant_placeholder24
0while_while_cond_584041___redundant_placeholder3
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
(__inference_lstm_36_layer_call_fn_586582
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5839012
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
�
)__inference_dense_18_layer_call_fn_587903

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
D__inference_dense_18_layer_call_and_return_conditional_losses_5853432
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
�
�
while_cond_587782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_587782___redundant_placeholder04
0while_while_cond_587782___redundant_placeholder14
0while_while_cond_587782___redundant_placeholder24
0while_while_cond_587782___redundant_placeholder3
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
.__inference_sequential_18_layer_call_fn_585835
lstm_36_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_18_layer_call_and_return_conditional_losses_5857952
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
_user_specified_namelstm_36_input
�
�
while_cond_584671
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_584671___redundant_placeholder04
0while_while_cond_584671___redundant_placeholder14
0while_while_cond_584671___redundant_placeholder24
0while_while_cond_584671___redundant_placeholder3
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
-__inference_lstm_cell_73_layer_call_fn_588045

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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5845942
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
�
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_585883
lstm_36_input!
lstm_36_585862:	�!
lstm_36_585864:	@�
lstm_36_585866:	�!
lstm_37_585869:	@�!
lstm_37_585871:	 �
lstm_37_585873:	�!
dense_18_585877: 
dense_18_585879:
identity�� dense_18/StatefulPartitionedCall�"dropout_18/StatefulPartitionedCall�lstm_36/StatefulPartitionedCall�lstm_37/StatefulPartitionedCall�
lstm_36/StatefulPartitionedCallStatefulPartitionedCalllstm_36_inputlstm_36_585862lstm_36_585864lstm_36_585866*
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5857392!
lstm_36/StatefulPartitionedCall�
lstm_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0lstm_37_585869lstm_37_585871lstm_37_585873*
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5855662!
lstm_37/StatefulPartitionedCall�
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_37/StatefulPartitionedCall:output:0*
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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853992$
"dropout_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_18_585877dense_18_585879*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_5853432"
 dense_18/StatefulPartitionedCall�
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_18/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_36_input
�
�
while_cond_583831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_583831___redundant_placeholder04
0while_while_cond_583831___redundant_placeholder14
0while_while_cond_583831___redundant_placeholder24
0while_while_cond_583831___redundant_placeholder3
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
�%
�
while_body_583832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_72_583856_0:	�.
while_lstm_cell_72_583858_0:	@�*
while_lstm_cell_72_583860_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_72_583856:	�,
while_lstm_cell_72_583858:	@�(
while_lstm_cell_72_583860:	���*while/lstm_cell_72/StatefulPartitionedCall�
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
*while/lstm_cell_72/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_72_583856_0while_lstm_cell_72_583858_0while_lstm_cell_72_583860_0*
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5838182,
*while/lstm_cell_72/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_72/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_72/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_72/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_72/StatefulPartitionedCall*"
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
while_lstm_cell_72_583856while_lstm_cell_72_583856_0"8
while_lstm_cell_72_583858while_lstm_cell_72_583858_0"8
while_lstm_cell_72_583860while_lstm_cell_72_583860_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_72/StatefulPartitionedCall*while/lstm_cell_72/StatefulPartitionedCall: 
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
�
(__inference_lstm_36_layer_call_fn_586593
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5841112
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
while_cond_587329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_587329___redundant_placeholder04
0while_while_cond_587329___redundant_placeholder14
0while_while_cond_587329___redundant_placeholder24
0while_while_cond_587329___redundant_placeholder3
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_588011

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
�	
�
$__inference_signature_wrapper_585912
lstm_36_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_5837432
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
_user_specified_namelstm_36_input
��
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_586259

inputsF
3lstm_36_lstm_cell_72_matmul_readvariableop_resource:	�H
5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource:	@�C
4lstm_36_lstm_cell_72_biasadd_readvariableop_resource:	�F
3lstm_37_lstm_cell_73_matmul_readvariableop_resource:	@�H
5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource:	 �C
4lstm_37_lstm_cell_73_biasadd_readvariableop_resource:	�9
'dense_18_matmul_readvariableop_resource: 6
(dense_18_biasadd_readvariableop_resource:
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�*lstm_36/lstm_cell_72/MatMul/ReadVariableOp�,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�lstm_36/while�+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�*lstm_37/lstm_cell_73/MatMul/ReadVariableOp�,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�lstm_37/whileT
lstm_36/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_36/Shape�
lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice/stack�
lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_36/strided_slice/stack_1�
lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_36/strided_slice/stack_2�
lstm_36/strided_sliceStridedSlicelstm_36/Shape:output:0$lstm_36/strided_slice/stack:output:0&lstm_36/strided_slice/stack_1:output:0&lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_36/strided_slicel
lstm_36/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros/mul/y�
lstm_36/zeros/mulMullstm_36/strided_slice:output:0lstm_36/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros/mulo
lstm_36/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_36/zeros/Less/y�
lstm_36/zeros/LessLesslstm_36/zeros/mul:z:0lstm_36/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros/Lessr
lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros/packed/1�
lstm_36/zeros/packedPacklstm_36/strided_slice:output:0lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_36/zeros/packedo
lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/zeros/Const�
lstm_36/zerosFilllstm_36/zeros/packed:output:0lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/zerosp
lstm_36/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros_1/mul/y�
lstm_36/zeros_1/mulMullstm_36/strided_slice:output:0lstm_36/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros_1/muls
lstm_36/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_36/zeros_1/Less/y�
lstm_36/zeros_1/LessLesslstm_36/zeros_1/mul:z:0lstm_36/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros_1/Lessv
lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros_1/packed/1�
lstm_36/zeros_1/packedPacklstm_36/strided_slice:output:0!lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_36/zeros_1/packeds
lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/zeros_1/Const�
lstm_36/zeros_1Filllstm_36/zeros_1/packed:output:0lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/zeros_1�
lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_36/transpose/perm�
lstm_36/transpose	Transposeinputslstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_36/transposeg
lstm_36/Shape_1Shapelstm_36/transpose:y:0*
T0*
_output_shapes
:2
lstm_36/Shape_1�
lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice_1/stack�
lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_1/stack_1�
lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_1/stack_2�
lstm_36/strided_slice_1StridedSlicelstm_36/Shape_1:output:0&lstm_36/strided_slice_1/stack:output:0(lstm_36/strided_slice_1/stack_1:output:0(lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_36/strided_slice_1�
#lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_36/TensorArrayV2/element_shape�
lstm_36/TensorArrayV2TensorListReserve,lstm_36/TensorArrayV2/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_36/TensorArrayV2�
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_36/transpose:y:0Flstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_36/TensorArrayUnstack/TensorListFromTensor�
lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice_2/stack�
lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_2/stack_1�
lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_2/stack_2�
lstm_36/strided_slice_2StridedSlicelstm_36/transpose:y:0&lstm_36/strided_slice_2/stack:output:0(lstm_36/strided_slice_2/stack_1:output:0(lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_36/strided_slice_2�
*lstm_36/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_36/lstm_cell_72/MatMul/ReadVariableOp�
lstm_36/lstm_cell_72/MatMulMatMul lstm_36/strided_slice_2:output:02lstm_36/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/MatMul�
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_36/lstm_cell_72/MatMul_1MatMullstm_36/zeros:output:04lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/MatMul_1�
lstm_36/lstm_cell_72/addAddV2%lstm_36/lstm_cell_72/MatMul:product:0'lstm_36/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/add�
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_36/lstm_cell_72/BiasAddBiasAddlstm_36/lstm_cell_72/add:z:03lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/BiasAdd�
$lstm_36/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_36/lstm_cell_72/split/split_dim�
lstm_36/lstm_cell_72/splitSplit-lstm_36/lstm_cell_72/split/split_dim:output:0%lstm_36/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_36/lstm_cell_72/split�
lstm_36/lstm_cell_72/SigmoidSigmoid#lstm_36/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Sigmoid�
lstm_36/lstm_cell_72/Sigmoid_1Sigmoid#lstm_36/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_36/lstm_cell_72/Sigmoid_1�
lstm_36/lstm_cell_72/mulMul"lstm_36/lstm_cell_72/Sigmoid_1:y:0lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul�
lstm_36/lstm_cell_72/ReluRelu#lstm_36/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Relu�
lstm_36/lstm_cell_72/mul_1Mul lstm_36/lstm_cell_72/Sigmoid:y:0'lstm_36/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul_1�
lstm_36/lstm_cell_72/add_1AddV2lstm_36/lstm_cell_72/mul:z:0lstm_36/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/add_1�
lstm_36/lstm_cell_72/Sigmoid_2Sigmoid#lstm_36/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_36/lstm_cell_72/Sigmoid_2�
lstm_36/lstm_cell_72/Relu_1Relulstm_36/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Relu_1�
lstm_36/lstm_cell_72/mul_2Mul"lstm_36/lstm_cell_72/Sigmoid_2:y:0)lstm_36/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul_2�
%lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_36/TensorArrayV2_1/element_shape�
lstm_36/TensorArrayV2_1TensorListReserve.lstm_36/TensorArrayV2_1/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_36/TensorArrayV2_1^
lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_36/time�
 lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_36/while/maximum_iterationsz
lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_36/while/loop_counter�
lstm_36/whileWhile#lstm_36/while/loop_counter:output:0)lstm_36/while/maximum_iterations:output:0lstm_36/time:output:0 lstm_36/TensorArrayV2_1:handle:0lstm_36/zeros:output:0lstm_36/zeros_1:output:0 lstm_36/strided_slice_1:output:0?lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_36_lstm_cell_72_matmul_readvariableop_resource5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource4lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
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
lstm_36_while_body_586021*%
condR
lstm_36_while_cond_586020*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_36/while�
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_36/TensorArrayV2Stack/TensorListStackTensorListStacklstm_36/while:output:3Alstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_36/TensorArrayV2Stack/TensorListStack�
lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_36/strided_slice_3/stack�
lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_36/strided_slice_3/stack_1�
lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_3/stack_2�
lstm_36/strided_slice_3StridedSlice3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_36/strided_slice_3/stack:output:0(lstm_36/strided_slice_3/stack_1:output:0(lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_36/strided_slice_3�
lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_36/transpose_1/perm�
lstm_36/transpose_1	Transpose3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_36/transpose_1v
lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/runtimee
lstm_37/ShapeShapelstm_36/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_37/Shape�
lstm_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice/stack�
lstm_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_37/strided_slice/stack_1�
lstm_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_37/strided_slice/stack_2�
lstm_37/strided_sliceStridedSlicelstm_37/Shape:output:0$lstm_37/strided_slice/stack:output:0&lstm_37/strided_slice/stack_1:output:0&lstm_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_37/strided_slicel
lstm_37/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros/mul/y�
lstm_37/zeros/mulMullstm_37/strided_slice:output:0lstm_37/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros/mulo
lstm_37/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_37/zeros/Less/y�
lstm_37/zeros/LessLesslstm_37/zeros/mul:z:0lstm_37/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros/Lessr
lstm_37/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros/packed/1�
lstm_37/zeros/packedPacklstm_37/strided_slice:output:0lstm_37/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_37/zeros/packedo
lstm_37/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/zeros/Const�
lstm_37/zerosFilllstm_37/zeros/packed:output:0lstm_37/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/zerosp
lstm_37/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros_1/mul/y�
lstm_37/zeros_1/mulMullstm_37/strided_slice:output:0lstm_37/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros_1/muls
lstm_37/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_37/zeros_1/Less/y�
lstm_37/zeros_1/LessLesslstm_37/zeros_1/mul:z:0lstm_37/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros_1/Lessv
lstm_37/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros_1/packed/1�
lstm_37/zeros_1/packedPacklstm_37/strided_slice:output:0!lstm_37/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_37/zeros_1/packeds
lstm_37/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/zeros_1/Const�
lstm_37/zeros_1Filllstm_37/zeros_1/packed:output:0lstm_37/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/zeros_1�
lstm_37/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_37/transpose/perm�
lstm_37/transpose	Transposelstm_36/transpose_1:y:0lstm_37/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_37/transposeg
lstm_37/Shape_1Shapelstm_37/transpose:y:0*
T0*
_output_shapes
:2
lstm_37/Shape_1�
lstm_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice_1/stack�
lstm_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_1/stack_1�
lstm_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_1/stack_2�
lstm_37/strided_slice_1StridedSlicelstm_37/Shape_1:output:0&lstm_37/strided_slice_1/stack:output:0(lstm_37/strided_slice_1/stack_1:output:0(lstm_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_37/strided_slice_1�
#lstm_37/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_37/TensorArrayV2/element_shape�
lstm_37/TensorArrayV2TensorListReserve,lstm_37/TensorArrayV2/element_shape:output:0 lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_37/TensorArrayV2�
=lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_37/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_37/transpose:y:0Flstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_37/TensorArrayUnstack/TensorListFromTensor�
lstm_37/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice_2/stack�
lstm_37/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_2/stack_1�
lstm_37/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_2/stack_2�
lstm_37/strided_slice_2StridedSlicelstm_37/transpose:y:0&lstm_37/strided_slice_2/stack:output:0(lstm_37/strided_slice_2/stack_1:output:0(lstm_37/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_37/strided_slice_2�
*lstm_37/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3lstm_37_lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_37/lstm_cell_73/MatMul/ReadVariableOp�
lstm_37/lstm_cell_73/MatMulMatMul lstm_37/strided_slice_2:output:02lstm_37/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/MatMul�
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_37/lstm_cell_73/MatMul_1MatMullstm_37/zeros:output:04lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/MatMul_1�
lstm_37/lstm_cell_73/addAddV2%lstm_37/lstm_cell_73/MatMul:product:0'lstm_37/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/add�
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_37/lstm_cell_73/BiasAddBiasAddlstm_37/lstm_cell_73/add:z:03lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/BiasAdd�
$lstm_37/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_37/lstm_cell_73/split/split_dim�
lstm_37/lstm_cell_73/splitSplit-lstm_37/lstm_cell_73/split/split_dim:output:0%lstm_37/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_37/lstm_cell_73/split�
lstm_37/lstm_cell_73/SigmoidSigmoid#lstm_37/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Sigmoid�
lstm_37/lstm_cell_73/Sigmoid_1Sigmoid#lstm_37/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_37/lstm_cell_73/Sigmoid_1�
lstm_37/lstm_cell_73/mulMul"lstm_37/lstm_cell_73/Sigmoid_1:y:0lstm_37/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul�
lstm_37/lstm_cell_73/ReluRelu#lstm_37/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Relu�
lstm_37/lstm_cell_73/mul_1Mul lstm_37/lstm_cell_73/Sigmoid:y:0'lstm_37/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul_1�
lstm_37/lstm_cell_73/add_1AddV2lstm_37/lstm_cell_73/mul:z:0lstm_37/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/add_1�
lstm_37/lstm_cell_73/Sigmoid_2Sigmoid#lstm_37/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_37/lstm_cell_73/Sigmoid_2�
lstm_37/lstm_cell_73/Relu_1Relulstm_37/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Relu_1�
lstm_37/lstm_cell_73/mul_2Mul"lstm_37/lstm_cell_73/Sigmoid_2:y:0)lstm_37/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul_2�
%lstm_37/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_37/TensorArrayV2_1/element_shape�
lstm_37/TensorArrayV2_1TensorListReserve.lstm_37/TensorArrayV2_1/element_shape:output:0 lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_37/TensorArrayV2_1^
lstm_37/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/time�
 lstm_37/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_37/while/maximum_iterationsz
lstm_37/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/while/loop_counter�
lstm_37/whileWhile#lstm_37/while/loop_counter:output:0)lstm_37/while/maximum_iterations:output:0lstm_37/time:output:0 lstm_37/TensorArrayV2_1:handle:0lstm_37/zeros:output:0lstm_37/zeros_1:output:0 lstm_37/strided_slice_1:output:0?lstm_37/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_37_lstm_cell_73_matmul_readvariableop_resource5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource4lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
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
lstm_37_while_body_586168*%
condR
lstm_37_while_cond_586167*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_37/while�
8lstm_37/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_37/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_37/TensorArrayV2Stack/TensorListStackTensorListStacklstm_37/while:output:3Alstm_37/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_37/TensorArrayV2Stack/TensorListStack�
lstm_37/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_37/strided_slice_3/stack�
lstm_37/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_37/strided_slice_3/stack_1�
lstm_37/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_3/stack_2�
lstm_37/strided_slice_3StridedSlice3lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_37/strided_slice_3/stack:output:0(lstm_37/strided_slice_3/stack_1:output:0(lstm_37/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_37/strided_slice_3�
lstm_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_37/transpose_1/perm�
lstm_37/transpose_1	Transpose3lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_37/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_37/transpose_1v
lstm_37/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/runtime�
dropout_18/IdentityIdentity lstm_37/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2
dropout_18/Identity�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMuldropout_18/Identity:output:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_18/BiasAddt
IdentityIdentitydense_18/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp,^lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp+^lstm_36/lstm_cell_72/MatMul/ReadVariableOp-^lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp^lstm_36/while,^lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp+^lstm_37/lstm_cell_73/MatMul/ReadVariableOp-^lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp^lstm_37/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2Z
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp2X
*lstm_36/lstm_cell_72/MatMul/ReadVariableOp*lstm_36/lstm_cell_72/MatMul/ReadVariableOp2\
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp2
lstm_36/whilelstm_36/while2Z
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp2X
*lstm_37/lstm_cell_73/MatMul/ReadVariableOp*lstm_37/lstm_cell_73/MatMul/ReadVariableOp2\
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp2
lstm_37/whilelstm_37/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
lstm_36_while_cond_586325,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3.
*lstm_36_while_less_lstm_36_strided_slice_1D
@lstm_36_while_lstm_36_while_cond_586325___redundant_placeholder0D
@lstm_36_while_lstm_36_while_cond_586325___redundant_placeholder1D
@lstm_36_while_lstm_36_while_cond_586325___redundant_placeholder2D
@lstm_36_while_lstm_36_while_cond_586325___redundant_placeholder3
lstm_36_while_identity
�
lstm_36/while/LessLesslstm_36_while_placeholder*lstm_36_while_less_lstm_36_strided_slice_1*
T0*
_output_shapes
: 2
lstm_36/while/Lessu
lstm_36/while/IdentityIdentitylstm_36/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_36/while/Identity"9
lstm_36_while_identitylstm_36/while/Identity:output:0*(
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_587068

inputs>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_586984*
condR
while_cond_586983*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_586983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_586983___redundant_placeholder04
0while_while_cond_586983___redundant_placeholder14
0while_while_cond_586983___redundant_placeholder24
0while_while_cond_586983___redundant_placeholder3
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
ԓ
�	
!__inference__wrapped_model_583743
lstm_36_inputT
Asequential_18_lstm_36_lstm_cell_72_matmul_readvariableop_resource:	�V
Csequential_18_lstm_36_lstm_cell_72_matmul_1_readvariableop_resource:	@�Q
Bsequential_18_lstm_36_lstm_cell_72_biasadd_readvariableop_resource:	�T
Asequential_18_lstm_37_lstm_cell_73_matmul_readvariableop_resource:	@�V
Csequential_18_lstm_37_lstm_cell_73_matmul_1_readvariableop_resource:	 �Q
Bsequential_18_lstm_37_lstm_cell_73_biasadd_readvariableop_resource:	�G
5sequential_18_dense_18_matmul_readvariableop_resource: D
6sequential_18_dense_18_biasadd_readvariableop_resource:
identity��-sequential_18/dense_18/BiasAdd/ReadVariableOp�,sequential_18/dense_18/MatMul/ReadVariableOp�9sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�8sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp�:sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�sequential_18/lstm_36/while�9sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�8sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp�:sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�sequential_18/lstm_37/whilew
sequential_18/lstm_36/ShapeShapelstm_36_input*
T0*
_output_shapes
:2
sequential_18/lstm_36/Shape�
)sequential_18/lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_18/lstm_36/strided_slice/stack�
+sequential_18/lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_18/lstm_36/strided_slice/stack_1�
+sequential_18/lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_18/lstm_36/strided_slice/stack_2�
#sequential_18/lstm_36/strided_sliceStridedSlice$sequential_18/lstm_36/Shape:output:02sequential_18/lstm_36/strided_slice/stack:output:04sequential_18/lstm_36/strided_slice/stack_1:output:04sequential_18/lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_18/lstm_36/strided_slice�
!sequential_18/lstm_36/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_18/lstm_36/zeros/mul/y�
sequential_18/lstm_36/zeros/mulMul,sequential_18/lstm_36/strided_slice:output:0*sequential_18/lstm_36/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_18/lstm_36/zeros/mul�
"sequential_18/lstm_36/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_18/lstm_36/zeros/Less/y�
 sequential_18/lstm_36/zeros/LessLess#sequential_18/lstm_36/zeros/mul:z:0+sequential_18/lstm_36/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_18/lstm_36/zeros/Less�
$sequential_18/lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_18/lstm_36/zeros/packed/1�
"sequential_18/lstm_36/zeros/packedPack,sequential_18/lstm_36/strided_slice:output:0-sequential_18/lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_18/lstm_36/zeros/packed�
!sequential_18/lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_18/lstm_36/zeros/Const�
sequential_18/lstm_36/zerosFill+sequential_18/lstm_36/zeros/packed:output:0*sequential_18/lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_18/lstm_36/zeros�
#sequential_18/lstm_36/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_18/lstm_36/zeros_1/mul/y�
!sequential_18/lstm_36/zeros_1/mulMul,sequential_18/lstm_36/strided_slice:output:0,sequential_18/lstm_36/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_18/lstm_36/zeros_1/mul�
$sequential_18/lstm_36/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_18/lstm_36/zeros_1/Less/y�
"sequential_18/lstm_36/zeros_1/LessLess%sequential_18/lstm_36/zeros_1/mul:z:0-sequential_18/lstm_36/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_18/lstm_36/zeros_1/Less�
&sequential_18/lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2(
&sequential_18/lstm_36/zeros_1/packed/1�
$sequential_18/lstm_36/zeros_1/packedPack,sequential_18/lstm_36/strided_slice:output:0/sequential_18/lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_18/lstm_36/zeros_1/packed�
#sequential_18/lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_18/lstm_36/zeros_1/Const�
sequential_18/lstm_36/zeros_1Fill-sequential_18/lstm_36/zeros_1/packed:output:0,sequential_18/lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_18/lstm_36/zeros_1�
$sequential_18/lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_18/lstm_36/transpose/perm�
sequential_18/lstm_36/transpose	Transposelstm_36_input-sequential_18/lstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:���������2!
sequential_18/lstm_36/transpose�
sequential_18/lstm_36/Shape_1Shape#sequential_18/lstm_36/transpose:y:0*
T0*
_output_shapes
:2
sequential_18/lstm_36/Shape_1�
+sequential_18/lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_18/lstm_36/strided_slice_1/stack�
-sequential_18/lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_36/strided_slice_1/stack_1�
-sequential_18/lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_36/strided_slice_1/stack_2�
%sequential_18/lstm_36/strided_slice_1StridedSlice&sequential_18/lstm_36/Shape_1:output:04sequential_18/lstm_36/strided_slice_1/stack:output:06sequential_18/lstm_36/strided_slice_1/stack_1:output:06sequential_18/lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_18/lstm_36/strided_slice_1�
1sequential_18/lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_18/lstm_36/TensorArrayV2/element_shape�
#sequential_18/lstm_36/TensorArrayV2TensorListReserve:sequential_18/lstm_36/TensorArrayV2/element_shape:output:0.sequential_18/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_18/lstm_36/TensorArrayV2�
Ksequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2M
Ksequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_18/lstm_36/transpose:y:0Tsequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensor�
+sequential_18/lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_18/lstm_36/strided_slice_2/stack�
-sequential_18/lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_36/strided_slice_2/stack_1�
-sequential_18/lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_36/strided_slice_2/stack_2�
%sequential_18/lstm_36/strided_slice_2StridedSlice#sequential_18/lstm_36/transpose:y:04sequential_18/lstm_36/strided_slice_2/stack:output:06sequential_18/lstm_36/strided_slice_2/stack_1:output:06sequential_18/lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2'
%sequential_18/lstm_36/strided_slice_2�
8sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOpReadVariableOpAsequential_18_lstm_36_lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02:
8sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp�
)sequential_18/lstm_36/lstm_cell_72/MatMulMatMul.sequential_18/lstm_36/strided_slice_2:output:0@sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_18/lstm_36/lstm_cell_72/MatMul�
:sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOpCsequential_18_lstm_36_lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02<
:sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�
+sequential_18/lstm_36/lstm_cell_72/MatMul_1MatMul$sequential_18/lstm_36/zeros:output:0Bsequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_18/lstm_36/lstm_cell_72/MatMul_1�
&sequential_18/lstm_36/lstm_cell_72/addAddV23sequential_18/lstm_36/lstm_cell_72/MatMul:product:05sequential_18/lstm_36/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_18/lstm_36/lstm_cell_72/add�
9sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOpBsequential_18_lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�
*sequential_18/lstm_36/lstm_cell_72/BiasAddBiasAdd*sequential_18/lstm_36/lstm_cell_72/add:z:0Asequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_18/lstm_36/lstm_cell_72/BiasAdd�
2sequential_18/lstm_36/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_18/lstm_36/lstm_cell_72/split/split_dim�
(sequential_18/lstm_36/lstm_cell_72/splitSplit;sequential_18/lstm_36/lstm_cell_72/split/split_dim:output:03sequential_18/lstm_36/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2*
(sequential_18/lstm_36/lstm_cell_72/split�
*sequential_18/lstm_36/lstm_cell_72/SigmoidSigmoid1sequential_18/lstm_36/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2,
*sequential_18/lstm_36/lstm_cell_72/Sigmoid�
,sequential_18/lstm_36/lstm_cell_72/Sigmoid_1Sigmoid1sequential_18/lstm_36/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2.
,sequential_18/lstm_36/lstm_cell_72/Sigmoid_1�
&sequential_18/lstm_36/lstm_cell_72/mulMul0sequential_18/lstm_36/lstm_cell_72/Sigmoid_1:y:0&sequential_18/lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:���������@2(
&sequential_18/lstm_36/lstm_cell_72/mul�
'sequential_18/lstm_36/lstm_cell_72/ReluRelu1sequential_18/lstm_36/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2)
'sequential_18/lstm_36/lstm_cell_72/Relu�
(sequential_18/lstm_36/lstm_cell_72/mul_1Mul.sequential_18/lstm_36/lstm_cell_72/Sigmoid:y:05sequential_18/lstm_36/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_18/lstm_36/lstm_cell_72/mul_1�
(sequential_18/lstm_36/lstm_cell_72/add_1AddV2*sequential_18/lstm_36/lstm_cell_72/mul:z:0,sequential_18/lstm_36/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2*
(sequential_18/lstm_36/lstm_cell_72/add_1�
,sequential_18/lstm_36/lstm_cell_72/Sigmoid_2Sigmoid1sequential_18/lstm_36/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2.
,sequential_18/lstm_36/lstm_cell_72/Sigmoid_2�
)sequential_18/lstm_36/lstm_cell_72/Relu_1Relu,sequential_18/lstm_36/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2+
)sequential_18/lstm_36/lstm_cell_72/Relu_1�
(sequential_18/lstm_36/lstm_cell_72/mul_2Mul0sequential_18/lstm_36/lstm_cell_72/Sigmoid_2:y:07sequential_18/lstm_36/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2*
(sequential_18/lstm_36/lstm_cell_72/mul_2�
3sequential_18/lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   25
3sequential_18/lstm_36/TensorArrayV2_1/element_shape�
%sequential_18/lstm_36/TensorArrayV2_1TensorListReserve<sequential_18/lstm_36/TensorArrayV2_1/element_shape:output:0.sequential_18/lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_18/lstm_36/TensorArrayV2_1z
sequential_18/lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_18/lstm_36/time�
.sequential_18/lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_18/lstm_36/while/maximum_iterations�
(sequential_18/lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_18/lstm_36/while/loop_counter�
sequential_18/lstm_36/whileWhile1sequential_18/lstm_36/while/loop_counter:output:07sequential_18/lstm_36/while/maximum_iterations:output:0#sequential_18/lstm_36/time:output:0.sequential_18/lstm_36/TensorArrayV2_1:handle:0$sequential_18/lstm_36/zeros:output:0&sequential_18/lstm_36/zeros_1:output:0.sequential_18/lstm_36/strided_slice_1:output:0Msequential_18/lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_18_lstm_36_lstm_cell_72_matmul_readvariableop_resourceCsequential_18_lstm_36_lstm_cell_72_matmul_1_readvariableop_resourceBsequential_18_lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
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
'sequential_18_lstm_36_while_body_583505*3
cond+R)
'sequential_18_lstm_36_while_cond_583504*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
sequential_18/lstm_36/while�
Fsequential_18/lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2H
Fsequential_18/lstm_36/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_18/lstm_36/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_18/lstm_36/while:output:3Osequential_18/lstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02:
8sequential_18/lstm_36/TensorArrayV2Stack/TensorListStack�
+sequential_18/lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_18/lstm_36/strided_slice_3/stack�
-sequential_18/lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_18/lstm_36/strided_slice_3/stack_1�
-sequential_18/lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_36/strided_slice_3/stack_2�
%sequential_18/lstm_36/strided_slice_3StridedSliceAsequential_18/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:04sequential_18/lstm_36/strided_slice_3/stack:output:06sequential_18/lstm_36/strided_slice_3/stack_1:output:06sequential_18/lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_18/lstm_36/strided_slice_3�
&sequential_18/lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_18/lstm_36/transpose_1/perm�
!sequential_18/lstm_36/transpose_1	TransposeAsequential_18/lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_18/lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2#
!sequential_18/lstm_36/transpose_1�
sequential_18/lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_18/lstm_36/runtime�
sequential_18/lstm_37/ShapeShape%sequential_18/lstm_36/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_18/lstm_37/Shape�
)sequential_18/lstm_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_18/lstm_37/strided_slice/stack�
+sequential_18/lstm_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_18/lstm_37/strided_slice/stack_1�
+sequential_18/lstm_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_18/lstm_37/strided_slice/stack_2�
#sequential_18/lstm_37/strided_sliceStridedSlice$sequential_18/lstm_37/Shape:output:02sequential_18/lstm_37/strided_slice/stack:output:04sequential_18/lstm_37/strided_slice/stack_1:output:04sequential_18/lstm_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_18/lstm_37/strided_slice�
!sequential_18/lstm_37/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_18/lstm_37/zeros/mul/y�
sequential_18/lstm_37/zeros/mulMul,sequential_18/lstm_37/strided_slice:output:0*sequential_18/lstm_37/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_18/lstm_37/zeros/mul�
"sequential_18/lstm_37/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2$
"sequential_18/lstm_37/zeros/Less/y�
 sequential_18/lstm_37/zeros/LessLess#sequential_18/lstm_37/zeros/mul:z:0+sequential_18/lstm_37/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_18/lstm_37/zeros/Less�
$sequential_18/lstm_37/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_18/lstm_37/zeros/packed/1�
"sequential_18/lstm_37/zeros/packedPack,sequential_18/lstm_37/strided_slice:output:0-sequential_18/lstm_37/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_18/lstm_37/zeros/packed�
!sequential_18/lstm_37/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_18/lstm_37/zeros/Const�
sequential_18/lstm_37/zerosFill+sequential_18/lstm_37/zeros/packed:output:0*sequential_18/lstm_37/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_18/lstm_37/zeros�
#sequential_18/lstm_37/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_18/lstm_37/zeros_1/mul/y�
!sequential_18/lstm_37/zeros_1/mulMul,sequential_18/lstm_37/strided_slice:output:0,sequential_18/lstm_37/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_18/lstm_37/zeros_1/mul�
$sequential_18/lstm_37/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2&
$sequential_18/lstm_37/zeros_1/Less/y�
"sequential_18/lstm_37/zeros_1/LessLess%sequential_18/lstm_37/zeros_1/mul:z:0-sequential_18/lstm_37/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_18/lstm_37/zeros_1/Less�
&sequential_18/lstm_37/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_18/lstm_37/zeros_1/packed/1�
$sequential_18/lstm_37/zeros_1/packedPack,sequential_18/lstm_37/strided_slice:output:0/sequential_18/lstm_37/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_18/lstm_37/zeros_1/packed�
#sequential_18/lstm_37/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_18/lstm_37/zeros_1/Const�
sequential_18/lstm_37/zeros_1Fill-sequential_18/lstm_37/zeros_1/packed:output:0,sequential_18/lstm_37/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_18/lstm_37/zeros_1�
$sequential_18/lstm_37/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_18/lstm_37/transpose/perm�
sequential_18/lstm_37/transpose	Transpose%sequential_18/lstm_36/transpose_1:y:0-sequential_18/lstm_37/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2!
sequential_18/lstm_37/transpose�
sequential_18/lstm_37/Shape_1Shape#sequential_18/lstm_37/transpose:y:0*
T0*
_output_shapes
:2
sequential_18/lstm_37/Shape_1�
+sequential_18/lstm_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_18/lstm_37/strided_slice_1/stack�
-sequential_18/lstm_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_37/strided_slice_1/stack_1�
-sequential_18/lstm_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_37/strided_slice_1/stack_2�
%sequential_18/lstm_37/strided_slice_1StridedSlice&sequential_18/lstm_37/Shape_1:output:04sequential_18/lstm_37/strided_slice_1/stack:output:06sequential_18/lstm_37/strided_slice_1/stack_1:output:06sequential_18/lstm_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_18/lstm_37/strided_slice_1�
1sequential_18/lstm_37/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������23
1sequential_18/lstm_37/TensorArrayV2/element_shape�
#sequential_18/lstm_37/TensorArrayV2TensorListReserve:sequential_18/lstm_37/TensorArrayV2/element_shape:output:0.sequential_18/lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_18/lstm_37/TensorArrayV2�
Ksequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2M
Ksequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape�
=sequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_18/lstm_37/transpose:y:0Tsequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensor�
+sequential_18/lstm_37/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_18/lstm_37/strided_slice_2/stack�
-sequential_18/lstm_37/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_37/strided_slice_2/stack_1�
-sequential_18/lstm_37/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_37/strided_slice_2/stack_2�
%sequential_18/lstm_37/strided_slice_2StridedSlice#sequential_18/lstm_37/transpose:y:04sequential_18/lstm_37/strided_slice_2/stack:output:06sequential_18/lstm_37/strided_slice_2/stack_1:output:06sequential_18/lstm_37/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2'
%sequential_18/lstm_37/strided_slice_2�
8sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOpReadVariableOpAsequential_18_lstm_37_lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02:
8sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp�
)sequential_18/lstm_37/lstm_cell_73/MatMulMatMul.sequential_18/lstm_37/strided_slice_2:output:0@sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_18/lstm_37/lstm_cell_73/MatMul�
:sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOpCsequential_18_lstm_37_lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02<
:sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�
+sequential_18/lstm_37/lstm_cell_73/MatMul_1MatMul$sequential_18/lstm_37/zeros:output:0Bsequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2-
+sequential_18/lstm_37/lstm_cell_73/MatMul_1�
&sequential_18/lstm_37/lstm_cell_73/addAddV23sequential_18/lstm_37/lstm_cell_73/MatMul:product:05sequential_18/lstm_37/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2(
&sequential_18/lstm_37/lstm_cell_73/add�
9sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOpBsequential_18_lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02;
9sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�
*sequential_18/lstm_37/lstm_cell_73/BiasAddBiasAdd*sequential_18/lstm_37/lstm_cell_73/add:z:0Asequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_18/lstm_37/lstm_cell_73/BiasAdd�
2sequential_18/lstm_37/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_18/lstm_37/lstm_cell_73/split/split_dim�
(sequential_18/lstm_37/lstm_cell_73/splitSplit;sequential_18/lstm_37/lstm_cell_73/split/split_dim:output:03sequential_18/lstm_37/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2*
(sequential_18/lstm_37/lstm_cell_73/split�
*sequential_18/lstm_37/lstm_cell_73/SigmoidSigmoid1sequential_18/lstm_37/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2,
*sequential_18/lstm_37/lstm_cell_73/Sigmoid�
,sequential_18/lstm_37/lstm_cell_73/Sigmoid_1Sigmoid1sequential_18/lstm_37/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2.
,sequential_18/lstm_37/lstm_cell_73/Sigmoid_1�
&sequential_18/lstm_37/lstm_cell_73/mulMul0sequential_18/lstm_37/lstm_cell_73/Sigmoid_1:y:0&sequential_18/lstm_37/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2(
&sequential_18/lstm_37/lstm_cell_73/mul�
'sequential_18/lstm_37/lstm_cell_73/ReluRelu1sequential_18/lstm_37/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2)
'sequential_18/lstm_37/lstm_cell_73/Relu�
(sequential_18/lstm_37/lstm_cell_73/mul_1Mul.sequential_18/lstm_37/lstm_cell_73/Sigmoid:y:05sequential_18/lstm_37/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_18/lstm_37/lstm_cell_73/mul_1�
(sequential_18/lstm_37/lstm_cell_73/add_1AddV2*sequential_18/lstm_37/lstm_cell_73/mul:z:0,sequential_18/lstm_37/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2*
(sequential_18/lstm_37/lstm_cell_73/add_1�
,sequential_18/lstm_37/lstm_cell_73/Sigmoid_2Sigmoid1sequential_18/lstm_37/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2.
,sequential_18/lstm_37/lstm_cell_73/Sigmoid_2�
)sequential_18/lstm_37/lstm_cell_73/Relu_1Relu,sequential_18/lstm_37/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2+
)sequential_18/lstm_37/lstm_cell_73/Relu_1�
(sequential_18/lstm_37/lstm_cell_73/mul_2Mul0sequential_18/lstm_37/lstm_cell_73/Sigmoid_2:y:07sequential_18/lstm_37/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2*
(sequential_18/lstm_37/lstm_cell_73/mul_2�
3sequential_18/lstm_37/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    25
3sequential_18/lstm_37/TensorArrayV2_1/element_shape�
%sequential_18/lstm_37/TensorArrayV2_1TensorListReserve<sequential_18/lstm_37/TensorArrayV2_1/element_shape:output:0.sequential_18/lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_18/lstm_37/TensorArrayV2_1z
sequential_18/lstm_37/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_18/lstm_37/time�
.sequential_18/lstm_37/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������20
.sequential_18/lstm_37/while/maximum_iterations�
(sequential_18/lstm_37/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_18/lstm_37/while/loop_counter�
sequential_18/lstm_37/whileWhile1sequential_18/lstm_37/while/loop_counter:output:07sequential_18/lstm_37/while/maximum_iterations:output:0#sequential_18/lstm_37/time:output:0.sequential_18/lstm_37/TensorArrayV2_1:handle:0$sequential_18/lstm_37/zeros:output:0&sequential_18/lstm_37/zeros_1:output:0.sequential_18/lstm_37/strided_slice_1:output:0Msequential_18/lstm_37/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_18_lstm_37_lstm_cell_73_matmul_readvariableop_resourceCsequential_18_lstm_37_lstm_cell_73_matmul_1_readvariableop_resourceBsequential_18_lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
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
'sequential_18_lstm_37_while_body_583652*3
cond+R)
'sequential_18_lstm_37_while_cond_583651*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
sequential_18/lstm_37/while�
Fsequential_18/lstm_37/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2H
Fsequential_18/lstm_37/TensorArrayV2Stack/TensorListStack/element_shape�
8sequential_18/lstm_37/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_18/lstm_37/while:output:3Osequential_18/lstm_37/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02:
8sequential_18/lstm_37/TensorArrayV2Stack/TensorListStack�
+sequential_18/lstm_37/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2-
+sequential_18/lstm_37/strided_slice_3/stack�
-sequential_18/lstm_37/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_18/lstm_37/strided_slice_3/stack_1�
-sequential_18/lstm_37/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_18/lstm_37/strided_slice_3/stack_2�
%sequential_18/lstm_37/strided_slice_3StridedSliceAsequential_18/lstm_37/TensorArrayV2Stack/TensorListStack:tensor:04sequential_18/lstm_37/strided_slice_3/stack:output:06sequential_18/lstm_37/strided_slice_3/stack_1:output:06sequential_18/lstm_37/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2'
%sequential_18/lstm_37/strided_slice_3�
&sequential_18/lstm_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_18/lstm_37/transpose_1/perm�
!sequential_18/lstm_37/transpose_1	TransposeAsequential_18/lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_18/lstm_37/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2#
!sequential_18/lstm_37/transpose_1�
sequential_18/lstm_37/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_18/lstm_37/runtime�
!sequential_18/dropout_18/IdentityIdentity.sequential_18/lstm_37/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2#
!sequential_18/dropout_18/Identity�
,sequential_18/dense_18/MatMul/ReadVariableOpReadVariableOp5sequential_18_dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_18/dense_18/MatMul/ReadVariableOp�
sequential_18/dense_18/MatMulMatMul*sequential_18/dropout_18/Identity:output:04sequential_18/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_18/dense_18/MatMul�
-sequential_18/dense_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_18/BiasAdd/ReadVariableOp�
sequential_18/dense_18/BiasAddBiasAdd'sequential_18/dense_18/MatMul:product:05sequential_18/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
sequential_18/dense_18/BiasAdd�
IdentityIdentity'sequential_18/dense_18/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp.^sequential_18/dense_18/BiasAdd/ReadVariableOp-^sequential_18/dense_18/MatMul/ReadVariableOp:^sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp9^sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp;^sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp^sequential_18/lstm_36/while:^sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp9^sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp;^sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp^sequential_18/lstm_37/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2^
-sequential_18/dense_18/BiasAdd/ReadVariableOp-sequential_18/dense_18/BiasAdd/ReadVariableOp2\
,sequential_18/dense_18/MatMul/ReadVariableOp,sequential_18/dense_18/MatMul/ReadVariableOp2v
9sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp9sequential_18/lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp2t
8sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp8sequential_18/lstm_36/lstm_cell_72/MatMul/ReadVariableOp2x
:sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp:sequential_18/lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp2:
sequential_18/lstm_36/whilesequential_18/lstm_36/while2v
9sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp9sequential_18/lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp2t
8sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp8sequential_18/lstm_37/lstm_cell_73/MatMul/ReadVariableOp2x
:sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp:sequential_18/lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp2:
sequential_18/lstm_37/whilesequential_18/lstm_37/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_36_input
�?
�
while_body_587135
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
��
�
"__inference__traced_restore_588328
file_prefix2
 assignvariableop_dense_18_kernel: .
 assignvariableop_1_dense_18_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_36_lstm_cell_72_kernel:	�K
8assignvariableop_8_lstm_36_lstm_cell_72_recurrent_kernel:	@�;
,assignvariableop_9_lstm_36_lstm_cell_72_bias:	�B
/assignvariableop_10_lstm_37_lstm_cell_73_kernel:	@�L
9assignvariableop_11_lstm_37_lstm_cell_73_recurrent_kernel:	 �<
-assignvariableop_12_lstm_37_lstm_cell_73_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: <
*assignvariableop_15_adam_dense_18_kernel_m: 6
(assignvariableop_16_adam_dense_18_bias_m:I
6assignvariableop_17_adam_lstm_36_lstm_cell_72_kernel_m:	�S
@assignvariableop_18_adam_lstm_36_lstm_cell_72_recurrent_kernel_m:	@�C
4assignvariableop_19_adam_lstm_36_lstm_cell_72_bias_m:	�I
6assignvariableop_20_adam_lstm_37_lstm_cell_73_kernel_m:	@�S
@assignvariableop_21_adam_lstm_37_lstm_cell_73_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_lstm_37_lstm_cell_73_bias_m:	�<
*assignvariableop_23_adam_dense_18_kernel_v: 6
(assignvariableop_24_adam_dense_18_bias_v:I
6assignvariableop_25_adam_lstm_36_lstm_cell_72_kernel_v:	�S
@assignvariableop_26_adam_lstm_36_lstm_cell_72_recurrent_kernel_v:	@�C
4assignvariableop_27_adam_lstm_36_lstm_cell_72_bias_v:	�I
6assignvariableop_28_adam_lstm_37_lstm_cell_73_kernel_v:	@�S
@assignvariableop_29_adam_lstm_37_lstm_cell_73_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_lstm_37_lstm_cell_73_bias_v:	�
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
AssignVariableOpAssignVariableOp assignvariableop_dense_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_18_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_36_lstm_cell_72_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_36_lstm_cell_72_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_36_lstm_cell_72_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_37_lstm_cell_73_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_37_lstm_cell_73_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_37_lstm_cell_73_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_18_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_18_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_36_lstm_cell_72_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_36_lstm_cell_72_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_36_lstm_cell_72_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_37_lstm_cell_73_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_37_lstm_cell_73_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_37_lstm_cell_73_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_18_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_18_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_36_lstm_cell_72_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_36_lstm_cell_72_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_36_lstm_cell_72_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_37_lstm_cell_73_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_37_lstm_cell_73_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_37_lstm_cell_73_bias_vIdentity_30:output:0"/device:CPU:0*
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
�?
�
while_body_585655
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
while_body_585076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
�
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_585331

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

�
.__inference_sequential_18_layer_call_fn_585954

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
I__inference_sequential_18_layer_call_and_return_conditional_losses_5857952
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
�\
�
C__inference_lstm_37_layer_call_and_return_conditional_losses_587414
inputs_0>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_587330*
condR
while_cond_587329*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
G
+__inference_dropout_18_layer_call_fn_587872

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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853312
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
�
�
while_cond_587480
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_587480___redundant_placeholder04
0while_while_cond_587480___redundant_placeholder14
0while_while_cond_587480___redundant_placeholder24
0while_while_cond_587480___redundant_placeholder3
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_585160

inputs>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_585076*
condR
while_cond_585075*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
.__inference_sequential_18_layer_call_fn_585369
lstm_36_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
I__inference_sequential_18_layer_call_and_return_conditional_losses_5853502
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
_user_specified_namelstm_36_input
�
�
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588109

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
�?
�
while_body_585482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
while_cond_587134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_587134___redundant_placeholder04
0while_while_cond_587134___redundant_placeholder14
0while_while_cond_587134___redundant_placeholder24
0while_while_cond_587134___redundant_placeholder3
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
while_body_586984
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
'sequential_18_lstm_36_while_body_583505H
Dsequential_18_lstm_36_while_sequential_18_lstm_36_while_loop_counterN
Jsequential_18_lstm_36_while_sequential_18_lstm_36_while_maximum_iterations+
'sequential_18_lstm_36_while_placeholder-
)sequential_18_lstm_36_while_placeholder_1-
)sequential_18_lstm_36_while_placeholder_2-
)sequential_18_lstm_36_while_placeholder_3G
Csequential_18_lstm_36_while_sequential_18_lstm_36_strided_slice_1_0�
sequential_18_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_36_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_18_lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0:	�^
Ksequential_18_lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�Y
Jsequential_18_lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0:	�(
$sequential_18_lstm_36_while_identity*
&sequential_18_lstm_36_while_identity_1*
&sequential_18_lstm_36_while_identity_2*
&sequential_18_lstm_36_while_identity_3*
&sequential_18_lstm_36_while_identity_4*
&sequential_18_lstm_36_while_identity_5E
Asequential_18_lstm_36_while_sequential_18_lstm_36_strided_slice_1�
}sequential_18_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_36_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_18_lstm_36_while_lstm_cell_72_matmul_readvariableop_resource:	�\
Isequential_18_lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource:	@�W
Hsequential_18_lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource:	���?sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�>sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�@sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
Msequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2O
Msequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_18_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_36_tensorarrayunstack_tensorlistfromtensor_0'sequential_18_lstm_36_while_placeholderVsequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02A
?sequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItem�
>sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOpIsequential_18_lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02@
>sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�
/sequential_18/lstm_36/while/lstm_cell_72/MatMulMatMulFsequential_18/lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_18/lstm_36/while/lstm_cell_72/MatMul�
@sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOpKsequential_18_lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02B
@sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
1sequential_18/lstm_36/while/lstm_cell_72/MatMul_1MatMul)sequential_18_lstm_36_while_placeholder_2Hsequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_18/lstm_36/while/lstm_cell_72/MatMul_1�
,sequential_18/lstm_36/while/lstm_cell_72/addAddV29sequential_18/lstm_36/while/lstm_cell_72/MatMul:product:0;sequential_18/lstm_36/while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_18/lstm_36/while/lstm_cell_72/add�
?sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOpJsequential_18_lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�
0sequential_18/lstm_36/while/lstm_cell_72/BiasAddBiasAdd0sequential_18/lstm_36/while/lstm_cell_72/add:z:0Gsequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_18/lstm_36/while/lstm_cell_72/BiasAdd�
8sequential_18/lstm_36/while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_18/lstm_36/while/lstm_cell_72/split/split_dim�
.sequential_18/lstm_36/while/lstm_cell_72/splitSplitAsequential_18/lstm_36/while/lstm_cell_72/split/split_dim:output:09sequential_18/lstm_36/while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split20
.sequential_18/lstm_36/while/lstm_cell_72/split�
0sequential_18/lstm_36/while/lstm_cell_72/SigmoidSigmoid7sequential_18/lstm_36/while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@22
0sequential_18/lstm_36/while/lstm_cell_72/Sigmoid�
2sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_1Sigmoid7sequential_18/lstm_36/while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@24
2sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_1�
,sequential_18/lstm_36/while/lstm_cell_72/mulMul6sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_1:y:0)sequential_18_lstm_36_while_placeholder_3*
T0*'
_output_shapes
:���������@2.
,sequential_18/lstm_36/while/lstm_cell_72/mul�
-sequential_18/lstm_36/while/lstm_cell_72/ReluRelu7sequential_18/lstm_36/while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2/
-sequential_18/lstm_36/while/lstm_cell_72/Relu�
.sequential_18/lstm_36/while/lstm_cell_72/mul_1Mul4sequential_18/lstm_36/while/lstm_cell_72/Sigmoid:y:0;sequential_18/lstm_36/while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_18/lstm_36/while/lstm_cell_72/mul_1�
.sequential_18/lstm_36/while/lstm_cell_72/add_1AddV20sequential_18/lstm_36/while/lstm_cell_72/mul:z:02sequential_18/lstm_36/while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@20
.sequential_18/lstm_36/while/lstm_cell_72/add_1�
2sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_2Sigmoid7sequential_18/lstm_36/while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@24
2sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_2�
/sequential_18/lstm_36/while/lstm_cell_72/Relu_1Relu2sequential_18/lstm_36/while/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@21
/sequential_18/lstm_36/while/lstm_cell_72/Relu_1�
.sequential_18/lstm_36/while/lstm_cell_72/mul_2Mul6sequential_18/lstm_36/while/lstm_cell_72/Sigmoid_2:y:0=sequential_18/lstm_36/while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@20
.sequential_18/lstm_36/while/lstm_cell_72/mul_2�
@sequential_18/lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_18_lstm_36_while_placeholder_1'sequential_18_lstm_36_while_placeholder2sequential_18/lstm_36/while/lstm_cell_72/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_18/lstm_36/while/TensorArrayV2Write/TensorListSetItem�
!sequential_18/lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_18/lstm_36/while/add/y�
sequential_18/lstm_36/while/addAddV2'sequential_18_lstm_36_while_placeholder*sequential_18/lstm_36/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_18/lstm_36/while/add�
#sequential_18/lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_18/lstm_36/while/add_1/y�
!sequential_18/lstm_36/while/add_1AddV2Dsequential_18_lstm_36_while_sequential_18_lstm_36_while_loop_counter,sequential_18/lstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_18/lstm_36/while/add_1�
$sequential_18/lstm_36/while/IdentityIdentity%sequential_18/lstm_36/while/add_1:z:0!^sequential_18/lstm_36/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_18/lstm_36/while/Identity�
&sequential_18/lstm_36/while/Identity_1IdentityJsequential_18_lstm_36_while_sequential_18_lstm_36_while_maximum_iterations!^sequential_18/lstm_36/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_36/while/Identity_1�
&sequential_18/lstm_36/while/Identity_2Identity#sequential_18/lstm_36/while/add:z:0!^sequential_18/lstm_36/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_36/while/Identity_2�
&sequential_18/lstm_36/while/Identity_3IdentityPsequential_18/lstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_18/lstm_36/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_36/while/Identity_3�
&sequential_18/lstm_36/while/Identity_4Identity2sequential_18/lstm_36/while/lstm_cell_72/mul_2:z:0!^sequential_18/lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_18/lstm_36/while/Identity_4�
&sequential_18/lstm_36/while/Identity_5Identity2sequential_18/lstm_36/while/lstm_cell_72/add_1:z:0!^sequential_18/lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2(
&sequential_18/lstm_36/while/Identity_5�
 sequential_18/lstm_36/while/NoOpNoOp@^sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp?^sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOpA^sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_18/lstm_36/while/NoOp"U
$sequential_18_lstm_36_while_identity-sequential_18/lstm_36/while/Identity:output:0"Y
&sequential_18_lstm_36_while_identity_1/sequential_18/lstm_36/while/Identity_1:output:0"Y
&sequential_18_lstm_36_while_identity_2/sequential_18/lstm_36/while/Identity_2:output:0"Y
&sequential_18_lstm_36_while_identity_3/sequential_18/lstm_36/while/Identity_3:output:0"Y
&sequential_18_lstm_36_while_identity_4/sequential_18/lstm_36/while/Identity_4:output:0"Y
&sequential_18_lstm_36_while_identity_5/sequential_18/lstm_36/while/Identity_5:output:0"�
Hsequential_18_lstm_36_while_lstm_cell_72_biasadd_readvariableop_resourceJsequential_18_lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0"�
Isequential_18_lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resourceKsequential_18_lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0"�
Gsequential_18_lstm_36_while_lstm_cell_72_matmul_readvariableop_resourceIsequential_18_lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0"�
Asequential_18_lstm_36_while_sequential_18_lstm_36_strided_slice_1Csequential_18_lstm_36_while_sequential_18_lstm_36_strided_slice_1_0"�
}sequential_18_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_36_tensorarrayunstack_tensorlistfromtensorsequential_18_lstm_36_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2�
?sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp?sequential_18/lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp2�
>sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp>sequential_18/lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp2�
@sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp@sequential_18/lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
while_body_584042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_72_584066_0:	�.
while_lstm_cell_72_584068_0:	@�*
while_lstm_cell_72_584070_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_72_584066:	�,
while_lstm_cell_72_584068:	@�(
while_lstm_cell_72_584070:	���*while/lstm_cell_72/StatefulPartitionedCall�
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
*while/lstm_cell_72/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_72_584066_0while_lstm_cell_72_584068_0while_lstm_cell_72_584070_0*
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5839642,
*while/lstm_cell_72/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_72/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_72/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_72/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_72/StatefulPartitionedCall*"
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
while_lstm_cell_72_584066while_lstm_cell_72_584066_0"8
while_lstm_cell_72_584068while_lstm_cell_72_584068_0"8
while_lstm_cell_72_584070while_lstm_cell_72_584070_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_72/StatefulPartitionedCall*while/lstm_cell_72/StatefulPartitionedCall: 
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
-__inference_lstm_cell_72_layer_call_fn_587947

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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5839642
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
�
while_cond_586832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_586832___redundant_placeholder04
0while_while_cond_586832___redundant_placeholder14
0while_while_cond_586832___redundant_placeholder24
0while_while_cond_586832___redundant_placeholder3
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
(__inference_lstm_37_layer_call_fn_587241
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5847412
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
�
d
+__inference_dropout_18_layer_call_fn_587877

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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853992
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
'sequential_18_lstm_37_while_cond_583651H
Dsequential_18_lstm_37_while_sequential_18_lstm_37_while_loop_counterN
Jsequential_18_lstm_37_while_sequential_18_lstm_37_while_maximum_iterations+
'sequential_18_lstm_37_while_placeholder-
)sequential_18_lstm_37_while_placeholder_1-
)sequential_18_lstm_37_while_placeholder_2-
)sequential_18_lstm_37_while_placeholder_3J
Fsequential_18_lstm_37_while_less_sequential_18_lstm_37_strided_slice_1`
\sequential_18_lstm_37_while_sequential_18_lstm_37_while_cond_583651___redundant_placeholder0`
\sequential_18_lstm_37_while_sequential_18_lstm_37_while_cond_583651___redundant_placeholder1`
\sequential_18_lstm_37_while_sequential_18_lstm_37_while_cond_583651___redundant_placeholder2`
\sequential_18_lstm_37_while_sequential_18_lstm_37_while_cond_583651___redundant_placeholder3(
$sequential_18_lstm_37_while_identity
�
 sequential_18/lstm_37/while/LessLess'sequential_18_lstm_37_while_placeholderFsequential_18_lstm_37_while_less_sequential_18_lstm_37_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_18/lstm_37/while/Less�
$sequential_18/lstm_37/while/IdentityIdentity$sequential_18/lstm_37/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_18/lstm_37/while/Identity"U
$sequential_18_lstm_37_while_identity-sequential_18/lstm_37/while/Identity:output:0*(
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
�
'sequential_18_lstm_36_while_cond_583504H
Dsequential_18_lstm_36_while_sequential_18_lstm_36_while_loop_counterN
Jsequential_18_lstm_36_while_sequential_18_lstm_36_while_maximum_iterations+
'sequential_18_lstm_36_while_placeholder-
)sequential_18_lstm_36_while_placeholder_1-
)sequential_18_lstm_36_while_placeholder_2-
)sequential_18_lstm_36_while_placeholder_3J
Fsequential_18_lstm_36_while_less_sequential_18_lstm_36_strided_slice_1`
\sequential_18_lstm_36_while_sequential_18_lstm_36_while_cond_583504___redundant_placeholder0`
\sequential_18_lstm_36_while_sequential_18_lstm_36_while_cond_583504___redundant_placeholder1`
\sequential_18_lstm_36_while_sequential_18_lstm_36_while_cond_583504___redundant_placeholder2`
\sequential_18_lstm_36_while_sequential_18_lstm_36_while_cond_583504___redundant_placeholder3(
$sequential_18_lstm_36_while_identity
�
 sequential_18/lstm_36/while/LessLess'sequential_18_lstm_36_while_placeholderFsequential_18_lstm_36_while_less_sequential_18_lstm_36_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_18/lstm_36/while/Less�
$sequential_18/lstm_36/while/IdentityIdentity$sequential_18/lstm_36/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_18/lstm_36/while/Identity"U
$sequential_18_lstm_36_while_identity-sequential_18/lstm_36/while/Identity:output:0*(
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
(__inference_lstm_36_layer_call_fn_586604

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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5851602
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

�
lstm_37_while_cond_586472,
(lstm_37_while_lstm_37_while_loop_counter2
.lstm_37_while_lstm_37_while_maximum_iterations
lstm_37_while_placeholder
lstm_37_while_placeholder_1
lstm_37_while_placeholder_2
lstm_37_while_placeholder_3.
*lstm_37_while_less_lstm_37_strided_slice_1D
@lstm_37_while_lstm_37_while_cond_586472___redundant_placeholder0D
@lstm_37_while_lstm_37_while_cond_586472___redundant_placeholder1D
@lstm_37_while_lstm_37_while_cond_586472___redundant_placeholder2D
@lstm_37_while_lstm_37_while_cond_586472___redundant_placeholder3
lstm_37_while_identity
�
lstm_37/while/LessLesslstm_37_while_placeholder*lstm_37_while_less_lstm_37_strided_slice_1*
T0*
_output_shapes
: 2
lstm_37/while/Lessu
lstm_37/while/IdentityIdentitylstm_37/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_37/while/Identity"9
lstm_37_while_identitylstm_37/while/Identity:output:0*(
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
�
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_585350

inputs!
lstm_36_585161:	�!
lstm_36_585163:	@�
lstm_36_585165:	�!
lstm_37_585319:	@�!
lstm_37_585321:	 �
lstm_37_585323:	�!
dense_18_585344: 
dense_18_585346:
identity�� dense_18/StatefulPartitionedCall�lstm_36/StatefulPartitionedCall�lstm_37/StatefulPartitionedCall�
lstm_36/StatefulPartitionedCallStatefulPartitionedCallinputslstm_36_585161lstm_36_585163lstm_36_585165*
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5851602!
lstm_36/StatefulPartitionedCall�
lstm_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0lstm_37_585319lstm_37_585321lstm_37_585323*
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5853182!
lstm_37/StatefulPartitionedCall�
dropout_18/PartitionedCallPartitionedCall(lstm_37/StatefulPartitionedCall:output:0*
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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853312
dropout_18/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_18_585344dense_18_585346*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_5853432"
 dense_18/StatefulPartitionedCall�
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_18/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
while_cond_585654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_585654___redundant_placeholder04
0while_while_cond_585654___redundant_placeholder14
0while_while_cond_585654___redundant_placeholder24
0while_while_cond_585654___redundant_placeholder3
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
while_cond_585481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_585481___redundant_placeholder04
0while_while_cond_585481___redundant_placeholder14
0while_while_cond_585481___redundant_placeholder24
0while_while_cond_585481___redundant_placeholder3
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_583818

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
�?
�
while_body_587481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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

lstm_37_while_body_586168,
(lstm_37_while_lstm_37_while_loop_counter2
.lstm_37_while_lstm_37_while_maximum_iterations
lstm_37_while_placeholder
lstm_37_while_placeholder_1
lstm_37_while_placeholder_2
lstm_37_while_placeholder_3+
'lstm_37_while_lstm_37_strided_slice_1_0g
clstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0:	@�P
=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �K
<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
lstm_37_while_identity
lstm_37_while_identity_1
lstm_37_while_identity_2
lstm_37_while_identity_3
lstm_37_while_identity_4
lstm_37_while_identity_5)
%lstm_37_while_lstm_37_strided_slice_1e
alstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensorL
9lstm_37_while_lstm_cell_73_matmul_readvariableop_resource:	@�N
;lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource:	 �I
:lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource:	���1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
?lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_37/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0lstm_37_while_placeholderHlstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_37/while/TensorArrayV2Read/TensorListGetItem�
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�
!lstm_37/while/lstm_cell_73/MatMulMatMul8lstm_37/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_37/while/lstm_cell_73/MatMul�
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
#lstm_37/while/lstm_cell_73/MatMul_1MatMullstm_37_while_placeholder_2:lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_37/while/lstm_cell_73/MatMul_1�
lstm_37/while/lstm_cell_73/addAddV2+lstm_37/while/lstm_cell_73/MatMul:product:0-lstm_37/while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_37/while/lstm_cell_73/add�
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�
"lstm_37/while/lstm_cell_73/BiasAddBiasAdd"lstm_37/while/lstm_cell_73/add:z:09lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_37/while/lstm_cell_73/BiasAdd�
*lstm_37/while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_37/while/lstm_cell_73/split/split_dim�
 lstm_37/while/lstm_cell_73/splitSplit3lstm_37/while/lstm_cell_73/split/split_dim:output:0+lstm_37/while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_37/while/lstm_cell_73/split�
"lstm_37/while/lstm_cell_73/SigmoidSigmoid)lstm_37/while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_37/while/lstm_cell_73/Sigmoid�
$lstm_37/while/lstm_cell_73/Sigmoid_1Sigmoid)lstm_37/while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_37/while/lstm_cell_73/Sigmoid_1�
lstm_37/while/lstm_cell_73/mulMul(lstm_37/while/lstm_cell_73/Sigmoid_1:y:0lstm_37_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_37/while/lstm_cell_73/mul�
lstm_37/while/lstm_cell_73/ReluRelu)lstm_37/while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_37/while/lstm_cell_73/Relu�
 lstm_37/while/lstm_cell_73/mul_1Mul&lstm_37/while/lstm_cell_73/Sigmoid:y:0-lstm_37/while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/mul_1�
 lstm_37/while/lstm_cell_73/add_1AddV2"lstm_37/while/lstm_cell_73/mul:z:0$lstm_37/while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/add_1�
$lstm_37/while/lstm_cell_73/Sigmoid_2Sigmoid)lstm_37/while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_37/while/lstm_cell_73/Sigmoid_2�
!lstm_37/while/lstm_cell_73/Relu_1Relu$lstm_37/while/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_37/while/lstm_cell_73/Relu_1�
 lstm_37/while/lstm_cell_73/mul_2Mul(lstm_37/while/lstm_cell_73/Sigmoid_2:y:0/lstm_37/while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/mul_2�
2lstm_37/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_37_while_placeholder_1lstm_37_while_placeholder$lstm_37/while/lstm_cell_73/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_37/while/TensorArrayV2Write/TensorListSetIteml
lstm_37/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_37/while/add/y�
lstm_37/while/addAddV2lstm_37_while_placeholderlstm_37/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_37/while/addp
lstm_37/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_37/while/add_1/y�
lstm_37/while/add_1AddV2(lstm_37_while_lstm_37_while_loop_counterlstm_37/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_37/while/add_1�
lstm_37/while/IdentityIdentitylstm_37/while/add_1:z:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity�
lstm_37/while/Identity_1Identity.lstm_37_while_lstm_37_while_maximum_iterations^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_1�
lstm_37/while/Identity_2Identitylstm_37/while/add:z:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_2�
lstm_37/while/Identity_3IdentityBlstm_37/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_3�
lstm_37/while/Identity_4Identity$lstm_37/while/lstm_cell_73/mul_2:z:0^lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_37/while/Identity_4�
lstm_37/while/Identity_5Identity$lstm_37/while/lstm_cell_73/add_1:z:0^lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_37/while/Identity_5�
lstm_37/while/NoOpNoOp2^lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp1^lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp3^lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_37/while/NoOp"9
lstm_37_while_identitylstm_37/while/Identity:output:0"=
lstm_37_while_identity_1!lstm_37/while/Identity_1:output:0"=
lstm_37_while_identity_2!lstm_37/while/Identity_2:output:0"=
lstm_37_while_identity_3!lstm_37/while/Identity_3:output:0"=
lstm_37_while_identity_4!lstm_37/while/Identity_4:output:0"=
lstm_37_while_identity_5!lstm_37/while/Identity_5:output:0"P
%lstm_37_while_lstm_37_strided_slice_1'lstm_37_while_lstm_37_strided_slice_1_0"z
:lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0"|
;lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0"x
9lstm_37_while_lstm_cell_73_matmul_readvariableop_resource;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0"�
alstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensorclstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp2d
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp2h
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587867

inputs>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_587783*
condR
while_cond_587782*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�\
�
C__inference_lstm_36_layer_call_and_return_conditional_losses_586917
inputs_0>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_586833*
condR
while_cond_586832*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�F
�
C__inference_lstm_37_layer_call_and_return_conditional_losses_584741

inputs&
lstm_cell_73_584659:	@�&
lstm_cell_73_584661:	 �"
lstm_cell_73_584663:	�
identity��$lstm_cell_73/StatefulPartitionedCall�whileD
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
$lstm_cell_73/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_73_584659lstm_cell_73_584661lstm_cell_73_584663*
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5845942&
$lstm_cell_73/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_73_584659lstm_cell_73_584661lstm_cell_73_584663*
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
while_body_584672*
condR
while_cond_584671*K
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
NoOpNoOp%^lstm_cell_73/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_73/StatefulPartitionedCall$lstm_cell_73/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
(__inference_lstm_37_layer_call_fn_587263

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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5855662
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_584594

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
�
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_587882

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
��
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_586571

inputsF
3lstm_36_lstm_cell_72_matmul_readvariableop_resource:	�H
5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource:	@�C
4lstm_36_lstm_cell_72_biasadd_readvariableop_resource:	�F
3lstm_37_lstm_cell_73_matmul_readvariableop_resource:	@�H
5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource:	 �C
4lstm_37_lstm_cell_73_biasadd_readvariableop_resource:	�9
'dense_18_matmul_readvariableop_resource: 6
(dense_18_biasadd_readvariableop_resource:
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�*lstm_36/lstm_cell_72/MatMul/ReadVariableOp�,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�lstm_36/while�+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�*lstm_37/lstm_cell_73/MatMul/ReadVariableOp�,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�lstm_37/whileT
lstm_36/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_36/Shape�
lstm_36/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice/stack�
lstm_36/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_36/strided_slice/stack_1�
lstm_36/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_36/strided_slice/stack_2�
lstm_36/strided_sliceStridedSlicelstm_36/Shape:output:0$lstm_36/strided_slice/stack:output:0&lstm_36/strided_slice/stack_1:output:0&lstm_36/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_36/strided_slicel
lstm_36/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros/mul/y�
lstm_36/zeros/mulMullstm_36/strided_slice:output:0lstm_36/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros/mulo
lstm_36/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_36/zeros/Less/y�
lstm_36/zeros/LessLesslstm_36/zeros/mul:z:0lstm_36/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros/Lessr
lstm_36/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros/packed/1�
lstm_36/zeros/packedPacklstm_36/strided_slice:output:0lstm_36/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_36/zeros/packedo
lstm_36/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/zeros/Const�
lstm_36/zerosFilllstm_36/zeros/packed:output:0lstm_36/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/zerosp
lstm_36/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros_1/mul/y�
lstm_36/zeros_1/mulMullstm_36/strided_slice:output:0lstm_36/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros_1/muls
lstm_36/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_36/zeros_1/Less/y�
lstm_36/zeros_1/LessLesslstm_36/zeros_1/mul:z:0lstm_36/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_36/zeros_1/Lessv
lstm_36/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_36/zeros_1/packed/1�
lstm_36/zeros_1/packedPacklstm_36/strided_slice:output:0!lstm_36/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_36/zeros_1/packeds
lstm_36/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/zeros_1/Const�
lstm_36/zeros_1Filllstm_36/zeros_1/packed:output:0lstm_36/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/zeros_1�
lstm_36/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_36/transpose/perm�
lstm_36/transpose	Transposeinputslstm_36/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_36/transposeg
lstm_36/Shape_1Shapelstm_36/transpose:y:0*
T0*
_output_shapes
:2
lstm_36/Shape_1�
lstm_36/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice_1/stack�
lstm_36/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_1/stack_1�
lstm_36/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_1/stack_2�
lstm_36/strided_slice_1StridedSlicelstm_36/Shape_1:output:0&lstm_36/strided_slice_1/stack:output:0(lstm_36/strided_slice_1/stack_1:output:0(lstm_36/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_36/strided_slice_1�
#lstm_36/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_36/TensorArrayV2/element_shape�
lstm_36/TensorArrayV2TensorListReserve,lstm_36/TensorArrayV2/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_36/TensorArrayV2�
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_36/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_36/transpose:y:0Flstm_36/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_36/TensorArrayUnstack/TensorListFromTensor�
lstm_36/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_36/strided_slice_2/stack�
lstm_36/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_2/stack_1�
lstm_36/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_2/stack_2�
lstm_36/strided_slice_2StridedSlicelstm_36/transpose:y:0&lstm_36/strided_slice_2/stack:output:0(lstm_36/strided_slice_2/stack_1:output:0(lstm_36/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_36/strided_slice_2�
*lstm_36/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3lstm_36_lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_36/lstm_cell_72/MatMul/ReadVariableOp�
lstm_36/lstm_cell_72/MatMulMatMul lstm_36/strided_slice_2:output:02lstm_36/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/MatMul�
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_36/lstm_cell_72/MatMul_1MatMullstm_36/zeros:output:04lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/MatMul_1�
lstm_36/lstm_cell_72/addAddV2%lstm_36/lstm_cell_72/MatMul:product:0'lstm_36/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/add�
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_36/lstm_cell_72/BiasAddBiasAddlstm_36/lstm_cell_72/add:z:03lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_36/lstm_cell_72/BiasAdd�
$lstm_36/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_36/lstm_cell_72/split/split_dim�
lstm_36/lstm_cell_72/splitSplit-lstm_36/lstm_cell_72/split/split_dim:output:0%lstm_36/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_36/lstm_cell_72/split�
lstm_36/lstm_cell_72/SigmoidSigmoid#lstm_36/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Sigmoid�
lstm_36/lstm_cell_72/Sigmoid_1Sigmoid#lstm_36/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_36/lstm_cell_72/Sigmoid_1�
lstm_36/lstm_cell_72/mulMul"lstm_36/lstm_cell_72/Sigmoid_1:y:0lstm_36/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul�
lstm_36/lstm_cell_72/ReluRelu#lstm_36/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Relu�
lstm_36/lstm_cell_72/mul_1Mul lstm_36/lstm_cell_72/Sigmoid:y:0'lstm_36/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul_1�
lstm_36/lstm_cell_72/add_1AddV2lstm_36/lstm_cell_72/mul:z:0lstm_36/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/add_1�
lstm_36/lstm_cell_72/Sigmoid_2Sigmoid#lstm_36/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_36/lstm_cell_72/Sigmoid_2�
lstm_36/lstm_cell_72/Relu_1Relulstm_36/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/Relu_1�
lstm_36/lstm_cell_72/mul_2Mul"lstm_36/lstm_cell_72/Sigmoid_2:y:0)lstm_36/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_36/lstm_cell_72/mul_2�
%lstm_36/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_36/TensorArrayV2_1/element_shape�
lstm_36/TensorArrayV2_1TensorListReserve.lstm_36/TensorArrayV2_1/element_shape:output:0 lstm_36/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_36/TensorArrayV2_1^
lstm_36/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_36/time�
 lstm_36/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_36/while/maximum_iterationsz
lstm_36/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_36/while/loop_counter�
lstm_36/whileWhile#lstm_36/while/loop_counter:output:0)lstm_36/while/maximum_iterations:output:0lstm_36/time:output:0 lstm_36/TensorArrayV2_1:handle:0lstm_36/zeros:output:0lstm_36/zeros_1:output:0 lstm_36/strided_slice_1:output:0?lstm_36/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_36_lstm_cell_72_matmul_readvariableop_resource5lstm_36_lstm_cell_72_matmul_1_readvariableop_resource4lstm_36_lstm_cell_72_biasadd_readvariableop_resource*
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
lstm_36_while_body_586326*%
condR
lstm_36_while_cond_586325*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_36/while�
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_36/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_36/TensorArrayV2Stack/TensorListStackTensorListStacklstm_36/while:output:3Alstm_36/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_36/TensorArrayV2Stack/TensorListStack�
lstm_36/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_36/strided_slice_3/stack�
lstm_36/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_36/strided_slice_3/stack_1�
lstm_36/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_36/strided_slice_3/stack_2�
lstm_36/strided_slice_3StridedSlice3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_36/strided_slice_3/stack:output:0(lstm_36/strided_slice_3/stack_1:output:0(lstm_36/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_36/strided_slice_3�
lstm_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_36/transpose_1/perm�
lstm_36/transpose_1	Transpose3lstm_36/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_36/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_36/transpose_1v
lstm_36/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_36/runtimee
lstm_37/ShapeShapelstm_36/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_37/Shape�
lstm_37/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice/stack�
lstm_37/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_37/strided_slice/stack_1�
lstm_37/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_37/strided_slice/stack_2�
lstm_37/strided_sliceStridedSlicelstm_37/Shape:output:0$lstm_37/strided_slice/stack:output:0&lstm_37/strided_slice/stack_1:output:0&lstm_37/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_37/strided_slicel
lstm_37/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros/mul/y�
lstm_37/zeros/mulMullstm_37/strided_slice:output:0lstm_37/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros/mulo
lstm_37/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_37/zeros/Less/y�
lstm_37/zeros/LessLesslstm_37/zeros/mul:z:0lstm_37/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros/Lessr
lstm_37/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros/packed/1�
lstm_37/zeros/packedPacklstm_37/strided_slice:output:0lstm_37/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_37/zeros/packedo
lstm_37/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/zeros/Const�
lstm_37/zerosFilllstm_37/zeros/packed:output:0lstm_37/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/zerosp
lstm_37/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros_1/mul/y�
lstm_37/zeros_1/mulMullstm_37/strided_slice:output:0lstm_37/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros_1/muls
lstm_37/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_37/zeros_1/Less/y�
lstm_37/zeros_1/LessLesslstm_37/zeros_1/mul:z:0lstm_37/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_37/zeros_1/Lessv
lstm_37/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/zeros_1/packed/1�
lstm_37/zeros_1/packedPacklstm_37/strided_slice:output:0!lstm_37/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_37/zeros_1/packeds
lstm_37/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/zeros_1/Const�
lstm_37/zeros_1Filllstm_37/zeros_1/packed:output:0lstm_37/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/zeros_1�
lstm_37/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_37/transpose/perm�
lstm_37/transpose	Transposelstm_36/transpose_1:y:0lstm_37/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_37/transposeg
lstm_37/Shape_1Shapelstm_37/transpose:y:0*
T0*
_output_shapes
:2
lstm_37/Shape_1�
lstm_37/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice_1/stack�
lstm_37/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_1/stack_1�
lstm_37/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_1/stack_2�
lstm_37/strided_slice_1StridedSlicelstm_37/Shape_1:output:0&lstm_37/strided_slice_1/stack:output:0(lstm_37/strided_slice_1/stack_1:output:0(lstm_37/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_37/strided_slice_1�
#lstm_37/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_37/TensorArrayV2/element_shape�
lstm_37/TensorArrayV2TensorListReserve,lstm_37/TensorArrayV2/element_shape:output:0 lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_37/TensorArrayV2�
=lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_37/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_37/transpose:y:0Flstm_37/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_37/TensorArrayUnstack/TensorListFromTensor�
lstm_37/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_37/strided_slice_2/stack�
lstm_37/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_2/stack_1�
lstm_37/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_2/stack_2�
lstm_37/strided_slice_2StridedSlicelstm_37/transpose:y:0&lstm_37/strided_slice_2/stack:output:0(lstm_37/strided_slice_2/stack_1:output:0(lstm_37/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_37/strided_slice_2�
*lstm_37/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3lstm_37_lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_37/lstm_cell_73/MatMul/ReadVariableOp�
lstm_37/lstm_cell_73/MatMulMatMul lstm_37/strided_slice_2:output:02lstm_37/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/MatMul�
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_37/lstm_cell_73/MatMul_1MatMullstm_37/zeros:output:04lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/MatMul_1�
lstm_37/lstm_cell_73/addAddV2%lstm_37/lstm_cell_73/MatMul:product:0'lstm_37/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/add�
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_37/lstm_cell_73/BiasAddBiasAddlstm_37/lstm_cell_73/add:z:03lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_37/lstm_cell_73/BiasAdd�
$lstm_37/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_37/lstm_cell_73/split/split_dim�
lstm_37/lstm_cell_73/splitSplit-lstm_37/lstm_cell_73/split/split_dim:output:0%lstm_37/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_37/lstm_cell_73/split�
lstm_37/lstm_cell_73/SigmoidSigmoid#lstm_37/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Sigmoid�
lstm_37/lstm_cell_73/Sigmoid_1Sigmoid#lstm_37/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_37/lstm_cell_73/Sigmoid_1�
lstm_37/lstm_cell_73/mulMul"lstm_37/lstm_cell_73/Sigmoid_1:y:0lstm_37/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul�
lstm_37/lstm_cell_73/ReluRelu#lstm_37/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Relu�
lstm_37/lstm_cell_73/mul_1Mul lstm_37/lstm_cell_73/Sigmoid:y:0'lstm_37/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul_1�
lstm_37/lstm_cell_73/add_1AddV2lstm_37/lstm_cell_73/mul:z:0lstm_37/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/add_1�
lstm_37/lstm_cell_73/Sigmoid_2Sigmoid#lstm_37/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_37/lstm_cell_73/Sigmoid_2�
lstm_37/lstm_cell_73/Relu_1Relulstm_37/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/Relu_1�
lstm_37/lstm_cell_73/mul_2Mul"lstm_37/lstm_cell_73/Sigmoid_2:y:0)lstm_37/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_37/lstm_cell_73/mul_2�
%lstm_37/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_37/TensorArrayV2_1/element_shape�
lstm_37/TensorArrayV2_1TensorListReserve.lstm_37/TensorArrayV2_1/element_shape:output:0 lstm_37/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_37/TensorArrayV2_1^
lstm_37/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/time�
 lstm_37/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_37/while/maximum_iterationsz
lstm_37/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_37/while/loop_counter�
lstm_37/whileWhile#lstm_37/while/loop_counter:output:0)lstm_37/while/maximum_iterations:output:0lstm_37/time:output:0 lstm_37/TensorArrayV2_1:handle:0lstm_37/zeros:output:0lstm_37/zeros_1:output:0 lstm_37/strided_slice_1:output:0?lstm_37/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_37_lstm_cell_73_matmul_readvariableop_resource5lstm_37_lstm_cell_73_matmul_1_readvariableop_resource4lstm_37_lstm_cell_73_biasadd_readvariableop_resource*
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
lstm_37_while_body_586473*%
condR
lstm_37_while_cond_586472*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_37/while�
8lstm_37/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_37/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_37/TensorArrayV2Stack/TensorListStackTensorListStacklstm_37/while:output:3Alstm_37/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_37/TensorArrayV2Stack/TensorListStack�
lstm_37/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_37/strided_slice_3/stack�
lstm_37/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_37/strided_slice_3/stack_1�
lstm_37/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_37/strided_slice_3/stack_2�
lstm_37/strided_slice_3StridedSlice3lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_37/strided_slice_3/stack:output:0(lstm_37/strided_slice_3/stack_1:output:0(lstm_37/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_37/strided_slice_3�
lstm_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_37/transpose_1/perm�
lstm_37/transpose_1	Transpose3lstm_37/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_37/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_37/transpose_1v
lstm_37/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_37/runtimey
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_18/dropout/Const�
dropout_18/dropout/MulMul lstm_37/strided_slice_3:output:0!dropout_18/dropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout_18/dropout/Mul�
dropout_18/dropout/ShapeShape lstm_37/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape�
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype021
/dropout_18/dropout/random_uniform/RandomUniform�
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2#
!dropout_18/dropout/GreaterEqual/y�
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2!
dropout_18/dropout/GreaterEqual�
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout_18/dropout/Cast�
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout_18/dropout/Mul_1�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_18/MatMul/ReadVariableOp�
dense_18/MatMulMatMuldropout_18/dropout/Mul_1:z:0&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_18/MatMul�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_18/BiasAdd/ReadVariableOp�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_18/BiasAddt
IdentityIdentitydense_18/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp,^lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp+^lstm_36/lstm_cell_72/MatMul/ReadVariableOp-^lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp^lstm_36/while,^lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp+^lstm_37/lstm_cell_73/MatMul/ReadVariableOp-^lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp^lstm_37/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2Z
+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp+lstm_36/lstm_cell_72/BiasAdd/ReadVariableOp2X
*lstm_36/lstm_cell_72/MatMul/ReadVariableOp*lstm_36/lstm_cell_72/MatMul/ReadVariableOp2\
,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp,lstm_36/lstm_cell_72/MatMul_1/ReadVariableOp2
lstm_36/whilelstm_36/while2Z
+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp+lstm_37/lstm_cell_73/BiasAdd/ReadVariableOp2X
*lstm_37/lstm_cell_73/MatMul/ReadVariableOp*lstm_37/lstm_cell_73/MatMul/ReadVariableOp2\
,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp,lstm_37/lstm_cell_73/MatMul_1/ReadVariableOp2
lstm_37/whilelstm_37/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_585859
lstm_36_input!
lstm_36_585838:	�!
lstm_36_585840:	@�
lstm_36_585842:	�!
lstm_37_585845:	@�!
lstm_37_585847:	 �
lstm_37_585849:	�!
dense_18_585853: 
dense_18_585855:
identity�� dense_18/StatefulPartitionedCall�lstm_36/StatefulPartitionedCall�lstm_37/StatefulPartitionedCall�
lstm_36/StatefulPartitionedCallStatefulPartitionedCalllstm_36_inputlstm_36_585838lstm_36_585840lstm_36_585842*
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5851602!
lstm_36/StatefulPartitionedCall�
lstm_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0lstm_37_585845lstm_37_585847lstm_37_585849*
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5853182!
lstm_37/StatefulPartitionedCall�
dropout_18/PartitionedCallPartitionedCall(lstm_37/StatefulPartitionedCall:output:0*
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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853312
dropout_18/PartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_18_585853dense_18_585855*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_5853432"
 dense_18/StatefulPartitionedCall�
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_18/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_36_input
�
�
(__inference_lstm_37_layer_call_fn_587230
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5845312
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
�
�
I__inference_sequential_18_layer_call_and_return_conditional_losses_585795

inputs!
lstm_36_585774:	�!
lstm_36_585776:	@�
lstm_36_585778:	�!
lstm_37_585781:	@�!
lstm_37_585783:	 �
lstm_37_585785:	�!
dense_18_585789: 
dense_18_585791:
identity�� dense_18/StatefulPartitionedCall�"dropout_18/StatefulPartitionedCall�lstm_36/StatefulPartitionedCall�lstm_37/StatefulPartitionedCall�
lstm_36/StatefulPartitionedCallStatefulPartitionedCallinputslstm_36_585774lstm_36_585776lstm_36_585778*
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5857392!
lstm_36/StatefulPartitionedCall�
lstm_37/StatefulPartitionedCallStatefulPartitionedCall(lstm_36/StatefulPartitionedCall:output:0lstm_37_585781lstm_37_585783lstm_37_585785*
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5855662!
lstm_37/StatefulPartitionedCall�
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_37/StatefulPartitionedCall:output:0*
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
F__inference_dropout_18_layer_call_and_return_conditional_losses_5853992$
"dropout_18/StatefulPartitionedCall�
 dense_18/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_18_585789dense_18_585791*
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
D__inference_dense_18_layer_call_and_return_conditional_losses_5853432"
 dense_18/StatefulPartitionedCall�
IdentityIdentity)dense_18/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^dense_18/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall ^lstm_36/StatefulPartitionedCall ^lstm_37/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2D
 dense_18/StatefulPartitionedCall dense_18/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2B
lstm_36/StatefulPartitionedCalllstm_36/StatefulPartitionedCall2B
lstm_37/StatefulPartitionedCalllstm_37/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_583964

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
�
�
(__inference_lstm_36_layer_call_fn_586615

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
C__inference_lstm_36_layer_call_and_return_conditional_losses_5857392
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
�F
�
C__inference_lstm_36_layer_call_and_return_conditional_losses_584111

inputs&
lstm_cell_72_584029:	�&
lstm_cell_72_584031:	@�"
lstm_cell_72_584033:	�
identity��$lstm_cell_72/StatefulPartitionedCall�whileD
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
$lstm_cell_72/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_72_584029lstm_cell_72_584031lstm_cell_72_584033*
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5839642&
$lstm_cell_72/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_72_584029lstm_cell_72_584031lstm_cell_72_584033*
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
while_body_584042*
condR
while_cond_584041*K
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
NoOpNoOp%^lstm_cell_72/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_72/StatefulPartitionedCall$lstm_cell_72/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_584448

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
while_body_587330
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
while_body_587632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
.__inference_sequential_18_layer_call_fn_585933

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
I__inference_sequential_18_layer_call_and_return_conditional_losses_5853502
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

�
D__inference_dense_18_layer_call_and_return_conditional_losses_587913

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
�J
�

lstm_36_while_body_586326,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3+
'lstm_36_while_lstm_36_strided_slice_1_0g
clstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0:	�P
=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�K
<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
lstm_36_while_identity
lstm_36_while_identity_1
lstm_36_while_identity_2
lstm_36_while_identity_3
lstm_36_while_identity_4
lstm_36_while_identity_5)
%lstm_36_while_lstm_36_strided_slice_1e
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorL
9lstm_36_while_lstm_cell_72_matmul_readvariableop_resource:	�N
;lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource:	@�I
:lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource:	���1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_36/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0lstm_36_while_placeholderHlstm_36/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_36/while/TensorArrayV2Read/TensorListGetItem�
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp�
!lstm_36/while/lstm_cell_72/MatMulMatMul8lstm_36/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_36/while/lstm_cell_72/MatMul�
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp�
#lstm_36/while/lstm_cell_72/MatMul_1MatMullstm_36_while_placeholder_2:lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_36/while/lstm_cell_72/MatMul_1�
lstm_36/while/lstm_cell_72/addAddV2+lstm_36/while/lstm_cell_72/MatMul:product:0-lstm_36/while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_36/while/lstm_cell_72/add�
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp�
"lstm_36/while/lstm_cell_72/BiasAddBiasAdd"lstm_36/while/lstm_cell_72/add:z:09lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_36/while/lstm_cell_72/BiasAdd�
*lstm_36/while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_36/while/lstm_cell_72/split/split_dim�
 lstm_36/while/lstm_cell_72/splitSplit3lstm_36/while/lstm_cell_72/split/split_dim:output:0+lstm_36/while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_36/while/lstm_cell_72/split�
"lstm_36/while/lstm_cell_72/SigmoidSigmoid)lstm_36/while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_36/while/lstm_cell_72/Sigmoid�
$lstm_36/while/lstm_cell_72/Sigmoid_1Sigmoid)lstm_36/while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_36/while/lstm_cell_72/Sigmoid_1�
lstm_36/while/lstm_cell_72/mulMul(lstm_36/while/lstm_cell_72/Sigmoid_1:y:0lstm_36_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_36/while/lstm_cell_72/mul�
lstm_36/while/lstm_cell_72/ReluRelu)lstm_36/while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_36/while/lstm_cell_72/Relu�
 lstm_36/while/lstm_cell_72/mul_1Mul&lstm_36/while/lstm_cell_72/Sigmoid:y:0-lstm_36/while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/mul_1�
 lstm_36/while/lstm_cell_72/add_1AddV2"lstm_36/while/lstm_cell_72/mul:z:0$lstm_36/while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/add_1�
$lstm_36/while/lstm_cell_72/Sigmoid_2Sigmoid)lstm_36/while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_36/while/lstm_cell_72/Sigmoid_2�
!lstm_36/while/lstm_cell_72/Relu_1Relu$lstm_36/while/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_36/while/lstm_cell_72/Relu_1�
 lstm_36/while/lstm_cell_72/mul_2Mul(lstm_36/while/lstm_cell_72/Sigmoid_2:y:0/lstm_36/while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_36/while/lstm_cell_72/mul_2�
2lstm_36/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_36_while_placeholder_1lstm_36_while_placeholder$lstm_36/while/lstm_cell_72/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_36/while/TensorArrayV2Write/TensorListSetIteml
lstm_36/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_36/while/add/y�
lstm_36/while/addAddV2lstm_36_while_placeholderlstm_36/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_36/while/addp
lstm_36/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_36/while/add_1/y�
lstm_36/while/add_1AddV2(lstm_36_while_lstm_36_while_loop_counterlstm_36/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_36/while/add_1�
lstm_36/while/IdentityIdentitylstm_36/while/add_1:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity�
lstm_36/while/Identity_1Identity.lstm_36_while_lstm_36_while_maximum_iterations^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_1�
lstm_36/while/Identity_2Identitylstm_36/while/add:z:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_2�
lstm_36/while/Identity_3IdentityBlstm_36/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_36/while/NoOp*
T0*
_output_shapes
: 2
lstm_36/while/Identity_3�
lstm_36/while/Identity_4Identity$lstm_36/while/lstm_cell_72/mul_2:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_36/while/Identity_4�
lstm_36/while/Identity_5Identity$lstm_36/while/lstm_cell_72/add_1:z:0^lstm_36/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_36/while/Identity_5�
lstm_36/while/NoOpNoOp2^lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp1^lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp3^lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_36/while/NoOp"9
lstm_36_while_identitylstm_36/while/Identity:output:0"=
lstm_36_while_identity_1!lstm_36/while/Identity_1:output:0"=
lstm_36_while_identity_2!lstm_36/while/Identity_2:output:0"=
lstm_36_while_identity_3!lstm_36/while/Identity_3:output:0"=
lstm_36_while_identity_4!lstm_36/while/Identity_4:output:0"=
lstm_36_while_identity_5!lstm_36/while/Identity_5:output:0"P
%lstm_36_while_lstm_36_strided_slice_1'lstm_36_while_lstm_36_strided_slice_1_0"z
:lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource<lstm_36_while_lstm_cell_72_biasadd_readvariableop_resource_0"|
;lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource=lstm_36_while_lstm_cell_72_matmul_1_readvariableop_resource_0"x
9lstm_36_while_lstm_cell_72_matmul_readvariableop_resource;lstm_36_while_lstm_cell_72_matmul_readvariableop_resource_0"�
alstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensorclstm_36_while_tensorarrayv2read_tensorlistgetitem_lstm_36_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp1lstm_36/while/lstm_cell_72/BiasAdd/ReadVariableOp2d
0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp0lstm_36/while/lstm_cell_72/MatMul/ReadVariableOp2h
2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp2lstm_36/while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
while_cond_584461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_584461___redundant_placeholder04
0while_while_cond_584461___redundant_placeholder14
0while_while_cond_584461___redundant_placeholder24
0while_while_cond_584461___redundant_placeholder3
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_585566

inputs>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_585482*
condR
while_cond_585481*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�?
�
while_body_586682
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
while_body_587783
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
�\
�
C__inference_lstm_37_layer_call_and_return_conditional_losses_587565
inputs_0>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_587481*
condR
while_cond_587480*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_585233
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_585233___redundant_placeholder04
0while_while_cond_585233___redundant_placeholder14
0while_while_cond_585233___redundant_placeholder24
0while_while_cond_585233___redundant_placeholder3
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
�%
�
while_body_584462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_73_584486_0:	@�.
while_lstm_cell_73_584488_0:	 �*
while_lstm_cell_73_584490_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_73_584486:	@�,
while_lstm_cell_73_584488:	 �(
while_lstm_cell_73_584490:	���*while/lstm_cell_73/StatefulPartitionedCall�
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
*while/lstm_cell_73/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_73_584486_0while_lstm_cell_73_584488_0while_lstm_cell_73_584490_0*
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5844482,
*while/lstm_cell_73/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_73/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_73/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_73/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_73/StatefulPartitionedCall*"
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
while_lstm_cell_73_584486while_lstm_cell_73_584486_0"8
while_lstm_cell_73_584488while_lstm_cell_73_584488_0"8
while_lstm_cell_73_584490while_lstm_cell_73_584490_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_73/StatefulPartitionedCall*while/lstm_cell_73/StatefulPartitionedCall: 
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
-__inference_lstm_cell_72_layer_call_fn_587930

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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5838182
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
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_587894

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
�F
�
C__inference_lstm_36_layer_call_and_return_conditional_losses_583901

inputs&
lstm_cell_72_583819:	�&
lstm_cell_72_583821:	@�"
lstm_cell_72_583823:	�
identity��$lstm_cell_72/StatefulPartitionedCall�whileD
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
$lstm_cell_72/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_72_583819lstm_cell_72_583821lstm_cell_72_583823*
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_5838182&
$lstm_cell_72/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_72_583819lstm_cell_72_583821lstm_cell_72_583823*
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
while_body_583832*
condR
while_cond_583831*K
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
NoOpNoOp%^lstm_cell_72/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_72/StatefulPartitionedCall$lstm_cell_72/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�[
�
C__inference_lstm_36_layer_call_and_return_conditional_losses_587219

inputs>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_587135*
condR
while_cond_587134*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_586833
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_72_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_72_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_72_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_72_matmul_readvariableop_resource:	�F
3while_lstm_cell_72_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_72_biasadd_readvariableop_resource:	���)while/lstm_cell_72/BiasAdd/ReadVariableOp�(while/lstm_cell_72/MatMul/ReadVariableOp�*while/lstm_cell_72/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_72/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_72_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_72/MatMul/ReadVariableOp�
while/lstm_cell_72/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul�
*while/lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_72_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_72/MatMul_1/ReadVariableOp�
while/lstm_cell_72/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/MatMul_1�
while/lstm_cell_72/addAddV2#while/lstm_cell_72/MatMul:product:0%while/lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/add�
)while/lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_72_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_72/BiasAdd/ReadVariableOp�
while/lstm_cell_72/BiasAddBiasAddwhile/lstm_cell_72/add:z:01while/lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_72/BiasAdd�
"while/lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_72/split/split_dim�
while/lstm_cell_72/splitSplit+while/lstm_cell_72/split/split_dim:output:0#while/lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_72/split�
while/lstm_cell_72/SigmoidSigmoid!while/lstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid�
while/lstm_cell_72/Sigmoid_1Sigmoid!while/lstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_1�
while/lstm_cell_72/mulMul while/lstm_cell_72/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul�
while/lstm_cell_72/ReluRelu!while/lstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu�
while/lstm_cell_72/mul_1Mulwhile/lstm_cell_72/Sigmoid:y:0%while/lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_1�
while/lstm_cell_72/add_1AddV2while/lstm_cell_72/mul:z:0while/lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/add_1�
while/lstm_cell_72/Sigmoid_2Sigmoid!while/lstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Sigmoid_2�
while/lstm_cell_72/Relu_1Reluwhile/lstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/Relu_1�
while/lstm_cell_72/mul_2Mul while/lstm_cell_72/Sigmoid_2:y:0'while/lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_72/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_72/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_72/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_72/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_72/BiasAdd/ReadVariableOp)^while/lstm_cell_72/MatMul/ReadVariableOp+^while/lstm_cell_72/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_72_biasadd_readvariableop_resource4while_lstm_cell_72_biasadd_readvariableop_resource_0"l
3while_lstm_cell_72_matmul_1_readvariableop_resource5while_lstm_cell_72_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_72_matmul_readvariableop_resource3while_lstm_cell_72_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_72/BiasAdd/ReadVariableOp)while/lstm_cell_72/BiasAdd/ReadVariableOp2T
(while/lstm_cell_72/MatMul/ReadVariableOp(while/lstm_cell_72/MatMul/ReadVariableOp2X
*while/lstm_cell_72/MatMul_1/ReadVariableOp*while/lstm_cell_72/MatMul_1/ReadVariableOp: 
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
while_body_585234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_73_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_73_matmul_readvariableop_resource:	@�F
3while_lstm_cell_73_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_73_biasadd_readvariableop_resource:	���)while/lstm_cell_73/BiasAdd/ReadVariableOp�(while/lstm_cell_73/MatMul/ReadVariableOp�*while/lstm_cell_73/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_73/MatMul/ReadVariableOp�
while/lstm_cell_73/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul�
*while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_73/MatMul_1/ReadVariableOp�
while/lstm_cell_73/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/MatMul_1�
while/lstm_cell_73/addAddV2#while/lstm_cell_73/MatMul:product:0%while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/add�
)while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_73/BiasAdd/ReadVariableOp�
while/lstm_cell_73/BiasAddBiasAddwhile/lstm_cell_73/add:z:01while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_73/BiasAdd�
"while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_73/split/split_dim�
while/lstm_cell_73/splitSplit+while/lstm_cell_73/split/split_dim:output:0#while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_73/split�
while/lstm_cell_73/SigmoidSigmoid!while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid�
while/lstm_cell_73/Sigmoid_1Sigmoid!while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_1�
while/lstm_cell_73/mulMul while/lstm_cell_73/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul�
while/lstm_cell_73/ReluRelu!while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu�
while/lstm_cell_73/mul_1Mulwhile/lstm_cell_73/Sigmoid:y:0%while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_1�
while/lstm_cell_73/add_1AddV2while/lstm_cell_73/mul:z:0while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/add_1�
while/lstm_cell_73/Sigmoid_2Sigmoid!while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Sigmoid_2�
while/lstm_cell_73/Relu_1Reluwhile/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/Relu_1�
while/lstm_cell_73/mul_2Mul while/lstm_cell_73/Sigmoid_2:y:0'while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_73/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_73/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_73/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_73/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_73/BiasAdd/ReadVariableOp)^while/lstm_cell_73/MatMul/ReadVariableOp+^while/lstm_cell_73/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_73_biasadd_readvariableop_resource4while_lstm_cell_73_biasadd_readvariableop_resource_0"l
3while_lstm_cell_73_matmul_1_readvariableop_resource5while_lstm_cell_73_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_73_matmul_readvariableop_resource3while_lstm_cell_73_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_73/BiasAdd/ReadVariableOp)while/lstm_cell_73/BiasAdd/ReadVariableOp2T
(while/lstm_cell_73/MatMul/ReadVariableOp(while/lstm_cell_73/MatMul/ReadVariableOp2X
*while/lstm_cell_73/MatMul_1/ReadVariableOp*while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_585739

inputs>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_585655*
condR
while_cond_585654*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�I
�
__inference__traced_save_588225
file_prefix.
*savev2_dense_18_kernel_read_readvariableop,
(savev2_dense_18_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_36_lstm_cell_72_kernel_read_readvariableopD
@savev2_lstm_36_lstm_cell_72_recurrent_kernel_read_readvariableop8
4savev2_lstm_36_lstm_cell_72_bias_read_readvariableop:
6savev2_lstm_37_lstm_cell_73_kernel_read_readvariableopD
@savev2_lstm_37_lstm_cell_73_recurrent_kernel_read_readvariableop8
4savev2_lstm_37_lstm_cell_73_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_dense_18_kernel_m_read_readvariableop3
/savev2_adam_dense_18_bias_m_read_readvariableopA
=savev2_adam_lstm_36_lstm_cell_72_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_36_lstm_cell_72_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_36_lstm_cell_72_bias_m_read_readvariableopA
=savev2_adam_lstm_37_lstm_cell_73_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_37_lstm_cell_73_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_37_lstm_cell_73_bias_m_read_readvariableop5
1savev2_adam_dense_18_kernel_v_read_readvariableop3
/savev2_adam_dense_18_bias_v_read_readvariableopA
=savev2_adam_lstm_36_lstm_cell_72_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_36_lstm_cell_72_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_36_lstm_cell_72_bias_v_read_readvariableopA
=savev2_adam_lstm_37_lstm_cell_73_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_37_lstm_cell_73_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_37_lstm_cell_73_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_18_kernel_read_readvariableop(savev2_dense_18_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_36_lstm_cell_72_kernel_read_readvariableop@savev2_lstm_36_lstm_cell_72_recurrent_kernel_read_readvariableop4savev2_lstm_36_lstm_cell_72_bias_read_readvariableop6savev2_lstm_37_lstm_cell_73_kernel_read_readvariableop@savev2_lstm_37_lstm_cell_73_recurrent_kernel_read_readvariableop4savev2_lstm_37_lstm_cell_73_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_dense_18_kernel_m_read_readvariableop/savev2_adam_dense_18_bias_m_read_readvariableop=savev2_adam_lstm_36_lstm_cell_72_kernel_m_read_readvariableopGsavev2_adam_lstm_36_lstm_cell_72_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_36_lstm_cell_72_bias_m_read_readvariableop=savev2_adam_lstm_37_lstm_cell_73_kernel_m_read_readvariableopGsavev2_adam_lstm_37_lstm_cell_73_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_37_lstm_cell_73_bias_m_read_readvariableop1savev2_adam_dense_18_kernel_v_read_readvariableop/savev2_adam_dense_18_bias_v_read_readvariableop=savev2_adam_lstm_36_lstm_cell_72_kernel_v_read_readvariableopGsavev2_adam_lstm_36_lstm_cell_72_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_36_lstm_cell_72_bias_v_read_readvariableop=savev2_adam_lstm_37_lstm_cell_73_kernel_v_read_readvariableopGsavev2_adam_lstm_37_lstm_cell_73_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_37_lstm_cell_73_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�

�
lstm_36_while_cond_586020,
(lstm_36_while_lstm_36_while_loop_counter2
.lstm_36_while_lstm_36_while_maximum_iterations
lstm_36_while_placeholder
lstm_36_while_placeholder_1
lstm_36_while_placeholder_2
lstm_36_while_placeholder_3.
*lstm_36_while_less_lstm_36_strided_slice_1D
@lstm_36_while_lstm_36_while_cond_586020___redundant_placeholder0D
@lstm_36_while_lstm_36_while_cond_586020___redundant_placeholder1D
@lstm_36_while_lstm_36_while_cond_586020___redundant_placeholder2D
@lstm_36_while_lstm_36_while_cond_586020___redundant_placeholder3
lstm_36_while_identity
�
lstm_36/while/LessLesslstm_36_while_placeholder*lstm_36_while_less_lstm_36_strided_slice_1*
T0*
_output_shapes
: 2
lstm_36/while/Lessu
lstm_36/while/IdentityIdentitylstm_36/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_36/while/Identity"9
lstm_36_while_identitylstm_36/while/Identity:output:0*(
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
�^
�
'sequential_18_lstm_37_while_body_583652H
Dsequential_18_lstm_37_while_sequential_18_lstm_37_while_loop_counterN
Jsequential_18_lstm_37_while_sequential_18_lstm_37_while_maximum_iterations+
'sequential_18_lstm_37_while_placeholder-
)sequential_18_lstm_37_while_placeholder_1-
)sequential_18_lstm_37_while_placeholder_2-
)sequential_18_lstm_37_while_placeholder_3G
Csequential_18_lstm_37_while_sequential_18_lstm_37_strided_slice_1_0�
sequential_18_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_37_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_18_lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0:	@�^
Ksequential_18_lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �Y
Jsequential_18_lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0:	�(
$sequential_18_lstm_37_while_identity*
&sequential_18_lstm_37_while_identity_1*
&sequential_18_lstm_37_while_identity_2*
&sequential_18_lstm_37_while_identity_3*
&sequential_18_lstm_37_while_identity_4*
&sequential_18_lstm_37_while_identity_5E
Asequential_18_lstm_37_while_sequential_18_lstm_37_strided_slice_1�
}sequential_18_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_37_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_18_lstm_37_while_lstm_cell_73_matmul_readvariableop_resource:	@�\
Isequential_18_lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource:	 �W
Hsequential_18_lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource:	���?sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�>sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�@sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
Msequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2O
Msequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape�
?sequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_18_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_37_tensorarrayunstack_tensorlistfromtensor_0'sequential_18_lstm_37_while_placeholderVsequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02A
?sequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItem�
>sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOpIsequential_18_lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02@
>sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�
/sequential_18/lstm_37/while/lstm_cell_73/MatMulMatMulFsequential_18/lstm_37/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_18/lstm_37/while/lstm_cell_73/MatMul�
@sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOpKsequential_18_lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02B
@sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
1sequential_18/lstm_37/while/lstm_cell_73/MatMul_1MatMul)sequential_18_lstm_37_while_placeholder_2Hsequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������23
1sequential_18/lstm_37/while/lstm_cell_73/MatMul_1�
,sequential_18/lstm_37/while/lstm_cell_73/addAddV29sequential_18/lstm_37/while/lstm_cell_73/MatMul:product:0;sequential_18/lstm_37/while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2.
,sequential_18/lstm_37/while/lstm_cell_73/add�
?sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOpJsequential_18_lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02A
?sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�
0sequential_18/lstm_37/while/lstm_cell_73/BiasAddBiasAdd0sequential_18/lstm_37/while/lstm_cell_73/add:z:0Gsequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_18/lstm_37/while/lstm_cell_73/BiasAdd�
8sequential_18/lstm_37/while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_18/lstm_37/while/lstm_cell_73/split/split_dim�
.sequential_18/lstm_37/while/lstm_cell_73/splitSplitAsequential_18/lstm_37/while/lstm_cell_73/split/split_dim:output:09sequential_18/lstm_37/while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split20
.sequential_18/lstm_37/while/lstm_cell_73/split�
0sequential_18/lstm_37/while/lstm_cell_73/SigmoidSigmoid7sequential_18/lstm_37/while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 22
0sequential_18/lstm_37/while/lstm_cell_73/Sigmoid�
2sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_1Sigmoid7sequential_18/lstm_37/while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 24
2sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_1�
,sequential_18/lstm_37/while/lstm_cell_73/mulMul6sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_1:y:0)sequential_18_lstm_37_while_placeholder_3*
T0*'
_output_shapes
:��������� 2.
,sequential_18/lstm_37/while/lstm_cell_73/mul�
-sequential_18/lstm_37/while/lstm_cell_73/ReluRelu7sequential_18/lstm_37/while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2/
-sequential_18/lstm_37/while/lstm_cell_73/Relu�
.sequential_18/lstm_37/while/lstm_cell_73/mul_1Mul4sequential_18/lstm_37/while/lstm_cell_73/Sigmoid:y:0;sequential_18/lstm_37/while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_18/lstm_37/while/lstm_cell_73/mul_1�
.sequential_18/lstm_37/while/lstm_cell_73/add_1AddV20sequential_18/lstm_37/while/lstm_cell_73/mul:z:02sequential_18/lstm_37/while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 20
.sequential_18/lstm_37/while/lstm_cell_73/add_1�
2sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_2Sigmoid7sequential_18/lstm_37/while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 24
2sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_2�
/sequential_18/lstm_37/while/lstm_cell_73/Relu_1Relu2sequential_18/lstm_37/while/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 21
/sequential_18/lstm_37/while/lstm_cell_73/Relu_1�
.sequential_18/lstm_37/while/lstm_cell_73/mul_2Mul6sequential_18/lstm_37/while/lstm_cell_73/Sigmoid_2:y:0=sequential_18/lstm_37/while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 20
.sequential_18/lstm_37/while/lstm_cell_73/mul_2�
@sequential_18/lstm_37/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_18_lstm_37_while_placeholder_1'sequential_18_lstm_37_while_placeholder2sequential_18/lstm_37/while/lstm_cell_73/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_18/lstm_37/while/TensorArrayV2Write/TensorListSetItem�
!sequential_18/lstm_37/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_18/lstm_37/while/add/y�
sequential_18/lstm_37/while/addAddV2'sequential_18_lstm_37_while_placeholder*sequential_18/lstm_37/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_18/lstm_37/while/add�
#sequential_18/lstm_37/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_18/lstm_37/while/add_1/y�
!sequential_18/lstm_37/while/add_1AddV2Dsequential_18_lstm_37_while_sequential_18_lstm_37_while_loop_counter,sequential_18/lstm_37/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_18/lstm_37/while/add_1�
$sequential_18/lstm_37/while/IdentityIdentity%sequential_18/lstm_37/while/add_1:z:0!^sequential_18/lstm_37/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_18/lstm_37/while/Identity�
&sequential_18/lstm_37/while/Identity_1IdentityJsequential_18_lstm_37_while_sequential_18_lstm_37_while_maximum_iterations!^sequential_18/lstm_37/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_37/while/Identity_1�
&sequential_18/lstm_37/while/Identity_2Identity#sequential_18/lstm_37/while/add:z:0!^sequential_18/lstm_37/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_37/while/Identity_2�
&sequential_18/lstm_37/while/Identity_3IdentityPsequential_18/lstm_37/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_18/lstm_37/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_18/lstm_37/while/Identity_3�
&sequential_18/lstm_37/while/Identity_4Identity2sequential_18/lstm_37/while/lstm_cell_73/mul_2:z:0!^sequential_18/lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_18/lstm_37/while/Identity_4�
&sequential_18/lstm_37/while/Identity_5Identity2sequential_18/lstm_37/while/lstm_cell_73/add_1:z:0!^sequential_18/lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2(
&sequential_18/lstm_37/while/Identity_5�
 sequential_18/lstm_37/while/NoOpNoOp@^sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp?^sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOpA^sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_18/lstm_37/while/NoOp"U
$sequential_18_lstm_37_while_identity-sequential_18/lstm_37/while/Identity:output:0"Y
&sequential_18_lstm_37_while_identity_1/sequential_18/lstm_37/while/Identity_1:output:0"Y
&sequential_18_lstm_37_while_identity_2/sequential_18/lstm_37/while/Identity_2:output:0"Y
&sequential_18_lstm_37_while_identity_3/sequential_18/lstm_37/while/Identity_3:output:0"Y
&sequential_18_lstm_37_while_identity_4/sequential_18/lstm_37/while/Identity_4:output:0"Y
&sequential_18_lstm_37_while_identity_5/sequential_18/lstm_37/while/Identity_5:output:0"�
Hsequential_18_lstm_37_while_lstm_cell_73_biasadd_readvariableop_resourceJsequential_18_lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0"�
Isequential_18_lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resourceKsequential_18_lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0"�
Gsequential_18_lstm_37_while_lstm_cell_73_matmul_readvariableop_resourceIsequential_18_lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0"�
Asequential_18_lstm_37_while_sequential_18_lstm_37_strided_slice_1Csequential_18_lstm_37_while_sequential_18_lstm_37_strided_slice_1_0"�
}sequential_18_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_37_tensorarrayunstack_tensorlistfromtensorsequential_18_lstm_37_while_tensorarrayv2read_tensorlistgetitem_sequential_18_lstm_37_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2�
?sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp?sequential_18/lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp2�
>sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp>sequential_18/lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp2�
@sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp@sequential_18/lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
-__inference_lstm_cell_73_layer_call_fn_588028

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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5844482
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
�
�
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588077

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
F__inference_dropout_18_layer_call_and_return_conditional_losses_585399

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
C__inference_lstm_36_layer_call_and_return_conditional_losses_586766
inputs_0>
+lstm_cell_72_matmul_readvariableop_resource:	�@
-lstm_cell_72_matmul_1_readvariableop_resource:	@�;
,lstm_cell_72_biasadd_readvariableop_resource:	�
identity��#lstm_cell_72/BiasAdd/ReadVariableOp�"lstm_cell_72/MatMul/ReadVariableOp�$lstm_cell_72/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_72/MatMul/ReadVariableOpReadVariableOp+lstm_cell_72_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_72/MatMul/ReadVariableOp�
lstm_cell_72/MatMulMatMulstrided_slice_2:output:0*lstm_cell_72/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul�
$lstm_cell_72/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_72_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_72/MatMul_1/ReadVariableOp�
lstm_cell_72/MatMul_1MatMulzeros:output:0,lstm_cell_72/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/MatMul_1�
lstm_cell_72/addAddV2lstm_cell_72/MatMul:product:0lstm_cell_72/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/add�
#lstm_cell_72/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_72_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_72/BiasAdd/ReadVariableOp�
lstm_cell_72/BiasAddBiasAddlstm_cell_72/add:z:0+lstm_cell_72/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_72/BiasAdd~
lstm_cell_72/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_72/split/split_dim�
lstm_cell_72/splitSplit%lstm_cell_72/split/split_dim:output:0lstm_cell_72/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_72/split�
lstm_cell_72/SigmoidSigmoidlstm_cell_72/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid�
lstm_cell_72/Sigmoid_1Sigmoidlstm_cell_72/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_1�
lstm_cell_72/mulMullstm_cell_72/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul}
lstm_cell_72/ReluRelulstm_cell_72/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu�
lstm_cell_72/mul_1Mullstm_cell_72/Sigmoid:y:0lstm_cell_72/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_1�
lstm_cell_72/add_1AddV2lstm_cell_72/mul:z:0lstm_cell_72/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/add_1�
lstm_cell_72/Sigmoid_2Sigmoidlstm_cell_72/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Sigmoid_2|
lstm_cell_72/Relu_1Relulstm_cell_72/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/Relu_1�
lstm_cell_72/mul_2Mullstm_cell_72/Sigmoid_2:y:0!lstm_cell_72/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_72/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_72_matmul_readvariableop_resource-lstm_cell_72_matmul_1_readvariableop_resource,lstm_cell_72_biasadd_readvariableop_resource*
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
while_body_586682*
condR
while_cond_586681*K
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
NoOpNoOp$^lstm_cell_72/BiasAdd/ReadVariableOp#^lstm_cell_72/MatMul/ReadVariableOp%^lstm_cell_72/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_72/BiasAdd/ReadVariableOp#lstm_cell_72/BiasAdd/ReadVariableOp2H
"lstm_cell_72/MatMul/ReadVariableOp"lstm_cell_72/MatMul/ReadVariableOp2L
$lstm_cell_72/MatMul_1/ReadVariableOp$lstm_cell_72/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�F
�
C__inference_lstm_37_layer_call_and_return_conditional_losses_584531

inputs&
lstm_cell_73_584449:	@�&
lstm_cell_73_584451:	 �"
lstm_cell_73_584453:	�
identity��$lstm_cell_73/StatefulPartitionedCall�whileD
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
$lstm_cell_73/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_73_584449lstm_cell_73_584451lstm_cell_73_584453*
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5844482&
$lstm_cell_73/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_73_584449lstm_cell_73_584451lstm_cell_73_584453*
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
while_body_584462*
condR
while_cond_584461*K
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
NoOpNoOp%^lstm_cell_73/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_73/StatefulPartitionedCall$lstm_cell_73/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
while_cond_586681
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_586681___redundant_placeholder04
0while_while_cond_586681___redundant_placeholder14
0while_while_cond_586681___redundant_placeholder24
0while_while_cond_586681___redundant_placeholder3
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_585318

inputs>
+lstm_cell_73_matmul_readvariableop_resource:	@�@
-lstm_cell_73_matmul_1_readvariableop_resource:	 �;
,lstm_cell_73_biasadd_readvariableop_resource:	�
identity��#lstm_cell_73/BiasAdd/ReadVariableOp�"lstm_cell_73/MatMul/ReadVariableOp�$lstm_cell_73/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_73/MatMul/ReadVariableOpReadVariableOp+lstm_cell_73_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_73/MatMul/ReadVariableOp�
lstm_cell_73/MatMulMatMulstrided_slice_2:output:0*lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul�
$lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_73_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_73/MatMul_1/ReadVariableOp�
lstm_cell_73/MatMul_1MatMulzeros:output:0,lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/MatMul_1�
lstm_cell_73/addAddV2lstm_cell_73/MatMul:product:0lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/add�
#lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_73_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_73/BiasAdd/ReadVariableOp�
lstm_cell_73/BiasAddBiasAddlstm_cell_73/add:z:0+lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_73/BiasAdd~
lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_73/split/split_dim�
lstm_cell_73/splitSplit%lstm_cell_73/split/split_dim:output:0lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_73/split�
lstm_cell_73/SigmoidSigmoidlstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid�
lstm_cell_73/Sigmoid_1Sigmoidlstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_1�
lstm_cell_73/mulMullstm_cell_73/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul}
lstm_cell_73/ReluRelulstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu�
lstm_cell_73/mul_1Mullstm_cell_73/Sigmoid:y:0lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_1�
lstm_cell_73/add_1AddV2lstm_cell_73/mul:z:0lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/add_1�
lstm_cell_73/Sigmoid_2Sigmoidlstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Sigmoid_2|
lstm_cell_73/Relu_1Relulstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/Relu_1�
lstm_cell_73/mul_2Mullstm_cell_73/Sigmoid_2:y:0!lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_73/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_73_matmul_readvariableop_resource-lstm_cell_73_matmul_1_readvariableop_resource,lstm_cell_73_biasadd_readvariableop_resource*
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
while_body_585234*
condR
while_cond_585233*K
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
NoOpNoOp$^lstm_cell_73/BiasAdd/ReadVariableOp#^lstm_cell_73/MatMul/ReadVariableOp%^lstm_cell_73/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_73/BiasAdd/ReadVariableOp#lstm_cell_73/BiasAdd/ReadVariableOp2H
"lstm_cell_73/MatMul/ReadVariableOp"lstm_cell_73/MatMul/ReadVariableOp2L
$lstm_cell_73/MatMul_1/ReadVariableOp$lstm_cell_73/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_587631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_587631___redundant_placeholder04
0while_while_cond_587631___redundant_placeholder14
0while_while_cond_587631___redundant_placeholder24
0while_while_cond_587631___redundant_placeholder3
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
D__inference_dense_18_layer_call_and_return_conditional_losses_585343

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
�
�
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_587979

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
�

�
lstm_37_while_cond_586167,
(lstm_37_while_lstm_37_while_loop_counter2
.lstm_37_while_lstm_37_while_maximum_iterations
lstm_37_while_placeholder
lstm_37_while_placeholder_1
lstm_37_while_placeholder_2
lstm_37_while_placeholder_3.
*lstm_37_while_less_lstm_37_strided_slice_1D
@lstm_37_while_lstm_37_while_cond_586167___redundant_placeholder0D
@lstm_37_while_lstm_37_while_cond_586167___redundant_placeholder1D
@lstm_37_while_lstm_37_while_cond_586167___redundant_placeholder2D
@lstm_37_while_lstm_37_while_cond_586167___redundant_placeholder3
lstm_37_while_identity
�
lstm_37/while/LessLesslstm_37_while_placeholder*lstm_37_while_less_lstm_37_strided_slice_1*
T0*
_output_shapes
: 2
lstm_37/while/Lessu
lstm_37/while/IdentityIdentitylstm_37/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_37/while/Identity"9
lstm_37_while_identitylstm_37/while/Identity:output:0*(
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

lstm_37_while_body_586473,
(lstm_37_while_lstm_37_while_loop_counter2
.lstm_37_while_lstm_37_while_maximum_iterations
lstm_37_while_placeholder
lstm_37_while_placeholder_1
lstm_37_while_placeholder_2
lstm_37_while_placeholder_3+
'lstm_37_while_lstm_37_strided_slice_1_0g
clstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0:	@�P
=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0:	 �K
<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0:	�
lstm_37_while_identity
lstm_37_while_identity_1
lstm_37_while_identity_2
lstm_37_while_identity_3
lstm_37_while_identity_4
lstm_37_while_identity_5)
%lstm_37_while_lstm_37_strided_slice_1e
alstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensorL
9lstm_37_while_lstm_cell_73_matmul_readvariableop_resource:	@�N
;lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource:	 �I
:lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource:	���1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
?lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_37/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0lstm_37_while_placeholderHlstm_37/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_37/while/TensorArrayV2Read/TensorListGetItem�
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOpReadVariableOp;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp�
!lstm_37/while/lstm_cell_73/MatMulMatMul8lstm_37/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_37/while/lstm_cell_73/MatMul�
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOpReadVariableOp=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp�
#lstm_37/while/lstm_cell_73/MatMul_1MatMullstm_37_while_placeholder_2:lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_37/while/lstm_cell_73/MatMul_1�
lstm_37/while/lstm_cell_73/addAddV2+lstm_37/while/lstm_cell_73/MatMul:product:0-lstm_37/while/lstm_cell_73/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_37/while/lstm_cell_73/add�
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOpReadVariableOp<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp�
"lstm_37/while/lstm_cell_73/BiasAddBiasAdd"lstm_37/while/lstm_cell_73/add:z:09lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_37/while/lstm_cell_73/BiasAdd�
*lstm_37/while/lstm_cell_73/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_37/while/lstm_cell_73/split/split_dim�
 lstm_37/while/lstm_cell_73/splitSplit3lstm_37/while/lstm_cell_73/split/split_dim:output:0+lstm_37/while/lstm_cell_73/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_37/while/lstm_cell_73/split�
"lstm_37/while/lstm_cell_73/SigmoidSigmoid)lstm_37/while/lstm_cell_73/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_37/while/lstm_cell_73/Sigmoid�
$lstm_37/while/lstm_cell_73/Sigmoid_1Sigmoid)lstm_37/while/lstm_cell_73/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_37/while/lstm_cell_73/Sigmoid_1�
lstm_37/while/lstm_cell_73/mulMul(lstm_37/while/lstm_cell_73/Sigmoid_1:y:0lstm_37_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_37/while/lstm_cell_73/mul�
lstm_37/while/lstm_cell_73/ReluRelu)lstm_37/while/lstm_cell_73/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_37/while/lstm_cell_73/Relu�
 lstm_37/while/lstm_cell_73/mul_1Mul&lstm_37/while/lstm_cell_73/Sigmoid:y:0-lstm_37/while/lstm_cell_73/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/mul_1�
 lstm_37/while/lstm_cell_73/add_1AddV2"lstm_37/while/lstm_cell_73/mul:z:0$lstm_37/while/lstm_cell_73/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/add_1�
$lstm_37/while/lstm_cell_73/Sigmoid_2Sigmoid)lstm_37/while/lstm_cell_73/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_37/while/lstm_cell_73/Sigmoid_2�
!lstm_37/while/lstm_cell_73/Relu_1Relu$lstm_37/while/lstm_cell_73/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_37/while/lstm_cell_73/Relu_1�
 lstm_37/while/lstm_cell_73/mul_2Mul(lstm_37/while/lstm_cell_73/Sigmoid_2:y:0/lstm_37/while/lstm_cell_73/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_37/while/lstm_cell_73/mul_2�
2lstm_37/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_37_while_placeholder_1lstm_37_while_placeholder$lstm_37/while/lstm_cell_73/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_37/while/TensorArrayV2Write/TensorListSetIteml
lstm_37/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_37/while/add/y�
lstm_37/while/addAddV2lstm_37_while_placeholderlstm_37/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_37/while/addp
lstm_37/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_37/while/add_1/y�
lstm_37/while/add_1AddV2(lstm_37_while_lstm_37_while_loop_counterlstm_37/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_37/while/add_1�
lstm_37/while/IdentityIdentitylstm_37/while/add_1:z:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity�
lstm_37/while/Identity_1Identity.lstm_37_while_lstm_37_while_maximum_iterations^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_1�
lstm_37/while/Identity_2Identitylstm_37/while/add:z:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_2�
lstm_37/while/Identity_3IdentityBlstm_37/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_37/while/NoOp*
T0*
_output_shapes
: 2
lstm_37/while/Identity_3�
lstm_37/while/Identity_4Identity$lstm_37/while/lstm_cell_73/mul_2:z:0^lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_37/while/Identity_4�
lstm_37/while/Identity_5Identity$lstm_37/while/lstm_cell_73/add_1:z:0^lstm_37/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_37/while/Identity_5�
lstm_37/while/NoOpNoOp2^lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp1^lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp3^lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_37/while/NoOp"9
lstm_37_while_identitylstm_37/while/Identity:output:0"=
lstm_37_while_identity_1!lstm_37/while/Identity_1:output:0"=
lstm_37_while_identity_2!lstm_37/while/Identity_2:output:0"=
lstm_37_while_identity_3!lstm_37/while/Identity_3:output:0"=
lstm_37_while_identity_4!lstm_37/while/Identity_4:output:0"=
lstm_37_while_identity_5!lstm_37/while/Identity_5:output:0"P
%lstm_37_while_lstm_37_strided_slice_1'lstm_37_while_lstm_37_strided_slice_1_0"z
:lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource<lstm_37_while_lstm_cell_73_biasadd_readvariableop_resource_0"|
;lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource=lstm_37_while_lstm_cell_73_matmul_1_readvariableop_resource_0"x
9lstm_37_while_lstm_cell_73_matmul_readvariableop_resource;lstm_37_while_lstm_cell_73_matmul_readvariableop_resource_0"�
alstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensorclstm_37_while_tensorarrayv2read_tensorlistgetitem_lstm_37_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp1lstm_37/while/lstm_cell_73/BiasAdd/ReadVariableOp2d
0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp0lstm_37/while/lstm_cell_73/MatMul/ReadVariableOp2h
2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp2lstm_37/while/lstm_cell_73/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_37_layer_call_fn_587252

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
C__inference_lstm_37_layer_call_and_return_conditional_losses_5853182
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
while_cond_585075
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_585075___redundant_placeholder04
0while_while_cond_585075___redundant_placeholder14
0while_while_cond_585075___redundant_placeholder24
0while_while_cond_585075___redundant_placeholder3
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
�%
�
while_body_584672
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_73_584696_0:	@�.
while_lstm_cell_73_584698_0:	 �*
while_lstm_cell_73_584700_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_73_584696:	@�,
while_lstm_cell_73_584698:	 �(
while_lstm_cell_73_584700:	���*while/lstm_cell_73/StatefulPartitionedCall�
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
*while/lstm_cell_73/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_73_584696_0while_lstm_cell_73_584698_0while_lstm_cell_73_584700_0*
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_5845942,
*while/lstm_cell_73/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_73/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_73/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_73/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_73/StatefulPartitionedCall*"
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
while_lstm_cell_73_584696while_lstm_cell_73_584696_0"8
while_lstm_cell_73_584698while_lstm_cell_73_584698_0"8
while_lstm_cell_73_584700while_lstm_cell_73_584700_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_73/StatefulPartitionedCall*while/lstm_cell_73/StatefulPartitionedCall: 
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
: "�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_36_input:
serving_default_lstm_36_input:0���������<
dense_180
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
!: 2dense_18/kernel
:2dense_18/bias
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
.:,	�2lstm_36/lstm_cell_72/kernel
8:6	@�2%lstm_36/lstm_cell_72/recurrent_kernel
(:&�2lstm_36/lstm_cell_72/bias
.:,	@�2lstm_37/lstm_cell_73/kernel
8:6	 �2%lstm_37/lstm_cell_73/recurrent_kernel
(:&�2lstm_37/lstm_cell_73/bias
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
&:$ 2Adam/dense_18/kernel/m
 :2Adam/dense_18/bias/m
3:1	�2"Adam/lstm_36/lstm_cell_72/kernel/m
=:;	@�2,Adam/lstm_36/lstm_cell_72/recurrent_kernel/m
-:+�2 Adam/lstm_36/lstm_cell_72/bias/m
3:1	@�2"Adam/lstm_37/lstm_cell_73/kernel/m
=:;	 �2,Adam/lstm_37/lstm_cell_73/recurrent_kernel/m
-:+�2 Adam/lstm_37/lstm_cell_73/bias/m
&:$ 2Adam/dense_18/kernel/v
 :2Adam/dense_18/bias/v
3:1	�2"Adam/lstm_36/lstm_cell_72/kernel/v
=:;	@�2,Adam/lstm_36/lstm_cell_72/recurrent_kernel/v
-:+�2 Adam/lstm_36/lstm_cell_72/bias/v
3:1	@�2"Adam/lstm_37/lstm_cell_73/kernel/v
=:;	 �2,Adam/lstm_37/lstm_cell_73/recurrent_kernel/v
-:+�2 Adam/lstm_37/lstm_cell_73/bias/v
�B�
!__inference__wrapped_model_583743lstm_36_input"�
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
.__inference_sequential_18_layer_call_fn_585369
.__inference_sequential_18_layer_call_fn_585933
.__inference_sequential_18_layer_call_fn_585954
.__inference_sequential_18_layer_call_fn_585835�
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
I__inference_sequential_18_layer_call_and_return_conditional_losses_586259
I__inference_sequential_18_layer_call_and_return_conditional_losses_586571
I__inference_sequential_18_layer_call_and_return_conditional_losses_585859
I__inference_sequential_18_layer_call_and_return_conditional_losses_585883�
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
(__inference_lstm_36_layer_call_fn_586582
(__inference_lstm_36_layer_call_fn_586593
(__inference_lstm_36_layer_call_fn_586604
(__inference_lstm_36_layer_call_fn_586615�
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_586766
C__inference_lstm_36_layer_call_and_return_conditional_losses_586917
C__inference_lstm_36_layer_call_and_return_conditional_losses_587068
C__inference_lstm_36_layer_call_and_return_conditional_losses_587219�
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
(__inference_lstm_37_layer_call_fn_587230
(__inference_lstm_37_layer_call_fn_587241
(__inference_lstm_37_layer_call_fn_587252
(__inference_lstm_37_layer_call_fn_587263�
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587414
C__inference_lstm_37_layer_call_and_return_conditional_losses_587565
C__inference_lstm_37_layer_call_and_return_conditional_losses_587716
C__inference_lstm_37_layer_call_and_return_conditional_losses_587867�
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
+__inference_dropout_18_layer_call_fn_587872
+__inference_dropout_18_layer_call_fn_587877�
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
F__inference_dropout_18_layer_call_and_return_conditional_losses_587882
F__inference_dropout_18_layer_call_and_return_conditional_losses_587894�
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
)__inference_dense_18_layer_call_fn_587903�
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
D__inference_dense_18_layer_call_and_return_conditional_losses_587913�
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
$__inference_signature_wrapper_585912lstm_36_input"�
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
-__inference_lstm_cell_72_layer_call_fn_587930
-__inference_lstm_cell_72_layer_call_fn_587947�
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_587979
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_588011�
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
-__inference_lstm_cell_73_layer_call_fn_588028
-__inference_lstm_cell_73_layer_call_fn_588045�
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588077
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588109�
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
!__inference__wrapped_model_583743{&'()*+:�7
0�-
+�(
lstm_36_input���������
� "3�0
.
dense_18"�
dense_18����������
D__inference_dense_18_layer_call_and_return_conditional_losses_587913\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� |
)__inference_dense_18_layer_call_fn_587903O/�,
%�"
 �
inputs��������� 
� "�����������
F__inference_dropout_18_layer_call_and_return_conditional_losses_587882\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
F__inference_dropout_18_layer_call_and_return_conditional_losses_587894\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� ~
+__inference_dropout_18_layer_call_fn_587872O3�0
)�&
 �
inputs��������� 
p 
� "���������� ~
+__inference_dropout_18_layer_call_fn_587877O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_lstm_36_layer_call_and_return_conditional_losses_586766�&'(O�L
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_586917�&'(O�L
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_587068q&'(?�<
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
C__inference_lstm_36_layer_call_and_return_conditional_losses_587219q&'(?�<
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
(__inference_lstm_36_layer_call_fn_586582}&'(O�L
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
(__inference_lstm_36_layer_call_fn_586593}&'(O�L
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
(__inference_lstm_36_layer_call_fn_586604d&'(?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
(__inference_lstm_36_layer_call_fn_586615d&'(?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
C__inference_lstm_37_layer_call_and_return_conditional_losses_587414})*+O�L
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587565})*+O�L
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587716m)*+?�<
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
C__inference_lstm_37_layer_call_and_return_conditional_losses_587867m)*+?�<
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
(__inference_lstm_37_layer_call_fn_587230p)*+O�L
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
(__inference_lstm_37_layer_call_fn_587241p)*+O�L
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
(__inference_lstm_37_layer_call_fn_587252`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
(__inference_lstm_37_layer_call_fn_587263`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_587979�&'(��}
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
H__inference_lstm_cell_72_layer_call_and_return_conditional_losses_588011�&'(��}
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
-__inference_lstm_cell_72_layer_call_fn_587930�&'(��}
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
-__inference_lstm_cell_72_layer_call_fn_587947�&'(��}
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588077�)*+��}
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
H__inference_lstm_cell_73_layer_call_and_return_conditional_losses_588109�)*+��}
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
-__inference_lstm_cell_73_layer_call_fn_588028�)*+��}
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
-__inference_lstm_cell_73_layer_call_fn_588045�)*+��}
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
I__inference_sequential_18_layer_call_and_return_conditional_losses_585859u&'()*+B�?
8�5
+�(
lstm_36_input���������
p 

 
� "%�"
�
0���������
� �
I__inference_sequential_18_layer_call_and_return_conditional_losses_585883u&'()*+B�?
8�5
+�(
lstm_36_input���������
p

 
� "%�"
�
0���������
� �
I__inference_sequential_18_layer_call_and_return_conditional_losses_586259n&'()*+;�8
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
I__inference_sequential_18_layer_call_and_return_conditional_losses_586571n&'()*+;�8
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
.__inference_sequential_18_layer_call_fn_585369h&'()*+B�?
8�5
+�(
lstm_36_input���������
p 

 
� "�����������
.__inference_sequential_18_layer_call_fn_585835h&'()*+B�?
8�5
+�(
lstm_36_input���������
p

 
� "�����������
.__inference_sequential_18_layer_call_fn_585933a&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "�����������
.__inference_sequential_18_layer_call_fn_585954a&'()*+;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_585912�&'()*+K�H
� 
A�>
<
lstm_36_input+�(
lstm_36_input���������"3�0
.
dense_18"�
dense_18���������