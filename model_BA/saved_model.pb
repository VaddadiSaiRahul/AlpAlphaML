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
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

: *
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
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
lstm_16/lstm_cell_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*,
shared_namelstm_16/lstm_cell_32/kernel
�
/lstm_16/lstm_cell_32/kernel/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_32/kernel*
_output_shapes
:	�*
dtype0
�
%lstm_16/lstm_cell_32/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*6
shared_name'%lstm_16/lstm_cell_32/recurrent_kernel
�
9lstm_16/lstm_cell_32/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_16/lstm_cell_32/recurrent_kernel*
_output_shapes
:	@�*
dtype0
�
lstm_16/lstm_cell_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_16/lstm_cell_32/bias
�
-lstm_16/lstm_cell_32/bias/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_32/bias*
_output_shapes	
:�*
dtype0
�
lstm_17/lstm_cell_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*,
shared_namelstm_17/lstm_cell_33/kernel
�
/lstm_17/lstm_cell_33/kernel/Read/ReadVariableOpReadVariableOplstm_17/lstm_cell_33/kernel*
_output_shapes
:	@�*
dtype0
�
%lstm_17/lstm_cell_33/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*6
shared_name'%lstm_17/lstm_cell_33/recurrent_kernel
�
9lstm_17/lstm_cell_33/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_17/lstm_cell_33/recurrent_kernel*
_output_shapes
:	 �*
dtype0
�
lstm_17/lstm_cell_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namelstm_17/lstm_cell_33/bias
�
-lstm_17/lstm_cell_33/bias/Read/ReadVariableOpReadVariableOplstm_17/lstm_cell_33/bias*
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
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
�
"Adam/lstm_16/lstm_cell_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_16/lstm_cell_32/kernel/m
�
6Adam/lstm_16/lstm_cell_32/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_32/kernel/m*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_16/lstm_cell_32/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_16/lstm_cell_32/recurrent_kernel/m
�
@Adam/lstm_16/lstm_cell_32/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_32/recurrent_kernel/m*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_16/lstm_cell_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_16/lstm_cell_32/bias/m
�
4Adam/lstm_16/lstm_cell_32/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_32/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_17/lstm_cell_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_17/lstm_cell_33/kernel/m
�
6Adam/lstm_17/lstm_cell_33/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_17/lstm_cell_33/kernel/m*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_17/lstm_cell_33/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_17/lstm_cell_33/recurrent_kernel/m
�
@Adam/lstm_17/lstm_cell_33/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_17/lstm_cell_33/recurrent_kernel/m*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_17/lstm_cell_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_17/lstm_cell_33/bias/m
�
4Adam/lstm_17/lstm_cell_33/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_17/lstm_cell_33/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
�
"Adam/lstm_16/lstm_cell_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*3
shared_name$"Adam/lstm_16/lstm_cell_32/kernel/v
�
6Adam/lstm_16/lstm_cell_32/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_32/kernel/v*
_output_shapes
:	�*
dtype0
�
,Adam/lstm_16/lstm_cell_32/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*=
shared_name.,Adam/lstm_16/lstm_cell_32/recurrent_kernel/v
�
@Adam/lstm_16/lstm_cell_32/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_32/recurrent_kernel/v*
_output_shapes
:	@�*
dtype0
�
 Adam/lstm_16/lstm_cell_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_16/lstm_cell_32/bias/v
�
4Adam/lstm_16/lstm_cell_32/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_32/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/lstm_17/lstm_cell_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*3
shared_name$"Adam/lstm_17/lstm_cell_33/kernel/v
�
6Adam/lstm_17/lstm_cell_33/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_17/lstm_cell_33/kernel/v*
_output_shapes
:	@�*
dtype0
�
,Adam/lstm_17/lstm_cell_33/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 �*=
shared_name.,Adam/lstm_17/lstm_cell_33/recurrent_kernel/v
�
@Adam/lstm_17/lstm_cell_33/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_17/lstm_cell_33/recurrent_kernel/v*
_output_shapes
:	 �*
dtype0
�
 Adam/lstm_17/lstm_cell_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/lstm_17/lstm_cell_33/bias/v
�
4Adam/lstm_17/lstm_cell_33/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_17/lstm_cell_33/bias/v*
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
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_16/lstm_cell_32/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_16/lstm_cell_32/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_16/lstm_cell_32/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_17/lstm_cell_33/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_17/lstm_cell_33/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_17/lstm_cell_33/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_32/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_16/lstm_cell_32/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_16/lstm_cell_32/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_17/lstm_cell_33/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_17/lstm_cell_33/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_17/lstm_cell_33/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_32/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_16/lstm_cell_32/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_16/lstm_cell_32/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/lstm_17/lstm_cell_33/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/lstm_17/lstm_cell_33/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/lstm_17/lstm_cell_33/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_lstm_16_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_16_inputlstm_16/lstm_cell_32/kernel%lstm_16/lstm_cell_32/recurrent_kernellstm_16/lstm_cell_32/biaslstm_17/lstm_cell_33/kernel%lstm_17/lstm_cell_33/recurrent_kernellstm_17/lstm_cell_33/biasdense_8/kerneldense_8/bias*
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
$__inference_signature_wrapper_281262
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_16/lstm_cell_32/kernel/Read/ReadVariableOp9lstm_16/lstm_cell_32/recurrent_kernel/Read/ReadVariableOp-lstm_16/lstm_cell_32/bias/Read/ReadVariableOp/lstm_17/lstm_cell_33/kernel/Read/ReadVariableOp9lstm_17/lstm_cell_33/recurrent_kernel/Read/ReadVariableOp-lstm_17/lstm_cell_33/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_32/kernel/m/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_32/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_32/bias/m/Read/ReadVariableOp6Adam/lstm_17/lstm_cell_33/kernel/m/Read/ReadVariableOp@Adam/lstm_17/lstm_cell_33/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_17/lstm_cell_33/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_32/kernel/v/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_32/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_32/bias/v/Read/ReadVariableOp6Adam/lstm_17/lstm_cell_33/kernel/v/Read/ReadVariableOp@Adam/lstm_17/lstm_cell_33/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_17/lstm_cell_33/bias/v/Read/ReadVariableOpConst*,
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
__inference__traced_save_283575
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_16/lstm_cell_32/kernel%lstm_16/lstm_cell_32/recurrent_kernellstm_16/lstm_cell_32/biaslstm_17/lstm_cell_33/kernel%lstm_17/lstm_cell_33/recurrent_kernellstm_17/lstm_cell_33/biastotalcountAdam/dense_8/kernel/mAdam/dense_8/bias/m"Adam/lstm_16/lstm_cell_32/kernel/m,Adam/lstm_16/lstm_cell_32/recurrent_kernel/m Adam/lstm_16/lstm_cell_32/bias/m"Adam/lstm_17/lstm_cell_33/kernel/m,Adam/lstm_17/lstm_cell_33/recurrent_kernel/m Adam/lstm_17/lstm_cell_33/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v"Adam/lstm_16/lstm_cell_32/kernel/v,Adam/lstm_16/lstm_cell_32/recurrent_kernel/v Adam/lstm_16/lstm_cell_32/bias/v"Adam/lstm_17/lstm_cell_33/kernel/v,Adam/lstm_17/lstm_cell_33/recurrent_kernel/v Adam/lstm_17/lstm_cell_33/bias/v*+
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
"__inference__traced_restore_283678��#
�?
�
while_body_282485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
(__inference_lstm_17_layer_call_fn_282602

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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2806682
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
�[
�
C__inference_lstm_16_layer_call_and_return_conditional_losses_281089

inputs>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_281005*
condR
while_cond_281004*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�[
�
C__inference_lstm_17_layer_call_and_return_conditional_losses_283066

inputs>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_282982*
condR
while_cond_282981*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�%
�
while_body_279392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_32_279416_0:	�.
while_lstm_cell_32_279418_0:	@�*
while_lstm_cell_32_279420_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_32_279416:	�,
while_lstm_cell_32_279418:	@�(
while_lstm_cell_32_279420:	���*while/lstm_cell_32/StatefulPartitionedCall�
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
*while/lstm_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_32_279416_0while_lstm_cell_32_279418_0while_lstm_cell_32_279420_0*
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2793142,
*while/lstm_cell_32/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_32/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_32/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_32/StatefulPartitionedCall*"
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
while_lstm_cell_32_279416while_lstm_cell_32_279416_0"8
while_lstm_cell_32_279418while_lstm_cell_32_279418_0"8
while_lstm_cell_32_279420while_lstm_cell_32_279420_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_32/StatefulPartitionedCall*while/lstm_cell_32/StatefulPartitionedCall: 
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
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_281921

inputsF
3lstm_16_lstm_cell_32_matmul_readvariableop_resource:	�H
5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource:	@�C
4lstm_16_lstm_cell_32_biasadd_readvariableop_resource:	�F
3lstm_17_lstm_cell_33_matmul_readvariableop_resource:	@�H
5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource:	 �C
4lstm_17_lstm_cell_33_biasadd_readvariableop_resource:	�8
&dense_8_matmul_readvariableop_resource: 5
'dense_8_biasadd_readvariableop_resource:
identity��dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�*lstm_16/lstm_cell_32/MatMul/ReadVariableOp�,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�lstm_16/while�+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�*lstm_17/lstm_cell_33/MatMul/ReadVariableOp�,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�lstm_17/whileT
lstm_16/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_16/Shape�
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice/stack�
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_1�
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_2�
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slicel
lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros/mul/y�
lstm_16/zeros/mulMullstm_16/strided_slice:output:0lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/mulo
lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_16/zeros/Less/y�
lstm_16/zeros/LessLesslstm_16/zeros/mul:z:0lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/Lessr
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros/packed/1�
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros/packedo
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros/Const�
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/zerosp
lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros_1/mul/y�
lstm_16/zeros_1/mulMullstm_16/strided_slice:output:0lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/muls
lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_16/zeros_1/Less/y�
lstm_16/zeros_1/LessLesslstm_16/zeros_1/mul:z:0lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/Lessv
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros_1/packed/1�
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros_1/packeds
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros_1/Const�
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/zeros_1�
lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose/perm�
lstm_16/transpose	Transposeinputslstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_16/transposeg
lstm_16/Shape_1Shapelstm_16/transpose:y:0*
T0*
_output_shapes
:2
lstm_16/Shape_1�
lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_1/stack�
lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_1�
lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_2�
lstm_16/strided_slice_1StridedSlicelstm_16/Shape_1:output:0&lstm_16/strided_slice_1/stack:output:0(lstm_16/strided_slice_1/stack_1:output:0(lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slice_1�
#lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_16/TensorArrayV2/element_shape�
lstm_16/TensorArrayV2TensorListReserve,lstm_16/TensorArrayV2/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2�
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_16/transpose:y:0Flstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_16/TensorArrayUnstack/TensorListFromTensor�
lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_2/stack�
lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_1�
lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_2�
lstm_16/strided_slice_2StridedSlicelstm_16/transpose:y:0&lstm_16/strided_slice_2/stack:output:0(lstm_16/strided_slice_2/stack_1:output:0(lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_16/strided_slice_2�
*lstm_16/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3lstm_16_lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_16/lstm_cell_32/MatMul/ReadVariableOp�
lstm_16/lstm_cell_32/MatMulMatMul lstm_16/strided_slice_2:output:02lstm_16/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/MatMul�
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_16/lstm_cell_32/MatMul_1MatMullstm_16/zeros:output:04lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/MatMul_1�
lstm_16/lstm_cell_32/addAddV2%lstm_16/lstm_cell_32/MatMul:product:0'lstm_16/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/add�
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_16/lstm_cell_32/BiasAddBiasAddlstm_16/lstm_cell_32/add:z:03lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/BiasAdd�
$lstm_16/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_16/lstm_cell_32/split/split_dim�
lstm_16/lstm_cell_32/splitSplit-lstm_16/lstm_cell_32/split/split_dim:output:0%lstm_16/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_16/lstm_cell_32/split�
lstm_16/lstm_cell_32/SigmoidSigmoid#lstm_16/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Sigmoid�
lstm_16/lstm_cell_32/Sigmoid_1Sigmoid#lstm_16/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_16/lstm_cell_32/Sigmoid_1�
lstm_16/lstm_cell_32/mulMul"lstm_16/lstm_cell_32/Sigmoid_1:y:0lstm_16/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul�
lstm_16/lstm_cell_32/ReluRelu#lstm_16/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Relu�
lstm_16/lstm_cell_32/mul_1Mul lstm_16/lstm_cell_32/Sigmoid:y:0'lstm_16/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul_1�
lstm_16/lstm_cell_32/add_1AddV2lstm_16/lstm_cell_32/mul:z:0lstm_16/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/add_1�
lstm_16/lstm_cell_32/Sigmoid_2Sigmoid#lstm_16/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_16/lstm_cell_32/Sigmoid_2�
lstm_16/lstm_cell_32/Relu_1Relulstm_16/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Relu_1�
lstm_16/lstm_cell_32/mul_2Mul"lstm_16/lstm_cell_32/Sigmoid_2:y:0)lstm_16/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul_2�
%lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_16/TensorArrayV2_1/element_shape�
lstm_16/TensorArrayV2_1TensorListReserve.lstm_16/TensorArrayV2_1/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2_1^
lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/time�
 lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_16/while/maximum_iterationsz
lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/while/loop_counter�
lstm_16/whileWhile#lstm_16/while/loop_counter:output:0)lstm_16/while/maximum_iterations:output:0lstm_16/time:output:0 lstm_16/TensorArrayV2_1:handle:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0 lstm_16/strided_slice_1:output:0?lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_16_lstm_cell_32_matmul_readvariableop_resource5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource4lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
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
lstm_16_while_body_281676*%
condR
lstm_16_while_cond_281675*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_16/while�
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_16/TensorArrayV2Stack/TensorListStackTensorListStacklstm_16/while:output:3Alstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_16/TensorArrayV2Stack/TensorListStack�
lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_16/strided_slice_3/stack�
lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_16/strided_slice_3/stack_1�
lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_3/stack_2�
lstm_16/strided_slice_3StridedSlice3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_16/strided_slice_3/stack:output:0(lstm_16/strided_slice_3/stack_1:output:0(lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_16/strided_slice_3�
lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose_1/perm�
lstm_16/transpose_1	Transpose3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_16/transpose_1v
lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/runtimee
lstm_17/ShapeShapelstm_16/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_17/Shape�
lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice/stack�
lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_1�
lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_2�
lstm_17/strided_sliceStridedSlicelstm_17/Shape:output:0$lstm_17/strided_slice/stack:output:0&lstm_17/strided_slice/stack_1:output:0&lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slicel
lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros/mul/y�
lstm_17/zeros/mulMullstm_17/strided_slice:output:0lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/mulo
lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_17/zeros/Less/y�
lstm_17/zeros/LessLesslstm_17/zeros/mul:z:0lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/Lessr
lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros/packed/1�
lstm_17/zeros/packedPacklstm_17/strided_slice:output:0lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros/packedo
lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros/Const�
lstm_17/zerosFilllstm_17/zeros/packed:output:0lstm_17/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/zerosp
lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros_1/mul/y�
lstm_17/zeros_1/mulMullstm_17/strided_slice:output:0lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/muls
lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_17/zeros_1/Less/y�
lstm_17/zeros_1/LessLesslstm_17/zeros_1/mul:z:0lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/Lessv
lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros_1/packed/1�
lstm_17/zeros_1/packedPacklstm_17/strided_slice:output:0!lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros_1/packeds
lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros_1/Const�
lstm_17/zeros_1Filllstm_17/zeros_1/packed:output:0lstm_17/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/zeros_1�
lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose/perm�
lstm_17/transpose	Transposelstm_16/transpose_1:y:0lstm_17/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_17/transposeg
lstm_17/Shape_1Shapelstm_17/transpose:y:0*
T0*
_output_shapes
:2
lstm_17/Shape_1�
lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_1/stack�
lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_1�
lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_2�
lstm_17/strided_slice_1StridedSlicelstm_17/Shape_1:output:0&lstm_17/strided_slice_1/stack:output:0(lstm_17/strided_slice_1/stack_1:output:0(lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slice_1�
#lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_17/TensorArrayV2/element_shape�
lstm_17/TensorArrayV2TensorListReserve,lstm_17/TensorArrayV2/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2�
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_17/transpose:y:0Flstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_17/TensorArrayUnstack/TensorListFromTensor�
lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_2/stack�
lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_1�
lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_2�
lstm_17/strided_slice_2StridedSlicelstm_17/transpose:y:0&lstm_17/strided_slice_2/stack:output:0(lstm_17/strided_slice_2/stack_1:output:0(lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_17/strided_slice_2�
*lstm_17/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3lstm_17_lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_17/lstm_cell_33/MatMul/ReadVariableOp�
lstm_17/lstm_cell_33/MatMulMatMul lstm_17/strided_slice_2:output:02lstm_17/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/MatMul�
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_17/lstm_cell_33/MatMul_1MatMullstm_17/zeros:output:04lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/MatMul_1�
lstm_17/lstm_cell_33/addAddV2%lstm_17/lstm_cell_33/MatMul:product:0'lstm_17/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/add�
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_17/lstm_cell_33/BiasAddBiasAddlstm_17/lstm_cell_33/add:z:03lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/BiasAdd�
$lstm_17/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_17/lstm_cell_33/split/split_dim�
lstm_17/lstm_cell_33/splitSplit-lstm_17/lstm_cell_33/split/split_dim:output:0%lstm_17/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_17/lstm_cell_33/split�
lstm_17/lstm_cell_33/SigmoidSigmoid#lstm_17/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Sigmoid�
lstm_17/lstm_cell_33/Sigmoid_1Sigmoid#lstm_17/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_17/lstm_cell_33/Sigmoid_1�
lstm_17/lstm_cell_33/mulMul"lstm_17/lstm_cell_33/Sigmoid_1:y:0lstm_17/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul�
lstm_17/lstm_cell_33/ReluRelu#lstm_17/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Relu�
lstm_17/lstm_cell_33/mul_1Mul lstm_17/lstm_cell_33/Sigmoid:y:0'lstm_17/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul_1�
lstm_17/lstm_cell_33/add_1AddV2lstm_17/lstm_cell_33/mul:z:0lstm_17/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/add_1�
lstm_17/lstm_cell_33/Sigmoid_2Sigmoid#lstm_17/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_17/lstm_cell_33/Sigmoid_2�
lstm_17/lstm_cell_33/Relu_1Relulstm_17/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Relu_1�
lstm_17/lstm_cell_33/mul_2Mul"lstm_17/lstm_cell_33/Sigmoid_2:y:0)lstm_17/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul_2�
%lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_17/TensorArrayV2_1/element_shape�
lstm_17/TensorArrayV2_1TensorListReserve.lstm_17/TensorArrayV2_1/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2_1^
lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/time�
 lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_17/while/maximum_iterationsz
lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/while/loop_counter�
lstm_17/whileWhile#lstm_17/while/loop_counter:output:0)lstm_17/while/maximum_iterations:output:0lstm_17/time:output:0 lstm_17/TensorArrayV2_1:handle:0lstm_17/zeros:output:0lstm_17/zeros_1:output:0 lstm_17/strided_slice_1:output:0?lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_17_lstm_cell_33_matmul_readvariableop_resource5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource4lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
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
lstm_17_while_body_281823*%
condR
lstm_17_while_cond_281822*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_17/while�
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_17/TensorArrayV2Stack/TensorListStackTensorListStacklstm_17/while:output:3Alstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_17/TensorArrayV2Stack/TensorListStack�
lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_17/strided_slice_3/stack�
lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_17/strided_slice_3/stack_1�
lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_3/stack_2�
lstm_17/strided_slice_3StridedSlice3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_17/strided_slice_3/stack:output:0(lstm_17/strided_slice_3/stack_1:output:0(lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_17/strided_slice_3�
lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose_1/perm�
lstm_17/transpose_1	Transpose3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_17/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_17/transpose_1v
lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/runtimew
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout_8/dropout/Const�
dropout_8/dropout/MulMul lstm_17/strided_slice_3:output:0 dropout_8/dropout/Const:output:0*
T0*'
_output_shapes
:��������� 2
dropout_8/dropout/Mul�
dropout_8/dropout/ShapeShape lstm_17/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/Shape�
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*'
_output_shapes
:��������� *
dtype020
.dropout_8/dropout/random_uniform/RandomUniform�
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>2"
 dropout_8/dropout/GreaterEqual/y�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:��������� 2 
dropout_8/dropout/GreaterEqual�
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:��������� 2
dropout_8/dropout/Cast�
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*'
_output_shapes
:��������� 2
dropout_8/dropout/Mul_1�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMuldropout_8/dropout/Mul_1:z:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp,^lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp+^lstm_16/lstm_cell_32/MatMul/ReadVariableOp-^lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp^lstm_16/while,^lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp+^lstm_17/lstm_cell_33/MatMul/ReadVariableOp-^lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp^lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2Z
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp2X
*lstm_16/lstm_cell_32/MatMul/ReadVariableOp*lstm_16/lstm_cell_32/MatMul/ReadVariableOp2\
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp2
lstm_16/whilelstm_16/while2Z
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp2X
*lstm_17/lstm_cell_33/MatMul/ReadVariableOp*lstm_17/lstm_cell_33/MatMul/ReadVariableOp2\
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp2
lstm_17/whilelstm_17/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
��
�	
!__inference__wrapped_model_279093
lstm_16_inputS
@sequential_8_lstm_16_lstm_cell_32_matmul_readvariableop_resource:	�U
Bsequential_8_lstm_16_lstm_cell_32_matmul_1_readvariableop_resource:	@�P
Asequential_8_lstm_16_lstm_cell_32_biasadd_readvariableop_resource:	�S
@sequential_8_lstm_17_lstm_cell_33_matmul_readvariableop_resource:	@�U
Bsequential_8_lstm_17_lstm_cell_33_matmul_1_readvariableop_resource:	 �P
Asequential_8_lstm_17_lstm_cell_33_biasadd_readvariableop_resource:	�E
3sequential_8_dense_8_matmul_readvariableop_resource: B
4sequential_8_dense_8_biasadd_readvariableop_resource:
identity��+sequential_8/dense_8/BiasAdd/ReadVariableOp�*sequential_8/dense_8/MatMul/ReadVariableOp�8sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�7sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp�9sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�sequential_8/lstm_16/while�8sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�7sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp�9sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�sequential_8/lstm_17/whileu
sequential_8/lstm_16/ShapeShapelstm_16_input*
T0*
_output_shapes
:2
sequential_8/lstm_16/Shape�
(sequential_8/lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/lstm_16/strided_slice/stack�
*sequential_8/lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_16/strided_slice/stack_1�
*sequential_8/lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_16/strided_slice/stack_2�
"sequential_8/lstm_16/strided_sliceStridedSlice#sequential_8/lstm_16/Shape:output:01sequential_8/lstm_16/strided_slice/stack:output:03sequential_8/lstm_16/strided_slice/stack_1:output:03sequential_8/lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_8/lstm_16/strided_slice�
 sequential_8/lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2"
 sequential_8/lstm_16/zeros/mul/y�
sequential_8/lstm_16/zeros/mulMul+sequential_8/lstm_16/strided_slice:output:0)sequential_8/lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_16/zeros/mul�
!sequential_8/lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_8/lstm_16/zeros/Less/y�
sequential_8/lstm_16/zeros/LessLess"sequential_8/lstm_16/zeros/mul:z:0*sequential_8/lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_8/lstm_16/zeros/Less�
#sequential_8/lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2%
#sequential_8/lstm_16/zeros/packed/1�
!sequential_8/lstm_16/zeros/packedPack+sequential_8/lstm_16/strided_slice:output:0,sequential_8/lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_8/lstm_16/zeros/packed�
 sequential_8/lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_8/lstm_16/zeros/Const�
sequential_8/lstm_16/zerosFill*sequential_8/lstm_16/zeros/packed:output:0)sequential_8/lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_8/lstm_16/zeros�
"sequential_8/lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_8/lstm_16/zeros_1/mul/y�
 sequential_8/lstm_16/zeros_1/mulMul+sequential_8/lstm_16/strided_slice:output:0+sequential_8/lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_16/zeros_1/mul�
#sequential_8/lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2%
#sequential_8/lstm_16/zeros_1/Less/y�
!sequential_8/lstm_16/zeros_1/LessLess$sequential_8/lstm_16/zeros_1/mul:z:0,sequential_8/lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_8/lstm_16/zeros_1/Less�
%sequential_8/lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_8/lstm_16/zeros_1/packed/1�
#sequential_8/lstm_16/zeros_1/packedPack+sequential_8/lstm_16/strided_slice:output:0.sequential_8/lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_8/lstm_16/zeros_1/packed�
"sequential_8/lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_8/lstm_16/zeros_1/Const�
sequential_8/lstm_16/zeros_1Fill,sequential_8/lstm_16/zeros_1/packed:output:0+sequential_8/lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
sequential_8/lstm_16/zeros_1�
#sequential_8/lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_8/lstm_16/transpose/perm�
sequential_8/lstm_16/transpose	Transposelstm_16_input,sequential_8/lstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:���������2 
sequential_8/lstm_16/transpose�
sequential_8/lstm_16/Shape_1Shape"sequential_8/lstm_16/transpose:y:0*
T0*
_output_shapes
:2
sequential_8/lstm_16/Shape_1�
*sequential_8/lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_16/strided_slice_1/stack�
,sequential_8/lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_1/stack_1�
,sequential_8/lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_1/stack_2�
$sequential_8/lstm_16/strided_slice_1StridedSlice%sequential_8/lstm_16/Shape_1:output:03sequential_8/lstm_16/strided_slice_1/stack:output:05sequential_8/lstm_16/strided_slice_1/stack_1:output:05sequential_8/lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_1�
0sequential_8/lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_8/lstm_16/TensorArrayV2/element_shape�
"sequential_8/lstm_16/TensorArrayV2TensorListReserve9sequential_8/lstm_16/TensorArrayV2/element_shape:output:0-sequential_8/lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_8/lstm_16/TensorArrayV2�
Jsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2L
Jsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_8/lstm_16/transpose:y:0Ssequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor�
*sequential_8/lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_16/strided_slice_2/stack�
,sequential_8/lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_2/stack_1�
,sequential_8/lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_2/stack_2�
$sequential_8/lstm_16/strided_slice_2StridedSlice"sequential_8/lstm_16/transpose:y:03sequential_8/lstm_16/strided_slice_2/stack:output:05sequential_8/lstm_16/strided_slice_2/stack_1:output:05sequential_8/lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_2�
7sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp@sequential_8_lstm_16_lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype029
7sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp�
(sequential_8/lstm_16/lstm_cell_32/MatMulMatMul-sequential_8/lstm_16/strided_slice_2:output:0?sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(sequential_8/lstm_16/lstm_cell_32/MatMul�
9sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOpBsequential_8_lstm_16_lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02;
9sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�
*sequential_8/lstm_16/lstm_cell_32/MatMul_1MatMul#sequential_8/lstm_16/zeros:output:0Asequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_8/lstm_16/lstm_cell_32/MatMul_1�
%sequential_8/lstm_16/lstm_cell_32/addAddV22sequential_8/lstm_16/lstm_cell_32/MatMul:product:04sequential_8/lstm_16/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%sequential_8/lstm_16/lstm_cell_32/add�
8sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOpAsequential_8_lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�
)sequential_8/lstm_16/lstm_cell_32/BiasAddBiasAdd)sequential_8/lstm_16/lstm_cell_32/add:z:0@sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_8/lstm_16/lstm_cell_32/BiasAdd�
1sequential_8/lstm_16/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_8/lstm_16/lstm_cell_32/split/split_dim�
'sequential_8/lstm_16/lstm_cell_32/splitSplit:sequential_8/lstm_16/lstm_cell_32/split/split_dim:output:02sequential_8/lstm_16/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2)
'sequential_8/lstm_16/lstm_cell_32/split�
)sequential_8/lstm_16/lstm_cell_32/SigmoidSigmoid0sequential_8/lstm_16/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2+
)sequential_8/lstm_16/lstm_cell_32/Sigmoid�
+sequential_8/lstm_16/lstm_cell_32/Sigmoid_1Sigmoid0sequential_8/lstm_16/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2-
+sequential_8/lstm_16/lstm_cell_32/Sigmoid_1�
%sequential_8/lstm_16/lstm_cell_32/mulMul/sequential_8/lstm_16/lstm_cell_32/Sigmoid_1:y:0%sequential_8/lstm_16/zeros_1:output:0*
T0*'
_output_shapes
:���������@2'
%sequential_8/lstm_16/lstm_cell_32/mul�
&sequential_8/lstm_16/lstm_cell_32/ReluRelu0sequential_8/lstm_16/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2(
&sequential_8/lstm_16/lstm_cell_32/Relu�
'sequential_8/lstm_16/lstm_cell_32/mul_1Mul-sequential_8/lstm_16/lstm_cell_32/Sigmoid:y:04sequential_8/lstm_16/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2)
'sequential_8/lstm_16/lstm_cell_32/mul_1�
'sequential_8/lstm_16/lstm_cell_32/add_1AddV2)sequential_8/lstm_16/lstm_cell_32/mul:z:0+sequential_8/lstm_16/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2)
'sequential_8/lstm_16/lstm_cell_32/add_1�
+sequential_8/lstm_16/lstm_cell_32/Sigmoid_2Sigmoid0sequential_8/lstm_16/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2-
+sequential_8/lstm_16/lstm_cell_32/Sigmoid_2�
(sequential_8/lstm_16/lstm_cell_32/Relu_1Relu+sequential_8/lstm_16/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2*
(sequential_8/lstm_16/lstm_cell_32/Relu_1�
'sequential_8/lstm_16/lstm_cell_32/mul_2Mul/sequential_8/lstm_16/lstm_cell_32/Sigmoid_2:y:06sequential_8/lstm_16/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2)
'sequential_8/lstm_16/lstm_cell_32/mul_2�
2sequential_8/lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   24
2sequential_8/lstm_16/TensorArrayV2_1/element_shape�
$sequential_8/lstm_16/TensorArrayV2_1TensorListReserve;sequential_8/lstm_16/TensorArrayV2_1/element_shape:output:0-sequential_8/lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_8/lstm_16/TensorArrayV2_1x
sequential_8/lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_8/lstm_16/time�
-sequential_8/lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_8/lstm_16/while/maximum_iterations�
'sequential_8/lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_8/lstm_16/while/loop_counter�
sequential_8/lstm_16/whileWhile0sequential_8/lstm_16/while/loop_counter:output:06sequential_8/lstm_16/while/maximum_iterations:output:0"sequential_8/lstm_16/time:output:0-sequential_8/lstm_16/TensorArrayV2_1:handle:0#sequential_8/lstm_16/zeros:output:0%sequential_8/lstm_16/zeros_1:output:0-sequential_8/lstm_16/strided_slice_1:output:0Lsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_8_lstm_16_lstm_cell_32_matmul_readvariableop_resourceBsequential_8_lstm_16_lstm_cell_32_matmul_1_readvariableop_resourceAsequential_8_lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
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
&sequential_8_lstm_16_while_body_278855*2
cond*R(
&sequential_8_lstm_16_while_cond_278854*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
sequential_8/lstm_16/while�
Esequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2G
Esequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_8/lstm_16/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_8/lstm_16/while:output:3Nsequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype029
7sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack�
*sequential_8/lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_8/lstm_16/strided_slice_3/stack�
,sequential_8/lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_8/lstm_16/strided_slice_3/stack_1�
,sequential_8/lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_3/stack_2�
$sequential_8/lstm_16/strided_slice_3StridedSlice@sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack:tensor:03sequential_8/lstm_16/strided_slice_3/stack:output:05sequential_8/lstm_16/strided_slice_3/stack_1:output:05sequential_8/lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_3�
%sequential_8/lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_8/lstm_16/transpose_1/perm�
 sequential_8/lstm_16/transpose_1	Transpose@sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_8/lstm_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2"
 sequential_8/lstm_16/transpose_1�
sequential_8/lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_8/lstm_16/runtime�
sequential_8/lstm_17/ShapeShape$sequential_8/lstm_16/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_8/lstm_17/Shape�
(sequential_8/lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/lstm_17/strided_slice/stack�
*sequential_8/lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_17/strided_slice/stack_1�
*sequential_8/lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_17/strided_slice/stack_2�
"sequential_8/lstm_17/strided_sliceStridedSlice#sequential_8/lstm_17/Shape:output:01sequential_8/lstm_17/strided_slice/stack:output:03sequential_8/lstm_17/strided_slice/stack_1:output:03sequential_8/lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_8/lstm_17/strided_slice�
 sequential_8/lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 sequential_8/lstm_17/zeros/mul/y�
sequential_8/lstm_17/zeros/mulMul+sequential_8/lstm_17/strided_slice:output:0)sequential_8/lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_17/zeros/mul�
!sequential_8/lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2#
!sequential_8/lstm_17/zeros/Less/y�
sequential_8/lstm_17/zeros/LessLess"sequential_8/lstm_17/zeros/mul:z:0*sequential_8/lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_8/lstm_17/zeros/Less�
#sequential_8/lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_8/lstm_17/zeros/packed/1�
!sequential_8/lstm_17/zeros/packedPack+sequential_8/lstm_17/strided_slice:output:0,sequential_8/lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_8/lstm_17/zeros/packed�
 sequential_8/lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_8/lstm_17/zeros/Const�
sequential_8/lstm_17/zerosFill*sequential_8/lstm_17/zeros/packed:output:0)sequential_8/lstm_17/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_8/lstm_17/zeros�
"sequential_8/lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_8/lstm_17/zeros_1/mul/y�
 sequential_8/lstm_17/zeros_1/mulMul+sequential_8/lstm_17/strided_slice:output:0+sequential_8/lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_17/zeros_1/mul�
#sequential_8/lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2%
#sequential_8/lstm_17/zeros_1/Less/y�
!sequential_8/lstm_17/zeros_1/LessLess$sequential_8/lstm_17/zeros_1/mul:z:0,sequential_8/lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_8/lstm_17/zeros_1/Less�
%sequential_8/lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_8/lstm_17/zeros_1/packed/1�
#sequential_8/lstm_17/zeros_1/packedPack+sequential_8/lstm_17/strided_slice:output:0.sequential_8/lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_8/lstm_17/zeros_1/packed�
"sequential_8/lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_8/lstm_17/zeros_1/Const�
sequential_8/lstm_17/zeros_1Fill,sequential_8/lstm_17/zeros_1/packed:output:0+sequential_8/lstm_17/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential_8/lstm_17/zeros_1�
#sequential_8/lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_8/lstm_17/transpose/perm�
sequential_8/lstm_17/transpose	Transpose$sequential_8/lstm_16/transpose_1:y:0,sequential_8/lstm_17/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2 
sequential_8/lstm_17/transpose�
sequential_8/lstm_17/Shape_1Shape"sequential_8/lstm_17/transpose:y:0*
T0*
_output_shapes
:2
sequential_8/lstm_17/Shape_1�
*sequential_8/lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_17/strided_slice_1/stack�
,sequential_8/lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_1/stack_1�
,sequential_8/lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_1/stack_2�
$sequential_8/lstm_17/strided_slice_1StridedSlice%sequential_8/lstm_17/Shape_1:output:03sequential_8/lstm_17/strided_slice_1/stack:output:05sequential_8/lstm_17/strided_slice_1/stack_1:output:05sequential_8/lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_1�
0sequential_8/lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential_8/lstm_17/TensorArrayV2/element_shape�
"sequential_8/lstm_17/TensorArrayV2TensorListReserve9sequential_8/lstm_17/TensorArrayV2/element_shape:output:0-sequential_8/lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_8/lstm_17/TensorArrayV2�
Jsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2L
Jsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape�
<sequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_8/lstm_17/transpose:y:0Ssequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor�
*sequential_8/lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_17/strided_slice_2/stack�
,sequential_8/lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_2/stack_1�
,sequential_8/lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_2/stack_2�
$sequential_8/lstm_17/strided_slice_2StridedSlice"sequential_8/lstm_17/transpose:y:03sequential_8/lstm_17/strided_slice_2/stack:output:05sequential_8/lstm_17/strided_slice_2/stack_1:output:05sequential_8/lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_2�
7sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp@sequential_8_lstm_17_lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype029
7sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp�
(sequential_8/lstm_17/lstm_cell_33/MatMulMatMul-sequential_8/lstm_17/strided_slice_2:output:0?sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2*
(sequential_8/lstm_17/lstm_cell_33/MatMul�
9sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOpBsequential_8_lstm_17_lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02;
9sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�
*sequential_8/lstm_17/lstm_cell_33/MatMul_1MatMul#sequential_8/lstm_17/zeros:output:0Asequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2,
*sequential_8/lstm_17/lstm_cell_33/MatMul_1�
%sequential_8/lstm_17/lstm_cell_33/addAddV22sequential_8/lstm_17/lstm_cell_33/MatMul:product:04sequential_8/lstm_17/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2'
%sequential_8/lstm_17/lstm_cell_33/add�
8sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOpAsequential_8_lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�
)sequential_8/lstm_17/lstm_cell_33/BiasAddBiasAdd)sequential_8/lstm_17/lstm_cell_33/add:z:0@sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2+
)sequential_8/lstm_17/lstm_cell_33/BiasAdd�
1sequential_8/lstm_17/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_8/lstm_17/lstm_cell_33/split/split_dim�
'sequential_8/lstm_17/lstm_cell_33/splitSplit:sequential_8/lstm_17/lstm_cell_33/split/split_dim:output:02sequential_8/lstm_17/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2)
'sequential_8/lstm_17/lstm_cell_33/split�
)sequential_8/lstm_17/lstm_cell_33/SigmoidSigmoid0sequential_8/lstm_17/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2+
)sequential_8/lstm_17/lstm_cell_33/Sigmoid�
+sequential_8/lstm_17/lstm_cell_33/Sigmoid_1Sigmoid0sequential_8/lstm_17/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2-
+sequential_8/lstm_17/lstm_cell_33/Sigmoid_1�
%sequential_8/lstm_17/lstm_cell_33/mulMul/sequential_8/lstm_17/lstm_cell_33/Sigmoid_1:y:0%sequential_8/lstm_17/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2'
%sequential_8/lstm_17/lstm_cell_33/mul�
&sequential_8/lstm_17/lstm_cell_33/ReluRelu0sequential_8/lstm_17/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2(
&sequential_8/lstm_17/lstm_cell_33/Relu�
'sequential_8/lstm_17/lstm_cell_33/mul_1Mul-sequential_8/lstm_17/lstm_cell_33/Sigmoid:y:04sequential_8/lstm_17/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2)
'sequential_8/lstm_17/lstm_cell_33/mul_1�
'sequential_8/lstm_17/lstm_cell_33/add_1AddV2)sequential_8/lstm_17/lstm_cell_33/mul:z:0+sequential_8/lstm_17/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2)
'sequential_8/lstm_17/lstm_cell_33/add_1�
+sequential_8/lstm_17/lstm_cell_33/Sigmoid_2Sigmoid0sequential_8/lstm_17/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2-
+sequential_8/lstm_17/lstm_cell_33/Sigmoid_2�
(sequential_8/lstm_17/lstm_cell_33/Relu_1Relu+sequential_8/lstm_17/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2*
(sequential_8/lstm_17/lstm_cell_33/Relu_1�
'sequential_8/lstm_17/lstm_cell_33/mul_2Mul/sequential_8/lstm_17/lstm_cell_33/Sigmoid_2:y:06sequential_8/lstm_17/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2)
'sequential_8/lstm_17/lstm_cell_33/mul_2�
2sequential_8/lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    24
2sequential_8/lstm_17/TensorArrayV2_1/element_shape�
$sequential_8/lstm_17/TensorArrayV2_1TensorListReserve;sequential_8/lstm_17/TensorArrayV2_1/element_shape:output:0-sequential_8/lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_8/lstm_17/TensorArrayV2_1x
sequential_8/lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_8/lstm_17/time�
-sequential_8/lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2/
-sequential_8/lstm_17/while/maximum_iterations�
'sequential_8/lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_8/lstm_17/while/loop_counter�
sequential_8/lstm_17/whileWhile0sequential_8/lstm_17/while/loop_counter:output:06sequential_8/lstm_17/while/maximum_iterations:output:0"sequential_8/lstm_17/time:output:0-sequential_8/lstm_17/TensorArrayV2_1:handle:0#sequential_8/lstm_17/zeros:output:0%sequential_8/lstm_17/zeros_1:output:0-sequential_8/lstm_17/strided_slice_1:output:0Lsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_8_lstm_17_lstm_cell_33_matmul_readvariableop_resourceBsequential_8_lstm_17_lstm_cell_33_matmul_1_readvariableop_resourceAsequential_8_lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
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
&sequential_8_lstm_17_while_body_279002*2
cond*R(
&sequential_8_lstm_17_while_cond_279001*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
sequential_8/lstm_17/while�
Esequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2G
Esequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shape�
7sequential_8/lstm_17/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_8/lstm_17/while:output:3Nsequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype029
7sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack�
*sequential_8/lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2,
*sequential_8/lstm_17/strided_slice_3/stack�
,sequential_8/lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_8/lstm_17/strided_slice_3/stack_1�
,sequential_8/lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_3/stack_2�
$sequential_8/lstm_17/strided_slice_3StridedSlice@sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack:tensor:03sequential_8/lstm_17/strided_slice_3/stack:output:05sequential_8/lstm_17/strided_slice_3/stack_1:output:05sequential_8/lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_3�
%sequential_8/lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_8/lstm_17/transpose_1/perm�
 sequential_8/lstm_17/transpose_1	Transpose@sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_8/lstm_17/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2"
 sequential_8/lstm_17/transpose_1�
sequential_8/lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_8/lstm_17/runtime�
sequential_8/dropout_8/IdentityIdentity-sequential_8/lstm_17/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2!
sequential_8/dropout_8/Identity�
*sequential_8/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_8_dense_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_8/dense_8/MatMul/ReadVariableOp�
sequential_8/dense_8/MatMulMatMul(sequential_8/dropout_8/Identity:output:02sequential_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_8/dense_8/MatMul�
+sequential_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_8/dense_8/BiasAdd/ReadVariableOp�
sequential_8/dense_8/BiasAddBiasAdd%sequential_8/dense_8/MatMul:product:03sequential_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential_8/dense_8/BiasAdd�
IdentityIdentity%sequential_8/dense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp,^sequential_8/dense_8/BiasAdd/ReadVariableOp+^sequential_8/dense_8/MatMul/ReadVariableOp9^sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp8^sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp:^sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp^sequential_8/lstm_16/while9^sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp8^sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp:^sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp^sequential_8/lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2Z
+sequential_8/dense_8/BiasAdd/ReadVariableOp+sequential_8/dense_8/BiasAdd/ReadVariableOp2X
*sequential_8/dense_8/MatMul/ReadVariableOp*sequential_8/dense_8/MatMul/ReadVariableOp2t
8sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp8sequential_8/lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp2r
7sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp7sequential_8/lstm_16/lstm_cell_32/MatMul/ReadVariableOp2v
9sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp9sequential_8/lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp28
sequential_8/lstm_16/whilesequential_8/lstm_16/while2t
8sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp8sequential_8/lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp2r
7sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp7sequential_8/lstm_17/lstm_cell_33/MatMul/ReadVariableOp2v
9sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp9sequential_8/lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp28
sequential_8/lstm_17/whilesequential_8/lstm_17/while:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_16_input
�
�
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_279944

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
�
�
"__inference__traced_restore_283678
file_prefix1
assignvariableop_dense_8_kernel: -
assignvariableop_1_dense_8_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_16_lstm_cell_32_kernel:	�K
8assignvariableop_8_lstm_16_lstm_cell_32_recurrent_kernel:	@�;
,assignvariableop_9_lstm_16_lstm_cell_32_bias:	�B
/assignvariableop_10_lstm_17_lstm_cell_33_kernel:	@�L
9assignvariableop_11_lstm_17_lstm_cell_33_recurrent_kernel:	 �<
-assignvariableop_12_lstm_17_lstm_cell_33_bias:	�#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_8_kernel_m: 5
'assignvariableop_16_adam_dense_8_bias_m:I
6assignvariableop_17_adam_lstm_16_lstm_cell_32_kernel_m:	�S
@assignvariableop_18_adam_lstm_16_lstm_cell_32_recurrent_kernel_m:	@�C
4assignvariableop_19_adam_lstm_16_lstm_cell_32_bias_m:	�I
6assignvariableop_20_adam_lstm_17_lstm_cell_33_kernel_m:	@�S
@assignvariableop_21_adam_lstm_17_lstm_cell_33_recurrent_kernel_m:	 �C
4assignvariableop_22_adam_lstm_17_lstm_cell_33_bias_m:	�;
)assignvariableop_23_adam_dense_8_kernel_v: 5
'assignvariableop_24_adam_dense_8_bias_v:I
6assignvariableop_25_adam_lstm_16_lstm_cell_32_kernel_v:	�S
@assignvariableop_26_adam_lstm_16_lstm_cell_32_recurrent_kernel_v:	@�C
4assignvariableop_27_adam_lstm_16_lstm_cell_32_bias_v:	�I
6assignvariableop_28_adam_lstm_17_lstm_cell_33_kernel_v:	@�S
@assignvariableop_29_adam_lstm_17_lstm_cell_33_recurrent_kernel_v:	 �C
4assignvariableop_30_adam_lstm_17_lstm_cell_33_bias_v:	�
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
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_16_lstm_cell_32_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_16_lstm_cell_32_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_16_lstm_cell_32_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_17_lstm_cell_33_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_17_lstm_cell_33_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_17_lstm_cell_33_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_8_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_8_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_lstm_16_lstm_cell_32_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp@assignvariableop_18_adam_lstm_16_lstm_cell_32_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_16_lstm_cell_32_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_lstm_17_lstm_cell_33_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_lstm_17_lstm_cell_33_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_17_lstm_cell_33_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_8_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_8_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_lstm_16_lstm_cell_32_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp@assignvariableop_26_adam_lstm_16_lstm_cell_32_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_16_lstm_cell_32_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_adam_lstm_17_lstm_cell_33_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_lstm_17_lstm_cell_33_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_17_lstm_cell_33_bias_vIdentity_30:output:0"/device:CPU:0*
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
�I
�
__inference__traced_save_283575
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_16_lstm_cell_32_kernel_read_readvariableopD
@savev2_lstm_16_lstm_cell_32_recurrent_kernel_read_readvariableop8
4savev2_lstm_16_lstm_cell_32_bias_read_readvariableop:
6savev2_lstm_17_lstm_cell_33_kernel_read_readvariableopD
@savev2_lstm_17_lstm_cell_33_recurrent_kernel_read_readvariableop8
4savev2_lstm_17_lstm_cell_33_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_32_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_32_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_32_bias_m_read_readvariableopA
=savev2_adam_lstm_17_lstm_cell_33_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_17_lstm_cell_33_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_17_lstm_cell_33_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_32_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_32_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_32_bias_v_read_readvariableopA
=savev2_adam_lstm_17_lstm_cell_33_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_17_lstm_cell_33_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_17_lstm_cell_33_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_16_lstm_cell_32_kernel_read_readvariableop@savev2_lstm_16_lstm_cell_32_recurrent_kernel_read_readvariableop4savev2_lstm_16_lstm_cell_32_bias_read_readvariableop6savev2_lstm_17_lstm_cell_33_kernel_read_readvariableop@savev2_lstm_17_lstm_cell_33_recurrent_kernel_read_readvariableop4savev2_lstm_17_lstm_cell_33_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop=savev2_adam_lstm_16_lstm_cell_32_kernel_m_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_32_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_16_lstm_cell_32_bias_m_read_readvariableop=savev2_adam_lstm_17_lstm_cell_33_kernel_m_read_readvariableopGsavev2_adam_lstm_17_lstm_cell_33_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_17_lstm_cell_33_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop=savev2_adam_lstm_16_lstm_cell_32_kernel_v_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_32_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_16_lstm_cell_32_bias_v_read_readvariableop=savev2_adam_lstm_17_lstm_cell_33_kernel_v_read_readvariableopGsavev2_adam_lstm_17_lstm_cell_33_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_17_lstm_cell_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�?
�
while_body_280426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_280510

inputs>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_280426*
condR
while_cond_280425*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_280749

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
while_body_279812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_33_279836_0:	@�.
while_lstm_cell_33_279838_0:	 �*
while_lstm_cell_33_279840_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_33_279836:	@�,
while_lstm_cell_33_279838:	 �(
while_lstm_cell_33_279840:	���*while/lstm_cell_33/StatefulPartitionedCall�
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
*while/lstm_cell_33/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_33_279836_0while_lstm_cell_33_279838_0while_lstm_cell_33_279840_0*
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2797982,
*while/lstm_cell_33/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_33/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_33/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_33/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_33/StatefulPartitionedCall*"
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
while_lstm_cell_33_279836while_lstm_cell_33_279836_0"8
while_lstm_cell_33_279838while_lstm_cell_33_279838_0"8
while_lstm_cell_33_279840while_lstm_cell_33_279840_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_33/StatefulPartitionedCall*while/lstm_cell_33/StatefulPartitionedCall: 
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
while_body_280832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
lstm_16_while_cond_281675,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3.
*lstm_16_while_less_lstm_16_strided_slice_1D
@lstm_16_while_lstm_16_while_cond_281675___redundant_placeholder0D
@lstm_16_while_lstm_16_while_cond_281675___redundant_placeholder1D
@lstm_16_while_lstm_16_while_cond_281675___redundant_placeholder2D
@lstm_16_while_lstm_16_while_cond_281675___redundant_placeholder3
lstm_16_while_identity
�
lstm_16/while/LessLesslstm_16_while_placeholder*lstm_16_while_less_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2
lstm_16/while/Lessu
lstm_16/while/IdentityIdentitylstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_16/while/Identity"9
lstm_16_while_identitylstm_16/while/Identity:output:0*(
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
�
$__inference_signature_wrapper_281262
lstm_16_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
!__inference__wrapped_model_2790932
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
_user_specified_namelstm_16_input
�
�
(__inference_lstm_17_layer_call_fn_282591
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2800912
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
�
�
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_279314

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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282116
inputs_0>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_282032*
condR
while_cond_282031*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�?
�
while_body_280584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
while_cond_279391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_279391___redundant_placeholder04
0while_while_cond_279391___redundant_placeholder14
0while_while_cond_279391___redundant_placeholder24
0while_while_cond_279391___redundant_placeholder3
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
while_cond_282031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282031___redundant_placeholder04
0while_while_cond_282031___redundant_placeholder14
0while_while_cond_282031___redundant_placeholder24
0while_while_cond_282031___redundant_placeholder3
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
while_cond_282981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282981___redundant_placeholder04
0while_while_cond_282981___redundant_placeholder14
0while_while_cond_282981___redundant_placeholder24
0while_while_cond_282981___redundant_placeholder3
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
�
F
*__inference_dropout_8_layer_call_fn_283222

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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2806812
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
�
�
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283427

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
(__inference_lstm_17_layer_call_fn_282613

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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2809162
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
while_cond_283132
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_283132___redundant_placeholder04
0while_while_cond_283132___redundant_placeholder14
0while_while_cond_283132___redundant_placeholder24
0while_while_cond_283132___redundant_placeholder3
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
-__inference_lstm_cell_33_layer_call_fn_283395

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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2799442
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

�
-__inference_sequential_8_layer_call_fn_280719
lstm_16_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_2807002
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
_user_specified_namelstm_16_input
�
�
(__inference_lstm_17_layer_call_fn_282580
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2798812
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
�%
�
while_body_279182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_32_279206_0:	�.
while_lstm_cell_32_279208_0:	@�*
while_lstm_cell_32_279210_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_32_279206:	�,
while_lstm_cell_32_279208:	@�(
while_lstm_cell_32_279210:	���*while/lstm_cell_32/StatefulPartitionedCall�
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
*while/lstm_cell_32/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_32_279206_0while_lstm_cell_32_279208_0while_lstm_cell_32_279210_0*
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2791682,
*while/lstm_cell_32/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_32/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_32/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_32/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_32/StatefulPartitionedCall*"
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
while_lstm_cell_32_279206while_lstm_cell_32_279206_0"8
while_lstm_cell_32_279208while_lstm_cell_32_279208_0"8
while_lstm_cell_32_279210while_lstm_cell_32_279210_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2X
*while/lstm_cell_32/StatefulPartitionedCall*while/lstm_cell_32/StatefulPartitionedCall: 
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
�
�
(__inference_dense_8_layer_call_fn_283253

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
C__inference_dense_8_layer_call_and_return_conditional_losses_2806932
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
�F
�
C__inference_lstm_16_layer_call_and_return_conditional_losses_279251

inputs&
lstm_cell_32_279169:	�&
lstm_cell_32_279171:	@�"
lstm_cell_32_279173:	�
identity��$lstm_cell_32/StatefulPartitionedCall�whileD
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
$lstm_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_32_279169lstm_cell_32_279171lstm_cell_32_279173*
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2791682&
$lstm_cell_32/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_32_279169lstm_cell_32_279171lstm_cell_32_279173*
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
while_body_279182*
condR
while_cond_279181*K
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
NoOpNoOp%^lstm_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_32/StatefulPartitionedCall$lstm_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�J
�

lstm_17_while_body_281518,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3+
'lstm_17_while_lstm_17_strided_slice_1_0g
clstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0:	@�P
=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �K
<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
lstm_17_while_identity
lstm_17_while_identity_1
lstm_17_while_identity_2
lstm_17_while_identity_3
lstm_17_while_identity_4
lstm_17_while_identity_5)
%lstm_17_while_lstm_17_strided_slice_1e
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorL
9lstm_17_while_lstm_cell_33_matmul_readvariableop_resource:	@�N
;lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource:	 �I
:lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource:	���1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0lstm_17_while_placeholderHlstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_17/while/TensorArrayV2Read/TensorListGetItem�
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�
!lstm_17/while/lstm_cell_33/MatMulMatMul8lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_17/while/lstm_cell_33/MatMul�
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
#lstm_17/while/lstm_cell_33/MatMul_1MatMullstm_17_while_placeholder_2:lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_17/while/lstm_cell_33/MatMul_1�
lstm_17/while/lstm_cell_33/addAddV2+lstm_17/while/lstm_cell_33/MatMul:product:0-lstm_17/while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_17/while/lstm_cell_33/add�
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�
"lstm_17/while/lstm_cell_33/BiasAddBiasAdd"lstm_17/while/lstm_cell_33/add:z:09lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_17/while/lstm_cell_33/BiasAdd�
*lstm_17/while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_17/while/lstm_cell_33/split/split_dim�
 lstm_17/while/lstm_cell_33/splitSplit3lstm_17/while/lstm_cell_33/split/split_dim:output:0+lstm_17/while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_17/while/lstm_cell_33/split�
"lstm_17/while/lstm_cell_33/SigmoidSigmoid)lstm_17/while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_17/while/lstm_cell_33/Sigmoid�
$lstm_17/while/lstm_cell_33/Sigmoid_1Sigmoid)lstm_17/while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_17/while/lstm_cell_33/Sigmoid_1�
lstm_17/while/lstm_cell_33/mulMul(lstm_17/while/lstm_cell_33/Sigmoid_1:y:0lstm_17_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_17/while/lstm_cell_33/mul�
lstm_17/while/lstm_cell_33/ReluRelu)lstm_17/while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_17/while/lstm_cell_33/Relu�
 lstm_17/while/lstm_cell_33/mul_1Mul&lstm_17/while/lstm_cell_33/Sigmoid:y:0-lstm_17/while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/mul_1�
 lstm_17/while/lstm_cell_33/add_1AddV2"lstm_17/while/lstm_cell_33/mul:z:0$lstm_17/while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/add_1�
$lstm_17/while/lstm_cell_33/Sigmoid_2Sigmoid)lstm_17/while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_17/while/lstm_cell_33/Sigmoid_2�
!lstm_17/while/lstm_cell_33/Relu_1Relu$lstm_17/while/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_17/while/lstm_cell_33/Relu_1�
 lstm_17/while/lstm_cell_33/mul_2Mul(lstm_17/while/lstm_cell_33/Sigmoid_2:y:0/lstm_17/while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/mul_2�
2lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_17_while_placeholder_1lstm_17_while_placeholder$lstm_17/while/lstm_cell_33/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_17/while/TensorArrayV2Write/TensorListSetIteml
lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add/y�
lstm_17/while/addAddV2lstm_17_while_placeholderlstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/addp
lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add_1/y�
lstm_17/while/add_1AddV2(lstm_17_while_lstm_17_while_loop_counterlstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/add_1�
lstm_17/while/IdentityIdentitylstm_17/while/add_1:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity�
lstm_17/while/Identity_1Identity.lstm_17_while_lstm_17_while_maximum_iterations^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_1�
lstm_17/while/Identity_2Identitylstm_17/while/add:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_2�
lstm_17/while/Identity_3IdentityBlstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_3�
lstm_17/while/Identity_4Identity$lstm_17/while/lstm_cell_33/mul_2:z:0^lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_17/while/Identity_4�
lstm_17/while/Identity_5Identity$lstm_17/while/lstm_cell_33/add_1:z:0^lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_17/while/Identity_5�
lstm_17/while/NoOpNoOp2^lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp1^lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp3^lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_17/while/NoOp"9
lstm_17_while_identitylstm_17/while/Identity:output:0"=
lstm_17_while_identity_1!lstm_17/while/Identity_1:output:0"=
lstm_17_while_identity_2!lstm_17/while/Identity_2:output:0"=
lstm_17_while_identity_3!lstm_17/while/Identity_3:output:0"=
lstm_17_while_identity_4!lstm_17/while/Identity_4:output:0"=
lstm_17_while_identity_5!lstm_17/while/Identity_5:output:0"P
%lstm_17_while_lstm_17_strided_slice_1'lstm_17_while_lstm_17_strided_slice_1_0"z
:lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0"|
;lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0"x
9lstm_17_while_lstm_cell_33_matmul_readvariableop_resource;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0"�
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp2d
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp2h
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
while_cond_282484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282484___redundant_placeholder04
0while_while_cond_282484___redundant_placeholder14
0while_while_cond_282484___redundant_placeholder24
0while_while_cond_282484___redundant_placeholder3
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
d
E__inference_dropout_8_layer_call_and_return_conditional_losses_283244

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
C__inference_lstm_17_layer_call_and_return_conditional_losses_279881

inputs&
lstm_cell_33_279799:	@�&
lstm_cell_33_279801:	 �"
lstm_cell_33_279803:	�
identity��$lstm_cell_33/StatefulPartitionedCall�whileD
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
$lstm_cell_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_33_279799lstm_cell_33_279801lstm_cell_33_279803*
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2797982&
$lstm_cell_33/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_33_279799lstm_cell_33_279801lstm_cell_33_279803*
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
while_body_279812*
condR
while_cond_279811*K
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
NoOpNoOp%^lstm_cell_33/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_33/StatefulPartitionedCall$lstm_cell_33/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
(__inference_lstm_16_layer_call_fn_281954

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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2805102
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
while_cond_282830
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282830___redundant_placeholder04
0while_while_cond_282830___redundant_placeholder14
0while_while_cond_282830___redundant_placeholder24
0while_while_cond_282830___redundant_placeholder3
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_283217

inputs>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_283133*
condR
while_cond_283132*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�J
�

lstm_16_while_body_281676,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3+
'lstm_16_while_lstm_16_strided_slice_1_0g
clstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0:	�P
=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�K
<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
lstm_16_while_identity
lstm_16_while_identity_1
lstm_16_while_identity_2
lstm_16_while_identity_3
lstm_16_while_identity_4
lstm_16_while_identity_5)
%lstm_16_while_lstm_16_strided_slice_1e
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorL
9lstm_16_while_lstm_cell_32_matmul_readvariableop_resource:	�N
;lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource:	@�I
:lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource:	���1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0lstm_16_while_placeholderHlstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_16/while/TensorArrayV2Read/TensorListGetItem�
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�
!lstm_16/while/lstm_cell_32/MatMulMatMul8lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_16/while/lstm_cell_32/MatMul�
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
#lstm_16/while/lstm_cell_32/MatMul_1MatMullstm_16_while_placeholder_2:lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_16/while/lstm_cell_32/MatMul_1�
lstm_16/while/lstm_cell_32/addAddV2+lstm_16/while/lstm_cell_32/MatMul:product:0-lstm_16/while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_16/while/lstm_cell_32/add�
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�
"lstm_16/while/lstm_cell_32/BiasAddBiasAdd"lstm_16/while/lstm_cell_32/add:z:09lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_16/while/lstm_cell_32/BiasAdd�
*lstm_16/while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_16/while/lstm_cell_32/split/split_dim�
 lstm_16/while/lstm_cell_32/splitSplit3lstm_16/while/lstm_cell_32/split/split_dim:output:0+lstm_16/while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_16/while/lstm_cell_32/split�
"lstm_16/while/lstm_cell_32/SigmoidSigmoid)lstm_16/while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_16/while/lstm_cell_32/Sigmoid�
$lstm_16/while/lstm_cell_32/Sigmoid_1Sigmoid)lstm_16/while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_16/while/lstm_cell_32/Sigmoid_1�
lstm_16/while/lstm_cell_32/mulMul(lstm_16/while/lstm_cell_32/Sigmoid_1:y:0lstm_16_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_16/while/lstm_cell_32/mul�
lstm_16/while/lstm_cell_32/ReluRelu)lstm_16/while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_16/while/lstm_cell_32/Relu�
 lstm_16/while/lstm_cell_32/mul_1Mul&lstm_16/while/lstm_cell_32/Sigmoid:y:0-lstm_16/while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/mul_1�
 lstm_16/while/lstm_cell_32/add_1AddV2"lstm_16/while/lstm_cell_32/mul:z:0$lstm_16/while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/add_1�
$lstm_16/while/lstm_cell_32/Sigmoid_2Sigmoid)lstm_16/while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_16/while/lstm_cell_32/Sigmoid_2�
!lstm_16/while/lstm_cell_32/Relu_1Relu$lstm_16/while/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_16/while/lstm_cell_32/Relu_1�
 lstm_16/while/lstm_cell_32/mul_2Mul(lstm_16/while/lstm_cell_32/Sigmoid_2:y:0/lstm_16/while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/mul_2�
2lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_16_while_placeholder_1lstm_16_while_placeholder$lstm_16/while/lstm_cell_32/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_16/while/TensorArrayV2Write/TensorListSetIteml
lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add/y�
lstm_16/while/addAddV2lstm_16_while_placeholderlstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/addp
lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add_1/y�
lstm_16/while/add_1AddV2(lstm_16_while_lstm_16_while_loop_counterlstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/add_1�
lstm_16/while/IdentityIdentitylstm_16/while/add_1:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity�
lstm_16/while/Identity_1Identity.lstm_16_while_lstm_16_while_maximum_iterations^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_1�
lstm_16/while/Identity_2Identitylstm_16/while/add:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_2�
lstm_16/while/Identity_3IdentityBlstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_3�
lstm_16/while/Identity_4Identity$lstm_16/while/lstm_cell_32/mul_2:z:0^lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_16/while/Identity_4�
lstm_16/while/Identity_5Identity$lstm_16/while/lstm_cell_32/add_1:z:0^lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_16/while/Identity_5�
lstm_16/while/NoOpNoOp2^lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp1^lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp3^lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_16/while/NoOp"9
lstm_16_while_identitylstm_16/while/Identity:output:0"=
lstm_16_while_identity_1!lstm_16/while/Identity_1:output:0"=
lstm_16_while_identity_2!lstm_16/while/Identity_2:output:0"=
lstm_16_while_identity_3!lstm_16/while/Identity_3:output:0"=
lstm_16_while_identity_4!lstm_16/while/Identity_4:output:0"=
lstm_16_while_identity_5!lstm_16/while/Identity_5:output:0"P
%lstm_16_while_lstm_16_strided_slice_1'lstm_16_while_lstm_16_strided_slice_1_0"z
:lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0"|
;lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0"x
9lstm_16_while_lstm_cell_32_matmul_readvariableop_resource;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0"�
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp2d
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp2h
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
-__inference_lstm_cell_32_layer_call_fn_283297

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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2793142
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

�
-__inference_sequential_8_layer_call_fn_281304

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
H__inference_sequential_8_layer_call_and_return_conditional_losses_2811452
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
�?
�
while_body_282032
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_282334
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
�
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_281145

inputs!
lstm_16_281124:	�!
lstm_16_281126:	@�
lstm_16_281128:	�!
lstm_17_281131:	@�!
lstm_17_281133:	 �
lstm_17_281135:	� 
dense_8_281139: 
dense_8_281141:
identity��dense_8/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�lstm_16/StatefulPartitionedCall�lstm_17/StatefulPartitionedCall�
lstm_16/StatefulPartitionedCallStatefulPartitionedCallinputslstm_16_281124lstm_16_281126lstm_16_281128*
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2810892!
lstm_16/StatefulPartitionedCall�
lstm_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0lstm_17_281131lstm_17_281133lstm_17_281135*
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2809162!
lstm_17/StatefulPartitionedCall�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2807492#
!dropout_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_281139dense_8_281141*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_2806932!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_8/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_282183
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
&sequential_8_lstm_17_while_cond_279001F
Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counterL
Hsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations*
&sequential_8_lstm_17_while_placeholder,
(sequential_8_lstm_17_while_placeholder_1,
(sequential_8_lstm_17_while_placeholder_2,
(sequential_8_lstm_17_while_placeholder_3H
Dsequential_8_lstm_17_while_less_sequential_8_lstm_17_strided_slice_1^
Zsequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_279001___redundant_placeholder0^
Zsequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_279001___redundant_placeholder1^
Zsequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_279001___redundant_placeholder2^
Zsequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_279001___redundant_placeholder3'
#sequential_8_lstm_17_while_identity
�
sequential_8/lstm_17/while/LessLess&sequential_8_lstm_17_while_placeholderDsequential_8_lstm_17_while_less_sequential_8_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_8/lstm_17/while/Less�
#sequential_8/lstm_17/while/IdentityIdentity#sequential_8/lstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_8/lstm_17/while/Identity"S
#sequential_8_lstm_17_while_identity,sequential_8/lstm_17/while/Identity:output:0*(
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
while_cond_282182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282182___redundant_placeholder04
0while_while_cond_282182___redundant_placeholder14
0while_while_cond_282182___redundant_placeholder24
0while_while_cond_282182___redundant_placeholder3
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
lstm_17_while_cond_281517,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3.
*lstm_17_while_less_lstm_17_strided_slice_1D
@lstm_17_while_lstm_17_while_cond_281517___redundant_placeholder0D
@lstm_17_while_lstm_17_while_cond_281517___redundant_placeholder1D
@lstm_17_while_lstm_17_while_cond_281517___redundant_placeholder2D
@lstm_17_while_lstm_17_while_cond_281517___redundant_placeholder3
lstm_17_while_identity
�
lstm_17/while/LessLesslstm_17_while_placeholder*lstm_17_while_less_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2
lstm_17/while/Lessu
lstm_17/while/IdentityIdentitylstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_17/while/Identity"9
lstm_17_while_identitylstm_17/while/Identity:output:0*(
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_281209
lstm_16_input!
lstm_16_281188:	�!
lstm_16_281190:	@�
lstm_16_281192:	�!
lstm_17_281195:	@�!
lstm_17_281197:	 �
lstm_17_281199:	� 
dense_8_281203: 
dense_8_281205:
identity��dense_8/StatefulPartitionedCall�lstm_16/StatefulPartitionedCall�lstm_17/StatefulPartitionedCall�
lstm_16/StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputlstm_16_281188lstm_16_281190lstm_16_281192*
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2805102!
lstm_16/StatefulPartitionedCall�
lstm_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0lstm_17_281195lstm_17_281197lstm_17_281199*
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2806682!
lstm_17/StatefulPartitionedCall�
dropout_8/PartitionedCallPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2806812
dropout_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_281203dense_8_281205*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_2806932!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_16_input
�

�
lstm_16_while_cond_281370,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3.
*lstm_16_while_less_lstm_16_strided_slice_1D
@lstm_16_while_lstm_16_while_cond_281370___redundant_placeholder0D
@lstm_16_while_lstm_16_while_cond_281370___redundant_placeholder1D
@lstm_16_while_lstm_16_while_cond_281370___redundant_placeholder2D
@lstm_16_while_lstm_16_while_cond_281370___redundant_placeholder3
lstm_16_while_identity
�
lstm_16/while/LessLesslstm_16_while_placeholder*lstm_16_while_less_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2
lstm_16/while/Lessu
lstm_16/while/IdentityIdentitylstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_16/while/Identity"9
lstm_16_while_identitylstm_16/while/Identity:output:0*(
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
while_body_282680
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_279168

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
�
while_cond_279181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_279181___redundant_placeholder04
0while_while_cond_279181___redundant_placeholder14
0while_while_cond_279181___redundant_placeholder24
0while_while_cond_279181___redundant_placeholder3
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
&sequential_8_lstm_16_while_cond_278854F
Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counterL
Hsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations*
&sequential_8_lstm_16_while_placeholder,
(sequential_8_lstm_16_while_placeholder_1,
(sequential_8_lstm_16_while_placeholder_2,
(sequential_8_lstm_16_while_placeholder_3H
Dsequential_8_lstm_16_while_less_sequential_8_lstm_16_strided_slice_1^
Zsequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_278854___redundant_placeholder0^
Zsequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_278854___redundant_placeholder1^
Zsequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_278854___redundant_placeholder2^
Zsequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_278854___redundant_placeholder3'
#sequential_8_lstm_16_while_identity
�
sequential_8/lstm_16/while/LessLess&sequential_8_lstm_16_while_placeholderDsequential_8_lstm_16_while_less_sequential_8_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_8/lstm_16/while/Less�
#sequential_8/lstm_16/while/IdentityIdentity#sequential_8/lstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_8/lstm_16/while/Identity"S
#sequential_8_lstm_16_while_identity,sequential_8/lstm_16/while/Identity:output:0*(
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_280091

inputs&
lstm_cell_33_280009:	@�&
lstm_cell_33_280011:	 �"
lstm_cell_33_280013:	�
identity��$lstm_cell_33/StatefulPartitionedCall�whileD
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
$lstm_cell_33/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_33_280009lstm_cell_33_280011lstm_cell_33_280013*
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2799442&
$lstm_cell_33/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_33_280009lstm_cell_33_280011lstm_cell_33_280013*
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
while_body_280022*
condR
while_cond_280021*K
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
NoOpNoOp%^lstm_cell_33/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2L
$lstm_cell_33/StatefulPartitionedCall$lstm_cell_33/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_280700

inputs!
lstm_16_280511:	�!
lstm_16_280513:	@�
lstm_16_280515:	�!
lstm_17_280669:	@�!
lstm_17_280671:	 �
lstm_17_280673:	� 
dense_8_280694: 
dense_8_280696:
identity��dense_8/StatefulPartitionedCall�lstm_16/StatefulPartitionedCall�lstm_17/StatefulPartitionedCall�
lstm_16/StatefulPartitionedCallStatefulPartitionedCallinputslstm_16_280511lstm_16_280513lstm_16_280515*
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2805102!
lstm_16/StatefulPartitionedCall�
lstm_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0lstm_17_280669lstm_17_280671lstm_17_280673*
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2806682!
lstm_17/StatefulPartitionedCall�
dropout_8/PartitionedCallPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2806812
dropout_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0dense_8_280694dense_8_280696*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_2806932!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�?
�
while_body_281005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_32_matmul_readvariableop_resource_0:	�H
5while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�C
4while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_32_matmul_readvariableop_resource:	�F
3while_lstm_cell_32_matmul_1_readvariableop_resource:	@�A
2while_lstm_cell_32_biasadd_readvariableop_resource:	���)while/lstm_cell_32/BiasAdd/ReadVariableOp�(while/lstm_cell_32/MatMul/ReadVariableOp�*while/lstm_cell_32/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02*
(while/lstm_cell_32/MatMul/ReadVariableOp�
while/lstm_cell_32/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul�
*while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02,
*while/lstm_cell_32/MatMul_1/ReadVariableOp�
while/lstm_cell_32/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/MatMul_1�
while/lstm_cell_32/addAddV2#while/lstm_cell_32/MatMul:product:0%while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/add�
)while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_32/BiasAdd/ReadVariableOp�
while/lstm_cell_32/BiasAddBiasAddwhile/lstm_cell_32/add:z:01while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_32/BiasAdd�
"while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_32/split/split_dim�
while/lstm_cell_32/splitSplit+while/lstm_cell_32/split/split_dim:output:0#while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
while/lstm_cell_32/split�
while/lstm_cell_32/SigmoidSigmoid!while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid�
while/lstm_cell_32/Sigmoid_1Sigmoid!while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_1�
while/lstm_cell_32/mulMul while/lstm_cell_32/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul�
while/lstm_cell_32/ReluRelu!while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu�
while/lstm_cell_32/mul_1Mulwhile/lstm_cell_32/Sigmoid:y:0%while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_1�
while/lstm_cell_32/add_1AddV2while/lstm_cell_32/mul:z:0while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/add_1�
while/lstm_cell_32/Sigmoid_2Sigmoid!while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Sigmoid_2�
while/lstm_cell_32/Relu_1Reluwhile/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/Relu_1�
while/lstm_cell_32/mul_2Mul while/lstm_cell_32/Sigmoid_2:y:0'while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
while/lstm_cell_32/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_32/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_32/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_32/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:���������@2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_32/BiasAdd/ReadVariableOp)^while/lstm_cell_32/MatMul/ReadVariableOp+^while/lstm_cell_32/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_32_biasadd_readvariableop_resource4while_lstm_cell_32_biasadd_readvariableop_resource_0"l
3while_lstm_cell_32_matmul_1_readvariableop_resource5while_lstm_cell_32_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_32_matmul_readvariableop_resource3while_lstm_cell_32_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2V
)while/lstm_cell_32/BiasAdd/ReadVariableOp)while/lstm_cell_32/BiasAdd/ReadVariableOp2T
(while/lstm_cell_32/MatMul/ReadVariableOp(while/lstm_cell_32/MatMul/ReadVariableOp2X
*while/lstm_cell_32/MatMul_1/ReadVariableOp*while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
while_body_283133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
while_cond_282333
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282333___redundant_placeholder04
0while_while_cond_282333___redundant_placeholder14
0while_while_cond_282333___redundant_placeholder24
0while_while_cond_282333___redundant_placeholder3
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
while_body_282982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_280668

inputs>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_280584*
condR
while_cond_280583*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
while_cond_280831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_280831___redundant_placeholder04
0while_while_cond_280831___redundant_placeholder14
0while_while_cond_280831___redundant_placeholder24
0while_while_cond_280831___redundant_placeholder3
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_282915
inputs_0>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_282831*
condR
while_cond_282830*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_279811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_279811___redundant_placeholder04
0while_while_cond_279811___redundant_placeholder14
0while_while_cond_279811___redundant_placeholder24
0while_while_cond_279811___redundant_placeholder3
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
�
c
*__inference_dropout_8_layer_call_fn_283227

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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2807492
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
�
(__inference_lstm_16_layer_call_fn_281965

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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2810892
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
�
(__inference_lstm_16_layer_call_fn_281943
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2794612
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
��
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_281609

inputsF
3lstm_16_lstm_cell_32_matmul_readvariableop_resource:	�H
5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource:	@�C
4lstm_16_lstm_cell_32_biasadd_readvariableop_resource:	�F
3lstm_17_lstm_cell_33_matmul_readvariableop_resource:	@�H
5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource:	 �C
4lstm_17_lstm_cell_33_biasadd_readvariableop_resource:	�8
&dense_8_matmul_readvariableop_resource: 5
'dense_8_biasadd_readvariableop_resource:
identity��dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�*lstm_16/lstm_cell_32/MatMul/ReadVariableOp�,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�lstm_16/while�+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�*lstm_17/lstm_cell_33/MatMul/ReadVariableOp�,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�lstm_17/whileT
lstm_16/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_16/Shape�
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice/stack�
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_1�
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_2�
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slicel
lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros/mul/y�
lstm_16/zeros/mulMullstm_16/strided_slice:output:0lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/mulo
lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_16/zeros/Less/y�
lstm_16/zeros/LessLesslstm_16/zeros/mul:z:0lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/Lessr
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros/packed/1�
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros/packedo
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros/Const�
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/zerosp
lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros_1/mul/y�
lstm_16/zeros_1/mulMullstm_16/strided_slice:output:0lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/muls
lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_16/zeros_1/Less/y�
lstm_16/zeros_1/LessLesslstm_16/zeros_1/mul:z:0lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/Lessv
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_16/zeros_1/packed/1�
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros_1/packeds
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros_1/Const�
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/zeros_1�
lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose/perm�
lstm_16/transpose	Transposeinputslstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:���������2
lstm_16/transposeg
lstm_16/Shape_1Shapelstm_16/transpose:y:0*
T0*
_output_shapes
:2
lstm_16/Shape_1�
lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_1/stack�
lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_1�
lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_2�
lstm_16/strided_slice_1StridedSlicelstm_16/Shape_1:output:0&lstm_16/strided_slice_1/stack:output:0(lstm_16/strided_slice_1/stack_1:output:0(lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slice_1�
#lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_16/TensorArrayV2/element_shape�
lstm_16/TensorArrayV2TensorListReserve,lstm_16/TensorArrayV2/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2�
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_16/transpose:y:0Flstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_16/TensorArrayUnstack/TensorListFromTensor�
lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_2/stack�
lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_1�
lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_2�
lstm_16/strided_slice_2StridedSlicelstm_16/transpose:y:0&lstm_16/strided_slice_2/stack:output:0(lstm_16/strided_slice_2/stack_1:output:0(lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask2
lstm_16/strided_slice_2�
*lstm_16/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp3lstm_16_lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02,
*lstm_16/lstm_cell_32/MatMul/ReadVariableOp�
lstm_16/lstm_cell_32/MatMulMatMul lstm_16/strided_slice_2:output:02lstm_16/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/MatMul�
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02.
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_16/lstm_cell_32/MatMul_1MatMullstm_16/zeros:output:04lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/MatMul_1�
lstm_16/lstm_cell_32/addAddV2%lstm_16/lstm_cell_32/MatMul:product:0'lstm_16/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/add�
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp4lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_16/lstm_cell_32/BiasAddBiasAddlstm_16/lstm_cell_32/add:z:03lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_16/lstm_cell_32/BiasAdd�
$lstm_16/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_16/lstm_cell_32/split/split_dim�
lstm_16/lstm_cell_32/splitSplit-lstm_16/lstm_cell_32/split/split_dim:output:0%lstm_16/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_16/lstm_cell_32/split�
lstm_16/lstm_cell_32/SigmoidSigmoid#lstm_16/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Sigmoid�
lstm_16/lstm_cell_32/Sigmoid_1Sigmoid#lstm_16/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2 
lstm_16/lstm_cell_32/Sigmoid_1�
lstm_16/lstm_cell_32/mulMul"lstm_16/lstm_cell_32/Sigmoid_1:y:0lstm_16/zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul�
lstm_16/lstm_cell_32/ReluRelu#lstm_16/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Relu�
lstm_16/lstm_cell_32/mul_1Mul lstm_16/lstm_cell_32/Sigmoid:y:0'lstm_16/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul_1�
lstm_16/lstm_cell_32/add_1AddV2lstm_16/lstm_cell_32/mul:z:0lstm_16/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/add_1�
lstm_16/lstm_cell_32/Sigmoid_2Sigmoid#lstm_16/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2 
lstm_16/lstm_cell_32/Sigmoid_2�
lstm_16/lstm_cell_32/Relu_1Relulstm_16/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/Relu_1�
lstm_16/lstm_cell_32/mul_2Mul"lstm_16/lstm_cell_32/Sigmoid_2:y:0)lstm_16/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_16/lstm_cell_32/mul_2�
%lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2'
%lstm_16/TensorArrayV2_1/element_shape�
lstm_16/TensorArrayV2_1TensorListReserve.lstm_16/TensorArrayV2_1/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2_1^
lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/time�
 lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_16/while/maximum_iterationsz
lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/while/loop_counter�
lstm_16/whileWhile#lstm_16/while/loop_counter:output:0)lstm_16/while/maximum_iterations:output:0lstm_16/time:output:0 lstm_16/TensorArrayV2_1:handle:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0 lstm_16/strided_slice_1:output:0?lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_16_lstm_cell_32_matmul_readvariableop_resource5lstm_16_lstm_cell_32_matmul_1_readvariableop_resource4lstm_16_lstm_cell_32_biasadd_readvariableop_resource*
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
lstm_16_while_body_281371*%
condR
lstm_16_while_cond_281370*K
output_shapes:
8: : : : :���������@:���������@: : : : : *
parallel_iterations 2
lstm_16/while�
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2:
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_16/TensorArrayV2Stack/TensorListStackTensorListStacklstm_16/while:output:3Alstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:���������@*
element_dtype02,
*lstm_16/TensorArrayV2Stack/TensorListStack�
lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_16/strided_slice_3/stack�
lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_16/strided_slice_3/stack_1�
lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_3/stack_2�
lstm_16/strided_slice_3StridedSlice3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_16/strided_slice_3/stack:output:0(lstm_16/strided_slice_3/stack_1:output:0(lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_16/strided_slice_3�
lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose_1/perm�
lstm_16/transpose_1	Transpose3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_16/transpose_1/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_16/transpose_1v
lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/runtimee
lstm_17/ShapeShapelstm_16/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_17/Shape�
lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice/stack�
lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_1�
lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_2�
lstm_17/strided_sliceStridedSlicelstm_17/Shape:output:0$lstm_17/strided_slice/stack:output:0&lstm_17/strided_slice/stack_1:output:0&lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slicel
lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros/mul/y�
lstm_17/zeros/mulMullstm_17/strided_slice:output:0lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/mulo
lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_17/zeros/Less/y�
lstm_17/zeros/LessLesslstm_17/zeros/mul:z:0lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/Lessr
lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros/packed/1�
lstm_17/zeros/packedPacklstm_17/strided_slice:output:0lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros/packedo
lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros/Const�
lstm_17/zerosFilllstm_17/zeros/packed:output:0lstm_17/zeros/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/zerosp
lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros_1/mul/y�
lstm_17/zeros_1/mulMullstm_17/strided_slice:output:0lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/muls
lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
lstm_17/zeros_1/Less/y�
lstm_17/zeros_1/LessLesslstm_17/zeros_1/mul:z:0lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/Lessv
lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/zeros_1/packed/1�
lstm_17/zeros_1/packedPacklstm_17/strided_slice:output:0!lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros_1/packeds
lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros_1/Const�
lstm_17/zeros_1Filllstm_17/zeros_1/packed:output:0lstm_17/zeros_1/Const:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/zeros_1�
lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose/perm�
lstm_17/transpose	Transposelstm_16/transpose_1:y:0lstm_17/transpose/perm:output:0*
T0*+
_output_shapes
:���������@2
lstm_17/transposeg
lstm_17/Shape_1Shapelstm_17/transpose:y:0*
T0*
_output_shapes
:2
lstm_17/Shape_1�
lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_1/stack�
lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_1�
lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_2�
lstm_17/strided_slice_1StridedSlicelstm_17/Shape_1:output:0&lstm_17/strided_slice_1/stack:output:0(lstm_17/strided_slice_1/stack_1:output:0(lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slice_1�
#lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#lstm_17/TensorArrayV2/element_shape�
lstm_17/TensorArrayV2TensorListReserve,lstm_17/TensorArrayV2/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2�
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape�
/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_17/transpose:y:0Flstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_17/TensorArrayUnstack/TensorListFromTensor�
lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_2/stack�
lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_1�
lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_2�
lstm_17/strided_slice_2StridedSlicelstm_17/transpose:y:0&lstm_17/strided_slice_2/stack:output:0(lstm_17/strided_slice_2/stack_1:output:0(lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*
shrink_axis_mask2
lstm_17/strided_slice_2�
*lstm_17/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3lstm_17_lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02,
*lstm_17/lstm_cell_33/MatMul/ReadVariableOp�
lstm_17/lstm_cell_33/MatMulMatMul lstm_17/strided_slice_2:output:02lstm_17/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/MatMul�
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02.
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_17/lstm_cell_33/MatMul_1MatMullstm_17/zeros:output:04lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/MatMul_1�
lstm_17/lstm_cell_33/addAddV2%lstm_17/lstm_cell_33/MatMul:product:0'lstm_17/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/add�
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02-
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_17/lstm_cell_33/BiasAddBiasAddlstm_17/lstm_cell_33/add:z:03lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_17/lstm_cell_33/BiasAdd�
$lstm_17/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_17/lstm_cell_33/split/split_dim�
lstm_17/lstm_cell_33/splitSplit-lstm_17/lstm_cell_33/split/split_dim:output:0%lstm_17/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_17/lstm_cell_33/split�
lstm_17/lstm_cell_33/SigmoidSigmoid#lstm_17/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Sigmoid�
lstm_17/lstm_cell_33/Sigmoid_1Sigmoid#lstm_17/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2 
lstm_17/lstm_cell_33/Sigmoid_1�
lstm_17/lstm_cell_33/mulMul"lstm_17/lstm_cell_33/Sigmoid_1:y:0lstm_17/zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul�
lstm_17/lstm_cell_33/ReluRelu#lstm_17/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Relu�
lstm_17/lstm_cell_33/mul_1Mul lstm_17/lstm_cell_33/Sigmoid:y:0'lstm_17/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul_1�
lstm_17/lstm_cell_33/add_1AddV2lstm_17/lstm_cell_33/mul:z:0lstm_17/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/add_1�
lstm_17/lstm_cell_33/Sigmoid_2Sigmoid#lstm_17/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2 
lstm_17/lstm_cell_33/Sigmoid_2�
lstm_17/lstm_cell_33/Relu_1Relulstm_17/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/Relu_1�
lstm_17/lstm_cell_33/mul_2Mul"lstm_17/lstm_cell_33/Sigmoid_2:y:0)lstm_17/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_17/lstm_cell_33/mul_2�
%lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2'
%lstm_17/TensorArrayV2_1/element_shape�
lstm_17/TensorArrayV2_1TensorListReserve.lstm_17/TensorArrayV2_1/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2_1^
lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/time�
 lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������2"
 lstm_17/while/maximum_iterationsz
lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/while/loop_counter�
lstm_17/whileWhile#lstm_17/while/loop_counter:output:0)lstm_17/while/maximum_iterations:output:0lstm_17/time:output:0 lstm_17/TensorArrayV2_1:handle:0lstm_17/zeros:output:0lstm_17/zeros_1:output:0 lstm_17/strided_slice_1:output:0?lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_17_lstm_cell_33_matmul_readvariableop_resource5lstm_17_lstm_cell_33_matmul_1_readvariableop_resource4lstm_17_lstm_cell_33_biasadd_readvariableop_resource*
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
lstm_17_while_body_281518*%
condR
lstm_17_while_cond_281517*K
output_shapes:
8: : : : :��������� :��������� : : : : : *
parallel_iterations 2
lstm_17/while�
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����    2:
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shape�
*lstm_17/TensorArrayV2Stack/TensorListStackTensorListStacklstm_17/while:output:3Alstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:��������� *
element_dtype02,
*lstm_17/TensorArrayV2Stack/TensorListStack�
lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
lstm_17/strided_slice_3/stack�
lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_17/strided_slice_3/stack_1�
lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_3/stack_2�
lstm_17/strided_slice_3StridedSlice3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_17/strided_slice_3/stack:output:0(lstm_17/strided_slice_3/stack_1:output:0(lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:��������� *
shrink_axis_mask2
lstm_17/strided_slice_3�
lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose_1/perm�
lstm_17/transpose_1	Transpose3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_17/transpose_1/perm:output:0*
T0*+
_output_shapes
:��������� 2
lstm_17/transpose_1v
lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/runtime�
dropout_8/IdentityIdentity lstm_17/strided_slice_3:output:0*
T0*'
_output_shapes
:��������� 2
dropout_8/Identity�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMuldropout_8/Identity:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp,^lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp+^lstm_16/lstm_cell_32/MatMul/ReadVariableOp-^lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp^lstm_16/while,^lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp+^lstm_17/lstm_cell_33/MatMul/ReadVariableOp-^lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp^lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2Z
+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp+lstm_16/lstm_cell_32/BiasAdd/ReadVariableOp2X
*lstm_16/lstm_cell_32/MatMul/ReadVariableOp*lstm_16/lstm_cell_32/MatMul/ReadVariableOp2\
,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp,lstm_16/lstm_cell_32/MatMul_1/ReadVariableOp2
lstm_16/whilelstm_16/while2Z
+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp+lstm_17/lstm_cell_33/BiasAdd/ReadVariableOp2X
*lstm_17/lstm_cell_33/MatMul/ReadVariableOp*lstm_17/lstm_cell_33/MatMul/ReadVariableOp2\
,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp,lstm_17/lstm_cell_33/MatMul_1/ReadVariableOp2
lstm_17/whilelstm_17/while:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_8_layer_call_and_return_conditional_losses_280693

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

�
-__inference_sequential_8_layer_call_fn_281185
lstm_16_input
unknown:	�
	unknown_0:	@�
	unknown_1:	�
	unknown_2:	@�
	unknown_3:	 �
	unknown_4:	�
	unknown_5: 
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_2811452
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
_user_specified_namelstm_16_input
�

�
lstm_17_while_cond_281822,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3.
*lstm_17_while_less_lstm_17_strided_slice_1D
@lstm_17_while_lstm_17_while_cond_281822___redundant_placeholder0D
@lstm_17_while_lstm_17_while_cond_281822___redundant_placeholder1D
@lstm_17_while_lstm_17_while_cond_281822___redundant_placeholder2D
@lstm_17_while_lstm_17_while_cond_281822___redundant_placeholder3
lstm_17_while_identity
�
lstm_17/while/LessLesslstm_17_while_placeholder*lstm_17_while_less_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2
lstm_17/while/Lessu
lstm_17/while/IdentityIdentitylstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_17/while/Identity"9
lstm_17_while_identitylstm_17/while/Identity:output:0*(
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
C__inference_dense_8_layer_call_and_return_conditional_losses_283263

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
�
(__inference_lstm_16_layer_call_fn_281932
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2792512
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
�%
�
while_body_280022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0.
while_lstm_cell_33_280046_0:	@�.
while_lstm_cell_33_280048_0:	 �*
while_lstm_cell_33_280050_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor,
while_lstm_cell_33_280046:	@�,
while_lstm_cell_33_280048:	 �(
while_lstm_cell_33_280050:	���*while/lstm_cell_33/StatefulPartitionedCall�
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
*while/lstm_cell_33/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_33_280046_0while_lstm_cell_33_280048_0while_lstm_cell_33_280050_0*
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2799442,
*while/lstm_cell_33/StatefulPartitionedCall�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_33/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_33/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identity3while/lstm_cell_33/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp+^while/lstm_cell_33/StatefulPartitionedCall*"
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
while_lstm_cell_33_280046while_lstm_cell_33_280046_0"8
while_lstm_cell_33_280048while_lstm_cell_33_280048_0"8
while_lstm_cell_33_280050while_lstm_cell_33_280050_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2X
*while/lstm_cell_33/StatefulPartitionedCall*while/lstm_cell_33/StatefulPartitionedCall: 
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
�
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_280681

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
�]
�
&sequential_8_lstm_17_while_body_279002F
Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counterL
Hsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations*
&sequential_8_lstm_17_while_placeholder,
(sequential_8_lstm_17_while_placeholder_1,
(sequential_8_lstm_17_while_placeholder_2,
(sequential_8_lstm_17_while_placeholder_3E
Asequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1_0�
}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_8_lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0:	@�]
Jsequential_8_lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �X
Isequential_8_lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0:	�'
#sequential_8_lstm_17_while_identity)
%sequential_8_lstm_17_while_identity_1)
%sequential_8_lstm_17_while_identity_2)
%sequential_8_lstm_17_while_identity_3)
%sequential_8_lstm_17_while_identity_4)
%sequential_8_lstm_17_while_identity_5C
?sequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1
{sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensorY
Fsequential_8_lstm_17_while_lstm_cell_33_matmul_readvariableop_resource:	@�[
Hsequential_8_lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource:	 �V
Gsequential_8_lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource:	���>sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�=sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�?sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
Lsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2N
Lsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape�
>sequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0&sequential_8_lstm_17_while_placeholderUsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype02@
>sequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem�
=sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOpHsequential_8_lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02?
=sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�
.sequential_8/lstm_17/while/lstm_cell_33/MatMulMatMulEsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������20
.sequential_8/lstm_17/while/lstm_cell_33/MatMul�
?sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOpJsequential_8_lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02A
?sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
0sequential_8/lstm_17/while/lstm_cell_33/MatMul_1MatMul(sequential_8_lstm_17_while_placeholder_2Gsequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_8/lstm_17/while/lstm_cell_33/MatMul_1�
+sequential_8/lstm_17/while/lstm_cell_33/addAddV28sequential_8/lstm_17/while/lstm_cell_33/MatMul:product:0:sequential_8/lstm_17/while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2-
+sequential_8/lstm_17/while/lstm_cell_33/add�
>sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOpIsequential_8_lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02@
>sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�
/sequential_8/lstm_17/while/lstm_cell_33/BiasAddBiasAdd/sequential_8/lstm_17/while/lstm_cell_33/add:z:0Fsequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_8/lstm_17/while/lstm_cell_33/BiasAdd�
7sequential_8/lstm_17/while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_8/lstm_17/while/lstm_cell_33/split/split_dim�
-sequential_8/lstm_17/while/lstm_cell_33/splitSplit@sequential_8/lstm_17/while/lstm_cell_33/split/split_dim:output:08sequential_8/lstm_17/while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2/
-sequential_8/lstm_17/while/lstm_cell_33/split�
/sequential_8/lstm_17/while/lstm_cell_33/SigmoidSigmoid6sequential_8/lstm_17/while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 21
/sequential_8/lstm_17/while/lstm_cell_33/Sigmoid�
1sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_1Sigmoid6sequential_8/lstm_17/while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 23
1sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_1�
+sequential_8/lstm_17/while/lstm_cell_33/mulMul5sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_1:y:0(sequential_8_lstm_17_while_placeholder_3*
T0*'
_output_shapes
:��������� 2-
+sequential_8/lstm_17/while/lstm_cell_33/mul�
,sequential_8/lstm_17/while/lstm_cell_33/ReluRelu6sequential_8/lstm_17/while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2.
,sequential_8/lstm_17/while/lstm_cell_33/Relu�
-sequential_8/lstm_17/while/lstm_cell_33/mul_1Mul3sequential_8/lstm_17/while/lstm_cell_33/Sigmoid:y:0:sequential_8/lstm_17/while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2/
-sequential_8/lstm_17/while/lstm_cell_33/mul_1�
-sequential_8/lstm_17/while/lstm_cell_33/add_1AddV2/sequential_8/lstm_17/while/lstm_cell_33/mul:z:01sequential_8/lstm_17/while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2/
-sequential_8/lstm_17/while/lstm_cell_33/add_1�
1sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_2Sigmoid6sequential_8/lstm_17/while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 23
1sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_2�
.sequential_8/lstm_17/while/lstm_cell_33/Relu_1Relu1sequential_8/lstm_17/while/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 20
.sequential_8/lstm_17/while/lstm_cell_33/Relu_1�
-sequential_8/lstm_17/while/lstm_cell_33/mul_2Mul5sequential_8/lstm_17/while/lstm_cell_33/Sigmoid_2:y:0<sequential_8/lstm_17/while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2/
-sequential_8/lstm_17/while/lstm_cell_33/mul_2�
?sequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_8_lstm_17_while_placeholder_1&sequential_8_lstm_17_while_placeholder1sequential_8/lstm_17/while/lstm_cell_33/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItem�
 sequential_8/lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_8/lstm_17/while/add/y�
sequential_8/lstm_17/while/addAddV2&sequential_8_lstm_17_while_placeholder)sequential_8/lstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_17/while/add�
"sequential_8/lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_8/lstm_17/while/add_1/y�
 sequential_8/lstm_17/while/add_1AddV2Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counter+sequential_8/lstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_17/while/add_1�
#sequential_8/lstm_17/while/IdentityIdentity$sequential_8/lstm_17/while/add_1:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_8/lstm_17/while/Identity�
%sequential_8/lstm_17/while/Identity_1IdentityHsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_1�
%sequential_8/lstm_17/while/Identity_2Identity"sequential_8/lstm_17/while/add:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_2�
%sequential_8/lstm_17/while/Identity_3IdentityOsequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_3�
%sequential_8/lstm_17/while/Identity_4Identity1sequential_8/lstm_17/while/lstm_cell_33/mul_2:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2'
%sequential_8/lstm_17/while/Identity_4�
%sequential_8/lstm_17/while/Identity_5Identity1sequential_8/lstm_17/while/lstm_cell_33/add_1:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2'
%sequential_8/lstm_17/while/Identity_5�
sequential_8/lstm_17/while/NoOpNoOp?^sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp>^sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp@^sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_8/lstm_17/while/NoOp"S
#sequential_8_lstm_17_while_identity,sequential_8/lstm_17/while/Identity:output:0"W
%sequential_8_lstm_17_while_identity_1.sequential_8/lstm_17/while/Identity_1:output:0"W
%sequential_8_lstm_17_while_identity_2.sequential_8/lstm_17/while/Identity_2:output:0"W
%sequential_8_lstm_17_while_identity_3.sequential_8/lstm_17/while/Identity_3:output:0"W
%sequential_8_lstm_17_while_identity_4.sequential_8/lstm_17/while/Identity_4:output:0"W
%sequential_8_lstm_17_while_identity_5.sequential_8/lstm_17/while/Identity_5:output:0"�
Gsequential_8_lstm_17_while_lstm_cell_33_biasadd_readvariableop_resourceIsequential_8_lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0"�
Hsequential_8_lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resourceJsequential_8_lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0"�
Fsequential_8_lstm_17_while_lstm_cell_33_matmul_readvariableop_resourceHsequential_8_lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0"�
?sequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1Asequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1_0"�
{sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2�
>sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp>sequential_8/lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp2~
=sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp=sequential_8/lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp2�
?sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp?sequential_8/lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283329

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
�
�
while_cond_280021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_280021___redundant_placeholder04
0while_while_cond_280021___redundant_placeholder14
0while_while_cond_280021___redundant_placeholder24
0while_while_cond_280021___redundant_placeholder3
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
�]
�
&sequential_8_lstm_16_while_body_278855F
Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counterL
Hsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations*
&sequential_8_lstm_16_while_placeholder,
(sequential_8_lstm_16_while_placeholder_1,
(sequential_8_lstm_16_while_placeholder_2,
(sequential_8_lstm_16_while_placeholder_3E
Asequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1_0�
}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_8_lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0:	�]
Jsequential_8_lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�X
Isequential_8_lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0:	�'
#sequential_8_lstm_16_while_identity)
%sequential_8_lstm_16_while_identity_1)
%sequential_8_lstm_16_while_identity_2)
%sequential_8_lstm_16_while_identity_3)
%sequential_8_lstm_16_while_identity_4)
%sequential_8_lstm_16_while_identity_5C
?sequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1
{sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensorY
Fsequential_8_lstm_16_while_lstm_cell_32_matmul_readvariableop_resource:	�[
Hsequential_8_lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource:	@�V
Gsequential_8_lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource:	���>sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�=sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�?sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
Lsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2N
Lsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape�
>sequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0&sequential_8_lstm_16_while_placeholderUsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype02@
>sequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem�
=sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOpHsequential_8_lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype02?
=sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�
.sequential_8/lstm_16/while/lstm_cell_32/MatMulMatMulEsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������20
.sequential_8/lstm_16/while/lstm_cell_32/MatMul�
?sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOpJsequential_8_lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02A
?sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
0sequential_8/lstm_16/while/lstm_cell_32/MatMul_1MatMul(sequential_8_lstm_16_while_placeholder_2Gsequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������22
0sequential_8/lstm_16/while/lstm_cell_32/MatMul_1�
+sequential_8/lstm_16/while/lstm_cell_32/addAddV28sequential_8/lstm_16/while/lstm_cell_32/MatMul:product:0:sequential_8/lstm_16/while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2-
+sequential_8/lstm_16/while/lstm_cell_32/add�
>sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOpIsequential_8_lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02@
>sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�
/sequential_8/lstm_16/while/lstm_cell_32/BiasAddBiasAdd/sequential_8/lstm_16/while/lstm_cell_32/add:z:0Fsequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������21
/sequential_8/lstm_16/while/lstm_cell_32/BiasAdd�
7sequential_8/lstm_16/while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_8/lstm_16/while/lstm_cell_32/split/split_dim�
-sequential_8/lstm_16/while/lstm_cell_32/splitSplit@sequential_8/lstm_16/while/lstm_cell_32/split/split_dim:output:08sequential_8/lstm_16/while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2/
-sequential_8/lstm_16/while/lstm_cell_32/split�
/sequential_8/lstm_16/while/lstm_cell_32/SigmoidSigmoid6sequential_8/lstm_16/while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@21
/sequential_8/lstm_16/while/lstm_cell_32/Sigmoid�
1sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_1Sigmoid6sequential_8/lstm_16/while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@23
1sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_1�
+sequential_8/lstm_16/while/lstm_cell_32/mulMul5sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_1:y:0(sequential_8_lstm_16_while_placeholder_3*
T0*'
_output_shapes
:���������@2-
+sequential_8/lstm_16/while/lstm_cell_32/mul�
,sequential_8/lstm_16/while/lstm_cell_32/ReluRelu6sequential_8/lstm_16/while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2.
,sequential_8/lstm_16/while/lstm_cell_32/Relu�
-sequential_8/lstm_16/while/lstm_cell_32/mul_1Mul3sequential_8/lstm_16/while/lstm_cell_32/Sigmoid:y:0:sequential_8/lstm_16/while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2/
-sequential_8/lstm_16/while/lstm_cell_32/mul_1�
-sequential_8/lstm_16/while/lstm_cell_32/add_1AddV2/sequential_8/lstm_16/while/lstm_cell_32/mul:z:01sequential_8/lstm_16/while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2/
-sequential_8/lstm_16/while/lstm_cell_32/add_1�
1sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_2Sigmoid6sequential_8/lstm_16/while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@23
1sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_2�
.sequential_8/lstm_16/while/lstm_cell_32/Relu_1Relu1sequential_8/lstm_16/while/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@20
.sequential_8/lstm_16/while/lstm_cell_32/Relu_1�
-sequential_8/lstm_16/while/lstm_cell_32/mul_2Mul5sequential_8/lstm_16/while/lstm_cell_32/Sigmoid_2:y:0<sequential_8/lstm_16/while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2/
-sequential_8/lstm_16/while/lstm_cell_32/mul_2�
?sequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_8_lstm_16_while_placeholder_1&sequential_8_lstm_16_while_placeholder1sequential_8/lstm_16/while/lstm_cell_32/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItem�
 sequential_8/lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_8/lstm_16/while/add/y�
sequential_8/lstm_16/while/addAddV2&sequential_8_lstm_16_while_placeholder)sequential_8/lstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_16/while/add�
"sequential_8/lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_8/lstm_16/while/add_1/y�
 sequential_8/lstm_16/while/add_1AddV2Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counter+sequential_8/lstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_16/while/add_1�
#sequential_8/lstm_16/while/IdentityIdentity$sequential_8/lstm_16/while/add_1:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_8/lstm_16/while/Identity�
%sequential_8/lstm_16/while/Identity_1IdentityHsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_1�
%sequential_8/lstm_16/while/Identity_2Identity"sequential_8/lstm_16/while/add:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_2�
%sequential_8/lstm_16/while/Identity_3IdentityOsequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_3�
%sequential_8/lstm_16/while/Identity_4Identity1sequential_8/lstm_16/while/lstm_cell_32/mul_2:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2'
%sequential_8/lstm_16/while/Identity_4�
%sequential_8/lstm_16/while/Identity_5Identity1sequential_8/lstm_16/while/lstm_cell_32/add_1:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2'
%sequential_8/lstm_16/while/Identity_5�
sequential_8/lstm_16/while/NoOpNoOp?^sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp>^sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp@^sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_8/lstm_16/while/NoOp"S
#sequential_8_lstm_16_while_identity,sequential_8/lstm_16/while/Identity:output:0"W
%sequential_8_lstm_16_while_identity_1.sequential_8/lstm_16/while/Identity_1:output:0"W
%sequential_8_lstm_16_while_identity_2.sequential_8/lstm_16/while/Identity_2:output:0"W
%sequential_8_lstm_16_while_identity_3.sequential_8/lstm_16/while/Identity_3:output:0"W
%sequential_8_lstm_16_while_identity_4.sequential_8/lstm_16/while/Identity_4:output:0"W
%sequential_8_lstm_16_while_identity_5.sequential_8/lstm_16/while/Identity_5:output:0"�
Gsequential_8_lstm_16_while_lstm_cell_32_biasadd_readvariableop_resourceIsequential_8_lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0"�
Hsequential_8_lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resourceJsequential_8_lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0"�
Fsequential_8_lstm_16_while_lstm_cell_32_matmul_readvariableop_resourceHsequential_8_lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0"�
?sequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1Asequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1_0"�
{sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2�
>sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp>sequential_8/lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp2~
=sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp=sequential_8/lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp2�
?sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp?sequential_8/lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
while_cond_280583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_280583___redundant_placeholder04
0while_while_cond_280583___redundant_placeholder14
0while_while_cond_280583___redundant_placeholder24
0while_while_cond_280583___redundant_placeholder3
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_279461

inputs&
lstm_cell_32_279379:	�&
lstm_cell_32_279381:	@�"
lstm_cell_32_279383:	�
identity��$lstm_cell_32/StatefulPartitionedCall�whileD
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
$lstm_cell_32/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_32_279379lstm_cell_32_279381lstm_cell_32_279383*
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2793142&
$lstm_cell_32/StatefulPartitionedCall�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_32_279379lstm_cell_32_279381lstm_cell_32_279383*
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
while_body_279392*
condR
while_cond_279391*K
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
NoOpNoOp%^lstm_cell_32/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2L
$lstm_cell_32/StatefulPartitionedCall$lstm_cell_32/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283361

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
�
�
while_cond_281004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_281004___redundant_placeholder04
0while_while_cond_281004___redundant_placeholder14
0while_while_cond_281004___redundant_placeholder24
0while_while_cond_281004___redundant_placeholder3
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
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_283232

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
�\
�
C__inference_lstm_17_layer_call_and_return_conditional_losses_282764
inputs_0>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_282680*
condR
while_cond_282679*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������@
"
_user_specified_name
inputs/0
�
�
while_cond_282679
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_282679___redundant_placeholder04
0while_while_cond_282679___redundant_placeholder14
0while_while_cond_282679___redundant_placeholder24
0while_while_cond_282679___redundant_placeholder3
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282418

inputs>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_282334*
condR
while_cond_282333*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_33_layer_call_fn_283378

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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_2797982
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_279798

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
�[
�
C__inference_lstm_17_layer_call_and_return_conditional_losses_280916

inputs>
+lstm_cell_33_matmul_readvariableop_resource:	@�@
-lstm_cell_33_matmul_1_readvariableop_resource:	 �;
,lstm_cell_33_biasadd_readvariableop_resource:	�
identity��#lstm_cell_33/BiasAdd/ReadVariableOp�"lstm_cell_33/MatMul/ReadVariableOp�$lstm_cell_33/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_33/MatMul/ReadVariableOpReadVariableOp+lstm_cell_33_matmul_readvariableop_resource*
_output_shapes
:	@�*
dtype02$
"lstm_cell_33/MatMul/ReadVariableOp�
lstm_cell_33/MatMulMatMulstrided_slice_2:output:0*lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul�
$lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_33_matmul_1_readvariableop_resource*
_output_shapes
:	 �*
dtype02&
$lstm_cell_33/MatMul_1/ReadVariableOp�
lstm_cell_33/MatMul_1MatMulzeros:output:0,lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/MatMul_1�
lstm_cell_33/addAddV2lstm_cell_33/MatMul:product:0lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/add�
#lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_33_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_33/BiasAdd/ReadVariableOp�
lstm_cell_33/BiasAddBiasAddlstm_cell_33/add:z:0+lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_33/BiasAdd~
lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_33/split/split_dim�
lstm_cell_33/splitSplit%lstm_cell_33/split/split_dim:output:0lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
lstm_cell_33/split�
lstm_cell_33/SigmoidSigmoidlstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid�
lstm_cell_33/Sigmoid_1Sigmoidlstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_1�
lstm_cell_33/mulMullstm_cell_33/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul}
lstm_cell_33/ReluRelulstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu�
lstm_cell_33/mul_1Mullstm_cell_33/Sigmoid:y:0lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_1�
lstm_cell_33/add_1AddV2lstm_cell_33/mul:z:0lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/add_1�
lstm_cell_33/Sigmoid_2Sigmoidlstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Sigmoid_2|
lstm_cell_33/Relu_1Relulstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/Relu_1�
lstm_cell_33/mul_2Mullstm_cell_33/Sigmoid_2:y:0!lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
lstm_cell_33/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_33_matmul_readvariableop_resource-lstm_cell_33_matmul_1_readvariableop_resource,lstm_cell_33_biasadd_readvariableop_resource*
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
while_body_280832*
condR
while_cond_280831*K
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
NoOpNoOp$^lstm_cell_33/BiasAdd/ReadVariableOp#^lstm_cell_33/MatMul/ReadVariableOp%^lstm_cell_33/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������@: : : 2J
#lstm_cell_33/BiasAdd/ReadVariableOp#lstm_cell_33/BiasAdd/ReadVariableOp2H
"lstm_cell_33/MatMul/ReadVariableOp"lstm_cell_33/MatMul/ReadVariableOp2L
$lstm_cell_33/MatMul_1/ReadVariableOp$lstm_cell_33/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�[
�
C__inference_lstm_16_layer_call_and_return_conditional_losses_282569

inputs>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileD
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_282485*
condR
while_cond_282484*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:���������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_lstm_cell_32_layer_call_fn_283280

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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_2791682
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
while_cond_280425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_280425___redundant_placeholder04
0while_while_cond_280425___redundant_placeholder14
0while_while_cond_280425___redundant_placeholder24
0while_while_cond_280425___redundant_placeholder3
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
while_body_282831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_33_matmul_readvariableop_resource_0:	@�H
5while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �C
4while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_33_matmul_readvariableop_resource:	@�F
3while_lstm_cell_33_matmul_1_readvariableop_resource:	 �A
2while_lstm_cell_33_biasadd_readvariableop_resource:	���)while/lstm_cell_33/BiasAdd/ReadVariableOp�(while/lstm_cell_33/MatMul/ReadVariableOp�*while/lstm_cell_33/MatMul_1/ReadVariableOp�
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
(while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype02*
(while/lstm_cell_33/MatMul/ReadVariableOp�
while/lstm_cell_33/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul�
*while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype02,
*while/lstm_cell_33/MatMul_1/ReadVariableOp�
while/lstm_cell_33/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/MatMul_1�
while/lstm_cell_33/addAddV2#while/lstm_cell_33/MatMul:product:0%while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/add�
)while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype02+
)while/lstm_cell_33/BiasAdd/ReadVariableOp�
while/lstm_cell_33/BiasAddBiasAddwhile/lstm_cell_33/add:z:01while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
while/lstm_cell_33/BiasAdd�
"while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_33/split/split_dim�
while/lstm_cell_33/splitSplit+while/lstm_cell_33/split/split_dim:output:0#while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2
while/lstm_cell_33/split�
while/lstm_cell_33/SigmoidSigmoid!while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid�
while/lstm_cell_33/Sigmoid_1Sigmoid!while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_1�
while/lstm_cell_33/mulMul while/lstm_cell_33/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul�
while/lstm_cell_33/ReluRelu!while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu�
while/lstm_cell_33/mul_1Mulwhile/lstm_cell_33/Sigmoid:y:0%while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_1�
while/lstm_cell_33/add_1AddV2while/lstm_cell_33/mul:z:0while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/add_1�
while/lstm_cell_33/Sigmoid_2Sigmoid!while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Sigmoid_2�
while/lstm_cell_33/Relu_1Reluwhile/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/Relu_1�
while/lstm_cell_33/mul_2Mul while/lstm_cell_33/Sigmoid_2:y:0'while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2
while/lstm_cell_33/mul_2�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_33/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_33/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_4�
while/Identity_5Identitywhile/lstm_cell_33/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:��������� 2
while/Identity_5�

while/NoOpNoOp*^while/lstm_cell_33/BiasAdd/ReadVariableOp)^while/lstm_cell_33/MatMul/ReadVariableOp+^while/lstm_cell_33/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_33_biasadd_readvariableop_resource4while_lstm_cell_33_biasadd_readvariableop_resource_0"l
3while_lstm_cell_33_matmul_1_readvariableop_resource5while_lstm_cell_33_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_33_matmul_readvariableop_resource3while_lstm_cell_33_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2V
)while/lstm_cell_33/BiasAdd/ReadVariableOp)while/lstm_cell_33/BiasAdd/ReadVariableOp2T
(while/lstm_cell_33/MatMul/ReadVariableOp(while/lstm_cell_33/MatMul/ReadVariableOp2X
*while/lstm_cell_33/MatMul_1/ReadVariableOp*while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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

lstm_16_while_body_281371,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3+
'lstm_16_while_lstm_16_strided_slice_1_0g
clstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0:	�P
=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0:	@�K
<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0:	�
lstm_16_while_identity
lstm_16_while_identity_1
lstm_16_while_identity_2
lstm_16_while_identity_3
lstm_16_while_identity_4
lstm_16_while_identity_5)
%lstm_16_while_lstm_16_strided_slice_1e
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorL
9lstm_16_while_lstm_cell_32_matmul_readvariableop_resource:	�N
;lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource:	@�I
:lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource:	���1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2A
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0lstm_16_while_placeholderHlstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype023
1lstm_16/while/TensorArrayV2Read/TensorListGetItem�
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOpReadVariableOp;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype022
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp�
!lstm_16/while/lstm_cell_32/MatMulMatMul8lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_16/while/lstm_cell_32/MatMul�
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0*
_output_shapes
:	@�*
dtype024
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp�
#lstm_16/while/lstm_cell_32/MatMul_1MatMullstm_16_while_placeholder_2:lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_16/while/lstm_cell_32/MatMul_1�
lstm_16/while/lstm_cell_32/addAddV2+lstm_16/while/lstm_cell_32/MatMul:product:0-lstm_16/while/lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_16/while/lstm_cell_32/add�
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp�
"lstm_16/while/lstm_cell_32/BiasAddBiasAdd"lstm_16/while/lstm_cell_32/add:z:09lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_16/while/lstm_cell_32/BiasAdd�
*lstm_16/while/lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_16/while/lstm_cell_32/split/split_dim�
 lstm_16/while/lstm_cell_32/splitSplit3lstm_16/while/lstm_cell_32/split/split_dim:output:0+lstm_16/while/lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2"
 lstm_16/while/lstm_cell_32/split�
"lstm_16/while/lstm_cell_32/SigmoidSigmoid)lstm_16/while/lstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2$
"lstm_16/while/lstm_cell_32/Sigmoid�
$lstm_16/while/lstm_cell_32/Sigmoid_1Sigmoid)lstm_16/while/lstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2&
$lstm_16/while/lstm_cell_32/Sigmoid_1�
lstm_16/while/lstm_cell_32/mulMul(lstm_16/while/lstm_cell_32/Sigmoid_1:y:0lstm_16_while_placeholder_3*
T0*'
_output_shapes
:���������@2 
lstm_16/while/lstm_cell_32/mul�
lstm_16/while/lstm_cell_32/ReluRelu)lstm_16/while/lstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2!
lstm_16/while/lstm_cell_32/Relu�
 lstm_16/while/lstm_cell_32/mul_1Mul&lstm_16/while/lstm_cell_32/Sigmoid:y:0-lstm_16/while/lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/mul_1�
 lstm_16/while/lstm_cell_32/add_1AddV2"lstm_16/while/lstm_cell_32/mul:z:0$lstm_16/while/lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/add_1�
$lstm_16/while/lstm_cell_32/Sigmoid_2Sigmoid)lstm_16/while/lstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2&
$lstm_16/while/lstm_cell_32/Sigmoid_2�
!lstm_16/while/lstm_cell_32/Relu_1Relu$lstm_16/while/lstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2#
!lstm_16/while/lstm_cell_32/Relu_1�
 lstm_16/while/lstm_cell_32/mul_2Mul(lstm_16/while/lstm_cell_32/Sigmoid_2:y:0/lstm_16/while/lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2"
 lstm_16/while/lstm_cell_32/mul_2�
2lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_16_while_placeholder_1lstm_16_while_placeholder$lstm_16/while/lstm_cell_32/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_16/while/TensorArrayV2Write/TensorListSetIteml
lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add/y�
lstm_16/while/addAddV2lstm_16_while_placeholderlstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/addp
lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add_1/y�
lstm_16/while/add_1AddV2(lstm_16_while_lstm_16_while_loop_counterlstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/add_1�
lstm_16/while/IdentityIdentitylstm_16/while/add_1:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity�
lstm_16/while/Identity_1Identity.lstm_16_while_lstm_16_while_maximum_iterations^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_1�
lstm_16/while/Identity_2Identitylstm_16/while/add:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_2�
lstm_16/while/Identity_3IdentityBlstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_3�
lstm_16/while/Identity_4Identity$lstm_16/while/lstm_cell_32/mul_2:z:0^lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_16/while/Identity_4�
lstm_16/while/Identity_5Identity$lstm_16/while/lstm_cell_32/add_1:z:0^lstm_16/while/NoOp*
T0*'
_output_shapes
:���������@2
lstm_16/while/Identity_5�
lstm_16/while/NoOpNoOp2^lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp1^lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp3^lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_16/while/NoOp"9
lstm_16_while_identitylstm_16/while/Identity:output:0"=
lstm_16_while_identity_1!lstm_16/while/Identity_1:output:0"=
lstm_16_while_identity_2!lstm_16/while/Identity_2:output:0"=
lstm_16_while_identity_3!lstm_16/while/Identity_3:output:0"=
lstm_16_while_identity_4!lstm_16/while/Identity_4:output:0"=
lstm_16_while_identity_5!lstm_16/while/Identity_5:output:0"P
%lstm_16_while_lstm_16_strided_slice_1'lstm_16_while_lstm_16_strided_slice_1_0"z
:lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource<lstm_16_while_lstm_cell_32_biasadd_readvariableop_resource_0"|
;lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource=lstm_16_while_lstm_cell_32_matmul_1_readvariableop_resource_0"x
9lstm_16_while_lstm_cell_32_matmul_readvariableop_resource;lstm_16_while_lstm_cell_32_matmul_readvariableop_resource_0"�
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������@:���������@: : : : : 2f
1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp1lstm_16/while/lstm_cell_32/BiasAdd/ReadVariableOp2d
0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp0lstm_16/while/lstm_cell_32/MatMul/ReadVariableOp2h
2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp2lstm_16/while/lstm_cell_32/MatMul_1/ReadVariableOp: 
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282267
inputs_0>
+lstm_cell_32_matmul_readvariableop_resource:	�@
-lstm_cell_32_matmul_1_readvariableop_resource:	@�;
,lstm_cell_32_biasadd_readvariableop_resource:	�
identity��#lstm_cell_32/BiasAdd/ReadVariableOp�"lstm_cell_32/MatMul/ReadVariableOp�$lstm_cell_32/MatMul_1/ReadVariableOp�whileF
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
"lstm_cell_32/MatMul/ReadVariableOpReadVariableOp+lstm_cell_32_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02$
"lstm_cell_32/MatMul/ReadVariableOp�
lstm_cell_32/MatMulMatMulstrided_slice_2:output:0*lstm_cell_32/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul�
$lstm_cell_32/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_32_matmul_1_readvariableop_resource*
_output_shapes
:	@�*
dtype02&
$lstm_cell_32/MatMul_1/ReadVariableOp�
lstm_cell_32/MatMul_1MatMulzeros:output:0,lstm_cell_32/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/MatMul_1�
lstm_cell_32/addAddV2lstm_cell_32/MatMul:product:0lstm_cell_32/MatMul_1:product:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/add�
#lstm_cell_32/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_32_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02%
#lstm_cell_32/BiasAdd/ReadVariableOp�
lstm_cell_32/BiasAddBiasAddlstm_cell_32/add:z:0+lstm_cell_32/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
lstm_cell_32/BiasAdd~
lstm_cell_32/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_32/split/split_dim�
lstm_cell_32/splitSplit%lstm_cell_32/split/split_dim:output:0lstm_cell_32/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������@:���������@:���������@:���������@*
	num_split2
lstm_cell_32/split�
lstm_cell_32/SigmoidSigmoidlstm_cell_32/split:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid�
lstm_cell_32/Sigmoid_1Sigmoidlstm_cell_32/split:output:1*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_1�
lstm_cell_32/mulMullstm_cell_32/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul}
lstm_cell_32/ReluRelulstm_cell_32/split:output:2*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu�
lstm_cell_32/mul_1Mullstm_cell_32/Sigmoid:y:0lstm_cell_32/Relu:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_1�
lstm_cell_32/add_1AddV2lstm_cell_32/mul:z:0lstm_cell_32/mul_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/add_1�
lstm_cell_32/Sigmoid_2Sigmoidlstm_cell_32/split:output:3*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Sigmoid_2|
lstm_cell_32/Relu_1Relulstm_cell_32/add_1:z:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/Relu_1�
lstm_cell_32/mul_2Mullstm_cell_32/Sigmoid_2:y:0!lstm_cell_32/Relu_1:activations:0*
T0*'
_output_shapes
:���������@2
lstm_cell_32/mul_2�
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_32_matmul_readvariableop_resource-lstm_cell_32_matmul_1_readvariableop_resource,lstm_cell_32_biasadd_readvariableop_resource*
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
while_body_282183*
condR
while_cond_282182*K
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
NoOpNoOp$^lstm_cell_32/BiasAdd/ReadVariableOp#^lstm_cell_32/MatMul/ReadVariableOp%^lstm_cell_32/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2J
#lstm_cell_32/BiasAdd/ReadVariableOp#lstm_cell_32/BiasAdd/ReadVariableOp2H
"lstm_cell_32/MatMul/ReadVariableOp"lstm_cell_32/MatMul/ReadVariableOp2L
$lstm_cell_32/MatMul_1/ReadVariableOp$lstm_cell_32/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs/0
�

�
-__inference_sequential_8_layer_call_fn_281283

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
H__inference_sequential_8_layer_call_and_return_conditional_losses_2807002
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

lstm_17_while_body_281823,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3+
'lstm_17_while_lstm_17_strided_slice_1_0g
clstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0:	@�P
=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0:	 �K
<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0:	�
lstm_17_while_identity
lstm_17_while_identity_1
lstm_17_while_identity_2
lstm_17_while_identity_3
lstm_17_while_identity_4
lstm_17_while_identity_5)
%lstm_17_while_lstm_17_strided_slice_1e
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorL
9lstm_17_while_lstm_cell_33_matmul_readvariableop_resource:	@�N
;lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource:	 �I
:lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource:	���1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����@   2A
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape�
1lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0lstm_17_while_placeholderHlstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������@*
element_dtype023
1lstm_17/while/TensorArrayV2Read/TensorListGetItem�
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOpReadVariableOp;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0*
_output_shapes
:	@�*
dtype022
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp�
!lstm_17/while/lstm_cell_33/MatMulMatMul8lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!lstm_17/while/lstm_cell_33/MatMul�
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOpReadVariableOp=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0*
_output_shapes
:	 �*
dtype024
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp�
#lstm_17/while/lstm_cell_33/MatMul_1MatMullstm_17_while_placeholder_2:lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2%
#lstm_17/while/lstm_cell_33/MatMul_1�
lstm_17/while/lstm_cell_33/addAddV2+lstm_17/while/lstm_cell_33/MatMul:product:0-lstm_17/while/lstm_cell_33/MatMul_1:product:0*
T0*(
_output_shapes
:����������2 
lstm_17/while/lstm_cell_33/add�
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOpReadVariableOp<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0*
_output_shapes	
:�*
dtype023
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp�
"lstm_17/while/lstm_cell_33/BiasAddBiasAdd"lstm_17/while/lstm_cell_33/add:z:09lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2$
"lstm_17/while/lstm_cell_33/BiasAdd�
*lstm_17/while/lstm_cell_33/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_17/while/lstm_cell_33/split/split_dim�
 lstm_17/while/lstm_cell_33/splitSplit3lstm_17/while/lstm_cell_33/split/split_dim:output:0+lstm_17/while/lstm_cell_33/BiasAdd:output:0*
T0*`
_output_shapesN
L:��������� :��������� :��������� :��������� *
	num_split2"
 lstm_17/while/lstm_cell_33/split�
"lstm_17/while/lstm_cell_33/SigmoidSigmoid)lstm_17/while/lstm_cell_33/split:output:0*
T0*'
_output_shapes
:��������� 2$
"lstm_17/while/lstm_cell_33/Sigmoid�
$lstm_17/while/lstm_cell_33/Sigmoid_1Sigmoid)lstm_17/while/lstm_cell_33/split:output:1*
T0*'
_output_shapes
:��������� 2&
$lstm_17/while/lstm_cell_33/Sigmoid_1�
lstm_17/while/lstm_cell_33/mulMul(lstm_17/while/lstm_cell_33/Sigmoid_1:y:0lstm_17_while_placeholder_3*
T0*'
_output_shapes
:��������� 2 
lstm_17/while/lstm_cell_33/mul�
lstm_17/while/lstm_cell_33/ReluRelu)lstm_17/while/lstm_cell_33/split:output:2*
T0*'
_output_shapes
:��������� 2!
lstm_17/while/lstm_cell_33/Relu�
 lstm_17/while/lstm_cell_33/mul_1Mul&lstm_17/while/lstm_cell_33/Sigmoid:y:0-lstm_17/while/lstm_cell_33/Relu:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/mul_1�
 lstm_17/while/lstm_cell_33/add_1AddV2"lstm_17/while/lstm_cell_33/mul:z:0$lstm_17/while/lstm_cell_33/mul_1:z:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/add_1�
$lstm_17/while/lstm_cell_33/Sigmoid_2Sigmoid)lstm_17/while/lstm_cell_33/split:output:3*
T0*'
_output_shapes
:��������� 2&
$lstm_17/while/lstm_cell_33/Sigmoid_2�
!lstm_17/while/lstm_cell_33/Relu_1Relu$lstm_17/while/lstm_cell_33/add_1:z:0*
T0*'
_output_shapes
:��������� 2#
!lstm_17/while/lstm_cell_33/Relu_1�
 lstm_17/while/lstm_cell_33/mul_2Mul(lstm_17/while/lstm_cell_33/Sigmoid_2:y:0/lstm_17/while/lstm_cell_33/Relu_1:activations:0*
T0*'
_output_shapes
:��������� 2"
 lstm_17/while/lstm_cell_33/mul_2�
2lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_17_while_placeholder_1lstm_17_while_placeholder$lstm_17/while/lstm_cell_33/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_17/while/TensorArrayV2Write/TensorListSetIteml
lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add/y�
lstm_17/while/addAddV2lstm_17_while_placeholderlstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/addp
lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add_1/y�
lstm_17/while/add_1AddV2(lstm_17_while_lstm_17_while_loop_counterlstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/add_1�
lstm_17/while/IdentityIdentitylstm_17/while/add_1:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity�
lstm_17/while/Identity_1Identity.lstm_17_while_lstm_17_while_maximum_iterations^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_1�
lstm_17/while/Identity_2Identitylstm_17/while/add:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_2�
lstm_17/while/Identity_3IdentityBlstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_3�
lstm_17/while/Identity_4Identity$lstm_17/while/lstm_cell_33/mul_2:z:0^lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_17/while/Identity_4�
lstm_17/while/Identity_5Identity$lstm_17/while/lstm_cell_33/add_1:z:0^lstm_17/while/NoOp*
T0*'
_output_shapes
:��������� 2
lstm_17/while/Identity_5�
lstm_17/while/NoOpNoOp2^lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp1^lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp3^lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_17/while/NoOp"9
lstm_17_while_identitylstm_17/while/Identity:output:0"=
lstm_17_while_identity_1!lstm_17/while/Identity_1:output:0"=
lstm_17_while_identity_2!lstm_17/while/Identity_2:output:0"=
lstm_17_while_identity_3!lstm_17/while/Identity_3:output:0"=
lstm_17_while_identity_4!lstm_17/while/Identity_4:output:0"=
lstm_17_while_identity_5!lstm_17/while/Identity_5:output:0"P
%lstm_17_while_lstm_17_strided_slice_1'lstm_17_while_lstm_17_strided_slice_1_0"z
:lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource<lstm_17_while_lstm_cell_33_biasadd_readvariableop_resource_0"|
;lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource=lstm_17_while_lstm_cell_33_matmul_1_readvariableop_resource_0"x
9lstm_17_while_lstm_cell_33_matmul_readvariableop_resource;lstm_17_while_lstm_cell_33_matmul_readvariableop_resource_0"�
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :��������� :��������� : : : : : 2f
1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp1lstm_17/while/lstm_cell_33/BiasAdd/ReadVariableOp2d
0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp0lstm_17/while/lstm_cell_33/MatMul/ReadVariableOp2h
2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp2lstm_17/while/lstm_cell_33/MatMul_1/ReadVariableOp: 
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283459

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
�
�
H__inference_sequential_8_layer_call_and_return_conditional_losses_281233
lstm_16_input!
lstm_16_281212:	�!
lstm_16_281214:	@�
lstm_16_281216:	�!
lstm_17_281219:	@�!
lstm_17_281221:	 �
lstm_17_281223:	� 
dense_8_281227: 
dense_8_281229:
identity��dense_8/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�lstm_16/StatefulPartitionedCall�lstm_17/StatefulPartitionedCall�
lstm_16/StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputlstm_16_281212lstm_16_281214lstm_16_281216*
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_2810892!
lstm_16/StatefulPartitionedCall�
lstm_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0lstm_17_281219lstm_17_281221lstm_17_281223*
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_2809162!
lstm_17/StatefulPartitionedCall�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_2807492#
!dropout_8/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0dense_8_281227dense_8_281229*
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
C__inference_dense_8_layer_call_and_return_conditional_losses_2806932!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_8/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:Z V
+
_output_shapes
:���������
'
_user_specified_namelstm_16_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
lstm_16_input:
serving_default_lstm_16_input:0���������;
dense_80
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
 : 2dense_8/kernel
:2dense_8/bias
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
.:,	�2lstm_16/lstm_cell_32/kernel
8:6	@�2%lstm_16/lstm_cell_32/recurrent_kernel
(:&�2lstm_16/lstm_cell_32/bias
.:,	@�2lstm_17/lstm_cell_33/kernel
8:6	 �2%lstm_17/lstm_cell_33/recurrent_kernel
(:&�2lstm_17/lstm_cell_33/bias
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
%:# 2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
3:1	�2"Adam/lstm_16/lstm_cell_32/kernel/m
=:;	@�2,Adam/lstm_16/lstm_cell_32/recurrent_kernel/m
-:+�2 Adam/lstm_16/lstm_cell_32/bias/m
3:1	@�2"Adam/lstm_17/lstm_cell_33/kernel/m
=:;	 �2,Adam/lstm_17/lstm_cell_33/recurrent_kernel/m
-:+�2 Adam/lstm_17/lstm_cell_33/bias/m
%:# 2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
3:1	�2"Adam/lstm_16/lstm_cell_32/kernel/v
=:;	@�2,Adam/lstm_16/lstm_cell_32/recurrent_kernel/v
-:+�2 Adam/lstm_16/lstm_cell_32/bias/v
3:1	@�2"Adam/lstm_17/lstm_cell_33/kernel/v
=:;	 �2,Adam/lstm_17/lstm_cell_33/recurrent_kernel/v
-:+�2 Adam/lstm_17/lstm_cell_33/bias/v
�B�
!__inference__wrapped_model_279093lstm_16_input"�
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
-__inference_sequential_8_layer_call_fn_280719
-__inference_sequential_8_layer_call_fn_281283
-__inference_sequential_8_layer_call_fn_281304
-__inference_sequential_8_layer_call_fn_281185�
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_281609
H__inference_sequential_8_layer_call_and_return_conditional_losses_281921
H__inference_sequential_8_layer_call_and_return_conditional_losses_281209
H__inference_sequential_8_layer_call_and_return_conditional_losses_281233�
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
(__inference_lstm_16_layer_call_fn_281932
(__inference_lstm_16_layer_call_fn_281943
(__inference_lstm_16_layer_call_fn_281954
(__inference_lstm_16_layer_call_fn_281965�
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282116
C__inference_lstm_16_layer_call_and_return_conditional_losses_282267
C__inference_lstm_16_layer_call_and_return_conditional_losses_282418
C__inference_lstm_16_layer_call_and_return_conditional_losses_282569�
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
(__inference_lstm_17_layer_call_fn_282580
(__inference_lstm_17_layer_call_fn_282591
(__inference_lstm_17_layer_call_fn_282602
(__inference_lstm_17_layer_call_fn_282613�
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_282764
C__inference_lstm_17_layer_call_and_return_conditional_losses_282915
C__inference_lstm_17_layer_call_and_return_conditional_losses_283066
C__inference_lstm_17_layer_call_and_return_conditional_losses_283217�
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
*__inference_dropout_8_layer_call_fn_283222
*__inference_dropout_8_layer_call_fn_283227�
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
E__inference_dropout_8_layer_call_and_return_conditional_losses_283232
E__inference_dropout_8_layer_call_and_return_conditional_losses_283244�
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
(__inference_dense_8_layer_call_fn_283253�
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
C__inference_dense_8_layer_call_and_return_conditional_losses_283263�
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
$__inference_signature_wrapper_281262lstm_16_input"�
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
-__inference_lstm_cell_32_layer_call_fn_283280
-__inference_lstm_cell_32_layer_call_fn_283297�
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283329
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283361�
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
-__inference_lstm_cell_33_layer_call_fn_283378
-__inference_lstm_cell_33_layer_call_fn_283395�
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283427
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283459�
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
!__inference__wrapped_model_279093y&'()*+:�7
0�-
+�(
lstm_16_input���������
� "1�.
,
dense_8!�
dense_8����������
C__inference_dense_8_layer_call_and_return_conditional_losses_283263\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� {
(__inference_dense_8_layer_call_fn_283253O/�,
%�"
 �
inputs��������� 
� "�����������
E__inference_dropout_8_layer_call_and_return_conditional_losses_283232\3�0
)�&
 �
inputs��������� 
p 
� "%�"
�
0��������� 
� �
E__inference_dropout_8_layer_call_and_return_conditional_losses_283244\3�0
)�&
 �
inputs��������� 
p
� "%�"
�
0��������� 
� }
*__inference_dropout_8_layer_call_fn_283222O3�0
)�&
 �
inputs��������� 
p 
� "���������� }
*__inference_dropout_8_layer_call_fn_283227O3�0
)�&
 �
inputs��������� 
p
� "���������� �
C__inference_lstm_16_layer_call_and_return_conditional_losses_282116�&'(O�L
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282267�&'(O�L
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282418q&'(?�<
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
C__inference_lstm_16_layer_call_and_return_conditional_losses_282569q&'(?�<
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
(__inference_lstm_16_layer_call_fn_281932}&'(O�L
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
(__inference_lstm_16_layer_call_fn_281943}&'(O�L
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
(__inference_lstm_16_layer_call_fn_281954d&'(?�<
5�2
$�!
inputs���������

 
p 

 
� "����������@�
(__inference_lstm_16_layer_call_fn_281965d&'(?�<
5�2
$�!
inputs���������

 
p

 
� "����������@�
C__inference_lstm_17_layer_call_and_return_conditional_losses_282764})*+O�L
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_282915})*+O�L
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_283066m)*+?�<
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
C__inference_lstm_17_layer_call_and_return_conditional_losses_283217m)*+?�<
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
(__inference_lstm_17_layer_call_fn_282580p)*+O�L
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
(__inference_lstm_17_layer_call_fn_282591p)*+O�L
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
(__inference_lstm_17_layer_call_fn_282602`)*+?�<
5�2
$�!
inputs���������@

 
p 

 
� "���������� �
(__inference_lstm_17_layer_call_fn_282613`)*+?�<
5�2
$�!
inputs���������@

 
p

 
� "���������� �
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283329�&'(��}
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
H__inference_lstm_cell_32_layer_call_and_return_conditional_losses_283361�&'(��}
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
-__inference_lstm_cell_32_layer_call_fn_283280�&'(��}
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
-__inference_lstm_cell_32_layer_call_fn_283297�&'(��}
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283427�)*+��}
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
H__inference_lstm_cell_33_layer_call_and_return_conditional_losses_283459�)*+��}
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
-__inference_lstm_cell_33_layer_call_fn_283378�)*+��}
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
-__inference_lstm_cell_33_layer_call_fn_283395�)*+��}
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_281209u&'()*+B�?
8�5
+�(
lstm_16_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_sequential_8_layer_call_and_return_conditional_losses_281233u&'()*+B�?
8�5
+�(
lstm_16_input���������
p

 
� "%�"
�
0���������
� �
H__inference_sequential_8_layer_call_and_return_conditional_losses_281609n&'()*+;�8
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
H__inference_sequential_8_layer_call_and_return_conditional_losses_281921n&'()*+;�8
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
-__inference_sequential_8_layer_call_fn_280719h&'()*+B�?
8�5
+�(
lstm_16_input���������
p 

 
� "�����������
-__inference_sequential_8_layer_call_fn_281185h&'()*+B�?
8�5
+�(
lstm_16_input���������
p

 
� "�����������
-__inference_sequential_8_layer_call_fn_281283a&'()*+;�8
1�.
$�!
inputs���������
p 

 
� "�����������
-__inference_sequential_8_layer_call_fn_281304a&'()*+;�8
1�.
$�!
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_281262�&'()*+K�H
� 
A�>
<
lstm_16_input+�(
lstm_16_input���������"1�.
,
dense_8!�
dense_8���������