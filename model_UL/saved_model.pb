Рб%
щ┐
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
╛
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
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
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
л
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
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
И"serve*2.6.22v2.6.1-9-gc2363d6d0258ар#
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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
П
lstm_2/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А**
shared_namelstm_2/lstm_cell_4/kernel
И
-lstm_2/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_4/kernel*
_output_shapes
:	А*
dtype0
г
#lstm_2/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*4
shared_name%#lstm_2/lstm_cell_4/recurrent_kernel
Ь
7lstm_2/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_2/lstm_cell_4/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm_2/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_2/lstm_cell_4/bias
А
+lstm_2/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_2/lstm_cell_4/bias*
_output_shapes	
:А*
dtype0
П
lstm_3/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А**
shared_namelstm_3/lstm_cell_5/kernel
И
-lstm_3/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_5/kernel*
_output_shapes
:	@А*
dtype0
г
#lstm_3/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*4
shared_name%#lstm_3/lstm_cell_5/recurrent_kernel
Ь
7lstm_3/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_5/recurrent_kernel*
_output_shapes
:	 А*
dtype0
З
lstm_3/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_3/lstm_cell_5/bias
А
+lstm_3/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_5/bias*
_output_shapes	
:А*
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
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
Э
 Adam/lstm_2/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_2/lstm_cell_4/kernel/m
Ц
4Adam/lstm_2/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_4/kernel/m*
_output_shapes
:	А*
dtype0
▒
*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m
к
>Adam/lstm_2/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_2/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_2/lstm_cell_4/bias/m
О
2Adam/lstm_2/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_4/bias/m*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_3/lstm_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_3/lstm_cell_5/kernel/m
Ц
4Adam/lstm_3/lstm_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_5/kernel/m*
_output_shapes
:	@А*
dtype0
▒
*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m
к
>Adam/lstm_3/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_3/lstm_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_3/lstm_cell_5/bias/m
О
2Adam/lstm_3/lstm_cell_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_5/bias/m*
_output_shapes	
:А*
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
Э
 Adam/lstm_2/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/lstm_2/lstm_cell_4/kernel/v
Ц
4Adam/lstm_2/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_2/lstm_cell_4/kernel/v*
_output_shapes
:	А*
dtype0
▒
*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v
к
>Adam/lstm_2/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_2/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_2/lstm_cell_4/bias/v
О
2Adam/lstm_2/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_2/lstm_cell_4/bias/v*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_3/lstm_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_3/lstm_cell_5/kernel/v
Ц
4Adam/lstm_3/lstm_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_5/kernel/v*
_output_shapes
:	@А*
dtype0
▒
*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v
к
>Adam/lstm_3/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_3/lstm_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_3/lstm_cell_5/bias/v
О
2Adam/lstm_3/lstm_cell_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_5/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
Т3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*═2
value├2B└2 B╣2
є
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
╨
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
н
trainable_variables
regularization_losses
,layer_metrics

-layers
	variables
.metrics
/non_trainable_variables
0layer_regularization_losses
 
О
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
╣
trainable_variables
regularization_losses
6layer_metrics

7layers
	variables
8layer_regularization_losses
9metrics
:non_trainable_variables

;states
О
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
╣
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
н
trainable_variables
regularization_losses
Glayer_metrics

Hlayers
	variables
Imetrics
Jnon_trainable_variables
Klayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
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
_]
VARIABLE_VALUElstm_2/lstm_cell_4/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_2/lstm_cell_4/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_2/lstm_cell_4/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_3/lstm_cell_5/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_3/lstm_cell_5/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_3/lstm_cell_5/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
н
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
н
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
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_2/lstm_cell_4/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_2/lstm_cell_4/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_3/lstm_cell_5/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_5/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
З
serving_default_lstm_2_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
Ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_2_inputlstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_50431
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╪
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_2/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_2/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_2/lstm_cell_4/bias/Read/ReadVariableOp-lstm_3/lstm_cell_5/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp4Adam/lstm_2/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_2/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_2/lstm_cell_4/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_5/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_5/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_5/bias/v/Read/ReadVariableOpConst*,
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
GPU 2J 8В *'
f"R 
__inference__traced_save_52744
ч
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_2/lstm_cell_4/kernel#lstm_2/lstm_cell_4/recurrent_kernellstm_2/lstm_cell_4/biaslstm_3/lstm_cell_5/kernel#lstm_3/lstm_cell_5/recurrent_kernellstm_3/lstm_cell_5/biastotalcountAdam/dense_1/kernel/mAdam/dense_1/bias/m Adam/lstm_2/lstm_cell_4/kernel/m*Adam/lstm_2/lstm_cell_4/recurrent_kernel/mAdam/lstm_2/lstm_cell_4/bias/m Adam/lstm_3/lstm_cell_5/kernel/m*Adam/lstm_3/lstm_cell_5/recurrent_kernel/mAdam/lstm_3/lstm_cell_5/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/v Adam/lstm_2/lstm_cell_4/kernel/v*Adam/lstm_2/lstm_cell_4/recurrent_kernel/vAdam/lstm_2/lstm_cell_4/bias/v Adam/lstm_3/lstm_cell_5/kernel/v*Adam/lstm_3/lstm_cell_5/recurrent_kernel/vAdam/lstm_3/lstm_cell_5/bias/v*+
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_52847ш═"
└H
ж

lstm_2_while_body_50845*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0:	АN
;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АI
:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorJ
7lstm_2_while_lstm_cell_4_matmul_readvariableop_resource:	АL
9lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource:	@АG
8lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource:	АИв/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpв.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpв0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp╤
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2@
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype022
0lstm_2/while/TensorArrayV2Read/TensorListGetItem█
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype020
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpЁ
lstm_2/while/lstm_cell_4/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_2/while/lstm_cell_4/MatMulс
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype022
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp┘
!lstm_2/while/lstm_cell_4/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_2/while/lstm_cell_4/MatMul_1╨
lstm_2/while/lstm_cell_4/addAddV2)lstm_2/while/lstm_cell_4/MatMul:product:0+lstm_2/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_2/while/lstm_cell_4/add┌
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp▌
 lstm_2/while/lstm_cell_4/BiasAddBiasAdd lstm_2/while/lstm_cell_4/add:z:07lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_2/while/lstm_cell_4/BiasAddЦ
(lstm_2/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_2/while/lstm_cell_4/split/split_dimг
lstm_2/while/lstm_cell_4/splitSplit1lstm_2/while/lstm_cell_4/split/split_dim:output:0)lstm_2/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2 
lstm_2/while/lstm_cell_4/splitк
 lstm_2/while/lstm_cell_4/SigmoidSigmoid'lstm_2/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2"
 lstm_2/while/lstm_cell_4/Sigmoidо
"lstm_2/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2$
"lstm_2/while/lstm_cell_4/Sigmoid_1╣
lstm_2/while/lstm_cell_4/mulMul&lstm_2/while/lstm_cell_4/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:         @2
lstm_2/while/lstm_cell_4/mulб
lstm_2/while/lstm_cell_4/ReluRelu'lstm_2/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_2/while/lstm_cell_4/Relu╠
lstm_2/while/lstm_cell_4/mul_1Mul$lstm_2/while/lstm_cell_4/Sigmoid:y:0+lstm_2/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/mul_1┴
lstm_2/while/lstm_cell_4/add_1AddV2 lstm_2/while/lstm_cell_4/mul:z:0"lstm_2/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/add_1о
"lstm_2/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2$
"lstm_2/while/lstm_cell_4/Sigmoid_2а
lstm_2/while/lstm_cell_4/Relu_1Relu"lstm_2/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2!
lstm_2/while/lstm_cell_4/Relu_1╨
lstm_2/while/lstm_cell_4/mul_2Mul&lstm_2/while/lstm_cell_4/Sigmoid_2:y:0-lstm_2/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/mul_2В
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_2/while/TensorArrayV2Write/TensorListSetItemj
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add/yЕ
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/addn
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add_1/yЩ
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/add_1З
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identityб
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_1Й
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_2╢
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_3и
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_4/mul_2:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_2/while/Identity_4и
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_4/add_1:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_2/while/Identity_5■
lstm_2/while/NoOpNoOp0^lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_2/while/NoOp"7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_4_matmul_readvariableop_resource9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0"─
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2b
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
¤E
№
A__inference_lstm_3_layer_call_and_return_conditional_losses_49260

inputs$
lstm_cell_5_49178:	@А$
lstm_cell_5_49180:	 А 
lstm_cell_5_49182:	А
identityИв#lstm_cell_5/StatefulPartitionedCallвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2У
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_49178lstm_cell_5_49180lstm_cell_5_49182*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_491132%
#lstm_cell_5/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╕
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_49178lstm_cell_5_49180lstm_cell_5_49182*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_49191*
condR
while_cond_49190*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
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
:          2

Identity|
NoOpNoOp$^lstm_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╨
╛
while_cond_51653
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51653___redundant_placeholder03
/while_while_cond_51653___redundant_placeholder13
/while_while_cond_51653___redundant_placeholder23
/while_while_cond_51653___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
д
╞
G__inference_sequential_1_layer_call_and_return_conditional_losses_49869

inputs
lstm_2_49680:	А
lstm_2_49682:	@А
lstm_2_49684:	А
lstm_3_49838:	@А
lstm_3_49840:	 А
lstm_3_49842:	А
dense_1_49863: 
dense_1_49865:
identityИвdense_1/StatefulPartitionedCallвlstm_2/StatefulPartitionedCallвlstm_3/StatefulPartitionedCallЫ
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_49680lstm_2_49682lstm_2_49684*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_496792 
lstm_2/StatefulPartitionedCall╕
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_49838lstm_3_49840lstm_3_49842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_498372 
lstm_3/StatefulPartitionedCallў
dropout_1/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_498502
dropout_1/PartitionedCallи
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_49863dense_1_49865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_498622!
dense_1/StatefulPartitionedCallГ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity▓
NoOpNoOp ^dense_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╧[
Х
A__inference_lstm_2_layer_call_and_return_conditional_losses_51436
inputs_0=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_51352*
condR
while_cond_51351*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
 :                  @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╡К
й	
 __inference__wrapped_model_48262
lstm_2_inputQ
>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resource:	АS
@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resource:	@АN
?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resource:	АQ
>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resource:	@АS
@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resource:	 АN
?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource:	АE
3sequential_1_dense_1_matmul_readvariableop_resource: B
4sequential_1_dense_1_biasadd_readvariableop_resource:
identityИв+sequential_1/dense_1/BiasAdd/ReadVariableOpв*sequential_1/dense_1/MatMul/ReadVariableOpв6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpв5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOpв7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpвsequential_1/lstm_2/whileв6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpв5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOpв7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpвsequential_1/lstm_3/whiler
sequential_1/lstm_2/ShapeShapelstm_2_input*
T0*
_output_shapes
:2
sequential_1/lstm_2/ShapeЬ
'sequential_1/lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_2/strided_slice/stackа
)sequential_1/lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_2/strided_slice/stack_1а
)sequential_1/lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_2/strided_slice/stack_2┌
!sequential_1/lstm_2/strided_sliceStridedSlice"sequential_1/lstm_2/Shape:output:00sequential_1/lstm_2/strided_slice/stack:output:02sequential_1/lstm_2/strided_slice/stack_1:output:02sequential_1/lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_2/strided_sliceД
sequential_1/lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2!
sequential_1/lstm_2/zeros/mul/y╝
sequential_1/lstm_2/zeros/mulMul*sequential_1/lstm_2/strided_slice:output:0(sequential_1/lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_2/zeros/mulЗ
 sequential_1/lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_1/lstm_2/zeros/Less/y╖
sequential_1/lstm_2/zeros/LessLess!sequential_1/lstm_2/zeros/mul:z:0)sequential_1/lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_2/zeros/LessК
"sequential_1/lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_1/lstm_2/zeros/packed/1╙
 sequential_1/lstm_2/zeros/packedPack*sequential_1/lstm_2/strided_slice:output:0+sequential_1/lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_2/zeros/packedЗ
sequential_1/lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_2/zeros/Const┼
sequential_1/lstm_2/zerosFill)sequential_1/lstm_2/zeros/packed:output:0(sequential_1/lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
sequential_1/lstm_2/zerosИ
!sequential_1/lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_1/lstm_2/zeros_1/mul/y┬
sequential_1/lstm_2/zeros_1/mulMul*sequential_1/lstm_2/strided_slice:output:0*sequential_1/lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_2/zeros_1/mulЛ
"sequential_1/lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_1/lstm_2/zeros_1/Less/y┐
 sequential_1/lstm_2/zeros_1/LessLess#sequential_1/lstm_2/zeros_1/mul:z:0+sequential_1/lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_2/zeros_1/LessО
$sequential_1/lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/lstm_2/zeros_1/packed/1┘
"sequential_1/lstm_2/zeros_1/packedPack*sequential_1/lstm_2/strided_slice:output:0-sequential_1/lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_2/zeros_1/packedЛ
!sequential_1/lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_2/zeros_1/Const═
sequential_1/lstm_2/zeros_1Fill+sequential_1/lstm_2/zeros_1/packed:output:0*sequential_1/lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
sequential_1/lstm_2/zeros_1Э
"sequential_1/lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_2/transpose/perm╝
sequential_1/lstm_2/transpose	Transposelstm_2_input+sequential_1/lstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:         2
sequential_1/lstm_2/transposeЛ
sequential_1/lstm_2/Shape_1Shape!sequential_1/lstm_2/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_2/Shape_1а
)sequential_1/lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_2/strided_slice_1/stackд
+sequential_1/lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_1/stack_1д
+sequential_1/lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_1/stack_2ц
#sequential_1/lstm_2/strided_slice_1StridedSlice$sequential_1/lstm_2/Shape_1:output:02sequential_1/lstm_2/strided_slice_1/stack:output:04sequential_1/lstm_2/strided_slice_1/stack_1:output:04sequential_1/lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_1н
/sequential_1/lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         21
/sequential_1/lstm_2/TensorArrayV2/element_shapeВ
!sequential_1/lstm_2/TensorArrayV2TensorListReserve8sequential_1/lstm_2/TensorArrayV2/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_2/TensorArrayV2ч
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2K
Isequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape╚
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_2/transpose:y:0Rsequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensorа
)sequential_1/lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_2/strided_slice_2/stackд
+sequential_1/lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_2/stack_1д
+sequential_1/lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_2/stack_2Ї
#sequential_1/lstm_2/strided_slice_2StridedSlice!sequential_1/lstm_2/transpose:y:02sequential_1/lstm_2/strided_slice_2/stack:output:04sequential_1/lstm_2/strided_slice_2/stack_1:output:04sequential_1/lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_2ю
5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype027
5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp·
&sequential_1/lstm_2/lstm_cell_4/MatMulMatMul,sequential_1/lstm_2/strided_slice_2:output:0=sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2(
&sequential_1/lstm_2/lstm_cell_4/MatMulЇ
7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype029
7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpЎ
(sequential_1/lstm_2/lstm_cell_4/MatMul_1MatMul"sequential_1/lstm_2/zeros:output:0?sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(sequential_1/lstm_2/lstm_cell_4/MatMul_1ь
#sequential_1/lstm_2/lstm_cell_4/addAddV20sequential_1/lstm_2/lstm_cell_4/MatMul:product:02sequential_1/lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2%
#sequential_1/lstm_2/lstm_cell_4/addэ
6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp∙
'sequential_1/lstm_2/lstm_cell_4/BiasAddBiasAdd'sequential_1/lstm_2/lstm_cell_4/add:z:0>sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2)
'sequential_1/lstm_2/lstm_cell_4/BiasAddд
/sequential_1/lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_2/lstm_cell_4/split/split_dim┐
%sequential_1/lstm_2/lstm_cell_4/splitSplit8sequential_1/lstm_2/lstm_cell_4/split/split_dim:output:00sequential_1/lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2'
%sequential_1/lstm_2/lstm_cell_4/split┐
'sequential_1/lstm_2/lstm_cell_4/SigmoidSigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2)
'sequential_1/lstm_2/lstm_cell_4/Sigmoid├
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_1Sigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2+
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_1╪
#sequential_1/lstm_2/lstm_cell_4/mulMul-sequential_1/lstm_2/lstm_cell_4/Sigmoid_1:y:0$sequential_1/lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:         @2%
#sequential_1/lstm_2/lstm_cell_4/mul╢
$sequential_1/lstm_2/lstm_cell_4/ReluRelu.sequential_1/lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2&
$sequential_1/lstm_2/lstm_cell_4/Reluш
%sequential_1/lstm_2/lstm_cell_4/mul_1Mul+sequential_1/lstm_2/lstm_cell_4/Sigmoid:y:02sequential_1/lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2'
%sequential_1/lstm_2/lstm_cell_4/mul_1▌
%sequential_1/lstm_2/lstm_cell_4/add_1AddV2'sequential_1/lstm_2/lstm_cell_4/mul:z:0)sequential_1/lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2'
%sequential_1/lstm_2/lstm_cell_4/add_1├
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_2Sigmoid.sequential_1/lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2+
)sequential_1/lstm_2/lstm_cell_4/Sigmoid_2╡
&sequential_1/lstm_2/lstm_cell_4/Relu_1Relu)sequential_1/lstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2(
&sequential_1/lstm_2/lstm_cell_4/Relu_1ь
%sequential_1/lstm_2/lstm_cell_4/mul_2Mul-sequential_1/lstm_2/lstm_cell_4/Sigmoid_2:y:04sequential_1/lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2'
%sequential_1/lstm_2/lstm_cell_4/mul_2╖
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   23
1sequential_1/lstm_2/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_2/TensorArrayV2_1TensorListReserve:sequential_1/lstm_2/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_2/TensorArrayV2_1v
sequential_1/lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_2/timeз
,sequential_1/lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,sequential_1/lstm_2/while/maximum_iterationsТ
&sequential_1/lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_2/while/loop_counter▓
sequential_1/lstm_2/whileWhile/sequential_1/lstm_2/while/loop_counter:output:05sequential_1/lstm_2/while/maximum_iterations:output:0!sequential_1/lstm_2/time:output:0,sequential_1/lstm_2/TensorArrayV2_1:handle:0"sequential_1/lstm_2/zeros:output:0$sequential_1/lstm_2/zeros_1:output:0,sequential_1/lstm_2/strided_slice_1:output:0Ksequential_1/lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_2_lstm_cell_4_matmul_readvariableop_resource@sequential_1_lstm_2_lstm_cell_4_matmul_1_readvariableop_resource?sequential_1_lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_2_while_body_48024*0
cond(R&
$sequential_1_lstm_2_while_cond_48023*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
sequential_1/lstm_2/while▌
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2F
Dsequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape╕
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_2/while:output:3Msequential_1/lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype028
6sequential_1/lstm_2/TensorArrayV2Stack/TensorListStackй
)sequential_1/lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2+
)sequential_1/lstm_2/strided_slice_3/stackд
+sequential_1/lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_2/strided_slice_3/stack_1д
+sequential_1/lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_2/strided_slice_3/stack_2Т
#sequential_1/lstm_2/strided_slice_3StridedSlice?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_2/strided_slice_3/stack:output:04sequential_1/lstm_2/strided_slice_3/stack_1:output:04sequential_1/lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2%
#sequential_1/lstm_2/strided_slice_3б
$sequential_1/lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_2/transpose_1/permї
sequential_1/lstm_2/transpose_1	Transpose?sequential_1/lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2!
sequential_1/lstm_2/transpose_1О
sequential_1/lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_2/runtimeЙ
sequential_1/lstm_3/ShapeShape#sequential_1/lstm_2/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/ShapeЬ
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_3/strided_slice/stackа
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_1а
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_2┌
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_3/strided_sliceД
sequential_1/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_1/lstm_3/zeros/mul/y╝
sequential_1/lstm_3/zeros/mulMul*sequential_1/lstm_3/strided_slice:output:0(sequential_1/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/zeros/mulЗ
 sequential_1/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2"
 sequential_1/lstm_3/zeros/Less/y╖
sequential_1/lstm_3/zeros/LessLess!sequential_1/lstm_3/zeros/mul:z:0)sequential_1/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/zeros/LessК
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_1/lstm_3/zeros/packed/1╙
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_3/zeros/packedЗ
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_3/zeros/Const┼
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:          2
sequential_1/lstm_3/zerosИ
!sequential_1/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_1/lstm_3/zeros_1/mul/y┬
sequential_1/lstm_3/zeros_1/mulMul*sequential_1/lstm_3/strided_slice:output:0*sequential_1/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/zeros_1/mulЛ
"sequential_1/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_1/lstm_3/zeros_1/Less/y┐
 sequential_1/lstm_3/zeros_1/LessLess#sequential_1/lstm_3/zeros_1/mul:z:0+sequential_1/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_3/zeros_1/LessО
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_1/lstm_3/zeros_1/packed/1┘
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_3/zeros_1/packedЛ
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_3/zeros_1/Const═
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
sequential_1/lstm_3/zeros_1Э
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_3/transpose/perm╙
sequential_1/lstm_3/transpose	Transpose#sequential_1/lstm_2/transpose_1:y:0+sequential_1/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
sequential_1/lstm_3/transposeЛ
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/Shape_1а
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_1/stackд
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_1д
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_2ц
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_1н
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         21
/sequential_1/lstm_3/TensorArrayV2/element_shapeВ
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_3/TensorArrayV2ч
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2K
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape╚
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorа
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_2/stackд
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_1д
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_2Ї
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_2ю
5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype027
5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp·
&sequential_1/lstm_3/lstm_cell_5/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0=sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2(
&sequential_1/lstm_3/lstm_cell_5/MatMulЇ
7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype029
7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpЎ
(sequential_1/lstm_3/lstm_cell_5/MatMul_1MatMul"sequential_1/lstm_3/zeros:output:0?sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(sequential_1/lstm_3/lstm_cell_5/MatMul_1ь
#sequential_1/lstm_3/lstm_cell_5/addAddV20sequential_1/lstm_3/lstm_cell_5/MatMul:product:02sequential_1/lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2%
#sequential_1/lstm_3/lstm_cell_5/addэ
6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp∙
'sequential_1/lstm_3/lstm_cell_5/BiasAddBiasAdd'sequential_1/lstm_3/lstm_cell_5/add:z:0>sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2)
'sequential_1/lstm_3/lstm_cell_5/BiasAddд
/sequential_1/lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_3/lstm_cell_5/split/split_dim┐
%sequential_1/lstm_3/lstm_cell_5/splitSplit8sequential_1/lstm_3/lstm_cell_5/split/split_dim:output:00sequential_1/lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2'
%sequential_1/lstm_3/lstm_cell_5/split┐
'sequential_1/lstm_3/lstm_cell_5/SigmoidSigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2)
'sequential_1/lstm_3/lstm_cell_5/Sigmoid├
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_1Sigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2+
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_1╪
#sequential_1/lstm_3/lstm_cell_5/mulMul-sequential_1/lstm_3/lstm_cell_5/Sigmoid_1:y:0$sequential_1/lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:          2%
#sequential_1/lstm_3/lstm_cell_5/mul╢
$sequential_1/lstm_3/lstm_cell_5/ReluRelu.sequential_1/lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2&
$sequential_1/lstm_3/lstm_cell_5/Reluш
%sequential_1/lstm_3/lstm_cell_5/mul_1Mul+sequential_1/lstm_3/lstm_cell_5/Sigmoid:y:02sequential_1/lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2'
%sequential_1/lstm_3/lstm_cell_5/mul_1▌
%sequential_1/lstm_3/lstm_cell_5/add_1AddV2'sequential_1/lstm_3/lstm_cell_5/mul:z:0)sequential_1/lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2'
%sequential_1/lstm_3/lstm_cell_5/add_1├
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_2Sigmoid.sequential_1/lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2+
)sequential_1/lstm_3/lstm_cell_5/Sigmoid_2╡
&sequential_1/lstm_3/lstm_cell_5/Relu_1Relu)sequential_1/lstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2(
&sequential_1/lstm_3/lstm_cell_5/Relu_1ь
%sequential_1/lstm_3/lstm_cell_5/mul_2Mul-sequential_1/lstm_3/lstm_cell_5/Sigmoid_2:y:04sequential_1/lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2'
%sequential_1/lstm_3/lstm_cell_5/mul_2╖
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_3/TensorArrayV2_1v
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_3/timeз
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,sequential_1/lstm_3/while/maximum_iterationsТ
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_3/while/loop_counter▓
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_3_lstm_cell_5_matmul_readvariableop_resource@sequential_1_lstm_3_lstm_cell_5_matmul_1_readvariableop_resource?sequential_1_lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *0
body(R&
$sequential_1_lstm_3_while_body_48171*0
cond(R&
$sequential_1_lstm_3_while_cond_48170*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
sequential_1/lstm_3/while▌
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape╕
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype028
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackй
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2+
)sequential_1/lstm_3/strided_slice_3/stackд
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_3/strided_slice_3/stack_1д
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_3/stack_2Т
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_3б
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_3/transpose_1/permї
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2!
sequential_1/lstm_3/transpose_1О
sequential_1/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_3/runtimeо
sequential_1/dropout_1/IdentityIdentity,sequential_1/lstm_3/strided_slice_3:output:0*
T0*'
_output_shapes
:          2!
sequential_1/dropout_1/Identity╠
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp╘
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/MatMul╦
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp╒
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_1/dense_1/BiasAddА
IdentityIdentity%sequential_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╖
NoOpNoOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp7^sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp6^sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp8^sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp^sequential_1/lstm_2/while7^sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp6^sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp8^sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp^sequential_1/lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2p
6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp6sequential_1/lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp5sequential_1/lstm_2/lstm_cell_4/MatMul/ReadVariableOp2r
7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp7sequential_1/lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp26
sequential_1/lstm_2/whilesequential_1/lstm_2/while2p
6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp6sequential_1/lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp5sequential_1/lstm_3/lstm_cell_5/MatMul/ReadVariableOp2r
7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp7sequential_1/lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp26
sequential_1/lstm_3/whilesequential_1/lstm_3/while:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
в
│
&__inference_lstm_3_layer_call_fn_51771

inputs
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_498372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
к
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_52413

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
∙
Д
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52596

inputs
states_0
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:          2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2Щ
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
?:         @:          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
╨
╛
while_cond_48560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_48560___redundant_placeholder03
/while_while_cond_48560___redundant_placeholder13
/while_while_cond_48560___redundant_placeholder23
/while_while_cond_48560___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
┴>
╞
while_body_51201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
▓
Ї
+__inference_lstm_cell_4_layer_call_fn_52449

inputs
states_0
states_1
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_483372
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @2

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
?:         :         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
┴>
╞
while_body_49595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
┴>
╞
while_body_51352
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
г

╩
lstm_2_while_cond_50539*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_50539___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_50539___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_50539___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_50539___redundant_placeholder3
lstm_2_while_identity
У
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
lstm_2/while/Lessr
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_2/while/Identity"7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╨
╛
while_cond_51351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51351___redundant_placeholder03
/while_while_cond_51351___redundant_placeholder13
/while_while_cond_51351___redundant_placeholder23
/while_while_cond_51351___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
║
╡
&__inference_lstm_3_layer_call_fn_51760
inputs_0
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_492602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
д

є
B__inference_dense_1_layer_call_and_return_conditional_losses_52432

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┴>
╞
while_body_51849
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
ё
В
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_48483

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
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
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates
БF
№
A__inference_lstm_2_layer_call_and_return_conditional_losses_48630

inputs$
lstm_cell_4_48548:	А$
lstm_cell_4_48550:	@А 
lstm_cell_4_48552:	А
identityИв#lstm_cell_4/StatefulPartitionedCallвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2У
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_48548lstm_cell_4_48550lstm_cell_4_48552*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_484832%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╕
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_48548lstm_cell_4_48550lstm_cell_4_48552*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_48561*
condR
while_cond_48560*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
 :                  @2

Identity|
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
г

╩
lstm_3_while_cond_50686*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_50686___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_50686___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_50686___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_50686___redundant_placeholder3
lstm_3_while_identity
У
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
ї▀
р
G__inference_sequential_1_layer_call_and_return_conditional_losses_51090

inputsD
1lstm_2_lstm_cell_4_matmul_readvariableop_resource:	АF
3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource:	@АA
2lstm_2_lstm_cell_4_biasadd_readvariableop_resource:	АD
1lstm_3_lstm_cell_5_matmul_readvariableop_resource:	@АF
3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource:	 АA
2lstm_3_lstm_cell_5_biasadd_readvariableop_resource:	А8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpв(lstm_2/lstm_cell_4/MatMul/ReadVariableOpв*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpвlstm_2/whileв)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpв(lstm_3/lstm_cell_5/MatMul/ReadVariableOpв*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpвlstm_3/whileR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/ShapeВ
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stackЖ
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1Ж
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2М
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicej
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/mul/yИ
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_2/zeros/Less/yГ
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessp
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/packed/1Я
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/ConstС
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_2/zerosn
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/mul/yО
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_2/zeros_1/Less/yЛ
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lesst
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/packed/1е
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/ConstЩ
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_2/zeros_1Г
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/permП
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1Ж
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stackК
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1К
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2Ш
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1У
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_2/TensorArrayV2/element_shape╬
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2═
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensorЖ
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stackК
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1К
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2ж
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_2/strided_slice_2╟
(lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02*
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp╞
lstm_2/lstm_cell_4/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/MatMul═
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp┬
lstm_2/lstm_cell_4/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/MatMul_1╕
lstm_2/lstm_cell_4/addAddV2#lstm_2/lstm_cell_4/MatMul:product:0%lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/add╞
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp┼
lstm_2/lstm_cell_4/BiasAddBiasAddlstm_2/lstm_cell_4/add:z:01lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/BiasAddК
"lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_4/split/split_dimЛ
lstm_2/lstm_cell_4/splitSplit+lstm_2/lstm_cell_4/split/split_dim:output:0#lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_2/lstm_cell_4/splitШ
lstm_2/lstm_cell_4/SigmoidSigmoid!lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/SigmoidЬ
lstm_2/lstm_cell_4/Sigmoid_1Sigmoid!lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Sigmoid_1д
lstm_2/lstm_cell_4/mulMul lstm_2/lstm_cell_4/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mulП
lstm_2/lstm_cell_4/ReluRelu!lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Relu┤
lstm_2/lstm_cell_4/mul_1Mullstm_2/lstm_cell_4/Sigmoid:y:0%lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mul_1й
lstm_2/lstm_cell_4/add_1AddV2lstm_2/lstm_cell_4/mul:z:0lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/add_1Ь
lstm_2/lstm_cell_4/Sigmoid_2Sigmoid!lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Sigmoid_2О
lstm_2/lstm_cell_4/Relu_1Relulstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Relu_1╕
lstm_2/lstm_cell_4/mul_2Mul lstm_2/lstm_cell_4/Sigmoid_2:y:0'lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mul_2Э
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_2/TensorArrayV2_1/element_shape╘
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/timeН
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counterя
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_4_matmul_readvariableop_resource3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_50845*#
condR
lstm_2_while_cond_50844*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
lstm_2/while├
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStackП
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_2/strided_slice_3/stackК
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1К
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2─
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_2/strided_slice_3З
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm┴
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtimeb
lstm_3/ShapeShapelstm_2/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:          2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/packed/1е
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/permЯ
lstm_3/transpose	Transposelstm_2/transpose_1:y:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_3/TensorArrayV2/element_shape╬
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2═
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2ж
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_3/strided_slice_2╟
(lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp╞
lstm_3/lstm_cell_5/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/MatMul═
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp┬
lstm_3/lstm_cell_5/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/MatMul_1╕
lstm_3/lstm_cell_5/addAddV2#lstm_3/lstm_cell_5/MatMul:product:0%lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/add╞
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp┼
lstm_3/lstm_cell_5/BiasAddBiasAddlstm_3/lstm_cell_5/add:z:01lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/BiasAddК
"lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_5/split/split_dimЛ
lstm_3/lstm_cell_5/splitSplit+lstm_3/lstm_cell_5/split/split_dim:output:0#lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_3/lstm_cell_5/splitШ
lstm_3/lstm_cell_5/SigmoidSigmoid!lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/SigmoidЬ
lstm_3/lstm_cell_5/Sigmoid_1Sigmoid!lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Sigmoid_1д
lstm_3/lstm_cell_5/mulMul lstm_3/lstm_cell_5/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mulП
lstm_3/lstm_cell_5/ReluRelu!lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Relu┤
lstm_3/lstm_cell_5/mul_1Mullstm_3/lstm_cell_5/Sigmoid:y:0%lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mul_1й
lstm_3/lstm_cell_5/add_1AddV2lstm_3/lstm_cell_5/mul:z:0lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/add_1Ь
lstm_3/lstm_cell_5/Sigmoid_2Sigmoid!lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Sigmoid_2О
lstm_3/lstm_cell_5/Relu_1Relulstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Relu_1╕
lstm_3/lstm_cell_5/mul_2Mul lstm_3/lstm_cell_5/Sigmoid_2:y:0'lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_3/TensorArrayV2_1/element_shape╘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterя
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_5_matmul_readvariableop_resource3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_50992*#
condR
lstm_3_while_cond_50991*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
lstm_3/while├
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2─
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm┴
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_1/dropout/Constк
dropout_1/dropout/MulMullstm_3/strided_slice_3:output:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout_1/dropout/MulБ
dropout_1/dropout/ShapeShapelstm_3/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╥
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2"
 dropout_1/dropout/GreaterEqual/yц
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2 
dropout_1/dropout/GreaterEqualЭ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_1/dropout/Castв
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_1/dropout/Mul_1е
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╡
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_4/MatMul/ReadVariableOp+^lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp^lstm_2/while*^lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_5/MatMul/ReadVariableOp+^lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp^lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp(lstm_2/lstm_cell_4/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while2V
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp(lstm_3/lstm_cell_5/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╨
╛
while_cond_49190
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_49190___redundant_placeholder03
/while_while_cond_49190___redundant_placeholder13
/while_while_cond_49190___redundant_placeholder23
/while_while_cond_49190___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╨
╛
while_cond_52150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_52150___redundant_placeholder03
/while_while_cond_52150___redundant_placeholder13
/while_while_cond_52150___redundant_placeholder23
/while_while_cond_52150___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╦[
Х
A__inference_lstm_3_layer_call_and_return_conditional_losses_51933
inputs_0=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_51849*
condR
while_cond_51848*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
╒
╡
&__inference_lstm_2_layer_call_fn_51101
inputs_0
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_484202
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
╧[
Х
A__inference_lstm_2_layer_call_and_return_conditional_losses_51285
inputs_0=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_51201*
condR
while_cond_51200*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
 :                  @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
┴>
╞
while_body_50001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
└H
ж

lstm_3_while_body_50687*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0:	@АN
;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АI
:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
7lstm_3_while_lstm_cell_5_matmul_readvariableop_resource:	@АL
9lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource:	 АG
8lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource:	АИв/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpв.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpв0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp╤
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem█
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpЁ
lstm_3/while/lstm_cell_5/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_3/while/lstm_cell_5/MatMulс
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp┘
!lstm_3/while/lstm_cell_5/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_3/while/lstm_cell_5/MatMul_1╨
lstm_3/while/lstm_cell_5/addAddV2)lstm_3/while/lstm_cell_5/MatMul:product:0+lstm_3/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_3/while/lstm_cell_5/add┌
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp▌
 lstm_3/while/lstm_cell_5/BiasAddBiasAdd lstm_3/while/lstm_cell_5/add:z:07lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_3/while/lstm_cell_5/BiasAddЦ
(lstm_3/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_5/split/split_dimг
lstm_3/while/lstm_cell_5/splitSplit1lstm_3/while/lstm_cell_5/split/split_dim:output:0)lstm_3/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2 
lstm_3/while/lstm_cell_5/splitк
 lstm_3/while/lstm_cell_5/SigmoidSigmoid'lstm_3/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2"
 lstm_3/while/lstm_cell_5/Sigmoidо
"lstm_3/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2$
"lstm_3/while/lstm_cell_5/Sigmoid_1╣
lstm_3/while/lstm_cell_5/mulMul&lstm_3/while/lstm_cell_5/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:          2
lstm_3/while/lstm_cell_5/mulб
lstm_3/while/lstm_cell_5/ReluRelu'lstm_3/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_3/while/lstm_cell_5/Relu╠
lstm_3/while/lstm_cell_5/mul_1Mul$lstm_3/while/lstm_cell_5/Sigmoid:y:0+lstm_3/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/mul_1┴
lstm_3/while/lstm_cell_5/add_1AddV2 lstm_3/while/lstm_cell_5/mul:z:0"lstm_3/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/add_1о
"lstm_3/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2$
"lstm_3/while/lstm_cell_5/Sigmoid_2а
lstm_3/while/lstm_cell_5/Relu_1Relu"lstm_3/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2!
lstm_3/while/lstm_cell_5/Relu_1╨
lstm_3/while/lstm_cell_5/mul_2Mul&lstm_3/while/lstm_cell_5/Sigmoid_2:y:0-lstm_3/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/mul_2В
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/yЕ
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/yЩ
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1З
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identityб
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1Й
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2╢
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3и
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_5/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:          2
lstm_3/while/Identity_4и
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_5/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:          2
lstm_3/while/Identity_5■
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_5_matmul_readvariableop_resource9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0"─
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2b
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
┴>
╞
while_body_52000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
ЫК
▌
!__inference__traced_restore_52847
file_prefix1
assignvariableop_dense_1_kernel: -
assignvariableop_1_dense_1_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ?
,assignvariableop_7_lstm_2_lstm_cell_4_kernel:	АI
6assignvariableop_8_lstm_2_lstm_cell_4_recurrent_kernel:	@А9
*assignvariableop_9_lstm_2_lstm_cell_4_bias:	А@
-assignvariableop_10_lstm_3_lstm_cell_5_kernel:	@АJ
7assignvariableop_11_lstm_3_lstm_cell_5_recurrent_kernel:	 А:
+assignvariableop_12_lstm_3_lstm_cell_5_bias:	А#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_1_kernel_m: 5
'assignvariableop_16_adam_dense_1_bias_m:G
4assignvariableop_17_adam_lstm_2_lstm_cell_4_kernel_m:	АQ
>assignvariableop_18_adam_lstm_2_lstm_cell_4_recurrent_kernel_m:	@АA
2assignvariableop_19_adam_lstm_2_lstm_cell_4_bias_m:	АG
4assignvariableop_20_adam_lstm_3_lstm_cell_5_kernel_m:	@АQ
>assignvariableop_21_adam_lstm_3_lstm_cell_5_recurrent_kernel_m:	 АA
2assignvariableop_22_adam_lstm_3_lstm_cell_5_bias_m:	А;
)assignvariableop_23_adam_dense_1_kernel_v: 5
'assignvariableop_24_adam_dense_1_bias_v:G
4assignvariableop_25_adam_lstm_2_lstm_cell_4_kernel_v:	АQ
>assignvariableop_26_adam_lstm_2_lstm_cell_4_recurrent_kernel_v:	@АA
2assignvariableop_27_adam_lstm_2_lstm_cell_4_bias_v:	АG
4assignvariableop_28_adam_lstm_3_lstm_cell_5_kernel_v:	@АQ
>assignvariableop_29_adam_lstm_3_lstm_cell_5_recurrent_kernel_v:	 АA
2assignvariableop_30_adam_lstm_3_lstm_cell_5_bias_v:	А
identity_32ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Ж
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Т
valueИBЕ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_2_lstm_cell_4_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╗
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_2_lstm_cell_4_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9п
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_2_lstm_cell_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╡
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_3_lstm_cell_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┐
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_3_lstm_cell_5_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12│
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_3_lstm_cell_5_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13б
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14б
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15▒
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16п
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╝
AssignVariableOp_17AssignVariableOp4assignvariableop_17_adam_lstm_2_lstm_cell_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18╞
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_lstm_2_lstm_cell_4_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19║
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_lstm_2_lstm_cell_4_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╝
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_lstm_3_lstm_cell_5_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╞
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_lstm_3_lstm_cell_5_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22║
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_lstm_3_lstm_cell_5_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▒
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24п
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╝
AssignVariableOp_25AssignVariableOp4assignvariableop_25_adam_lstm_2_lstm_cell_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╞
AssignVariableOp_26AssignVariableOp>assignvariableop_26_adam_lstm_2_lstm_cell_4_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27║
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_lstm_2_lstm_cell_4_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╝
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_lstm_3_lstm_cell_5_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╞
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_lstm_3_lstm_cell_5_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30║
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_lstm_3_lstm_cell_5_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31f
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_32Ё
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
┴>
╞
while_body_49753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
║
╡
&__inference_lstm_3_layer_call_fn_51749
inputs_0
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_490502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
└H
ж

lstm_2_while_body_50540*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3)
%lstm_2_while_lstm_2_strided_slice_1_0e
alstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0:	АN
;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АI
:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
lstm_2_while_identity
lstm_2_while_identity_1
lstm_2_while_identity_2
lstm_2_while_identity_3
lstm_2_while_identity_4
lstm_2_while_identity_5'
#lstm_2_while_lstm_2_strided_slice_1c
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensorJ
7lstm_2_while_lstm_cell_4_matmul_readvariableop_resource:	АL
9lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource:	@АG
8lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource:	АИв/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpв.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpв0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp╤
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2@
>lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0lstm_2_while_placeholderGlstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype022
0lstm_2/while/TensorArrayV2Read/TensorListGetItem█
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype020
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpЁ
lstm_2/while/lstm_cell_4/MatMulMatMul7lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_2/while/lstm_cell_4/MatMulс
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype022
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp┘
!lstm_2/while/lstm_cell_4/MatMul_1MatMullstm_2_while_placeholder_28lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_2/while/lstm_cell_4/MatMul_1╨
lstm_2/while/lstm_cell_4/addAddV2)lstm_2/while/lstm_cell_4/MatMul:product:0+lstm_2/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_2/while/lstm_cell_4/add┌
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp▌
 lstm_2/while/lstm_cell_4/BiasAddBiasAdd lstm_2/while/lstm_cell_4/add:z:07lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_2/while/lstm_cell_4/BiasAddЦ
(lstm_2/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_2/while/lstm_cell_4/split/split_dimг
lstm_2/while/lstm_cell_4/splitSplit1lstm_2/while/lstm_cell_4/split/split_dim:output:0)lstm_2/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2 
lstm_2/while/lstm_cell_4/splitк
 lstm_2/while/lstm_cell_4/SigmoidSigmoid'lstm_2/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2"
 lstm_2/while/lstm_cell_4/Sigmoidо
"lstm_2/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_2/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2$
"lstm_2/while/lstm_cell_4/Sigmoid_1╣
lstm_2/while/lstm_cell_4/mulMul&lstm_2/while/lstm_cell_4/Sigmoid_1:y:0lstm_2_while_placeholder_3*
T0*'
_output_shapes
:         @2
lstm_2/while/lstm_cell_4/mulб
lstm_2/while/lstm_cell_4/ReluRelu'lstm_2/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_2/while/lstm_cell_4/Relu╠
lstm_2/while/lstm_cell_4/mul_1Mul$lstm_2/while/lstm_cell_4/Sigmoid:y:0+lstm_2/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/mul_1┴
lstm_2/while/lstm_cell_4/add_1AddV2 lstm_2/while/lstm_cell_4/mul:z:0"lstm_2/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/add_1о
"lstm_2/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_2/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2$
"lstm_2/while/lstm_cell_4/Sigmoid_2а
lstm_2/while/lstm_cell_4/Relu_1Relu"lstm_2/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2!
lstm_2/while/lstm_cell_4/Relu_1╨
lstm_2/while/lstm_cell_4/mul_2Mul&lstm_2/while/lstm_cell_4/Sigmoid_2:y:0-lstm_2/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2 
lstm_2/while/lstm_cell_4/mul_2В
1lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_2_while_placeholder_1lstm_2_while_placeholder"lstm_2/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_2/while/TensorArrayV2Write/TensorListSetItemj
lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add/yЕ
lstm_2/while/addAddV2lstm_2_while_placeholderlstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/addn
lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_2/while/add_1/yЩ
lstm_2/while/add_1AddV2&lstm_2_while_lstm_2_while_loop_counterlstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_2/while/add_1З
lstm_2/while/IdentityIdentitylstm_2/while/add_1:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identityб
lstm_2/while/Identity_1Identity,lstm_2_while_lstm_2_while_maximum_iterations^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_1Й
lstm_2/while/Identity_2Identitylstm_2/while/add:z:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_2╢
lstm_2/while/Identity_3IdentityAlstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_2/while/NoOp*
T0*
_output_shapes
: 2
lstm_2/while/Identity_3и
lstm_2/while/Identity_4Identity"lstm_2/while/lstm_cell_4/mul_2:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_2/while/Identity_4и
lstm_2/while/Identity_5Identity"lstm_2/while/lstm_cell_4/add_1:z:0^lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2
lstm_2/while/Identity_5■
lstm_2/while/NoOpNoOp0^lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp/^lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp1^lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_2/while/NoOp"7
lstm_2_while_identitylstm_2/while/Identity:output:0";
lstm_2_while_identity_1 lstm_2/while/Identity_1:output:0";
lstm_2_while_identity_2 lstm_2/while/Identity_2:output:0";
lstm_2_while_identity_3 lstm_2/while/Identity_3:output:0";
lstm_2_while_identity_4 lstm_2/while/Identity_4:output:0";
lstm_2_while_identity_5 lstm_2/while/Identity_5:output:0"L
#lstm_2_while_lstm_2_strided_slice_1%lstm_2_while_lstm_2_strided_slice_1_0"v
8lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_2_while_lstm_cell_4_matmul_readvariableop_resource9lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0"─
_lstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensoralstm_2_while_tensorarrayv2read_tensorlistgetitem_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2b
/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp2`
.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp.lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp2d
0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp0lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╝
E
)__inference_dropout_1_layer_call_fn_52391

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_498502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╨
╛
while_cond_51200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51200___redundant_placeholder03
/while_while_cond_51200___redundant_placeholder13
/while_while_cond_51200___redundant_placeholder23
/while_while_cond_51200___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
¤E
№
A__inference_lstm_3_layer_call_and_return_conditional_losses_49050

inputs$
lstm_cell_5_48968:	@А$
lstm_cell_5_48970:	 А 
lstm_cell_5_48972:	А
identityИв#lstm_cell_5/StatefulPartitionedCallвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2У
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_48968lstm_cell_5_48970lstm_cell_5_48972*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_489672%
#lstm_cell_5/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╕
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_48968lstm_cell_5_48970lstm_cell_5_48972*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_48981*
condR
while_cond_48980*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
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
:          2

Identity|
NoOpNoOp$^lstm_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╨
╛
while_cond_51502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51502___redundant_placeholder03
/while_while_cond_51502___redundant_placeholder13
/while_while_cond_51502___redundant_placeholder23
/while_while_cond_51502___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
Ч

╦
,__inference_sequential_1_layer_call_fn_50354
lstm_2_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:	 А
	unknown_4:	А
	unknown_5: 
	unknown_6:
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_503142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
г

╩
lstm_3_while_cond_50991*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1A
=lstm_3_while_lstm_3_while_cond_50991___redundant_placeholder0A
=lstm_3_while_lstm_3_while_cond_50991___redundant_placeholder1A
=lstm_3_while_lstm_3_while_cond_50991___redundant_placeholder2A
=lstm_3_while_lstm_3_while_cond_50991___redundant_placeholder3
lstm_3_while_identity
У
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
д

є
B__inference_dense_1_layer_call_and_return_conditional_losses_49862

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▓
Ї
+__inference_lstm_cell_5_layer_call_fn_52564

inputs
states_0
states_1
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_491132
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          2

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
?:         @:          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
ё
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_52401

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:          2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
к
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_49918

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
г

╩
lstm_2_while_cond_50844*
&lstm_2_while_lstm_2_while_loop_counter0
,lstm_2_while_lstm_2_while_maximum_iterations
lstm_2_while_placeholder
lstm_2_while_placeholder_1
lstm_2_while_placeholder_2
lstm_2_while_placeholder_3,
(lstm_2_while_less_lstm_2_strided_slice_1A
=lstm_2_while_lstm_2_while_cond_50844___redundant_placeholder0A
=lstm_2_while_lstm_2_while_cond_50844___redundant_placeholder1A
=lstm_2_while_lstm_2_while_cond_50844___redundant_placeholder2A
=lstm_2_while_lstm_2_while_cond_50844___redundant_placeholder3
lstm_2_while_identity
У
lstm_2/while/LessLesslstm_2_while_placeholder(lstm_2_while_less_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2
lstm_2/while/Lessr
lstm_2/while/IdentityIdentitylstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_2/while/Identity"7
lstm_2_while_identitylstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
┴>
╞
while_body_52151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╨
╛
while_cond_49752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_49752___redundant_placeholder03
/while_while_cond_49752___redundant_placeholder13
/while_while_cond_49752___redundant_placeholder23
/while_while_cond_49752___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
в
│
&__inference_lstm_3_layer_call_fn_51782

inputs
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИвStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_500852
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Д[
╞
$sequential_1_lstm_2_while_body_48024D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3C
?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0
{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_1_lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0:	А[
Hsequential_1_lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АV
Gsequential_1_lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0:	А&
"sequential_1_lstm_2_while_identity(
$sequential_1_lstm_2_while_identity_1(
$sequential_1_lstm_2_while_identity_2(
$sequential_1_lstm_2_while_identity_3(
$sequential_1_lstm_2_while_identity_4(
$sequential_1_lstm_2_while_identity_5A
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1}
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensorW
Dsequential_1_lstm_2_while_lstm_cell_4_matmul_readvariableop_resource:	АY
Fsequential_1_lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource:	@АT
Esequential_1_lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource:	АИв<sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpв;sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpв=sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOpы
Ksequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2M
Ksequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape╦
=sequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_2_while_placeholderTsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02?
=sequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItemВ
;sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02=
;sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOpд
,sequential_1/lstm_2/while/lstm_cell_4/MatMulMatMulDsequential_1/lstm_2/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,sequential_1/lstm_2/while/lstm_cell_4/MatMulИ
=sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02?
=sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOpН
.sequential_1/lstm_2/while/lstm_cell_4/MatMul_1MatMul'sequential_1_lstm_2_while_placeholder_2Esequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А20
.sequential_1/lstm_2/while/lstm_cell_4/MatMul_1Д
)sequential_1/lstm_2/while/lstm_cell_4/addAddV26sequential_1/lstm_2/while/lstm_cell_4/MatMul:product:08sequential_1/lstm_2/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2+
)sequential_1/lstm_2/while/lstm_cell_4/addБ
<sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOpС
-sequential_1/lstm_2/while/lstm_cell_4/BiasAddBiasAdd-sequential_1/lstm_2/while/lstm_cell_4/add:z:0Dsequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-sequential_1/lstm_2/while/lstm_cell_4/BiasAdd░
5sequential_1/lstm_2/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_2/while/lstm_cell_4/split/split_dim╫
+sequential_1/lstm_2/while/lstm_cell_4/splitSplit>sequential_1/lstm_2/while/lstm_cell_4/split/split_dim:output:06sequential_1/lstm_2/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2-
+sequential_1/lstm_2/while/lstm_cell_4/split╤
-sequential_1/lstm_2/while/lstm_cell_4/SigmoidSigmoid4sequential_1/lstm_2/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2/
-sequential_1/lstm_2/while/lstm_cell_4/Sigmoid╒
/sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_1Sigmoid4sequential_1/lstm_2/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @21
/sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_1э
)sequential_1/lstm_2/while/lstm_cell_4/mulMul3sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_1:y:0'sequential_1_lstm_2_while_placeholder_3*
T0*'
_output_shapes
:         @2+
)sequential_1/lstm_2/while/lstm_cell_4/mul╚
*sequential_1/lstm_2/while/lstm_cell_4/ReluRelu4sequential_1/lstm_2/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2,
*sequential_1/lstm_2/while/lstm_cell_4/ReluА
+sequential_1/lstm_2/while/lstm_cell_4/mul_1Mul1sequential_1/lstm_2/while/lstm_cell_4/Sigmoid:y:08sequential_1/lstm_2/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2-
+sequential_1/lstm_2/while/lstm_cell_4/mul_1ї
+sequential_1/lstm_2/while/lstm_cell_4/add_1AddV2-sequential_1/lstm_2/while/lstm_cell_4/mul:z:0/sequential_1/lstm_2/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2-
+sequential_1/lstm_2/while/lstm_cell_4/add_1╒
/sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_2Sigmoid4sequential_1/lstm_2/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @21
/sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_2╟
,sequential_1/lstm_2/while/lstm_cell_4/Relu_1Relu/sequential_1/lstm_2/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2.
,sequential_1/lstm_2/while/lstm_cell_4/Relu_1Д
+sequential_1/lstm_2/while/lstm_cell_4/mul_2Mul3sequential_1/lstm_2/while/lstm_cell_4/Sigmoid_2:y:0:sequential_1/lstm_2/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2-
+sequential_1/lstm_2/while/lstm_cell_4/mul_2├
>sequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_2_while_placeholder_1%sequential_1_lstm_2_while_placeholder/sequential_1/lstm_2/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItemД
sequential_1/lstm_2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_2/while/add/y╣
sequential_1/lstm_2/while/addAddV2%sequential_1_lstm_2_while_placeholder(sequential_1/lstm_2/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_2/while/addИ
!sequential_1/lstm_2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_2/while/add_1/y┌
sequential_1/lstm_2/while/add_1AddV2@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counter*sequential_1/lstm_2/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_2/while/add_1╗
"sequential_1/lstm_2/while/IdentityIdentity#sequential_1/lstm_2/while/add_1:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_1/lstm_2/while/Identityт
$sequential_1/lstm_2/while/Identity_1IdentityFsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_2/while/Identity_1╜
$sequential_1/lstm_2/while/Identity_2Identity!sequential_1/lstm_2/while/add:z:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_2/while/Identity_2ъ
$sequential_1/lstm_2/while/Identity_3IdentityNsequential_1/lstm_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_2/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_2/while/Identity_3▄
$sequential_1/lstm_2/while/Identity_4Identity/sequential_1/lstm_2/while/lstm_cell_4/mul_2:z:0^sequential_1/lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2&
$sequential_1/lstm_2/while/Identity_4▄
$sequential_1/lstm_2/while/Identity_5Identity/sequential_1/lstm_2/while/lstm_cell_4/add_1:z:0^sequential_1/lstm_2/while/NoOp*
T0*'
_output_shapes
:         @2&
$sequential_1/lstm_2/while/Identity_5┐
sequential_1/lstm_2/while/NoOpNoOp=^sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp<^sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp>^sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_1/lstm_2/while/NoOp"Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0"U
$sequential_1_lstm_2_while_identity_1-sequential_1/lstm_2/while/Identity_1:output:0"U
$sequential_1_lstm_2_while_identity_2-sequential_1/lstm_2/while/Identity_2:output:0"U
$sequential_1_lstm_2_while_identity_3-sequential_1/lstm_2/while/Identity_3:output:0"U
$sequential_1_lstm_2_while_identity_4-sequential_1/lstm_2/while/Identity_4:output:0"U
$sequential_1_lstm_2_while_identity_5-sequential_1/lstm_2/while/Identity_5:output:0"Р
Esequential_1_lstm_2_while_lstm_cell_4_biasadd_readvariableop_resourceGsequential_1_lstm_2_while_lstm_cell_4_biasadd_readvariableop_resource_0"Т
Fsequential_1_lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resourceHsequential_1_lstm_2_while_lstm_cell_4_matmul_1_readvariableop_resource_0"О
Dsequential_1_lstm_2_while_lstm_cell_4_matmul_readvariableop_resourceFsequential_1_lstm_2_while_lstm_cell_4_matmul_readvariableop_resource_0"А
=sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1?sequential_1_lstm_2_while_sequential_1_lstm_2_strided_slice_1_0"°
ysequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_2_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_2_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2|
<sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp<sequential_1/lstm_2/while/lstm_cell_4/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp;sequential_1/lstm_2/while/lstm_cell_4/MatMul/ReadVariableOp2~
=sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp=sequential_1/lstm_2/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╦[
Х
A__inference_lstm_3_layer_call_and_return_conditional_losses_52084
inputs_0=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileF
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_52000*
condR
while_cond_51999*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                   *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                   2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  @
"
_user_specified_name
inputs/0
╒
╡
&__inference_lstm_2_layer_call_fn_51112
inputs_0
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_486302
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
Х[
У
A__inference_lstm_3_layer_call_and_return_conditional_losses_52386

inputs=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
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
:         @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_52302*
condR
while_cond_52301*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╨
╛
while_cond_51999
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51999___redundant_placeholder03
/while_while_cond_51999___redundant_placeholder13
/while_while_cond_51999___redundant_placeholder23
/while_while_cond_51999___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
к
╬
$sequential_1_lstm_3_while_cond_48170D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3F
Bsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_48170___redundant_placeholder0[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_48170___redundant_placeholder1[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_48170___redundant_placeholder2[
Wsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_48170___redundant_placeholder3&
"sequential_1_lstm_3_while_identity
╘
sequential_1/lstm_3/while/LessLess%sequential_1_lstm_3_while_placeholderBsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/while/LessЩ
"sequential_1/lstm_3/while/IdentityIdentity"sequential_1/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identity"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
к
│
&__inference_lstm_2_layer_call_fn_51123

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_496792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
└H
ж

lstm_3_while_body_50992*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0:	@АN
;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АI
:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
7lstm_3_while_lstm_cell_5_matmul_readvariableop_resource:	@АL
9lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource:	 АG
8lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource:	АИв/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpв.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpв0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp╤
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape¤
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItem█
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpЁ
lstm_3/while/lstm_cell_5/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
lstm_3/while/lstm_cell_5/MatMulс
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp┘
!lstm_3/while/lstm_cell_5/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_3/while/lstm_cell_5/MatMul_1╨
lstm_3/while/lstm_cell_5/addAddV2)lstm_3/while/lstm_cell_5/MatMul:product:0+lstm_3/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_3/while/lstm_cell_5/add┌
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp▌
 lstm_3/while/lstm_cell_5/BiasAddBiasAdd lstm_3/while/lstm_cell_5/add:z:07lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2"
 lstm_3/while/lstm_cell_5/BiasAddЦ
(lstm_3/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_5/split/split_dimг
lstm_3/while/lstm_cell_5/splitSplit1lstm_3/while/lstm_cell_5/split/split_dim:output:0)lstm_3/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2 
lstm_3/while/lstm_cell_5/splitк
 lstm_3/while/lstm_cell_5/SigmoidSigmoid'lstm_3/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2"
 lstm_3/while/lstm_cell_5/Sigmoidо
"lstm_3/while/lstm_cell_5/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2$
"lstm_3/while/lstm_cell_5/Sigmoid_1╣
lstm_3/while/lstm_cell_5/mulMul&lstm_3/while/lstm_cell_5/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:          2
lstm_3/while/lstm_cell_5/mulб
lstm_3/while/lstm_cell_5/ReluRelu'lstm_3/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_3/while/lstm_cell_5/Relu╠
lstm_3/while/lstm_cell_5/mul_1Mul$lstm_3/while/lstm_cell_5/Sigmoid:y:0+lstm_3/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/mul_1┴
lstm_3/while/lstm_cell_5/add_1AddV2 lstm_3/while/lstm_cell_5/mul:z:0"lstm_3/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/add_1о
"lstm_3/while/lstm_cell_5/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2$
"lstm_3/while/lstm_cell_5/Sigmoid_2а
lstm_3/while/lstm_cell_5/Relu_1Relu"lstm_3/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2!
lstm_3/while/lstm_cell_5/Relu_1╨
lstm_3/while/lstm_cell_5/mul_2Mul&lstm_3/while/lstm_cell_5/Sigmoid_2:y:0-lstm_3/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2 
lstm_3/while/lstm_cell_5/mul_2В
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/yЕ
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/yЩ
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1З
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identityб
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1Й
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2╢
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3и
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_5/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:          2
lstm_3/while/Identity_4и
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_5/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:          2
lstm_3/while/Identity_5■
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_5_matmul_readvariableop_resource9lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0"─
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2b
/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
ц
Ё
G__inference_sequential_1_layer_call_and_return_conditional_losses_50402
lstm_2_input
lstm_2_50381:	А
lstm_2_50383:	@А
lstm_2_50385:	А
lstm_3_50388:	@А
lstm_3_50390:	 А
lstm_3_50392:	А
dense_1_50396: 
dense_1_50398:
identityИвdense_1/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallвlstm_2/StatefulPartitionedCallвlstm_3/StatefulPartitionedCallб
lstm_2/StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputlstm_2_50381lstm_2_50383lstm_2_50385*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_502582 
lstm_2/StatefulPartitionedCall╕
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_50388lstm_3_50390lstm_3_50392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_500852 
lstm_3/StatefulPartitionedCallП
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_499182#
!dropout_1/StatefulPartitionedCall░
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_50396dense_1_50398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_498622!
dense_1/StatefulPartitionedCallГ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╓
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
▓
Ї
+__inference_lstm_cell_4_layer_call_fn_52466

inputs
states_0
states_1
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_484832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         @2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         @2

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
?:         :         @:         @: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
Р[
У
A__inference_lstm_2_layer_call_and_return_conditional_losses_50258

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
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
:         2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_50174*
condR
while_cond_50173*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
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
:         @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
о%
╒
while_body_48561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_4_48585_0:	А,
while_lstm_cell_4_48587_0:	@А(
while_lstm_cell_4_48589_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_4_48585:	А*
while_lstm_cell_4_48587:	@А&
while_lstm_cell_4_48589:	АИв)while/lstm_cell_4/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╫
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_48585_0while_lstm_cell_4_48587_0while_lstm_cell_4_48589_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_484832+
)while/lstm_cell_4/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_48585while_lstm_cell_4_48585_0"4
while_lstm_cell_4_48587while_lstm_cell_4_48587_0"4
while_lstm_cell_4_48589while_lstm_cell_4_48589_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Р[
У
A__inference_lstm_2_layer_call_and_return_conditional_losses_51738

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
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
:         2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_51654*
condR
while_cond_51653*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
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
:         @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Д[
╞
$sequential_1_lstm_3_while_body_48171D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3C
?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0
{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_1_lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0:	@А[
Hsequential_1_lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АV
Gsequential_1_lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0:	А&
"sequential_1_lstm_3_while_identity(
$sequential_1_lstm_3_while_identity_1(
$sequential_1_lstm_3_while_identity_2(
$sequential_1_lstm_3_while_identity_3(
$sequential_1_lstm_3_while_identity_4(
$sequential_1_lstm_3_while_identity_5A
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1}
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensorW
Dsequential_1_lstm_3_while_lstm_cell_5_matmul_readvariableop_resource:	@АY
Fsequential_1_lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource:	 АT
Esequential_1_lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource:	АИв<sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpв;sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpв=sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOpы
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2M
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape╦
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_3_while_placeholderTsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02?
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemВ
;sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02=
;sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOpд
,sequential_1/lstm_3/while/lstm_cell_5/MatMulMatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2.
,sequential_1/lstm_3/while/lstm_cell_5/MatMulИ
=sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02?
=sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOpН
.sequential_1/lstm_3/while/lstm_cell_5/MatMul_1MatMul'sequential_1_lstm_3_while_placeholder_2Esequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А20
.sequential_1/lstm_3/while/lstm_cell_5/MatMul_1Д
)sequential_1/lstm_3/while/lstm_cell_5/addAddV26sequential_1/lstm_3/while/lstm_cell_5/MatMul:product:08sequential_1/lstm_3/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2+
)sequential_1/lstm_3/while/lstm_cell_5/addБ
<sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOpС
-sequential_1/lstm_3/while/lstm_cell_5/BiasAddBiasAdd-sequential_1/lstm_3/while/lstm_cell_5/add:z:0Dsequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-sequential_1/lstm_3/while/lstm_cell_5/BiasAdd░
5sequential_1/lstm_3/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_3/while/lstm_cell_5/split/split_dim╫
+sequential_1/lstm_3/while/lstm_cell_5/splitSplit>sequential_1/lstm_3/while/lstm_cell_5/split/split_dim:output:06sequential_1/lstm_3/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2-
+sequential_1/lstm_3/while/lstm_cell_5/split╤
-sequential_1/lstm_3/while/lstm_cell_5/SigmoidSigmoid4sequential_1/lstm_3/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2/
-sequential_1/lstm_3/while/lstm_cell_5/Sigmoid╒
/sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_1Sigmoid4sequential_1/lstm_3/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          21
/sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_1э
)sequential_1/lstm_3/while/lstm_cell_5/mulMul3sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_1:y:0'sequential_1_lstm_3_while_placeholder_3*
T0*'
_output_shapes
:          2+
)sequential_1/lstm_3/while/lstm_cell_5/mul╚
*sequential_1/lstm_3/while/lstm_cell_5/ReluRelu4sequential_1/lstm_3/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2,
*sequential_1/lstm_3/while/lstm_cell_5/ReluА
+sequential_1/lstm_3/while/lstm_cell_5/mul_1Mul1sequential_1/lstm_3/while/lstm_cell_5/Sigmoid:y:08sequential_1/lstm_3/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2-
+sequential_1/lstm_3/while/lstm_cell_5/mul_1ї
+sequential_1/lstm_3/while/lstm_cell_5/add_1AddV2-sequential_1/lstm_3/while/lstm_cell_5/mul:z:0/sequential_1/lstm_3/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2-
+sequential_1/lstm_3/while/lstm_cell_5/add_1╒
/sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_2Sigmoid4sequential_1/lstm_3/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          21
/sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_2╟
,sequential_1/lstm_3/while/lstm_cell_5/Relu_1Relu/sequential_1/lstm_3/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2.
,sequential_1/lstm_3/while/lstm_cell_5/Relu_1Д
+sequential_1/lstm_3/while/lstm_cell_5/mul_2Mul3sequential_1/lstm_3/while/lstm_cell_5/Sigmoid_2:y:0:sequential_1/lstm_3/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2-
+sequential_1/lstm_3/while/lstm_cell_5/mul_2├
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_3_while_placeholder_1%sequential_1_lstm_3_while_placeholder/sequential_1/lstm_3/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemД
sequential_1/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_3/while/add/y╣
sequential_1/lstm_3/while/addAddV2%sequential_1_lstm_3_while_placeholder(sequential_1/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/while/addИ
!sequential_1/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_3/while/add_1/y┌
sequential_1/lstm_3/while/add_1AddV2@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counter*sequential_1/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/while/add_1╗
"sequential_1/lstm_3/while/IdentityIdentity#sequential_1/lstm_3/while/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identityт
$sequential_1/lstm_3/while/Identity_1IdentityFsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_1╜
$sequential_1/lstm_3/while/Identity_2Identity!sequential_1/lstm_3/while/add:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_2ъ
$sequential_1/lstm_3/while/Identity_3IdentityNsequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_3▄
$sequential_1/lstm_3/while/Identity_4Identity/sequential_1/lstm_3/while/lstm_cell_5/mul_2:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:          2&
$sequential_1/lstm_3/while/Identity_4▄
$sequential_1/lstm_3/while/Identity_5Identity/sequential_1/lstm_3/while/lstm_cell_5/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:          2&
$sequential_1/lstm_3/while/Identity_5┐
sequential_1/lstm_3/while/NoOpNoOp=^sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp<^sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp>^sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_1/lstm_3/while/NoOp"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0"U
$sequential_1_lstm_3_while_identity_1-sequential_1/lstm_3/while/Identity_1:output:0"U
$sequential_1_lstm_3_while_identity_2-sequential_1/lstm_3/while/Identity_2:output:0"U
$sequential_1_lstm_3_while_identity_3-sequential_1/lstm_3/while/Identity_3:output:0"U
$sequential_1_lstm_3_while_identity_4-sequential_1/lstm_3/while/Identity_4:output:0"U
$sequential_1_lstm_3_while_identity_5-sequential_1/lstm_3/while/Identity_5:output:0"Р
Esequential_1_lstm_3_while_lstm_cell_5_biasadd_readvariableop_resourceGsequential_1_lstm_3_while_lstm_cell_5_biasadd_readvariableop_resource_0"Т
Fsequential_1_lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resourceHsequential_1_lstm_3_while_lstm_cell_5_matmul_1_readvariableop_resource_0"О
Dsequential_1_lstm_3_while_lstm_cell_5_matmul_readvariableop_resourceFsequential_1_lstm_3_while_lstm_cell_5_matmul_readvariableop_resource_0"А
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0"°
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2|
<sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp<sequential_1/lstm_3/while/lstm_cell_5/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp;sequential_1/lstm_3/while/lstm_cell_5/MatMul/ReadVariableOp2~
=sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp=sequential_1/lstm_3/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
╘
ъ
G__inference_sequential_1_layer_call_and_return_conditional_losses_50314

inputs
lstm_2_50293:	А
lstm_2_50295:	@А
lstm_2_50297:	А
lstm_3_50300:	@А
lstm_3_50302:	 А
lstm_3_50304:	А
dense_1_50308: 
dense_1_50310:
identityИвdense_1/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallвlstm_2/StatefulPartitionedCallвlstm_3/StatefulPartitionedCallЫ
lstm_2/StatefulPartitionedCallStatefulPartitionedCallinputslstm_2_50293lstm_2_50295lstm_2_50297*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_502582 
lstm_2/StatefulPartitionedCall╕
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_50300lstm_3_50302lstm_3_50304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_500852 
lstm_3/StatefulPartitionedCallП
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_499182#
!dropout_1/StatefulPartitionedCall░
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_50308dense_1_50310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_498622!
dense_1/StatefulPartitionedCallГ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╓
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
▓
Ї
+__inference_lstm_cell_5_layer_call_fn_52547

inputs
states_0
states_1
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identity

identity_1

identity_2ИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_489672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:          2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:          2

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
?:         @:          :          : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1
ё
В
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_48337

inputs

states
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
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
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         @
 
_user_specified_namestates:OK
'
_output_shapes
:         @
 
_user_specified_namestates
Ч

╦
,__inference_sequential_1_layer_call_fn_49888
lstm_2_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:	 А
	unknown_4:	А
	unknown_5: 
	unknown_6:
identityИвStatefulPartitionedCall╦
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_498692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
ч	
┬
#__inference_signature_wrapper_50431
lstm_2_input
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:	 А
	unknown_4:	А
	unknown_5: 
	unknown_6:
identityИвStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_482622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
Х[
У
A__inference_lstm_3_layer_call_and_return_conditional_losses_49837

inputs=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
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
:         @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_49753*
condR
while_cond_49752*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╢
╠
G__inference_sequential_1_layer_call_and_return_conditional_losses_50378
lstm_2_input
lstm_2_50357:	А
lstm_2_50359:	@А
lstm_2_50361:	А
lstm_3_50364:	@А
lstm_3_50366:	 А
lstm_3_50368:	А
dense_1_50372: 
dense_1_50374:
identityИвdense_1/StatefulPartitionedCallвlstm_2/StatefulPartitionedCallвlstm_3/StatefulPartitionedCallб
lstm_2/StatefulPartitionedCallStatefulPartitionedCalllstm_2_inputlstm_2_50357lstm_2_50359lstm_2_50361*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_496792 
lstm_2/StatefulPartitionedCall╕
lstm_3/StatefulPartitionedCallStatefulPartitionedCall'lstm_2/StatefulPartitionedCall:output:0lstm_3_50364lstm_3_50366lstm_3_50368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_498372 
lstm_3/StatefulPartitionedCallў
dropout_1/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_498502
dropout_1/PartitionedCallи
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_50372dense_1_50374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_498622!
dense_1/StatefulPartitionedCallГ
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity▓
NoOpNoOp ^dense_1/StatefulPartitionedCall^lstm_2/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
lstm_2/StatefulPartitionedCalllstm_2/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_namelstm_2_input
Ю
b
)__inference_dropout_1_layer_call_fn_52396

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_499182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
о%
╒
while_body_48351
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_4_48375_0:	А,
while_lstm_cell_4_48377_0:	@А(
while_lstm_cell_4_48379_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_4_48375:	А*
while_lstm_cell_4_48377:	@А&
while_lstm_cell_4_48379:	АИв)while/lstm_cell_4/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╫
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_48375_0while_lstm_cell_4_48377_0while_lstm_cell_4_48379_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_483372+
)while/lstm_cell_4/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_4_48375while_lstm_cell_4_48375_0"4
while_lstm_cell_4_48377while_lstm_cell_4_48377_0"4
while_lstm_cell_4_48379while_lstm_cell_4_48379_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╨
╛
while_cond_50000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_50000___redundant_placeholder03
/while_while_cond_50000___redundant_placeholder13
/while_while_cond_50000___redundant_placeholder23
/while_while_cond_50000___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
о%
╒
while_body_48981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_5_49005_0:	@А,
while_lstm_cell_5_49007_0:	 А(
while_lstm_cell_5_49009_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_5_49005:	@А*
while_lstm_cell_5_49007:	 А&
while_lstm_cell_5_49009:	АИв)while/lstm_cell_5/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╫
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_49005_0while_lstm_cell_5_49007_0while_lstm_cell_5_49009_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_489672+
)while/lstm_cell_5/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5_49005while_lstm_cell_5_49005_0"4
while_lstm_cell_5_49007while_lstm_cell_5_49007_0"4
while_lstm_cell_5_49009while_lstm_cell_5_49009_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
Х[
У
A__inference_lstm_3_layer_call_and_return_conditional_losses_52235

inputs=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
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
:         @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_52151*
condR
while_cond_52150*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╨
╛
while_cond_49594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_49594___redundant_placeholder03
/while_while_cond_49594___redundant_placeholder13
/while_while_cond_49594___redundant_placeholder23
/while_while_cond_49594___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
к
╬
$sequential_1_lstm_2_while_cond_48023D
@sequential_1_lstm_2_while_sequential_1_lstm_2_while_loop_counterJ
Fsequential_1_lstm_2_while_sequential_1_lstm_2_while_maximum_iterations)
%sequential_1_lstm_2_while_placeholder+
'sequential_1_lstm_2_while_placeholder_1+
'sequential_1_lstm_2_while_placeholder_2+
'sequential_1_lstm_2_while_placeholder_3F
Bsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_48023___redundant_placeholder0[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_48023___redundant_placeholder1[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_48023___redundant_placeholder2[
Wsequential_1_lstm_2_while_sequential_1_lstm_2_while_cond_48023___redundant_placeholder3&
"sequential_1_lstm_2_while_identity
╘
sequential_1/lstm_2/while/LessLess%sequential_1_lstm_2_while_placeholderBsequential_1_lstm_2_while_less_sequential_1_lstm_2_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_2/while/LessЩ
"sequential_1/lstm_2/while/IdentityIdentity"sequential_1/lstm_2/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_2/while/Identity"Q
"sequential_1_lstm_2_while_identity+sequential_1/lstm_2/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
ЦI
▀
__inference__traced_save_52744
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_2_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_2_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_2_lstm_cell_4_bias_read_readvariableop8
4savev2_lstm_3_lstm_cell_5_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_5_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_4_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_5_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_5_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop?
;savev2_adam_lstm_2_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_2_lstm_cell_4_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_5_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_5_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameА
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*Т
valueИBЕ B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╚
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╤
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_2_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_2_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_2_lstm_cell_4_bias_read_readvariableop4savev2_lstm_3_lstm_cell_5_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_5_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop;savev2_adam_lstm_2_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_2_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_2_lstm_cell_4_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_5_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_5_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*Е
_input_shapesє
Ё: : :: : : : : :	А:	@А:А:	@А:	 А:А: : : ::	А:	@А:А:	@А:	 А:А: ::	А:	@А:А:	@А:	 А:А: 2(
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
:	А:%	!

_output_shapes
:	@А:!


_output_shapes	
:А:%!

_output_shapes
:	@А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:
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
:	А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	@А:%!

_output_shapes
:	 А:!

_output_shapes	
:А: 

_output_shapes
: 
┴>
╞
while_body_50174
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
ё
В
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_49113

inputs

states
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:          2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2Щ
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
?:         @:          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
Е

┼
,__inference_sequential_1_layer_call_fn_50452

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:	 А
	unknown_4:	А
	unknown_5: 
	unknown_6:
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_498692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Х[
У
A__inference_lstm_3_layer_call_and_return_conditional_losses_50085

inputs=
*lstm_cell_5_matmul_readvariableop_resource:	@А?
,lstm_cell_5_matmul_1_readvariableop_resource:	 А:
+lstm_cell_5_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_5/BiasAdd/ReadVariableOpв!lstm_cell_5/MatMul/ReadVariableOpв#lstm_cell_5/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:          2
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
B :ш2
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
zeros_1/packed/1Й
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
:          2	
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
:         @2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOpк
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul╕
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOpж
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/MatMul_1Ь
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/add▒
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOpй
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_5/BiasAdd|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dimя
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_cell_5/splitГ
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/SigmoidЗ
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_1И
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_cell_5/ReluШ
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_1Н
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/add_1З
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_cell_5/Relu_1Ь
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_cell_5/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_50001*
condR
while_cond_50000*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
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
:          2

Identity┼
NoOpNoOp#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         @: : : 2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
┴>
╞
while_body_52302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_5_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_5_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_5_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_5_matmul_readvariableop_resource:	@АE
2while_lstm_cell_5_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_5_biasadd_readvariableop_resource:	АИв(while/lstm_cell_5/BiasAdd/ReadVariableOpв'while/lstm_cell_5/MatMul/ReadVariableOpв)while/lstm_cell_5/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp╘
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul╠
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp╜
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/MatMul_1┤
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/add┼
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp┴
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_5/BiasAddИ
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dimЗ
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
while/lstm_cell_5/splitХ
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/SigmoidЩ
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_1Э
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mulМ
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu░
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_1е
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/add_1Щ
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Sigmoid_2Л
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/Relu_1┤
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
while/lstm_cell_5/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
Р[
У
A__inference_lstm_2_layer_call_and_return_conditional_losses_51587

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
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
:         2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_51503*
condR
while_cond_51502*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
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
:         @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ё
В
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_48967

inputs

states
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:          2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2Щ
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
?:         @:          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:OK
'
_output_shapes
:          
 
_user_specified_namestates:OK
'
_output_shapes
:          
 
_user_specified_namestates
э
Ф
'__inference_dense_1_layer_call_fn_52422

inputs
unknown: 
	unknown_0:
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_498622
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
∙
Д
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52498

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
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
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
к
│
&__inference_lstm_2_layer_call_fn_51134

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_2_layer_call_and_return_conditional_losses_502582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╨
╛
while_cond_51848
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_51848___redundant_placeholder03
/while_while_cond_51848___redundant_placeholder13
/while_while_cond_51848___redundant_placeholder23
/while_while_cond_51848___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╨
╛
while_cond_50173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_50173___redundant_placeholder03
/while_while_cond_50173___redundant_placeholder13
/while_while_cond_50173___redundant_placeholder23
/while_while_cond_50173___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
Е

┼
,__inference_sequential_1_layer_call_fn_50473

inputs
unknown:	А
	unknown_0:	@А
	unknown_1:	А
	unknown_2:	@А
	unknown_3:	 А
	unknown_4:	А
	unknown_5: 
	unknown_6:
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_503142
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┴>
╞
while_body_51654
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
∙
Д
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52530

inputs
states_0
states_11
matmul_readvariableop_resource:	А3
 matmul_1_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         @2

Identity_2Щ
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
?:         :         @:         @: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         @
"
_user_specified_name
states/0:QM
'
_output_shapes
:         @
"
_user_specified_name
states/1
Р[
У
A__inference_lstm_2_layer_call_and_return_conditional_losses_49679

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	А?
,lstm_cell_4_matmul_1_readvariableop_resource:	@А:
+lstm_cell_4_biasadd_readvariableop_resource:	А
identityИв"lstm_cell_4/BiasAdd/ReadVariableOpв!lstm_cell_4/MatMul/ReadVariableOpв#lstm_cell_4/MatMul_1/ReadVariableOpвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
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
:         2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2▓
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02#
!lstm_cell_4/MatMul/ReadVariableOpк
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul╕
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOpж
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/MatMul_1Ь
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/add▒
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOpй
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_4/BiasAdd|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimя
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_cell_4/splitГ
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/SigmoidЗ
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_1И
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_cell_4/ReluШ
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_1Н
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/add_1З
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_cell_4/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterЖ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_49595*
condR
while_cond_49594*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
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
:         @2

Identity┼
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┴>
╞
while_body_51503
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	АG
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	@АB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	АE
2while_lstm_cell_4_matmul_1_readvariableop_resource:	@А@
1while_lstm_cell_4_biasadd_readvariableop_resource:	АИв(while/lstm_cell_4/BiasAdd/ReadVariableOpв'while/lstm_cell_4/MatMul/ReadVariableOpв)while/lstm_cell_4/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╞
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	А*
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp╘
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul╠
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp╜
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/MatMul_1┤
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/add┼
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp┴
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_4/BiasAddИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dimЗ
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
while/lstm_cell_4/splitХ
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/SigmoidЩ
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_1Э
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mulМ
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu░
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_1е
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/add_1Щ
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/Relu_1┤
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
while/lstm_cell_4/mul_2▀
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3М
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         @2
while/Identity_5█

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         @:         @: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
╨
╛
while_cond_48980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_48980___redundant_placeholder03
/while_while_cond_48980___redundant_placeholder13
/while_while_cond_48980___redundant_placeholder23
/while_while_cond_48980___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
ё
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_49850

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:          2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
о%
╒
while_body_49191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_5_49215_0:	@А,
while_lstm_cell_5_49217_0:	 А(
while_lstm_cell_5_49219_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_5_49215:	@А*
while_lstm_cell_5_49217:	 А&
while_lstm_cell_5_49219:	АИв)while/lstm_cell_5/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         @*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╫
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_49215_0while_lstm_cell_5_49217_0while_lstm_cell_5_49219_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:          :          :          *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_491132+
)while/lstm_cell_5/StatefulPartitionedCallЎ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3г
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_4г
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:          2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"4
while_lstm_cell_5_49215while_lstm_cell_5_49215_0"4
while_lstm_cell_5_49217while_lstm_cell_5_49217_0"4
while_lstm_cell_5_49219while_lstm_cell_5_49219_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :          :          : : : : : 2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
: 
БF
№
A__inference_lstm_2_layer_call_and_return_conditional_losses_48420

inputs$
lstm_cell_4_48338:	А$
lstm_cell_4_48340:	@А 
lstm_cell_4_48342:	А
identityИв#lstm_cell_4/StatefulPartitionedCallвwhileD
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
strided_slice/stack_2т
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
B :ш2
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
zeros/packed/1Г
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
:         @2
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
B :ш2
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
zeros_1/packed/1Й
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
:         @2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  2
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
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
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
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2У
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_48338lstm_cell_4_48340lstm_cell_4_48342*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         @:         @:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_483372%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
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
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╕
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_48338lstm_cell_4_48340lstm_cell_4_48342*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_48351*
condR
while_cond_48350*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
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
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @2
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
 :                  @2

Identity|
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╨
╛
while_cond_52301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_52301___redundant_placeholder03
/while_while_cond_52301___redundant_placeholder13
/while_while_cond_52301___redundant_placeholder23
/while_while_cond_52301___redundant_placeholder3
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
@: : : : :          :          : ::::: 
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
:          :-)
'
_output_shapes
:          :

_output_shapes
: :

_output_shapes
:
╞╓
р
G__inference_sequential_1_layer_call_and_return_conditional_losses_50778

inputsD
1lstm_2_lstm_cell_4_matmul_readvariableop_resource:	АF
3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource:	@АA
2lstm_2_lstm_cell_4_biasadd_readvariableop_resource:	АD
1lstm_3_lstm_cell_5_matmul_readvariableop_resource:	@АF
3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource:	 АA
2lstm_3_lstm_cell_5_biasadd_readvariableop_resource:	А8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identityИвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpв)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpв(lstm_2/lstm_cell_4/MatMul/ReadVariableOpв*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpвlstm_2/whileв)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpв(lstm_3/lstm_cell_5/MatMul/ReadVariableOpв*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpвlstm_3/whileR
lstm_2/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_2/ShapeВ
lstm_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice/stackЖ
lstm_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_1Ж
lstm_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_2/strided_slice/stack_2М
lstm_2/strided_sliceStridedSlicelstm_2/Shape:output:0#lstm_2/strided_slice/stack:output:0%lstm_2/strided_slice/stack_1:output:0%lstm_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slicej
lstm_2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/mul/yИ
lstm_2/zeros/mulMullstm_2/strided_slice:output:0lstm_2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/mulm
lstm_2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_2/zeros/Less/yГ
lstm_2/zeros/LessLesslstm_2/zeros/mul:z:0lstm_2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros/Lessp
lstm_2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros/packed/1Я
lstm_2/zeros/packedPacklstm_2/strided_slice:output:0lstm_2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros/packedm
lstm_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros/ConstС
lstm_2/zerosFilllstm_2/zeros/packed:output:0lstm_2/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_2/zerosn
lstm_2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/mul/yО
lstm_2/zeros_1/mulMullstm_2/strided_slice:output:0lstm_2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/mulq
lstm_2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_2/zeros_1/Less/yЛ
lstm_2/zeros_1/LessLesslstm_2/zeros_1/mul:z:0lstm_2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_2/zeros_1/Lesst
lstm_2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
lstm_2/zeros_1/packed/1е
lstm_2/zeros_1/packedPacklstm_2/strided_slice:output:0 lstm_2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_2/zeros_1/packedq
lstm_2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/zeros_1/ConstЩ
lstm_2/zeros_1Filllstm_2/zeros_1/packed:output:0lstm_2/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @2
lstm_2/zeros_1Г
lstm_2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose/permП
lstm_2/transpose	Transposeinputslstm_2/transpose/perm:output:0*
T0*+
_output_shapes
:         2
lstm_2/transposed
lstm_2/Shape_1Shapelstm_2/transpose:y:0*
T0*
_output_shapes
:2
lstm_2/Shape_1Ж
lstm_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_1/stackК
lstm_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_1К
lstm_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_1/stack_2Ш
lstm_2/strided_slice_1StridedSlicelstm_2/Shape_1:output:0%lstm_2/strided_slice_1/stack:output:0'lstm_2/strided_slice_1/stack_1:output:0'lstm_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_2/strided_slice_1У
"lstm_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_2/TensorArrayV2/element_shape╬
lstm_2/TensorArrayV2TensorListReserve+lstm_2/TensorArrayV2/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2═
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2>
<lstm_2/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_2/transpose:y:0Elstm_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_2/TensorArrayUnstack/TensorListFromTensorЖ
lstm_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_2/strided_slice_2/stackК
lstm_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_1К
lstm_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_2/stack_2ж
lstm_2/strided_slice_2StridedSlicelstm_2/transpose:y:0%lstm_2/strided_slice_2/stack:output:0'lstm_2/strided_slice_2/stack_1:output:0'lstm_2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
lstm_2/strided_slice_2╟
(lstm_2/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_2_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype02*
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp╞
lstm_2/lstm_cell_4/MatMulMatMullstm_2/strided_slice_2:output:00lstm_2/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/MatMul═
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	@А*
dtype02,
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp┬
lstm_2/lstm_cell_4/MatMul_1MatMullstm_2/zeros:output:02lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/MatMul_1╕
lstm_2/lstm_cell_4/addAddV2#lstm_2/lstm_cell_4/MatMul:product:0%lstm_2/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/add╞
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp┼
lstm_2/lstm_cell_4/BiasAddBiasAddlstm_2/lstm_cell_4/add:z:01lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_2/lstm_cell_4/BiasAddК
"lstm_2/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_2/lstm_cell_4/split/split_dimЛ
lstm_2/lstm_cell_4/splitSplit+lstm_2/lstm_cell_4/split/split_dim:output:0#lstm_2/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:         @:         @:         @:         @*
	num_split2
lstm_2/lstm_cell_4/splitШ
lstm_2/lstm_cell_4/SigmoidSigmoid!lstm_2/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/SigmoidЬ
lstm_2/lstm_cell_4/Sigmoid_1Sigmoid!lstm_2/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Sigmoid_1д
lstm_2/lstm_cell_4/mulMul lstm_2/lstm_cell_4/Sigmoid_1:y:0lstm_2/zeros_1:output:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mulП
lstm_2/lstm_cell_4/ReluRelu!lstm_2/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Relu┤
lstm_2/lstm_cell_4/mul_1Mullstm_2/lstm_cell_4/Sigmoid:y:0%lstm_2/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mul_1й
lstm_2/lstm_cell_4/add_1AddV2lstm_2/lstm_cell_4/mul:z:0lstm_2/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/add_1Ь
lstm_2/lstm_cell_4/Sigmoid_2Sigmoid!lstm_2/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Sigmoid_2О
lstm_2/lstm_cell_4/Relu_1Relulstm_2/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/Relu_1╕
lstm_2/lstm_cell_4/mul_2Mul lstm_2/lstm_cell_4/Sigmoid_2:y:0'lstm_2/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:         @2
lstm_2/lstm_cell_4/mul_2Э
$lstm_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2&
$lstm_2/TensorArrayV2_1/element_shape╘
lstm_2/TensorArrayV2_1TensorListReserve-lstm_2/TensorArrayV2_1/element_shape:output:0lstm_2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_2/TensorArrayV2_1\
lstm_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/timeН
lstm_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_2/while/maximum_iterationsx
lstm_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_2/while/loop_counterя
lstm_2/whileWhile"lstm_2/while/loop_counter:output:0(lstm_2/while/maximum_iterations:output:0lstm_2/time:output:0lstm_2/TensorArrayV2_1:handle:0lstm_2/zeros:output:0lstm_2/zeros_1:output:0lstm_2/strided_slice_1:output:0>lstm_2/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_2_lstm_cell_4_matmul_readvariableop_resource3lstm_2_lstm_cell_4_matmul_1_readvariableop_resource2lstm_2_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         @:         @: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_2_while_body_50540*#
condR
lstm_2_while_cond_50539*K
output_shapes:
8: : : : :         @:         @: : : : : *
parallel_iterations 2
lstm_2/while├
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   29
7lstm_2/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_2/TensorArrayV2Stack/TensorListStackTensorListStacklstm_2/while:output:3@lstm_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         @*
element_dtype02+
)lstm_2/TensorArrayV2Stack/TensorListStackП
lstm_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_2/strided_slice_3/stackК
lstm_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_2/strided_slice_3/stack_1К
lstm_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_2/strided_slice_3/stack_2─
lstm_2/strided_slice_3StridedSlice2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_2/strided_slice_3/stack:output:0'lstm_2/strided_slice_3/stack_1:output:0'lstm_2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_2/strided_slice_3З
lstm_2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_2/transpose_1/perm┴
lstm_2/transpose_1	Transpose2lstm_2/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_2/transpose_1/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_2/transpose_1t
lstm_2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_2/runtimeb
lstm_3/ShapeShapelstm_2/transpose_1:y:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:          2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/packed/1е
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:          2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/permЯ
lstm_3/transpose	Transposelstm_2/transpose_1:y:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:         @2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2$
"lstm_3/TensorArrayV2/element_shape╬
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2═
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2ж
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
lstm_3/strided_slice_2╟
(lstm_3/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp╞
lstm_3/lstm_cell_5/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/MatMul═
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp┬
lstm_3/lstm_cell_5/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/MatMul_1╕
lstm_3/lstm_cell_5/addAddV2#lstm_3/lstm_cell_5/MatMul:product:0%lstm_3/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/add╞
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp┼
lstm_3/lstm_cell_5/BiasAddBiasAddlstm_3/lstm_cell_5/add:z:01lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_3/lstm_cell_5/BiasAddК
"lstm_3/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_5/split/split_dimЛ
lstm_3/lstm_cell_5/splitSplit+lstm_3/lstm_cell_5/split/split_dim:output:0#lstm_3/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
lstm_3/lstm_cell_5/splitШ
lstm_3/lstm_cell_5/SigmoidSigmoid!lstm_3/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/SigmoidЬ
lstm_3/lstm_cell_5/Sigmoid_1Sigmoid!lstm_3/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Sigmoid_1д
lstm_3/lstm_cell_5/mulMul lstm_3/lstm_cell_5/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mulП
lstm_3/lstm_cell_5/ReluRelu!lstm_3/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Relu┤
lstm_3/lstm_cell_5/mul_1Mullstm_3/lstm_cell_5/Sigmoid:y:0%lstm_3/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mul_1й
lstm_3/lstm_cell_5/add_1AddV2lstm_3/lstm_cell_5/mul:z:0lstm_3/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/add_1Ь
lstm_3/lstm_cell_5/Sigmoid_2Sigmoid!lstm_3/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Sigmoid_2О
lstm_3/lstm_cell_5/Relu_1Relulstm_3/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/Relu_1╕
lstm_3/lstm_cell_5/mul_2Mul lstm_3/lstm_cell_5/Sigmoid_2:y:0'lstm_3/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:          2
lstm_3/lstm_cell_5/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        2&
$lstm_3/TensorArrayV2_1/element_shape╘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterя
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_5_matmul_readvariableop_resource3lstm_3_lstm_cell_5_matmul_1_readvariableop_resource2lstm_3_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :          :          : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
lstm_3_while_body_50687*#
condR
lstm_3_while_cond_50686*K
output_shapes:
8: : : : :          :          : : : : : *
parallel_iterations 2
lstm_3/while├
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:          *
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2─
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/perm┴
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:          2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeЗ
dropout_1/IdentityIdentitylstm_3/strided_slice_3:output:0*
T0*'
_output_shapes
:          2
dropout_1/Identityе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_1/MatMul/ReadVariableOpа
dense_1/MatMulMatMuldropout_1/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╡
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*^lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp)^lstm_2/lstm_cell_4/MatMul/ReadVariableOp+^lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp^lstm_2/while*^lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_5/MatMul/ReadVariableOp+^lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp^lstm_3/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2V
)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp)lstm_2/lstm_cell_4/BiasAdd/ReadVariableOp2T
(lstm_2/lstm_cell_4/MatMul/ReadVariableOp(lstm_2/lstm_cell_4/MatMul/ReadVariableOp2X
*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp*lstm_2/lstm_cell_4/MatMul_1/ReadVariableOp2
lstm_2/whilelstm_2/while2V
)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_5/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_5/MatMul/ReadVariableOp(lstm_3/lstm_cell_5/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_5/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╨
╛
while_cond_48350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_48350___redundant_placeholder03
/while_while_cond_48350___redundant_placeholder13
/while_while_cond_48350___redundant_placeholder23
/while_while_cond_48350___redundant_placeholder3
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
@: : : : :         @:         @: ::::: 
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
:         @:-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
∙
Д
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52628

inputs
states_0
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:          :          :          :          *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:          2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:          2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:          2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:          2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:          2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:          2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:          2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:          2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:          2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:          2

Identity_2Щ
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
?:         @:          :          : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs:QM
'
_output_shapes
:          
"
_user_specified_name
states/0:QM
'
_output_shapes
:          
"
_user_specified_name
states/1"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╕
serving_defaultд
I
lstm_2_input9
serving_default_lstm_2_input:0         ;
dense_10
StatefulPartitionedCall:0         tensorflow/serving/predict:Те
ш
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
├
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
├
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
е
trainable_variables
regularization_losses
	variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
_tf_keras_layer
у
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
╩
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
с
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
╣
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
с
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
╣
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
н
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
 : 2dense_1/kernel
:2dense_1/bias
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
н
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
,:*	А2lstm_2/lstm_cell_4/kernel
6:4	@А2#lstm_2/lstm_cell_4/recurrent_kernel
&:$А2lstm_2/lstm_cell_4/bias
,:*	@А2lstm_3/lstm_cell_5/kernel
6:4	 А2#lstm_3/lstm_cell_5/recurrent_kernel
&:$А2lstm_3/lstm_cell_5/bias
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
н
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
н
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
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
1:/	А2 Adam/lstm_2/lstm_cell_4/kernel/m
;:9	@А2*Adam/lstm_2/lstm_cell_4/recurrent_kernel/m
+:)А2Adam/lstm_2/lstm_cell_4/bias/m
1:/	@А2 Adam/lstm_3/lstm_cell_5/kernel/m
;:9	 А2*Adam/lstm_3/lstm_cell_5/recurrent_kernel/m
+:)А2Adam/lstm_3/lstm_cell_5/bias/m
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
1:/	А2 Adam/lstm_2/lstm_cell_4/kernel/v
;:9	@А2*Adam/lstm_2/lstm_cell_4/recurrent_kernel/v
+:)А2Adam/lstm_2/lstm_cell_4/bias/v
1:/	@А2 Adam/lstm_3/lstm_cell_5/kernel/v
;:9	 А2*Adam/lstm_3/lstm_cell_5/recurrent_kernel/v
+:)А2Adam/lstm_3/lstm_cell_5/bias/v
╨B═
 __inference__wrapped_model_48262lstm_2_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■2√
,__inference_sequential_1_layer_call_fn_49888
,__inference_sequential_1_layer_call_fn_50452
,__inference_sequential_1_layer_call_fn_50473
,__inference_sequential_1_layer_call_fn_50354└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_50778
G__inference_sequential_1_layer_call_and_return_conditional_losses_51090
G__inference_sequential_1_layer_call_and_return_conditional_losses_50378
G__inference_sequential_1_layer_call_and_return_conditional_losses_50402└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
√2°
&__inference_lstm_2_layer_call_fn_51101
&__inference_lstm_2_layer_call_fn_51112
&__inference_lstm_2_layer_call_fn_51123
&__inference_lstm_2_layer_call_fn_51134╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
A__inference_lstm_2_layer_call_and_return_conditional_losses_51285
A__inference_lstm_2_layer_call_and_return_conditional_losses_51436
A__inference_lstm_2_layer_call_and_return_conditional_losses_51587
A__inference_lstm_2_layer_call_and_return_conditional_losses_51738╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
√2°
&__inference_lstm_3_layer_call_fn_51749
&__inference_lstm_3_layer_call_fn_51760
&__inference_lstm_3_layer_call_fn_51771
&__inference_lstm_3_layer_call_fn_51782╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ч2ф
A__inference_lstm_3_layer_call_and_return_conditional_losses_51933
A__inference_lstm_3_layer_call_and_return_conditional_losses_52084
A__inference_lstm_3_layer_call_and_return_conditional_losses_52235
A__inference_lstm_3_layer_call_and_return_conditional_losses_52386╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Р2Н
)__inference_dropout_1_layer_call_fn_52391
)__inference_dropout_1_layer_call_fn_52396┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_1_layer_call_and_return_conditional_losses_52401
D__inference_dropout_1_layer_call_and_return_conditional_losses_52413┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╤2╬
'__inference_dense_1_layer_call_fn_52422в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_52432в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧B╠
#__inference_signature_wrapper_50431lstm_2_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
+__inference_lstm_cell_4_layer_call_fn_52449
+__inference_lstm_cell_4_layer_call_fn_52466╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52498
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52530╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
+__inference_lstm_cell_5_layer_call_fn_52547
+__inference_lstm_cell_5_layer_call_fn_52564╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52596
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52628╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 Ь
 __inference__wrapped_model_48262x&'()*+9в6
/в,
*К'
lstm_2_input         
к "1к.
,
dense_1!К
dense_1         в
B__inference_dense_1_layer_call_and_return_conditional_losses_52432\/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ z
'__inference_dense_1_layer_call_fn_52422O/в,
%в"
 К
inputs          
к "К         д
D__inference_dropout_1_layer_call_and_return_conditional_losses_52401\3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ д
D__inference_dropout_1_layer_call_and_return_conditional_losses_52413\3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ |
)__inference_dropout_1_layer_call_fn_52391O3в0
)в&
 К
inputs          
p 
к "К          |
)__inference_dropout_1_layer_call_fn_52396O3в0
)в&
 К
inputs          
p
к "К          ╨
A__inference_lstm_2_layer_call_and_return_conditional_losses_51285К&'(OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "2в/
(К%
0                  @
Ъ ╨
A__inference_lstm_2_layer_call_and_return_conditional_losses_51436К&'(OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "2в/
(К%
0                  @
Ъ ╢
A__inference_lstm_2_layer_call_and_return_conditional_losses_51587q&'(?в<
5в2
$К!
inputs         

 
p 

 
к ")в&
К
0         @
Ъ ╢
A__inference_lstm_2_layer_call_and_return_conditional_losses_51738q&'(?в<
5в2
$К!
inputs         

 
p

 
к ")в&
К
0         @
Ъ з
&__inference_lstm_2_layer_call_fn_51101}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p 

 
к "%К"                  @з
&__inference_lstm_2_layer_call_fn_51112}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  

 
p

 
к "%К"                  @О
&__inference_lstm_2_layer_call_fn_51123d&'(?в<
5в2
$К!
inputs         

 
p 

 
к "К         @О
&__inference_lstm_2_layer_call_fn_51134d&'(?в<
5в2
$К!
inputs         

 
p

 
к "К         @┬
A__inference_lstm_3_layer_call_and_return_conditional_losses_51933})*+OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "%в"
К
0          
Ъ ┬
A__inference_lstm_3_layer_call_and_return_conditional_losses_52084})*+OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "%в"
К
0          
Ъ ▓
A__inference_lstm_3_layer_call_and_return_conditional_losses_52235m)*+?в<
5в2
$К!
inputs         @

 
p 

 
к "%в"
К
0          
Ъ ▓
A__inference_lstm_3_layer_call_and_return_conditional_losses_52386m)*+?в<
5в2
$К!
inputs         @

 
p

 
к "%в"
К
0          
Ъ Ъ
&__inference_lstm_3_layer_call_fn_51749p)*+OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p 

 
к "К          Ъ
&__inference_lstm_3_layer_call_fn_51760p)*+OвL
EвB
4Ъ1
/К,
inputs/0                  @

 
p

 
к "К          К
&__inference_lstm_3_layer_call_fn_51771`)*+?в<
5в2
$К!
inputs         @

 
p 

 
к "К          К
&__inference_lstm_3_layer_call_fn_51782`)*+?в<
5в2
$К!
inputs         @

 
p

 
к "К          ╚
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52498¤&'(Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p 
к "sвp
iвf
К
0/0         @
EЪB
К
0/1/0         @
К
0/1/1         @
Ъ ╚
F__inference_lstm_cell_4_layer_call_and_return_conditional_losses_52530¤&'(Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p
к "sвp
iвf
К
0/0         @
EЪB
К
0/1/0         @
К
0/1/1         @
Ъ Э
+__inference_lstm_cell_4_layer_call_fn_52449э&'(Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p 
к "cв`
К
0         @
AЪ>
К
1/0         @
К
1/1         @Э
+__inference_lstm_cell_4_layer_call_fn_52466э&'(Ав}
vвs
 К
inputs         
KвH
"К
states/0         @
"К
states/1         @
p
к "cв`
К
0         @
AЪ>
К
1/0         @
К
1/1         @╚
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52596¤)*+Ав}
vвs
 К
inputs         @
KвH
"К
states/0          
"К
states/1          
p 
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ ╚
F__inference_lstm_cell_5_layer_call_and_return_conditional_losses_52628¤)*+Ав}
vвs
 К
inputs         @
KвH
"К
states/0          
"К
states/1          
p
к "sвp
iвf
К
0/0          
EЪB
К
0/1/0          
К
0/1/1          
Ъ Э
+__inference_lstm_cell_5_layer_call_fn_52547э)*+Ав}
vвs
 К
inputs         @
KвH
"К
states/0          
"К
states/1          
p 
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          Э
+__inference_lstm_cell_5_layer_call_fn_52564э)*+Ав}
vвs
 К
inputs         @
KвH
"К
states/0          
"К
states/1          
p
к "cв`
К
0          
AЪ>
К
1/0          
К
1/1          ┐
G__inference_sequential_1_layer_call_and_return_conditional_losses_50378t&'()*+Aв>
7в4
*К'
lstm_2_input         
p 

 
к "%в"
К
0         
Ъ ┐
G__inference_sequential_1_layer_call_and_return_conditional_losses_50402t&'()*+Aв>
7в4
*К'
lstm_2_input         
p

 
к "%в"
К
0         
Ъ ╣
G__inference_sequential_1_layer_call_and_return_conditional_losses_50778n&'()*+;в8
1в.
$К!
inputs         
p 

 
к "%в"
К
0         
Ъ ╣
G__inference_sequential_1_layer_call_and_return_conditional_losses_51090n&'()*+;в8
1в.
$К!
inputs         
p

 
к "%в"
К
0         
Ъ Ч
,__inference_sequential_1_layer_call_fn_49888g&'()*+Aв>
7в4
*К'
lstm_2_input         
p 

 
к "К         Ч
,__inference_sequential_1_layer_call_fn_50354g&'()*+Aв>
7в4
*К'
lstm_2_input         
p

 
к "К         С
,__inference_sequential_1_layer_call_fn_50452a&'()*+;в8
1в.
$К!
inputs         
p 

 
к "К         С
,__inference_sequential_1_layer_call_fn_50473a&'()*+;в8
1в.
$К!
inputs         
p

 
к "К         ░
#__inference_signature_wrapper_50431И&'()*+IвF
в 
?к<
:
lstm_2_input*К'
lstm_2_input         "1к.
,
dense_1!К
dense_1         