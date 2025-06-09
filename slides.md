---
# You can also start simply with 'default'
theme: seriph
colorSchema: light
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: #D8BFD8
background: '#f5f0ff'
# some information about your slides (markdown enabled)
title: Full-stack quantum machine learning on hybrid quantum-classical platforms 
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
---

## Full-stack quantum machine learning on hybrid quantum-classical platforms
<br>

### Andrea Papaluca
#### andrea.papaluca@unimi.it

<div class="absolute bottom-4 left-4 flex space-x-8">
<img src="https://www.unidformazione.com/wp-content/uploads/2020/12/unimi-universita-milano-statale-1.png" width="300"></img>
<img src="https://raw.githubusercontent.com/qiboteam/qibo/refs/heads/master/doc/source/_static/qibo_logo_dark.svg" width=250></img>
</div>

<div class="abs-br m-6 text-xl">
  <a href="https://github.com/BrunoLiegiBastonLiegi" target="_blank" class="slidev-icon-btn">
    <carbon:logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

# A worldwide open source collaboration

<div class="grid grid-cols-[1fr_2fr] gap-4 place-items-center">
  <img src="/qibo_github.png">
  <img src="/qibo_collaboration.png">
</div>
<br>
<div class="grid grid-cols-[2fr_1fr] gap-4 place-items-center">
<div>
<a href="https://arxiv.org/abs/2009.01845" class="text-blue-600">"Qibo: a framework for quantum simulation with hardware acceleration"</a>
<br>
<t style="font-size: 15px;">Efthymiou et al, 2020.</t>
</div>
<img src="/qibo_arxiv_qr.svg" width=100>
</div>

---

# A flexible full-stack ecosystem

<svg class="absolute w-0 h-0">
  <defs>
    <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto">
      <polygon points="0 0, 6 2, 0 4" fill="orange" />
    </marker>
  </defs>
</svg>

<div class="flex justify-center h-full">
<div class="relative w-[600px]">
  <!-- SVG Image -->
  <img
    src="https://raw.githubusercontent.com/qiboteam/xmind-diagrams/refs/heads/master/docs/qibo_ecosystem_webpage.svg"
    class="w-full"
  />

  <!-- Highlight box -->
  <div v-click>
  <div class="absolute top-[1px] left-[423px] w-[180px] h-[211px] border-2 border-blue-500 rounded-md"></div>

  <!-- Annotation label -->
  <div class="absolute top-[-20px] left-[420px] text-xs bg-white px-1 rounded shadow">
    Simulation
  </div>
  </div>
  
  <div v-click>
  <div class="absolute top-[211px] left-[423px] w-[80px] h-[70px] border-2 border-black-500 rounded-md"></div>
  <div class="absolute top-[221px] left-[511px] text-xs bg-white px-1 rounded shadow">
    Remote <br> Access
  </div>
  </div>

  <div v-click>
  <div class="absolute top-[320px] left-[200px] w-[330px] h-[120px] border-2 border-green-500 rounded-md"></div>
  <div class="absolute top-[325px] left-[540px] text-xs bg-white px-1 rounded shadow">
    Selfhosted Hardware<br>(calibration & control)
  </div>
  </div>

  
  <div v-click>
  <div class="absolute top-[281px] left-[108px] w-[83px] h-[35px] border-2 border-orange-500 rounded-md"></div>

<!-- Arrow to Simulation -->
<svg class="absolute top-[115px] left-[170px] w-[260px] h-[160px]" viewBox="0 0 260 160">
  <line x1="0" y1="160" x2="245" y2="0" stroke="orange" stroke-width="1.2" marker-end="url(#arrowhead)" />
</svg>

<!-- Arrow to Remote Access -->
<svg class="absolute top-[245px] left-[200px] w-[245px] h-[120px]" viewBox="0 0 240 60">
  <line x1="0" y1="20" x2="210" y2="-20" stroke="orange" stroke-width="1.2" marker-end="url(#arrowhead)" />
</svg>

<!-- Arrow to Hardware -->
<svg class="absolute top-[312px] left-[170px] w-[250px] h-[60px]" viewBox="0 0 250 60">
  <line x1="0" y1="10" x2="25" y2="20" stroke="orange" stroke-width="1.2" marker-end="url(#arrowhead)" />
</svg>

 <div class="absolute top-[261px] left-[88px] text-xs bg-white px-1 rounded shadow">
    ML Interface
  </div>

  </div>
   
  
</div>
</div>

---

# The Quantum Computer

<div class="grid grid-cols-[2fr_1fr] gap-4 items-stretch">

<div>
<img src="/cryostat.png">

<div class="flex justifu-start">
<img src="/nqch_logo.jpg" width=128>
</div>

</div>

<div>
<img src="/quantum_computer.png" width=200>
<br>
<img src="/chips.png" width=250>
</div>

</div>

---

# Quantum Computation through circuits

<div class="grid grid-cols-[2fr_1fr] gap-4 place-items-center">
<div class="grid grid-cols-1 place-items-center">
<img src="/example_circuit.svg" width="400"/>
<br>
<img src="https://user-images.githubusercontent.com/89847233/263536880-171e4364-a42f-4e63-92d3-da00cbcd9fbb.gif" width="200"/>
</div>
<div>
```python {all|5|8,9,11|15,16}
import numpy as np
from qibo import Circuit, gates

# Construct the circuit
circuit  = Circuit(3)
# Add the gates
for q in range(3):
	circuit.add(gates.RY(q=q, theta=np.random.randn()))
	circuit.add(gates.RZ(q=q, theta=np.random.randn()))
	for q in range(3):
		circuit.add(gates.CRX(
			q%3, (q+1)%3, theta=np.random.randn())
		)
# Execute the circuit and get the final state
result = circuit() 
print(result.state())
```
</div>
</div>

---

# Quantum Machine Learning (QML)

<div class="grid grid-cols-1 place-items-center">
	<img src="/example_circuit_qml.svg" width=700/>
	
</div>

- Decode the quantum information contained in the final state
- Compute the loss
- Update the gates' parameters to minimize the loss

---

# The Qiboml pipeline

<div class="grid grid-cols-1 place-items-center">
	<img src="/qiboml_pipeline.png" width=700/>
</div>

---

# QML models made easy

<div class="grid grid-cols-2 gap-4 items-stretch">

<div>

#### Interface agnostic

```python {all|5|6,9|11}
from qiboml.models.encoding import PhaseEncoding
from qiboml.models.decoding import Expectation

# Encoding layer
encoding = PhaseEncoding(3, encoding_gate=gates.RX)
structure = []
# Alternate encoding and trainable layers
for _ in range(5):
    structure.extend([encoding, circuit.copy()])	
# Decoding layer
decoding = Expectation(3)
# Prepare some data
x = np.random.uniform(
	0, 2, 300
	).reshape(100, 3) * np.pi
y = np.sin(x).sum(-1) 
```
</div>

<div>
<div v-click=[4]>

#### Torch interface

```python
from qiboml.interfaces.pytorch import QuantumModel
import torch

# This is a torch.nn.Module
q_model = QuantumModel(structure, decoding=decoding)
# you can train it as you normally do for torch 
# models
optimizer = torch.optim.Adam(q_model.parameters())
for epoch in range(10):
    optimizer.zero_grad()
    prediction = torch.stack([
		    q_model(data) 
		    for data in torch.as_tensor(x)
		])
    loss = torch.nn.functional.mse_loss(
		    prediction, 
		    torch.as_tensor(y)
		)
    loss.backward()
    optimizer.step()

```
</div>
<div v-click=[5] v-motion
  :initial="{ x: -50, y: -410}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">
  
#### Keras interface

```python
from qiboml.interfaces.keras import QuantumModel
import tensorflow as tf

# This is a keras.Model
q_model = QuantumModel(structure, decoding=decoding)
# you can train it as you normally do for keras 
# models
q_model.compile("adam", "mean_squared_error")
q_model.fit(
	tf.convert_to_tensor(x), 
	tf.convert_to_tensor(y), 
	batch_size=1, 
	epochs=10
)
```
</div>
</div>

</div>

---

# Deploy on any device: even selfhosted QPUs!

<div class="grid grid-cols-[1fr_1.5fr] gap-4 items-stretch">

<div>

<br>
<br>

<v-clicks every="2">
	
- ## *Just In Time* (JIT) CPU 

<br>

- ## GPUs

<br>

- ## Cloud Providers

<br>

- ## Selfhosted QPUs
	
</v-clicks>
</div>

<div>

<div v-click=[1] class="flex flex-col space-y-2">

  <!-- Title and icons -->
  <div class="flex justify-end">
    <img src="https://numba.pydata.org/_static/numba-blue-icon-rgb.svg" width="24" />
    <img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" width=36>
  </div>

  <!-- Code block -->
  <div>
```python
from qibo import set_backend

# Use numba
set_backend("qibojit", platform="numba")

# Use jax
set_backend("qiboml", platform="jax")
```
</div>
</div>

<div v-click=[2] class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: -170}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">

  <!-- Title and icons -->
  <div class="flex justify-end">
    <img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" width="64" />
	<img src="https://camo.githubusercontent.com/59f1d1cd5dc5748444d385a7c17d0652001ba959ada26e6bc93a0f82569900a1/68747470733a2f2f646576656c6f7065722e6e76696469612e636f6d2f73697465732f64656661756c742f66696c65732f616b616d61692f6e76696469612d63757175616e74756d2d69636f6e2e737667" width=24/>
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" width=128>
	<img src="https://www.gstatic.com/devrel-devsite/prod/vd9663438c989ac592eff7c92ff013bc8fa2578bc40babda19f4e44265d95782f/tensorflow/images/lockup.svg" width=128>
  </div>

  <!-- Code block -->
  <div>
```python
from qibo import set_backend

# Use cupy
set_backend("qibojit", platform="cupy")

# Use nvidia cuquantum
set_backend("qibojit", platform="cuquantum")

# Use torch 
set_backend("qiboml", platform="pytorch")

# Use tensorflow
set_backend("qiboml", platform="tensorflow")
```
</div>
</div>

<div v-click=[3] class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: -470}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">

  <!-- Title and icons -->
  <div class="flex justify-end">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/250px-IBM_logo.svg.png" width="48"/>
	<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/250px-Amazon_Web_Services_Logo.svg.png" width="36">
	<img src="https://avatars.githubusercontent.com/u/25356822?s=200&v=4" width=24>
	<img src="https://avatars.githubusercontent.com/u/59836348?s=200&v=4" width=24>
  </div>

  <!-- Code block -->
  <div>
```python
from qibo import set_backend

# Use ibmq
set_backend("qibo-cloud-backends", platform="qiskit-client")

# Use aws
set_backend("qibo-cloud-backends", platform="braket-client")

# Use ionq
set_backend("qibo-cloud-backends", platform="ionq-client")

# Use qibo at TII
set_backend("qibo-cloud-backends", platform="qibo-client")
```
</div>
</div>


<div v-click.after="4" class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: -670}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">

  <!-- Title and icons -->
  <div class="flex justify-end">
    <img src="https://raw.githubusercontent.com/qiboteam/qibo/refs/heads/master/doc/source/_static/qibo_logo_dark.svg" width=48></img>
  </div>

  <!-- Code block -->
  <div>
```python
from qibo import set_backend

# Use your selfhosted device through qibolab!
set_backend("qibolab", platform="my_local_platform")
```

</div>
</div>


</div>

</div>

---

# Fitting functions on Quantum Hardware

<div class="grid grid-cols-[1fr_1.5fr] gap-4 items-stretch">

<div>
```python
# Target function
def f(x):
    return torch.sin(x) ** 2 - 0.3 * torch.cos(x)
# Training Dataset
num_samples = 30
x_train = torch.linspace(
	0, 2 * np.pi, num_samples
	).unsqueeze(1)
y_train = f(x_train)
```
<div v-click.after="1" class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: 0}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">
```python {all|all|2|4-6|12|10}
# Using qibolab at NQCH
set_backend("qibolab", platform="sinq-20")
# Define the transpiler
glist = [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
natives = NativeGates(0).from_gatelist(glist)
transpiler = Passes([Unroller(natives)])
# Decoding layer
decoding = Expectation(
	nqubits=1,
	wire_names=[19], # select the qubit to use
	nshots=1024, # set the number of shots
	transpiler=transpiler
)
```
</div>
</div>

<div>
<div class="grid grid-cols-1 gap-2 place-items-center">

<div>
<img src="/target_function.png" width=250>
</div>

<div>
<div v-click.after="1" class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: 0}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">
<img src="/fit_sinq.png" width=250>
<div class="flex justify-end">
<img src="/nqch_logo.jpg" width=128>
</div>
</div>
</div>

</div>
</div>

</div>

---

# Mitigating Errors

<div class="grid grid-cols-2 gap-4 items-stretch">

<div>

#### Simulated Pauli Noise 

<img src="/animation_noise.gif" width=300>

</div>

<div v-click.after="1" class="flex flex-col space-y-2" v-motion
  :initial="{ x: -40, y: -5}"
  :enter="{ x: 0 }"
  :leave="{ x: 50 }">

#### CDR mitigation

<img src="/animation_mit.gif" width=300>

<div v-motion
  :initial="{ x: -280, y: -5}"
  :enter="{ x: -240 }"
  :leave="{ x: -190 }">
```python
decoding_circ = Expectation(
    nshots=5000,
    nqubits=1, 
    mitigation_config={
    "real_time": True,
    "method": "CDR",
    "method_kwargs": {"n_training_samples": 50}
	}
)
```
</div>

</div>

</div>


---


# Surpassing qubit fidelity

<!-- Top section: figure with top-right icon -->
<div class="relative w-[400px] mx-auto">
  <!-- Main image -->
  <img src="/rtqem_quark.png" width="350" />

  <!-- Positioned icon -->
  <img
    src="https://avatars.githubusercontent.com/u/59836348?s=200&v=4"
    class="absolute top-2 right-2"
    width="40"
  />
</div>

<!-- Bottom section: citation and QR code -->
<div class="grid grid-cols-[4fr_1fr] items-center mt-4">

  <div>
    <a href="https://arxiv.org/abs/2311.05680" class="text-blue-600">
      “Real-time error mitigation for variational optimization on quantum hardware”
    </a>
    <br>
    Robbiati et al.
  </div>

  <img src="/rtqem_qr.svg" width="128" />

</div>

---

# Finding ground states through VQEs

<div class="grid grid-cols-[1fr_3fr] gap-4 items-stretch">

<div>
```python
# by default this computes <Z0 + Z1 + Z2>
# any observable can be used though
decoder = Expectation(
	nqubits=3, 
	wire_names=[8, 3, 13], 
	nshots=1000, 
	transpiler=transpiler
	)
circuit = HardwareEfficient(3, nlayers=2)
model = QuantumModel([circuit,], decoder)
optimizer = torch.optim.Adam(
	model.parameters(), 
	lr=0.25
	)
for epoch in range(50):
	optimizer.zero_grad()
	cost = model()
	cost.backward()
	optimizer.step()    
```
</div>

<div>
<img src="/vqe_example.svg">
<div class="flex justify-end">
<img src="/nqch_logo.jpg" width=128>
</div>
</div>

</div>

---

