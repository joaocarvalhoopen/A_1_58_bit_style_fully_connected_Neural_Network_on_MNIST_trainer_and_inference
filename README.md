# A 1.58 bit style fully connected Neural Network on MNIST trainer and inference
A interesting concept, that trains the neural network at each epoch with 1.58 bits ternary states -1, 0, 1.

## Description
This is a recent type of neural network that some researchers have reasearched in same papers, and it as the weights in f32 for trainning and backpropagation but that at each step the f32 weights are quantitizes to ternary values

```
Ternary Weights { -1, 0 ,1} 
log2( 3 ) = 1.58 bits
```

And evaluated with this encoding. This quantization aware training is suposed to produce better results then a posterior quantization from BP16 or F32 into a lower number of bits. Because it is trainned on those bits. Also, the weights that in a normal network were multiplied by the inputs of each layer in this case ( because we only have the value -1, 0, 1 ) they are integer summed up. And this is a less costly operation, then a multiplication. And if we encode the each weight 1.58 bits into a ternary (trits instead of bits) representation and then generate the decimal and from the decimal the binary encoding, we can fit into each 8 bit u8, 5x 1.58 bits, that means that this representation has the possibility to be 5x smaller then the i8 byte representation and 20x more compressed and less bandwith intensive then f32 representation. In this project the training is multi threaded, but the procesing of the 1.58 bits is not SIMD (at least for now). This was a small project for me to learn more about this recent techniques. This program is made in the Odin programming language.

## Note
You have to decompress the zip file into its diretory.

## Output

### Train on CPU

```
time ./bitnet_mnist.exe
--train-images mnist_idx/train-images-idx3-ubyte
--train-labels mnist_idx/train-labels-idx1-ubyte
--test-images  mnist_idx/t10k-images-idx3-ubyte
--test-labels  mnist_idx/t10k-labels-idx1-ubyte
--epochs 15 --batch 512 --hidden 256
--lr_start 0.005 --lr_stop 0.0002 --threads 24 
--limit-train 50000 --limit-test 10000


Loaded MNIST: train=50000 ( of 60000 ), test=10000 ( of 10000 )

Training with 24 threads

Epoch 1 / 15  loss=0.31832  test_acc=0.9536  learning_rate=0.007820
Epoch 2 / 15  loss=0.11515  test_acc=0.9679  learning_rate=0.006115
Epoch 3 / 15  loss=0.07958  test_acc=0.9682  learning_rate=0.004782
Epoch 4 / 15  loss=0.05762  test_acc=0.9760  learning_rate=0.003739
Epoch 5 / 15  loss=0.03995  test_acc=0.9758  learning_rate=0.002924
Epoch 6 / 15  loss=0.02726  test_acc=0.9782  learning_rate=0.002287
Epoch 7 / 15  loss=0.02096  test_acc=0.9777  learning_rate=0.001788
Epoch 8 / 15  loss=0.01488  test_acc=0.9796  learning_rate=0.001398
Epoch 9 / 15  loss=0.01146  test_acc=0.9816  learning_rate=0.001093
Epoch 10 / 15  loss=0.00868  test_acc=0.9811  learning_rate=0.000855
Epoch 11 / 15  loss=0.00645  test_acc=0.9834  learning_rate=0.000669
Epoch 12 / 15  loss=0.00529  test_acc=0.9824  learning_rate=0.000523
Epoch 13 / 15  loss=0.00468  test_acc=0.9820  learning_rate=0.000409
Epoch 14 / 15  loss=0.00388  test_acc=0.9826  learning_rate=0.000320
Epoch 15 / 15  loss=0.00350  test_acc=0.9827  learning_rate=0.000250
Saved weights to weights.bla

real    0m14.005s
user    2m37.214s
sys     0m2.195s
```

### Inference on CPU

``` 
./bitnet_mnist.exe --weights weights.bla --infer-index 300

Loaded MNIST: train=50000 ( of 60000 ), test=10000 ( of 10000 )

Loaded weights from weights.bla ( hidden=256, step=1770 )

Infer split=test  index=300
True label: 4
Pred label: 4
Probabilities:
  0: 0.000040
  1: 0.000051
  2: 0.048937
  3: 0.000006
  4: 0.947977
  5: 0.000002
  6: 0.002937
  7: 0.000048
  8: 0.000000
  9: 0.000001

Image ( ASCII ):





                              ****
                            ..%%##
                            ::@@##
                            ##%%++
                          ==%%%%::
                        --%%%%%%..
                      ..%%%%##%%..
                    ..%%%%..%%%%..
                  ..##%%..  %%##
                  ##%%--  ::%%==            ::::
                ==%%==    ##%%          --==**--
                ==%%%%****%%%%==--******%%##--
                  ::++%%%%%%%%%%%%####++
                        %%**
                      ==%%::
                      **%%
                      ##**
                      %%--
                      %%..
                    ..**
```

## References
1. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits <br>
   by Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei <br>
   [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)

2. BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks <br>
   by Jacob Nielsen, Peter Schneider-Kamp <br>
   [https://arxiv.org/abs/2407.09527](https://arxiv.org/abs/2407.09527)

3. GitHub Microsoft project bitnet.cpp <br>
   [https://github.com/microsoft/bitnet](https://github.com/microsoft/bitnet)

## License
MIT Open Source License

## Have fun
Best regards, <br>
Joao Carvalho
