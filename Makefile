
all:
	odin build . -out:bitnet_mnist.exe -o:speed

all_no_bounds_check:
	odin build . -out:bitnet_mnist.exe -o:speed -no-bounds-check

all_opti:
	odin build . -out:bitnet_mnist.exe -o:aggressive -no-bounds-check -microarch:native


clean:
	rm -f ./bitnet_mnist.exe

run:
	./bitnet_mnist.exe --epochs 3 --batch 64 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24

run_2:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 6 --batch 512 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_3:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 10 --batch 64 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_4:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 15 --batch 512 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_5:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 30 --batch 512 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_6:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 50 --batch 64 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_7:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 50 --batch 512 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_8:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 100 --batch 64 --hidden 256 --lr 0.0005 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_9:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 100 --batch 256 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_10:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 100 --batch 512 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_11:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 100 --batch 1024 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_12:
	./bitnet_mnist.exe \
	--train-images mnist_idx/train-images-idx3-ubyte \
	--train-labels mnist_idx/train-labels-idx1-ubyte \
	--test-images  mnist_idx/t10k-images-idx3-ubyte \
	--test-labels  mnist_idx/t10k-labels-idx1-ubyte \
	--epochs 150 --batch 2048 --hidden 256 --lr_start 0.005 --lr_stop 0.0002 --threads 24 \
	--limit-train 60000 --limit-test 10000

run_inference:
	# ./bitnet_mnist.exe --weights weights.bla --infer-index 123
	./bitnet_mnist.exe --weights weights.bla --infer-index 300
