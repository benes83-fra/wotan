mkdir -p data/mnist
cd data/mnist
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
curl -O https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
