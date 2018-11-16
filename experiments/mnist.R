setwd("~/Desktop/t-SNE/FIt-SNE-master")
source('fast_tsne.R')

> setwd("~/Desktop/t-SNE/FIt-SNE-master")
mnist <- read_csv("https://pjreddie.com/media/files/mnist_train.csv", col_names = FALSE)
mnist <- as.matrix(mnist)


# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org

load_mnist <- function() {
    load_image_file <- function(filename) {
        ret = list()
        f = file(filename,'rb')
        readBin(f,'integer',n=1,size=4,endian='big')
        ret$n = readBin(f,'integer',n=1,size=4,endian='big')
        nrow = readBin(f,'integer',n=1,size=4,endian='big')
        ncol = readBin(f,'integer',n=1,size=4,endian='big')
        x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
        ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
        close(f)
        ret
    }
    load_label_file <- function(filename) {
        f = file(filename,'rb')
        readBin(f,'integer',n=1,size=4,endian='big')
        n = readBin(f,'integer',n=1,size=4,endian='big')
        y = readBin(f,'integer',n=n,size=1,signed=F)
        close(f)
        y
    }
    train <<- load_image_file('mnist/train-images-idx3-ubyte')

    
    train$y <<- load_label_file('mnist/train-labels-idx1-ubyte')

}

train <- data.frame()

show_digit <- function(arr784, col=gray(12:1/12), ...) {
    image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

load_mnist()

PCA1=prcomp(train$x,center = T,scale. = F)

mnist_tsne<- fftRtsne(PCA1$x[,1:20],2, rand_seed = 11000, ann_not_vptree = FALSE)
plot(mnist_tsne[,1],mnist_tsne[,2], col = train$y+1, main = 'With K/2 near neighbors shuffled, seed 11000')

