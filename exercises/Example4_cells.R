#install.packages("pbapply")
library(pbapply)

library("devtools")
#install_github("aoles/EBImage")
library(EBImage)
library(keras)
library(deepLearningR)



data(cells)



grayScale <- FALSE
width <- 64
height <- 64



packageDataDir = system.file("extdata", package="deepLearningR")
trainData <- extract_feature(paste0(packageDataDir, "/cells/train/"),
            width, height, grayScale, TRUE)
trainData$y <- cellCounts[21:200]
print(range(trainData$X))



testData <- extract_feature(paste0(packageDataDir, "/cells/test/"), 
            width, height, grayScale, TRUE)
testData$y <- cellCounts[1:20]
print(range(testData$X))




numInputChannels <- 3
train_array <- t(trainData$X)
dim(train_array) <- c(width, height, numInputChannels, 
                      nrow(trainData$X))
train_array <- aperm(train_array, c(4,1,2,3))

test_array <- t(testData$X)
dim(test_array) <- c(width, height, numInputChannels, 
                     nrow(testData$X))
test_array <- aperm(test_array, c(4,1,2,3))




datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 0,
  shear_range = 0,
  zoom_range = 0,
  horizontal_flip = TRUE,
  vertical_flip = TRUE,
  fill_mode = "wrap",
  width_shift_range = 0.05,
  height_shift_range = 0.05
)




numNewImages <- 300
batchSize = 50
d <- dim(train_array)
train_array_augmented <- array(NA, c(d[1] + numNewImages, d[2:4]))
train_array_augmented[1:d[1], , , ] <- train_array
y_augmented <- trainData$y
for(i in 1:(numNewImages/batchSize))
{
  augmentation_generator <- flow_images_from_data(x = train_array,
                                                  y = trainData$y,generator = datagen,
                                                  batch_size = batchSize)
  newImages <- generator_next(augmentation_generator)
  fromInd = d[1] + (i-1)*batchSize + 1
  toInd = d[1] + i*batchSize
  train_array_augmented[fromInd:toInd, , , ] <- newImages[[1]]
  y_augmented <- c(y_augmented, newImages[[2]])
}





model <- keras_model_sequential()

model %>% layer_conv_2d(kernel_size = c(3,3), filters = 8, strides = 1,
                        activation = "relu", padding = "same",
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")

model %>% layer_conv_2d(kernel_size = c(3,3), filters = 16, strides = 1,
                        activation = "relu", padding = "same",
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")






model %>% layer_conv_2d(kernel_size = c(3,3), filters = 32, strides = 1,
                        activation = "relu", padding = "same",
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")





model %>% layer_flatten()
model %>% layer_dropout(0)
model %>% layer_dense(units = 64, activation = "relu")
model %>% layer_dropout(0)
model %>% layer_dense(units = 64, activation = "relu")
model %>% layer_dropout(0)
model %>% layer_dense(units = 64, activation = "relu")
model %>% layer_dropout(0)
model %>% layer_dense(units = 1, activation = "relu")




model %>% compile(
  loss = 'mse',
  optimizer = optimizer_rmsprop(lr = 0.001),
  metrics = c('mae')
)



useAugmentedData = TRUE
if(useAugmentedData)
{
  history <- model %>% fit(
    x = train_array_augmented, y = as.numeric(y_augmented),
    epochs = 20, batch_size = 8, validation_data = list(test_array, as.numeric(testData$y))
  )
}
if(!useAugmentedData)
{
  history <- model %>% fit(
    x = train_array, y = as.numeric(trainData$y),
    epochs = 20, batch_size = 8, validation_data = list(test_array, as.numeric(testData$y))
  )
}




plot(history)



predictions <-  predict(model, test_array)
truth <- testData$y
print(mean(predictions - truth))
print(sd(predictions - truth))




plot(truth, predictions, xlab = "True Cell Count", 
     ylab = "Predicted Cell Count", xlim = c(0, 350), ylim = c(0, 350))
abline(0,1, col = "red")





