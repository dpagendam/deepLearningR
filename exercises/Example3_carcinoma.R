#install.packages("pbapply")
library(pbapply)

library("devtools")
#install_github("aoles/EBImage")
library(EBImage)
library(keras)
library(deepLearningR)




width <- 50
height <- 50
grayScale <- FALSE
packageDataDir = system.file("extdata", package="deepLearningR")
trainData <- extract_feature(paste0(packageDataDir, "/carcinoma/train/"),
                             width, height, grayScale, TRUE)
testData <- extract_feature(paste0(packageDataDir, "/carcinoma/test/"), 
                            width, height, grayScale, TRUE)





numInputChannels <- 3
train_array <- t(trainData$X)
dim(train_array) <- c(width, height, numInputChannels, 
                      nrow(trainData$X))
train_array <- aperm(train_array, c(4,1,2,3))

test_array <- t(testData$X)
dim(test_array) <- c(width, height, numInputChannels, 
                     nrow(testData$X))
test_array <- aperm(test_array, c(4,1,2,3))





model <- keras_model_sequential()

model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")

model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")





model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")

model %>% layer_flatten()
model %>% layer_dense(units = 8, activation = "relu")
model %>% layer_dense(units = 8, activation = "relu")
model %>% layer_dense(units = 8, activation = "relu")
model %>%layer_dense(units = 1, activation = "sigmoid")




model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metrics = c('accuracy')
)




history <- model %>% fit(
  x = train_array, y = as.numeric(trainData$y),
  epochs = 50, batch_size = 32, validation_data =
    list(test_array,as.numeric(testData$y))
)




plot(history)




model <- keras_model_sequential()

model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")

model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")




model %>% layer_conv_2d(kernel_size = c(5,5), filters = 8,
                        strides = 1, activation = "relu", padding = "same", 
                        input_shape = c(width, height, numInputChannels),
                        data_format="channels_last") %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2,2), padding = "same")

model %>% layer_flatten()
model %>% layer_dense(units = 8, activation = "relu")
model %>% layer_dense(units = 8, activation = "relu")
model %>% layer_dense(units = 8, activation = "relu")
model %>%layer_dense(units = 1, activation = "sigmoid")




model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metrics = c('accuracy')
)



history <- model %>% fit(
  x = train_array, y = as.numeric(trainData$y),
  epochs = 20, batch_size = 32, validation_data =
    list(test_array,as.numeric(testData$y))
)




plot(history)




# We can obtain the outputs of the model (sigmoid activation)
probabilities <- predict_proba(model, test_array)
# We obtain the predicted classes for the test/validation dataset
predictions <-  predict_classes(model, test_array)
truth <- testData$y
propCorrect <- sum(predictions == truth)/length(truth)
print(propCorrect)





random <- sample(1:nrow(testData$X), 16)
plot_preds <- predictions[random,]
plot_probs <- as.vector(round(probabilities[random,], 2))
plot_truth <- truth[random]

par(mfrow = c(4, 4), mar = rep(0, 4))
for(i in 1:length(random)){
  if(grayScale)
  {
    image(t(apply(test_array[random[i],,,], 2, rev)),
          col = gray.colors(12), axes = F)
  }
  if(!grayScale)
  {
    image(t(apply(test_array[random[i],,,1], 2, rev)),
          col = fade(hcl.colors(12, "YlOrRd", rev = FALSE), 100), axes = F)
  }
  legend("top", legend = paste0("Pred: ", ifelse(plot_preds[i] == 0, "IDC Neg.", "IDC Pos.")),
         text.col = ifelse(plot_preds[i] == 0, 4, 2), bty = "n", text.font = 2)
  legend("center", legend = plot_probs[i], bty = "n", cex = 2, text.col = "black")
  legend("bottom", legend = paste0("Truth: ", ifelse(plot_truth[i] == 0, "IDC Neg.", "IDC Pos.")), text.col = ifelse(plot_truth[i] == 0, 4, 2), bty = "n", text.font = 2)
}


index <- 70
test_output <- model$output[, 1]
last_conv_layer <- model %>% get_layer("conv2d_3")
grads <- k_gradients(test_output, last_conv_layer$output)[[1]]
pooled_grads <- k_mean(grads, axis = c(1,2,3))
iterate <- k_function(list(model$input), list(pooled_grads, last_conv_layer$output[1, , , ]))
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(array(train_array[index, , , ], c(1, 50, 50, 3))))

for(i in 1:8)
{
  conv_layer_output_value[, , i] <- conv_layer_output_value[, , i]*pooled_grads_value[[i]]
}

heatmap <- apply(conv_layer_output_value, c(1,2), mean)
heatmap <- (heatmap - min(heatmap))/(max(heatmap) - min(heatmap))
testImage <- train_array[index, , , ]
red <- raster(testImage[, , 1]*255)
green <- raster(testImage[, , 2]*255)
blue <- raster(testImage[, , 3]*255)
testImageBrick <- brick(red, green, blue)
par(mfrow = c(1, 2))
plotRGB(testImageBrick)
heatmap_red <- resample(raster(heatmap*255), red, method = "ngb")
heatmap_green <- resample(raster(heatmap*0), green, method = "ngb")
heatmap_blue <- resample(raster(heatmap*0), blue, method = "ngb")
heatBrick <- brick(heatmap_red, heatmap_green, heatmap_blue)
plotRGB(heatBrick)

