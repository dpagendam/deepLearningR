library(deepLearningR)
data("buzz_X1")
data("buzz_X2")
data("buzz_X3")
data("buzz_X4")
data("buzz_X5")
data("buzz_X6")
data("buzz_Y")
X <- rbind(X1, X2, X3, X4, X5, X6)



str(X)
str(Y)
unique(Y)


trainInds <- sample(1:nrow(X), round(nrow(X)*0.9))
valInds <- (1:nrow(X))[-trainInds]


Xtrain <- X[trainInds, ]
Ytrain_classes <- Y[trainInds]


Xtest <- X[valInds, ]
Ytest_classes <- Y[valInds]



rescaleCols <- function(rowX, colMins, colMaxs)
{
  r <- (rowX - colMins)/(colMaxs - colMins)
  r[is.nan(r)] <- 0
  return(r)
}



colMinsX <- apply(Xtrain, 2, min)
colMaxsX <- apply(Xtrain, 2, max)



Xtrain_scaled <- t(apply(Xtrain, 1, rescaleCols, colMinsX, colMaxsX))
Xtest_scaled <- t(apply(Xtest, 1, rescaleCols, colMinsX, colMaxsX))



range(Xtrain_scaled)
range(Xtest_scaled)



library(keras)
Ytrain <- to_categorical(Ytrain_classes)
Ytest <- to_categorical(Ytest_classes)


head(Ytrain)



model <- keras_model_sequential()
inDim <- ncol(Xtrain_scaled)
model %>% layer_dense(512, "relu", input_shape = inDim)
model %>% layer_dense(units = 512, activation = "relu")
model %>% layer_dense(units = 512, activation = "relu")
model %>% layer_dense(units = 6, activation = "softmax")



print(model)



model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.01, decay = 0.01),
  metrics = c('accuracy')
)




history <- model %>% fit(
  x = Xtrain_scaled, y = Ytrain,
  epochs = 20, batch_size = 32, 
  validation_data = list(Xtest_scaled, Ytest))




plot(history)




predictions <- predict_classes(model, Xtest_scaled)
truth <- Ytest_classes
table(predictions, truth)









