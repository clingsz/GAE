Guided Auto Encoder (GAE)
=======

Train an autoencoder with a guided value, where the codes are informative of the guided value.

# Example

```Training
import gae.model.trainer as trainer
gae = trainer.build_gae()
gae.fit(train_x,train_y)
```

```Prediction
gae.predict(train_x)
```

# Prerequisites:

Numpy, Scipy, Theano, Sklearn

