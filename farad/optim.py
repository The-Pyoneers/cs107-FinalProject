class SGD(object):

    def __init__(self, params, lr=1e-3, decay=1e-6):
        """Implementation of stochastic gradient descent optimizer.

        Parameters
        ==========
        params : list
            Iterable list of model hyperparameters.
        lr : float
            The learning rate step used during each optimization stage.
        decay : magnitude of L2 regularization parameter.

        Returns
        =======
        NoReturn : None
            Performs weight update via stochastic gradient descent optimization
            procedure.
        
        Notes
        =====
        This implementation is a basic form of stochastic gradient descent. More
        involved and complex gradient descent algorithms can be derived from this class
        such as Adam, AdaDelta, etc.
        """

        if params is None or not isinstance(params, list):
            raise ValueError("Parameter list is not of type list.")
        if lr is None or lr < 0.0 or lr > 1.0:
            raise ValueError("Learning rate must be a positive value between zero and one.")
        
        self.params = params
        self.lr = lr
        self.decay = decay


    def step(self):
        """
        Performs a single optimization step of SGD optimizer.
        """
        for param in self.params:
            if self.decay:
                param._der += self.decay * param._der
            param._val -= self.lr * param._der